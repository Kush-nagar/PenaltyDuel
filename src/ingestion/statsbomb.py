"""
StatsBomb Open Data ingestion adapter.

Reads raw JSON from data/ and returns clean DataFrames.
Nothing in this module saves files — saving is the pipeline's job.

Public API (called by pipelines/build_dataset.py):
    build_competitions_df()
    build_matches_df(competitions_df)
    collect_penalty_events(matches_df)  -> (rows, failed_ids)
    build_lineup_cache(penalty_match_ids)
    build_penalties_df(penalty_event_rows, lineup_cache, matches_df)
    add_derived_fields(df)
    build_lineup_reference(lineup_cache)
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

from conf.settings import (
    COMPS_FILE,
    EVENTS_DIR,
    LINEUPS_DIR,
    MATCHES_DIR,
    GOAL_Y_CENTER_HIGH,
    GOAL_Y_CENTER_LOW,
    GOALKEEPER_POSITION_NAME,
    OUTCOME_3_MAP,
    SHOT_Z_HIGH_THRESHOLD,
)


# ── Private helpers ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[list]:
    """Read a JSON file; return None if file doesn't exist or is empty/invalid."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return None
        return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return None


def _normalize_name(name: Optional[str]) -> Optional[str]:
    """
    Lowercase, strip diacritics, collapse whitespace.
    This is a join-key helper for future fuzzy matching against other sources.
    It is NOT entity resolution — two different players can share a normalised name.
    """
    if not name:
        return None
    return re.sub(r"\s+", " ", unidecode(str(name)).lower().strip())


def _classify_shot_zone(
    end_y: Optional[float],
    end_z: Optional[float],
) -> Optional[str]:
    """
    Bin a shot's end-location into one of six goal-mouth zones (keeper's POV):
        {low, high} × {left, center, right}

    Returns None when end_y is missing (the zone model simply won't train
    on those rows — see blueprint §3.2).

    ⚠️  Validate the boundary constants (GOAL_Y_CENTER_LOW / HIGH in settings.py)
    by looking at the zone distribution printed at the end of the pipeline run.
    Raw (x, y) are preserved so you can re-bin without re-running.
    """
    if end_y is None:
        return None
    if end_y < GOAL_Y_CENTER_LOW:
        col = "left"
    elif end_y > GOAL_Y_CENTER_HIGH:
        col = "right"
    else:
        col = "center"
    row = "high" if (end_z is not None and end_z >= SHOT_Z_HIGH_THRESHOLD) else "low"
    return f"{row}_{col}"


def _resolve_keeper(
    lineup_json: Optional[list],
    shooting_team_id: Optional[int],
    period: Optional[int],
) -> tuple:
    """
    Derive the keeper for a penalty from the opposing team's lineup.

    StatsBomb does NOT attach a keeper_id to shot events — it must be
    derived from the lineup.  Strategy:
      1. Identify all players on the opposing team who held the Goalkeeper
         position at some point in the match.
      2. Narrow to those whose period window covers the kick's period.
      3. If exactly one candidate remains → "lineup_position_window" (clean).
      4. Otherwise fall back to the first goalkeeper listed → "lineup_fallback_first_gk"
         (still almost always correct; mid-match keeper changes are rare).
      5. If no goalkeeper found at all → "unresolved".

    Returns (keeper_id, keeper_name, keeper_nationality, resolution_method).
    """
    if lineup_json is None:
        return None, None, None, "no_lineup_file"
    if shooting_team_id is None:
        return None, None, None, "no_shooting_team_id"

    opposing_teams = [t for t in lineup_json if t.get("team_id") != shooting_team_id]
    if not opposing_teams:
        return None, None, None, "opposing_team_not_found"

    opp_players = opposing_teams[0].get("lineup", [])

    # All players who ever held the Goalkeeper position in this match
    all_gks = [
        p for p in opp_players
        if any(
            pos.get("position") == GOALKEEPER_POSITION_NAME
            for pos in p.get("positions", [])
        )
    ]

    if not all_gks:
        return None, None, None, "unresolved"

    # Narrow to those whose logged period window covers the kick's period.
    # from_period / to_period are integers in StatsBomb lineups.
    # to_period can be None (meaning "still active at the end of the match").
    if period is not None:
        period_gks = [
            p for p in all_gks
            if any(
                pos.get("position") == GOALKEEPER_POSITION_NAME
                and pos.get("from_period") is not None
                and pos.get("from_period") <= period
                and (pos.get("to_period") is None or pos.get("to_period") >= period)
                for pos in p.get("positions", [])
            )
        ]
    else:
        period_gks = []

    if len(period_gks) == 1:
        p = period_gks[0]
        return (
            p.get("player_id"),
            p.get("player_name"),
            (p.get("country") or {}).get("name"),
            "lineup_position_window",
        )

    # Fallback: first goalkeeper listed (handles period-window ambiguity and
    # the rare case of no from_period data in older competition files)
    p = all_gks[0]
    return (
        p.get("player_id"),
        p.get("player_name"),
        (p.get("country") or {}).get("name"),
        "lineup_fallback_first_gk",
    )


def _extract_gk_freeze_position(freeze_frame: list) -> tuple:
    """
    Pull the goalkeeper's x, y coordinates from the shot's freeze frame.

    ⚠️  This is the keeper's position AT THE INSTANT THE KICK WAS STRUCK
    (their pre-dive stance / weight distribution), NOT their completed dive
    direction.  Do not label this as "dive direction" anywhere.
    Returns (gk_x, gk_y) or (None, None) if the keeper isn't in the frame.
    """
    gk_entry = next(
        (
            ff for ff in freeze_frame
            if (ff.get("position") or {}).get("name") == GOALKEEPER_POSITION_NAME
            and ff.get("teammate") is False
        ),
        None,
    )
    if gk_entry is None:
        return None, None
    loc = gk_entry.get("location") or []
    return (
        loc[0] if len(loc) > 0 else None,
        loc[1] if len(loc) > 1 else None,
    )


# ── Public functions ───────────────────────────────────────────────────────────

def build_competitions_df() -> pd.DataFrame:
    """Load competitions.json into a clean, flat DataFrame."""
    data = _load_json(COMPS_FILE)
    if data is None:
        raise FileNotFoundError(
            f"competitions.json not found at {COMPS_FILE}\n"
            "Make sure your data/ folder is at the project root."
        )
    return pd.DataFrame(data)[[
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "competition_gender",
        "competition_youth",
        "competition_international",
        "season_name",
    ]]


def build_matches_df(competitions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load every matches/{competition_id}/{season_id}.json file and
    flatten them into one master match table.
    """
    rows = []
    for _, comp_row in tqdm(
        competitions_df.iterrows(),
        total=len(competitions_df),
        desc="Loading match files",
    ):
        path = MATCHES_DIR / str(comp_row["competition_id"]) / f"{comp_row['season_id']}.json"
        data = _load_json(path)
        if data is None:
            continue
        for m in data:
            rows.append({
                "match_id":               m.get("match_id"),
                "match_date":             m.get("match_date"),
                "kick_off":               m.get("kick_off"),
                "competition_id":         comp_row["competition_id"],
                "season_id":              comp_row["season_id"],
                "competition_name":       (m.get("competition") or {}).get("competition_name"),
                "season_name":            (m.get("season") or {}).get("season_name"),
                "country_name":           (m.get("competition") or {}).get("country_name"),
                "competition_stage_name": (m.get("competition_stage") or {}).get("name"),
                "home_team_id":           (m.get("home_team") or {}).get("home_team_id"),
                "home_team_name":         (m.get("home_team") or {}).get("home_team_name"),
                "away_team_id":           (m.get("away_team") or {}).get("away_team_id"),
                "away_team_name":         (m.get("away_team") or {}).get("away_team_name"),
                "home_score":             m.get("home_score"),
                "away_score":             m.get("away_score"),
                "match_week":             m.get("match_week"),
                "match_status":           m.get("match_status"),
                "stadium_name":           (m.get("stadium") or {}).get("name"),
                "referee_name":           (m.get("referee") or {}).get("name"),
            })

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset="match_id")
        .reset_index(drop=True)
    )


def collect_penalty_events(matches_df: pd.DataFrame) -> tuple:
    """
    Scan every match's events file and collect Shot events where
    shot.type.name == "Penalty".  Includes both in-game penalties
    (periods 1–4) and shootout kicks (period 5).

    Returns:
        penalty_rows  — list of {"match_id": ..., "event": {...}} dicts
        failed_ids    — match_ids whose events file couldn't be loaded
    """
    penalty_rows: list[dict] = []
    failed_ids: list = []

    for match_id in tqdm(matches_df["match_id"].tolist(), desc="Scanning for penalties"):
        events = _load_json(EVENTS_DIR / f"{match_id}.json")
        if events is None:
            failed_ids.append(match_id)
            continue
        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            if (e.get("shot") or {}).get("type", {}).get("name") != "Penalty":
                continue
            penalty_rows.append({"match_id": match_id, "event": e})

    return penalty_rows, failed_ids


def build_lineup_cache(penalty_match_ids: list) -> dict:
    """
    Load lineup files only for matches that contained at least one penalty.
    Avoids loading thousands of lineup files we'd never use.
    Returns {match_id: lineup_json_or_None}.
    """
    return {
        mid: _load_json(LINEUPS_DIR / f"{mid}.json")
        for mid in tqdm(penalty_match_ids, desc="Loading lineups")
    }


def build_penalties_df(
    penalty_event_rows: list,
    lineup_cache: dict,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise raw penalty event dicts into a clean, flat DataFrame.
    Derives the goalkeeper identity from the lineup cache.
    Does NOT add the derived target labels or shootout sequence —
    those are added by add_derived_fields() below.
    """
    matches_lookup = matches_df.set_index("match_id").to_dict(orient="index")
    rows = []

    for r in tqdm(penalty_event_rows, desc="Normalising events"):
        match_id  = r["match_id"]
        e         = r["event"]
        shot      = e.get("shot") or {}
        minfo     = matches_lookup.get(match_id, {})
        team      = e.get("team") or {}
        player    = e.get("player") or {}
        position  = e.get("position") or {}
        loc       = e.get("location") or []
        end_loc   = shot.get("end_location") or []
        ff        = shot.get("freeze_frame") or []

        end_x = end_loc[0] if len(end_loc) > 0 else None
        end_y = end_loc[1] if len(end_loc) > 1 else None
        end_z = end_loc[2] if len(end_loc) > 2 else None

        gk_x, gk_y = _extract_gk_freeze_position(ff)

        k_id, k_name, k_nat, k_method = _resolve_keeper(
            lineup_cache.get(match_id),
            team.get("id"),
            e.get("period"),
        )

        rows.append({
            # ── Provenance ────────────────────────────────────────────────
            "source":                    "statsbomb_open_data",
            "source_event_id":           e.get("id"),
            "match_id":                  match_id,
            "index_in_match":            e.get("index"),

            # ── Match / competition context ───────────────────────────────
            "competition_id":            minfo.get("competition_id"),
            "season_id":                 minfo.get("season_id"),
            "competition_name":          minfo.get("competition_name"),
            "season_name":               minfo.get("season_name"),
            "country_name":              minfo.get("country_name"),
            "competition_stage_name":    minfo.get("competition_stage_name"),
            "match_date":                minfo.get("match_date"),
            "home_team_id":              minfo.get("home_team_id"),
            "home_team_name":            minfo.get("home_team_name"),
            "away_team_id":              minfo.get("away_team_id"),
            "away_team_name":            minfo.get("away_team_name"),
            "stadium_name":              minfo.get("stadium_name"),
            "referee_name":              minfo.get("referee_name"),

            # ── Timing ────────────────────────────────────────────────────
            "period":                    e.get("period"),
            "minute":                    e.get("minute"),
            "second":                    e.get("second"),

            # ── Shooter ───────────────────────────────────────────────────
            "shooter_id":                player.get("id"),
            "shooter_name":              player.get("name"),
            "shooter_name_normalized":   _normalize_name(player.get("name")),
            "shooter_team_id":           team.get("id"),
            "shooter_team_name":         team.get("name"),
            "shooter_position_name":     position.get("name"),

            # ── Shot ──────────────────────────────────────────────────────
            "location_x":                loc[0] if len(loc) > 0 else None,
            "location_y":                loc[1] if len(loc) > 1 else None,
            "shot_end_location_x":       end_x,
            "shot_end_location_y":       end_y,
            "shot_end_location_z":       end_z,
            "shot_outcome_name":         (shot.get("outcome") or {}).get("name"),
            "shot_body_part_name":       (shot.get("body_part") or {}).get("name"),
            "shot_technique_name":       (shot.get("technique") or {}).get("name"),
            "shot_statsbomb_xg":         shot.get("statsbomb_xg"),
            "n_freeze_frame_players":    len(ff),

            # ── Goalkeeper (derived from lineup) ──────────────────────────
            "keeper_id":                 k_id,
            "keeper_name":               k_name,
            "keeper_name_normalized":    _normalize_name(k_name),
            "keeper_nationality":        k_nat,
            "keeper_resolution_method":  k_method,
            "gk_freeze_location_x":      gk_x,
            "gk_freeze_location_y":      gk_y,
            "gk_freeze_available":       gk_x is not None,

            # ── Placeholder: NOT available in StatsBomb open data ─────────
            # Reserved for future enrichment (Kaggle dive-label sets,
            # video pipeline).  Never populated with a guess here.
            "keeper_dive_direction":     np.nan,

            # ── Raw freeze frame: keep for deeper future use ─────────────
            # Dropped from the lean CSV export; kept in the full parquet.
            "shot_freeze_frame_raw":     json.dumps(ff),
        })

    return pd.DataFrame(rows)


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all columns that are computed from the already-extracted raw fields:
      - shot_zone, outcome_bin, outcome_3
      - is_shootout
      - shootout_kick_index, shootout_score_for_before, shootout_score_against_before
      - penalty_id, ingested_at
    """
    df = df.copy()

    # Zone
    df["shot_zone"] = df.apply(
        lambda r: _classify_shot_zone(r["shot_end_location_y"], r["shot_end_location_z"]),
        axis=1,
    )

    # Outcome labels
    df["outcome_bin"] = (df["shot_outcome_name"] == "Goal").astype(int)
    df["outcome_3"]   = df["shot_outcome_name"].map(OUTCOME_3_MAP).fillna("unknown")

    # Shootout flag — period 5 by StatsBomb convention.
    # The pipeline prints a period breakdown so you can verify this.
    df["is_shootout"] = df["period"] == 5

    # Sort chronologically within each match before deriving sequence
    df = df.sort_values(["match_id", "index_in_match"]).reset_index(drop=True)

    # Shootout sequence (only meaningful for period-5 kicks)
    df["shootout_kick_index"]          = pd.NA
    df["shootout_score_for_before"]    = pd.NA
    df["shootout_score_against_before"] = pd.NA

    shootout_df = df[df["is_shootout"]]
    for match_id, group_indices in shootout_df.groupby("match_id").groups.items():
        group = df.loc[group_indices].sort_values("index_in_match")
        running_scores: dict = {}

        for kick_num, (row_idx, row) in enumerate(group.iterrows(), start=1):
            s_team  = row["shooter_team_id"]
            home    = row["home_team_id"]
            away    = row["away_team_id"]
            opp     = away if s_team == home else home

            df.loc[row_idx, "shootout_kick_index"]           = kick_num
            df.loc[row_idx, "shootout_score_for_before"]     = running_scores.get(s_team, 0)
            df.loc[row_idx, "shootout_score_against_before"] = running_scores.get(opp, 0)

            if row["outcome_bin"] == 1:
                running_scores[s_team] = running_scores.get(s_team, 0) + 1

    # Stable surrogate key for multi-source merging later
    df["penalty_id"] = "statsbomb_open_data_" + df["source_event_id"].astype(str)

    # Pipeline timestamp (UTC)
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()

    return df


def build_lineup_reference(lineup_cache: dict) -> pd.DataFrame:
    """
    Build a flat player-level reference table from the lineup cache.
    One row per (match_id, player_id).
    Useful later for resolving player attributes from FBref / Transfermarkt.
    The `played_goalkeeper` flag makes it easy to filter just keepers.
    """
    rows = []
    for match_id, lineup_json in lineup_cache.items():
        if lineup_json is None:
            continue
        for team in lineup_json:
            for p in team.get("lineup", []):
                rows.append({
                    "match_id":               match_id,
                    "team_id":                team.get("team_id"),
                    "team_name":              team.get("team_name"),
                    "player_id":              p.get("player_id"),
                    "player_name":            p.get("player_name"),
                    "player_name_normalized": _normalize_name(p.get("player_name")),
                    "player_nickname":        p.get("player_nickname"),
                    "jersey_number":          p.get("jersey_number"),
                    "country_name":           (p.get("country") or {}).get("name"),
                    "played_goalkeeper": any(
                        pos.get("position") == GOALKEEPER_POSITION_NAME
                        for pos in p.get("positions", [])
                    ),
                })
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["match_id", "player_id"])
        .reset_index(drop=True)
    )