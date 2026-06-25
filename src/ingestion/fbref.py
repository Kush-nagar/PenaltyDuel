"""
FBref ingestion via SoccerData.

We use SoccerData EXCLUSIVELY for FBref (its Transfermarkt support is weaker
than ScraperFC's — see transfermarkt.py for that side).

What we pull:
  * standard season stats -> PKatt / PK (penalty counts) + minutes per player-season
  * misc season stats     -> a foot / preferred_foot column *if FBref exposes one*
                             (it usually does NOT at season level — we degrade to
                             None and let Transfermarkt fill it in).

Important SoccerData behaviour handled here:
  * It returns a MultiIndex DataFrame (both rows and columns).  We flatten with
    ``reset_index`` and join the column levels into flat snake-ish names.
  * Column names differ between versions, so NOTHING is hardcoded — we locate the
    PK / PKatt / minutes / foot / player-id columns by fuzzy suffix matching and
    log what we actually found.
  * SoccerData caches to ``data_dir`` automatically; we point that at
    ``outputs/enrichment/cache/fbref/`` and never disable caching.

Public API:
    statsbomb_seasons_to_fbref(season_names, lookback_years=3) -> (domestic, intl)
    fetch_fbref_career_penalty_stats(...) -> pd.DataFrame
    fetch_fbref_foot(...)                 -> pd.DataFrame
    build_fbref_index(career_df)          -> dict[str, list[dict]]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.enrichment.entity_resolution import normalize_name

# SoccerData league keys (verified via FBref.available_leagues()).
DOMESTIC_LEAGUES = ["Big 5 European Leagues Combined"]
INTERNATIONAL_LEAGUES = [
    "INT-World Cup",
    "INT-European Championship",
    "INT-Women's World Cup",
]


# ── Season handling ────────────────────────────────────────────────────────────

# FBref Big 5 European Leagues data starts ~1996-97.
# International tournament data on SoccerData starts ~2002.
# Seasons before these years don't exist in FBref → skip them to avoid
# burning subprocess time on guaranteed failures.
MIN_DOMESTIC_START_YEAR = 1996
MIN_INTL_YEAR = 2002


def _expand_domestic(start_end: tuple[int, int], lookback_years: int) -> set[str]:
    """Return FBref season strings for a domestic season plus N prior seasons."""
    start, _end = start_end
    out: set[str] = set()
    for s in range(start - lookback_years, start + 1):
        if s >= MIN_DOMESTIC_START_YEAR:
            out.add(f"{s}-{s + 1}")
    return out


def statsbomb_seasons_to_fbref(
    season_names: Iterable[str],
    lookback_years: int = 3,
) -> tuple[list[str], list[str]]:
    """Convert StatsBomb ``season_name`` values into FBref season strings.

    StatsBomb uses two shapes:
      * "2018/2019" -> domestic league season -> FBref "2018-2019" (+lookback)
      * "2022"      -> single-year international tournament -> FBref "2022"

    Returns ``(domestic_seasons, international_seasons)`` as sorted lists, since
    they map to different league groups.  Seasons before FBref's coverage window
    (MIN_DOMESTIC_START_YEAR / MIN_INTL_YEAR) are silently dropped.
    """
    domestic: set[str] = set()
    intl: set[str] = set()
    for raw in season_names:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        name = str(raw).strip()
        if "/" in name:
            a, b = name.split("/", 1)
            if a.isdigit() and b.isdigit():
                domestic |= _expand_domestic((int(a), int(b)), lookback_years)
        elif name.isdigit() and int(name) >= MIN_INTL_YEAR:
            intl.add(name)
    return sorted(domestic), sorted(intl)


def statsbomb_to_fbref_season_str(season_name: Optional[str]) -> Optional[str]:
    """Map a single StatsBomb ``season_name`` to its FBref string (no lookback)."""
    if not season_name:
        return None
    name = str(season_name).strip()
    if "/" in name:
        a, b = name.split("/", 1)
        if a.isdigit() and b.isdigit():
            return f"{a}-{b}"
        return None
    return name if name.isdigit() else None


# ── Column discovery (version-agnostic) ────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a SoccerData MultiIndex column DataFrame into flat string names."""
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for tup in df.columns:
            parts = [str(p) for p in tup if p not in (None, "") and not str(p).startswith("Unnamed")]
            flat.append("_".join(parts) if parts else "_".join(str(p) for p in tup))
        df.columns = flat
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """Find the first column whose (lowercased) name matches any regex pattern."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    return None


def _extract_fbref_player_id(df: pd.DataFrame) -> pd.Series:
    """Best-effort FBref player id.

    SoccerData sometimes exposes the FBref URL-slug id; if absent we derive a
    stable surrogate from the normalised name so the column is never empty.
    """
    id_col = _find_col(df, r"^player_id$", r"player.*id$", r"\bid\b")
    name_col = _find_col(df, r"^player$", r"player_name", r"player$")
    names = df[name_col] if name_col else pd.Series([None] * len(df))
    if id_col and df[id_col].notna().any():
        return df[id_col].astype("string")
    return names.map(lambda n: f"fbref_name::{normalize_name(n)}" if n else None).astype("string")


# ── Public fetchers ────────────────────────────────────────────────────────────

def _read_one(
    leagues: list[str],
    season: str,
    cache_dir: Path,
    stat_type: str,
):
    """Read one stat type for ONE season.

    Seasons are fetched individually on purpose: SoccerData raises if *any*
    season in a batch is unsupported (e.g. pre-1996 Big-5 seasons), which would
    otherwise discard every valid season too.  Per-season isolation lets the
    unsupported ones fail without taking the good ones down with them.
    """
    import soccerdata as sd  # imported lazily so the module loads without a browser

    fb = sd.FBref(leagues=leagues, seasons=season, data_dir=cache_dir)
    return fb.read_player_season_stats(stat_type=stat_type)


def _read_seasons_isolated(
    label: str,
    leagues: list[str],
    seasons: list[str],
    cache_dir: Path,
    stat_type: str,
    verbose: bool,
):
    """Yield flattened DataFrames per season, skipping any that fail."""
    ok, skipped = [], []
    for season in seasons:
        try:
            raw = _read_one(leagues, season, cache_dir, stat_type)
        except Exception as exc:  # noqa: BLE001 — unsupported season / transient error
            skipped.append(f"{season}({exc})")
            continue
        if raw is None or len(raw) == 0:
            continue
        ok.append(season)
        yield _flatten_columns(raw)
    if verbose:
        print(f"    FBref {stat_type} [{label}]: {len(ok)} ok, {len(skipped)} skipped")
        if skipped:
            print(f"      skipped: {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}")


def fetch_fbref_career_penalty_stats(
    domestic_seasons: list[str],
    intl_seasons: list[str],
    cache_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pull standard season stats and return raw career penalty counts.

    Output columns (one row per player-season, NO rates — counts only):
        player_name_normalized, fbref_player_id, season_name, squad,
        competition, pk_attempted, pk_scored, minutes_played, nation
    """
    frames: list[pd.DataFrame] = []
    groups = [
        ("domestic", DOMESTIC_LEAGUES, domestic_seasons),
        ("international", INTERNATIONAL_LEAGUES, intl_seasons),
    ]
    logged_cols = False
    for label, leagues, seasons in groups:
        if not seasons:
            continue
        for flat in _read_seasons_isolated(label, leagues, seasons, cache_dir, "standard", verbose):
            if verbose and not logged_cols:
                print(f"      columns sample: {flat.columns.tolist()[:12]}")
                logged_cols = True
            frames.append(_normalise_standard(flat))

    if not frames:
        return _empty_career_df()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(
        subset=["player_name_normalized", "season_name", "squad", "competition"]
    ).reset_index(drop=True)


def _normalise_standard(flat: pd.DataFrame) -> pd.DataFrame:
    name_col = _find_col(flat, r"^player$", r"player_name", r"player$")
    season_col = _find_col(flat, r"^season$", r"season")
    squad_col = _find_col(flat, r"^squad$", r"team", r"squad")
    comp_col = _find_col(flat, r"^league$", r"comp", r"competition")
    pkatt_col = _find_col(flat, r"pkatt$", r"pk.?att", r"pens.*att")
    pk_col = _find_col(flat, r"performance_pk$", r"(?<!att)_pk$", r"^pk$", r"pens.*made")
    min_col = _find_col(flat, r"playing.*time_min", r"_min$", r"^min$", r"minutes")
    nation_col = _find_col(flat, r"^nation$", r"nation")

    names = flat[name_col] if name_col else pd.Series([None] * len(flat))
    out = pd.DataFrame(
        {
            "player_name_normalized": names.map(normalize_name),
            "fbref_player_id": _extract_fbref_player_id(flat),
            "season_name": flat[season_col].astype("string") if season_col else pd.NA,
            "squad": flat[squad_col].astype("string") if squad_col else pd.NA,
            "competition": flat[comp_col].astype("string") if comp_col else pd.NA,
            "pk_attempted": pd.to_numeric(flat[pkatt_col], errors="coerce") if pkatt_col else pd.NA,
            "pk_scored": pd.to_numeric(flat[pk_col], errors="coerce") if pk_col else pd.NA,
            "minutes_played": pd.to_numeric(flat[min_col], errors="coerce") if min_col else pd.NA,
            "nation": flat[nation_col].astype("string") if nation_col else pd.NA,
        }
    )
    return out[out["player_name_normalized"].notna()].reset_index(drop=True)


def fetch_fbref_foot(
    domestic_seasons: list[str],
    intl_seasons: list[str],
    cache_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pull misc season stats and return a per-player foot table IF available.

    FBref usually does NOT expose foot at season level.  When no foot column is
    present we return an empty frame (never raise) so Transfermarkt fills the gap.

    Output columns: player_name_normalized, preferred_foot
    """
    frames: list[pd.DataFrame] = []
    groups = [
        ("domestic", DOMESTIC_LEAGUES, domestic_seasons),
        ("international", INTERNATIONAL_LEAGUES, intl_seasons),
    ]
    foot_missing_logged = False
    for label, leagues, seasons in groups:
        if not seasons:
            continue
        for flat in _read_seasons_isolated(label, leagues, seasons, cache_dir, "misc", verbose):
            foot_col = _find_col(flat, r"preferred.?foot", r"^foot$", r"_foot$")
            name_col = _find_col(flat, r"^player$", r"player_name", r"player$")
            if foot_col is None or name_col is None:
                if verbose and not foot_missing_logged:
                    print("      FBref misc: no foot column present (expected) — TM will fill foot")
                    foot_missing_logged = True
                continue
            sub = pd.DataFrame(
                {
                    "player_name_normalized": flat[name_col].map(normalize_name),
                    "preferred_foot": flat[foot_col].map(_normalise_foot),
                }
            ).dropna(subset=["player_name_normalized", "preferred_foot"])
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["player_name_normalized", "preferred_foot"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset="player_name_normalized").reset_index(drop=True)


def _normalise_foot(value) -> Optional[str]:
    """Map any foot string to {'left','right','both'} or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = str(value).strip().lower()
    if "both" in v:
        return "both"
    has_left, has_right = "left" in v, "right" in v
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _empty_career_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_name_normalized",
            "fbref_player_id",
            "season_name",
            "squad",
            "competition",
            "pk_attempted",
            "pk_scored",
            "minutes_played",
            "nation",
        ]
    )


def build_fbref_index(career_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Index FBref career rows by normalised name for entity resolution.

    Returns ``{normalised_name: [{fbref_player_id, squads, nation,
    is_goalkeeper}]}``.  ``is_goalkeeper`` is None here (standard stats don't
    cleanly flag GKs), letting the resolver fall back to other signals.
    """
    index: dict[str, list[dict]] = {}
    if career_df.empty:
        return index
    for name, grp in career_df.groupby("player_name_normalized"):
        squads = sorted({s for s in grp["squad"].dropna().astype(str)})
        nations = [n for n in grp["nation"].dropna().astype(str)]
        index[name] = [
            {
                "fbref_player_id": grp["fbref_player_id"].dropna().iloc[0]
                if grp["fbref_player_id"].notna().any()
                else None,
                "squads": squads,
                "nation": nations[0] if nations else None,
                "is_goalkeeper": None,
            }
        ]
    return index


def season_part_paths(out_dir: Path, group: str, season: str) -> tuple[Path, Path]:
    """Per-season output parquet paths (career, foot) for the subprocess worker."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{group}_{season}")
    return out_dir / f"_career_{safe}.parquet", out_dir / f"_foot_{safe}.parquet"


# ── CLI worker (ONE season per invocation) ───────────────────────────────────────
# Run as a SUBPROCESS by pipelines/enrich_players.py.  SoccerData's headless
# browser can crash hard enough to kill the whole interpreter (not just raise), so
# isolating each season in its own process means a crash loses at most that one
# season.  On success the worker writes per-season parquets; the orchestrator then
# concatenates them WITHOUT touching the browser, so the main pipeline can never be
# killed by FBref and always proceeds to the Transfermarkt / join / output stages.
#
#   python -m src.ingestion.fbref --group domestic --season 2022-2023 \
#       --cache-dir <cache> --out-dir <out>
def _cli_main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="FBref single-season worker.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group", required=True, choices=["domestic", "international"])
    parser.add_argument("--season", required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domestic = [args.season] if args.group == "domestic" else []
    intl = [args.season] if args.group == "international" else []

    career = fetch_fbref_career_penalty_stats(domestic, intl, cache_dir, verbose=False)
    foot = fetch_fbref_foot(domestic, intl, cache_dir, verbose=False)

    career_path, foot_path = season_part_paths(out_dir, args.group, args.season)
    career.to_parquet(career_path, index=False)
    foot.to_parquet(foot_path, index=False)
    print(f"WORKER_DONE {args.group} {args.season} career={len(career)} foot={len(foot)}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    raise SystemExit(_cli_main())
