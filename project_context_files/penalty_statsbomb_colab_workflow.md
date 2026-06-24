# StatsBomb Open Data → Clean Penalty Dataset: Google Colab Workflow

*Phase 1 of the penalty shootout predictor: data acquisition and cleaning, StatsBomb Open Data only. No modeling, no enrichment from other sources yet — this produces the clean, one-row-per-penalty foundation you'll merge other sources into later.*

**A note on certainty before you start.** The StatsBomb Open Data repo's JSON structure has been stable and widely documented for years, and the code below is built on that well-established structure. I was not able to verify it against the live repo in this session (my search tool was unavailable). So the workflow is designed to **verify itself**: Cell 3 loads one real file from each folder and prints its actual keys before anything downstream depends on them. If a field name has changed since this was written, you'll see it immediately in Cell 3's output rather than getting a silent failure three cells later. Treat Cell 3's output as the source of truth over anything written here.

---

## Section 1: What to download from StatsBomb Open Data

The repo (`github.com/statsbomb/open-data`) has one folder that matters: `data/`. Everything inside it is JSON, organized as five sub-folders. Here's what each one is, and what you need it for.

| Folder / file | What it contains | Do you need it? |
|---|---|---|
| **`data/competitions.json`** | A single flat list: every competition+season combination available in the open release (e.g. "FIFA World Cup, 2022", "Women's Euro, 2022"), with IDs, country, gender. | **Yes — required.** This is your index. Without it you don't know what `competition_id`/`season_id` pairs exist. |
| **`data/matches/{competition_id}/{season_id}.json`** | One file per competition-season; a list of every match in it, with date, teams, score, competition stage (e.g. "Final"), stadium, referee. | **Yes — required.** This is your match metadata dimension table, and the only source of `competition_stage_name` (needed to flag finals/knockout rounds later). |
| **`data/lineups/{match_id}.json`** | One file per match; full squad lists for both teams, with player IDs, names, nicknames, nationality, jersey numbers, and a `positions` array logging which position each player held and when. | **Yes — required.** This is the *only* reliable way to identify **who the goalkeeper was** for a given penalty (see Section 2 — this is not a field directly on the shot event). |
| **`data/events/{match_id}.json`** | One file per match; every logged event (passes, duels, shots, etc.) — usually 2,000–3,500 events per match. | **Yes — required.** Penalty kicks live in here as `Shot` events. This is where outcome, shot placement, and the shot-level freeze frame come from. |
| **`data/three-sixty/{match_id}.json`** | Frame-by-frame positional "360" data — but only for a **limited subset of competitions/matches** (mainly recent major tournaments StatsBomb chose to release it for). | **Skip for now.** It adds positional detail beyond the shot-level `freeze_frame` already in the events file, but coverage is sparse and inconsistent, and parsing it is non-trivial. This is squarely a **future video/positional-upgrade-phase** asset (it's conceptually the closest thing StatsBomb's open data has to your eventual computer-vision features) — revisit it then, not now. |
| Repo root files (`README.md`, `LICENSE`, `.github/`, any data-spec docs) | Documentation and licensing terms. | **Skip for execution** — but do read the README/LICENSE once for the usage terms (StatsBomb's open data carries a non-commercial attribution license; respect it in any write-up). No data to process here. |

**Practical implication:** you only need to clone the repo once (it's the `data/` tree above — a shallow clone keeps this manageable in Colab; see Cell 2). You do **not** need to touch `three-sixty/`, and you do **not** need any paid StatsBomb product for this phase.

---

## Section 2: Exact dataset schema for the first version

One row = one penalty kick (in-game **and** shootout — they're the same event type, distinguished by `period`). Every column below is tagged **Direct** (read straight from a StatsBomb field) or **Derived** (computed by this pipeline from direct fields).

### Identifiers / provenance
| Column | Direct / Derived | Notes |
|---|---|---|
| `penalty_id` | Derived | `"statsbomb_open_data_" + source_event_id` — stable surrogate key, designed for later multi-source merging (Section 7). |
| `source` | Derived (constant) | `"statsbomb_open_data"` — every row, for provenance once you add other sources. |
| `source_event_id` | Direct | The event's StatsBomb UUID (`event.id`). Guaranteed unique per kick — your dedup check. |
| `match_id` | Direct | |
| `index_in_match` | Direct | Event's position in the match's chronological event stream — used to derive shootout order. |

### Match / competition context
| Column | Direct / Derived | Notes |
|---|---|---|
| `competition_id`, `season_id`, `competition_name`, `season_name`, `country_name` | Direct | From `competitions.json` / `matches.json`. |
| `competition_stage_name` | Direct | e.g. `"Final"`, `"Group Stage"` — your only source for stage/importance later. |
| `match_date` | Direct | |
| `home_team_id`, `home_team_name`, `away_team_id`, `away_team_name` | Direct | |
| `stadium_name`, `referee_name` | Direct | Low-priority, free metadata — kept since it costs nothing. |
| `period`, `minute`, `second` | Direct | |

### Shooter
| Column | Direct / Derived | Notes |
|---|---|---|
| `shooter_id`, `shooter_name` | Direct | |
| `shooter_name_normalized` | Derived | Lowercased, diacritics stripped — a join-key helper for Section 7, **not** entity resolution itself. |
| `shooter_team_id`, `shooter_team_name` | Direct | |
| `shooter_position_name` | Direct | Position tag *at the moment of this event* (from the event's own `position` field). |
| `shot_body_part_name` | Direct | `"Left Foot"` / `"Right Foot"` / `"Head"` / `"Other"` — **which foot struck this specific kick.** Useful, but see the "not available" box below — this is not the same as a player's *preferred* foot. |

### Shot outcome & placement
| Column | Direct / Derived | Notes |
|---|---|---|
| `location_x`, `location_y` | Direct | Where the kick was *taken from* — near-constant for penalties, kept for QA. |
| `shot_end_location_x`, `shot_end_location_y`, `shot_end_location_z` | Direct | **The placement field.** `z` (height) is only populated when StatsBomb logged the ball leaving the ground — expect nulls for some rows. |
| `shot_outcome_name` | Direct | `"Goal"`, `"Saved"`, `"Off T"`, `"Post"`, `"Wayward"`, `"Blocked"`, etc. |
| `shot_technique_name` | Direct | Usually `"Normal"` for penalties; kept for completeness. |
| `shot_statsbomb_xg` | Direct | StatsBomb's own xG for this shot — a free baseline to compare your model against later. |
| `shot_zone` | **Derived** | 6-bin classification (`{low,high} × {left,center,right}`) of `shot_end_location`, using documented pitch-coordinate assumptions. **Flagged for visual validation in Cell 9** — see Section 5. |
| `outcome_bin` | Derived | `1` if `shot_outcome_name == "Goal"`, else `0`. **Primary target label.** |
| `outcome_3` | Derived | `{scored, saved, missed}`, grouped from `shot_outcome_name`. **Secondary target label.** |

### Goalkeeper (this is the engineered part — read this carefully)
StatsBomb does **not** put a `keeper_id` field on the shot event. There is no field anywhere in the open data that says "this is the goalkeeper this shot was taken against." It has to be derived from the lineup.

| Column | Direct / Derived | Notes |
|---|---|---|
| `keeper_id`, `keeper_name` | **Derived** | Resolved from the opposing team's lineup — the player whose logged `"Goalkeeper"` position window covers this event's match-time. Method in Cell 6. |
| `keeper_name_normalized` | Derived | Same purpose as the shooter version. |
| `keeper_nationality` | Direct (via lineup, once resolved) | |
| `keeper_resolution_method` | Derived | One of `lineup_position_window` (clean), `lineup_fallback_first_gk` (used when the time-window match was ambiguous — still almost always correct, since mid-match keeper changes are rare), or `unresolved`. **Inspect this column's distribution before trusting `keeper_id`** (Section 5). |
| `gk_freeze_location_x`, `gk_freeze_location_y` | Derived | The goalkeeper's position **at the moment the kick was struck**, pulled from the shot's `freeze_frame`. |
| `gk_freeze_available` | Derived | Whether a freeze frame was present for this shot at all (coverage is not universal). |

> ### ⚠️ Fields that do **not** exist in StatsBomb Open Data — do not expect them, and this pipeline does not invent them
> - **Keeper dive direction.** There is no field anywhere that records which way the keeper actually dove or whether they guessed correctly. The `gk_freeze_location` above is the closest proxy available, but it is the keeper's **position at the instant of the kick** (essentially their starting stance / pre-dive read), **not** the outcome of their dive — don't treat the two as equivalent. The schema includes an explicit empty `keeper_dive_direction` column (all `NaN` in this phase) as a placeholder for future enrichment (Kaggle hand-coded sets, or your future video pipeline) — it is **never populated with a guess** here.
> - **Preferred foot** (as a player attribute, e.g. "this player is left-footed"). `shot_body_part_name` tells you which foot was used for *this kick only*. A true preferred-foot attribute will need to come from FBref or Transfermarkt later.
> - **Date of birth, age, height** for players.
> - **Attendance.**
> - **Score state before in-game (non-shootout) penalties.** This requires walking the *entire* match's goal timeline, not just the penalty events — out of scope for this acquisition phase. `match_id`, `period`, `minute`, `second` are preserved precisely so this can be derived later by joining back to full match events.
> - **"Stakes"/pressure classification** (must-score-or-lose, etc.) for shootout kicks. This phase derives the *raw sequence facts* needed to compute it (`shootout_kick_index`, running score) — the semantic labeling is feature-engineering work for the next phase, not data acquisition.

### Shootout sequence (derived, shootout rows only)
| Column | Direct / Derived | Notes |
|---|---|---|
| `is_shootout` | Derived | `period == 5`, by StatsBomb convention. **Verify this against the live `period` value counts printed in Cell 5** before trusting it — see the caveat in Section 5. |
| `shootout_kick_index` | Derived | 1, 2, 3… in chronological order within the shootout, computed by sorting on `index_in_match`. |
| `shootout_score_for_before`, `shootout_score_against_before` | Derived | Running shootout tally *before* this kick. |

### Housekeeping
| Column | Direct / Derived | Notes |
|---|---|---|
| `n_freeze_frame_players` | Derived | Count of players in the shot's freeze frame — a QA signal (0 means no freeze frame at all for this shot). |
| `shot_end_location_x_is_missing`, `shot_end_location_y_is_missing`, `keeper_id_is_missing`, `gk_freeze_location_x_is_missing` | Derived | Explicit missingness flags. **No imputation happens in this phase** — that's a modeling-phase decision, not a data-cleaning one. |
| `shot_freeze_frame_raw` | Direct (kept as JSON string) | The full raw freeze frame, preserved for deeper future use (e.g. extracting *all* nearby player positions, not just the keeper). Dropped from the lean CSV export, kept in one parquet file. |
| `ingested_at` | Derived | Pipeline run timestamp. |

---

## Section 3: Step-by-step Google Colab workflow

| Cell | Purpose | Produces |
|---|---|---|
| **1** | Install/import everything needed. | A ready Python environment. |
| **2** | Shallow-clone the repo. | Local `open-data/data/` folder. |
| **3** | Load one real file from each folder; print actual keys. | Live verification of field names before you build anything on top of them. |
| **4** | Load `competitions.json` and **every** `matches/*.json` into one master `matches_df`; define reusable `load_events()`/`load_lineup()` functions. | `competitions_df`, `matches_df`, two loader functions. |
| **5** | Scan every match's events file; keep only `Shot` events where `shot.type.name == "Penalty"`. | A raw list of penalty event dicts (in-game **and** shootout), plus a `period` value-count check. |
| **6** | Normalize the raw events into clean columns; resolve the goalkeeper via the lineup; classify shot zone; derive shootout sequencing. | `penalties_df` — the full, feature-rich (but not yet validated) table. |
| **7** | Validate: uniqueness, outcome vocabulary, missingness flags, drop exact duplicates. | A cleaned, asserted-correct `penalties_df`. |
| **8** | Save CSV + Parquet outputs, plus the supporting `matches` and `lineups` reference tables. | The files described in Section 6. |
| **9** | Sanity checks: distributions, a visual scatter to validate the zoning constants, a manual spot-check of one shootout's derived sequence. | Confidence the dataset is correct before you build on it. |

A **shallow clone** (`--depth 1`) is used in Cell 2 rather than a full clone, because the full git history of this repo is much larger than the current snapshot of files and you never need past versions — just the current JSON. Cells 5–6 scan **every** match in the repo (not just a hand-picked subset) because in-game penalties can occur in any competition, not just the tournaments that went to shootouts; this takes a few minutes but is well within Colab's free tier.

---

## Section 4: Exact code

Paste these into Colab in order, one cell at a time. Run Cell 3 first and actually read its output before trusting anything downstream — it's your live check that the field names below are still accurate.

### Cell 1 — Setup, installs, imports
```python
# ============================================================
# CELL 1 — Setup, installs, imports
# ============================================================
!pip install -q pyarrow unidecode tqdm

import os
import re
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from unidecode import unidecode
from tqdm.auto import tqdm

pd.set_option("display.max_columns", 100)
warnings.filterwarnings("ignore")

print("pandas:", pd.__version__)
```

### Cell 2 — Clone the repo (shallow)
```python
# ============================================================
# CELL 2 — Clone the StatsBomb Open Data repo (shallow clone)
# ============================================================
REPO_DIR = "open-data"

if not os.path.isdir(REPO_DIR):
    # --depth 1: only the current snapshot of files, no git history.
    # The full history is much larger and we never need past versions.
    !git clone --depth 1 https://github.com/statsbomb/open-data.git
else:
    print(f"'{REPO_DIR}' already exists — skipping clone.")

DATA_DIR = os.path.join(REPO_DIR, "data")
assert os.path.isdir(DATA_DIR), "data/ folder not found — the clone may have failed."

print("Top-level contents of data/:")
print(sorted(os.listdir(DATA_DIR)))
```

### Cell 3 — Inspect folder structure and verify field names live
```python
# ============================================================
# CELL 3 — Inspect the folder structure and verify field names
# ============================================================
# We do NOT blindly assume the JSON schema. We load one real file from each
# folder and print its keys, so we can confirm field names BEFORE relying on
# them in later cells. If StatsBomb has renamed something since this was
# written, you will see it here — adjust the .get() key paths in Cell 6 accordingly.

COMP_PATH = os.path.join(DATA_DIR, "competitions.json")
with open(COMP_PATH, "r", encoding="utf-8") as f:
    competitions_raw = json.load(f)
print(f"competitions.json: {len(competitions_raw)} competition-season entries")
print("Sample competition entry keys:", list(competitions_raw[0].keys()))
print(competitions_raw[0])

# Peek one matches file
sample_comp_id = competitions_raw[0]["competition_id"]
sample_season_id = competitions_raw[0]["season_id"]
matches_sample_path = os.path.join(DATA_DIR, "matches", str(sample_comp_id), f"{sample_season_id}.json")
with open(matches_sample_path, "r", encoding="utf-8") as f:
    matches_sample = json.load(f)
print(f"\nSample matches file: {matches_sample_path}")
print("Sample match entry top-level keys:", list(matches_sample[0].keys()))

# Peek one lineups file
sample_match_id = matches_sample[0]["match_id"]
lineup_sample_path = os.path.join(DATA_DIR, "lineups", f"{sample_match_id}.json")
with open(lineup_sample_path, "r", encoding="utf-8") as f:
    lineup_sample = json.load(f)
print(f"\nSample lineups file: {lineup_sample_path}")
print("Sample lineup team-entry keys:", list(lineup_sample[0].keys()))
print("Sample player entry keys:", list(lineup_sample[0]["lineup"][0].keys()))
if lineup_sample[0]["lineup"][0]["positions"]:
    print("Sample player 'positions' entry:", lineup_sample[0]["lineup"][0]["positions"][0])

# Peek one events file, specifically a Shot event if this match happens to have one
events_sample_path = os.path.join(DATA_DIR, "events", f"{sample_match_id}.json")
with open(events_sample_path, "r", encoding="utf-8") as f:
    events_sample = json.load(f)
print(f"\nSample events file: {events_sample_path}  ({len(events_sample)} events)")

shot_events = [e for e in events_sample if e.get("type", {}).get("name") == "Shot"]
print(f"Shot events in this match: {len(shot_events)}")
if shot_events:
    print("Sample Shot event top-level keys:", list(shot_events[0].keys()))
    print("Sample 'shot' sub-object keys:", list(shot_events[0]["shot"].keys()))
```

### Cell 4 — Load competitions + all matches; define reusable loaders
```python
# ============================================================
# CELL 4 — Load competitions and matches into master tables;
#           define reusable loaders for events and lineups
# ============================================================

# --- competitions_df ----------------------------------------------------
competitions_df = pd.DataFrame(competitions_raw)[
    ["competition_id", "season_id", "country_name", "competition_name",
     "competition_gender", "competition_youth", "competition_international",
     "season_name"]
]
print(f"competitions_df: {competitions_df.shape[0]} competition-season rows")
display(competitions_df.head())

# --- matches_df: loop over every (competition_id, season_id) pair -------
def load_matches_for(competition_id, season_id):
    """Load and flatten one matches/{comp}/{season}.json file into clean rows."""
    path = os.path.join(DATA_DIR, "matches", str(competition_id), f"{season_id}.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        matches = json.load(f)

    rows = []
    for m in matches:
        rows.append({
            "match_id": m.get("match_id"),
            "match_date": m.get("match_date"),
            "kick_off": m.get("kick_off"),
            "competition_id": competition_id,
            "season_id": season_id,
            "competition_name": (m.get("competition") or {}).get("competition_name"),
            "season_name": (m.get("season") or {}).get("season_name"),
            "country_name": (m.get("competition") or {}).get("country_name"),
            "competition_stage_name": (m.get("competition_stage") or {}).get("name"),
            "home_team_id": (m.get("home_team") or {}).get("home_team_id"),
            "home_team_name": (m.get("home_team") or {}).get("home_team_name"),
            "away_team_id": (m.get("away_team") or {}).get("away_team_id"),
            "away_team_name": (m.get("away_team") or {}).get("away_team_name"),
            "home_score": m.get("home_score"),
            "away_score": m.get("away_score"),
            "match_status": m.get("match_status"),
            "match_week": m.get("match_week"),
            "stadium_name": (m.get("stadium") or {}).get("name"),
            "referee_name": (m.get("referee") or {}).get("name"),
        })
    return rows

all_match_rows = []
for _, row in tqdm(competitions_df.iterrows(), total=len(competitions_df), desc="Loading matches"):
    all_match_rows.extend(load_matches_for(row["competition_id"], row["season_id"]))

matches_df = pd.DataFrame(all_match_rows).drop_duplicates(subset="match_id").reset_index(drop=True)
print(f"matches_df: {matches_df.shape[0]} unique matches across all competitions")
display(matches_df.head())

# --- reusable loaders for events / lineups (called per-match in Cell 5/6) ---
def load_events(match_id):
    path = os.path.join(DATA_DIR, "events", f"{match_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_lineup(match_id):
    path = os.path.join(DATA_DIR, "lineups", f"{match_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

### Cell 5 — Identify penalty events
```python
# ============================================================
# CELL 5 — Scan every match's events file and pull out penalty
#           shot events only (both in-game and shootout kicks)
# ============================================================
# A "penalty" event = type.name == "Shot" AND shot.type.name == "Penalty".
# This captures BOTH penalties awarded during normal play and shootout kicks.
# We distinguish the two using `period` once we inspect its values below.

penalty_event_rows = []
failed_match_ids = []

for match_id in tqdm(matches_df["match_id"].tolist(), desc="Scanning matches for penalties"):
    events = load_events(match_id)
    if events is None:
        failed_match_ids.append(match_id)
        continue
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        shot = e.get("shot", {})
        if shot.get("type", {}).get("name") != "Penalty":
            continue
        # Keep the raw nested event; we normalize it in Cell 6.
        penalty_event_rows.append({"match_id": match_id, "event": e})

print(f"Found {len(penalty_event_rows)} penalty shot events "
      f"across {matches_df['match_id'].nunique()} matches.")
print(f"{len(failed_match_ids)} matches had no events file or failed to load.")

# Quick look at `period` values among penalty events — this is how we tell
# shootout kicks (period 5, by StatsBomb convention) apart from in-game
# penalties (periods 1-4). VERIFY this before relying on it downstream.
period_counts = pd.Series([r["event"].get("period") for r in penalty_event_rows]).value_counts()
print("\nPenalty events by `period` value (verify period 5 = shootout):")
print(period_counts)
```

### Cell 6 — Extract, normalize, and engineer fields
```python
# ============================================================
# CELL 6 — Extract, normalize, and engineer the core fields
# ============================================================

# ---------- helpers -----------------------------------------------------

def normalize_name(name):
    """Lowercase, strip diacritics, collapse whitespace — a join-key helper
    for future fuzzy matching against other sources. The original display
    name is preserved separately; this is NOT entity resolution itself."""
    if not name:
        return None
    name = unidecode(str(name)).lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name

def classify_shot_zone(end_x, end_y, end_z=None):
    """
    Bin a shot end-location into a 6-zone goal-mouth grid (keeper's POV):
    columns {left, center, right} x rows {low, high}.

    ASSUMPTIONS (StatsBomb 120x80 pitch convention, attacking toward x=120):
      - goal line is at x = 120
      - goal posts are at y ~= 36 and y ~= 44 (goal centered on y = 40, ~8 wide)
      - "high" vs "low" uses end_z when present (z is only attached for shots
        that left the ground); below ~2.0 units is treated as "low"
    These constants are a documented approximation — VALIDATE them in Cell 9
    by plotting end_location for goals before trusting this zoning in modeling.
    """
    if end_y is None:
        return None
    if end_y < 38.67:
        col = "left"      # keeper's left
    elif end_y > 41.33:
        col = "right"     # keeper's right
    else:
        col = "center"
    row = "high" if (end_z is not None and end_z >= 2.0) else "low"
    return f"{row}_{col}"

def resolve_keeper(lineup_json, shooting_team_id, period, minute, second):
    """
    Identify the goalkeeper facing this penalty.
    StatsBomb does NOT tag a 'keeper_id' field directly on shot events, so we
    derive it from the opposing team's lineup, matching the player whose
    logged 'Goalkeeper' position window covers this event's match-time.

    NOTE: the from/to time-window match below is a best-effort heuristic.
    Validate it via the `keeper_resolution_method` value counts in Cell 7/9 —
    mid-match keeper changes are rare, so the fallback below is correct the
    overwhelming majority of the time even when the precise window doesn't match.

    Returns (keeper_id, keeper_name, keeper_nationality, resolution_method).
    """
    if lineup_json is None:
        return None, None, None, "no_lineup_file"

    opposing = [t for t in lineup_json if t.get("team_id") != shooting_team_id]
    if not opposing:
        return None, None, None, "opposing_team_not_found"
    opp_lineup = opposing[0].get("lineup", [])

    event_seconds = (minute or 0) * 60 + (second or 0)

    def to_seconds(ts):
        if not ts:
            return None
        parts = ts.split(":")[:2]
        m, s = parts[0], parts[1]
        return int(m) * 60 + int(float(s))

    candidates = []
    for p in opp_lineup:
        for pos in p.get("positions", []):
            if pos.get("position") != "Goalkeeper":
                continue
            if pos.get("from_period") != period:
                continue
            start_s = to_seconds(pos.get("from"))
            end_s = to_seconds(pos.get("to")) if pos.get("to") else None
            if start_s is None:
                continue
            if end_s is None or start_s <= event_seconds <= end_s:
                candidates.append(p)

    if len(candidates) == 1:
        p = candidates[0]
        return (p.get("player_id"), p.get("player_name"),
                (p.get("country") or {}).get("name"), "lineup_position_window")

    # Fallback: first/only player who ever logged 'Goalkeeper' for this team this match
    fallback = [p for p in opp_lineup
                if any(pos.get("position") == "Goalkeeper" for pos in p.get("positions", []))]
    if fallback:
        p = fallback[0]
        return (p.get("player_id"), p.get("player_name"),
                (p.get("country") or {}).get("name"), "lineup_fallback_first_gk")

    return None, None, None, "unresolved"

# ---------- pre-load lineups only for matches that actually have penalties ----------
penalty_match_ids = sorted({r["match_id"] for r in penalty_event_rows})
lineup_cache = {mid: load_lineup(mid)
                for mid in tqdm(penalty_match_ids, desc="Loading lineups (penalty matches only)")}

# ---------- build one row per penalty event ----------
matches_lookup = matches_df.set_index("match_id").to_dict(orient="index")

rows = []
for r in tqdm(penalty_event_rows, desc="Normalizing penalty events"):
    match_id = r["match_id"]
    e = r["event"]
    shot = e.get("shot", {})
    minfo = matches_lookup.get(match_id, {})

    team = e.get("team", {}) or {}
    player = e.get("player", {}) or {}
    position = e.get("position", {}) or {}
    loc = e.get("location") or [None, None]
    end_loc = shot.get("end_location") or [None, None, None]
    end_x, end_y = end_loc[0], end_loc[1]
    end_z = end_loc[2] if len(end_loc) > 2 else None
    outcome_name = (shot.get("outcome") or {}).get("name")
    freeze_frame = shot.get("freeze_frame") or []

    # goalkeeper proxy location from the shot freeze frame (NOT dive direction)
    gk_frame = next((ff for ff in freeze_frame
                      if (ff.get("position") or {}).get("name") == "Goalkeeper"
                      and ff.get("teammate") is False), None)

    keeper_id, keeper_name, keeper_nat, keeper_method = resolve_keeper(
        lineup_cache.get(match_id), team.get("id"), e.get("period"), e.get("minute"), e.get("second")
    )

    rows.append({
        # ---- identifiers / provenance ----
        "source": "statsbomb_open_data",
        "source_event_id": e.get("id"),
        "match_id": match_id,
        "index_in_match": e.get("index"),

        # ---- match / competition context (direct, via matches_df) ----
        "competition_id": minfo.get("competition_id"),
        "season_id": minfo.get("season_id"),
        "competition_name": minfo.get("competition_name"),
        "season_name": minfo.get("season_name"),
        "country_name": minfo.get("country_name"),
        "competition_stage_name": minfo.get("competition_stage_name"),
        "match_date": minfo.get("match_date"),
        "home_team_id": minfo.get("home_team_id"),
        "home_team_name": minfo.get("home_team_name"),
        "away_team_id": minfo.get("away_team_id"),
        "away_team_name": minfo.get("away_team_name"),
        "stadium_name": minfo.get("stadium_name"),
        "referee_name": minfo.get("referee_name"),

        # ---- timing (direct) ----
        "period": e.get("period"),
        "minute": e.get("minute"),
        "second": e.get("second"),

        # ---- shooter (direct, from event + lineup) ----
        "shooter_id": player.get("id"),
        "shooter_name": player.get("name"),
        "shooter_name_normalized": normalize_name(player.get("name")),
        "shooter_team_id": team.get("id"),
        "shooter_team_name": team.get("name"),
        "shooter_position_name": position.get("name"),

        # ---- shot details (direct) ----
        "location_x": loc[0],
        "location_y": loc[1],
        "shot_end_location_x": end_x,
        "shot_end_location_y": end_y,
        "shot_end_location_z": end_z,
        "shot_outcome_name": outcome_name,
        "shot_body_part_name": (shot.get("body_part") or {}).get("name"),
        "shot_technique_name": (shot.get("technique") or {}).get("name"),
        "shot_statsbomb_xg": shot.get("statsbomb_xg"),
        "n_freeze_frame_players": len(freeze_frame),

        # ---- goalkeeper (derived) ----
        "keeper_id": keeper_id,
        "keeper_name": keeper_name,
        "keeper_name_normalized": normalize_name(keeper_name),
        "keeper_nationality": keeper_nat,
        "keeper_resolution_method": keeper_method,
        "gk_freeze_location_x": (gk_frame or {}).get("location", [None, None])[0],
        "gk_freeze_location_y": (gk_frame or {}).get("location", [None, None])[1],
        "gk_freeze_available": gk_frame is not None,

        # ---- explicitly NOT available from StatsBomb open data (placeholder) ----
        "keeper_dive_direction": np.nan,   # reserved for future enrichment; never invented here

        # ---- raw freeze frame, kept for optional deeper use later ----
        "shot_freeze_frame_raw": json.dumps(freeze_frame),
    })

penalties_df = pd.DataFrame(rows)
print(f"penalties_df (pre-cleaning): {penalties_df.shape}")
display(penalties_df.head())
```

```python
# ---------- continued: derived fields — zone, outcome bins, shootout sequencing ----------

penalties_df["shot_zone"] = penalties_df.apply(
    lambda r: classify_shot_zone(r["shot_end_location_x"], r["shot_end_location_y"], r["shot_end_location_z"]),
    axis=1
)

penalties_df["outcome_bin"] = (penalties_df["shot_outcome_name"] == "Goal").astype(int)

OUTCOME_3_MAP = {
    "Goal": "scored",
    "Saved": "saved",
    "Saved Off Target": "saved",
    "Off T": "missed",
    "Post": "missed",
    "Wayward": "missed",
    "Blocked": "missed",
}
penalties_df["outcome_3"] = penalties_df["shot_outcome_name"].map(OUTCOME_3_MAP).fillna("unknown")

# is_shootout: VERIFY against the period_counts printed in Cell 5 before trusting this.
penalties_df["is_shootout"] = penalties_df["period"] == 5

# shootout_kick_index + running score state, derived by sorting shootout kicks
# within each match using the event's global `index_in_match` (chronological order).
penalties_df = penalties_df.sort_values(["match_id", "index_in_match"]).reset_index(drop=True)

def add_shootout_sequence(group):
    group = group.copy()
    so = group[group["is_shootout"]].sort_values("index_in_match")
    if so.empty:
        return group
    kick_idx, score_for, score_against, running = {}, {}, {}, {}
    for i, (_, row) in enumerate(so.iterrows(), start=1):
        eid = row["source_event_id"]
        shooter_team = row["shooter_team_id"]
        opp_team = row["away_team_id"] if shooter_team == row["home_team_id"] else row["home_team_id"]
        kick_idx[eid] = i
        score_for[eid] = running.get(shooter_team, 0)
        score_against[eid] = running.get(opp_team, 0)
        if row["outcome_bin"] == 1:
            running[shooter_team] = running.get(shooter_team, 0) + 1
    group["shootout_kick_index"] = group["source_event_id"].map(kick_idx)
    group["shootout_score_for_before"] = group["source_event_id"].map(score_for)
    group["shootout_score_against_before"] = group["source_event_id"].map(score_against)
    return group

penalties_df = penalties_df.groupby("match_id", group_keys=False).apply(add_shootout_sequence)

# stable surrogate key — designed for later multi-source merging (Section 7)
penalties_df["penalty_id"] = penalties_df["source"] + "_" + penalties_df["source_event_id"].astype(str)

print("Derived fields added. Shape:", penalties_df.shape)
display(penalties_df[["is_shootout", "shootout_kick_index", "shot_zone", "outcome_bin"]].sample(5))
```

### Cell 7 — Clean and validate
```python
# ============================================================
# CELL 7 — Clean and validate
# ============================================================

# ---- 1) duplicate check (hard requirement: event ids must be unique) ----
dupe_events = penalties_df["source_event_id"].duplicated().sum()
assert dupe_events == 0, f"Found {dupe_events} duplicate source_event_id values — investigate before continuing."

dupe_penalty_id = penalties_df["penalty_id"].duplicated().sum()
assert dupe_penalty_id == 0, "Duplicate penalty_id — should be impossible if source_event_id is unique."

# ---- 2) outcome vocabulary check ----
unexpected_outcomes = set(penalties_df["shot_outcome_name"].dropna().unique()) - set(OUTCOME_3_MAP.keys())
if unexpected_outcomes:
    print("WARNING — unmapped shot_outcome_name values found, review OUTCOME_3_MAP:", unexpected_outcomes)

# ---- 3) missingness flags (we FLAG, we do not impute — imputation belongs to
#         the feature-engineering/modeling phase, not data acquisition) ----
for col in ["shot_end_location_x", "shot_end_location_y", "keeper_id", "gk_freeze_location_x"]:
    penalties_df[f"{col}_is_missing"] = penalties_df[col].isna()

print("Missingness rates (key fields):")
print(penalties_df[[c for c in penalties_df.columns if c.endswith("_is_missing")]].mean().round(3))

print("\nkeeper_resolution_method breakdown:")
print(penalties_df["keeper_resolution_method"].value_counts(dropna=False))

print("\ngk_freeze_available rate:", round(penalties_df["gk_freeze_available"].mean(), 3))

# ---- 4) sanity ranges ----
assert penalties_df["outcome_bin"].isin([0, 1]).all()
assert penalties_df["period"].notna().all(), "Every penalty event must have a period."

# ---- 5) drop exact full-row duplicates if any slipped in ----
before = len(penalties_df)
penalties_df = penalties_df.drop_duplicates(subset="penalty_id").reset_index(drop=True)
print(f"\nDropped {before - len(penalties_df)} exact-duplicate rows (if any).")

print(f"\nFinal cleaned shape: {penalties_df.shape}")
```

### Cell 8 — Save outputs
```python
# ============================================================
# CELL 8 — Save outputs (CSV + Parquet)
# ============================================================
OUTPUT_DIR = "penalty_predictor_data/statsbomb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: mount Google Drive instead of (or in addition to) local Colab
# storage, so outputs persist after the runtime disconnects. Uncomment to use:
#
# from google.colab import drive
# drive.mount('/content/drive')
# OUTPUT_DIR = "/content/drive/MyDrive/penalty_predictor_data/statsbomb"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

ingested_at = datetime.now(timezone.utc).isoformat()
penalties_df["ingested_at"] = ingested_at
matches_df["ingested_at"] = ingested_at

# 1) The final, cleaned, one-row-per-penalty table — your core deliverable.
penalties_out = penalties_df.drop(columns=["shot_freeze_frame_raw"])  # drop the heavy raw blob from the lean export
penalties_out.to_csv(f"{OUTPUT_DIR}/penalties_statsbomb_clean.csv", index=False)
penalties_out.to_parquet(f"{OUTPUT_DIR}/penalties_statsbomb_clean.parquet", index=False)

# 2) The same table WITH the raw freeze-frame JSON kept, for deeper future use
#    (e.g. extracting all nearby player positions, not just the goalkeeper).
penalties_df.to_parquet(f"{OUTPUT_DIR}/penalties_statsbomb_with_freeze_frame.parquet", index=False)

# 3) The match metadata dimension table — a separate join target for later.
matches_df.to_csv(f"{OUTPUT_DIR}/matches_statsbomb.csv", index=False)
matches_df.to_parquet(f"{OUTPUT_DIR}/matches_statsbomb.parquet", index=False)

# 4) A reference table of every lineup entry for matches that had a penalty —
#    useful later for enrichment / cross-checking player attributes.
lineup_rows = []
for mid, lu in lineup_cache.items():
    if lu is None:
        continue
    for team in lu:
        for p in team.get("lineup", []):
            lineup_rows.append({
                "match_id": mid,
                "team_id": team.get("team_id"),
                "team_name": team.get("team_name"),
                "player_id": p.get("player_id"),
                "player_name": p.get("player_name"),
                "player_name_normalized": normalize_name(p.get("player_name")),
                "player_nickname": p.get("player_nickname"),
                "jersey_number": p.get("jersey_number"),
                "country_name": (p.get("country") or {}).get("name"),
            })
lineups_out = pd.DataFrame(lineup_rows).drop_duplicates()
lineups_out.to_csv(f"{OUTPUT_DIR}/lineups_statsbomb_penalty_matches.csv", index=False)
lineups_out.to_parquet(f"{OUTPUT_DIR}/lineups_statsbomb_penalty_matches.parquet", index=False)

print("Saved files:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / 1024
    print(f"  {fn:55s} {size_kb:8.1f} KB")
```

### Cell 9 — Quick sanity checks / EDA
```python
# ============================================================
# CELL 9 — Quick sanity checks / EDA
# ============================================================
import matplotlib.pyplot as plt

print("=== Overview ===")
print(f"Total penalties: {len(penalties_df)}")
print(f"  In-game:  {(~penalties_df['is_shootout']).sum()}")
print(f"  Shootout: {penalties_df['is_shootout'].sum()}")
print(f"Overall conversion rate: {penalties_df['outcome_bin'].mean():.3f}")

print("\n=== Outcome distribution ===")
print(penalties_df["shot_outcome_name"].value_counts())

print("\n=== Penalties per competition ===")
print(penalties_df["competition_name"].value_counts())

print("\n=== Shot zone distribution (sanity-check the zoning constants) ===")
print(penalties_df["shot_zone"].value_counts(dropna=False))

print("\n=== Keeper resolution success rate ===")
print(penalties_df["keeper_resolution_method"].value_counts(normalize=True).round(3))

# Visual check: do goals cluster near the corners as expected for penalties?
goals = penalties_df[penalties_df["outcome_bin"] == 1]
plt.figure(figsize=(6, 4))
plt.scatter(goals["shot_end_location_y"], goals["shot_end_location_z"].fillna(0), alpha=0.3, s=10)
plt.axvline(36, color="red", linestyle="--", label="goal post (approx)")
plt.axvline(44, color="red", linestyle="--")
plt.xlabel("end_location_y (goal width)")
plt.ylabel("end_location_z (height)")
plt.title("Scored penalties: end-location scatter (visually confirm goal-frame constants)")
plt.legend()
plt.show()

# Manual spot-check: pick one well-known shootout and print its derived
# sequence to eyeball against a real source (e.g. Wikipedia) for correctness.
example_match = penalties_df[penalties_df["is_shootout"]]["match_id"].mode()
if not example_match.empty:
    mid = example_match.iloc[0]
    cols = ["shootout_kick_index", "shooter_team_name", "shooter_name", "outcome_bin",
            "shootout_score_for_before", "shootout_score_against_before"]
    print(f"\n=== Spot-check: shootout sequence for match_id {mid} ===")
    print(penalties_df[penalties_df["match_id"] == mid].sort_values("shootout_kick_index")[cols]
          .to_string(index=False))
```

---

## Section 5: Data quality checks

**What to run (beyond what's already built into Cells 7 and 9):**

| Check | How | What it catches |
|---|---|---|
| **Event uniqueness** | `assert penalties_df["source_event_id"].duplicated().sum() == 0` (Cell 7) | A penalty counted twice — would silently inflate sample sizes and corrupt rates. |
| **`period` assumption for shootouts** | Inspect the `period_counts` printed in Cell 5 *before* trusting `is_shootout = (period == 5)`. | If a competition encodes shootouts differently, every shootout-derived feature downstream (kick index, score state) would be silently wrong for that subset. |
| **Keeper resolution rate** | `penalties_df["keeper_resolution_method"].value_counts()` (Cell 7/9). | A high `unresolved` rate means many rows will have no usable `keeper_id` — you'd want to know that before building keeper features on top of this table. |
| **Outcome vocabulary** | The `unexpected_outcomes` check in Cell 7. | A new/renamed `shot_outcome_name` value that silently falls into `outcome_3 = "unknown"` instead of being correctly bucketed. |
| **Shot-zone visual sanity** | The scatter plot in Cell 9. | Confirms the hand-coded goal-post constants (`y ≈ 36/44`, `z ≥ 2.0`) actually line up with where scored penalties cluster, before you rely on `shot_zone` as a model target. |
| **Coverage by competition** | `penalties_df["competition_name"].value_counts()` and the per-field missingness table (Cell 7). | Tells you which competitions actually have freeze-frame/end-location data and which don't — directly informs what your placement/keeper sub-models can be trained on later. |
| **Manual shootout spot-check** | The Cell 9 spot-check, cross-referenced against an external source (e.g. Wikipedia) for one well-known shootout. | Confirms `shootout_kick_index` and the running score state are in the correct order — the single most failure-prone derived feature in this pipeline, since it depends on `index_in_match` sorting behaving as expected. |

**Common mistakes to avoid:**
- **Trusting `period == 5` as "shootout" without checking.** It's a well-established convention, but verify it against the printed value counts for *your* pulled data before building on it — don't take it on faith.
- **Treating the absence of shootout events for a match you know went to penalties as a bug.** It usually means StatsBomb's open release simply didn't capture that match's shootout in event-level detail — a real data-completeness gap, not a pipeline error. Don't try to "fix" it by inventing rows.
- **Treating `gk_freeze_location` as the keeper's completed dive.** It's the keeper's position **at the instant of the kick** — essentially their starting stance, not where they ended up. Mislabeling this as "dive direction" in your eventual write-up would overstate what the data supports.
- **Not checking the `keeper_resolution_method` distribution.** If `unresolved` is non-trivial, downstream keeper features will have real gaps — you want to know the size of that gap now, not discover it during modeling.
- **Loading lineups for every match in the repo.** Cell 6 deliberately restricts lineup loading to the matches that actually contain a penalty (`penalty_match_ids`) — loading lineups for thousands of irrelevant matches wastes time for no benefit.
- **Forgetting that StatsBomb's open data covers only specific competitions.** This pipeline pulls *everything currently in the open release*, but that's still a curated subset of world football (mostly major international tournaments and a handful of league releases) — not a comprehensive penalty history. Say this plainly in any write-up; don't imply broader coverage than you have.

**Detecting duplicate or bad penalty rows specifically:**
- Duplicates: checked via `source_event_id` (hard assertion) and `penalty_id` (derived from it) in Cell 7. If a duplicate ever appears, it almost certainly means the same match file was scanned twice — check `matches_df` for accidental duplicate `match_id` rows across competitions first.
- Bad rows: a penalty with `shot_outcome_name` missing entirely, or with `shooter_id`/`shooter_team_id` null, indicates a malformed event — these are rare but worth an explicit `penalties_df[penalties_df["shot_outcome_name"].isna()]` check before saving.
- Confirming penalties "make sense": the Cell 9 overview (`outcome_bin.mean()`) should land in a plausible range for penalty conversion (high-70s to low-80s percent is the well-known long-run norm) — if your pulled subset shows something wildly different, that's worth investigating before proceeding, not just noting.

---

## Section 6: Final saved files

All written by Cell 8 into `penalty_predictor_data/statsbomb/` (locally in the Colab runtime by default; switch to the commented-out Google Drive path in Cell 8 if you want them to persist across sessions).

| File | Format | Purpose |
|---|---|---|
| `penalties_statsbomb_clean.csv` | CSV | **The core deliverable** — one row per penalty, fully cleaned, all derived fields, lean (freeze-frame JSON dropped). Use this for anything that doesn't need the raw freeze frame. |
| `penalties_statsbomb_clean.parquet` | Parquet | Same content as the CSV, typed and compressed — prefer this for any further pandas work (faster load, preserves dtypes). |
| `penalties_statsbomb_with_freeze_frame.parquet` | Parquet | Same table, but with the full raw `shot_freeze_frame_raw` JSON string retained per row — kept separately because it's large and most downstream work won't need it. Use this if you later want to extract *all* nearby player positions, not just the keeper. |
| `matches_statsbomb.csv` / `.parquet` | CSV + Parquet | The match metadata dimension table (every match across every pulled competition, not just ones with penalties) — a reusable join target for any future match-level feature you add. |
| `lineups_statsbomb_penalty_matches.csv` / `.parquet` | CSV + Parquet | Every lineup entry (player, team, nationality, nickname) for matches that contained at least one penalty — your reference table for player attribute enrichment and entity resolution against other sources. |

**Suggested folder structure** (works identically in Colab-local storage or a mounted Google Drive):
```
penalty_predictor_data/
└── statsbomb/
    ├── penalties_statsbomb_clean.csv
    ├── penalties_statsbomb_clean.parquet
    ├── penalties_statsbomb_with_freeze_frame.parquet
    ├── matches_statsbomb.csv
    ├── matches_statsbomb.parquet
    ├── lineups_statsbomb_penalty_matches.csv
    └── lineups_statsbomb_penalty_matches.parquet
```
When you add the next source (FBref, Understat, etc.), create sibling folders — `penalty_predictor_data/fbref/`, `penalty_predictor_data/understat/` — rather than mixing sources into one folder. This mirrors the `src/ingestion/{source}.py` adapter-per-source pattern from the project architecture, and keeps provenance unambiguous at the filesystem level too.

---

## Section 7: Next-step readiness

This phase deliberately produces a **StatsBomb-only** table, but every design choice above was made with the next phase — merging in FBref, Understat, Transfermarkt, and any future enrichment — in mind.

**Keys to preserve (already in the schema above) — do not drop these when you move to feature engineering:**
- `penalty_id`, `source`, `source_event_id` — provenance and a stable join key for this row, forever. Once you add other sources, every new source's adapter should emit the same three-column provenance pattern.
- `shooter_id`, `keeper_id` — StatsBomb's own internal player IDs. **Important:** these IDs are StatsBomb-specific and will **not** match FBref's or Transfermarkt's player identifiers — there is no shared ID namespace across these sources. This is exactly why `shooter_name_normalized` and `keeper_name_normalized` were built now.
- `shooter_name_normalized`, `keeper_name_normalized` — the diacritic-stripped, lowercased name strings. These are your **starting point** for fuzzy name matching against other sources later — but matching on normalized name alone is not enough on its own (two different players can share a name). When you build the cross-source entity-resolution step, pair this with a second signal — `match_date`/`season_name` (for career-window plausibility) and `keeper_nationality`/player nationality where available — before accepting a match, exactly as planned in the project architecture.
- `match_date`, `season_name`, `competition_name` — needed to disambiguate same-named players by era, and to align this dataset's time range with whatever time range FBref/Understat/Transfermarkt pulls cover.
- `shooter_team_name`, `home_team_name`, `away_team_name` — team names will also need their own normalization/mapping step later (StatsBomb's team naming conventions won't always match other sources' either) — not handled in this phase, but the raw names are preserved so that step is possible.

**What this phase deliberately leaves undone, on purpose, for the next phase:**
- No cross-source entity resolution — that requires having at least one other source loaded to resolve *against*.
- No imputation of missing values — only `*_is_missing` flags. Imputation/shrinkage is a feature-engineering decision, not a cleaning one.
- No "preferred foot," DOB, height, or pressure/stakes classification — these come from later enrichment or later feature engineering, not from StatsBomb, and were not invented here (see the Section 2 callout box).
- No score-state-before for in-game penalties — `match_id`/`period`/`minute`/`second` are preserved precisely so this can be computed later by re-walking each match's full event timeline if needed.

When you start the FBref/Understat/Transfermarkt phase, the practical workflow will be: load each new source through its own adapter into the same provenance pattern (`source`, `source_event_id`, plus that source's own ID), build a `name_map` keyed on `(normalized_name, plausible_birth_year_or_season, nationality)`, propose matches with fuzzy string matching, and require a human-reviewable confirmation step before merging — never auto-merge on name alone. The `*_normalized` columns and the full provenance trail in this StatsBomb table are exactly what make that next phase tractable instead of a re-do.
