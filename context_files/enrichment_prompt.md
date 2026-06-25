# Prompt: FBref + Transfermarkt Data Enrichment Pipeline

## Who you are and what you are building

You are an expert data engineer and sports analytics developer. You are building
a data enrichment pipeline for a penalty shootout prediction system for
football/soccer. Phase 1 (StatsBomb data acquisition and cleaning) is already
complete. You are now building Phase 2: enriching the clean penalty dataset with
player-level attributes from FBref (via SoccerData) and Transfermarkt (via
ScraperFC).

Do not build any models, dashboards, or APIs. This prompt is only about data
enrichment and producing clean output files.

---

## What already exists

The project has this folder structure. Every file listed here already exists and
is working. Do not modify any existing file. Only add new files.

```
penalty-predictor/
├── data/                                   # StatsBomb raw JSON — read-only, never touch
├── conf/
│   ├── __init__.py
│   └── settings.py                         # paths + constants (details below)
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── statsbomb.py                    # existing StatsBomb adapter — do not modify
│   └── cleaning/
│       ├── __init__.py
│       └── validate.py                     # existing validator — do not modify
├── pipelines/
│   └── build_dataset.py                    # existing pipeline — do not modify
├── outputs/
│   └── statsbomb/
│       ├── penalties_statsbomb_clean.csv
│       ├── penalties_statsbomb_clean.parquet      ← primary input for this pipeline
│       ├── penalties_statsbomb_with_freeze_frame.parquet
│       ├── matches_statsbomb.csv
│       ├── matches_statsbomb.parquet
│       ├── lineups_statsbomb_penalty_matches.csv
│       └── lineups_statsbomb_penalty_matches.parquet  ← player name reference
└── requirements.txt
```

### What conf/settings.py contains

```python
from pathlib import Path

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "statsbomb"

# All other settings are pitch coordinates and outcome maps — not relevant here.
```

---

## The existing penalty dataset schema

`outputs/statsbomb/penalties_statsbomb_clean.parquet` has one row per penalty
kick. These are the columns that matter most for the enrichment join:

| Column | Type | Notes |
|---|---|---|
| `penalty_id` | str | Stable surrogate key — `"statsbomb_open_data_" + event_uuid` |
| `source_event_id` | str | StatsBomb event UUID |
| `match_date` | str | e.g. `"2022-11-20"` — use for temporal leakage prevention |
| `season_name` | str | e.g. `"2022/2023"` — use to scope FBref season queries |
| `shooter_id` | int | StatsBomb internal player ID (NOT an FBref or Transfermarkt ID) |
| `shooter_name` | str | Display name e.g. `"Lionel Messi"` |
| `shooter_name_normalized` | str | Lowercase, diacritics stripped e.g. `"lionel messi"` |
| `shooter_position_name` | str | e.g. `"Center Forward"` |
| `shooter_team_name` | str | Team at time of kick |
| `shot_body_part_name` | str | `"Left Foot"` / `"Right Foot"` — which foot struck THIS kick only, not preferred foot |
| `keeper_id` | int | StatsBomb internal keeper ID |
| `keeper_name` | str | Display name |
| `keeper_name_normalized` | str | Same normalization as shooter |
| `keeper_nationality` | str | Nationality from StatsBomb lineup |
| `outcome_bin` | int | 1 = scored, 0 = not scored |
| `is_shootout` | bool | True for period 5 kicks |

The `lineups_statsbomb_penalty_matches.parquet` file has one row per
(match_id, player_id) with these columns:

| Column | Notes |
|---|---|
| `player_id` | StatsBomb ID |
| `player_name` | Display name |
| `player_name_normalized` | Normalized join key |
| `country_name` | Nationality — use as a second signal during entity resolution |
| `played_goalkeeper` | bool — True if this player held the Goalkeeper position |

---

## What you need to add

You need to produce two new enrichment tables and one final merged output:

### Table 1: `player_attributes.parquet`
One row per unique player (shooters and keepers combined). These are the
attributes to add. Every column is nullable — never invent a value.

| Column | Source | Notes |
|---|---|---|
| `player_name_normalized` | derived | The join key — matches the column of the same name in the penalty table |
| `statsbomb_player_id` | existing data | Carried over from the lineup reference |
| `preferred_foot` | Transfermarkt (primary), FBref misc (fallback) | `"left"`, `"right"`, `"both"`, or `None` |
| `date_of_birth` | Transfermarkt | `"YYYY-MM-DD"` string or `None` |
| `height_cm` | Transfermarkt | integer or `None` |
| `nationality` | Transfermarkt (cross-check against StatsBomb `country_name`) | |
| `foot_source` | derived | `"transfermarkt"`, `"fbref_misc"`, or `"unresolved"` |
| `tm_player_url` | Transfermarkt | Full URL — keep for auditing and future re-scraping |
| `fbref_player_id` | FBref | FBref's internal player ID (from their URL slug) — keep for future joins |

### Table 2: `player_career_penalty_stats.parquet`
One row per (player_name_normalized, season_name). These are aggregated career
penalty counts from FBref, used to build richer prior estimates in the feature
engineering phase. Do NOT compute rates here — just the raw counts.

| Column | Source | Notes |
|---|---|---|
| `player_name_normalized` | derived | Join key |
| `fbref_player_id` | FBref | |
| `season_name` | FBref | e.g. `"2022-2023"` — normalize to match the penalty table's format |
| `squad` | FBref | Team name that season |
| `competition` | FBref | League / competition name |
| `pk_attempted` | FBref `standard` stats | `PKatt` column — penalties attempted that season |
| `pk_scored` | FBref `standard` stats | `PK` column — penalties scored that season |
| `minutes_played` | FBref `standard` stats | For context |

### Final output: `penalties_enriched.parquet`
The original `penalties_statsbomb_clean.parquet` with new columns joined in.
Every original column stays untouched. You only add columns. The join is on
`player_name_normalized` using the entity-resolved mapping (see below).

New columns added to each penalty row:

| Column | Notes |
|---|---|
| `shooter_preferred_foot` | From `player_attributes` for the shooter |
| `shooter_dob` | From `player_attributes` |
| `shooter_height_cm` | From `player_attributes` |
| `shooter_foot_source` | Which source resolved the shooter's foot |
| `keeper_preferred_foot` | From `player_attributes` for the keeper |
| `keeper_height_cm` | From `player_attributes` |
| `keeper_foot_source` | |
| `shooter_preferred_foot_is_missing` | bool flag — never drop rows for missing enrichment |
| `keeper_preferred_foot_is_missing` | bool flag |
| `enrichment_version` | string constant e.g. `"fbref_tm_v1"` — for provenance |

---

## Tool assignments — read this carefully

### SoccerData → FBref only

Install: `pip install soccerdata`

Use SoccerData exclusively for pulling FBref data. Do not use it for
Transfermarkt (its Transfermarkt support is weaker than ScraperFC's).

**What to pull from FBref via SoccerData:**

1. **Standard season stats** — to get `PKatt` (penalties attempted) and `PK`
   (penalties scored) per player per season. Use `stat_type="standard"`.

2. **Misc season stats** — to get preferred foot where FBref exposes it at the
   season level. Use `stat_type="misc"`. Check whether a `foot` or
   `preferred_foot` column is present in the returned DataFrame. If it is not
   present, do not error — just set `preferred_foot = None` for those rows and
   let Transfermarkt fill it in.

**SoccerData FBref usage pattern:**

```python
import soccerdata as sd

# SoccerData caches responses in ~/.cache/soccerdata/ by default.
# This means the second run is instant — do not disable caching.

fbref = sd.FBref(leagues="Big 5 European Leagues", seasons="2020-21")
# seasons can be a list: ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
# leagues can be a list of specific league names

# Pull standard stats (has PKatt and PK columns)
standard_stats = fbref.read_player_season_stats(stat_type="standard")

# Pull misc stats (may have foot column)
misc_stats = fbref.read_player_season_stats(stat_type="misc")
```

**Critical SoccerData behaviour to be aware of:**

- SoccerData returns a MultiIndex DataFrame. Flatten it with
  `df.reset_index()` before processing.
- Column names after flattening may include the stat category as a prefix.
  Print `df.columns.tolist()` and `df.head()` at the start and handle
  whatever names are actually returned. Do not hardcode column names without
  first checking.
- SoccerData uses its own internal player IDs (derived from FBref URL slugs).
  These are NOT StatsBomb player IDs. The join between SoccerData output and
  the StatsBomb penalty table must go through `player_name_normalized`, not
  through IDs.
- Rate limiting: SoccerData handles this automatically when caching is enabled.
  Do not add extra sleep() calls — they are not needed and will slow the script.
- SoccerData covers specific leagues and seasons. It will NOT return data for
  all players in the penalty table (especially international tournament players
  who only appear in World Cup / Euros data, which FBref has but SoccerData
  may not cover in all versions). Players with no FBref match should get
  `pk_attempted = None` and should NOT cause an error.

**FBref seasons to query:**
Pull data for all seasons that overlap with the StatsBomb penalty data. The
penalty table has a `season_name` column (e.g. `"2018/2019"`). Build the
season list dynamically from the unique values in that column, converting
to FBref format (e.g. `"2018/2019"` → `"2018-19"`). Include 2–3 seasons
before the earliest season in the data to capture career history.

---

### ScraperFC → Transfermarkt only

Install: `pip install ScraperFC`

Use ScraperFC exclusively for Transfermarkt. Do not use ScraperFC for FBref —
SoccerData handles FBref better.

**What to pull from Transfermarkt via ScraperFC:**

- `preferred_foot` — this is the primary reason for scraping Transfermarkt.
  It is more consistently populated here than anywhere else.
- `date_of_birth`
- `height_cm`
- `nationality` (for cross-checking against StatsBomb `country_name`)

**ScraperFC Transfermarkt usage pattern:**

```python
from ScraperFC import Transfermarkt

tm = Transfermarkt()
```

Read the ScraperFC documentation (https://scraperfc.readthedocs.io/) to find
the correct method for retrieving individual player profiles. The exact method
name may be `get_player_info`, `scrape_player`, or similar — check the docs
and use whatever is current. Do not guess method names.

**Critical ScraperFC / Transfermarkt behaviour to be aware of:**

- You do NOT have Transfermarkt player URLs ahead of time. You will need to
  search for each player by name and then fetch their profile. Check whether
  ScraperFC provides a search method; if not, construct Transfermarkt search
  URLs manually (Transfermarkt's search endpoint is
  `https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=NAME`).
- Rate limit manually: add `time.sleep(3)` between every Transfermarkt request.
  Transfermarkt blocks scrapers aggressively. Do not skip this.
- Scrape only the players who appear in the penalty table. Build a deduplicated
  list of unique (player_name_normalized, country_name) pairs from the
  penalty table and the lineup reference. This is your scrape queue.
- Cache every raw response to `outputs/enrichment/cache/tm/` before parsing it.
  If a run is interrupted, the cache means you do not re-scrape players you
  already have. Check the cache before making any network request.
- Preferred foot values from Transfermarkt come back as strings like `"left"`,
  `"right"`, `"both"`. Normalize to lowercase. Map anything unexpected to None.

---

## Entity resolution — the hardest part

**The core problem:** StatsBomb uses its own internal integer player IDs. FBref
uses its own URL-slug-based IDs. Transfermarkt uses its own numeric IDs. None
of these ID namespaces overlap. The only common signal across all three is the
player's name, and names are inconsistent (diacritics, abbreviations, spelling
variants, "First Last" vs "Last, First").

**The required approach:**

1. Build a **player pool** from the existing data. Start with
   `lineups_statsbomb_penalty_matches.parquet` which has every player's
   StatsBomb ID and their `player_name_normalized`. This is your ground truth.

2. When FBref (SoccerData) returns player names, normalize them using the
   exact same function: lowercase, strip diacritics with `unidecode`, collapse
   whitespace. Implement this as a function called `normalize_name()` — do not
   use any other normalization.

   ```python
   import re
   from unidecode import unidecode

   def normalize_name(name: str) -> str:
       if not name:
           return None
       return re.sub(r"\s+", " ", unidecode(str(name)).lower().strip())
   ```

3. Match on `player_name_normalized` as the **first signal**. But never
   auto-accept a name-only match. Require a **second signal** from this list
   before accepting:
   - Nationality matches (StatsBomb `country_name` vs Transfermarkt `nationality`)
   - Team-season overlap (the player's team in FBref for a given season matches
     their team in the StatsBomb penalty table for a kick in that season)
   - The player's position (goalkeeper or outfield — never match a goalkeeper
     to an outfield player)

4. Use **rapidfuzz** (install: `pip install rapidfuzz`) for fuzzy name
   matching as a fallback when exact normalized names don't match. Use
   `rapidfuzz.fuzz.token_sort_ratio` with a threshold of 90. Only propose
   fuzzy matches — do not auto-accept them.

5. Output a file called `outputs/enrichment/entity_resolution/name_map.csv`
   with these columns:
   - `player_name_normalized` — from StatsBomb
   - `statsbomb_player_id`
   - `fbref_player_id` — matched FBref ID or empty
   - `tm_player_url` — matched Transfermarkt URL or empty
   - `match_method` — `"exact_name"`, `"exact_name_plus_nationality"`,
     `"exact_name_plus_team_season"`, `"fuzzy_proposed"`, or `"unresolved"`
   - `match_confidence` — `"high"`, `"medium"`, `"low"`, `"unresolved"`
   - `needs_review` — bool: True when match_method is `"fuzzy_proposed"` or
     `"unresolved"`. These rows must be reviewed by a human before being used.

6. Output a separate file called `outputs/enrichment/entity_resolution/unresolved.csv`
   containing only the rows where `needs_review == True`. This is the human
   review queue.

7. When producing `player_attributes.parquet` and `penalties_enriched.parquet`,
   only use matches where `match_confidence` is `"high"` or `"medium"`.
   Rows with `"low"` or `"unresolved"` confidence get `preferred_foot = None`
   and the appropriate `*_is_missing = True` flag. Never guess.

---

## Files to create

Create exactly these new files. Do not modify any existing file.

```
penalty-predictor/
├── src/
│   └── ingestion/
│       ├── fbref.py           # SoccerData/FBref scraping functions
│       └── transfermarkt.py   # ScraperFC/Transfermarkt scraping functions
├── src/
│   └── enrichment/
│       ├── __init__.py
│       └── entity_resolution.py   # name normalization + matching logic
├── pipelines/
│   └── enrich_players.py      # orchestrator: run this one file end to end
└── outputs/
    └── enrichment/
        ├── cache/
        │   ├── fbref/         # SoccerData caches here automatically
        │   └── tm/            # manual Transfermarkt response cache
        ├── entity_resolution/
        │   ├── name_map.csv
        │   └── unresolved.csv
        ├── player_attributes.parquet
        ├── player_attributes.csv
        ├── player_career_penalty_stats.parquet
        ├── player_career_penalty_stats.csv
        ├── penalties_enriched.parquet
        └── penalties_enriched.csv
```

---

## The orchestrator script: `pipelines/enrich_players.py`

This is the single file the user runs:

```
python pipelines/enrich_players.py
```

It must run end to end and produce all output files. Structure it with these
clearly labelled steps:

```
Step 1  Load the existing StatsBomb penalty table and player lineup reference
Step 2  Build the unique player scrape queue (deduplicated names + nationalities)
Step 3  Pull FBref career penalty stats via SoccerData (all relevant seasons)
Step 4  Pull player attributes from Transfermarkt via ScraperFC
Step 5  Run entity resolution → produce name_map.csv + unresolved.csv
Step 6  Build player_attributes.parquet
Step 7  Build player_career_penalty_stats.parquet
Step 8  Join everything back to the penalty table → penalties_enriched.parquet
Step 9  Print a summary of coverage, match rates, and unresolved counts
```

The script must add `ROOT = Path(__file__).parent.parent` and
`sys.path.insert(0, str(ROOT))` at the top so all imports work when run from
the project root.

---

## Constraints — follow all of these

**What to never do:**
- Never modify `penalties_statsbomb_clean.parquet` in place. Only write
  `penalties_enriched.parquet` as a new file.
- Never drop a row from the penalty table because enrichment data is missing.
  Every row in `penalties_statsbomb_clean.parquet` must appear in
  `penalties_enriched.parquet`, with `None` / `NaN` and a `*_is_missing` flag
  where enrichment could not be resolved.
- Never invent or guess a preferred foot. If it cannot be confirmed from a
  source, it is `None`.
- Never merge two players on name alone without a second confirming signal.
- Never hardcode FBref or SoccerData column names without first printing
  and checking the actual returned DataFrame columns. Column names differ
  between SoccerData versions.
- Never use `time.sleep()` values under 3 seconds between Transfermarkt
  requests. Use `time.sleep(3)` as the minimum.

**What to always do:**
- Cache every raw Transfermarkt response before parsing it. Check the cache
  before making any network request.
- Let SoccerData handle its own caching. Do not disable it.
- Add `*_is_missing` bool companion columns for every nullable enrichment field.
- Add `enrichment_version = "fbref_tm_v1"` as a constant column in
  `penalties_enriched.parquet` for provenance.
- Print progress at every step. The enrichment pipeline may take 30–60 minutes
  on first run (mostly Transfermarkt rate limiting).
- Handle network errors and timeouts gracefully. If a Transfermarkt request
  fails, log the failure, set all fields to None for that player, and continue
  to the next player. Do not crash.
- Use `pathlib.Path` throughout — no `os.path`.
- Use `pandas` throughout — no raw dicts as the final output format.
- Follow the same code style as the existing project: one module per source,
  private helper functions prefixed with `_`, public functions with docstrings,
  orchestrator imports from `src/`.

---

## Summary of what "done" looks like

1. `python pipelines/enrich_players.py` runs end to end without crashing.
2. `outputs/enrichment/penalties_enriched.parquet` exists and has the same
   number of rows as `outputs/statsbomb/penalties_statsbomb_clean.parquet`.
3. `outputs/enrichment/entity_resolution/name_map.csv` exists with one row
   per unique player.
4. `outputs/enrichment/entity_resolution/unresolved.csv` exists and contains
   only the players that need human review.
5. The summary printed at Step 9 shows:
   - What % of unique shooters have preferred foot resolved
   - What % of unique keepers have preferred foot resolved
   - How many players are in the unresolved list
   - The breakdown of `foot_source` values (transfermarkt / fbref_misc / unresolved)
6. `preferred_foot` in `penalties_enriched.parquet` is never a guessed value —
   every non-None value traces back to a high- or medium-confidence entity match.
