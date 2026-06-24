"""
Pipeline: StatsBomb Open Data → clean penalty dataset.

Run from your project root:
    python pipelines/build_dataset.py

What it does (mirrors the 9-cell Colab workflow):
  Step 0  Verify the data folder structure (live field-name check)
  Step 1  Load competitions + matches index
  Step 2  Scan every events file for penalty kicks
  Step 3  Load lineups for matches that had penalties
  Step 4  Normalise penalty events into a flat DataFrame
  Step 5  Add derived fields (zone, outcome labels, shootout sequence)
  Step 6  Validate (hard assertions + soft warnings + missingness flags)
  Step 7  Save outputs to outputs/statsbomb/
  Step 8  Print a summary so you can sanity-check the result

Expected runtime: 5–15 minutes (scanning ~1,000+ events files).
Expected output: ~600–1,500 penalty rows depending on the open-data release.
"""

import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids UnicodeEncodeError with checkmark/arrow chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Make conf/ and src/ importable regardless of working directory ─────────────
# Always add the project root (the folder that contains data/, conf/, src/).
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from conf.settings import (
    COMPS_FILE,
    DATA_DIR,
    EVENTS_DIR,
    LINEUPS_DIR,
    MATCHES_DIR,
    OUTPUTS_DIR,
)
from src.ingestion.statsbomb import (
    add_derived_fields,
    build_competitions_df,
    build_lineup_cache,
    build_lineup_reference,
    build_matches_df,
    build_penalties_df,
    collect_penalty_events,
)
from src.cleaning.validate import run_all_validations


# ── Step 0: Live structure verification ───────────────────────────────────────

def verify_structure() -> None:
    """
    Load one real file from each data sub-folder and print its actual JSON keys.
    This catches any field-name changes in the StatsBomb repo before they
    silently corrupt downstream columns.

    If keys printed here don't match what you expect, update:
      - conf/settings.py       (constant names)
      - src/ingestion/statsbomb.py  (.get() key paths)
    """
    SEP = "=" * 64
    print(SEP)
    print("STEP 0 — Verifying data folder structure (live key check)")
    print(SEP)

    # competitions.json
    comps = json.loads(COMPS_FILE.read_text(encoding="utf-8"))
    print(f"\n  competitions.json   {len(comps)} entries")
    print(f"  Entry keys: {list(comps[0].keys())}")

    # One matches file
    comp_dirs       = sorted(MATCHES_DIR.iterdir())
    first_match_dir = comp_dirs[0]
    match_file      = sorted(first_match_dir.iterdir())[0]
    matches_sample  = json.loads(match_file.read_text(encoding="utf-8"))
    print(f"\n  Sample matches file: {match_file.relative_to(DATA_DIR)}")
    print(f"  Entry keys: {list(matches_sample[0].keys())}")

    # Lineup file — find the first match in the sample file that has a lineup
    sample_match_id = None
    lineup_data = None
    for m in matches_sample:
        candidate_id = m["match_id"]
        candidate_file = LINEUPS_DIR / f"{candidate_id}.json"
        if candidate_file.is_file():
            sample_match_id = candidate_id
            lineup_data = json.loads(candidate_file.read_text(encoding="utf-8"))
            lineup_file = candidate_file
            break
    if sample_match_id is None:
        print("  (No lineup file found for any match in this sample — skipping lineup check.)")
        sample_match_id = matches_sample[0]["match_id"]
        lineup_data = None
    if lineup_data is not None:
        print(f"\n  Sample lineup file: lineups/{sample_match_id}.json")
        print(f"  Team-entry keys:   {list(lineup_data[0].keys())}")
        print(f"  Player-entry keys: {list(lineup_data[0]['lineup'][0].keys())}")
        positions = lineup_data[0]["lineup"][0].get("positions", [])
        if positions:
            print(f"  Position-entry keys: {list(positions[0].keys())}")

    # Events file for the same match
    events_file = EVENTS_DIR / f"{sample_match_id}.json"
    if events_file.is_file():
        events_data = json.loads(events_file.read_text(encoding="utf-8"))
        shot_events = [e for e in events_data if e.get("type", {}).get("name") == "Shot"]
        print(f"\n  Sample events file: events/{sample_match_id}.json")
        print(f"  Total events: {len(events_data)},  Shot events: {len(shot_events)}")
        if shot_events:
            print(f"  Shot top-level keys:  {list(shot_events[0].keys())}")
            print(f"  'shot' sub-obj keys:  {list((shot_events[0].get('shot') or {}).keys())}")
        else:
            print("  (No shot events in this match — that's fine, just a sample.)")
    else:
        print(f"  (No events file for match {sample_match_id} — skipping events check.)")

    print(f"\n  ✓ If these keys look correct, the pipeline is safe to continue.")
    print(f"    If anything looks wrong, fix conf/settings.py or src/ingestion/statsbomb.py")
    print(f"    BEFORE running the rest of the pipeline.")
    print(SEP)


# ── Step 8: Summary printout ──────────────────────────────────────────────────

def print_summary(penalties_df, matches_df, competitions_df) -> None:
    SEP = "=" * 64
    print(f"\n{SEP}")
    print("SUMMARY — Final dataset")
    print(SEP)

    total    = len(penalties_df)
    ingame   = (~penalties_df["is_shootout"]).sum()
    shootout = penalties_df["is_shootout"].sum()
    conv     = penalties_df["outcome_bin"].mean()

    print(f"\n  Total penalties:          {total:,}")
    print(f"    In-game (periods 1–4):  {ingame:,}")
    print(f"    Shootout (period 5):    {shootout:,}")
    print(f"  Unique shooters:          {penalties_df['shooter_id'].nunique():,}")
    print(f"  Unique keepers resolved:  {penalties_df['keeper_id'].nunique():,}")
    print(f"  Competitions covered:     {penalties_df['competition_name'].nunique():,}")
    print(f"  Matches with penalties:   {penalties_df['match_id'].nunique():,}")
    print(f"\n  Overall conversion rate:  {conv:.1%}  (expect ~75–82% — if very different, investigate)")

    print("\n  Penalties per competition:")
    for name, count in penalties_df["competition_name"].value_counts().items():
        print(f"    {str(name):<48s} {count:>4d}")

    print("\n  Shot outcome distribution:")
    for outcome, count in penalties_df["shot_outcome_name"].value_counts(dropna=False).items():
        print(f"    {str(outcome):<22s} {count:>4d}")

    print("\n  Shot zone distribution (validate the boundary constants in settings.py):")
    for zone, count in penalties_df["shot_zone"].value_counts(dropna=False).items():
        pct = count / total * 100
        print(f"    {str(zone):<16s} {count:>4d}  ({pct:.0f}%)")

    print("\n  Freeze frame available:")
    ff_rate = penalties_df["gk_freeze_available"].mean()
    print(f"    {ff_rate:.1%} of kicks have keeper position data")

    print(f"\n  Shootout kicks with sequence derived: "
          f"{penalties_df['shootout_kick_index'].notna().sum():,}")

    # Spot-check: print one full shootout so you can cross-reference it
    # against a known result (e.g. Wikipedia) to verify kick ordering is correct.
    so_matches = penalties_df[penalties_df["is_shootout"]]["match_id"].unique()
    if len(so_matches) > 0:
        example_mid = so_matches[0]
        so_kicks = (
            penalties_df[penalties_df["match_id"] == example_mid]
            .sort_values("shootout_kick_index")[
                ["shootout_kick_index", "shooter_team_name",
                 "shooter_name", "shot_outcome_name",
                 "shootout_score_for_before", "shootout_score_against_before"]
            ]
        )
        comp = penalties_df[penalties_df["match_id"] == example_mid]["competition_name"].iloc[0]
        date = penalties_df[penalties_df["match_id"] == example_mid]["match_date"].iloc[0]
        print(f"\n  Spot-check — shootout in match {example_mid} ({comp}, {date}):")
        print(f"  Cross-reference this against a known result to verify ordering.")
        print(so_kicks.to_string(index=False))

    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Verify structure ─────────────────────────────────────────────
    verify_structure()

    # ── Step 1: Competitions + matches ───────────────────────────────────────
    print("\nSTEP 1 — Loading competitions and matches")
    competitions_df = build_competitions_df()
    print(f"  Competitions: {len(competitions_df)} competition-season pairs")

    matches_df = build_matches_df(competitions_df)
    print(f"  Matches:      {len(matches_df):,} unique matches")

    # ── Step 2: Collect penalty events ───────────────────────────────────────
    print("\nSTEP 2 — Scanning events files for penalty kicks")
    print("  (This is the slowest step — scanning all matches. ~5–15 min.)")
    penalty_event_rows, failed_ids = collect_penalty_events(matches_df)
    print(f"  Found {len(penalty_event_rows):,} penalty events")
    if failed_ids:
        print(f"  ⚠  {len(failed_ids)} matches had no events file (not an error — some")
        print("     competitions in competitions.json have no matching events yet).")

    if len(penalty_event_rows) == 0:
        print("\n  ERROR: No penalty events found.")
        print("  Check that your data/events/ folder is populated and that")
        print("  the events files contain Shot events with shot.type.name == 'Penalty'.")
        return

    # ── Step 3: Lineups (penalty matches only) ────────────────────────────────
    print("\nSTEP 3 — Loading lineups for matches with penalties")
    penalty_match_ids = sorted({r["match_id"] for r in penalty_event_rows})
    print(f"  {len(penalty_match_ids)} matches to load")
    lineup_cache = build_lineup_cache(penalty_match_ids)

    # ── Step 4 + 5: Build and derive ─────────────────────────────────────────
    print("\nSTEP 4 — Normalising and deriving fields")
    penalties_df = build_penalties_df(penalty_event_rows, lineup_cache, matches_df)
    penalties_df = add_derived_fields(penalties_df)
    print(f"  penalties_df shape: {penalties_df.shape}")

    # ── Step 6: Validate ──────────────────────────────────────────────────────
    print("\nSTEP 6 — Validating")
    penalties_df = run_all_validations(penalties_df)

    # ── Step 7: Save ─────────────────────────────────────────────────────────
    print(f"\nSTEP 7 — Saving to {OUTPUTS_DIR.relative_to(ROOT)}/")

    # (a) Lean CSV + Parquet — the main deliverable, no raw freeze frame
    lean_df = penalties_df.drop(columns=["shot_freeze_frame_raw"])
    lean_df.to_csv(OUTPUTS_DIR / "penalties_statsbomb_clean.csv", index=False)
    lean_df.to_parquet(OUTPUTS_DIR / "penalties_statsbomb_clean.parquet", index=False)
    print("  ✓ penalties_statsbomb_clean.csv + .parquet")

    # (b) Full Parquet with raw freeze frame (for future deeper use)
    penalties_df.to_parquet(
        OUTPUTS_DIR / "penalties_statsbomb_with_freeze_frame.parquet",
        index=False,
    )
    print("  ✓ penalties_statsbomb_with_freeze_frame.parquet")

    # (c) Match dimension table
    matches_df.to_csv(OUTPUTS_DIR / "matches_statsbomb.csv", index=False)
    matches_df.to_parquet(OUTPUTS_DIR / "matches_statsbomb.parquet", index=False)
    print("  ✓ matches_statsbomb.csv + .parquet")

    # (d) Player reference table (for attribute enrichment from FBref / Transfermarkt)
    lineup_ref = build_lineup_reference(lineup_cache)
    lineup_ref.to_csv(OUTPUTS_DIR / "lineups_statsbomb_penalty_matches.csv", index=False)
    lineup_ref.to_parquet(
        OUTPUTS_DIR / "lineups_statsbomb_penalty_matches.parquet",
        index=False,
    )
    print("  ✓ lineups_statsbomb_penalty_matches.csv + .parquet")

    # ── Step 8: Summary ───────────────────────────────────────────────────────
    print_summary(penalties_df, matches_df, competitions_df)

    print("\nDone. Your clean penalty dataset is in outputs/statsbomb/")
    print("Next: open penalties_statsbomb_clean.parquet and run EDA (Step 4 of the plan).")


if __name__ == "__main__":
    main()