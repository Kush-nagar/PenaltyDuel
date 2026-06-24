"""
Validation and quality-check functions for the cleaned penalties DataFrame.

Design rule: this module only reads and checks — it never saves files and
never imputes values.  Imputation is a feature-engineering decision (Step 5
of the implementation plan), not a cleaning one.

Call run_all_validations(df) from the pipeline.
"""

import pandas as pd

from conf.settings import OUTCOME_3_MAP


# ── Individual checks ──────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame) -> None:
    """
    Hard assertion: every penalty must have a unique penalty_id and a
    unique source_event_id.  A failure here almost always means the same
    match was scanned twice — look for duplicate match_ids in matches_df.
    """
    n_pid = df["penalty_id"].duplicated().sum()
    assert n_pid == 0, (
        f"Found {n_pid} duplicate penalty_id values. "
        "Check for duplicate match_ids in matches_df."
    )
    n_eid = df["source_event_id"].duplicated().sum()
    assert n_eid == 0, (
        f"Found {n_eid} duplicate source_event_id values."
    )


def check_outcome_vocabulary(df: pd.DataFrame) -> None:
    """
    Warn (not fail) if any shot_outcome_name values fall outside the known
    vocabulary.  Those rows will have outcome_3 == "unknown", which will
    be excluded from the 3-class placement model but kept for outcome_bin.
    """
    known    = set(OUTCOME_3_MAP.keys())
    observed = set(df["shot_outcome_name"].dropna().unique())
    unmapped = observed - known
    if unmapped:
        print(f"  ⚠  Unmapped shot_outcome_name values: {unmapped}")
        print("     These rows get outcome_3='unknown'. Update OUTCOME_3_MAP in conf/settings.py if needed.")


def check_period_values(df: pd.DataFrame) -> None:
    """
    Print a period breakdown.
    Verify that period == 5 really does mean shootout kicks for your data.
    This is a well-established StatsBomb convention but worth confirming.
    """
    print("\n  Period breakdown (confirm period 5 = shootout):")
    for period, count in df["period"].value_counts().sort_index().items():
        tag = "  ← shootout (StatsBomb convention)" if period == 5 else ""
        print(f"    period {period}: {count:>5,d} kicks{tag}")


def check_keeper_resolution(df: pd.DataFrame) -> None:
    """
    Print the keeper resolution method breakdown.
    Inspect the 'unresolved' rate before trusting keeper_id downstream.
    """
    print("\n  Keeper resolution methods:")
    vc = df["keeper_resolution_method"].value_counts(dropna=False)
    for method, count in vc.items():
        pct = count / len(df) * 100
        print(f"    {str(method):<40s} {count:>5,d}  ({pct:.1f}%)")
    n_unresolved = (df["keeper_resolution_method"] == "unresolved").sum()
    if n_unresolved > 0:
        print(f"  ⚠  {n_unresolved} penalties have an unresolved keeper_id.")
        print("     These rows will not contribute to the keeper-side model.")


def check_ranges(df: pd.DataFrame) -> None:
    """Hard assertions on values that must always be valid."""
    assert df["outcome_bin"].isin([0, 1]).all(), \
        "outcome_bin must be 0 or 1 for every row."
    assert df["period"].notna().all(), \
        "Every penalty event must have a non-null period."
    n_unknown_3 = (df["outcome_3"] == "unknown").sum()
    if n_unknown_3 > 0:
        print(f"  ⚠  {n_unknown_3} rows have outcome_3 == 'unknown'.")


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean *_is_missing companion column for each field that can
    legitimately be absent in the StatsBomb open data.
    No values are imputed here — that belongs in the feature engineering phase.
    """
    df = df.copy()
    for col in [
        "shot_end_location_x",
        "shot_end_location_y",
        "shot_end_location_z",
        "keeper_id",
        "gk_freeze_location_x",
        "gk_freeze_location_y",
        "shot_zone",
    ]:
        df[f"{col}_is_missing"] = df[col].isna()
    return df


# ── Main entry point ───────────────────────────────────────────────────────────

def run_all_validations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every validation check against the cleaned penalties DataFrame.
    Hard checks raise AssertionError on failure.
    Soft checks print a warning and continue.
    Returns the DataFrame with missingness flags added.
    """
    print("\nRunning validations ...")

    check_duplicates(df)
    print("  ✓ No duplicate penalty_id or source_event_id")

    check_ranges(df)
    print("  ✓ outcome_bin ∈ {0,1} and period non-null")

    check_outcome_vocabulary(df)
    check_period_values(df)
    check_keeper_resolution(df)

    df = add_missingness_flags(df)

    print("\n  Missingness rates (key fields):")
    miss_cols = [c for c in df.columns if c.endswith("_is_missing")]
    for col in miss_cols:
        rate = df[col].mean()
        flag = "  ← expected" if "z_is_missing" in col else ""
        print(f"    {col:<45s} {rate:>6.1%}{flag}")

    print("\n  ✓ All validations complete.")
    return df