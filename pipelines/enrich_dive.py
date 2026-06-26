"""
Dive enrichment pipeline (Step 8.5).

Loads both Kaggle dive datasets, builds keeper dive distributions,
attaches P(dive|keeper) features to penalties_enriched.parquet,
and saves an extended resolution table for the full zone×dive formula.

Usage:
    python pipelines/enrich_dive.py
    python pipelines/enrich_dive.py --rodrigo data/kaggle/real_data.xlsx
                                    --pablo   data/kaggle/WorldCupShootouts.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.enrichment.entity_resolution import normalize_name
from src.ingestion.kaggle_dive import (
    DIVE_DIRECTIONS,
    build_keeper_dive_counts,
    build_population_prior_counts,
    load_pablo,
    load_rodrigo,
)
from src.models.dive_prior import (
    attach_dive_features,
    build_population_prior,
    estimate_keeper_dive_distributions,
)
from src.models.resolution import fit_dive_resolution_table

DEFAULT_RODRIGO = ROOT / "data" / "kaggle" / "real_data.xlsx"
DEFAULT_PABLO = ROOT / "data" / "kaggle" / "WorldCupShootouts.csv"
ENRICHED_PATH = ROOT / "outputs" / "enrichment" / "penalties_enriched.parquet"
DIVE_DISTS_PATH = ROOT / "outputs" / "enrichment" / "keeper_dive_distributions.json"
DIVE_TABLE_PATH = ROOT / "outputs" / "enrichment" / "dive_resolution_table.json"


def _match_keeper_to_statsbomb(
    keeper_dive_counts: dict[str, dict[str, float]],
    statsbomb_keepers: set[str],
    *,
    fuzzy_threshold: float = 85.0,
) -> dict[str, dict[str, float]]:
    """
    Map DS1 keeper dive counts to StatsBomb keeper_name_normalized keys.

    Exact match first; fuzzy fallback using rapidfuzz token_sort_ratio.
    Returns {statsbomb_keeper_name_normalized: {"left": n, "center": n, "right": n}}.
    """
    try:
        from rapidfuzz.fuzz import token_sort_ratio
    except ImportError:
        token_sort_ratio = None

    result: dict[str, dict[str, float]] = {}

    for sb_keeper in statsbomb_keepers:
        if not sb_keeper:
            continue
        # Exact match
        if sb_keeper in keeper_dive_counts:
            result[sb_keeper] = keeper_dive_counts[sb_keeper]
            continue
        # Fuzzy match
        if token_sort_ratio is None:
            continue
        best_score, best_key = 0.0, None
        for kaggle_keeper in keeper_dive_counts:
            score = token_sort_ratio(sb_keeper, kaggle_keeper)
            if score > best_score:
                best_score, best_key = score, kaggle_keeper
        if best_score >= fuzzy_threshold and best_key is not None:
            result[sb_keeper] = keeper_dive_counts[best_key]

    return result


def run(
    rodrigo_path: Path = DEFAULT_RODRIGO,
    pablo_path: Path = DEFAULT_PABLO,
) -> None:
    print(f"Loading StatsBomb penalties from {ENRICHED_PATH}")
    statsbomb_df = pd.read_parquet(ENRICHED_PATH)
    print(f"  {len(statsbomb_df)} penalties, {statsbomb_df['keeper_id'].nunique()} unique keepers")

    # ── Load Kaggle datasets ──────────────────────────────────────────────────
    print(f"\nLoading DS1 (rodrigoarede2003): {rodrigo_path}")
    rodrigo_df = load_rodrigo(str(rodrigo_path))
    valid_rodrigo = rodrigo_df[rodrigo_df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    print(f"  {len(rodrigo_df)} kicks loaded, {len(valid_rodrigo)} with valid dive labels")
    print(f"  Unique keepers in DS1: {rodrigo_df['keeper_name_normalized'].nunique()}")

    print(f"\nLoading DS2 (pablollanderos33): {pablo_path}")
    pablo_df = load_pablo(str(pablo_path))
    valid_pablo = pablo_df[pablo_df["keeper_dive_direction"].isin(DIVE_DIRECTIONS)]
    print(f"  {len(pablo_df)} kicks loaded, {len(valid_pablo)} with valid dive labels")

    # ── Population prior from DS2 ─────────────────────────────────────────────
    print("\nBuilding population dive prior from DS2 WC data...")
    pop_counts = build_population_prior_counts(pablo_df)
    population_prior = {d: max(pop_counts.get(d, 1e-6), 1e-6) for d in DIVE_DIRECTIONS}
    total_prior = sum(population_prior.values())
    print(f"  Prior (normalized %): { {d: round(population_prior[d]/total_prior*100,1) for d in DIVE_DIRECTIONS} }")

    # ── Per-keeper dive counts from DS1 ───────────────────────────────────────
    print("\nBuilding per-keeper dive counts from DS1...")
    keeper_dive_counts = build_keeper_dive_counts(rodrigo_df)
    print(f"  DS1 keepers with dive data: {len(keeper_dive_counts)}")

    # ── Match DS1 keepers -> StatsBomb keepers ────────────────────────────────
    statsbomb_keepers = set(
        statsbomb_df["keeper_name_normalized"].dropna().unique()
    )
    print(f"\nMatching DS1 keeper names to {len(statsbomb_keepers)} StatsBomb keepers...")
    matched_counts = _match_keeper_to_statsbomb(keeper_dive_counts, statsbomb_keepers)
    print(f"  Matched: {len(matched_counts)} keepers ({len(matched_counts)/len(statsbomb_keepers)*100:.1f}%)")
    if matched_counts:
        print("  Sample matches:", list(matched_counts.keys())[:5])

    # ── Estimate P(dive|keeper) via Dirichlet ─────────────────────────────────
    print("\nEstimating keeper dive distributions (Dirichlet-shrunk)...")
    # Attach DS1 counts to statsbomb_df via keeper_id using name->id lookup
    name_to_id = (
        statsbomb_df[["keeper_name_normalized", "keeper_id"]]
        .dropna()
        .drop_duplicates("keeper_name_normalized")
        .set_index("keeper_name_normalized")["keeper_id"]
        .to_dict()
    )
    # Build a temporary df with keeper_id and dive direction from matched DS1 data
    kaggle_with_id_rows = []
    for name, counts in matched_counts.items():
        kid = name_to_id.get(name)
        if kid is None:
            continue
        for dive, n in counts.items():
            for _ in range(int(n)):
                kaggle_with_id_rows.append({"keeper_id": kid, "keeper_dive_direction": dive})
    kaggle_with_id = pd.DataFrame(kaggle_with_id_rows) if kaggle_with_id_rows else pd.DataFrame(
        columns=["keeper_id", "keeper_dive_direction"]
    )

    keeper_dive_distributions = estimate_keeper_dive_distributions(
        statsbomb_df,
        kaggle_with_id,
        population_prior=population_prior,
    )
    print(f"  Distributions estimated for {len(keeper_dive_distributions)} keepers")

    # ── Attach dive features to penalties_enriched.parquet ───────────────────
    print("\nAttaching dive features to penalties_enriched.parquet...")
    enriched = attach_dive_features(statsbomb_df, keeper_dive_distributions)
    print(f"  keeper_dive_left_prob: {enriched['keeper_dive_left_prob'].describe().to_dict()}")

    # ── Zone×dive resolution table from DS2 ───────────────────────────────────
    print("\nFitting zone×dive resolution table from DS2...")
    pablo_for_table = pablo_df.rename(columns={"outcome_bin": "outcome_bin"})
    dive_table = fit_dive_resolution_table(pablo_for_table)
    if dive_table is not None:
        labeled_coverage = len(valid_pablo) / max(len(pablo_df), 1)
        print(f"  Coverage: {labeled_coverage:.1%} -> table fitted ({len(dive_table)} cells)")
        dive_table_serializable = {f"{z}|{d}": v for (z, d), v in dive_table.items()}
    else:
        print("  Coverage below threshold -> dive table not fitted")
        dive_table_serializable = {}

    # ── Save outputs ───────────────────────────────────────────────────────────
    print(f"\nSaving enriched parquet -> {ENRICHED_PATH}")
    enriched.to_parquet(ENRICHED_PATH, index=False)
    enriched.to_csv(ENRICHED_PATH.with_suffix(".csv"), index=False)

    print(f"Saving keeper dive distributions -> {DIVE_DISTS_PATH}")
    with open(DIVE_DISTS_PATH, "w") as f:
        json.dump(keeper_dive_distributions, f, indent=2)

    if dive_table_serializable:
        print(f"Saving dive resolution table -> {DIVE_TABLE_PATH}")
        with open(DIVE_TABLE_PATH, "w") as f:
            json.dump(dive_table_serializable, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Dive Enrichment Summary ===")
    print(f"DS1 kicks used:              {len(valid_rodrigo)}")
    print(f"DS2 kicks used:              {len(valid_pablo)}")
    print(f"StatsBomb keepers matched:   {len(matched_counts)} / {len(statsbomb_keepers)}")
    print(f"Keepers with DS1 data:       {len(matched_counts)}")
    print(f"Dive table cells:            {len(dive_table_serializable)}")
    print(f"New feature cols added:      keeper_dive_left_prob, keeper_dive_center_prob, keeper_dive_right_prob, keeper_dive_entropy")
    print(f"keeper_dive_direction:       {enriched['keeper_dive_direction'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich penalties with keeper dive distributions")
    parser.add_argument("--rodrigo", default=str(DEFAULT_RODRIGO), help="Path to real_data.xlsx")
    parser.add_argument("--pablo", default=str(DEFAULT_PABLO), help="Path to WorldCupShootouts.csv")
    args = parser.parse_args()
    run(Path(args.rodrigo), Path(args.pablo))
