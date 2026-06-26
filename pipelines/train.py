"""
Train and serialize the full outcome pipeline.

Trains on all available enriched data, saves pipeline artifact + metadata.
Run this before starting the API server.

Usage:
    PYTHONIOENCODING=utf-8 python pipelines/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.features.build import build_modeling_table
from src.models.baselines import add_baseline_predictions
from src.models.training import fit_pipeline

ENRICHED_PATH = ROOT / "outputs" / "enrichment" / "penalties_enriched.parquet"
DIVE_DISTS_PATH = ROOT / "outputs" / "enrichment" / "keeper_dive_distributions.json"
DIVE_TABLE_PATH = ROOT / "outputs" / "enrichment" / "dive_resolution_table.json"
MODEL_DIR = ROOT / "outputs" / "model"
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def run() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {ENRICHED_PATH}")
    df = pd.read_parquet(ENRICHED_PATH)
    print(f"  {len(df)} penalties, {df['shooter_id'].nunique()} shooters, {df['keeper_id'].nunique()} keepers")

    kdd: dict | None = None
    if DIVE_DISTS_PATH.exists():
        with open(DIVE_DISTS_PATH) as f:
            kdd = json.load(f)
        print(f"  Loaded dive distributions for {len(kdd)} keepers")

    print("Building modeling table...")
    mt = build_modeling_table(df, keeper_dive_distributions=kdd)
    mt = add_baseline_predictions(mt)
    print(f"  {len(mt)} rows, {len(mt.columns)} columns")

    # Load pre-built dive resolution table from Kaggle WC data (enrich_dive.py)
    dive_table: dict | None = None
    if DIVE_TABLE_PATH.exists():
        with open(DIVE_TABLE_PATH) as f:
            raw = json.load(f)
        # Deserialize: "zone|dive" keys -> (zone, dive) tuple keys
        dive_table = {tuple(k.split("|")): v for k, v in raw.items()}
        print(f"  Loaded dive resolution table: {len(dive_table)} cells")

    print("Training pipeline on full dataset...")
    pipeline = fit_pipeline(mt)
    # Inject pre-built WC dive table (StatsBomb data has no kick-level dive labels)
    if dive_table is not None:
        pipeline.dive_table = dive_table
    print(f"  Calibration method: {pipeline.calibration_method}")
    print(f"  Global goal rate: {pipeline.global_rate:.3f}")
    print(f"  Dive table cells: {len(pipeline.dive_table) if pipeline.dive_table else 0}")

    print(f"Saving pipeline -> {PIPELINE_PATH}")
    joblib.dump(pipeline, PIPELINE_PATH)

    metadata = {
        "n_penalties": len(df),
        "n_shooters": int(df["shooter_id"].nunique()),
        "n_keepers": int(df["keeper_id"].nunique()),
        "global_goal_rate": pipeline.global_rate,
        "calibration_method": pipeline.calibration_method,
        "has_dive_table": pipeline.dive_table is not None,
        "dive_table_cells": len(pipeline.dive_table) if pipeline.dive_table else 0,
        "dive_table_source": "kaggle_world_cup_shootouts" if dive_table else None,
        "feature_names": pipeline.combined_model.feature_names,
        "date_range": {
            "min": str(df["match_date"].min()),
            "max": str(df["match_date"].max()),
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {METADATA_PATH}")
    print("\nDone. Pipeline ready for serving.")


if __name__ == "__main__":
    run()
