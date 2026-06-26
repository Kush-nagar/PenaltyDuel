"""
PenaltyDuel FastAPI serving application.

Endpoints:
  GET  /health              — liveness check
  GET  /players/shooters    — searchable shooter list
  GET  /players/keepers     — searchable keeper list
  POST /predict             — matchup prediction

Run:
  uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features.build import build_modeling_table
from src.models.baselines import add_baseline_predictions
from src.serving.predict import (
    PlayerIndex,
    MatchupPrediction,
    build_player_index,
    predict_matchup,
    search_players,
    SHOT_ZONES,
    DIVE_DIRS,
)
from src.enrichment.entity_resolution import normalize_name
from src.explain.shap_wrap import build_explainer, explain_matchup
from src.explain.cards import build_shap_card

# ── Paths ─────────────────────────────────────────────────────────────────────

ENRICHED_PATH = ROOT / "outputs" / "enrichment" / "penalties_enriched.parquet"
PIPELINE_PATH = ROOT / "outputs" / "model" / "pipeline.pkl"
DIVE_DISTS_PATH = ROOT / "outputs" / "enrichment" / "keeper_dive_distributions.json"
METADATA_PATH = ROOT / "outputs" / "model" / "metadata.json"


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    pipeline = None
    player_index: PlayerIndex | None = None
    metadata: dict = {}
    shap_explainer = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading pipeline...")
    state.pipeline = joblib.load(PIPELINE_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            state.metadata = json.load(f)

    print("Loading enriched dataset and building player index...")
    df = pd.read_parquet(ENRICHED_PATH)

    kdd: dict = {}
    if DIVE_DISTS_PATH.exists():
        with open(DIVE_DISTS_PATH) as f:
            kdd = json.load(f)

    mt = build_modeling_table(df, keeper_dive_distributions=kdd)
    mt = add_baseline_predictions(mt)
    state.player_index = build_player_index(df, mt, kdd)

    n_s = len(state.player_index.shooters)
    n_k = len(state.player_index.keepers)
    print(f"Ready: {n_s} shooters, {n_k} keepers indexed.")

    print("Building SHAP explainer...")
    state.shap_explainer = build_explainer(state.pipeline.combined_model, mt)
    print("SHAP explainer ready.")
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="PenaltyDuel API",
    description="Hierarchical penalty outcome prediction",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    shooter_name: str
    keeper_name: str


class ZoneProbability(BaseModel):
    zone: str
    prob: float
    goal_prob: float   # P(goal | this zone, keeper's dominant dive)


class ShapFactor(BaseModel):
    feature: str
    label: str
    shap_value: float
    direction: str
    pct_impact: float


class ShapCard(BaseModel):
    headline: str
    confidence: str
    low_data_warning: bool
    low_data_note: str | None
    shooter_summary: str
    keeper_summary: str
    top_factors: list[str]
    factors: list[ShapFactor]


class PredictResponse(BaseModel):
    goal_probability: float
    composed_probability: float
    explanation: str
    shooter: dict
    keeper: dict
    zone_distribution: list[ZoneProbability]
    dive_distribution: list[dict]
    shap_card: ShapCard


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    meta = state.metadata
    return {
        "status": "ok",
        "n_shooters": len(state.player_index.shooters) if state.player_index else 0,
        "n_keepers": len(state.player_index.keepers) if state.player_index else 0,
        "global_goal_rate": meta.get("global_goal_rate"),
        "n_penalties_trained": meta.get("n_penalties"),
    }


@app.get("/players/shooters")
def list_shooters(q: str = Query(default="", min_length=0)):
    if state.player_index is None:
        raise HTTPException(503, "Index not ready")
    if not q:
        # Return top 50 by penalty count
        profiles = sorted(
            state.player_index.shooters.values(),
            key=lambda p: p.n_penalties,
            reverse=True,
        )[:50]
        return [
            {"id": p.player_id, "name": p.name, "n_penalties": p.n_penalties}
            for p in profiles
        ]
    results = search_players(
        q,
        state.player_index.shooter_name_index,
        state.player_index.shooters,
    )
    return results


@app.get("/players/keepers")
def list_keepers(q: str = Query(default="", min_length=0)):
    if state.player_index is None:
        raise HTTPException(503, "Index not ready")
    if not q:
        profiles = sorted(
            state.player_index.keepers.values(),
            key=lambda p: p.n_faced,
            reverse=True,
        )[:50]
        return [
            {"id": p.player_id, "name": p.name, "n_faced": p.n_faced}
            for p in profiles
        ]
    results = search_players(
        q,
        state.player_index.keeper_name_index,
        state.player_index.keepers,
    )
    return results


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if state.player_index is None or state.pipeline is None:
        raise HTTPException(503, "Model not ready")

    idx = state.player_index

    # Resolve shooter
    shooter_norm = normalize_name(req.shooter_name) or ""
    shooter_id = idx.shooter_name_index.get(shooter_norm)
    if shooter_id is None:
        results = search_players(req.shooter_name, idx.shooter_name_index, idx.shooters, limit=1)
        if not results:
            raise HTTPException(404, f"Shooter not found: {req.shooter_name!r}")
        shooter_id = results[0]["id"]
    shooter = idx.shooters[shooter_id]

    # Resolve keeper
    keeper_norm = normalize_name(req.keeper_name) or ""
    keeper_id = idx.keeper_name_index.get(keeper_norm)
    if keeper_id is None:
        results = search_players(req.keeper_name, idx.keeper_name_index, idx.keepers, limit=1)
        if not results:
            raise HTTPException(404, f"Keeper not found: {req.keeper_name!r}")
        keeper_id = results[0]["id"]
    keeper = idx.keepers[keeper_id]

    prediction: MatchupPrediction = predict_matchup(shooter, keeper, state.pipeline)

    shap_factors = explain_matchup(state.shap_explainer, shooter, keeper, state.pipeline)
    shap_card_data = build_shap_card(
        shap_factors, shooter, keeper,
        goal_prob=prediction.goal_probability,
        global_rate=state.pipeline.combined_model.global_rate,
    )
    shap_card = ShapCard(
        headline=shap_card_data["headline"],
        confidence=shap_card_data["confidence"],
        low_data_warning=shap_card_data["low_data_warning"],
        low_data_note=shap_card_data["low_data_note"],
        shooter_summary=shap_card_data["shooter_summary"],
        keeper_summary=shap_card_data["keeper_summary"],
        top_factors=shap_card_data["top_factors"],
        factors=[ShapFactor(**f) for f in shap_card_data["factors"]],
    )

    zone_dist = [
        ZoneProbability(
            zone=z,
            prob=round(prediction.zone_probs.get(z, 0.0), 4),
            goal_prob=round(prediction.zone_dive_table.get(z, 0.74), 4),
        )
        for z in SHOT_ZONES
    ]

    dive_dist = [
        {"direction": d, "prob": round(prediction.dive_probs.get(d, 1/3), 4)}
        for d in DIVE_DIRS
    ]

    return PredictResponse(
        goal_probability=round(prediction.goal_probability, 4),
        composed_probability=round(prediction.composed_probability, 4),
        explanation=prediction.explanation,
        shooter={
            "id": shooter.player_id,
            "name": shooter.name,
            "n_penalties": shooter.n_penalties,
            "conversion_rate": round(shooter.conv_rate_shrunk, 4),
            "preferred_foot": shooter.preferred_foot,
            "height_cm": shooter.height_cm,
        },
        keeper={
            "id": keeper.player_id,
            "name": keeper.name,
            "n_faced": keeper.n_faced,
            "save_rate": round(keeper.save_rate_shrunk, 4),
            "preferred_foot": keeper.preferred_foot,
            "height_cm": keeper.height_cm,
        },
        zone_distribution=zone_dist,
        dive_distribution=dive_dist,
        shap_card=shap_card,
    )
