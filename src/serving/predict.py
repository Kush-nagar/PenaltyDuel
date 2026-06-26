"""
Serving layer: build per-player feature caches and assemble matchup predictions.

At startup: builds a player stats cache from the full enriched dataset so that
individual matchup predictions don't require re-running the as-of feature pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

ENRICHED_PATH = ROOT / "outputs" / "enrichment" / "penalties_enriched.parquet"
PIPELINE_PATH = ROOT / "outputs" / "model" / "pipeline.pkl"
DIVE_DISTS_PATH = ROOT / "outputs" / "enrichment" / "keeper_dive_distributions.json"

SHOT_ZONES = [
    "low_left", "low_center", "low_right",
    "high_left", "high_center", "high_right",
]
DIVE_DIRS = ["left", "center", "right"]
EPS = 1e-6


# ── Player stat caches ────────────────────────────────────────────────────────

@dataclass
class ShooterProfile:
    player_id: str
    name: str
    name_normalized: str
    n_penalties: int
    conv_rate_shrunk: float
    zone_probs: dict[str, float]
    zone_entropy: float
    preferred_foot: str | None
    height_cm: float | None
    foot_is_right: float
    foot_is_missing: float


@dataclass
class KeeperProfile:
    player_id: str
    name: str
    name_normalized: str
    n_faced: int
    save_rate_shrunk: float
    dive_probs: dict[str, float]
    dive_entropy: float
    preferred_foot: str | None
    height_cm: float | None
    foot_is_right: float
    foot_is_missing: float


@dataclass
class MatchupPrediction:
    goal_probability: float
    composed_probability: float
    shooter: ShooterProfile
    keeper: KeeperProfile
    zone_probs: dict[str, float]       # shooter zone distribution
    dive_probs: dict[str, float]       # keeper dive distribution
    zone_dive_table: dict[str, float]  # P(goal | zone, best_dive) for heatmap
    explanation: str


@dataclass
class PlayerIndex:
    shooters: dict[str, ShooterProfile] = field(default_factory=dict)   # id -> profile
    keepers: dict[str, KeeperProfile] = field(default_factory=dict)
    shooter_name_index: dict[str, str] = field(default_factory=dict)    # normalized_name -> id
    keeper_name_index: dict[str, str] = field(default_factory=dict)


def _entropy(probs: dict[str, float]) -> float:
    import math
    vals = [v for v in probs.values() if v > 0]
    if len(vals) <= 1:
        return 0.0
    total = sum(vals)
    norm = [v / total for v in vals]
    raw = -sum(v * math.log(v) for v in norm)
    return raw / math.log(len(norm))


def build_player_index(
    df: pd.DataFrame,
    modeling_table: pd.DataFrame,
    keeper_dive_distributions: dict[str, dict[str, float]],
) -> PlayerIndex:
    """
    Build shooter and keeper profile caches from the modeling table.

    Takes the LATEST row per player (sorted by match_date) as their
    current career state. As-of features in that row represent all kicks
    up to but not including the last one — close enough for serving.
    """
    idx = PlayerIndex()

    ordered = modeling_table.copy()
    ordered["_date"] = pd.to_datetime(ordered["match_date"], errors="coerce")
    ordered = ordered.sort_values(["_date", "penalty_id"], kind="mergesort")

    # ── Shooter profiles ──────────────────────────────────────────────────────
    shooter_last = ordered.drop_duplicates("shooter_id", keep="last")
    for _, row in shooter_last.iterrows():
        sid = str(row["shooter_id"])
        name = str(row.get("shooter_name", sid))
        name_norm = str(row.get("shooter_name_normalized", ""))
        zone_probs = {z: float(row.get(f"shooter_zone_prob_{z}", 1/6)) for z in SHOT_ZONES}
        profile = ShooterProfile(
            player_id=sid,
            name=name,
            name_normalized=name_norm,
            n_penalties=int(row.get("shooter_n_pens_before", 0)),
            conv_rate_shrunk=float(row.get("shooter_conv_rate_shrunk", 0.74)),
            zone_probs=zone_probs,
            zone_entropy=float(row.get("shooter_zone_entropy", _entropy(zone_probs))),
            preferred_foot=row.get("shooter_preferred_foot") if pd.notna(row.get("shooter_preferred_foot")) else None,
            height_cm=float(row["shooter_height_cm"]) if pd.notna(row.get("shooter_height_cm")) else None,
            foot_is_right=float(row.get("shooter_foot_is_right", 0.0)),
            foot_is_missing=float(row.get("shooter_preferred_foot_is_missing", 1.0)),
        )
        idx.shooters[sid] = profile
        if name_norm:
            idx.shooter_name_index[name_norm] = sid

    # ── Keeper profiles ───────────────────────────────────────────────────────
    keeper_last = ordered.dropna(subset=["keeper_id"]).drop_duplicates("keeper_id", keep="last")
    for _, row in keeper_last.iterrows():
        kid = str(row["keeper_id"])
        name = str(row.get("keeper_name", kid))
        name_norm = str(row.get("keeper_name_normalized", ""))
        dive_dist = keeper_dive_distributions.get(kid, {d: 1/3 for d in DIVE_DIRS})
        profile = KeeperProfile(
            player_id=kid,
            name=name,
            name_normalized=name_norm,
            n_faced=int(row.get("keeper_n_faced_before", 0)),
            save_rate_shrunk=float(row.get("keeper_save_rate_shrunk", 0.26)),
            dive_probs=dive_dist,
            dive_entropy=_entropy(dive_dist),
            preferred_foot=row.get("keeper_preferred_foot") if pd.notna(row.get("keeper_preferred_foot")) else None,
            height_cm=float(row["keeper_height_cm"]) if pd.notna(row.get("keeper_height_cm")) else None,
            foot_is_right=float(row.get("keeper_foot_is_right", 0.0)),
            foot_is_missing=float(row.get("keeper_preferred_foot_is_missing", 1.0)),
        )
        idx.keepers[kid] = profile
        if name_norm:
            idx.keeper_name_index[name_norm] = kid

    return idx


# ── Name search ───────────────────────────────────────────────────────────────

def search_players(
    query: str,
    name_index: dict[str, str],
    profiles: dict[str, ShooterProfile] | dict[str, KeeperProfile],
    *,
    limit: int = 10,
) -> list[dict]:
    """Fuzzy name search. Returns list of {id, name, score} dicts."""
    from rapidfuzz.fuzz import WRatio
    from src.enrichment.entity_resolution import normalize_name

    q_norm = normalize_name(query) or ""
    if not q_norm:
        return []

    scored = []
    for name_norm, pid in name_index.items():
        score = WRatio(q_norm, name_norm)
        scored.append((score, pid))

    scored.sort(key=lambda x: -x[0])
    results = []
    for score, pid in scored[:limit]:
        if score < 50:
            break
        p = profiles[pid]
        results.append({"id": pid, "name": p.name, "score": score})
    return results


# ── Matchup prediction ────────────────────────────────────────────────────────

def _build_feature_row(
    shooter: ShooterProfile,
    keeper: KeeperProfile,
    pipeline,
) -> pd.DataFrame:
    """Assemble a single synthetic feature row for a shooter×keeper matchup."""
    row: dict = {}

    # Shooter features
    row["shooter_conv_rate_shrunk"] = shooter.conv_rate_shrunk
    row["shooter_n_pens_before"] = shooter.n_penalties
    row["shooter_conv_rate_is_imputed"] = shooter.n_penalties == 0
    row["shooter_zone_entropy"] = shooter.zone_entropy
    for z in SHOT_ZONES:
        row[f"shooter_zone_prob_{z}"] = shooter.zone_probs.get(z, 1/6)
    row["shooter_foot_is_right"] = shooter.foot_is_right
    row["shooter_preferred_foot_is_missing"] = shooter.foot_is_missing
    row["shooter_height_cm"] = shooter.height_cm or 0.0

    # Keeper features
    row["keeper_save_rate_shrunk"] = keeper.save_rate_shrunk
    row["keeper_n_faced_before"] = keeper.n_faced
    row["keeper_save_rate_is_imputed"] = keeper.n_faced == 0
    for d in DIVE_DIRS:
        row[f"keeper_dive_{d}_prob"] = keeper.dive_probs.get(d, 1/3)
    row["keeper_dive_entropy"] = keeper.dive_entropy
    row["keeper_foot_is_right"] = keeper.foot_is_right
    row["keeper_preferred_foot_is_missing"] = keeper.foot_is_missing
    row["keeper_height_cm"] = keeper.height_cm or 0.0

    # Baseline predictions (required by LGBM feature set)
    row["pred_shooter_shrunk"] = shooter.conv_rate_shrunk
    row["pred_keeper_shrunk"] = np.clip(1 - keeper.save_rate_shrunk, EPS, 1 - EPS)

    # Match context defaults (in-game penalty, no shootout index)
    row["is_shootout"] = False
    row["shootout_kick_index"] = 0.0
    row["expected_guess_correct"] = 0.0

    return pd.DataFrame([row])


def _build_explanation(
    shooter: ShooterProfile,
    keeper: KeeperProfile,
    goal_prob: float,
    composed_prob: float,
) -> str:
    dominant_zone = max(shooter.zone_probs, key=lambda z: shooter.zone_probs[z])
    dominant_dive = max(keeper.dive_probs, key=lambda d: keeper.dive_probs[d])
    zone_pct = round(shooter.zone_probs[dominant_zone] * 100)
    dive_pct = round(keeper.dive_probs[dominant_dive] * 100)
    conv_pct = round(shooter.conv_rate_shrunk * 100)
    save_pct = round(keeper.save_rate_shrunk * 100)
    goal_pct = round(goal_prob * 100)

    foot_str = f" ({shooter.preferred_foot}-footed)" if shooter.preferred_foot else ""
    lines = [
        f"{shooter.name}{foot_str} converts {conv_pct}% of penalties (career, shrunk).",
        f"Favours {dominant_zone.replace('_', ' ')} zone ({zone_pct}% of shots).",
        f"{keeper.name} saves {save_pct}% and dives {dominant_dive} {dive_pct}% of the time.",
        f"Model prediction: {goal_pct}% chance of goal.",
    ]
    return " ".join(lines)


def predict_matchup(
    shooter: ShooterProfile,
    keeper: KeeperProfile,
    pipeline,
) -> MatchupPrediction:
    from src.models.training import predict_pipeline

    feature_row = _build_feature_row(shooter, keeper, pipeline)
    preds = predict_pipeline(pipeline, feature_row)

    goal_prob = float(np.clip(preds["combo_calibrated"][0], EPS, 1 - EPS))
    composed_prob = float(np.clip(preds["composed"][0], EPS, 1 - EPS))

    # Build per-zone heatmap: P(goal | zone) weighted by best dive alignment
    zone_dive_table: dict[str, float] = {}
    if pipeline.dive_table:
        dominant_dive = max(keeper.dive_probs, key=lambda d: keeper.dive_probs[d])
        for z in SHOT_ZONES:
            zone_dive_table[z] = pipeline.dive_table.get((z, dominant_dive), 0.74)
    else:
        for z in SHOT_ZONES:
            zone_dive_table[z] = pipeline.resolution_table.get(z, 0.74)

    explanation = _build_explanation(shooter, keeper, goal_prob, composed_prob)

    return MatchupPrediction(
        goal_probability=goal_prob,
        composed_probability=composed_prob,
        shooter=shooter,
        keeper=keeper,
        zone_probs=shooter.zone_probs,
        dive_probs=keeper.dive_probs,
        zone_dive_table=zone_dive_table,
        explanation=explanation,
    )
