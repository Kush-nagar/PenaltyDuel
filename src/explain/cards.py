"""Honesty-aware explanation card builder from SHAP factors."""
from __future__ import annotations

from src.serving.predict import KeeperProfile, ShooterProfile

CONFIDENCE_HIGH_THRESHOLD = 20
CONFIDENCE_MEDIUM_THRESHOLD = 5


def _confidence_level(n_penalties: int, n_faced: int) -> str:
    min_count = min(n_penalties, n_faced)
    if min_count >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if min_count >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _shooter_narrative(shooter: ShooterProfile) -> str:
    if shooter.n_penalties == 0:
        foot_str = shooter.preferred_foot or "unknown-footed"
        return f"No personal record — estimate based on typical {foot_str} forwards."
    conv_pct = round(shooter.conv_rate_shrunk * 100)
    dominant_zone = max(shooter.zone_probs, key=lambda z: shooter.zone_probs[z])
    zone_pct = round(shooter.zone_probs[dominant_zone] * 100)
    return (
        f"{shooter.name} converts {conv_pct}% over {shooter.n_penalties} penalties, "
        f"favouring {dominant_zone.replace('_', ' ')} ({zone_pct}%)."
    )


def _keeper_narrative(keeper: KeeperProfile) -> str:
    if keeper.n_faced == 0:
        return "No personal record — estimate based on typical goalkeepers."
    save_pct = round(keeper.save_rate_shrunk * 100)
    dominant_dive = max(keeper.dive_probs, key=lambda d: keeper.dive_probs[d])
    dive_pct = round(keeper.dive_probs[dominant_dive] * 100)
    return (
        f"{keeper.name} saves {save_pct}% over {keeper.n_faced} faced, "
        f"tending to dive {dominant_dive} ({dive_pct}%)."
    )


def build_shap_card(
    shap_factors: list[dict],
    shooter: ShooterProfile,
    keeper: KeeperProfile,
    goal_prob: float,
    global_rate: float,
) -> dict:
    """
    Build a structured explanation card from SHAP factors and player profiles.

    Honesty rules enforced:
    - Never say "converts X%" when n_penalties == 0 (uses prior language instead).
    - Never say "saves X%" when n_faced == 0.
    - Confidence level driven by min(n_penalties, n_faced).
    - low_data_warning=True when min < CONFIDENCE_MEDIUM_THRESHOLD.
    """
    goal_pct = round(goal_prob * 100)
    confidence = _confidence_level(shooter.n_penalties, keeper.n_faced)
    low_data = (
        shooter.n_penalties < CONFIDENCE_MEDIUM_THRESHOLD
        or keeper.n_faced < CONFIDENCE_MEDIUM_THRESHOLD
    )

    top_factors = shap_factors[:2] if len(shap_factors) >= 2 else shap_factors
    top_descriptions = []
    for f in top_factors:
        sign = "+" if f["direction"] == "positive" else "-" if f["direction"] == "negative" else "~"
        top_descriptions.append(
            f"{sign}{abs(f['shap_value']) * 100:.1f}pp from {f['label'].lower()}"
        )

    return {
        "headline": f"{goal_pct}% goal probability",
        "confidence": confidence,
        "low_data_warning": low_data,
        "low_data_note": (
            "Estimate based on limited data — treat as prior, not personal record."
            if low_data
            else None
        ),
        "shooter_summary": _shooter_narrative(shooter),
        "keeper_summary": _keeper_narrative(keeper),
        "top_factors": top_descriptions,
        "factors": shap_factors,
    }
