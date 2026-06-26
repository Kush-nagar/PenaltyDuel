"""
Honesty and correctness tests for the SHAP explanation layer.
Uses synthetic LogisticRegression fitted on synthetic data — no model artifacts required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.explain.cards import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    build_shap_card,
)
from src.explain.shap_wrap import FEATURE_LABELS, build_explainer, explain_matchup
from src.models.combo import COMBO_FEATURE_NAMES, CombinedOutcomeModel
from src.serving.predict import KeeperProfile, ShooterProfile

SHOT_ZONES = ["low_left", "low_center", "low_right", "high_left", "high_center", "high_right"]
DIVE_DIRS = ["left", "center", "right"]
N_FEATURES = len(COMBO_FEATURE_NAMES)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_lr_model() -> CombinedOutcomeModel:
    rng = np.random.default_rng(42)
    n = 300
    X = rng.standard_normal((n, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    lr = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    lr.fit(X, y)
    return CombinedOutcomeModel(
        estimator=lr,
        global_rate=0.74,
        feature_names=list(COMBO_FEATURE_NAMES),
    )


def _make_modeling_table(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "shooter_conv_rate_shrunk": rng.uniform(0.5, 0.9, n),
        "keeper_save_rate_shrunk": rng.uniform(0.1, 0.4, n),
        "shooter_zone_entropy": rng.uniform(0.5, 1.0, n),
        "is_shootout": rng.integers(0, 2, n).astype(float),
        "shooter_n_pens_before": rng.integers(0, 50, n).astype(float),
        "keeper_n_faced_before": rng.integers(0, 50, n).astype(float),
        **{f"shooter_zone_prob_{z}": rng.uniform(0, 1, n) for z in SHOT_ZONES},
    })


def _make_shooter(n_penalties: int = 20) -> ShooterProfile:
    return ShooterProfile(
        player_id="s1",
        name="Test Shooter",
        name_normalized="test shooter",
        n_penalties=n_penalties,
        conv_rate_shrunk=0.80,
        zone_probs={z: 1 / 6 for z in SHOT_ZONES},
        zone_entropy=1.0,
        preferred_foot="right",
        height_cm=180.0,
        foot_is_right=1.0,
        foot_is_missing=0.0,
    )


def _make_keeper(n_faced: int = 20) -> KeeperProfile:
    return KeeperProfile(
        player_id="k1",
        name="Test Keeper",
        name_normalized="test keeper",
        n_faced=n_faced,
        save_rate_shrunk=0.20,
        dive_probs={d: 1 / 3 for d in DIVE_DIRS},
        dive_entropy=1.0,
        preferred_foot="right",
        height_cm=190.0,
        foot_is_right=1.0,
        foot_is_missing=0.0,
    )


class _MockPipeline:
    """Minimal pipeline stub for explain_matchup — no disk access."""
    def __init__(self, model: CombinedOutcomeModel):
        self.combined_model = model
        self.dive_table = None
        self.resolution_table: dict = {}


# ── SHAP wrapper tests ────────────────────────────────────────────────────────

class TestShapWrap:
    def setup_method(self):
        self.model = _make_lr_model()
        self.mt = _make_modeling_table()
        self.explainer = build_explainer(self.model, self.mt)
        self.pipeline = _MockPipeline(self.model)

    def test_build_explainer_returns_linear_explainer(self):
        import shap
        assert isinstance(self.explainer, shap.LinearExplainer)

    def test_all_combo_features_have_human_labels(self):
        for feat in COMBO_FEATURE_NAMES:
            assert feat in FEATURE_LABELS, f"Missing label for feature {feat!r}"
            assert FEATURE_LABELS[feat].strip(), f"Empty label for feature {feat!r}"

    def test_explain_matchup_returns_one_factor_per_feature(self):
        shooter = _make_shooter()
        keeper = _make_keeper()
        factors = explain_matchup(self.explainer, shooter, keeper, self.pipeline)
        assert len(factors) == N_FEATURES

    def test_explain_matchup_factor_fields_present(self):
        factors = explain_matchup(
            self.explainer, _make_shooter(), _make_keeper(), self.pipeline
        )
        for f in factors:
            assert "feature" in f
            assert "label" in f
            assert "shap_value" in f
            assert "direction" in f
            assert "pct_impact" in f

    def test_factors_sorted_by_abs_shap_descending(self):
        factors = explain_matchup(
            self.explainer, _make_shooter(), _make_keeper(), self.pipeline
        )
        abs_vals = [abs(f["shap_value"]) for f in factors]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_pct_impact_sums_to_100(self):
        factors = explain_matchup(
            self.explainer, _make_shooter(), _make_keeper(), self.pipeline
        )
        total = sum(f["pct_impact"] for f in factors)
        assert abs(total - 100.0) < 0.5, f"pct_impact sum = {total}, expected ~100"

    def test_direction_matches_shap_sign(self):
        factors = explain_matchup(
            self.explainer, _make_shooter(), _make_keeper(), self.pipeline
        )
        for f in factors:
            sv = f["shap_value"]
            if sv > 0.001:
                assert f["direction"] == "positive", f"Expected positive for sv={sv}"
            elif sv < -0.001:
                assert f["direction"] == "negative", f"Expected negative for sv={sv}"


# ── Card builder honesty tests ────────────────────────────────────────────────

class TestShapCard:
    def _factors(self) -> list[dict]:
        return [
            {
                "feature": "shooter_logodds_offset",
                "label": "Shooter conversion record",
                "shap_value": 0.05,
                "direction": "positive",
                "pct_impact": 60.0,
            },
            {
                "feature": "keeper_concede_logodds_offset",
                "label": "Keeper vulnerability",
                "shap_value": -0.03,
                "direction": "negative",
                "pct_impact": 40.0,
            },
        ]

    def test_low_data_warning_when_shooter_no_penalties(self):
        shooter = _make_shooter(n_penalties=0)
        keeper = _make_keeper(n_faced=30)
        card = build_shap_card(self._factors(), shooter, keeper, 0.76, 0.74)
        assert card["low_data_warning"] is True
        assert card["confidence"] == "low"
        assert card["low_data_note"] is not None

    def test_low_data_warning_when_keeper_few_faced(self):
        shooter = _make_shooter(n_penalties=30)
        keeper = _make_keeper(n_faced=3)
        card = build_shap_card(self._factors(), shooter, keeper, 0.74, 0.74)
        assert card["low_data_warning"] is True

    def test_no_low_data_warning_for_well_sampled_players(self):
        shooter = _make_shooter(n_penalties=30)
        keeper = _make_keeper(n_faced=25)
        card = build_shap_card(self._factors(), shooter, keeper, 0.80, 0.74)
        assert card["low_data_warning"] is False
        assert card["confidence"] == "high"
        assert card["low_data_note"] is None

    def test_imputed_shooter_no_personal_language(self):
        """When n_penalties==0, shooter_summary must not claim personal history."""
        shooter = _make_shooter(n_penalties=0)
        keeper = _make_keeper(n_faced=20)
        card = build_shap_card([], shooter, keeper, 0.74, 0.74)
        assert "converts" not in card["shooter_summary"]
        assert "No personal record" in card["shooter_summary"]

    def test_imputed_keeper_no_personal_language(self):
        """When n_faced==0, keeper_summary must not claim personal history."""
        shooter = _make_shooter(n_penalties=20)
        keeper = _make_keeper(n_faced=0)
        card = build_shap_card([], shooter, keeper, 0.74, 0.74)
        assert "saves" not in card["keeper_summary"]
        assert "No personal record" in card["keeper_summary"]

    def test_well_sampled_shooter_uses_personal_language(self):
        shooter = _make_shooter(n_penalties=25)
        keeper = _make_keeper(n_faced=20)
        card = build_shap_card([], shooter, keeper, 0.80, 0.74)
        assert "converts" in card["shooter_summary"]
        assert str(shooter.n_penalties) in card["shooter_summary"]

    def test_well_sampled_keeper_uses_personal_language(self):
        shooter = _make_shooter(n_penalties=20)
        keeper = _make_keeper(n_faced=25)
        card = build_shap_card([], shooter, keeper, 0.74, 0.74)
        assert "saves" in card["keeper_summary"]
        assert str(keeper.n_faced) in card["keeper_summary"]

    def test_headline_is_integer_percent_no_decimal(self):
        shooter = _make_shooter(n_penalties=20)
        keeper = _make_keeper(n_faced=20)
        card = build_shap_card([], shooter, keeper, 0.813, 0.74)
        assert card["headline"] == "81% goal probability"
        assert "." not in card["headline"].split("%")[0]

    @pytest.mark.parametrize("n_pens,expected", [
        (0, "low"),
        (3, "low"),
        (4, "low"),
        (5, "medium"),
        (10, "medium"),
        (19, "medium"),
        (20, "high"),
        (50, "high"),
    ])
    def test_confidence_levels_by_shooter_count(self, n_pens, expected):
        shooter = _make_shooter(n_penalties=n_pens)
        keeper = _make_keeper(n_faced=100)
        card = build_shap_card([], shooter, keeper, 0.74, 0.74)
        assert card["confidence"] == expected

    @pytest.mark.parametrize("n_faced,expected", [
        (0, "low"),
        (4, "low"),
        (5, "medium"),
        (20, "high"),
    ])
    def test_confidence_levels_by_keeper_count(self, n_faced, expected):
        shooter = _make_shooter(n_penalties=100)
        keeper = _make_keeper(n_faced=n_faced)
        card = build_shap_card([], shooter, keeper, 0.74, 0.74)
        assert card["confidence"] == expected

    def test_top_factors_shows_two_most_impactful(self):
        shooter = _make_shooter(n_penalties=20)
        keeper = _make_keeper(n_faced=20)
        factors = self._factors()
        card = build_shap_card(factors, shooter, keeper, 0.80, 0.74)
        assert len(card["top_factors"]) == 2
