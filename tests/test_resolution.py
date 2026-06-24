import numpy as np
import pandas as pd

from src.models.resolution import (
    RESOLUTION_ZONES,
    compose_goal_prob,
    fit_resolution_table,
)


def train_rows() -> pd.DataFrame:
    rows = []
    # low_left almost always scores; high_center almost always misses.
    for _ in range(30):
        rows.append({"shot_zone": "low_left", "outcome_bin": 1})
    for _ in range(20):
        rows.append({"shot_zone": "high_center", "outcome_bin": 0})
    for _ in range(10):
        rows.append({"shot_zone": "low_right", "outcome_bin": 1})
    return pd.DataFrame(rows)


def test_fit_resolution_table_covers_all_zones_and_orders_by_difficulty():
    table = fit_resolution_table(train_rows(), smoothing=2.0)

    assert set(table) == set(RESOLUTION_ZONES)
    for value in table.values():
        assert 0.0 < value < 1.0
    assert table["low_left"] > table["high_center"]


def test_compose_goal_prob_uses_shooter_zone_distribution():
    table = fit_resolution_table(train_rows(), smoothing=2.0)

    df = pd.DataFrame(
        [
            {f"shooter_zone_prob_{z}": (1.0 if z == "low_left" else 0.0) for z in RESOLUTION_ZONES},
            {f"shooter_zone_prob_{z}": (1.0 if z == "high_center" else 0.0) for z in RESOLUTION_ZONES},
        ]
    )

    probs = compose_goal_prob(df, table)

    assert probs.shape == (2,)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert probs[0] > probs[1]
    assert probs[0] == np.float64(table["low_left"])
