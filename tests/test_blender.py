import numpy as np

from src.models.blender import predict_blend, train_meta_blender


def test_blender_predicts_unit_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=80)
    components = {
        "combo": np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.05, 80), 0.01, 0.99),
        "composed": np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.05, 80), 0.01, 0.99),
    }

    blender = train_meta_blender(components, y)
    blended = predict_blend(blender, components)

    assert blended.shape == (80,)
    assert np.all(blended >= 0) and np.all(blended <= 1)


def test_blender_is_invariant_to_component_dict_order():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=60)
    combo = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.05, 60), 0.01, 0.99)
    composed = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.05, 60), 0.01, 0.99)

    blender = train_meta_blender({"combo": combo, "composed": composed}, y)

    forward = predict_blend(blender, {"combo": combo, "composed": composed})
    reversed_order = predict_blend(blender, {"composed": composed, "combo": combo})

    assert np.allclose(forward, reversed_order)
