import pandas as pd

from src.features.providers import TabularFeatureProvider, VideoFeatureProvider


def test_tabular_provider_filters_features_by_penalty_id_order():
    table = pd.DataFrame(
        {
            "penalty_id": ["p1", "p2", "p3"],
            "feature": [0.1, 0.2, 0.3],
        }
    )
    provider = TabularFeatureProvider(table)

    result = provider.build(["p3", "p1"])

    assert result["penalty_id"].tolist() == ["p3", "p1"]
    assert result["feature"].tolist() == [0.3, 0.1]


def test_video_provider_returns_missing_flag_for_requested_penalties():
    provider = VideoFeatureProvider()

    result = provider.build(["p2", "p1"])

    assert result.to_dict("records") == [
        {"penalty_id": "p2", "video_is_missing": True},
        {"penalty_id": "p1", "video_is_missing": True},
    ]
