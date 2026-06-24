import pandas as pd

from src.evaluation.splitters import grouped_holdout_split, temporal_train_test_split


def toy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "penalty_id": ["p1", "p2", "p3", "p4", "p5"],
            "match_date": [
                "2020-01-01",
                "2020-01-05",
                "2020-02-01",
                "2020-03-01",
                "2020-04-01",
            ],
            "shooter_id": [1, 1, 2, 3, 4],
            "competition_name": ["A", "A", "B", "B", "C"],
        }
    )


def test_temporal_train_test_split_uses_cutoff_without_random_shuffle():
    df = toy_rows()

    train, test = temporal_train_test_split(df, cutoff="2020-02-01")

    assert train["penalty_id"].tolist() == ["p1", "p2"]
    assert test["penalty_id"].tolist() == ["p3", "p4", "p5"]
    assert pd.to_datetime(train["match_date"]).max() < pd.Timestamp("2020-02-01")
    assert pd.to_datetime(test["match_date"]).min() >= pd.Timestamp("2020-02-01")


def test_grouped_holdout_split_holds_out_entire_groups():
    df = toy_rows()

    train, test = grouped_holdout_split(df, group_col="shooter_id", holdout_groups=[1, 4])

    assert set(test["shooter_id"]) == {1, 4}
    assert set(train["shooter_id"]).isdisjoint(set(test["shooter_id"]))
    assert test["penalty_id"].tolist() == ["p1", "p2", "p5"]


def test_grouped_holdout_split_rejects_missing_group_column():
    df = toy_rows()

    try:
        grouped_holdout_split(df, group_col="keeper_id", holdout_groups=[1])
    except KeyError as exc:
        assert "keeper_id" in str(exc)
    else:
        raise AssertionError("expected KeyError")
