
import pandas as pd
import pytest

from src.modeling.splits import biweekly_walk_forward_splits


def test_biweekly_splits_preserve_chronology():
    dates = pd.date_range("2025-12-01", "2026-04-05")

    df = pd.DataFrame({
        "GAME_DATE": dates,
        "DK_POINTS": range(len(dates)),
    })

    folds = list(biweekly_walk_forward_splits(df))

    assert len(folds) == 7

    validation_dates = []

    for train_idx, val_idx in folds:
        train = pd.to_datetime(df.loc[train_idx, "GAME_DATE"])
        val = pd.to_datetime(df.loc[val_idx, "GAME_DATE"])

        assert train.max() < val.min()
        assert val.max() < pd.Timestamp("2026-04-01")

        validation_dates.extend(val.tolist())

    assert len(validation_dates) == len(set(validation_dates))
    assert min(validation_dates) == pd.Timestamp("2026-01-01")
    assert max(validation_dates) == pd.Timestamp("2026-03-31")


def test_biweekly_splits_keep_same_day_together():
    df = pd.DataFrame({
        "GAME_DATE": [
            "2025-12-31",
            "2026-01-01",
            "2026-01-01",
            "2026-01-15",
            "2026-01-15",
            "2026-01-29",
            "2026-02-12",
            "2026-02-26",
            "2026-03-12",
            "2026-03-26",
        ],
    })

    folds = list(biweekly_walk_forward_splits(df))

    first_train, first_val = folds[0]

    assert len(first_train) == 1
    assert len(first_val) == 2

    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))


def test_biweekly_splits_reject_invalid_window():
    df = pd.DataFrame({"GAME_DATE": ["2026-01-01"]})

    with pytest.raises(ValueError):
        list(biweekly_walk_forward_splits(df, window_days=0))
