"""Tests for chronological NBA modeling splits."""

import pandas as pd

from src.modeling.splits import monthly_walk_forward_splits


def test_monthly_walk_forward_splits():
    dates = pd.date_range(
        "2025-10-01",
        "2026-04-30",
        freq="D",
    )

    # Two players per date, simulating multiple player-games.
    df = pd.DataFrame({
        "GAME_DATE": dates.repeat(2),
        "PLAYER_ID": [1, 2] * len(dates),
    })

    splits = list(monthly_walk_forward_splits(df))

    assert len(splits) == 3

    expected_months = ["2026-01", "2026-02", "2026-03"]

    for (train_idx, val_idx), month in zip(splits, expected_months):
        train = df.loc[train_idx]
        val = df.loc[val_idx]

        # Both partitions must contain observations.
        assert not train.empty
        assert not val.empty

        # Training must strictly precede validation.
        assert train["GAME_DATE"].max() < val["GAME_DATE"].min()

        # Every validation row belongs to the expected month.
        assert (
            val["GAME_DATE"].dt.to_period("M").astype(str) == month
        ).all()

        # No dates may appear in both partitions.
        assert set(train["GAME_DATE"]).isdisjoint(
            set(val["GAME_DATE"])
        )

        # April is reserved for final testing.
        assert (train["GAME_DATE"] < "2026-04-01").all()
        assert (val["GAME_DATE"] < "2026-04-01").all()

        # Both players on each validation date stay together.
        assert val.groupby("GAME_DATE").size().eq(2).all()