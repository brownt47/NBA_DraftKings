"""Chronological train/validation splits for NBA modeling."""

import pandas as pd


def monthly_walk_forward_splits(
    df,
    validation_months=("2026-01", "2026-02", "2026-03"),
    date_column="GAME_DATE",
):
    """
    Generate expanding-window train/validation splits.

    All player-games on the same date stay together.
    April 2026 is reserved for final testing.
    """
    dates = pd.to_datetime(df[date_column])

    for month in validation_months:
        start = pd.Period(month, freq="M").start_time
        end = (pd.Period(month, freq="M") + 1).start_time

        train_idx = df.index[dates < start].to_numpy()
        val_idx = df.index[
            (dates >= start) & (dates < end)
        ].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Insufficient data for validation month {month}"
            )

        yield train_idx, val_idx
        