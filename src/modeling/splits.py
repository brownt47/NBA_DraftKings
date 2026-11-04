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


def biweekly_walk_forward_splits(
    df,
    start_date="2026-01-01",
    end_date="2026-04-01",
    window_days=14,
    date_column="GAME_DATE",
):
    """
    Generate expanding-window chronological validation splits.

    Validation starts January 1 and ends before April 1.
    Each validation window is 14 days, except the final
    window, which may be shorter.

    All player-games on the same date remain together.
    Indices are labels compatible with df.loc.
    """
    if window_days < 1:
        raise ValueError("window_days must be positive")

    dates = pd.to_datetime(df[date_column])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start >= end:
        raise ValueError("start_date must precede end_date")

    current = start

    while current < end:
        next_date = min(
            current + pd.Timedelta(days=window_days),
            end,
        )

        train_idx = df.index[dates < current].to_numpy()
        val_idx = df.index[
            (dates >= current) & (dates < next_date)
        ].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Insufficient data for validation window "
                f"{current.date()} to {next_date.date()}"
            )

        yield train_idx, val_idx
        current = next_date
