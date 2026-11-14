
"""Leakage-safe historical sequences for NBA DraftKings LSTM experiments."""

from collections import deque

import numpy as np
import pandas as pd


SEQUENCE_FEATURES = [
    "MIN", "DK_POINTS", "PTS", "REB", "AST", "STL", "BLK",
    "TOV", "FGA", "FG3A", "FTA",
]


def build_sequences(df: pd.DataFrame, sequence_length: int = 10):
    """Build previous-game sequences aligned to the original row order."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    required = ["PLAYER_ID", "GAME_DATE", "GAME_ID", *SEQUENCE_FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[["PLAYER_ID", "GAME_DATE", "GAME_ID"]].isna().any().any():
        raise ValueError("Missing player, date, or game identifiers")

    if df.duplicated(["PLAYER_ID", "GAME_ID"]).any():
        raise ValueError("Duplicate player/game records")

    n = len(df)
    X = np.zeros(
        (n, sequence_length, len(SEQUENCE_FEATURES)),
        dtype=np.float32,
    )
    lengths = np.zeros(n, dtype=np.int64)

    work = df[
        ["PLAYER_ID", "GAME_DATE", "GAME_ID", *SEQUENCE_FEATURES]
    ].copy()

    work["GAME_DATE"] = pd.to_datetime(
        work["GAME_DATE"], errors="raise"
    )

    for col in SEQUENCE_FEATURES:
        work[col] = pd.to_numeric(
            work[col], errors="coerce"
        ).fillna(0)

    work["_position"] = np.arange(n)

    work = work.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"],
        kind="stable",
    )

    for _, player in work.groupby("PLAYER_ID", sort=False):
        history = deque(maxlen=sequence_length)

        for _, same_day in player.groupby("GAME_DATE", sort=True):
            prior = np.asarray(list(history), dtype=np.float32)
            length = len(prior)

            for pos in same_day["_position"].to_numpy():
                lengths[pos] = length

                if length:
                    X[pos, -length:, :] = prior

            # Update history only after all same-day rows.
            for values in same_day[
                SEQUENCE_FEATURES
            ].to_numpy(dtype=np.float32):
                history.append(values)

    return X, lengths


def load_historical_sequences(
    path="data/nba_features_2025_26.csv",
    sequence_length=10,
):
    """Load historical data and construct LSTM sequences."""
    df = pd.read_csv(
        path,
        dtype={"PLAYER_ID": str, "GAME_ID": str},
    )

    X, lengths = build_sequences(
        df, sequence_length=sequence_length
    )

    return df, X, lengths
