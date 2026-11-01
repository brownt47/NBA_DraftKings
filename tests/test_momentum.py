
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------
# IMPORT SETUP
# Allow tests to import modules from src.
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from features.momentum import calculate_momentum_features


# ---------------------------------------------------------
# POSITIVE MOMENTUM TEST
# Verify EMA calculations for improving performance.
# ---------------------------------------------------------

def test_positive_momentum():

    games = pd.DataFrame({
        "PLAYER_ID": ["101"] * 5,
        "GAME_ID": ["G3", "G1", "G5", "G2", "G4"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-03",
            "2026-10-01",
            "2026-10-05",
            "2026-10-02",
            "2026-10-04",
        ]),
        "DK_POINTS": [30, 10, 50, 20, 40],
    })

    result = calculate_momentum_features(games)
    player = result.iloc[0]

    assert player["ESPN_PLAYER_ID"] == "101"

    # Shorter EMA should respond faster to rising scores
    assert player["DK_EMA_3"] > player["DK_EMA_10"]

    # Momentum should be positive
    assert player["DK_MOMENTUM"] > 0

    # Momentum should still be increasing
    assert player["DK_MOMENTUM_CHANGE"] > 0

    # No negative-to-positive crossover in the final game
    assert player["DK_CROSSOVER"] == 0

    # No repeated crossover reversals
    assert player["DK_WHIPSAW_5"] == 0


# ---------------------------------------------------------
# POSITIVE CROSSOVER TEST
# Verify detection of negative-to-positive EMA momentum.
# ---------------------------------------------------------

def test_positive_crossover():

    # Declining scores followed by a strong recovery
    scores = [50, 40, 30, 20, 10, 60]

    games = pd.DataFrame({
        "PLAYER_ID": ["101"] * len(scores),
        "GAME_ID": [f"G{i}" for i in range(len(scores))],
        "GAME_DATE": pd.date_range(
            "2026-10-01",
            periods=len(scores)
        ),
        "DK_POINTS": scores,
    })

    result = calculate_momentum_features(games)
    player = result.iloc[0]

    # Short-term EMA crosses above long-term EMA
    assert player["DK_MOMENTUM"] > 0
    assert player["DK_CROSSOVER"] == 1

    # A crossover should also count as one reversal
    assert player["DK_WHIPSAW_5"] == 1


# ---------------------------------------------------------
# NEGATIVE CROSSOVER TEST
# Verify detection of positive-to-negative EMA momentum.
# ---------------------------------------------------------

def test_negative_crossover():

    # Improving scores followed by a sharp decline
    scores = [10, 20, 30, 40, 50, 0]

    games = pd.DataFrame({
        "PLAYER_ID": ["101"] * len(scores),
        "GAME_ID": [f"G{i}" for i in range(len(scores))],
        "GAME_DATE": pd.date_range(
            "2026-10-01",
            periods=len(scores)
        ),
        "DK_POINTS": scores,
    })

    result = calculate_momentum_features(games)
    player = result.iloc[0]

    # Short-term EMA crosses below long-term EMA
    assert player["DK_MOMENTUM"] < 0
    assert player["DK_CROSSOVER"] == -1

    # One crossover reversal
    assert player["DK_WHIPSAW_5"] == 1

    # Momentum should be declining
    assert player["DK_MOMENTUM_CHANGE"] < 0


# ---------------------------------------------------------
# MOMENTUM WHIPSAW TEST
# Verify multiple EMA crossovers within five transitions.
# ---------------------------------------------------------

def test_momentum_whipsaw():

    # Alternating strong and weak performances
    scores = [30, 60, 0, 60, 0, 60, 0]

    games = pd.DataFrame({
        "PLAYER_ID": ["101"] * len(scores),
        "GAME_ID": [f"G{i}" for i in range(len(scores))],
        "GAME_DATE": pd.date_range(
            "2026-10-01",
            periods=len(scores)
        ),
        "DK_POINTS": scores,
    })

    result = calculate_momentum_features(games)
    player = result.iloc[0]

    # Five reversals across the last five transitions
    assert player["DK_WHIPSAW_5"] == 5

    # Final performance produces negative momentum
    assert player["DK_MOMENTUM"] < 0
