
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

from features.minutes import calculate_minutes_features


# ---------------------------------------------------------
# MINUTES FEATURE TEST
# Verify season averages, recent averages, and trends.
# ---------------------------------------------------------

def test_minutes_features():

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
        "MIN": [30, 20, 40, 25, 35],
        "DK_POINTS": [45, 30, 60, 37.5, 52.5],
    })

    result = calculate_minutes_features(games)
    player = result.iloc[0]

    # Season averages
    assert player["ESPN_PLAYER_ID"] == "101"
    assert player["MIN_AVG_SEASON"] == pytest.approx(30.0)
    assert player["DK_PER_MIN"] == pytest.approx(1.5)

    # Last 3 games: 30, 35, 40 minutes
    assert player["MIN_N_LAST_3"] == 3
    assert player["MIN_AVG_LAST_3"] == pytest.approx(35.0)

    # All 5 games
    assert player["MIN_N_LAST_5"] == 5
    assert player["MIN_AVG_LAST_5"] == pytest.approx(30.0)

    # Fewer than 10 games available
    assert player["MIN_N_LAST_10"] == 5
    assert player["MIN_AVG_LAST_10"] == pytest.approx(30.0)

    # Recent minutes trend relative to season
    assert player["MIN_TREND_3"] == pytest.approx(5.0)

    # Sample standard deviation of 20, 25, 30, 35, 40
    assert player["MIN_STD_LAST_5"] == pytest.approx(7.905694)


# ---------------------------------------------------------
# ZERO-MINUTE TEST
# Verify safe handling of players with no playing time.
# ---------------------------------------------------------

def test_zero_minutes():

    games = pd.DataFrame({
        "PLAYER_ID": ["101", "101"],
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-01",
            "2026-10-02",
        ]),
        "MIN": [0, 0],
        "DK_POINTS": [0, 0],
    })

    result = calculate_minutes_features(games)
    player = result.iloc[0]

    assert player["MIN_AVG_SEASON"] == pytest.approx(0.0)
    assert pd.isna(player["DK_PER_MIN"])
    assert player["MIN_N_LAST_3"] == 2
    assert player["MIN_AVG_LAST_3"] == pytest.approx(0.0)
