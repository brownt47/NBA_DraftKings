
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

from features.scoring import (
    calculate_scoring_features,
    calculate_rolling_scoring_features,
    calculate_value_per_1000,
)


# ---------------------------------------------------------
# VALUE PER $1,000 TEST
# Verify fantasy point averages relative to salary.
# ---------------------------------------------------------

def test_value_per_1000():

    players = pd.DataFrame({
        "Name": ["Player A", "Player B"],
        "Salary": [10000, 5000],
        "AVG_DK_POINTS": [50.0, 30.0],
    })

    result = calculate_value_per_1000(players)

    assert result.loc[0, "VALUE_PER_1000"] == pytest.approx(5.0)
    assert result.loc[1, "VALUE_PER_1000"] == pytest.approx(6.0)



# ---------------------------------------------------------
# SEASON SCORING TEST
# Verify game counts and season statistical averages.
# ---------------------------------------------------------

def test_scoring_features():

    games = pd.DataFrame({
        "PLAYER_ID": ["101", "101", "102"],
        "GAME_ID": ["G1", "G2", "G3"],
        "DK_POINTS": [40.0, 60.0, 30.0],
        "PTS": [20, 30, 15],
        "REB": [8, 12, 5],
        "AST": [4, 6, 3],
    })

    result = calculate_scoring_features(games)

    player = result.loc[
        result["ESPN_PLAYER_ID"] == "101"
    ].iloc[0]

    assert player["GAMES_PLAYED"] == 2
    assert player["AVG_DK_POINTS"] == pytest.approx(50.0)
    assert player["AVG_PTS"] == pytest.approx(25.0)
    assert player["AVG_REB"] == pytest.approx(10.0)
    assert player["AVG_AST"] == pytest.approx(5.0)



# ---------------------------------------------------------
# ROLLING SCORING TEST
# Verify recent-game averages and chronological ordering.
# ---------------------------------------------------------

def test_rolling_scoring_features():

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

    result = calculate_rolling_scoring_features(games)

    player = result.iloc[0]

    assert player["ESPN_PLAYER_ID"] == "101"

    # Most recent 3 games: 30, 40, 50
    assert player["DK_N_LAST_3"] == 3
    assert player["DK_AVG_LAST_3"] == pytest.approx(40.0)

    # All 5 games: 10, 20, 30, 40, 50
    assert player["DK_N_LAST_5"] == 5
    assert player["DK_AVG_LAST_5"] == pytest.approx(30.0)

    # Fewer than 10 games available
    assert player["DK_N_LAST_10"] == 5
    assert player["DK_AVG_LAST_10"] == pytest.approx(30.0)

    # Pandas sample standard deviation
    assert player["DK_STD_LAST_3"] == pytest.approx(10.0)
