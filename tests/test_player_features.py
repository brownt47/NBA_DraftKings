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

from data_processing.player_features import add_player_features



# ---------------------------------------------------------
# PLAYER FEATURE INTEGRATION TEST
# Verify scoring, minutes, momentum, and salary value
# are correctly combined for a DraftKings player.
# ---------------------------------------------------------

def test_add_player_features():

    players = pd.DataFrame({
        "ID": ["101"],
        "ESPN_PLAYER_ID": ["9001"],
        "Name": ["Player A"],
        "Salary": [8000],
    })

    games = pd.DataFrame({
        "PLAYER_ID": ["9001", "9001", "9001"],
        "GAME_ID": ["G1", "G2", "G3"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-25",
            "2026-10-27",
            "2026-10-29",
        ]),
        "MIN": [30, 32, 34],
        "PTS": [20, 25, 30],
        "REB": [5, 6, 7],
        "AST": [4, 5, 6],
        "DK_POINTS": [35, 45, 55],
    })

    result = add_player_features(players, games)

    # Player remains in the output
    assert len(result) == 1
    assert result.loc[0, "ESPN_PLAYER_ID"] == "9001"

    # Season scoring
    assert result.loc[0, "GAMES_PLAYED"] == 3
    assert result.loc[0, "AVG_DK_POINTS"] == 45

    # Rolling scoring
    assert result.loc[0, "DK_AVG_LAST_3"] == 45

    # Playing time
    assert result.loc[0, "MIN_AVG_SEASON"] == 32

    # Momentum features exist
    assert "DK_MOMENTUM" in result.columns
    assert pd.notna(result.loc[0, "DK_MOMENTUM"])

    # Fantasy points per $1,000 salary
    assert result.loc[0, "VALUE_PER_1000"] == pytest.approx(
        45 / 8
    )



# ---------------------------------------------------------
# PLAYER WITHOUT HISTORY TEST
# Verify players without historical games remain in
# the DraftKings slate with missing statistical features.
# ---------------------------------------------------------

def test_player_without_history():

    players = pd.DataFrame({
        "ID": ["101", "102"],
        "ESPN_PLAYER_ID": ["9001", "9002"],
        "Name": ["Player A", "Player B"],
        "Salary": [8000, 6000],
    })

    games = pd.DataFrame({
        "PLAYER_ID": ["9001", "9001", "9001"],
        "GAME_ID": ["G1", "G2", "G3"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-25",
            "2026-10-27",
            "2026-10-29",
        ]),
        "MIN": [30, 32, 34],
        "PTS": [20, 25, 30],
        "REB": [5, 6, 7],
        "AST": [4, 5, 6],
        "DK_POINTS": [35, 45, 55],
    })

    result = add_player_features(players, games)

    # Both DraftKings players remain
    assert len(result) == 2
    assert set(result["ID"]) == {"101", "102"}

    # Player A has historical statistics
    player_a = result.loc[
        result["ESPN_PLAYER_ID"] == "9001"
    ].iloc[0]

    assert player_a["GAMES_PLAYED"] == 3
    assert player_a["AVG_DK_POINTS"] == 45

    # Player B has no historical games
    player_b = result.loc[
        result["ESPN_PLAYER_ID"] == "9002"
    ].iloc[0]

    assert pd.isna(player_b["GAMES_PLAYED"])
    assert pd.isna(player_b["AVG_DK_POINTS"])
    assert pd.isna(player_b["DK_MOMENTUM"])
    assert pd.isna(player_b["VALUE_PER_1000"])


# ---------------------------------------------------------
# MULTIPLE PLAYER VALUE TEST
# Verify fantasy value is calculated independently
# for players with different salaries and scoring.
# ---------------------------------------------------------

def test_multiple_player_values():

    players = pd.DataFrame({
        "ID": ["101", "102"],
        "ESPN_PLAYER_ID": ["9001", "9002"],
        "Name": ["Player A", "Player B"],
        "Salary": [10000, 5000],
    })

    games = pd.DataFrame({
        "PLAYER_ID": ["9001", "9001", "9002", "9002"],
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-25",
            "2026-10-27",
            "2026-10-25",
            "2026-10-27",
        ]),
        "MIN": [30, 32, 25, 28],
        "PTS": [25, 35, 15, 20],
        "REB": [8, 10, 5, 6],
        "AST": [5, 7, 3, 4],
        "DK_POINTS": [40, 60, 25, 35],
    })

    result = add_player_features(players, games)

    player_a = result.loc[
        result["ESPN_PLAYER_ID"] == "9001"
    ].iloc[0]

    player_b = result.loc[
        result["ESPN_PLAYER_ID"] == "9002"
    ].iloc[0]

    # Player A: 50 average DK points / $10,000
    assert player_a["AVG_DK_POINTS"] == 50
    assert player_a["VALUE_PER_1000"] == pytest.approx(5.0)

    # Player B: 30 average DK points / $5,000
    assert player_b["AVG_DK_POINTS"] == 30
    assert player_b["VALUE_PER_1000"] == pytest.approx(6.0)

    # Lower-priced player provides better salary value
    assert player_b["VALUE_PER_1000"] > player_a["VALUE_PER_1000"]
