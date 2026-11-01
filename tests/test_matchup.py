
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

from features.matchup import (
    add_opponent,
    add_team_matchup_features,
    validate_matchup_features,
)


# ---------------------------------------------------------
# OPPONENT IDENTIFICATION TEST
# Verify home/away opponents and DK-to-ESPN abbreviations.
# ---------------------------------------------------------

def test_add_opponent():

    players = pd.DataFrame({
        "Name": [
            "Atlanta Player",
            "Boston Player",
            "San Antonio Player",
            "New York Player",
        ],
        "TeamAbbrev": ["ATL", "BOS", "SAS", "NYK"],
        "Game Info": [
            "ATL@BOS 10/31/2026 07:30PM ET",
            "ATL@BOS 10/31/2026 07:30PM ET",
            "SAS@NYK 10/31/2026 07:30PM ET",
            "SAS@NYK 10/31/2026 07:30PM ET",
        ],
    })

    result = add_opponent(players)

    assert result.loc[0, "OPPONENT_DK"] == "BOS"
    assert result.loc[0, "OPPONENT"] == "BOS"

    assert result.loc[1, "OPPONENT_DK"] == "ATL"
    assert result.loc[1, "OPPONENT"] == "ATL"

    assert result.loc[2, "OPPONENT_DK"] == "NYK"
    assert result.loc[2, "OPPONENT"] == "NY"

    assert result.loc[3, "OPPONENT_DK"] == "SAS"
    assert result.loc[3, "OPPONENT"] == "SA"


# ---------------------------------------------------------
# TEAM MATCHUP FEATURE TEST
# Verify own-team and opponent ratings and matchup pace.
# ---------------------------------------------------------

def test_team_matchup_features():

    players = pd.DataFrame({
        "ID": ["101", "102"],
        "TeamAbbrev": ["ATL", "BOS"],
        "Game Info": [
            "ATL@BOS 10/31/2026 07:30PM ET",
            "ATL@BOS 10/31/2026 07:30PM ET",
        ],
    })

    team_features = pd.DataFrame({
        "TEAM": ["ATL", "BOS"],
        "PACE_SEASON": [100.0, 110.0],
        "OFF_RATING_SEASON": [112.0, 120.0],
        "DEF_RATING_SEASON": [108.0, 102.0],
        "PACE_LAST_3": [102.0, 112.0],
        "PACE_LAST_5": [101.0, 111.0],
        "PACE_LAST_7": [100.0, 110.0],
    })

    players = add_opponent(players)
    result = add_team_matchup_features(players, team_features)

    atl = result.loc[result["TeamAbbrev"] == "ATL"].iloc[0]
    bos = result.loc[result["TeamAbbrev"] == "BOS"].iloc[0]

    # Atlanta's own-team statistics
    assert atl["TEAM_PACE_SEASON"] == pytest.approx(100.0)
    assert atl["TEAM_OFF_RATING_SEASON"] == pytest.approx(112.0)

    # Atlanta's opponent is Boston
    assert atl["OPP_DEF_RATING_SEASON"] == pytest.approx(102.0)

    # Boston's opponent is Atlanta
    assert bos["OPP_DEF_RATING_SEASON"] == pytest.approx(108.0)

    # Matchup pace is the average of both teams
    assert atl["MATCHUP_PACE_SEASON"] == pytest.approx(105.0)
    assert bos["MATCHUP_PACE_SEASON"] == pytest.approx(105.0)

    # Recent matchup pace
    assert atl["MATCHUP_PACE_LAST_3"] == pytest.approx(107.0)
    assert atl["MATCHUP_PACE_LAST_5"] == pytest.approx(106.0)
    assert atl["MATCHUP_PACE_LAST_7"] == pytest.approx(105.0)


# ---------------------------------------------------------
# INVALID MATCHUP VALIDATION TEST
# Verify that incorrect team assignments are rejected.
# ---------------------------------------------------------

def test_invalid_matchup_validation():

    players = pd.DataFrame({
        "ID": ["101"],
        "TeamAbbrev": ["CHI"],
        "Game Info": [
            "ATL@BOS 10/31/2026 07:30PM ET",
        ],
        "OPPONENT": ["ATL"],
        "OPP_DEF_RATING_SEASON": [108.0],
        "TEAM_DEF_RATING_SEASON": [110.0],
    })

    with pytest.raises(
        ValueError,
        match="Matchup feature validation failed"
    ):
        validate_matchup_features(players)


# ---------------------------------------------------------
# DUPLICATE DK ID VALIDATION TEST
# Verify that duplicate DraftKings IDs are rejected.
# ---------------------------------------------------------

def test_duplicate_dk_ids():

    players = pd.DataFrame({
        "ID": ["101", "101"],
        "TeamAbbrev": ["ATL", "BOS"],
        "Game Info": [
            "ATL@BOS 10/31/2026 07:30PM ET",
            "ATL@BOS 10/31/2026 07:30PM ET",
        ],
        "OPPONENT": ["BOS", "ATL"],
        "OPP_DEF_RATING_SEASON": [102.0, 108.0],
        "TEAM_DEF_RATING_SEASON": [108.0, 102.0],
    })

    with pytest.raises(
        ValueError,
        match="Matchup feature validation failed"
    ):
        validate_matchup_features(players)
