
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

from features.team import calculate_team_features


# ---------------------------------------------------------
# TEAM FEATURE TEST
# Verify season and recent team performance averages.
# ---------------------------------------------------------

def test_team_features():

    games = pd.DataFrame({
        "TEAM": ["ATL"] * 5,
        "GAME_ID": ["G3", "G1", "G5", "G2", "G4"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-03",
            "2026-10-01",
            "2026-10-05",
            "2026-10-02",
            "2026-10-04",
        ]),
        "PACE_EST": [102, 98, 106, 100, 104],
        "OFF_RATING_EST": [112, 108, 116, 110, 114],
        "DEF_RATING_EST": [110, 106, 114, 108, 112],
    })

    result = calculate_team_features(games)
    team = result.iloc[0]

    assert team["TEAM"] == "ATL"
    assert team["TEAM_GAMES_PLAYED"] == 5

    # Season averages
    assert team["PACE_SEASON"] == pytest.approx(102.0)
    assert team["OFF_RATING_SEASON"] == pytest.approx(112.0)
    assert team["DEF_RATING_SEASON"] == pytest.approx(110.0)

    # Most recent 3 games
    assert team["PACE_LAST_3"] == pytest.approx(104.0)
    assert team["OFF_RATING_LAST_3"] == pytest.approx(114.0)
    assert team["DEF_RATING_LAST_3"] == pytest.approx(112.0)

    # Fewer than 7 games available
    assert team["PACE_LAST_7"] == pytest.approx(102.0)
    assert team["OFF_RATING_LAST_7"] == pytest.approx(112.0)
    assert team["DEF_RATING_LAST_7"] == pytest.approx(110.0)


# ---------------------------------------------------------
# MULTIPLE TEAM TEST
# Verify that team statistics are calculated separately.
# ---------------------------------------------------------

def test_multiple_teams():

    games = pd.DataFrame({
        "TEAM": ["ATL", "BOS", "ATL", "BOS"],
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-01",
            "2026-10-01",
            "2026-10-02",
            "2026-10-02",
        ]),
        "PACE_EST": [100, 110, 104, 114],
        "OFF_RATING_EST": [110, 120, 114, 124],
        "DEF_RATING_EST": [108, 98, 112, 102],
    })

    result = calculate_team_features(games)

    atl = result.loc[result["TEAM"] == "ATL"].iloc[0]
    bos = result.loc[result["TEAM"] == "BOS"].iloc[0]

    # Each team should have two games
    assert atl["TEAM_GAMES_PLAYED"] == 2
    assert bos["TEAM_GAMES_PLAYED"] == 2

    # Atlanta averages
    assert atl["PACE_SEASON"] == pytest.approx(102.0)
    assert atl["OFF_RATING_SEASON"] == pytest.approx(112.0)
    assert atl["DEF_RATING_SEASON"] == pytest.approx(110.0)

    # Boston averages
    assert bos["PACE_SEASON"] == pytest.approx(112.0)
    assert bos["OFF_RATING_SEASON"] == pytest.approx(122.0)
    assert bos["DEF_RATING_SEASON"] == pytest.approx(100.0)

    # Each team has fewer than three games
    assert atl["PACE_LAST_3"] == pytest.approx(102.0)
    assert bos["PACE_LAST_3"] == pytest.approx(112.0)
