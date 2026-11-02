import pandas as pd
import pytest

from src.features.team_change import add_historical_team_change_features


def test_team_changes_and_return_to_former_team():
    players = pd.DataFrame({
        "PLAYER_ID": [101] * 6,
        "GAME_ID": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-07",
            "2026-10-09",
            "2026-10-11",
        ],
        "TEAM": ["ATL", "ATL", "BOS", "BOS", "ATL", "ATL"],
        "MIN": [20, 30, 35, 25, 40, 32],
        "DK_POINTS": [15, 25, 40, 20, 50, 35],
    })

    result = add_historical_team_change_features(players)
    result = result.sort_values("GAME_DATE").reset_index(drop=True)

    assert result["TEAM_CHANGE_FLAG"].tolist() == [0, 0, 1, 0, 1, 0]
    assert result["GAMES_WITH_CURRENT_TEAM"].tolist() == [0, 1, 0, 1, 0, 1]

    expected_minutes = [None, 20, None, 35, None, 40]
    expected_dk = [None, 15, None, 40, None, 50]

    for i in range(6):
        if expected_minutes[i] is None:
            assert pd.isna(result.loc[i, "MIN_AVG_CURRENT_TEAM"])
            assert pd.isna(result.loc[i, "DK_AVG_CURRENT_TEAM"])
        else:
            assert result.loc[i, "MIN_AVG_CURRENT_TEAM"] == pytest.approx(
                expected_minutes[i]
            )
            assert result.loc[i, "DK_AVG_CURRENT_TEAM"] == pytest.approx(
                expected_dk[i]
            )


def test_team_change_features_multiple_players_unsorted():
    players = pd.DataFrame({
        "PLAYER_ID": [202, 101, 101, 202, 101, 202],
        "GAME_ID": ["B2", "A3", "A1", "B1", "A2", "B3"],
        "GAME_DATE": [
            "2026-10-04",
            "2026-10-05",
            "2026-10-01",
            "2026-10-02",
            "2026-10-03",
            "2026-10-06",
        ],
        "TEAM": ["BOS", "ATL", "ATL", "BOS", "ATL", "NY"],
        "MIN": [40, 30, 10, 20, 20, 50],
        "DK_POINTS": [60, 30, 10, 40, 20, 70],
    })

    result = add_historical_team_change_features(players)
    result = result.set_index("GAME_ID")

    # Player 101: three games with Atlanta.
    assert result.loc["A3", "GAMES_WITH_CURRENT_TEAM"] == 2
    assert result.loc["A3", "MIN_AVG_CURRENT_TEAM"] == pytest.approx(15)
    assert result.loc["A3", "DK_AVG_CURRENT_TEAM"] == pytest.approx(15)

    # Player 202: Boston history must remain independent.
    assert result.loc["B2", "GAMES_WITH_CURRENT_TEAM"] == 1
    assert result.loc["B2", "MIN_AVG_CURRENT_TEAM"] == pytest.approx(20)
    assert result.loc["B2", "DK_AVG_CURRENT_TEAM"] == pytest.approx(40)

    # Player 202 changes teams for the third game.
    assert result.loc["B3", "TEAM_CHANGE_FLAG"] == 1
    assert result.loc["B3", "GAMES_WITH_CURRENT_TEAM"] == 0
    assert pd.isna(result.loc["B3", "MIN_AVG_CURRENT_TEAM"])
    assert pd.isna(result.loc["B3", "DK_AVG_CURRENT_TEAM"])


def test_current_team_averages_handle_missing_and_zero_values():
    players = pd.DataFrame({
        "PLAYER_ID": [101] * 4,
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-07",
        ],
        "TEAM": ["ATL"] * 4,
        "MIN": [20.0, float("nan"), 0.0, 30.0],
        "DK_POINTS": [15.0, float("nan"), 0.0, 25.0],
    })

    result = add_historical_team_change_features(players)
    result = result.set_index("GAME_ID")

    assert result.loc["G4", "GAMES_WITH_CURRENT_TEAM"] == 3

    # The missing game is excluded from the average denominator.
    # The legitimate zero-minute game remains included.
    assert result.loc["G4", "MIN_AVG_CURRENT_TEAM"] == pytest.approx(10.0)
    assert result.loc["G4", "DK_AVG_CURRENT_TEAM"] == pytest.approx(7.5)
