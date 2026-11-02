import pandas as pd
import pytest

from src.features.schedule import add_historical_schedule_features


def test_historical_rest_days_and_back_to_backs():
    games = pd.DataFrame({
        "GAME_ID": ["G3", "G1", "G4", "G2"],
        "GAME_DATE": [
            "2026-10-06",
            "2026-10-01",
            "2026-10-10",
            "2026-10-02",
        ],
        "TEAM": ["ATL"] * 4,
        "OPPONENT": ["BOS", "NY", "MIA", "CHA"],
        "HOME_AWAY": ["home", "home", "away", "away"],
    })

    result = add_historical_schedule_features(games)
    result = result.sort_values("GAME_DATE").reset_index(drop=True)

    assert result["GAME_ID"].tolist() == ["G1", "G2", "G3", "G4"]

    assert pd.isna(result.loc[0, "DAYS_REST"])
    assert result["DAYS_REST"].iloc[1:].tolist() == [0, 3, 3]

    assert result["IS_BACK_TO_BACK"].tolist() == [0, 1, 0, 0]


def test_schedule_congestion_and_road_trips():
    games = pd.DataFrame({
        "GAME_ID": ["G1", "G2", "G3", "G4", "G5"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-08",
            "2026-10-16",
        ],
        "TEAM": ["ATL"] * 5,
        "OPPONENT": ["BOS", "NY", "MIA", "CHA", "ORL"],
        "HOME_AWAY": ["home", "away", "away", "home", "home"],
    })

    result = add_historical_schedule_features(games)
    result = result.sort_values("GAME_DATE").reset_index(drop=True)

    assert result["GAMES_LAST_7_DAYS"].tolist() == [0, 1, 2, 3, 0]
    assert result["GAMES_LAST_14_DAYS"].tolist() == [0, 1, 2, 3, 3]

    assert result["ROAD_TRIP_GAME_NUMBER"].tolist() == [0, 1, 2, 0, 0]
    assert result["IS_FIRST_HOME_AFTER_TRIP"].tolist() == [0, 0, 0, 1, 0]


def test_historical_travel_distance_and_timezones():
    from src.features.schedule import add_historical_travel_features

    games = pd.DataFrame({
        "GAME_ID": ["G1", "G2", "G3"],
        "GAME_DATE": [
            "2026-10-21",
            "2026-10-23",
            "2026-10-25",
        ],
        "TEAM": ["ATL"] * 3,
        "OPPONENT": ["BOS", "HOU", "MIA"],
        "HOME_AWAY": ["home", "away", "home"],
    })

    games = add_historical_schedule_features(games)
    result = add_historical_travel_features(games)
    result = result.sort_values("GAME_DATE").reset_index(drop=True)

    assert pd.isna(result.loc[0, "TRAVEL_MILES"])
    assert pd.isna(result.loc[0, "TIMEZONE_CHANGE"])

    assert result.loc[1, "TRAVEL_MILES"] == pytest.approx(701, abs=5)
    assert result.loc[2, "TRAVEL_MILES"] == pytest.approx(701, abs=5)

    assert result["TIMEZONE_CHANGE"].iloc[1:].tolist() == [-1, 1]
    assert result["TIMEZONES_CROSSED"].iloc[1:].tolist() == [1, 1]


def test_player_recent_minutes_excludes_current_game():
    from src.features.schedule import add_player_recent_minutes

    players = pd.DataFrame({
        "PLAYER_ID": [101, 101, 202, 101, 101, 101],
        "GAME_ID": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-08",
            "2026-10-10",
            "2026-10-16",
        ],
        "MIN": [20, 30, 40, 25, 35, 50],
    })

    result = add_player_recent_minutes(players)
    result = result.set_index("GAME_ID")

    assert result.loc["G1", "PLAYER_MINUTES_LAST_7_DAYS"] == 0
    assert result.loc["G2", "PLAYER_MINUTES_LAST_7_DAYS"] == 20
    assert result.loc["G4", "PLAYER_MINUTES_LAST_7_DAYS"] == 50
    assert result.loc["G5", "PLAYER_MINUTES_LAST_7_DAYS"] == 55
    assert result.loc["G6", "PLAYER_MINUTES_LAST_7_DAYS"] == 35

    assert result.loc["G6", "PLAYER_MINUTES_LAST_14_DAYS"] == 90

    # Another player's minutes must not enter player 101's totals.
    assert result.loc["G3", "PLAYER_MINUTES_LAST_7_DAYS"] == 0


def test_player_workload_interactions():
    from src.features.schedule import add_player_workload_features

    players = pd.DataFrame({
        "PLAYER_ID": [101, 101, 202],
        "IS_BACK_TO_BACK": [1, 0, 0],
        "MIN_AVG_LAST_5": [30.0, 30.0, float("nan")],
        "LAST_GAME_MIN": [38.0, 38.0, float("nan")],
    })

    result = add_player_workload_features(players)

    assert result["B2B_MIN_LOAD"].iloc[:2].tolist() == [30.0, 0.0]
    assert result["B2B_LAST_GAME_MIN"].iloc[:2].tolist() == [38.0, 0.0]
    assert result["B2B_EXCESS_MIN_LOAD"].iloc[:2].tolist() == [8.0, 0.0]

    # No prior history should remain missing, not become invented zeros.
    assert pd.isna(result.loc[2, "B2B_MIN_LOAD"])
    assert pd.isna(result.loc[2, "B2B_LAST_GAME_MIN"])
    assert pd.isna(result.loc[2, "B2B_EXCESS_MIN_LOAD"])
