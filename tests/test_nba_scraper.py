
import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

# ---------------------------------------------------------
# IMPORT SETUP
# Allow tests to import modules from src.
# ---------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import nba_scraper
from nba_scraper import (
    split_shooting,
    discover_games,
    parse_boxscore,
)


# ---------------------------------------------------------
# SHOOTING STATISTICS PARSING TEST
# Verify made and attempted shots are extracted.
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "input_value, expected",
    [
        ("8-18", (8, 18)),
        ("0-5", (0, 5)),
        ("10-10", (10, 10)),
        ("0-0", (0, 0)),
    ]
)
def test_split_shooting(input_value, expected):

    result = split_shooting(input_value)

    assert result == expected


# ---------------------------------------------------------
# GAME DISCOVERY TEST
# Verify only completed regular-season games are included.
# ---------------------------------------------------------

def test_discover_games(monkeypatch):

    test_date = date(2026, 10, 25)

    # Limit discovery to one day
    monkeypatch.setattr(
        nba_scraper,
        "END_DATE",
        test_date
    )

    # Simulated ESPN scoreboard response
    scoreboard = {
        "events": [
            {
                "id": "401001",
                "name": "Atlanta Hawks at Boston Celtics",
                "date": "2026-10-26T00:00Z",
                "season": {"type": 2},
                "status": {"type": {"completed": True}},
            },
            {
                "id": "401002",
                "name": "Miami Heat at Orlando Magic",
                "season": {"type": 2},
                "status": {"type": {"completed": False}},
            },
            {
                "id": "401003",
                "name": "Chicago Bulls at Detroit Pistons",
                "season": {"type": 1},
                "status": {"type": {"completed": True}},
            },
        ]
    }

    # Replace ESPN API request with simulated data
    monkeypatch.setattr(
        nba_scraper,
        "get_json",
        lambda url, params: scoreboard
    )

    result = discover_games(start_date=test_date)

    # Only the completed regular-season game is included
    assert len(result) == 1
    assert "401001" in result

    # Verify game information
    assert result["401001"]["GAME_ID"] == "401001"
    assert result["401001"]["GAME_DATE"] == "2026-10-25"
    assert result["401001"]["MATCHUP"] == (
        "Atlanta Hawks at Boston Celtics"
    )


# ---------------------------------------------------------
# PLAYER BOX SCORE PARSING TEST
# Verify ESPN player statistics are extracted correctly.
# ---------------------------------------------------------

def test_parse_boxscore(monkeypatch):

    game = {
        "GAME_ID": "401001",
        "GAME_DATE": "2026-10-25",
        "MATCHUP": "Atlanta Hawks at Boston Celtics",
    }

    columns = [
        "MIN", "PTS", "FG", "3PT", "FT",
        "REB", "AST", "TO", "STL", "BLK",
        "OREB", "DREB", "PF", "+/-",
    ]

    def make_player(player_id, name, stats):
        return {
            "athlete": {
                "id": player_id,
                "displayName": name,
            },
            "stats": stats,
            "didNotPlay": False,
        }

    scoreboard = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "ATL"},
                    "statistics": [{
                        "names": columns,
                        "athletes": [
                            make_player(
                                "9001",
                                "Player A",
                                [
                                    "32", "25", "9-18", "3-7",
                                    "4-5", "8", "6", "2",
                                    "1", "2", "2", "6",
                                    "3", "+10",
                                ],
                            )
                        ],
                    }],
                },
                {
                    "team": {"abbreviation": "BOS"},
                    "statistics": [{
                        "names": columns,
                        "athletes": [
                            make_player(
                                "9002",
                                "Player B",
                                [
                                    "28", "18", "7-15", "2-6",
                                    "2-2", "5", "4", "1",
                                    "2", "0", "1", "4",
                                    "2", "-5",
                                ],
                            )
                        ],
                    }],
                },
            ]
        }
    }

    # Replace ESPN API request with simulated response
    monkeypatch.setattr(
        nba_scraper,
        "get_json",
        lambda url, params: scoreboard
    )

    result = parse_boxscore(game)

    # Two player records extracted
    assert len(result) == 2

    player_a = result[0]
    player_b = result[1]

    # Player identity and game information
    assert player_a["PLAYER_ID"] == "9001"
    assert player_a["GAME_ID"] == "401001"
    assert player_a["TEAM"] == "ATL"

    # Shooting statistics
    assert player_a["FGM"] == 9
    assert player_a["FGA"] == 18
    assert player_a["FG3M"] == 3
    assert player_a["FG3A"] == 7
    assert player_a["FTM"] == 4
    assert player_a["FTA"] == 5

    # Other statistics
    assert player_a["PTS"] == 25
    assert player_a["REB"] == 8
    assert player_a["AST"] == 6
    assert player_a["TOV"] == 2
    assert player_a["STL"] == 1
    assert player_a["BLK"] == 2

    # Second player
    assert player_b["PLAYER_ID"] == "9002"
    assert player_b["TEAM"] == "BOS"
    assert player_b["PTS"] == 18


# ---------------------------------------------------------
# DID NOT PLAY TEST
# Verify inactive players are excluded from box scores.
# ---------------------------------------------------------

def test_parse_boxscore_did_not_play(monkeypatch):

    columns = [
        "MIN", "PTS", "FG", "3PT", "FT",
        "REB", "AST", "TO", "STL", "BLK",
        "OREB", "DREB", "PF", "+/-",
    ]

    def make_player(player_id, name, did_not_play):

        return {
            "athlete": {
                "id": player_id,
                "displayName": name,
            },
            "didNotPlay": did_not_play,
            "stats": [
                "30", "20", "8-15", "2-5", "2-2",
                "5", "4", "1", "1", "0",
                "1", "4", "2", "+5",
            ],
        }

    boxscore = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "ATL"},
                    "statistics": [{
                        "names": columns,
                        "athletes": [
                            make_player("9001", "Active Player", False),
                            make_player("9002", "Inactive Player", True),
                        ],
                    }],
                },
                {
                    "team": {"abbreviation": "BOS"},
                    "statistics": [{
                        "names": columns,
                        "athletes": [
                            make_player("9003", "Opponent Player", False),
                        ],
                    }],
                },
            ]
        }
    }

    monkeypatch.setattr(
        nba_scraper,
        "get_json",
        lambda url, params: boxscore
    )

    game = {
        "GAME_ID": "401001",
        "GAME_DATE": "2026-10-25",
        "MATCHUP": "Atlanta Hawks at Boston Celtics",
    }

    result = parse_boxscore(game)

    # Only players who participated are included
    assert len(result) == 2

    player_ids = {record["PLAYER_ID"] for record in result}

    assert player_ids == {"9001", "9003"}
    assert "9002" not in player_ids


# ---------------------------------------------------------
# INCREMENTAL UPDATE TEST
# Verify refreshed games replace existing records
# while older historical games are preserved.
# ---------------------------------------------------------

def test_incremental_update(tmp_path, monkeypatch):

    output_file = tmp_path / "nba_player_games_2026_27.csv"

    # Existing historical data
    existing = pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": ["2026-10-25", "2026-10-29"],
        "PLAYER_ID": ["9001", "9002"],
        "PLAYER_NAME": ["Player A", "Player B"],
        "TEAM": ["ATL", "BOS"],
        "MIN": [30, 32],
        "PTS": [20, 25],
        "REB": [5, 6],
        "AST": [4, 5],
        "FG3M": [2, 3],
        "TOV": [2, 1],
        "STL": [1, 2],
        "BLK": [0, 1],
    })

    existing.to_csv(output_file, index=False)

    # Redirect scraper output to temporary file
    monkeypatch.setattr(
        nba_scraper,
        "OUTPUT_FILE",
        output_file
    )

    # Simulate discovered games
    discovered = {
        "G2": {
            "GAME_ID": "G2",
            "GAME_DATE": "2026-10-29",
            "MATCHUP": "Boston at Atlanta",
        },
        "G3": {
            "GAME_ID": "G3",
            "GAME_DATE": "2026-10-30",
            "MATCHUP": "Miami at Orlando",
        },
    }

    monkeypatch.setattr(
        nba_scraper,
        "discover_games",
        lambda start_date, end_date=None: discovered
    )

    # Simulated updated player records
    def fake_parse_boxscore(game):

        game_id = game["GAME_ID"]

        return [{
            **game,
            "PLAYER_ID": "9002" if game_id == "G2" else "9003",
            "PLAYER_NAME": "Player B" if game_id == "G2" else "Player C",
            "TEAM": "BOS" if game_id == "G2" else "MIA",
            "MIN": 35,
            "PTS": 30,
            "REB": 8,
            "AST": 6,
            "FG3M": 4,
            "TOV": 2,
            "STL": 1,
            "BLK": 1,
        }]

    monkeypatch.setattr(
        nba_scraper,
        "parse_boxscore",
        fake_parse_boxscore
    )

    # Avoid unnecessary delays
    monkeypatch.setattr(
        nba_scraper.time,
        "sleep",
        lambda seconds: None
    )

    nba_scraper.main()

    result = pd.read_csv(output_file, dtype={
        "GAME_ID": str,
        "PLAYER_ID": str,
    })

    # Old game preserved
    assert "G1" in result["GAME_ID"].values

    # Refreshed game appears exactly once
    assert (result["GAME_ID"] == "G2").sum() == 1

    # Updated statistics replace old values
    refreshed = result.loc[
        result["GAME_ID"] == "G2"
    ].iloc[0]

    assert refreshed["PTS"] == 30

    # New game included
    assert "G3" in result["GAME_ID"].values

    # No duplicate player-game records
    assert not result.duplicated(
        subset=["GAME_ID", "PLAYER_ID"]
    ).any()

    # All three games retained
    assert result["GAME_ID"].nunique() == 3


# ---------------------------------------------------------
# DUPLICATE PLAYER-GAME PROTECTION TEST
# Verify duplicate records stop the update
# and leave the existing CSV unchanged.
# ---------------------------------------------------------

def test_duplicate_player_game_protection(tmp_path, monkeypatch):

    output_file = tmp_path / "nba_player_games_2026_27.csv"

    # Existing data that must remain unchanged
    existing = pd.DataFrame({
        "GAME_ID": ["G1"],
        "GAME_DATE": ["2026-10-25"],
        "PLAYER_ID": ["9001"],
        "PLAYER_NAME": ["Player A"],
        "TEAM": ["ATL"],
        "MIN": [30],
        "PTS": [20],
        "REB": [5],
        "AST": [4],
        "FG3M": [2],
        "TOV": [2],
        "STL": [1],
        "BLK": [0],
    })

    existing.to_csv(output_file, index=False)

    # Save original file contents for comparison
    original_contents = output_file.read_bytes()

    monkeypatch.setattr(
        nba_scraper,
        "OUTPUT_FILE",
        output_file
    )

    game = {
        "GAME_ID": "G2",
        "GAME_DATE": "2026-10-30",
        "MATCHUP": "Boston at Atlanta",
    }

    monkeypatch.setattr(
        nba_scraper,
        "discover_games",
        lambda start_date, end_date=None: {"G2": game}
    )

    # Simulate ESPN returning the same player twice
    duplicate_record = {
        **game,
        "PLAYER_ID": "9002",
        "PLAYER_NAME": "Player B",
        "TEAM": "BOS",
        "MIN": 35,
        "PTS": 30,
        "REB": 8,
        "AST": 6,
        "FG3M": 4,
        "TOV": 2,
        "STL": 1,
        "BLK": 1,
    }

    monkeypatch.setattr(
        nba_scraper,
        "parse_boxscore",
        lambda game: [
            duplicate_record.copy(),
            duplicate_record.copy(),
        ]
    )

    monkeypatch.setattr(
        nba_scraper.time,
        "sleep",
        lambda seconds: None
    )

    # Duplicate detection must stop the update
    with pytest.raises(
        RuntimeError,
        match="Duplicate player-game records"
    ):
        nba_scraper.main()

    # Original CSV must remain completely unchanged
    assert output_file.read_bytes() == original_contents

    # No temporary file should remain
    assert not output_file.with_suffix(".tmp").exists()


# ---------------------------------------------------------
# BOX SCORE FAILURE PROTECTION TEST
# Verify an ESPN failure stops the update
# without modifying the existing CSV.
# ---------------------------------------------------------

def test_boxscore_failure_protection(tmp_path, monkeypatch):

    output_file = tmp_path / "nba_player_games_2026_27.csv"

    # Create existing historical data
    existing = pd.DataFrame({
        "GAME_ID": ["G1"],
        "GAME_DATE": ["2026-10-25"],
        "PLAYER_ID": ["9001"],
        "PLAYER_NAME": ["Player A"],
        "TEAM": ["ATL"],
        "MIN": [30],
        "PTS": [20],
        "REB": [5],
        "AST": [4],
        "FG3M": [2],
        "TOV": [2],
        "STL": [1],
        "BLK": [0],
    })

    existing.to_csv(output_file, index=False)

    # Preserve original CSV contents
    original_contents = output_file.read_bytes()

    monkeypatch.setattr(
        nba_scraper,
        "OUTPUT_FILE",
        output_file
    )

    game = {
        "GAME_ID": "G2",
        "GAME_DATE": "2026-10-30",
        "MATCHUP": "Boston at Atlanta",
    }

    monkeypatch.setattr(
        nba_scraper,
        "discover_games",
        lambda start_date, end_date=None: {"G2": game}
    )

    # Simulate ESPN failing to return the box score
    def fake_parse_boxscore(game):
        raise RuntimeError("ESPN box score unavailable")

    monkeypatch.setattr(
        nba_scraper,
        "parse_boxscore",
        fake_parse_boxscore
    )

    # The scraper should stop with an error
    with pytest.raises(
        RuntimeError,
        match="ESPN box score unavailable"
    ):
        nba_scraper.main()

    # Existing CSV must remain unchanged
    assert output_file.read_bytes() == original_contents

    # No temporary file should remain
    assert not output_file.with_suffix(".tmp").exists()



# ---------------------------------------------------------
# ESPN API RETRY SUCCESS TEST
# Verify a failed request is retried successfully.
# ---------------------------------------------------------

def test_get_json_retry_success(monkeypatch):

    attempts = []

    def fake_get(url, params, timeout):

        attempts.append(1)

        # First request fails
        if len(attempts) == 1:
            raise requests.ConnectionError(
                "Temporary connection failure"
            )

        # Second request succeeds
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "events": [{"id": "401001"}]
        }

        return response

    monkeypatch.setattr(
        nba_scraper.session,
        "get",
        fake_get
    )

    # Eliminate retry delays during testing
    monkeypatch.setattr(
        nba_scraper.time,
        "sleep",
        lambda seconds: None
    )

    result = nba_scraper.get_json(
        nba_scraper.SCOREBOARD_URL,
        {"dates": "20261025"}
    )

    # First attempt failed, second succeeded
    assert len(attempts) == 2

    # Correct data returned
    assert result == {
        "events": [{"id": "401001"}]
    }


# ---------------------------------------------------------
# ESPN API RETRY FAILURE TEST
# Verify three failed requests raise an error.
# ---------------------------------------------------------

def test_get_json_retry_exhausted(monkeypatch):

    attempts = []
    delays = []

    def fake_get(url, params, timeout):

        attempts.append(1)

        raise requests.ConnectionError(
            "ESPN connection unavailable"
        )

    monkeypatch.setattr(
        nba_scraper.session,
        "get",
        fake_get
    )

    # Record retry delays without actually waiting
    monkeypatch.setattr(
        nba_scraper.time,
        "sleep",
        lambda seconds: delays.append(seconds)
    )

    with pytest.raises(
        RuntimeError,
        match="Failed after 3 attempts"
    ):
        nba_scraper.get_json(
            nba_scraper.SCOREBOARD_URL,
            {"dates": "20261025"}
        )

    # Exactly three attempts
    assert len(attempts) == 3

    # Delays occur only between attempts
    assert delays == [3, 6]


def test_missing_player_id_with_no_minutes(monkeypatch):
    from src import nba_scraper

    game = {
        "GAME_ID": "401810150",
        "GAME_DATE": "2025-11-29",
        "MATCHUP": "Chicago Bulls at Indiana Pacers",
    }

    fake_data = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "CHI"},
                    "statistics": [
                        {
                            "names": [
                                "MIN", "PTS", "FG", "3PT", "FT",
                                "REB", "AST", "TO", "STL", "BLK",
                                "OREB", "DREB", "PF", "+/-",
                            ],
                            "athletes": [
                                {
                                    "didNotPlay": False,
                                    "athlete": {
                                        "links": [],
                                        "shortName": "Olbrich",
                                    },
                                    "stats": [
                                        "--", "0", "0-0", "0-0", "0-0",
                                        "0", "0", "0", "0", "0",
                                        "0", "0", "0", "0",
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "team": {"abbreviation": "IND"},
                    "statistics": [],
                },
            ]
        }
    }

    monkeypatch.setattr(
        nba_scraper,
        "get_json",
        lambda url, params: fake_data,
    )

    # The malformed record is skipped, leaving no valid players.
    # The scraper should raise its existing empty-box-score error.
    import pytest

    with pytest.raises(ValueError, match="No player records"):
        nba_scraper.parse_boxscore(game)