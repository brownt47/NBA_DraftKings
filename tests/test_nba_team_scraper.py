
import sys
from pathlib import Path
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

import nba_team_scraper
from nba_team_scraper import fetch_team_game


# ---------------------------------------------------------
# TEAM GAME CALCULATION TEST
# Verify possessions, pace, and efficiency ratings.
# ---------------------------------------------------------

def test_fetch_team_game_regulation(monkeypatch):

    def make_stats(fg, ft, oreb, tov):
        return [
            {"name": "fieldGoalsMade-fieldGoalsAttempted",
             "displayValue": fg},
            {"name": "freeThrowsMade-freeThrowsAttempted",
             "displayValue": ft},
            {"name": "offensiveRebounds",
             "displayValue": str(oreb)},
            {"name": "totalTurnovers",
             "displayValue": str(tov)},
        ]

    data = {
        "header": {
            "competitions": [{
                "date": "2026-10-25T23:00Z",
                "status": {"type": {"completed": True}},
                "competitors": [
                    {
                        "team": {
                            "id": "1",
                            "abbreviation": "ATL",
                        },
                        "homeAway": "away",
                        "score": "110",
                        "linescores": [{}] * 4,
                    },
                    {
                        "team": {
                            "id": "2",
                            "abbreviation": "BOS",
                        },
                        "homeAway": "home",
                        "score": "105",
                        "linescores": [{}] * 4,
                    },
                ],
            }]
        },
        "boxscore": {
            "teams": [
                {
                    "team": {"id": "1"},
                    "statistics": make_stats(
                        "40-85", "20-25", 10, 12
                    ),
                },
                {
                    "team": {"id": "2"},
                    "statistics": make_stats(
                        "38-82", "17-22", 8, 14
                    ),
                },
            ]
        },
    }

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = data

    monkeypatch.setattr(
        nba_team_scraper.requests,
        "get",
        lambda url, timeout: response
    )

    result = fetch_team_game("401001")

    assert len(result) == 2

    atl = next(
        team for team in result if team["TEAM"] == "ATL"
    )
    bos = next(
        team for team in result if team["TEAM"] == "BOS"
    )

    # ATL possessions: 85 + .44(25) - 10 + 12 = 98
    # BOS possessions: 82 + .44(22) - 8 + 14 = 97.68
    # Average possessions = 97.84

    expected_possessions = 97.84

    assert atl["GAME_MINUTES"] == 48
    assert atl["POSS_EST"] == pytest.approx(expected_possessions)
    assert atl["PACE_EST"] == pytest.approx(expected_possessions)

    assert atl["OFF_RATING_EST"] == pytest.approx(
        100 * 110 / expected_possessions
    )
    assert atl["DEF_RATING_EST"] == pytest.approx(
        100 * 105 / expected_possessions
    )

    assert atl["OPPONENT"] == "BOS"
    assert bos["OPPONENT"] == "ATL"

    assert bos["OFF_RATING_EST"] == pytest.approx(
        100 * 105 / expected_possessions
    )


# ---------------------------------------------------------
# OVERTIME TEAM GAME TEST
# Verify pace is adjusted for a 53-minute game.
# ---------------------------------------------------------

def test_fetch_team_game_overtime(monkeypatch):

    def make_stats(fg, ft, oreb, tov):
        return [
            {
                "name": "fieldGoalsMade-fieldGoalsAttempted",
                "displayValue": fg,
            },
            {
                "name": "freeThrowsMade-freeThrowsAttempted",
                "displayValue": ft,
            },
            {
                "name": "offensiveRebounds",
                "displayValue": str(oreb),
            },
            {
                "name": "totalTurnovers",
                "displayValue": str(tov),
            },
        ]

    data = {
        "header": {
            "competitions": [{
                "date": "2026-10-26T01:00Z",
                "status": {"type": {"completed": True}},
                "competitors": [
                    {
                        "team": {"id": "1", "abbreviation": "ATL"},
                        "homeAway": "away",
                        "score": "120",
                        "linescores": [{}] * 5,
                    },
                    {
                        "team": {"id": "2", "abbreviation": "BOS"},
                        "homeAway": "home",
                        "score": "115",
                        "linescores": [{}] * 5,
                    },
                ],
            }]
        },
        "boxscore": {
            "teams": [
                {
                    "team": {"id": "1"},
                    "statistics": make_stats(
                        "44-95", "24-30", 12, 14
                    ),
                },
                {
                    "team": {"id": "2"},
                    "statistics": make_stats(
                        "42-92", "23-28", 10, 15
                    ),
                },
            ]
        },
    }

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = data

    monkeypatch.setattr(
        nba_team_scraper.requests,
        "get",
        lambda url, timeout: response
    )

    result = fetch_team_game("401002")

    assert len(result) == 2

    atl = next(
        team for team in result if team["TEAM"] == "ATL"
    )

    # ATL: 95 + .44(30) - 12 + 14 = 110.2
    # BOS: 92 + .44(28) - 10 + 15 = 109.32
    # Average possessions = 109.76

    expected_possessions = 109.76
    expected_pace = expected_possessions * 48 / 53

    assert atl["GAME_MINUTES"] == 53
    assert atl["POSS_EST"] == pytest.approx(expected_possessions)
    assert atl["PACE_EST"] == pytest.approx(expected_pace)

    # Offensive rating uses total possessions,
    # not possessions normalized to 48 minutes.
    assert atl["OFF_RATING_EST"] == pytest.approx(
        100 * 120 / expected_possessions
    )


# ---------------------------------------------------------
# UNFINISHED GAME TEST
# Verify incomplete games produce no team records.
# ---------------------------------------------------------

def test_fetch_team_game_unfinished(monkeypatch):

    data = {
        "header": {
            "competitions": [{
                "date": "2026-10-26T01:00Z",
                "status": {
                    "type": {"completed": False}
                },
                "competitors": [
                    {
                        "team": {
                            "id": "1",
                            "abbreviation": "ATL",
                        },
                        "linescores": [{}] * 3,
                    },
                    {
                        "team": {
                            "id": "2",
                            "abbreviation": "BOS",
                        },
                        "linescores": [{}] * 3,
                    },
                ],
            }]
        }
    }

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = data

    monkeypatch.setattr(
        nba_team_scraper.requests,
        "get",
        lambda url, timeout: response
    )

    result = fetch_team_game("401003")

    assert result == []


# ---------------------------------------------------------
# INSUFFICIENT GAME PERIODS TEST
# Verify games with fewer than four periods are excluded.
# ---------------------------------------------------------

def test_fetch_team_game_insufficient_periods(monkeypatch):

    data = {
        "header": {
            "competitions": [{
                "date": "2026-10-26T01:00Z",
                "status": {
                    "type": {"completed": True}
                },
                "competitors": [
                    {
                        "team": {
                            "id": "1",
                            "abbreviation": "ATL",
                        },
                        "linescores": [{}] * 3,
                    },
                    {
                        "team": {
                            "id": "2",
                            "abbreviation": "BOS",
                        },
                        "linescores": [{}] * 3,
                    },
                ],
            }]
        }
    }

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = data

    monkeypatch.setattr(
        nba_team_scraper.requests,
        "get",
        lambda url, timeout: response
    )

    result = fetch_team_game("401004")

    assert result == []


# ---------------------------------------------------------
# TEAM SCRAPER INCREMENTAL UPDATE TEST
# Verify existing games are preserved and only
# missing games are downloaded.
# ---------------------------------------------------------

def test_team_scraper_incremental_update(tmp_path, monkeypatch):

    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    # Player data contains two completed games
    pd.DataFrame({
        "GAME_ID": ["G1", "G1", "G2", "G2"],
        "GAME_DATE": [
            "2026-10-20", "2026-10-20",
            "2026-10-26", "2026-10-26"
        ]
    }).to_csv(player_file, index=False)

    # Team data already contains both teams for G1
    existing = pd.DataFrame({
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": ["2026-10-25", "2026-10-25"],
        "TEAM": ["ATL", "BOS"],
        "PTS": [110, 105],
    })

    existing.to_csv(output_file, index=False)

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )

    requested_games = []

    def fake_fetch_team_game(game_id):

        requested_games.append(game_id)

        return [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "MIA",
                "PTS": 115,
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "ORL",
                "PTS": 108,
            },
        ]

    monkeypatch.setattr(
        nba_team_scraper,
        "fetch_team_game",
        fake_fetch_team_game
    )

    nba_team_scraper.main()

    result = pd.read_csv(
        output_file,
        dtype={"GAME_ID": str}
    )

    # Only missing game G2 was downloaded
    assert requested_games == ["G2"]

    # Existing and new games are preserved
    assert set(result["GAME_ID"]) == {"G1", "G2"}

    # Exactly two team records per game
    assert result.groupby("GAME_ID").size().to_dict() == {
        "G1": 2,
        "G2": 2,
    }

    # Existing scores remain unchanged
    atl = result.loc[
        (result["GAME_ID"] == "G1")
        & (result["TEAM"] == "ATL")
    ].iloc[0]

    assert atl["PTS"] == 110

    # New game scores were saved
    mia = result.loc[
        (result["GAME_ID"] == "G2")
        & (result["TEAM"] == "MIA")
    ].iloc[0]

    assert mia["PTS"] == 115


# ---------------------------------------------------------
# TEAM SCRAPER FAILURE PROTECTION TEST
# Verify an ESPN failure preserves existing data.
# ---------------------------------------------------------

def test_team_scraper_preserves_data_on_failure(
    tmp_path, monkeypatch
):

    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    # Two games exist in the player dataset
    pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "GAME_DATE": ["2026-10-20", "2026-10-26"]
    }).to_csv(player_file, index=False)

    # G1 has already been downloaded
    existing = pd.DataFrame({
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": ["2026-10-25", "2026-10-25"],
        "TEAM": ["ATL", "BOS"],
        "PTS": [110, 105],
    })

    existing.to_csv(output_file, index=False)

    original_contents = output_file.read_bytes()

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )

    def fake_fetch_team_game(game_id):
        raise requests.exceptions.Timeout(
            "ESPN request timed out"
        )

    monkeypatch.setattr(
        nba_team_scraper,
        "fetch_team_game",
        fake_fetch_team_game
    )

    with pytest.raises(
        RuntimeError,
        match="Team scraper failed to download 1 game"
    ):
        nba_team_scraper.main()

    # Existing CSV must remain completely unchanged
    assert output_file.read_bytes() == original_contents


# ---------------------------------------------------------
# TEAM SCRAPER PARTIAL FAILURE TEST
# Save successful games even if another game fails.
# ---------------------------------------------------------

def test_team_scraper_partial_failure(tmp_path, monkeypatch, capsys):

    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    pd.DataFrame({
        "GAME_ID": ["G1", "G2", "G3"],
        "GAME_DATE": [
            "2026-10-20",
            "2026-10-26",
            "2026-10-26"
        ]
    }).to_csv(player_file, index=False)

    existing = pd.DataFrame({
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": ["2026-10-25", "2026-10-25"],
        "TEAM": ["ATL", "BOS"],
        "PTS": [110, 105],
    })

    existing.to_csv(output_file, index=False)

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )

    requested_games = []

    def fake_fetch_team_game(game_id):

        requested_games.append(game_id)

        if game_id == "G3":
            raise requests.exceptions.Timeout(
                "ESPN request timed out"
            )

        return [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "MIA",
                "PTS": 115,
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "ORL",
                "PTS": 108,
            },
        ]

    monkeypatch.setattr(
        nba_team_scraper,
        "fetch_team_game",
        fake_fetch_team_game
    )

    with pytest.raises(
        RuntimeError,
        match="Team scraper failed to download 1 game"
    ):
        nba_team_scraper.main()

    result = pd.read_csv(
        output_file,
        dtype={"GAME_ID": str}
    )

    assert requested_games == ["G2", "G3"]

    # Existing and successful games are saved
    assert set(result["GAME_ID"]) == {"G1", "G2"}

    # Failed game is not marked complete
    assert "G3" not in result["GAME_ID"]

    # Every saved game has two team records
    assert result.groupby("GAME_ID").size().eq(2).all()

    # Verify only the failed game is reported
    output = capsys.readouterr().out

    assert "Failed games: 1" in output
    assert "Failed game IDs: G3" in output


# ---------------------------------------------------------
# TEAM SCRAPER INCOMPLETE GAME TEST
# Reject games with fewer than two team records.
# ---------------------------------------------------------

def test_team_scraper_rejects_incomplete_game(
    tmp_path, monkeypatch
):

    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    pd.DataFrame({
        "GAME_ID": ["G1"],
        "GAME_DATE": ["2026-10-26"]
    }).to_csv(player_file, index=False)

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )

    def fake_fetch_team_game(game_id):

        # Simulate ESPN returning only one team
        return [{
            "GAME_ID": game_id,
            "GAME_DATE": "2026-10-26",
            "TEAM": "ATL",
            "PTS": 110,
        }]

    monkeypatch.setattr(
        nba_team_scraper,
        "fetch_team_game",
        fake_fetch_team_game
    )

    with pytest.raises(
        RuntimeError,
        match="Team scraper failed to download 1 game"
    ):
        nba_team_scraper.main()

    # No incomplete CSV should have been created
    assert not output_file.exists()


# ---------------------------------------------------------
# RECENT GAME REFRESH TEST
# Verify recent games are downloaded again and replaced.
# ---------------------------------------------------------

def test_team_scraper_refreshes_recent_game(
    tmp_path, monkeypatch
):
    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    pd.DataFrame({
        "GAME_ID": ["G1"],
        "GAME_DATE": ["2026-10-26"]
    }).to_csv(player_file, index=False)

    pd.DataFrame({
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": ["2026-10-26", "2026-10-26"],
        "TEAM": ["ATL", "BOS"],
        "PTS": [100, 95]
    }).to_csv(output_file, index=False)

    requested_games = []

    def fake_fetch_team_game(game_id):
        requested_games.append(game_id)
        return [
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "ATL",
                "PTS": 115
            },
            {
                "GAME_ID": game_id,
                "GAME_DATE": "2026-10-26",
                "TEAM": "BOS",
                "PTS": 108
            }
        ]

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "fetch_team_game", fake_fetch_team_game
    )

    nba_team_scraper.main()

    result = pd.read_csv(output_file)

    assert requested_games == ["G1"]
    assert len(result) == 2

    scores = dict(zip(result["TEAM"], result["PTS"]))
    assert scores == {"ATL": 115, "BOS": 108}


# ---------------------------------------------------------
# FAILED RECENT GAME REFRESH TEST
# Preserve existing data when ESPN fails.
# ---------------------------------------------------------

def test_team_scraper_failed_refresh_preserves_data(
    tmp_path, monkeypatch, capsys
):
    player_file = tmp_path / "player_games.csv"
    output_file = tmp_path / "team_games.csv"

    pd.DataFrame({
        "GAME_ID": ["G1"],
        "GAME_DATE": ["2026-10-26"]
    }).to_csv(player_file, index=False)

    pd.DataFrame({
        "GAME_ID": ["G1", "G1"],
        "GAME_DATE": ["2026-10-26", "2026-10-26"],
        "TEAM": ["ATL", "BOS"],
        "PTS": [100, 95]
    }).to_csv(output_file, index=False)

    original_contents = output_file.read_bytes()
    requested_games = []

    def fake_fetch_team_game(game_id):
        requested_games.append(game_id)
        raise requests.exceptions.Timeout(
            "ESPN request timed out"
        )

    monkeypatch.setattr(
        nba_team_scraper, "PLAYER_FILE", player_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "OUTPUT_FILE", output_file
    )
    monkeypatch.setattr(
        nba_team_scraper, "fetch_team_game", fake_fetch_team_game
    )

    with pytest.raises(
        RuntimeError,
        match="Team scraper failed to download 1 game"
    ):
        nba_team_scraper.main()

    assert requested_games == ["G1"]
    assert output_file.read_bytes() == original_contents
    # Verify the failed refresh is reported
    output = capsys.readouterr().out

    assert "Failed games: 1" in output
    assert "Failed game IDs: G1" in output