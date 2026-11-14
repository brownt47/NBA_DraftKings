
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import nba_scraper
from src.nba_scraper import parse_boxscore


def test_participation_extraction():
    game = {
        "GAME_ID": "12345",
        "GAME_DATE": "2026-11-13",
        "MATCHUP": "Test",
    }

    def athlete(player_id, name, dnp=False):
        return {
            "athlete": {
                "id": player_id,
                "displayName": name,
            },
            "didNotPlay": dnp,
            "active": not dnp,
            "starter": False,
            "reason": "COACH'S DECISION" if dnp else None,
            "stats": [] if dnp else [
                "20", "2-5", "1-2", "2-2",
                "3", "4", "1", "1", "0",
                "0", "1", "0", "+5", "7"
            ],
        }

    data = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "DEN"},
                    "statistics": [{
                        "names": [
                            "MIN", "FG", "3PT", "FT",
                            "OREB", "DREB", "REB",
                            "AST", "STL", "BLK",
                            "TO", "PF", "+/-", "PTS"
                        ],
                        "athletes": [
                            athlete("1", "Player A"),
                            athlete("2", "Player B", dnp=True),
                        ],
                    }],
                },
                {
                    "team": {"abbreviation": "MEM"},
                    "statistics": [{
                        "names": [
                            "MIN", "FG", "3PT", "FT",
                            "OREB", "DREB", "REB",
                            "AST", "STL", "BLK",
                            "TO", "PF", "+/-", "PTS"
                        ],
                        "athletes": [
                            athlete("3", "Player C"),
                        ],
                    }],
                },
            ]
        }
    }

    with patch("src.nba_scraper.get_json", return_value=data):
        original = parse_boxscore(game)
        played, participation = parse_boxscore(
            game, include_participation=True
        )

    assert original == played
    assert len(played) == 2
    assert len(participation) == 3

    dnp = [p for p in participation if p["DID_NOT_PLAY"]]
    assert len(dnp) == 1
    assert dnp[0]["PLAYER_ID"] == "2"
    assert dnp[0]["DNP_REASON"] == "COACH'S DECISION"

    assert all(p["PLAYED"] for p in participation if not p["DID_NOT_PLAY"])

def test_backfill_preserves_historical_statistics(tmp_path, monkeypatch):
    import pandas as pd
    from datetime import date
    
    stats_file = tmp_path / "nba_player_games_2026_27.csv"
    participation_file = tmp_path / "nba_player_participation_2026_27.csv"

    existing = pd.DataFrame([
        {
            "GAME_ID": "G1",
            "GAME_DATE": "2026-10-25",
            "PLAYER_ID": "100",
            "PLAYER_NAME": "Historical Player",
            "TEAM": "DEN",
            "MIN": 30,
            "PTS": 99,
            "REB": 5,
            "AST": 4,
            "FG3M": 2,
            "TOV": 1,
            "STL": 1,
            "BLK": 0,
        },
        {
            "GAME_ID": "G2",
            "GAME_DATE": "2026-11-12",
            "PLAYER_ID": "200",
            "PLAYER_NAME": "Recent Player",
            "TEAM": "MEM",
            "MIN": 25,
            "PTS": 10,
            "REB": 3,
            "AST": 2,
            "FG3M": 1,
            "TOV": 1,
            "STL": 0,
            "BLK": 0,
        },
    ])
    existing.to_csv(stats_file, index=False)

    monkeypatch.setattr(nba_scraper, "OUTPUT_FILE", stats_file)
    monkeypatch.setattr(
        nba_scraper, "PARTICIPATION_OUTPUT_FILE", participation_file
    )
    monkeypatch.setattr(
        nba_scraper.time, "sleep", lambda seconds: None
    )

    games = {
        game_id: {
            "GAME_ID": game_id,
            "GAME_DATE": game_date,
            "MATCHUP": "Test matchup",
        }
        for game_id, game_date in [
            ("G1", "2026-10-25"),
            ("G2", "2026-11-12"),
        ]
    }

    monkeypatch.setattr(
        nba_scraper,
        "discover_games",
        lambda start_date, end_date=None: games,
    )

    def fake_parse_boxscore(game, include_participation=False):
        record = {
            **game,
            "PLAYER_ID": "100" if game["GAME_ID"] == "G1" else "200",
            "PLAYER_NAME": "Historical Player"
            if game["GAME_ID"] == "G1" else "Recent Player",
            "TEAM": "DEN" if game["GAME_ID"] == "G1" else "MEM",
            "MIN": 30,
            "PTS": 20,
            "REB": 5,
            "AST": 4,
            "FG3M": 2,
            "TOV": 1,
            "STL": 1,
            "BLK": 0,
        }
        participation = {
            "GAME_ID": game["GAME_ID"],
            "GAME_DATE": game["GAME_DATE"],
            "PLAYER_ID": record["PLAYER_ID"],
            "PLAYER_NAME": record["PLAYER_NAME"],
            "TEAM": record["TEAM"],
            "PLAYED": True,
            "DID_NOT_PLAY": False,
            "DNP_REASON": None,
            "ACTIVE": True,
            "STARTER": True,
            "MIN": 30,
        }
        if include_participation:
            return [record], [participation]
        return [record]

    monkeypatch.setattr(
        nba_scraper, "parse_boxscore", fake_parse_boxscore
    )

    nba_scraper.main(["--season", "2026-27", "--participation"])

    result = pd.read_csv(stats_file, dtype={"GAME_ID": str})

    historical = result.loc[result["GAME_ID"] == "G1"].iloc[0]
    recent = result.loc[result["GAME_ID"] == "G2"].iloc[0]

    assert historical["PTS"] == 99
    assert recent["PTS"] == 20
    assert result["GAME_ID"].nunique() == 2

    assert participation_file.exists()
    participation_result = pd.read_csv(participation_file)
    assert set(participation_result["GAME_ID"]) == {"G1", "G2"}


def test_incremental_participation_refresh(tmp_path, monkeypatch):
    import pandas as pd

    stats_file = tmp_path / "nba_player_games_2026_27.csv"
    participation_file = tmp_path / "nba_player_participation_2026_27.csv"

    # Existing historical and recent participation records
    existing_participation = pd.DataFrame([
        {
            "GAME_ID": "G1",
            "GAME_DATE": "2026-10-25",
            "PLAYER_ID": "100",
            "PLAYER_NAME": "Historical Player",
            "TEAM": "DEN",
            "PLAYED": False,
            "DID_NOT_PLAY": True,
            "DNP_REASON": "Coach's Decision",
            "ACTIVE": True,
            "STARTER": False,
            "MIN": 0,
        },
        {
            "GAME_ID": "G2",
            "GAME_DATE": "2026-11-12",
            "PLAYER_ID": "200",
            "PLAYER_NAME": "Recent Player",
            "TEAM": "MEM",
            "PLAYED": False,
            "DID_NOT_PLAY": True,
            "DNP_REASON": "Old reason",
            "ACTIVE": True,
            "STARTER": False,
            "MIN": 0,
        },
    ])
    existing_participation.to_csv(participation_file, index=False)

    monkeypatch.setattr(nba_scraper, "OUTPUT_FILE", stats_file)
    monkeypatch.setattr(
        nba_scraper, "PARTICIPATION_OUTPUT_FILE", participation_file
    )
    monkeypatch.setattr(nba_scraper.time, "sleep", lambda seconds: None)

    games = {
        "G2": {
            "GAME_ID": "G2",
            "GAME_DATE": "2026-11-12",
            "MATCHUP": "Test matchup",
        }
    }

    monkeypatch.setattr(
        nba_scraper,
        "discover_games",
        lambda start_date, end_date=None: games,
    )

    def fake_parse_boxscore(game, include_participation=False):
        stats = [{
            **game,
            "PLAYER_ID": "200",
            "PLAYER_NAME": "Recent Player",
            "TEAM": "MEM",
            "MIN": 30,
            "PTS": 20,
            "REB": 5,
            "AST": 4,
            "FG3M": 2,
            "TOV": 1,
            "STL": 1,
            "BLK": 0,
        }]

        participation = [{
            "GAME_ID": "G2",
            "GAME_DATE": "2026-11-12",
            "PLAYER_ID": "200",
            "PLAYER_NAME": "Recent Player",
            "TEAM": "MEM",
            "PLAYED": True,
            "DID_NOT_PLAY": False,
            "DNP_REASON": None,
            "ACTIVE": True,
            "STARTER": True,
            "MIN": 30,
        }]

        if include_participation:
            return stats, participation
        return stats

    monkeypatch.setattr(
        nba_scraper, "parse_boxscore", fake_parse_boxscore
    )

    # Run twice to verify repeatability.
    for _ in range(2):
        nba_scraper.main(["--season", "2026-27", "--participation"])

    result = pd.read_csv(
        participation_file,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    assert len(result) == 2
    assert not result.duplicated(
        ["GAME_ID", "PLAYER_ID"]
    ).any()

    historical = result.loc[result["GAME_ID"] == "G1"].iloc[0]
    recent = result.loc[result["GAME_ID"] == "G2"].iloc[0]

    # Historical DNP information must remain unchanged.
    assert bool(historical["DID_NOT_PLAY"])
    assert historical["DNP_REASON"] == "Coach's Decision"

    # Recent participation must reflect refreshed data.
    assert bool(recent["PLAYED"])
    assert not bool(recent["DID_NOT_PLAY"])
    assert recent["MIN"] == 30    

def test_participation_rolling_windows():
    import pandas as pd
    import pytest
    from src.features.rotation import add_participation_reliability_features

    # Twelve consecutive team games for one player.
    # Game 4 is a confirmed coach DNP.
    # Game 7 has unknown participation.
    # Game 12 is played, but must not affect its own features.
    records = []

    for game_number in range(1, 13):
        played = game_number not in (4, 7)

        if game_number == 7:
            continue

        records.append({
            "GAME_ID": str(game_number),
            "GAME_DATE": (
                pd.Timestamp("2026-10-20")
                + pd.Timedelta(days=game_number - 1)
            ),
            "TEAM": "DEN",
            "PLAYER_ID": "100",
            "PLAYED": played,
            "DID_NOT_PLAY": not played,
            "DNP_REASON": (
                "COACH'S DECISION"
                if game_number == 4 else None
            ),
        })

    participation = pd.DataFrame(records)

    teams = pd.DataFrame([
        {
            "GAME_ID": str(i),
            "GAME_DATE": (
                pd.Timestamp("2026-10-20")
                + pd.Timedelta(days=i - 1)
            ),
            "TEAM": "DEN",
        }
        for i in range(1, 13)
    ])

    players = teams.copy()
    players["PLAYER_ID"] = "100"

    result = add_participation_reliability_features(
        players, teams, participation
    )

    assert len(result) == 12
    assert not result.duplicated(
        ["GAME_ID", "PLAYER_ID"]
    ).any()

    result = result.set_index("GAME_ID")

    # Game 1 has no previous team games.
    assert pd.isna(
        result.loc["1", "APPEARANCE_RATE_LAST_5"]
    )

    # Before game 3, both previous games were played.
    assert result.loc["3", "PARTICIPATION_KNOWN_LAST_5"] == 2
    assert result.loc["3", "APPEARANCE_RATE_LAST_5"] == 1
    assert result.loc["3", "PARTICIPATION_COVERAGE_LAST_5"] == 1

    # Before game 5, game 4 is a confirmed coach DNP.
    assert result.loc["5", "COACH_DNP_LAST_5"] == 1
    assert result.loc["5", "APPEARANCE_RATE_LAST_5"] == 0.75

    # Before game 8, game 7 is unknown, not a confirmed DNP.
    assert result.loc["8", "PARTICIPATION_KNOWN_LAST_5"] == 4
    assert result.loc["8", "PARTICIPATION_COVERAGE_LAST_5"] == pytest.approx(0.8)
    assert result.loc["8", "COACH_DNP_LAST_5"] == 1

    # Before game 12:
    # Last 5 games: 7-11, four known and four played.
    # Last 10 games: 2-11, nine known, eight played,
    # and one confirmed coach DNP.
    assert result.loc["12", "PARTICIPATION_KNOWN_LAST_5"] == 4
    assert result.loc["12", "APPEARANCE_RATE_LAST_5"] == 1
    assert result.loc["12", "PARTICIPATION_COVERAGE_LAST_5"] == pytest.approx(0.8)
    assert result.loc["12", "COACH_DNP_LAST_5"] == 0

    assert result.loc["12", "PARTICIPATION_KNOWN_LAST_10"] == 9
    assert result.loc["12", "APPEARANCE_RATE_LAST_10"] == pytest.approx(8 / 9)
    assert result.loc["12", "PARTICIPATION_COVERAGE_LAST_10"] == pytest.approx(0.9)
    assert result.loc["12", "COACH_DNP_LAST_10"] == 1

    # Changing current-game participation must not change
    # current-game features.
    modified = participation.copy()
    modified.loc[
        modified["GAME_ID"] == "12", "PLAYED"
    ] = False
    modified.loc[
        modified["GAME_ID"] == "12", "DID_NOT_PLAY"
    ] = True
    modified.loc[
        modified["GAME_ID"] == "12", "DNP_REASON"
    ] = "COACH'S DECISION"

    modified_result = add_participation_reliability_features(
        players, teams, modified
    ).set_index("GAME_ID")

    feature_cols = [
        f"{prefix}_LAST_{w}"
        for w in (5, 10)
        for prefix in (
            "APPEARANCE_RATE",
            "PARTICIPATION_KNOWN",
            "COACH_DNP",
            "PARTICIPATION_COVERAGE",
        )
    ]

    pd.testing.assert_series_equal(
        result.loc["12", feature_cols],
        modified_result.loc["12", feature_cols],
    )

def test_team_games_since_last_appearance():
    import pandas as pd
    from src.features.rotation import add_participation_reliability_features

    teams = pd.DataFrame([
        {
            "GAME_ID": str(i),
            "GAME_DATE": f"2026-11-{i:02d}",
            "TEAM": "DEN",
        }
        for i in range(1, 7)
    ])

    players = teams.copy()
    players["PLAYER_ID"] = "100"

    # Game 1: played
    # Game 2: confirmed DNP
    # Game 3: unknown
    # Game 4: played
    # Game 5: confirmed DNP
    # Game 6: played
    participation = pd.DataFrame([
        {
            "GAME_ID": str(i),
            "GAME_DATE": f"2026-11-{i:02d}",
            "TEAM": "DEN",
            "PLAYER_ID": "100",
            "PLAYED": played,
            "DID_NOT_PLAY": not played,
            "DNP_REASON": (
                "COACH'S DECISION" if not played else None
            ),
        }
        for i, played in [
            (1, True),
            (2, False),
            (4, True),
            (5, False),
            (6, True),
        ]
    ])

    result = add_participation_reliability_features(
        players, teams, participation
    ).set_index("GAME_ID")

    col = "TEAM_GAMES_SINCE_LAST_APPEARANCE"

    assert pd.isna(result.loc["1", col])
    assert result.loc["2", col] == 0
    assert result.loc["3", col] == 1
    assert result.loc["4", col] == 2
    assert result.loc["5", col] == 0
    assert result.loc["6", col] == 1

    # Calendar days since the last confirmed appearance.
    days_col = "DAYS_SINCE_LAST_APPEARANCE"

    assert pd.isna(result.loc["1", days_col])
    assert result.loc["2", days_col] == 1
    assert result.loc["3", days_col] == 2
    assert result.loc["4", days_col] == 3
    assert result.loc["5", days_col] == 1
    assert result.loc["6", days_col] == 2



    # Changing game 6 must not affect its own pregame gap.
    modified = participation.copy()
    modified.loc[
        modified["GAME_ID"] == "6", "PLAYED"
    ] = False
    modified.loc[
        modified["GAME_ID"] == "6", "DID_NOT_PLAY"
    ] = True

    modified_result = add_participation_reliability_features(
        players, teams, modified
    ).set_index("GAME_ID")

    assert modified_result.loc["6", col] == result.loc["6", col]

    # Current-game participation must not affect
    # current-game calendar-day features.
    assert (
        modified_result.loc["6", days_col]
        == result.loc["6", days_col]
    )

def test_minutes_per_team_game():
    import pandas as pd
    import pytest
    from src.features.rotation import add_participation_reliability_features

    teams = pd.DataFrame([
        {
            "GAME_ID": str(i),
            "GAME_DATE": f"2026-11-{i:02d}",
            "TEAM": "DEN",
        }
        for i in range(1, 13)
    ])

    players = teams.copy()
    players["PLAYER_ID"] = "100"

    # Game 4: confirmed DNP, counted as zero.
    # Game 7: unknown participation, excluded.
    # Game 12: current game, excluded from its own features.
    minutes = {
        1: 30, 2: 28, 3: 32,
        4: None,
        5: 20, 6: 24,
        8: 36, 9: 30, 10: 34, 11: 26,
        12: 40,
    }

    participation = pd.DataFrame([
        {
            "GAME_ID": str(i),
            "GAME_DATE": f"2026-11-{i:02d}",
            "TEAM": "DEN",
            "PLAYER_ID": "100",
            "PLAYED": i != 4,
            "DID_NOT_PLAY": i == 4,
            "DNP_REASON": "COACH'S DECISION" if i == 4 else None,
            "MIN": value,
        }
        for i, value in minutes.items()
    ])

    result = add_participation_reliability_features(
        players, teams, participation
    ).set_index("GAME_ID")

    # Before game 5: 30, 28, 32, 0.
    assert result.loc["5", "MIN_PER_TEAM_GAME_LAST_5"] == pytest.approx(22.5)
    assert result.loc["5", "MINUTES_KNOWN_LAST_5"] == 4

    # Before game 8: games 3-7 = 32, 0, 20, 24, unknown.
    assert result.loc["8", "MIN_PER_TEAM_GAME_LAST_5"] == pytest.approx(19)
    assert result.loc["8", "MINUTES_KNOWN_LAST_5"] == 4

    # Before game 12: games 7-11 = unknown, 36, 30, 34, 26.
    assert result.loc["12", "MIN_PER_TEAM_GAME_LAST_5"] == pytest.approx(31.5)
    assert result.loc["12", "MINUTES_KNOWN_LAST_5"] == 4

    # Before game 12: games 2-11, nine known minutes records.
    assert result.loc["12", "MIN_PER_TEAM_GAME_LAST_10"] == pytest.approx(230 / 9)
    assert result.loc["12", "MINUTES_KNOWN_LAST_10"] == 9

    # Current-game minutes must not affect current-game features.
    modified = participation.copy()
    modified.loc[modified["GAME_ID"] == "12", "MIN"] = 5

    modified_result = add_participation_reliability_features(
        players, teams, modified
    ).set_index("GAME_ID")

    for w in (5, 10):
        for prefix in ("MIN_PER_TEAM_GAME", "MINUTES_KNOWN"):
            col = f"{prefix}_LAST_{w}"
            assert modified_result.loc["12", col] == result.loc["12", col]