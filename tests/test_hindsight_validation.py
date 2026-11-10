
"""Tests for hindsight slate validation."""

import pandas as pd
import pytest

from src.hindsight.validation import validate_slate


def test_november_8_validation():
    report = validate_slate("2026-11-08")

    assert len(report) == 5
    assert (report["STATUS"] == "PASS").all()
    assert report["PLAYERS"].sum() == 123


def test_missing_salary_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_slate(
            "2026-11-08",
            salary_dir=tmp_path,
        )


def test_missing_espn_data(tmp_path):
    empty_games = tmp_path / "games.csv"

    pd.DataFrame(columns=[
        "GAME_ID", "GAME_DATE", "MATCHUP",
        "TEAM", "PLAYER_ID", "MIN", "DK_POINTS",
    ]).to_csv(empty_games, index=False)

    report = validate_slate(
        "2026-11-08",
        games_file=empty_games,
    )

    assert (report["STATUS"] == "FAIL").all()


def test_missing_player_minutes(tmp_path):
    source = pd.read_csv("data/nba_player_games_2026_27.csv")

    target = (
        source["GAME_DATE"].eq("2026-11-08")
        & source["TEAM"].eq("SAC")
    )

    source.loc[target, "MIN"] = None

    modified_file = tmp_path / "games.csv"
    source.to_csv(modified_file, index=False)

    report = validate_slate(
        "2026-11-08",
        games_file=modified_file,
    )

    sac_game = report.loc[
        report["MATCHUP"].eq("MIA@SAC")
    ]

    assert sac_game.iloc[0]["STATUS"] == "FAIL"


def test_overtime_minutes_warning(tmp_path):
    source = pd.read_csv("data/nba_player_games_2026_27.csv")

    target = (
        source["GAME_DATE"].eq("2026-11-08")
        & source["TEAM"].eq("SAC")
    )

    source.loc[target, "MIN"] = 40

    modified_file = tmp_path / "games.csv"
    source.to_csv(modified_file, index=False)

    report = validate_slate(
        "2026-11-08",
        games_file=modified_file,
    )

    sac_game = report.loc[
        report["MATCHUP"].eq("MIA@SAC")
    ]

    assert sac_game.iloc[0]["STATUS"] == "WARNING"
