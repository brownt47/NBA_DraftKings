
import pandas as pd
import pytest

from src.slates.salaries import (
    load_dk_salaries,
    parse_game_info,
)


def test_parse_home_game():
    result = parse_game_info(
        "BOS@CHA 11/06/2026 07:00PM ET", "CHA"
    )
    assert result["OPPONENT"] == "BOS"
    assert result["HOME_AWAY"] == "HOME"


def test_parse_away_game():
    result = parse_game_info(
        "BOS@CHA 11/06/2026 07:00PM ET", "BOS"
    )
    assert result["OPPONENT"] == "CHA"
    assert result["HOME_AWAY"] == "AWAY"


def test_invalid_matchup():
    with pytest.raises(ValueError):
        parse_game_info(
            "BOS@CHA 11/06/2026 07:00PM ET", "ATL"
        )


def test_load_salary_file(tmp_path):
    path = tmp_path / "DKSalaries.csv"

    pd.DataFrame([{
        "ID": "12345",
        "Name": "Test Player",
        "Position": "PG/SG",
        "Roster Position": "PG/SG/G/UTIL",
        "Salary": 6500,
        "Game Info": "BOS@CHA 11/06/2026 07:00PM ET",
        "TeamAbbrev": "BOS",
        "Status": "Q",
        "AvgPointsPerGame": 32.5,
    }]).to_csv(path, index=False)

    result = load_dk_salaries(path)

    assert len(result) == 1
    assert result.iloc[0]["DK_ID"] == "12345"
    assert result.iloc[0]["OPPONENT"] == "CHA"
    assert result.iloc[0]["Salary"] == 6500
    assert result.iloc[0]["IS_QUESTIONABLE"]
    assert result.iloc[0]["ELIGIBLE_POSITIONS"] == [
        "PG", "SG", "G", "UTIL"
    ]
