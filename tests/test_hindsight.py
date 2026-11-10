
"""Tests for the independent hindsight optimizer."""

import pandas as pd
import pytest

from src.hindsight.actuals import prepare_actuals
from src.hindsight.lineup import build_perfect_lineup, save_lineup


def test_prepare_actuals_november_8():
    players, report = prepare_actuals("2026-11-08")

    assert report["salary_players"] == 176
    assert report["mapped_players"] == 176
    assert report["players_with_results"] == 123
    assert report["players_without_results"] == 53

    assert len(players) == 123
    assert players["DK_POINTS"].notna().all()
    assert players["OPTIMIZER_ELIGIBLE"].all()


def test_hindsight_lineup_november_8():
    result, report = build_perfect_lineup("2026-11-08")

    lineup = result["lineup"]

    assert len(lineup) == 8
    assert lineup["DK_ID"].is_unique
    assert result["total_salary"] <= 50000
    assert result["optimization_score"] == pytest.approx(345.25)
    assert report["date"] == "2026-11-08"


def test_save_hindsight_lineup(tmp_path):
    result, report = build_perfect_lineup("2026-11-08")

    lineup_file, summary_file = save_lineup(
        result, report, output_dir=tmp_path
    )

    assert lineup_file.exists()
    assert summary_file.exists()

    saved_lineup = pd.read_csv(lineup_file)
    saved_summary = pd.read_csv(summary_file)

    assert len(saved_lineup) == 8
    assert saved_summary.loc[0, "actual_dk_points"] == pytest.approx(345.25)
    assert not bool(saved_summary.loc[0, "certified_complete"])


def test_missing_historical_salary_file():
    with pytest.raises(FileNotFoundError):
        prepare_actuals("2020-01-01")
