
"""Tests for historical DraftKings lineup comparisons."""

import numpy as np
import pandas as pd
import pytest

from src.hindsight import comparison


DATE = "2026-11-06"


def make_fixture(tmp_path, monkeypatch, *, dnp=False, missing=False):
    """Build a synthetic eight-player lineup and actual results."""
    lineup_dir = tmp_path / "lineups"
    hindsight_dir = tmp_path / "hindsight"
    lineup_dir.mkdir()
    hindsight_dir.mkdir()

    players = pd.DataFrame({
        "LINEUP_NUMBER": [1] * 8,
        "DK_ID": list(range(101, 109)),
        "DK_NAME": [f"Player {i}" for i in range(1, 9)],
        "SALARY": [6000] * 8,
        "PROJECTED_DK_POINTS": [20.0] * 8,
    })

    players.to_csv(
        lineup_dir / "lineups_2026_11_06_Main_cash_details.csv",
        index=False,
    )

    # Player 108 has no recorded box score.
    actuals = pd.DataFrame({
        "DK_ID": list(range(101, 108)),
        "DK_POINTS": [25.0] * 7,
    })

    monkeypatch.setattr(
        comparison,
        "validate_slate",
        lambda date: pd.DataFrame({"STATUS": ["PASS"]}),
    )
    monkeypatch.setattr(
        comparison,
        "prepare_actuals",
        lambda date: (actuals, {}),
    )

    pd.DataFrame({
        "actual_dk_points": [300.0],
    }).to_csv(
        hindsight_dir / "perfect_summary_2026_11_06_Main.csv",
        index=False,
    )

    if dnp:
        pd.DataFrame({
            "GAME_DATE": [DATE],
            "DK_ID": [108],
            "DK_NAME": ["Player 8"],
            "RESULT_STATUS": ["CONFIRMED_DNP"],
            "REASON": ["Test injury"],
            "SOURCE": ["Test fixture"],
        }).to_csv(
            hindsight_dir / "player_status_overrides.csv",
            index=False,
        )

    if missing:
        # An override for another date must not resolve Player 108.
        pd.DataFrame({
            "GAME_DATE": ["2026-11-05"],
            "DK_ID": [108],
            "RESULT_STATUS": ["CONFIRMED_DNP"],
        }).to_csv(
            hindsight_dir / "player_status_overrides.csv",
            index=False,
        )

    return lineup_dir, hindsight_dir


def test_confirmed_dnp_counts_as_zero(tmp_path, monkeypatch):
    lineup_dir, hindsight_dir = make_fixture(
        tmp_path, monkeypatch, dnp=True
    )

    summary, details = comparison.compare_lineups(
        DATE, lineup_dir, hindsight_dir
    )

    row = summary.iloc[0]
    assert bool(row["COMPLETE"])
    assert row["MISSING_RESULTS"] == 0
    assert row["DNP_COUNT"] == 1
    assert row["DNP_PROJECTED_POINTS"] == 20.0
    assert row["ACTUAL_POINTS"] == 175.0
    assert row["PROJECTION_ERROR"] == -15.0
    assert row["POINTS_BELOW_PERFECT"] == 125.0

    dnp_player = details.loc[details["DK_ID"] == 108].iloc[0]
    assert dnp_player["RESULT_STATUS"] == "CONFIRMED_DNP"
    assert dnp_player["DK_POINTS"] == 0.0


def test_missing_result_remains_unknown(tmp_path, monkeypatch):
    lineup_dir, hindsight_dir = make_fixture(
        tmp_path, monkeypatch, missing=True
    )

    summary, details = comparison.compare_lineups(
        DATE, lineup_dir, hindsight_dir
    )

    row = summary.iloc[0]
    assert not bool(row["COMPLETE"])
    assert row["MISSING_RESULTS"] == 1
    assert row["DNP_COUNT"] == 0
    assert row["DNP_PROJECTED_POINTS"] == 0.0
    assert np.isnan(row["ACTUAL_POINTS"])

    unknown = details.loc[details["DK_ID"] == 108].iloc[0]
    assert unknown["RESULT_STATUS"] == "UNKNOWN"
    assert np.isnan(unknown["DK_POINTS"])


def test_failed_slate_validation_stops_comparison(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        comparison,
        "validate_slate",
        lambda date: pd.DataFrame({"STATUS": ["FAIL"]}),
    )

    with pytest.raises(ValueError, match="validation did not pass"):
        comparison.compare_lineups(
            DATE, tmp_path, tmp_path
        )


def test_missing_strategy_files_raise_error(
    tmp_path, monkeypatch
):
    lineup_dir, hindsight_dir = make_fixture(
        tmp_path, monkeypatch
    )

    for file in lineup_dir.glob("*.csv"):
        file.unlink()

    with pytest.raises(FileNotFoundError, match="No saved strategy"):
        comparison.compare_lineups(
            DATE, lineup_dir, hindsight_dir
        )
