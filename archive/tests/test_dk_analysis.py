
import sys
import importlib
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

import data_processing.slate as slate


# ---------------------------------------------------------
# COMPLETE ANALYSIS PIPELINE TEST
# Verify the analysis runs and exports a CSV.
# ---------------------------------------------------------

def test_analysis_pipeline(tmp_path, monkeypatch):

    slate_date = pd.Timestamp("2026-10-31")
    salary_file = tmp_path / "DKSalaries_2026_10_31_Main.csv"

    # Prevent real salary-file selection during import
    monkeypatch.setattr(
        slate,
        "select_salary_file",
        lambda folder: (salary_file, slate_date)
    )

    # Import analysis module after patching selection
    sys.modules.pop("dk_analysis", None)
    analysis = importlib.import_module("dk_analysis")


    # Sample DraftKings players
    dk = pd.DataFrame({
        "ID": ["101", "102"],
        "Name": ["Player A", "Player B"],
        "Salary": [8000, 6000],
        "TeamAbbrev": ["ATL", "BOS"],
        "Game Info": [
            "ATL@BOS 07:00PM ET",
            "ATL@BOS 07:00PM ET",
        ],
    })

    mapping = pd.DataFrame({
        "DK_ID": ["101", "102"],
        "ESPN_PLAYER_ID": ["9001", "9002"],
    })


    # Historical player games
    games = pd.DataFrame({
        "PLAYER_ID": ["9001", "9001", "9001"],
        "GAME_ID": ["G1", "G2", "G3"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-25",
            "2026-10-27",
            "2026-10-29",
        ]),
        "MIN": [30, 32, 34],
        "PTS": [20, 25, 30],
        "REB": [5, 6, 7],
        "AST": [4, 5, 6],
        "DK_POINTS": [35, 45, 55],
    })

    # Historical team games
    team_games = pd.DataFrame({
        "TEAM": ["ATL", "BOS"],
        "GAME_ID": ["T1", "T2"],
        "GAME_DATE": pd.to_datetime([
            "2026-10-28",
            "2026-10-28",
        ]),
        "PACE_EST": [100, 102],
        "OFF_RATING_EST": [110, 112],
        "DEF_RATING_EST": [108, 109],
    })

    # Replace file loading with test data
    monkeypatch.setattr(
        analysis,
        "load_slate_data",
        lambda *args: (dk, mapping)
    )
    monkeypatch.setattr(
        analysis,
        "load_player_games",
        lambda *args: games
    )
    monkeypatch.setattr(
        analysis,
        "load_team_games",
        lambda *args: team_games
    )

    # Write output into temporary directory
    monkeypatch.setattr(analysis, "DATA_FOLDER", tmp_path)

    analysis.main()

    output = (
        tmp_path / "dk_player_analysis_2026_10_31_Main.csv"
    )

    
    assert output.exists()

    result = pd.read_csv(output)

    # Both DraftKings players remain
    assert len(result) == 2
    assert set(result["Name"]) == {"Player A", "Player B"}

    player_a = result.loc[
        result["Name"] == "Player A"
    ].iloc[0]

    player_b = result.loc[
        result["Name"] == "Player B"
    ].iloc[0]

    # Player A has calculated statistics
    assert player_a["AVG_DK_POINTS"] == 45
    assert player_a["VALUE_PER_1000"] == pytest.approx(45 / 8)

    # Player B remains without historical statistics
    assert pd.isna(player_b["GAMES_PLAYED"])
    assert pd.isna(player_b["AVG_DK_POINTS"])
    assert pd.isna(player_b["VALUE_PER_1000"])

    # Players remain sorted by descending salary
    assert result["Salary"].tolist() == [8000, 6000]