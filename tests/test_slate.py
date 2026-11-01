
import sys
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

from data_processing.slate import (
    merge_player_mapping,
    build_output_path,
    select_salary_file,
)


# ---------------------------------------------------------
# PLAYER MAPPING TEST
# Verify DraftKings-to-ESPN player ID matching.
# ---------------------------------------------------------

def test_merge_player_mapping():

    dk = pd.DataFrame({
        "ID": ["101", "102", "103"],
        "Name": ["Player A", "Player B", "Player C"],
        "Salary": [10000, 8000, 6000],
    })

    mapping = pd.DataFrame({
        "DK_ID": ["101", "102"],
        "ESPN_PLAYER_ID": ["9001", "9002"],
    })

    result = merge_player_mapping(dk, mapping)

    # All DraftKings players should be preserved
    assert len(result) == 3

    # Correct ESPN IDs assigned
    assert result.loc[0, "ESPN_PLAYER_ID"] == "9001"
    assert result.loc[1, "ESPN_PLAYER_ID"] == "9002"

    # Unmapped player remains in the slate
    assert pd.isna(result.loc[2, "ESPN_PLAYER_ID"])

    # Salary information remains unchanged
    assert result["Salary"].tolist() == [
        10000, 8000, 6000
    ]


# ---------------------------------------------------------
# DUPLICATE PLAYER MAPPING TEST
# Verify duplicate DraftKings IDs are rejected.
# ---------------------------------------------------------

def test_duplicate_player_mapping():

    dk = pd.DataFrame({
        "ID": ["101", "102"],
        "Name": ["Player A", "Player B"],
        "Salary": [10000, 8000],
    })

    mapping = pd.DataFrame({
        "DK_ID": ["101", "101", "102"],
        "ESPN_PLAYER_ID": ["9001", "9999", "9002"],
    })

    # Duplicate DK_ID values should cause a merge error
    with pytest.raises(pd.errors.MergeError):
        merge_player_mapping(dk, mapping)


# ---------------------------------------------------------
# OUTPUT PATH TEST
# Verify filenames for Early, Main, and Late slates.
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "slate",
    ["Early", "Main", "Late"]
)
def test_build_output_path(slate):

    data_folder = Path("data")
    slate_date = pd.Timestamp("2026-10-31")

    dk_file = Path(
        f"DKSalaries_2026_10_31_{slate}.csv"
    )

    result = build_output_path(
        data_folder,
        slate_date,
        dk_file
    )

    expected = data_folder / (
        f"dk_player_analysis_2026_10_31_{slate}.csv"
    )

    assert result == expected


# ---------------------------------------------------------
# DEFAULT SALARY FILE SELECTION TEST
# Verify the latest Main slate is selected.
# ---------------------------------------------------------

def test_select_salary_file_default(tmp_path, monkeypatch):

    filenames = [
        "DKSalaries_2026_10_28_Main.csv",
        "DKSalaries_2026_10_30_Main.csv",
        "DKSalaries_2026_10_31_Main.csv",
        "DKSalaries_2026_11_01_Early.csv",
        "DKSalaries_invalid.csv",
    ]

    # Create temporary salary files
    for filename in filenames:
        (tmp_path / filename).touch()

    # Simulate running without command-line arguments
    monkeypatch.setattr(sys, "argv", ["dk_analysis.py"])

    selected_file, slate_date = select_salary_file(
        tmp_path
    )

    # Latest Main slate should be selected
    assert selected_file.name == (
        "DKSalaries_2026_10_31_Main.csv"
    )

    assert slate_date == pd.Timestamp("2026-10-31")


# ---------------------------------------------------------
# EXPLICIT SALARY FILE SELECTION TEST
# Verify --date and --slate arguments are respected.
# ---------------------------------------------------------

def test_select_salary_file_explicit(tmp_path, monkeypatch):

    filenames = [
        "DKSalaries_2026_10_30_Main.csv",
        "DKSalaries_2026_10_31_Early.csv",
        "DKSalaries_2026_10_31_Main.csv",
        "DKSalaries_2026_10_31_Late.csv",
        "DKSalaries_2026_11_01_Main.csv",
    ]

    for filename in filenames:
        (tmp_path / filename).touch()

    # Simulate command-line arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dk_analysis.py",
            "--date", "2026-10-31",
            "--slate", "Late",
        ]
    )

    selected_file, slate_date = select_salary_file(
        tmp_path
    )

    assert selected_file.name == (
        "DKSalaries_2026_10_31_Late.csv"
    )

    assert slate_date == pd.Timestamp("2026-10-31")


# ---------------------------------------------------------
# MISSING SALARY FILE TEST
# Verify an error is raised when no matching file exists.
# ---------------------------------------------------------

def test_select_salary_file_missing(tmp_path, monkeypatch):

    # Create a Main slate, but no Late slate
    (tmp_path / "DKSalaries_2026_10_31_Main.csv").touch()

    # Request a slate that does not exist
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dk_analysis.py",
            "--date", "2026-10-31",
            "--slate", "Late",
        ]
    )

    with pytest.raises(
        FileNotFoundError,
        match="No matching DraftKings salary file found"
    ):
        select_salary_file(tmp_path)
