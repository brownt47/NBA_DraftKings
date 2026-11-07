
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

from validation import validate_player_features


# ---------------------------------------------------------
# VALID PLAYER FEATURES TEST
# Verify player counts and feature completeness reporting.
# ---------------------------------------------------------

def test_validate_player_features(capsys):

    players = pd.DataFrame({
        "ID": ["101", "102", "103"],
        "GAMES_PLAYED": [10, 5, None],
        "AVG_DK_POINTS": [40.0, 25.0, None],
        "VALUE_PER_1000": [5.0, 4.2, None],
    })

    validate_player_features(players)

    # Capture printed validation results
    output = capsys.readouterr().out

    # Player counts
    assert "Total players: 3" in output
    assert "Players with history: 2" in output
    assert "Players without history: 1" in output

    # Historical players have complete features
    assert "Missing AVG_DK_POINTS: 0" in output
    assert "Missing VALUE_PER_1000: 0" in output


# ---------------------------------------------------------
# MISSING PLAYER FEATURES TEST
# Verify missing statistics are reported only for
# players with historical games.
# ---------------------------------------------------------

def test_validate_missing_player_features(capsys):

    players = pd.DataFrame({
        "ID": ["101", "102", "103", "104"],
        "GAMES_PLAYED": [10, 5, 3, None],
        "AVG_DK_POINTS": [40.0, None, 25.0, None],
        "VALUE_PER_1000": [5.0, 4.2, None, None],
    })

    validate_player_features(players)

    output = capsys.readouterr().out

    # Player counts
    assert "Total players: 4" in output
    assert "Players with history: 3" in output
    assert "Players without history: 1" in output

    # Missing features among historical players only
    assert "Missing AVG_DK_POINTS: 1" in output
    assert "Missing VALUE_PER_1000: 1" in output
