
import pandas as pd
import pytest

from src.optimizer.classic import optimize_classic


def make_slate():
    """Create a valid eight-player lineup plus one cheap alternative."""
    return pd.DataFrame([
        {
            "DK_ID": i,
            "DK_NAME": f"Player {i}",
            "SALARY": salary,
            "ELIGIBLE_POSITIONS": [position, "UTIL"],
            "PROJECTED_DK_POINTS": 20.0,
            "OPTIMIZER_ELIGIBLE": True,
        }
        for i, (position, salary) in enumerate([
            ("PG", 6000),
            ("SG", 6000),
            ("SF", 6000),
            ("PF", 6000),
            ("C", 6000),
            ("G", 6000),
            ("F", 6000),
            ("UTIL", 6000),
            ("UTIL", 3000),
        ], start=1)
    ])


def test_salary_floor_excludes_cheap_players():
    slate = make_slate()

    result = optimize_classic(slate, min_salary=4000)

    assert len(result["lineup"]) == 8
    assert result["lineup"]["SALARY"].min() >= 4000
    assert "Player 9" not in set(result["lineup"]["DK_NAME"])


def test_locked_player_below_floor_is_rejected():
    slate = make_slate()

    with pytest.raises(ValueError, match="Locked players unavailable"):
        optimize_classic(
            slate,
            min_salary=4000,
            locked_players=["Player 9"],
        )


def test_default_salary_floor_allows_3000_player():
    slate = make_slate()

    # Make the inexpensive player more valuable than alternatives.
    slate.loc[
        slate["DK_NAME"] == "Player 9",
        "PROJECTED_DK_POINTS",
    ] = 50.0

    result = optimize_classic(slate)

    assert "Player 9" in set(result["lineup"]["DK_NAME"])


def test_optimizer_requires_all_eight_roster_slots():
    """An unaffordable center must make the slate infeasible."""

    slate = make_slate()

    # Make the only center too expensive to fit
    # alongside the other seven required players.
    slate.loc[
        slate["DK_NAME"] == "Player 5",
        "SALARY",
    ] = 15000

    with pytest.raises(
        RuntimeError,
        match="Optimizer did not confirm an optimal lineup",
    ):
        optimize_classic(
            slate,
            min_salary=4000,
            salary_cap=50000,
        )