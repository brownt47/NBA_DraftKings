
"""DraftKings NBA Classic lineup optimizer."""

import ast

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROSTER_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
SALARY_CAP = 50000


def _parse_positions(value):
    """Accept position lists or their string representations."""
    if isinstance(value, (list, tuple, set)):
        return list(value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass

        return [part.strip() for part in value.split(",")]

    return []


def optimize_classic(
    projection_slate,
    salary_cap=SALARY_CAP,
    locked_players=None,
    previous_lineups=None,
    max_player_overlap=6,
):
    """Return the highest-projected valid NBA Classic lineup."""

    required = [
        "DK_ID",
        "DK_NAME",
        "SALARY",
        "ELIGIBLE_POSITIONS",
        "PROJECTED_DK_POINTS",
        "OPTIMIZER_ELIGIBLE",
    ]

    missing = [
        col for col in required
        if col not in projection_slate.columns
    ]

    if missing:
        raise ValueError(f"Missing optimizer columns: {missing}")

    locked_players = set(locked_players or [])

    previous_lineups = previous_lineups or []

    if not 0 <= max_player_overlap <= 8:
        raise ValueError("max_player_overlap must be between 0 and 8.")

    if previous_lineups and len(locked_players) > max_player_overlap:
        raise ValueError(
            "Number of locked players exceeds maximum allowed overlap."
        )

    if len(locked_players) > 8:
        raise ValueError("Cannot lock more than 8 players.")

    if locked_players:
        available = set(
            projection_slate.loc[
                projection_slate["OPTIMIZER_ELIGIBLE"].eq(True),
                "DK_NAME",
            ]
        )

        unavailable = locked_players - available

        if unavailable:
            raise ValueError(
                f"Locked players unavailable: {sorted(unavailable)}"
            )

    players = projection_slate.loc[
        projection_slate["OPTIMIZER_ELIGIBLE"].eq(True)
    ].copy()

    players["SALARY"] = pd.to_numeric(
        players["SALARY"], errors="coerce"
    )
    players["PROJECTED_DK_POINTS"] = pd.to_numeric(
        players["PROJECTED_DK_POINTS"], errors="coerce"
    )

    # Use strategy-adjusted scores when available.
    # Otherwise, use the original LightGBM projections.
    if "OPTIMIZATION_SCORE" not in players.columns:
        players["OPTIMIZATION_SCORE"] = (
            players["PROJECTED_DK_POINTS"]
        )

    players["OPTIMIZATION_SCORE"] = pd.to_numeric(
        players["OPTIMIZATION_SCORE"], errors="coerce"
    )

    players = players.dropna(
        subset=[
            "DK_ID",
            "SALARY",
            "PROJECTED_DK_POINTS",
            "OPTIMIZATION_SCORE",
        ]
    ).reset_index(drop=True)

    if players["DK_ID"].duplicated().any():
        raise ValueError("Duplicate DraftKings player IDs.")

    if (players["SALARY"] <= 0).any():
        raise ValueError("Invalid player salary.")

    players["ELIGIBLE_POSITIONS"] = (
        players["ELIGIBLE_POSITIONS"].apply(_parse_positions)
    )

    # One binary variable for each eligible player-slot pairing.
    assignments = [
        (i, slot)
        for i, positions in enumerate(players["ELIGIBLE_POSITIONS"])
        for slot in ROSTER_SLOTS
        if slot in positions
    ]

    n = len(assignments)

    if n == 0:
        raise ValueError("No eligible player-position assignments.")

    # Constraints:
    # 8 roster slots filled exactly once.
    # Each player selected at most once.
    # Total salary <= salary cap.
    rows = (
        len(ROSTER_SLOTS)
        + len(players)
        + 1
        + len(previous_lineups)
    )
    matrix = lil_matrix((rows, n), dtype=float)

    lower = np.zeros(rows)
    upper = np.ones(rows)

    salary_row = len(ROSTER_SLOTS) + len(players)
    lower[salary_row] = 0
    upper[salary_row] = salary_cap

    # Require each locked player to occupy exactly one roster slot.
    for player_idx, player_name in enumerate(players["DK_NAME"]):
        if player_name in locked_players:
            player_row = len(ROSTER_SLOTS) + player_idx
            lower[player_row] = 1
            upper[player_row] = 1

    costs = np.zeros(n)

    for j, (player_idx, slot) in enumerate(assignments):
        slot_idx = ROSTER_SLOTS.index(slot)

        matrix[slot_idx, j] = 1
        matrix[len(ROSTER_SLOTS) + player_idx, j] = 1
        matrix[salary_row, j] = players.at[player_idx, "SALARY"]

        # SciPy minimizes, so negate the optimization score.
        costs[j] = -players.at[
            player_idx, "OPTIMIZATION_SCORE"
        ]

    # Limit overlap with every previously generated lineup.
    for lineup_idx, previous_ids in enumerate(previous_lineups):
        overlap_row = salary_row + 1 + lineup_idx

        previous_ids = set(previous_ids)

        for j, (player_idx, slot) in enumerate(assignments):
            if players.at[player_idx, "DK_ID"] in previous_ids:
                matrix[overlap_row, j] = 1

        lower[overlap_row] = 0
        upper[overlap_row] = max_player_overlap

    result = milp(
        c=costs,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            matrix.tocsr(), lower, upper
        ),
        options={"time_limit": 30, "mip_rel_gap": 0.0001},
    )

    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"Optimizer did not confirm an optimal lineup: "
            f"{result.message}"
        )

    selected = []

    for j, value in enumerate(result.x):
        if value > 0.5:
            player_idx, slot = assignments[j]

            row = players.iloc[player_idx].copy()
            row["ROSTER_SLOT"] = slot
            selected.append(row)

    if len(selected) != 8:
        raise RuntimeError("Optimizer returned an invalid roster size.")

    lineup = pd.DataFrame(selected)

    lineup["ROSTER_SLOT"] = pd.Categorical(
        lineup["ROSTER_SLOT"],
        categories=ROSTER_SLOTS,
        ordered=True,
    )

    lineup = lineup.sort_values("ROSTER_SLOT")

    total_salary = lineup["SALARY"].sum()
    total_points = lineup["PROJECTED_DK_POINTS"].sum()
    total_optimization_score = lineup["OPTIMIZATION_SCORE"].sum()

    if total_salary > salary_cap:
        raise RuntimeError("Salary cap exceeded.")

    if lineup["DK_ID"].duplicated().any():
        raise RuntimeError("Duplicate player in lineup.")

    missing_locks = locked_players - set(lineup["DK_NAME"])

    if missing_locks:
        raise RuntimeError(
            f"Locked players missing from lineup: {sorted(missing_locks)}"
        )

    return {
        "lineup": lineup,
        "total_salary": int(total_salary),
        "remaining_salary": int(salary_cap - total_salary),
        "projected_points": float(total_points),
        "optimization_score": float(total_optimization_score),
    }
