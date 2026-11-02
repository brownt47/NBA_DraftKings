import pandas as pd
import pytest

from src.features.usage import add_historical_usage_features


def test_fga_per_36_uses_previous_games_only():
    players = pd.DataFrame({
        "PLAYER_ID": [101, 101, 101, 101],
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-07",
        ],
        "MIN": [24, 36, 18, 30],
        "FGA": [12, 24, 6, 15],
    })

    players['DK_POINTS'] = 0.0
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index("GAME_ID")

    assert pd.isna(result.loc["G1", "FGA_PER_36"])

    assert result.loc["G2", "FGA_PER_36"] == pytest.approx(18)
    assert result.loc["G3", "FGA_PER_36"] == pytest.approx(21)
    assert result.loc["G4", "FGA_PER_36"] == pytest.approx(18)


def test_fga_per_36_rolling_window_and_player_isolation():
    rows = []

    for game_number in range(1, 8):
        game_id = f"G{game_number}"
        date = f"2026-10-{game_number:02d}"

        rows.append({
            "PLAYER_ID": 101,
            "GAME_ID": game_id,
            "GAME_DATE": date,
            "MIN": 36,
            "FGA": game_number * 2,
        })

        rows.append({
            "PLAYER_ID": 202,
            "GAME_ID": game_id,
            "GAME_DATE": date,
            "MIN": 36,
            "FGA": 10,
        })

    # Deliberately scramble the input order.
    players = pd.DataFrame(rows).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    players['DK_POINTS'] = 0.0
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index(["PLAYER_ID", "GAME_ID"])

    # G6 uses G1 through G5: 2, 4, 6, 8, 10.
    assert result.loc[
        (101, "G6"), "FGA_PER_36"
    ] == pytest.approx(6)

    # G7 uses G2 through G6: 4, 6, 8, 10, 12.
    assert result.loc[
        (101, "G7"), "FGA_PER_36"
    ] == pytest.approx(8)

    # Player 202 always attempts 10 shots per 36 minutes.
    assert result.loc[
        (202, "G7"), "FGA_PER_36"
    ] == pytest.approx(10)

    # Neither player's first appearance has prior history.
    assert pd.isna(result.loc[(101, "G1"), "FGA_PER_36"])
    assert pd.isna(result.loc[(202, "G1"), "FGA_PER_36"])


def test_fga_per_36_handles_zero_and_missing_minutes():
    players = pd.DataFrame({
        "PLAYER_ID": [101] * 5,
        "GAME_ID": ["G1", "G2", "G3", "G4", "G5"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-02",
            "2026-10-03",
            "2026-10-04",
            "2026-10-05",
        ],
        "MIN": [24, 0, "--", 36, 30],
        "FGA": [12, 0, 0, 24, 15],
    })

    players['DK_POINTS'] = 0.0
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index("GAME_ID")

    # G1: 12 attempts in 24 minutes = 18 per 36.
    assert result.loc["G2", "FGA_PER_36"] == pytest.approx(18)

    # Zero and missing minutes must not alter the average.
    assert result.loc["G3", "FGA_PER_36"] == pytest.approx(18)
    assert result.loc["G4", "FGA_PER_36"] == pytest.approx(18)

    # G5 uses valid historical rates from G1 and G4.
    # G4: 24 attempts in 36 minutes = 24 per 36.
    assert result.loc["G5", "FGA_PER_36"] == pytest.approx(21)

    assert result["FGA_PER_36"].dropna().apply(
        lambda x: x != float("inf") and x != float("-inf")
    ).all()


def test_fga_per_36_trend_direction_and_rolling_boundary():
    rows = []

    patterns = {
        101: [2, 4, 6, 8, 10, 12, 14],
        202: [14, 12, 10, 8, 6, 4, 2],
        303: [10, 10, 10, 10, 10, 10, 10],
    }

    for player_id, attempts in patterns.items():
        for game_number, fga in enumerate(attempts, start=1):
            rows.append({
                "PLAYER_ID": player_id,
                "GAME_ID": f"G{game_number}",
                "GAME_DATE": f"2026-10-{game_number:02d}",
                "MIN": 36,
                "FGA": fga,
            })

    players = pd.DataFrame(rows)
    players['DK_POINTS'] = 0.0
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index(["PLAYER_ID", "GAME_ID"])

    # G7 uses G2 through G6 for AVG_5.
    # AVG_3 uses G4 through G6.
    assert result.loc[
        (101, "G7"), "FGA_PER_36_TREND"
    ] == pytest.approx(2)

    assert result.loc[
        (202, "G7"), "FGA_PER_36_TREND"
    ] == pytest.approx(-2)

    assert result.loc[
        (303, "G7"), "FGA_PER_36_TREND"
    ] == pytest.approx(0)

    # Three previous games are required for a trend.
    for player_id in patterns:
        assert pd.isna(
            result.loc[(player_id, "G3"), "FGA_PER_36_TREND"]
        )

    # G4 has exactly three previous games.
    assert result.loc[
        (101, "G4"), "FGA_PER_36_TREND"
    ] == pytest.approx(0)


def test_historical_efficiency_slope():
    import numpy as np
    from src.features.usage import _historical_efficiency_slope

    minutes = pd.Series([10, 20, 30, 40, 50])

    # Efficiency = 0.5 + 0.02 * minutes
    positive_efficiency = 0.5 + 0.02 * minutes
    positive_dk = positive_efficiency * minutes

    assert _historical_efficiency_slope(
        minutes, positive_dk
    ) == pytest.approx(0.02)

    # Efficiency = 1.5 - 0.01 * minutes
    negative_efficiency = 1.5 - 0.01 * minutes
    negative_dk = negative_efficiency * minutes

    assert _historical_efficiency_slope(
        minutes, negative_dk
    ) == pytest.approx(-0.01)

    # At least five valid observations are required.
    assert np.isnan(
        _historical_efficiency_slope(
            minutes.iloc[:4],
            positive_dk.iloc[:4],
        )
    )

    # Constant minutes cannot produce a regression slope.
    constant_minutes = pd.Series([30] * 5)
    constant_dk = pd.Series([20, 25, 30, 35, 40])

    assert np.isnan(
        _historical_efficiency_slope(
            constant_minutes, constant_dk
        )
    )


def test_dk_efficiency_slope_rolling_ten_games_and_no_leakage():
    rows = []

    for player_id, slope in [(101, 0.02), (202, -0.01)]:
        for game_number in range(1, 13):
            minutes = 10 + game_number * 2
            efficiency = 1.0 + slope * minutes

            # Player 101 has an unusual first game.
            # It must be excluded from G12's rolling window.
            if player_id == 101 and game_number == 1:
                efficiency = 5.0

            # G12 has an extreme current-game result.
            # It must not influence its own pregame feature.
            if game_number == 12:
                efficiency = 10.0

            rows.append({
                "PLAYER_ID": player_id,
                "GAME_ID": f"G{game_number}",
                "GAME_DATE": f"2026-10-{game_number:02d}",
                "MIN": minutes,
                "FGA": 10,
                "DK_POINTS": efficiency * minutes,
            })

    players = pd.DataFrame(rows)
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index(["PLAYER_ID", "GAME_ID"])

    # G5 has only four previous games.
    assert pd.isna(
        result.loc[(101, "G5"), "DK_PER_MIN_BY_MINUTES"]
    )

    # G12 uses G2 through G11, excluding unusual G1 and current G12.
    assert result.loc[
        (101, "G12"), "DK_PER_MIN_BY_MINUTES"
    ] == pytest.approx(0.02)

    assert result.loc[
        (202, "G12"), "DK_PER_MIN_BY_MINUTES"
    ] == pytest.approx(-0.01)

    # Changing current-game DK points must not change the feature.
    modified = players.copy()
    modified.loc[
        (modified["PLAYER_ID"] == 101)
        & (modified["GAME_ID"] == "G12"),
        "DK_POINTS"
    ] = 99999

    modified_result = add_historical_usage_features(modified)
    modified_result = modified_result.set_index(
        ["PLAYER_ID", "GAME_ID"]
    )

    assert modified_result.loc[
        (101, "G12"), "DK_PER_MIN_BY_MINUTES"
    ] == pytest.approx(
        result.loc[(101, "G12"), "DK_PER_MIN_BY_MINUTES"]
    )


def test_dk_efficiency_slope_ignores_invalid_minutes():
    rows = []

    for game_number in range(1, 9):
        minutes = 10 + game_number * 4
        efficiency = 1.0 + 0.02 * minutes

        if game_number == 3:
            minutes = 0
            dk_points = 0
        elif game_number == 5:
            minutes = "--"
            dk_points = 0
        else:
            dk_points = efficiency * minutes

        rows.append({
            "PLAYER_ID": 101,
            "GAME_ID": f"G{game_number}",
            "GAME_DATE": f"2026-10-{game_number:02d}",
            "MIN": minutes,
            "FGA": 10,
            "DK_POINTS": dk_points,
        })

    players = pd.DataFrame(rows)
    players['FTA'] = 0
    players['TOV'] = 0
    result = add_historical_usage_features(players)
    result = result.set_index("GAME_ID")

    # Before G7, only four valid observations exist.
    assert pd.isna(
        result.loc["G7", "DK_PER_MIN_BY_MINUTES"]
    )

    # Before G8, five valid observations exist.
    # Invalid games G3 and G5 must be excluded.
    assert result.loc[
        "G8", "DK_PER_MIN_BY_MINUTES"
    ] == pytest.approx(0.02)


def test_player_usage_proxy_uses_previous_games_only():
    players = pd.DataFrame({
        "PLAYER_ID": [101] * 4,
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-02",
            "2026-10-03",
            "2026-10-04",
        ],
        "MIN": [36, 36, 36, 36],
        "FGA": [10, 15, 20, 100],
        "FTA": [5, 10, 5, 20],
        "TOV": [2, 3, 4, 10],
        "DK_POINTS": [20, 30, 40, 100],
    })

    result = add_historical_usage_features(players)
    result = result.set_index("GAME_ID")

    assert pd.isna(result.loc["G1", "PLAYER_USAGE_PROXY"])
    assert result.loc["G2", "PLAYER_USAGE_PROXY"] == pytest.approx(14.2)
    assert result.loc["G3", "PLAYER_USAGE_PROXY"] == pytest.approx(18.3)
    assert result.loc["G4", "PLAYER_USAGE_PROXY"] == pytest.approx(
        (14.2 + 22.4 + 26.2) / 3
    )

    # Changing G4's current-game FGA cannot affect G4's feature.
    modified = players.copy()
    modified.loc[
        modified["GAME_ID"] == "G4", "FGA"
    ] = 9999

    modified_result = add_historical_usage_features(modified)
    modified_result = modified_result.set_index("GAME_ID")

    assert modified_result.loc[
        "G4", "PLAYER_USAGE_PROXY"
    ] == pytest.approx(
        result.loc["G4", "PLAYER_USAGE_PROXY"]
    )


def test_player_usage_proxy_rolling_five_and_player_isolation():
    rows = []

    for player_id in [101, 202]:
        for game_number in range(1, 8):
            if player_id == 101:
                fga = 100 if game_number == 1 else 10
            else:
                fga = 20

            rows.append({
                "PLAYER_ID": player_id,
                "GAME_ID": f"G{game_number}",
                "GAME_DATE": f"2026-10-{game_number:02d}",
                "MIN": 36,
                "FGA": fga,
                "FTA": 0,
                "TOV": 0,
                "DK_POINTS": 20,
            })

    players = pd.DataFrame(rows).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    result = add_historical_usage_features(players)
    result = result.set_index(["PLAYER_ID", "GAME_ID"])

    # G6 uses G1-G5, including the unusually high G1.
    assert result.loc[
        (101, "G6"), "PLAYER_USAGE_PROXY"
    ] == pytest.approx(28.0)

    # G7 uses G2-G6, so G1 must drop out.
    assert result.loc[
        (101, "G7"), "PLAYER_USAGE_PROXY"
    ] == pytest.approx(10.0)

    # Player 202's history must remain independent.
    assert result.loc[
        (202, "G7"), "PLAYER_USAGE_PROXY"
    ] == pytest.approx(20.0)
