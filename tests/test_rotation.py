import pandas as pd
import pytest

from src.features.rotation import add_historical_rotation_features


def test_rotation_concentration_uses_prior_games():
    minutes = {
        "G1": [30, 30, 30, 30],
        "G2": [60, 30, 20, 10],
        "G3": [40, 40, 20, 20],
    }

    rows = []
    for game_id, values in minutes.items():
        for player_id, value in enumerate(values, start=1):
            rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": {
                    "G1": "2026-10-01",
                    "G2": "2026-10-03",
                    "G3": "2026-10-05",
                }[game_id],
                "TEAM": "ATL",
                "PLAYER_ID": player_id,
                "MIN": value,
            })

    players = pd.DataFrame(rows)

    teams = players[
        ["GAME_ID", "GAME_DATE", "TEAM"]
    ].drop_duplicates()

    result = add_historical_rotation_features(players, teams)

    by_game = result.groupby("GAME_ID").first()

    hhi_g1 = 0.25
    hhi_g2 = (60**2 + 30**2 + 20**2 + 10**2) / 120**2

    assert pd.isna(
        result.loc[
            result["GAME_ID"] == "G1",
            "TEAM_ROTATION_CONCENTRATION"
        ]
    ).all()

    assert by_game.loc["G2", "TEAM_ROTATION_CONCENTRATION"] == pytest.approx(hhi_g1)
    assert by_game.loc["G3", "TEAM_ROTATION_CONCENTRATION"] == pytest.approx(hhi_g2)

    assert by_game.loc["G3", "TEAM_ROTATION_CONCENTRATION_AVG_5"] == pytest.approx(
        (hhi_g1 + hhi_g2) / 2
    )


def test_rotation_concentration_separates_teams_and_sorts_games():
    games = [
        ("B2", "2026-10-04", "BOS", [40, 20, 20, 20]),
        ("A2", "2026-10-03", "ATL", [60, 20, 10, 10]),
        ("B1", "2026-10-02", "BOS", [25, 25, 25, 25]),
        ("A1", "2026-10-01", "ATL", [50, 30, 10, 10]),
    ]

    rows = []
    for game_id, date, team, minutes in games:
        for player_id, value in enumerate(minutes, start=1):
            rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": date,
                "TEAM": team,
                "PLAYER_ID": f"{team}_{player_id}",
                "MIN": value,
            })

    players = pd.DataFrame(rows)
    teams = players[
        ["GAME_ID", "GAME_DATE", "TEAM"]
    ].drop_duplicates()

    result = add_historical_rotation_features(players, teams)
    by_game = result.groupby("GAME_ID").first()

    atl_hhi = (50**2 + 30**2 + 10**2 + 10**2) / 100**2
    bos_hhi = 4 * (25 / 100) ** 2

    assert by_game.loc["A2", "TEAM_ROTATION_CONCENTRATION"] == pytest.approx(
        atl_hhi
    )
    assert by_game.loc["B2", "TEAM_ROTATION_CONCENTRATION"] == pytest.approx(
        bos_hhi
    )

    assert result.loc[
        result["GAME_ID"].isin(["A1", "B1"]),
        "TEAM_ROTATION_CONCENTRATION"
    ].isna().all()


def test_rotation_concentration_rolling_five_game_boundary():
    rows = []
    expected_hhi = []

    for game_number in range(1, 8):
        game_id = f"G{game_number}"
        date = f"2026-10-{game_number:02d}"

        minutes = [10 * game_number, 20, 20, 20]
        total = sum(minutes)
        expected_hhi.append(sum(m**2 for m in minutes) / total**2)

        for player_id, value in enumerate(minutes, start=1):
            rows.append({
                "GAME_ID": game_id,
                "GAME_DATE": date,
                "TEAM": "ATL",
                "PLAYER_ID": player_id,
                "MIN": value,
            })

    players = pd.DataFrame(rows)
    teams = players[
        ["GAME_ID", "GAME_DATE", "TEAM"]
    ].drop_duplicates()

    result = add_historical_rotation_features(players, teams)
    by_game = result.groupby("GAME_ID").first()

    # Game 6 uses games 1 through 5.
    assert by_game.loc[
        "G6", "TEAM_ROTATION_CONCENTRATION_AVG_5"
    ] == pytest.approx(sum(expected_hhi[:5]) / 5)

    # Game 7 uses games 2 through 6, excluding game 1.
    assert by_game.loc[
        "G7", "TEAM_ROTATION_CONCENTRATION_AVG_5"
    ] == pytest.approx(sum(expected_hhi[1:6]) / 5)

    # The most recent completed game is game 6.
    assert by_game.loc[
        "G7", "TEAM_ROTATION_CONCENTRATION"
    ] == pytest.approx(expected_hhi[5])


def test_player_minutes_share_regulation_overtime_and_no_leakage():
    from src.features.rotation import add_historical_player_minutes_share

    players = pd.DataFrame({
        "PLAYER_ID": [101] * 4,
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "GAME_DATE": [
            "2026-10-01",
            "2026-10-03",
            "2026-10-05",
            "2026-10-07",
        ],
        "TEAM": ["ATL"] * 4,
        "MIN": [24, 53, 36, 48],
    })

    teams = pd.DataFrame({
        "GAME_ID": ["G1", "G2", "G3", "G4"],
        "TEAM": ["ATL"] * 4,
        "GAME_MINUTES": [48, 53, 48, 48],
    })

    result = add_historical_player_minutes_share(players, teams)
    result = result.set_index("GAME_ID")

    assert pd.isna(result.loc["G1", "PLAYER_MINUTES_SHARE_AVG_5"])
    assert pd.isna(result.loc["G1", "PLAYER_MINUTES_SHARE_STD_5"])
    assert pd.isna(result.loc["G1", "PLAYER_MINUTES_SHARE_TREND"])

    assert result.loc["G2", "PLAYER_MINUTES_SHARE_AVG_5"] == pytest.approx(0.10)
    assert result.loc["G3", "PLAYER_MINUTES_SHARE_AVG_5"] == pytest.approx(0.15)

    assert result.loc["G4", "PLAYER_MINUTES_SHARE_AVG_5"] == pytest.approx(0.15)
    assert result.loc["G4", "PLAYER_MINUTES_SHARE_STD_5"] == pytest.approx(0.05)
    assert result.loc["G4", "PLAYER_MINUTES_SHARE_TREND"] == pytest.approx(0.0)


def test_player_minutes_share_rolling_window_and_isolation():
    from src.features.rotation import add_historical_player_minutes_share

    rows = []
    team_rows = []

    for game_number in range(1, 8):
        game_id = f"G{game_number}"
        date = f"2026-10-{game_number:02d}"

        team_rows.append({
            "GAME_ID": game_id,
            "TEAM": "ATL",
            "GAME_MINUTES": 48,
        })

        rows.append({
            "PLAYER_ID": 101,
            "GAME_ID": game_id,
            "GAME_DATE": date,
            "TEAM": "ATL",
            "MIN": game_number * 5,
        })

        rows.append({
            "PLAYER_ID": 202,
            "GAME_ID": game_id,
            "GAME_DATE": date,
            "TEAM": "ATL",
            "MIN": 12,
        })

    players = pd.DataFrame(rows)
    teams = pd.DataFrame(team_rows)

    result = add_historical_player_minutes_share(players, teams)
    result = result.set_index(["PLAYER_ID", "GAME_ID"])

    # G7 uses G2 through G6: 10, 15, 20, 25, 30 minutes.
    expected_avg = 20 / 240

    assert result.loc[
        (101, "G7"), "PLAYER_MINUTES_SHARE_AVG_5"
    ] == pytest.approx(expected_avg)

    # Previous 3 games: G4, G5, G6 = 20, 25, 30 minutes.
    expected_trend = (25 - 20) / 240

    assert result.loc[
        (101, "G7"), "PLAYER_MINUTES_SHARE_TREND"
    ] == pytest.approx(expected_trend)

    # Player 202 consistently plays 12 minutes.
    assert result.loc[
        (202, "G7"), "PLAYER_MINUTES_SHARE_AVG_5"
    ] == pytest.approx(12 / 240)

    assert result.loc[
        (202, "G7"), "PLAYER_MINUTES_SHARE_STD_5"
    ] == pytest.approx(0.0)

    assert result.loc[
        (202, "G7"), "PLAYER_MINUTES_SHARE_TREND"
    ] == pytest.approx(0.0)
