
import pandas as pd

from src.dk_features import load_player_games


def test_load_player_games_preserves_zero_minute_appearances(tmp_path):
    file_path = tmp_path / "player_games.csv"

    games = pd.DataFrame({
        "GAME_ID": ["G3", "G1", "G4", "G2"],
        "GAME_DATE": [
            "2026-10-30",
            "2026-10-25",
            "2026-10-31",
            "2026-10-27",
        ],
        "PLAYER_ID": [101, 101, 102, 101],
        "PLAYER_NAME": [
            "Test Player",
            "Test Player",
            "Other Player",
            "Test Player",
        ],
        "MIN": ["0", "30", "--", "0"],
        "AST": [1, 4, 0, 0],
        "DK_POINTS": [1.5, 35.0, 0.0, 0.0],
    })

    games.to_csv(file_path, index=False)

    result = load_player_games(file_path)

    # Missing minutes excluded; zero-minute appearances retained.
    assert len(result) == 3

    # Player games remain in chronological order.
    assert result["GAME_ID"].tolist() == ["G1", "G2", "G3"]

    # A zero-minute appearance with an assist is preserved.
    cryer_case = result.loc[result["GAME_ID"] == "G3"].iloc[0]
    assert float(cryer_case["MIN"]) == 0
    assert cryer_case["AST"] == 1
    assert cryer_case["DK_POINTS"] == 1.5

    # A zero-minute appearance without statistics is also preserved.
    assert "G2" in result["GAME_ID"].values

    # The original player-game identifiers remain intact.
    assert result["PLAYER_ID"].nunique() == 1
    assert not result.duplicated(["GAME_ID", "PLAYER_ID"]).any()
