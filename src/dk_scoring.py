
import pandas as pd


def calculate_dk_points(df):
    """Calculate DraftKings NBA Classic fantasy points."""

    df = df.copy()

    # Base fantasy points
    df["DK_BASE"] = (
        df["PTS"]
        + 0.5 * df["FG3M"]
        + 1.25 * df["REB"]
        + 1.5 * df["AST"]
        + 2.0 * df["STL"]
        + 2.0 * df["BLK"]
        - 0.5 * df["TOV"]
    )

    # Count statistical categories with 10 or more
    # Only these five categories qualify for bonuses.
    categories = ["PTS", "REB", "AST", "STL", "BLK"]

    double_digit_count = (
        df[categories] >= 10
    ).sum(axis=1)

    df["DOUBLE_DOUBLE"] = (
        double_digit_count >= 2
    ).astype(int)

    df["TRIPLE_DOUBLE"] = (
        double_digit_count >= 3
    ).astype(int)

    # Bonuses are cumulative
    df["DK_BONUS"] = (
        1.5 * df["DOUBLE_DOUBLE"]
        + 3.0 * df["TRIPLE_DOUBLE"]
    )

    df["DK_POINTS"] = (
        df["DK_BASE"] + df["DK_BONUS"]
    )

    return df


if __name__ == "__main__":
    file_path = "data/nba_player_games_2026_27.csv"

    df = pd.read_csv(file_path)

    df = calculate_dk_points(df)

    print("\nTop 10 DraftKings performances:\n")

    print(
        df.nlargest(10, "DK_POINTS")[
            [
                "GAME_DATE",
                "PLAYER_NAME",
                "TEAM",
                "PTS",
                "REB",
                "AST",
                "DK_POINTS",
            ]
        ].to_string(index=False)
    )
