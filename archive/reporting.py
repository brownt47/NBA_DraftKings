
def print_data_summary(slate_date, games, team_games):
    """
    Report the selected slate date and the number
    of historical player and team game records.
    """

    print(f"\nSlate date: {slate_date.date()}")
    print(f"Historical player-game records: {len(games)}")
    print(f"Historical team records: {len(team_games)}")


def print_team_summary(team_features):
    """
    Report the number of NBA teams with
    calculated historical features.
    """

    print(
        f"Teams with historical features: {len(team_features)}"
    )


def print_analysis_summary(players, output):
    """
    Print DraftKings analysis results.
    """

    print(f"\nDraftKings players: {len(players)}")
    print(
        f"Players with ESPN IDs: "
        f"{players['ESPN_PLAYER_ID'].notna().sum()}"
    )
    print(
        f"Players with game history: "
        f"{players['GAMES_PLAYED'].notna().sum()}"
    )

    print("\nTop 10 by salary:")
    print(
        players[
            [
                "Name",
                "Salary",
                "GAMES_PLAYED",
                "AVG_DK_POINTS",
                "VALUE_PER_1000",
            ]
        ].head(10).round(2).to_string(index=False)
    )

    print(f"\nSaved to: {output}")
