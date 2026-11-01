
def validate_player_features(players):
    """
    Check completeness of calculated player features.

    Players without historical games are excluded
    from feature completeness checks.
    """

    print("\nPlayer feature validation")

    total_players = len(players)

    has_history = players["GAMES_PLAYED"].notna()
    historical_players = players[has_history]

    print(f"Total players: {total_players}")
    print(f"Players with history: {len(historical_players)}")
    print(f"Players without history: {(~has_history).sum()}")

    required_features = [
        "AVG_DK_POINTS",
        "VALUE_PER_1000",
    ]

    for feature in required_features:
        missing = historical_players[feature].isna().sum()
        print(f"Missing {feature}: {missing}")
