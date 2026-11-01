
from features.scoring import (
    calculate_scoring_features,
    calculate_rolling_scoring_features,
    calculate_value_per_1000,
)
from features.minutes import calculate_minutes_features
from features.momentum import calculate_momentum_features


def add_player_features(players, games):
    """
    Calculate and merge historical player features
    into the DraftKings slate.
    """

    # Season scoring
    stats = calculate_scoring_features(games)

    players = players.merge(
        stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )

    # Rolling scoring
    recent_stats = calculate_rolling_scoring_features(games)

    players = players.merge(
        recent_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )

    # Playing time
    minutes_stats = calculate_minutes_features(games)

    players = players.merge(
        minutes_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )

    # Momentum indicators
    momentum_stats = calculate_momentum_features(games)

    players = players.merge(
        momentum_stats,
        on="ESPN_PLAYER_ID",
        how="left",
        validate="many_to_one"
    )

    # Fantasy points per $1,000 salary
    players = calculate_value_per_1000(players)

    return players
