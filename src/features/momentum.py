import pandas as pd


def calculate_momentum_features(games):
    """
    Calculate EMA-based DraftKings scoring momentum.

    Expects historical games filtered to dates
    before the selected slate.
    """

    records = []

    for player_id, player_games in games.groupby("PLAYER_ID"):
        player_games = player_games.sort_values(
            ["GAME_DATE", "GAME_ID"]
        )

        dk = player_games["DK_POINTS"]

        ema_3 = dk.ewm(span=3, adjust=False).mean()
        ema_10 = dk.ewm(span=10, adjust=False).mean()

        momentum = ema_3 - ema_10

        # Count EMA crossover reversals within the last 5 game transitions.
        # Zero momentum is treated as neutral, not a crossover.
        momentum_sign = momentum.apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )

        recent_signs = momentum_sign.tail(6)
        previous_signs = recent_signs.shift(1)


        whipsaw_5 = int(
            (
                (recent_signs != 0)
                & previous_signs.notna()
                & (previous_signs != 0)
                & (recent_signs != previous_signs)
            ).sum()
        )

        
        record = {
            "ESPN_PLAYER_ID": player_id,
            "DK_EMA_3": ema_3.iloc[-1],
            "DK_EMA_10": ema_10.iloc[-1],
            "DK_MOMENTUM": momentum.iloc[-1],
            "DK_MOMENTUM_CHANGE": (
                momentum.iloc[-1] - momentum.iloc[-2]
                if len(momentum) >= 2
                else float("nan")
            ),
            "DK_CROSSOVER": (
                1 if len(momentum) >= 2
                and momentum.iloc[-2] <= 0
                and momentum.iloc[-1] > 0
                else -1 if len(momentum) >= 2
                and momentum.iloc[-2] >= 0
                and momentum.iloc[-1] < 0
                else 0
            ),
            "DK_WHIPSAW_5": whipsaw_5,
        }

        records.append(record)

    return pd.DataFrame(records)


def add_historical_momentum_features(games, short_span=3, long_span=10):
    """Calculate pregame EMA momentum features for every player-game."""
    games = games.copy()

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    games = games.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = games.groupby("PLAYER_ID")["DK_POINTS"]

    def calculate_player_momentum(dk):
        # Shift first so the current game's DK points are excluded
        previous = dk.shift(1)

        ema_short = previous.ewm(
            span=short_span, adjust=False
        ).mean()

        ema_long = previous.ewm(
            span=long_span, adjust=False
        ).mean()

        momentum = ema_short - ema_long

        sign = momentum.apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )

        crossover = pd.Series(0, index=dk.index)

        crossover.loc[
            (sign == 1) & (sign.shift(1) == -1)
        ] = 1

        crossover.loc[
            (sign == -1) & (sign.shift(1) == 1)
        ] = -1

        reversals = (
            (sign != 0)
            & (sign.shift(1) != 0)
            & (sign != sign.shift(1))
        ).astype(int)

        whipsaw = reversals.rolling(
            window=5, min_periods=1
        ).sum()

        return pd.DataFrame({
            "DK_EMA_3": ema_short,
            "DK_EMA_10": ema_long,
            "DK_MOMENTUM": momentum,
            "DK_MOMENTUM_CHANGE": momentum.diff(),
            "DK_CROSSOVER": crossover,
            "DK_WHIPSAW_5": whipsaw,
        })

    features = grouped.apply(
        calculate_player_momentum,
        include_groups=False
    )

    features.index = features.index.droplevel(0)

    return games.join(features)
