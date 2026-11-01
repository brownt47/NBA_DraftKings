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