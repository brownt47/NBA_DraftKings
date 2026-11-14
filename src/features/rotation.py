"""Pregame NBA rotation and playing-time distribution features."""

import pandas as pd


def add_historical_rotation_features(players, teams):
    """Add historical team rotation concentration without data leakage."""
    players = players.copy()
    teams = teams.copy()

    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    teams["GAME_DATE"] = pd.to_datetime(teams["GAME_DATE"])

    players["MIN_NUM"] = pd.to_numeric(players["MIN"], errors="coerce")

    # Calculate concentration for each completed team-game.
    keys = ["GAME_ID", "TEAM"]

    total_minutes = players.groupby(keys)["MIN_NUM"].transform("sum")

    players["_MIN_SHARE"] = players["MIN_NUM"] / total_minutes
    players["_SHARE_SQUARED"] = players["_MIN_SHARE"] ** 2

    concentration = (
        players.groupby(keys, as_index=False)["_SHARE_SQUARED"]
        .sum(min_count=1)
        .rename(columns={"_SHARE_SQUARED": "_ACTUAL_ROTATION_HHI"})
    )

    teams = teams.merge(
        concentration,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    teams = teams.sort_values(
        ["TEAM", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped = teams.groupby("TEAM")["_ACTUAL_ROTATION_HHI"]

    teams["TEAM_ROTATION_CONCENTRATION"] = grouped.shift(1)

    teams["TEAM_ROTATION_CONCENTRATION_AVG_5"] = (
        teams.groupby("TEAM")["_ACTUAL_ROTATION_HHI"]
        .transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    return players.drop(
        columns=["MIN_NUM", "_MIN_SHARE", "_SHARE_SQUARED"]
    ).merge(
        teams[
            keys
            + [
                "TEAM_ROTATION_CONCENTRATION",
                "TEAM_ROTATION_CONCENTRATION_AVG_5",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )


def add_historical_player_minutes_share(players, teams):
    """Calculate pregame player minutes-share features."""
    players = players.copy()

    game_lengths = teams[
        ["GAME_ID", "TEAM", "GAME_MINUTES"]
    ].copy()

    players = players.merge(
        game_lengths,
        on=["GAME_ID", "TEAM"],
        how="left",
        validate="many_to_one",
    )

    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    players = players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    minutes = pd.to_numeric(players["MIN"], errors="coerce")
    available = 5 * pd.to_numeric(
        players["GAME_MINUTES"], errors="coerce"
    )

    players["_MINUTES_SHARE"] = minutes / available

    grouped = players.groupby("PLAYER_ID")["_MINUTES_SHARE"]

    players["PLAYER_MINUTES_SHARE_AVG_5"] = grouped.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    players["PLAYER_MINUTES_SHARE_STD_5"] = grouped.transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std()
    )

    avg_3 = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )

    players["PLAYER_MINUTES_SHARE_TREND"] = (
        avg_3 - players["PLAYER_MINUTES_SHARE_AVG_5"]
    )

    return players.drop(columns=["GAME_MINUTES", "_MINUTES_SHARE"])

def add_participation_reliability_features(players, teams, participation, window=5):
    """Calculate pregame participation reliability over prior team games."""
    players = players.copy()
    teams = teams.copy()
    participation = participation.copy()

    for df in (players, teams, participation):
        df["GAME_ID"] = df["GAME_ID"].astype(str)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    for df in (players, participation):
        df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)

    if participation.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate participation records")

    if teams.duplicated(["GAME_ID", "TEAM"]).any():
        raise ValueError("Duplicate team-game records")

    # One row per team game, in chronological order.
    schedule = (
        teams[["GAME_ID", "GAME_DATE", "TEAM"]]
        .drop_duplicates()
        .sort_values(["TEAM", "GAME_DATE", "GAME_ID"])
        .copy()
    )

    schedule["_TEAM_GAME_NUMBER"] = schedule.groupby("TEAM").cumcount()

    participation = participation.merge(
        schedule[["GAME_ID", "TEAM", "_TEAM_GAME_NUMBER"]],
        on=["GAME_ID", "TEAM"],
        how="left",
        validate="many_to_one",
    )

    participation["PLAYED"] = (
        participation["PLAYED"].astype(str).str.lower().eq("true")
    )
    participation["DID_NOT_PLAY"] = (
        participation["DID_NOT_PLAY"].astype(str).str.lower().eq("true")
    )

    participation["_COACH_DNP"] = (
        participation["DID_NOT_PLAY"]
        & participation["DNP_REASON"]
        .fillna("")
        .str.upper()
        .eq("COACH'S DECISION")
    )


    # Convert minutes to numeric values.
    # Confirmed DNPs count as zero minutes.
    # Missing or invalid minutes for players marked played
    # remain unknown rather than becoming zero.
    # Synthetic tests or older datasets may lack MIN.
    # Missing minutes remain unknown unless a DNP is confirmed.
    if "MIN" in participation.columns:
        participation["_MINUTES_KNOWN"] = pd.to_numeric(
            participation["MIN"], errors="coerce"
        )
    else:
        participation["_MINUTES_KNOWN"] = float("nan")

    participation.loc[
        participation["DID_NOT_PLAY"],
        "_MINUTES_KNOWN",
    ] = 0.0


    # Build a complete team-game timeline for each observed player-team pair.
    player_teams = pd.concat(
        [
            participation[["PLAYER_ID", "TEAM"]],
            players[["PLAYER_ID", "TEAM"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    first_seen = (
        participation.groupby(["PLAYER_ID", "TEAM"])["GAME_DATE"]
        .min()
        .rename("_FIRST_SEEN")
        .reset_index()
    )

    timeline = player_teams.merge(
        first_seen,
        on=["PLAYER_ID", "TEAM"],
        how="left",
        validate="one_to_one",
    )

    timeline = timeline.merge(
        schedule,
        on="TEAM",
        how="left",
    )

    # Only calculate reliability after the first confirmed
    # participation record for this player-team combination.
    # Players with no participation history remain unknown.
    timeline = timeline.loc[
        timeline["_FIRST_SEEN"].notna()
        & (timeline["GAME_DATE"] >= timeline["_FIRST_SEEN"])
    ].copy()





    timeline = timeline.merge(
        participation[
            [
                "GAME_ID",
                "TEAM",
                "PLAYER_ID",
                "PLAYED",
                "_COACH_DNP",
                "_MINUTES_KNOWN",
            ]
        ],
        on=["GAME_ID", "TEAM", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
    )

    timeline["_KNOWN"] = timeline["PLAYED"].notna().astype(int)


    timeline["_PLAYED"] = (
        timeline["PLAYED"].eq(True).astype(int)
    )

    timeline["_COACH_DNP"] = (
        timeline["_COACH_DNP"].eq(True).astype(int)
    )


    timeline = timeline.sort_values(
        ["PLAYER_ID", "TEAM", "_TEAM_GAME_NUMBER"]
    )

    grouped = timeline.groupby(["PLAYER_ID", "TEAM"])


    # Most recent confirmed appearance before the current game.
    # A DNP or unknown participation does not reset the counter.
    timeline["_LAST_APPEARANCE_GAME"] = (
        timeline["_TEAM_GAME_NUMBER"]
        .where(timeline["_PLAYED"].eq(1))
    )

    timeline["_LAST_APPEARANCE_GAME"] = (
        timeline.groupby(["PLAYER_ID", "TEAM"])[
            "_LAST_APPEARANCE_GAME"
        ].transform(
            lambda s: s.shift(1).ffill()
        )
    )

    # Count intervening team games since the last appearance.
    # Zero means the player appeared in the previous team game.
    timeline["TEAM_GAMES_SINCE_LAST_APPEARANCE"] = (
        timeline["_TEAM_GAME_NUMBER"]
        - timeline["_LAST_APPEARANCE_GAME"]
        - 1
    )

    # Last confirmed appearance date before the current game.
    # Exclude the current game to prevent data leakage.
    timeline["_LAST_APPEARANCE_DATE"] = (
        timeline["GAME_DATE"]
        .where(timeline["_PLAYED"].eq(1))
    )

    timeline["_LAST_APPEARANCE_DATE"] = (
        timeline.groupby(["PLAYER_ID", "TEAM"])[
            "_LAST_APPEARANCE_DATE"
        ].transform(
            lambda s: s.shift(1).ffill()
        )
    )

    timeline["DAYS_SINCE_LAST_APPEARANCE"] = (
        timeline["GAME_DATE"]
        - timeline["_LAST_APPEARANCE_DATE"]
    ).dt.days


    # Count prior eligible team games, excluding the current game.
    # Early-season players may have fewer than five prior games.
    
    feature_columns = [
        "GAME_ID",
        "TEAM",
        "PLAYER_ID",
        "TEAM_GAMES_SINCE_LAST_APPEARANCE",
        "DAYS_SINCE_LAST_APPEARANCE",
    ]


    # Rolling minutes use only prior team games.
    # Confirmed DNPs contribute zero; unknown minutes are excluded.
    for w in (5, 10):
        minutes_col = f"MIN_PER_TEAM_GAME_LAST_{w}"
        count_col = f"MINUTES_KNOWN_LAST_{w}"

        timeline[minutes_col] = grouped["_MINUTES_KNOWN"].transform(
            lambda s: s.shift(1).rolling(
                w, min_periods=1
            ).mean()
        )

        timeline[count_col] = grouped["_MINUTES_KNOWN"].transform(
            lambda s: s.shift(1).rolling(
                w, min_periods=1
            ).count()
        )

        feature_columns.extend([
            minutes_col,
            count_col,
        ])


    for w in (5, 10):
        known_col = f"PARTICIPATION_KNOWN_LAST_{w}"
        played_col = f"_PARTICIPATION_PLAYED_LAST_{w}"
        dnp_col = f"COACH_DNP_LAST_{w}"
        rate_col = f"APPEARANCE_RATE_LAST_{w}"
        coverage_col = f"PARTICIPATION_COVERAGE_LAST_{w}"

        for source, target in [
            ("_KNOWN", known_col),
            ("_PLAYED", played_col),
            ("_COACH_DNP", dnp_col),
        ]:
            timeline[target] = grouped[source].transform(
                lambda s: s.shift(1).rolling(
                    w, min_periods=1
                ).sum()
            )

        timeline[rate_col] = (
            timeline[played_col]
            / timeline[known_col].replace(0, float("nan"))
        )

        # Number of observable prior team games since first seen.
        prior_games = grouped.cumcount()
        eligible_games = prior_games.clip(upper=w)

        timeline[coverage_col] = (
            timeline[known_col]
            / eligible_games.replace(0, float("nan"))
        )

        feature_columns.extend([
            rate_col,
            known_col,
            dnp_col,
            coverage_col,
        ])

    return players.merge(
        timeline[feature_columns],
        on=["GAME_ID", "TEAM", "PLAYER_ID"],
        how="left",
        validate="many_to_one",
    )