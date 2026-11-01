
import pandas as pd

DK_TO_ESPN = {
    "SAS": "SA",
    "NOP": "NO",
    "GSW": "GS",
    "NYK": "NY",
    "UTA": "UTAH",
    "WAS": "WSH",
}

def add_opponent(players):
    """Identify each player's opponent from the DK matchup."""
    players = players.copy()

    matchups = players["Game Info"].str.extract(
        r"^([A-Z]+)@([A-Z]+)"
    )

    away_team = matchups[0]
    home_team = matchups[1]

    players["OPPONENT_DK"] = home_team.where(
        players["TeamAbbrev"] == away_team,
        away_team
    )

    players["OPPONENT"] = (
        players["OPPONENT_DK"].replace(DK_TO_ESPN)
    )

    return players

def add_team_matchup_features(players, team_features):
    """Attach opponent and own-team performance statistics."""
    players = players.copy()

    # Opponent statistics
    opponent_features = team_features.rename(
        columns={
            "TEAM": "OPPONENT",
            **{
                col: f"OPP_{col}"
                for col in team_features.columns
                if col != "TEAM"
            }
        }
    )

    players = players.merge(
        opponent_features,
        on="OPPONENT",
        how="left",
        validate="many_to_one"
    )

    # Own-team statistics
    players["TEAM_ESPN"] = (
        players["TeamAbbrev"].replace(DK_TO_ESPN)
    )

    own_team_features = team_features.rename(
        columns={
            "TEAM": "TEAM_ESPN",
            **{
                col: f"TEAM_{col}"
                for col in team_features.columns
                if col != "TEAM"
            }
        }
    )

    players = players.merge(
        own_team_features,
        on="TEAM_ESPN",
        how="left",
        validate="many_to_one"
    )


    # Estimate game pace using both teams
    players["MATCHUP_PACE_SEASON"] = (
        players["TEAM_PACE_SEASON"]
        + players["OPP_PACE_SEASON"]
    ) / 2

    # Recent matchup pace estimates
    for window in (3, 5, 7):
        players[f"MATCHUP_PACE_LAST_{window}"] = (
            players[f"TEAM_PACE_LAST_{window}"]
            + players[f"OPP_PACE_LAST_{window}"]
        ) / 2


    return players

def validate_matchup_features(players):
    """Validate opponent assignments and team feature coverage."""

    matchups = players["Game Info"].str.extract(
        r"^([A-Z]+)@([A-Z]+)"
    )

    valid_matchup = (
        players["TeamAbbrev"].eq(matchups[0])
        | players["TeamAbbrev"].eq(matchups[1])
    )

    checks = {
        "Invalid team matchups": (~valid_matchup).sum(),
        "Missing opponents": players["OPPONENT"].isna().sum(),
        "Missing opponent ratings": (
            players["OPP_DEF_RATING_SEASON"].isna().sum()
        ),
        "Missing own-team ratings": (
            players["TEAM_DEF_RATING_SEASON"].isna().sum()
        ),
        "Duplicate DK IDs": players["ID"].duplicated().sum(),
    }

    print("\nMatchup feature validation")
    print(f"Total players: {len(players)}")

    for label, count in checks.items():
        print(f"{label}: {count}")

    if any(count > 0 for count in checks.values()):
        raise ValueError("Matchup feature validation failed.")

def add_matchup_features(players, team_features):
    """
    Identify opponents and attach team matchup statistics.
    """

    players = add_opponent(players)

    players = add_team_matchup_features(
        players,
        team_features
    )

    return players



def add_historical_matchup_features(players, team_games):
    """Attach pregame team and opponent features to player-game records."""
    players = players.copy()

    feature_columns = [
        col for col in team_games.columns
        if col == "TEAM_PRIOR_GAMES"
        or col.endswith("_SEASON")
        or "_LAST_" in col
    ]

    team_columns = [
        "GAME_ID", "TEAM", "OPPONENT", "HOME_AWAY"
    ] + feature_columns

    # Attach the player's own-team information
    own_team = team_games[team_columns].rename(
        columns={
            "OPPONENT": "OPPONENT",
            "HOME_AWAY": "HOME_AWAY",
            **{
                col: f"TEAM_{col}"
                for col in feature_columns
            },
        }
    )

    players = players.merge(
        own_team,
        on=["GAME_ID", "TEAM"],
        how="left",
        validate="many_to_one"
    )

    # Attach opponent features from the same game
    opponent = team_games[
        ["GAME_ID", "TEAM"] + feature_columns
    ].rename(
        columns={
            "TEAM": "OPPONENT",
            **{
                col: f"OPP_{col}"
                for col in feature_columns
            },
        }
    )

    players = players.merge(
        opponent,
        on=["GAME_ID", "OPPONENT"],
        how="left",
        validate="many_to_one"
    )

    # Estimated matchup pace
    players["MATCHUP_PACE_SEASON"] = (
        players["TEAM_PACE_SEASON"]
        + players["OPP_PACE_SEASON"]
    ) / 2

    for window in (3, 5, 7, 10):
        players[f"MATCHUP_PACE_LAST_{window}"] = (
            players[f"TEAM_PACE_LAST_{window}"]
            + players[f"OPP_PACE_LAST_{window}"]
        ) / 2

    return players
