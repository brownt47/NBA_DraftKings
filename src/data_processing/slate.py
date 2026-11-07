import argparse
import re
import pandas as pd


def normalize_player_name(name):
    """Normalize player names for matching across data sources."""
    if pd.isna(name):
        return ""

    name = str(name).casefold().strip()

    # Remove punctuation and spacing differences.
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)

    # Remove common suffixes.
    name = re.sub(
        r"\s+(jr|sr|ii|iii|iv)$",
        "",
        name,
    )

    return name

def resolve_player_mapping(dk, directory, mapping):
    """Resolve ESPN player IDs using names and historical mappings."""

    players = dk.copy()
    players["NAME_KEY"] = players["Name"].map(normalize_player_name)

    # ESPN directory lookup
    directory = directory.copy()
    directory["NAME_KEY"] = directory["PLAYER_NAME"].map(
        normalize_player_name
    )

    # Historical DraftKings mapping lookup
    historical = mapping.copy()
    historical["NAME_KEY"] = historical["DK_NAME"].map(
        normalize_player_name
    )

    # Reject names associated with multiple ESPN player IDs.
    for source, frame, id_column in (
        ("ESPN directory", directory, "PLAYER_ID"),
        ("Historical mapping", historical, "ESPN_PLAYER_ID"),
    ):
        # Only validate names appearing on the current slate.
        relevant = frame[
            frame["NAME_KEY"].isin(players["NAME_KEY"])
        ]

        conflicts = (
            relevant.groupby("NAME_KEY")[id_column]
            .nunique()
        )

        if (conflicts > 1).any():
            raise ValueError(
                f"Ambiguous player names in {source}: "
                f"{conflicts[conflicts > 1].index.tolist()}"
            )
        
    directory_lookup = (
        directory.drop_duplicates("NAME_KEY")
        [["NAME_KEY", "PLAYER_ID", "PLAYER_NAME", "TEAM"]]
    )

    historical_lookup = (
        historical.drop_duplicates("NAME_KEY")
        [["NAME_KEY", "ESPN_PLAYER_ID"]]
    )

    players = players.merge(
        directory_lookup,
        on="NAME_KEY",
        how="left",
        validate="many_to_one",
    )

    players = players.merge(
        historical_lookup,
        on="NAME_KEY",
        how="left",
        validate="many_to_one",
    )

    players["ESPN_PLAYER_ID"] = players["PLAYER_ID"].fillna(
        players["ESPN_PLAYER_ID"]
    )

    return players


def attach_slate_games(players, scheduled_games):
    """Attach ESPN game information to mapped DraftKings players."""

    from src.features.matchup import DK_TO_ESPN

    players = players.copy()
    players["ESPN_TEAM"] = players["TeamAbbrev"].replace(
        DK_TO_ESPN
    )

    records = []

    for game in scheduled_games:
        for team in (game["HOME_TEAM"], game["AWAY_TEAM"]):
            records.append({
                "ESPN_TEAM": team,
                "ESPN_GAME_ID": str(game["GAME_ID"]),
                "GAME_DATE": game["GAME_DATE"],
                "MATCHUP": game["MATCHUP"],
            })

    schedule = pd.DataFrame(records)

    if schedule["ESPN_TEAM"].duplicated().any():
        raise ValueError(
            "A team appears in multiple scheduled games."
        )

    players = players.merge(
        schedule,
        on="ESPN_TEAM",
        how="left",
        validate="many_to_one",
    )

    if players["ESPN_GAME_ID"].isna().any():
        missing = players.loc[
            players["ESPN_GAME_ID"].isna(),
            ["Name", "TeamAbbrev"]
        ]
        raise ValueError(
            f"Players without scheduled games:\n{missing}"
        )

    players["ESPN_PLAYER_NAME"] = players["PLAYER_NAME"].fillna(
        players["Name"]
    )

    return players


def merge_player_mapping(dk, mapping):
    """
    Connect DraftKings players to ESPN player IDs.

    Preserve all DraftKings players and validate
    that each DraftKings ID maps at most once.
    """

    players = dk.merge(
        mapping[["DK_ID", "ESPN_PLAYER_ID"]],
        left_on="ID",
        right_on="DK_ID",
        how="left",
        validate="one_to_one"
    )

    return players


def build_output_path(data_folder, slate_date, dk_file):
    """
    Construct the output CSV path using
    the slate date and DraftKings slate type.
    """

    return data_folder / (
        f"dk_player_analysis_{slate_date:%Y_%m_%d}_"
        f"{dk_file.stem.rsplit('_', 1)[-1]}.csv"
    )

# Automatically select the latest DraftKings salary file
def select_salary_file(dk_folder):
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", default=None)
    parser.add_argument("--slate", default="Main")

    args = parser.parse_args()

    candidates = []

    for file in dk_folder.glob("DKSalaries*.csv"):
        match = re.fullmatch(
            r"DKSalaries_(\d{4})_(\d{2})_(\d{2})_(Early|Main|Late)\.csv",
            file.name,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        year, month, day, slate = match.groups()
        file_date = pd.Timestamp(
            f"{year}-{month}-{day}"
        )

        if args.date and file_date != pd.Timestamp(args.date):
            continue

        if slate.lower() != args.slate.lower():
            continue

        candidates.append((file_date, file))

    if not candidates:
        raise FileNotFoundError(
            "No matching DraftKings salary file found."
        )

    slate_date, selected_file = max(
        candidates,
        key=lambda item: item[0]
    )


    return selected_file, slate_date