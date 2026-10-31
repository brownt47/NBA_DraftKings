from pathlib import Path
import re
import unicodedata
import requests
import time

import pandas as pd


# File locations
ESPN_FILE = Path("data/nba_player_games_2026_27.csv")
ROSTER_FILE = Path("data/espn_player_directory.csv")
MAPPING_FILE = Path("data/dk_espn_player_mapping.csv")

DK_FOLDER = Path("data/dk_salaries")

salary_files = list(DK_FOLDER.glob("DKSalaries*.csv"))

if not salary_files:
    raise FileNotFoundError(
        f"No DraftKings salary files found in {DK_FOLDER}"
    )

DK_FILE = max(salary_files, key=lambda file: file.stat().st_mtime)

print(f"Using DraftKings salary file: {DK_FILE.name}")

def normalize_name(name):
    """Standardize player names for matching."""

    name = str(name).lower().strip()

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    # Remove punctuation
    name = re.sub(r"[-']", " ", name)
    name = re.sub(r"[^\w\s]", "", name)

    # Remove common suffixes
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b$", "", name)

    # Remove extra spaces
    return " ".join(name.split())

def search_espn_player(player_name):
    """
    Search ESPN for an NBA player by name.

    Returns (ESPN_ID, ESPN_NAME) if exactly one
    NBA player has an exact normalized name match.

    Otherwise returns None.
    """
    url = "https://site.api.espn.com/apis/search/v2"

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params={
                    "query": player_name,
                    "limit": 20
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            break

        except requests.RequestException as error:
            print(f"  ESPN search error: {error}")

            if attempt == 2:
                return None

            time.sleep(2)

    matches = []

    for result in data.get("results", []):
        if result.get("type") != "player":
            continue

        for player in result.get("contents", []):
            if player.get("description") != "NBA":
                continue

            espn_name = player.get("displayName", "")

            if normalize_name(espn_name) != normalize_name(player_name):
                continue

            uid = player.get("uid", "")
            espn_id = uid.split("~a:")[-1]

            if not espn_id.isdigit():
                continue

            matches.append((espn_id, espn_name))

    # Remove duplicate search results
    matches = list(set(matches))

    if len(matches) == 1:
        return matches[0]

    return None

def main():
    print("DraftKings to ESPN Player Matching")
    print("=" * 40)

    # Load both datasets
    espn_df = pd.read_csv(
        ESPN_FILE,
        dtype={"PLAYER_ID": str}
    )

    dk_df = pd.read_csv(
        DK_FILE,
        dtype={"ID": str}
    )

    # Get one record per ESPN player
    espn_players = espn_df[
        ["PLAYER_ID", "PLAYER_NAME"]
    ].drop_duplicates(subset=["PLAYER_ID"])

    # Normalize player names in both datasets
    espn_players = espn_players.copy()

        # Load players from ESPN's current team rosters
    roster_df = pd.read_csv(
        ROSTER_FILE,
        dtype={"PLAYER_ID": str}
    )

    # Combine historical players with current roster players
    espn_players = pd.concat(
        [
            espn_players,
            roster_df[["PLAYER_ID", "PLAYER_NAME"]]
        ],
        ignore_index=True
    )

    # Keep one record per ESPN player ID
    espn_players = espn_players.drop_duplicates(
        subset=["PLAYER_ID"]
    )

    espn_players["MATCH_NAME"] = (
        espn_players["PLAYER_NAME"].apply(normalize_name)
    )

    dk_df["MATCH_NAME"] = (
        dk_df["Name"].apply(normalize_name)
    )

    print(f"ESPN players: {len(espn_players)}")
    print(f"DraftKings players: {len(dk_df)}")

        # Match DraftKings players to ESPN players
    matched_df = dk_df.merge(
        espn_players[
            ["PLAYER_ID", "PLAYER_NAME", "MATCH_NAME"]
        ],
        on="MATCH_NAME",
        how="left",
        validate="many_to_many"
    )

    # Identify ambiguous matches
    match_counts = matched_df.groupby("ID")[
        "PLAYER_ID"
    ].transform("count")

    matched_df["MATCH_STATUS"] = "MATCHED"

    matched_df.loc[
        matched_df["PLAYER_ID"].isna(),
        "MATCH_STATUS"
    ] = "UNMATCHED"

    matched_df.loc[
        match_counts > 1,
        "MATCH_STATUS"
    ] = "AMBIGUOUS"

        # Reuse previously saved DraftKings-to-ESPN mappings
    if MAPPING_FILE.exists():
        existing = pd.read_csv(
            MAPPING_FILE,
            dtype={"DK_ID": str, "ESPN_PLAYER_ID": str}
        )

        saved_lookup = existing.set_index("DK_ID")

        reused = 0

        for idx, player in matched_df.iterrows():
            if player["MATCH_STATUS"] != "UNMATCHED":
                continue

            dk_id = str(player["ID"])

            if dk_id not in saved_lookup.index:
                continue

            saved = saved_lookup.loc[dk_id]

            matched_df.loc[idx, "PLAYER_ID"] = saved["ESPN_PLAYER_ID"]
            matched_df.loc[idx, "PLAYER_NAME"] = saved["ESPN_PLAYER_NAME"]
            matched_df.loc[idx, "MATCH_STATUS"] = "MATCHED"

            reused += 1

        print(f"\nReused {reused} previously saved mappings.")

        # Search ESPN for players not found in our local data
    unmatched = matched_df[
        matched_df["MATCH_STATUS"] == "UNMATCHED"
    ]

    print(f"\nSearching ESPN for {len(unmatched)} unmatched players...")

    for idx, player in unmatched.iterrows():
        name = player["Name"]

        print(f"  Searching: {name}")

        result = search_espn_player(name)

        if result is None:
            print(f"    No unique NBA match found")
            continue

        espn_id, espn_name = result

        # Do not assign an ESPN ID already linked to another DK player
        conflict = matched_df[
            (matched_df["PLAYER_ID"] == espn_id)
            & (matched_df["ID"] != player["ID"])
        ]

        if not conflict.empty:
            print(f"    Conflicting ESPN ID: {espn_id}")
            continue

        matched_df.loc[idx, "PLAYER_ID"] = espn_id
        matched_df.loc[idx, "PLAYER_NAME"] = espn_name
        matched_df.loc[idx, "MATCH_STATUS"] = "MATCHED"

        print(f"    Matched: {espn_name} ({espn_id})")


     # Save confirmed one-to-one player matches
    confirmed = matched_df[
        matched_df["MATCH_STATUS"] == "MATCHED"
    ][["ID", "Name", "PLAYER_ID", "PLAYER_NAME"]].copy()

    confirmed = confirmed.rename(columns={
        "ID": "DK_ID",
        "Name": "DK_NAME",
        "PLAYER_ID": "ESPN_PLAYER_ID",
        "PLAYER_NAME": "ESPN_PLAYER_NAME"
    })

    confirmed = confirmed.drop_duplicates(subset=["DK_ID"])

    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Preserve previously confirmed mappings across salary slates
    if MAPPING_FILE.exists():
        existing = pd.read_csv(
            MAPPING_FILE,
            dtype={"DK_ID": str, "ESPN_PLAYER_ID": str}
        )

                # Detect conflicting ESPN IDs for the same DraftKings ID
        comparison = confirmed.merge(
            existing[["DK_ID", "ESPN_PLAYER_ID"]],
            on="DK_ID",
            how="inner",
            suffixes=("_NEW", "_OLD")
        )

        conflicts = comparison[
            comparison["ESPN_PLAYER_ID_NEW"]
            != comparison["ESPN_PLAYER_ID_OLD"]
        ]

        if not conflicts.empty:
            print("\nERROR: Conflicting player mappings detected!")
            print(
                conflicts[
                    ["DK_ID", "DK_NAME",
                     "ESPN_PLAYER_ID_NEW", "ESPN_PLAYER_ID_OLD"]
                ].to_string(index=False)
            )
            raise ValueError(
                "Player mapping conflicts require manual review."
            )

        confirmed = pd.concat(
            [existing, confirmed],
            ignore_index=True
        )

        # Keep the latest mapping for each DraftKings player ID
        confirmed = confirmed.drop_duplicates(
            subset=["DK_ID"],
            keep="last"
        )

        # Check whether an ESPN ID belongs to multiple DK IDs
    duplicate_espn = confirmed[
        confirmed.duplicated(
            subset=["ESPN_PLAYER_ID"],
            keep=False
        )
    ]

    if not duplicate_espn.empty:
        print("\nERROR: Multiple DraftKings IDs share an ESPN ID!")
        print(
            duplicate_espn[
                ["DK_ID", "DK_NAME",
                 "ESPN_PLAYER_ID", "ESPN_PLAYER_NAME"]
            ].sort_values("ESPN_PLAYER_ID").to_string(index=False)
        )

        raise ValueError(
            "Duplicate ESPN player mappings require review."
        )

    confirmed.to_csv(MAPPING_FILE, index=False)

    print(f"\nSaved {len(confirmed)} player mappings.")

    # Display results
    print("\nMatching Results")
    print("-" * 40)

    print(
        matched_df["MATCH_STATUS"]
        .value_counts()
        .to_string()
    )

    # Show players needing review
    needs_review = matched_df[
        matched_df["MATCH_STATUS"] != "MATCHED"
    ]

    if not needs_review.empty:
        print("\nPlayers needing review:")
        print(
            needs_review[
                ["Name", "TeamAbbrev", "MATCH_STATUS"]
            ].to_string(index=False)
        )

        # Look for similar ESPN names
    from difflib import get_close_matches

    print("\nPossible ESPN name matches:")
    print("-" * 40)

    espn_names = espn_players["MATCH_NAME"].tolist()

    for _, player in needs_review.iterrows():
        suggestions = get_close_matches(
            player["MATCH_NAME"],
            espn_names,
            n=3,
            cutoff=0.6
        )

        print(f"\n{player['Name']}:")

        if suggestions:
            for suggestion in suggestions:
                possible = espn_players[
                    espn_players["MATCH_NAME"] == suggestion
                ]

                for _, match in possible.iterrows():
                    print(
                        f"  {match['PLAYER_NAME']} "
                        f"(ESPN ID: {match['PLAYER_ID']})"
                    )
        else:
            print("  No similar names found")

        print("\nChecking unmatched players against ESPN records")
    print("-" * 40)

    for _, player in needs_review.iterrows():
        name = player["Name"]

        matches = espn_df[
            espn_df["PLAYER_NAME"].str.contains(
                name.split()[-1],
                case=False,
                regex=False,
                na=False
            )
        ]

        print(
            f"{name}: "
            f"{matches['PLAYER_ID'].nunique()} "
            f"ESPN players with matching last name"
        )

if __name__ == "__main__":
    main()