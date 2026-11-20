from pathlib import Path
from collections import Counter
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CONTESTS = DATA / "dk_contests"
OUTPUT = DATA / "dk_backtest" / "opponent_analysis"

WATCHLIST = {
    "griffalicious",
    "nickenyon6",
    "dachamp0824",
    "meatman7",
    "scohen21",
    "hyjamon",
}

SLOTS = r"PG|SG|SF|PF|UTIL|C|G|F"
PLAYER_PATTERN = re.compile(
    rf"(?:^|\s)({SLOTS})\s+(.+?)(?=\s+(?:{SLOTS})\s+|$)"
)

def normalize_username(name):
    name = str(name).strip()
    name = re.sub(r"\s+\(\d+/\d+\)$", "", name)
    return name.lower()


def parse_lineup(lineup):
    if pd.isna(lineup):
        return tuple()

    matches = PLAYER_PATTERN.findall(str(lineup))
    players = [name.strip() for _, name in matches]

    if len(players) != 8:
        return tuple()

    return tuple(sorted(players))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(CONTESTS.rglob("contest-standings-*.csv"))

    print(f"Found {len(paths)} contest files")

    all_entries = []

    for path in paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"SKIP {path.name}: {exc}")
            continue

        required = {"EntryName", "Lineup", "Points"}
        if not required.issubset(df.columns):
            print(f"SKIP {path.name}: missing required columns")
            continue

        match = re.search(
            r"contest-standings-(\d+)(?:_(\d{4}_\d{2}_\d{2}))?",
            path.stem
        )

        if not match:
            continue

        contest_id, date_tag = match.groups()

        # DK standings can include a secondary player-stat table.
        # Only actual contest entries have a valid EntryId.
        if "EntryId" in df.columns:
            df = df.loc[df["EntryId"].notna()].copy()

        df["NORMALIZED_USER"] = df["EntryName"].map(
            normalize_username
        )

        df["LINEUP_KEY"] = df["Lineup"].map(parse_lineup)

        df = df.loc[
            df["LINEUP_KEY"].map(len).eq(8)
        ].copy()

        df["CONTEST_ID"] = contest_id
        df["DATE"] = (
            date_tag.replace("_", "-") if date_tag else None
        )
        df["SOURCE_FILE"] = path.name

        all_entries.append(df)

    if not all_entries:
        raise RuntimeError("No valid contest entries found")

    entries = pd.concat(all_entries, ignore_index=True)

    # Deduplicate repeated downloads of the same contest entry.
    if "EntryId" in entries.columns:
        entries = entries.drop_duplicates(
            subset=["CONTEST_ID", "EntryId"]
        )

    entries["LINEUP_ID"] = entries["LINEUP_KEY"].map(
        lambda x: "|".join(x)
    )

    entries.to_csv(
        OUTPUT / "all_entries.csv",
        index=False
    )

    print(f"\nValid entries: {len(entries):,}")
    print(f"Unique users: {entries['NORMALIZED_USER'].nunique():,}")
    print(f"Unique lineups: {entries['LINEUP_ID'].nunique():,}")

    # Repeated usernames.
    users = entries.groupby("NORMALIZED_USER").agg(
        ENTRIES=("EntryName", "size"),
        CONTESTS=("CONTEST_ID", "nunique"),
        DATES=("DATE", "nunique"),
        AVG_POINTS=("Points", "mean"),
        BEST_POINTS=("Points", "max"),
    ).reset_index()

    users = users.sort_values(
        ["DATES", "CONTESTS", "ENTRIES"],
        ascending=False
    )

    users.to_csv(
        OUTPUT / "recurring_users.csv",
        index=False
    )

    print("\nTOP RECURRING USERS")
    print(users.head(25).to_string(index=False))

    # Same lineup in the same contest.
    duplicates = entries.groupby(
        ["CONTEST_ID", "DATE", "LINEUP_ID"],
        dropna=False
    ).agg(
        ENTRIES=("NORMALIZED_USER", "size"),
        UNIQUE_USERS=("NORMALIZED_USER", "nunique"),
        USERS=("NORMALIZED_USER", lambda x: ", ".join(sorted(set(x)))),
        POINTS=("Points", "first"),
    ).reset_index()

    duplicates = duplicates.loc[
        duplicates["ENTRIES"] >= 2
    ].sort_values(
        ["UNIQUE_USERS", "ENTRIES"],
        ascending=False
    )

    duplicates["DUPLICATE_TYPE"] = duplicates[
        "UNIQUE_USERS"
    ].map(
        lambda n: "CROSS_USER" if n > 1 else "SAME_USER"
    )

    duplicates.to_csv(
        OUTPUT / "duplicate_lineups.csv",
        index=False
    )

    print("\nTOP DUPLICATE LINEUP GROUPS")
    print(
        duplicates[
            [
                "DATE", "CONTEST_ID", "ENTRIES",
                "UNIQUE_USERS", "DUPLICATE_TYPE",
                "POINTS", "USERS"
            ]
        ].head(20).to_string(index=False)
    )

    # Track specific usernames, including Reddit-linked users.
    watched = entries.loc[
        entries["NORMALIZED_USER"].isin(WATCHLIST)
    ].copy()

    watched.to_csv(
        OUTPUT / "watched_users.csv",
        index=False
    )

    print("\nWATCHLIST")
    for username in sorted(WATCHLIST):
        subset = watched.loc[
            watched["NORMALIZED_USER"].eq(username)
        ]

        print(
            f"{username:20s} "
            f"entries={len(subset):3d} "
            f"contests={subset['CONTEST_ID'].nunique():3d} "
            f"dates={subset['DATE'].nunique():3d}"
        )

    # Common player selections across all entries.
    player_counts = Counter()

    for lineup in entries["LINEUP_KEY"]:
        player_counts.update(lineup)

    common = pd.DataFrame(
        player_counts.most_common(),
        columns=["PLAYER", "ENTRY_COUNT"]
    )

    common.to_csv(
        OUTPUT / "common_players.csv",
        index=False
    )

    print("\nMOST COMMON PLAYERS")
    print(common.head(25).to_string(index=False))

    print(f"\nReports saved to {OUTPUT}")


if __name__ == "__main__":
    main()
