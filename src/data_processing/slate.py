import argparse
import re
import pandas as pd

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