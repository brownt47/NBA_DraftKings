"""Audit archived DraftKings contest standings without assuming payout rules."""

from pathlib import Path
import re
import pandas as pd

CONTEST_DIR = Path("data/dk_contests")
OUTPUT_DIR = Path("reports/backtesting/contest_audit")

PATTERN = re.compile(
    r"contest-standings-(\d+)_(\d{4}_\d{2}_\d{2})\.csv$"
)

PERCENTILES = (0.40, 0.43, 0.45, 0.50)


def audit_contest(path):
    match = PATTERN.fullmatch(path.name)
    if not match:
        return None

    contest_id, date_tag = match.groups()
    date = date_tag.replace("_", "-")

    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")

    required = {"EntryId", "Rank", "Points", "TimeRemaining"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}")

    entries = df.dropna(
        subset=["EntryId", "Rank", "Points"]
    ).copy()

    for col in ("Rank", "Points", "TimeRemaining"):
        entries[col] = pd.to_numeric(
            entries[col], errors="coerce"
        )

    entries = entries.dropna(subset=["Rank", "Points"])

    n = len(entries)
    if n == 0:
        raise ValueError(f"{path.name}: no valid entries")

    scores = entries["Points"].sort_values(
        ascending=False
    ).reset_index(drop=True)

    remaining = entries["TimeRemaining"].dropna()
    max_remaining = remaining.max() if not remaining.empty else None

    duplicate_ids = int(entries["EntryId"].duplicated().sum())

    row = {
        "DATE": date,
        "CONTEST_ID": contest_id,
        "SOURCE_FILE": path.name,
        "ENTRIES": n,
        "UNIQUE_ENTRIES": entries["EntryId"].nunique(),
        "DUPLICATE_ENTRY_IDS": duplicate_ids,
        "MAX_RANK": entries["Rank"].max(),
        "MIN_SCORE": scores.min(),
        "MEDIAN_SCORE": scores.median(),
        "MAX_SCORE": scores.max(),
        "ZERO_SCORE_ENTRIES": int(scores.eq(0).sum()),
        "MAX_TIME_REMAINING": max_remaining,
        "STANDINGS_STATUS": (
            "REVIEW_TIME_REMAINING"
            if pd.notna(max_remaining) and max_remaining > 0
            else "FINAL_CANDIDATE"
        ),
        "CONTEST_TYPE": "UNVERIFIED",
        "PAID_PLACES": pd.NA,
        "PAYOUT_STATUS": "UNVERIFIED",
    }

    for fraction in PERCENTILES:
        position = max(1, int(n * fraction))
        cutoff = float(scores.iloc[position - 1])

        label = f"{int(fraction * 100)}"
        row[f"CUT_{label}"] = cutoff
        row[f"TIES_AT_{label}"] = int(scores.eq(cutoff).sum())

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    errors = []

    for path in sorted(CONTEST_DIR.glob("contest-standings-*.csv")):
        try:
            result = audit_contest(path)
            if result is not None:
                records.append(result)
        except Exception as exc:
            errors.append({
                "SOURCE_FILE": path.name,
                "ERROR": str(exc),
            })

    if not records:
        raise RuntimeError("No valid contest standings found")

    audit = pd.DataFrame(records)

    repeated_ids = audit["CONTEST_ID"].duplicated(keep=False)
    audit["REPEATED_CONTEST_ID"] = repeated_ids

    audit = audit.sort_values(
        ["DATE", "CONTEST_ID", "SOURCE_FILE"]
    )

    output = OUTPUT_DIR / "contest_audit.csv"
    audit.to_csv(output, index=False)

    print("\nCONTEST AUDIT")
    print(
        audit[
            [
                "DATE", "CONTEST_ID", "ENTRIES",
                "STANDINGS_STATUS", "REPEATED_CONTEST_ID",
                "CUT_43", "CUT_50"
            ]
        ].to_string(index=False)
    )

    print("\nSUMMARY")
    print("Files audited:", len(audit))
    print("Entry rows:", audit["ENTRIES"].sum())
    print(
        "Final candidates:",
        audit["STANDINGS_STATUS"].eq("FINAL_CANDIDATE").sum()
    )
    print(
        "Files with repeated contest IDs:",
        int(repeated_ids.sum())
    )
    print("Errors:", len(errors))

    if errors:
        error_path = OUTPUT_DIR / "audit_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print("Error report:", error_path)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
