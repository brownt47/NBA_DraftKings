"""Diagnose pregame minutes stability and CatBoost downside risk."""

from pathlib import Path
import numpy as np
import pandas as pd

PRED_PATH = Path("reports/modeling/boosting_comparison/predictions.csv")
FEATURE_PATH = Path("data/nba_features_2025_26.csv")
OUT_DIR = Path("reports/modeling/minutes_risk")

FEATURES = [
    "MIN_STD_LAST_5",
    "MIN_STD_LAST_10",
    "MIN_WHIPSAW_5",
    "MIN_DIRECTION_CHANGES_5",
    "MIN_MIN_LAST_5",
    "MIN_RATIO_5_10",
    "MIN_RATIO_5_7",
    "MIN_TREND_RECENT_3",
    "DK_BELOW_70_RATE_LAST_10",
    "DK_BELOW_80_RATE_LAST_10",
    "DK_BELOW_90_RATE_LAST_10",
]

def main():
    pred = pd.read_csv(
        PRED_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    hist = pd.read_csv(
        FEATURE_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )

    cols = [
        "GAME_ID", "PLAYER_ID", "MIN",
        "MIN_AVG_LAST_5",
        "DK_STD_LAST_10",
        *FEATURES,
    ]

    df = pred.merge(
        hist[cols],
        on=["GAME_ID", "PLAYER_ID"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        raise ValueError("Unmatched historical player-games")

    df["MIN_CHANGE"] = df["MIN"] - df["MIN_AVG_LAST_5"]
    df["DK_ERROR"] = df["CATBOOST"] - df["ACTUAL"]

    df["MIN_DROP_5"] = df["MIN_CHANGE"] <= -5
    df["MIN_DROP_10"] = df["MIN_CHANGE"] <= -10
    df["DK_OVER_10"] = df["DK_ERROR"] >= 10

    # Relative downside compared with CatBoost projection.
    # Exclude near-zero projections to avoid unstable percentages.
    df["DK_PCT_OF_PROJECTION"] = np.where(
        df["CATBOOST"] >= 10,
        df["ACTUAL"] / df["CATBOOST"],
        np.nan,
    )

    df["DK_BELOW_90_PCT"] = (
        df["DK_PCT_OF_PROJECTION"] < 0.90
    ).astype(float)

    df["DK_BELOW_80_PCT"] = (
        df["DK_PCT_OF_PROJECTION"] < 0.80
    ).astype(float)

    df["DK_BELOW_70_PCT"] = (
        df["DK_PCT_OF_PROJECTION"] < 0.70
    ).astype(float)

    # Missing ratios should not count as successful outcomes.
    for col in [
        "DK_BELOW_90_PCT",
        "DK_BELOW_80_PCT",
        "DK_BELOW_70_PCT",
    ]:
        df.loc[df["DK_PCT_OF_PROJECTION"].isna(), col] = np.nan


    df["ROLE"] = pd.cut(
        df["MIN_AVG_LAST_5"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["Under 10", "10-19", "20-29", "30+"],
        right=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nOVERALL DOWNSIDE RATES")
    print("Validation rows:", len(df))

    valid = df.dropna(subset=["MIN_CHANGE"])
    print("Rows with minutes baseline:", len(valid))
    print("Minutes drop >=5:", f"{valid['MIN_DROP_5'].mean():.2%}")
    print("Minutes drop >=10:", f"{valid['MIN_DROP_10'].mean():.2%}")
    print("CatBoost overprojection >=10:", f"{df['DK_OVER_10'].mean():.2%}")

    role_summary = valid.groupby(
        "ROLE", observed=True
    ).agg(
        N=("GAME_ID", "size"),
        MIN_DROP_5_RATE=("MIN_DROP_5", "mean"),
        MIN_DROP_10_RATE=("MIN_DROP_10", "mean"),
        DK_OVER_10_RATE=("DK_OVER_10", "mean"),
        DK_BELOW_90_RATE=("DK_BELOW_90_PCT", "mean"),
        DK_BELOW_80_RATE=("DK_BELOW_80_PCT", "mean"),
        DK_BELOW_70_RATE=("DK_BELOW_70_PCT", "mean"),
    )

    role_summary.to_csv(OUT_DIR / "role_summary.csv")

    print("\nDOWNSIDE BY EXPECTED MINUTES")
    print(role_summary.round(4).to_string())

    rows = []

    for feature in FEATURES:
        for role, group in valid.groupby("ROLE", observed=True):
            subset = group.dropna(subset=[feature]).copy()

            if len(subset) < 30:
                continue

            # Use quantiles of distinct values, avoiding
            # arbitrary rank-based splitting of ties.
            try:
                subset["BUCKET"] = pd.qcut(
                    subset[feature],
                    q=3,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue

            for bucket, part in subset.groupby("BUCKET"):
                rows.append({
                    "FEATURE": feature,
                    "ROLE": role,
                    "BUCKET": int(bucket) + 1,
                    "N": len(part),
                    "FEATURE_MEDIAN": part[feature].median(),
                    "MIN_DROP_5_RATE": part["MIN_DROP_5"].mean(),
                    "MIN_DROP_10_RATE": part["MIN_DROP_10"].mean(),
                    "DK_OVER_10_RATE": part["DK_OVER_10"].mean(),
                    "DK_ERROR_MEAN": part["DK_ERROR"].mean(),
                    "DK_BELOW_90_RATE": part["DK_BELOW_90_PCT"].mean(),
                    "DK_BELOW_80_RATE": part["DK_BELOW_80_PCT"].mean(),
                    "DK_BELOW_70_RATE": part["DK_BELOW_70_PCT"].mean(),
                })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "feature_risk_by_role.csv", index=False)

    print("\n20-29 MINUTE PLAYERS: FEATURE RISK")
    mid = detail.loc[detail["ROLE"] == "20-29"]

    if not mid.empty:
        print(mid.round(3).to_string(index=False))


    # Combined minutes reliability and fantasy volatility.
    # Analyze only players meeting our 22-minute floor.
    cash = df.loc[
        (df["MIN_AVG_LAST_5"] >= 22)
        & (df["CATBOOST"] >= 10)
    ].dropna(
        subset=[
            "MIN_MIN_LAST_5",
            "MIN_STD_LAST_10",
            "DK_STD_LAST_10",
            "DK_PCT_OF_PROJECTION",
        ]
    ).copy()

    cash["CASH_ROLE"] = np.where(
        cash["MIN_AVG_LAST_5"] < 30,
        "22-29",
        "30+",
    )

    combined_rows = []

    for role, group in cash.groupby("CASH_ROLE"):
        # Thresholds are calculated separately within each role.
        min_cutoff = group["MIN_MIN_LAST_5"].median()
        dk_cutoff = group["DK_STD_LAST_10"].median()

        group = group.copy()

        group["MINUTES_RELIABILITY"] = np.where(
            group["MIN_MIN_LAST_5"] >= min_cutoff,
            "Stable",
            "Unstable",
        )

        group["DK_VOLATILITY"] = np.where(
            group["DK_STD_LAST_10"] <= dk_cutoff,
            "Low",
            "High",
        )

        for (stability, volatility), part in group.groupby(
            ["MINUTES_RELIABILITY", "DK_VOLATILITY"]
        ):
            combined_rows.append({
                "ROLE": role,
                "MINUTES_RELIABILITY": stability,
                "DK_VOLATILITY": volatility,
                "N": len(part),
                "AVG_PROJECTION": part["CATBOOST"].mean(),
                "MIN_FLOOR_MEDIAN": part["MIN_MIN_LAST_5"].median(),
                "DK_STD_MEDIAN": part["DK_STD_LAST_10"].median(),
                "BELOW_90_RATE": part["DK_BELOW_90_PCT"].mean(),
                "BELOW_80_RATE": part["DK_BELOW_80_PCT"].mean(),
                "BELOW_70_RATE": part["DK_BELOW_70_PCT"].mean(),
                "MIN_DROP_10_RATE": part["MIN_DROP_10"].mean(),
            })

    combined = pd.DataFrame(combined_rows)

    combined.to_csv(
        OUT_DIR / "combined_cash_risk.csv",
        index=False,
    )

    print("\nCOMBINED CASH RISK: 22+ EXPECTED MINUTES")
    print("Eligible observations:", len(cash))

    print(
        combined.sort_values(
            ["ROLE", "MINUTES_RELIABILITY", "DK_VOLATILITY"]
        ).round(3).to_string(index=False)
    )


    # Historical downside frequency versus future CatBoost downside.
    # Only include players eligible under our 22-minute rule.
    downside_features = [
        "DK_BELOW_70_RATE_LAST_10",
        "DK_BELOW_80_RATE_LAST_10",
        "DK_BELOW_90_RATE_LAST_10",
    ]

    eligible = df.loc[
        (df["MIN_AVG_LAST_5"] >= 22)
        & (df["CATBOOST"] >= 10)
    ].copy()

    eligible["CASH_ROLE"] = np.where(
        eligible["MIN_AVG_LAST_5"] < 30,
        "22-29",
        "30+",
    )

    downside_rows = []

    for role, group in eligible.groupby("CASH_ROLE"):
        for feature in downside_features:
            subset = group.dropna(subset=[feature]).copy()

            if len(subset) < 30:
                continue

            # Separate players by historical downside frequency.
            # Duplicate boundaries are dropped automatically.
            subset["RISK_BUCKET"] = pd.qcut(
                subset[feature],
                q=3,
                labels=False,
                duplicates="drop",
            )

            for bucket, part in subset.groupby("RISK_BUCKET"):
                downside_rows.append({
                    "ROLE": role,
                    "FEATURE": feature,
                    "BUCKET": int(bucket) + 1,
                    "N": len(part),
                    "HISTORICAL_RATE_MEDIAN": part[feature].median(),
                    "AVG_PROJECTION": part["CATBOOST"].mean(),
                    "BELOW_80_RATE": part["DK_BELOW_80_PCT"].mean(),
                    "BELOW_70_RATE": part["DK_BELOW_70_PCT"].mean(),
                    "MIN_FLOOR_MEDIAN": part["MIN_MIN_LAST_5"].median(),
                })

    downside_report = pd.DataFrame(downside_rows)

    downside_report.to_csv(
        OUT_DIR / "historical_downside_risk.csv",
        index=False,
    )

    print("\nHISTORICAL DOWNSIDE FREQUENCY VS FUTURE RISK")
    print(
        downside_report.round(3).to_string(index=False)
    )



    print(f"\nSaved reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
