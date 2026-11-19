"""Test whether historical downside frequency adds predictive value."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PRED_PATH = Path("reports/modeling/boosting_comparison/predictions.csv")
FEATURE_PATH = Path("data/nba_features_2025_26.csv")
OUT_DIR = Path("reports/modeling/downside_risk")

BASELINE = [
    "CATBOOST",
    "MIN_AVG_LAST_5",
    "MIN_MIN_LAST_5",
    "MIN_STD_LAST_10",
]

PARTICIPATION = BASELINE + [
    "APPEARANCE_RATE_LAST_10",
    "COACH_DNP_LAST_10",
    "MIN_PER_TEAM_GAME_LAST_5",
    "PARTICIPATION_COVERAGE_LAST_10",
]

FULL_ROTATION = PARTICIPATION + [
    "PLAYER_MINUTES_SHARE_STD_5",
    "TEAM_ROTATION_CONCENTRATION_AVG_5",
]




def evaluate(train, test, columns):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        LogisticRegression(max_iter=2000),
    )

    model.fit(train[columns], train["TARGET"])
    probability = model.predict_proba(test[columns])[:, 1]

    return probability


def main():
    pred = pd.read_csv(
        PRED_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )
    hist = pd.read_csv(
        FEATURE_PATH, dtype={"GAME_ID": str, "PLAYER_ID": str}
    )

    cols = [
        "GAME_ID",
        "PLAYER_ID",
        *dict.fromkeys(
            BASELINE[1:] + FULL_ROTATION[1:]
        ),
    ]

    df = pred.merge(
        hist[cols],
        on=["GAME_ID", "PLAYER_ID"],
        validate="one_to_one",
        indicator=True,
    )

    if not df["_merge"].eq("both").all():
        raise ValueError("Unmatched player-games")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.loc[
        (df["MIN_AVG_LAST_5"] >= 22)
        & (df["CATBOOST"] >= 10)
    ].copy()

    df["TARGET"] = (
        df["ACTUAL"] < 0.70 * df["CATBOOST"]
    ).astype(int)

    df["ROLE"] = np.where(
        df["MIN_AVG_LAST_5"] < 30, "22-29", "30+"
    )

    # Require prior downside history so both models
    # are compared on the same eligible observations.
    df = df.dropna(
        subset=["MIN_STD_LAST_10"]
    )

    # Chronological expanding validation:
    # Train on earlier months, test on later months.
    months = sorted(df["GAME_DATE"].dt.to_period("M").unique())

    rows = []
    predictions = []

    for role, group in df.groupby("ROLE"):
        for i in range(1, len(months)):
            test_month = months[i]

            train = group.loc[
                group["GAME_DATE"].dt.to_period("M") < test_month
            ]
            test = group.loc[
                group["GAME_DATE"].dt.to_period("M") == test_month
            ]

            if len(train) < 200 or len(test) < 50:
                continue

            if train["TARGET"].nunique() < 2:
                continue

            for name, columns in [
                ("Baseline", BASELINE),
                ("Participation", PARTICIPATION),
                ("Full Rotation", FULL_ROTATION),
            ]:
                probability = evaluate(train, test, columns)

                rows.append({
                    "ROLE": role,
                    "MONTH": str(test_month),
                    "MODEL": name,
                    "N": len(test),
                    "EVENT_RATE": test["TARGET"].mean(),
                    "BRIER": brier_score_loss(
                        test["TARGET"], probability
                    ),
                    "LOG_LOSS": log_loss(
                        test["TARGET"], probability,
                        labels=[0, 1],
                    ),
                    "AUC": (
                        roc_auc_score(test["TARGET"], probability)
                        if test["TARGET"].nunique() == 2
                        else np.nan
                    ),
                })

                predictions.append(pd.DataFrame({
                    "ROLE": role,
                    "MONTH": str(test_month),
                    "MODEL": name,
                    "GAME_ID": test["GAME_ID"].values,
                    "PLAYER_ID": test["PLAYER_ID"].values,
                    "TARGET": test["TARGET"].values,
                    "PROBABILITY": probability,
                }))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)

    if results.empty:
        print("No eligible chronological validation folds.")
        return

    results.to_csv(OUT_DIR / "monthly_metrics.csv", index=False)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(
        OUT_DIR / "risk_predictions.csv", index=False
    )

    summary = []

    for (role, name), group in all_predictions.groupby(
        ["ROLE", "MODEL"]
    ):
        y = group["TARGET"]
        p = group["PROBABILITY"]

        summary.append({
            "ROLE": role,
            "MODEL": name,
            "N": len(group),
            "EVENT_RATE": y.mean(),
            "BRIER": brier_score_loss(y, p),
            "LOG_LOSS": log_loss(y, p, labels=[0, 1]),
            "AUC": (
                roc_auc_score(y, p)
                if y.nunique() == 2 else np.nan
            ),
        })

    print("\nCHRONOLOGICAL DOWNSIDE RISK VALIDATION")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nMONTHLY RESULTS")
    print(results.round(4).to_string(index=False))
    print(f"\nReports saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
