from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.features import PREGAME_FEATURES
from src.modeling.splits import biweekly_walk_forward_splits

DATA = Path("data/nba_features_2025_26.csv")
OUT = Path("reports/modeling/feature_ablation")
OUT.mkdir(parents=True, exist_ok=True)

PREFIXES = (
    "TEAM_PACE_",
    "TEAM_OFF_RATING_",
    "TEAM_DEF_RATING_",
)

removed = [f for f in PREGAME_FEATURES if f.startswith(PREFIXES)]
reduced = [f for f in PREGAME_FEATURES if f not in removed]

assert len(PREGAME_FEATURES) == 125
assert len(removed) == 15
assert len(reduced) == 110

df = pd.read_csv(DATA)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

assert df["DK_POINTS"].notna().all()

def prepare(data, features):
    X = data[features].copy()

    for col in ["OPPONENT", "HOME_AWAY"]:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN").astype(str)

    for col in X.columns.difference(["OPPONENT", "HOME_AWAY"]):
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X

rows = []
predictions = []

for fold, (train_idx, val_idx) in enumerate(
    biweekly_walk_forward_splits(df), start=1
):
    train = df.loc[train_idx]
    val = df.loc[val_idx]

    assert train["GAME_DATE"].max() < val["GAME_DATE"].min()

    print(
        f"\nFold {fold}: train={len(train):,}, "
        f"validation={len(val):,}",
        flush=True
    )

    for name, features in [
        
        ("NO_TEAM_SIDE_110", reduced),
    ]:
        X_train = prepare(train, features)
        X_val = prepare(val, features)

        model = CatBoostRegressor(
            iterations=500,
            depth=7,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            loss_function="RMSE",
            task_type="GPU",
            devices="0",
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
        )

        model.fit(
            X_train,
            train["DK_POINTS"],
            cat_features=[
                c for c in ["OPPONENT", "HOME_AWAY"]
                if c in features
            ],
        )

        p = np.maximum(model.predict(X_val), 0)
        y = val["DK_POINTS"].to_numpy()

        mae = mean_absolute_error(y, p)
        rmse = np.sqrt(mean_squared_error(y, p))
        bias = np.mean(p - y)

        rows.append({
            "FOLD": fold,
            "MODEL": name,
            "N": len(val),
            "MAE": mae,
            "RMSE": rmse,
            "BIAS": bias,
        })

        predictions.append(pd.DataFrame({
            "FOLD": fold,
            "GAME_DATE": val["GAME_DATE"].to_numpy(),
            "PLAYER_ID": val["PLAYER_ID"].to_numpy(),
            "MODEL": name,
            "ACTUAL": y,
            "PREDICTED": p,
        }))

        print(
            f"  {name}: MAE={mae:.4f} "
            f"RMSE={rmse:.4f} BIAS={bias:+.4f}",
            flush=True
        )

metrics = pd.DataFrame(rows)
preds = pd.concat(predictions, ignore_index=True)

metrics.to_csv(OUT / "team_side_folds.csv", index=False)
preds.to_csv(OUT / "team_side_predictions.csv", index=False)

print("\nPOOLED RESULTS")

for name, g in preds.groupby("MODEL"):
    print(
        f"{name:20s} "
        f"N={len(g):,} "
        f"MAE={mean_absolute_error(g.ACTUAL, g.PREDICTED):.4f} "
        f"RMSE={np.sqrt(mean_squared_error(g.ACTUAL, g.PREDICTED)):.4f} "
        f"BIAS={(g.PREDICTED-g.ACTUAL).mean():+.4f}"
    )

print(f"\nSaved results to {OUT}")
