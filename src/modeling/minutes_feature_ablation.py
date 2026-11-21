"""Leakage-safe, biweekly walk-forward ablation of NBA minutes predictors.

Run: python -m src.modeling.minutes_feature_ablation
Outputs: reports/modeling/minutes_feature_ablation/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.modeling.features import PREGAME_FEATURES
from src.modeling.lightgbm_model import DEFAULT_PARAMS, _minutes_to_float
from src.modeling.splits import biweekly_walk_forward_splits

DATA = Path('data/nba_features_2025_26.csv')
OUT = Path('reports/modeling/minutes_feature_ablation')

HISTORY = [
    'PRIOR_GAMES', 'MIN_AVG_SEASON',
    'MIN_AVG_LAST_3', 'MIN_AVG_LAST_5', 'MIN_AVG_LAST_7', 'MIN_AVG_LAST_10',
    'MIN_STD_LAST_3', 'MIN_STD_LAST_5', 'MIN_STD_LAST_7', 'MIN_STD_LAST_10',
    'LAST_GAME_MIN',
]
TRENDS = [
    'MIN_TREND_3', 'TEAM_CHANGE_FLAG', 'GAMES_WITH_CURRENT_TEAM',
    'MIN_AVG_CURRENT_TEAM', 'DAYS_REST', 'IS_BACK_TO_BACK',
    'GAMES_LAST_7_DAYS', 'GAMES_LAST_14_DAYS',
    'B2B_MIN_LOAD', 'B2B_LAST_GAME_MIN', 'B2B_EXCESS_MIN_LOAD',
    'PLAYER_MINUTES_LAST_7_DAYS', 'PLAYER_MINUTES_LAST_14_DAYS',
]
ROTATION = [
    'TEAM_ROTATION_CONCENTRATION', 'TEAM_ROTATION_CONCENTRATION_AVG_5',
    'PLAYER_MINUTES_SHARE_AVG_5', 'PLAYER_MINUTES_SHARE_STD_5',
    'PLAYER_MINUTES_SHARE_TREND',
]
PARTICIPATION = [
    'APPEARANCE_RATE_LAST_10', 'COACH_DNP_LAST_10',
    'MIN_PER_TEAM_GAME_LAST_5', 'PARTICIPATION_COVERAGE_LAST_10',
]

def unique(items):
    return list(dict.fromkeys(items))

GROUPS = {
    'A_HISTORY': unique(HISTORY),
    'B_TRENDS': unique(HISTORY + TRENDS),
    'C_ROTATION': unique(HISTORY + TRENDS + ROTATION),
    'D_PARTICIPATION': unique(HISTORY + TRENDS + ROTATION + PARTICIPATION),
    'E_FULL': unique(PREGAME_FEATURES),
}
PREDICTORS = ['BASELINE_5', 'BASELINE_10', *GROUPS]

def summarize(x):
    y = x['ACTUAL'].to_numpy()
    rows = []
    for name in PREDICTORS:
        p = x[name].to_numpy()
        valid = np.isfinite(y) & np.isfinite(p)
        e = p[valid] - y[valid]
        rows.append({'MODEL': name, 'N': int(valid.sum()),
                     'MAE': float(np.mean(np.abs(e))) if len(e) else np.nan,
                     'RMSE': float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
                     'BIAS': float(np.mean(e)) if len(e) else np.nan})
    return pd.DataFrame(rows)

def main():
    df = pd.read_csv(DATA, dtype={'GAME_ID': str, 'PLAYER_ID': str})
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['ACTUAL_MINUTES'] = df['MIN'].map(_minutes_to_float)
    needed = unique([c for cols in GROUPS.values() for c in cols])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing features: {missing}')
    if any(c in {'MIN', 'ACTUAL_MINUTES', 'MINUTES_TARGET', 'DK_POINTS'} for c in needed):
        raise ValueError('Current-game outcome included in predictors')
    for c in needed:
        if c in {'OPPONENT', 'HOME_AWAY'}:
            df[c] = df[c].astype('category')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    folds = list(biweekly_walk_forward_splits(df))
    all_preds = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        train = train.loc[train['ACTUAL_MINUTES'].notna()].copy()
        valid = test['ACTUAL_MINUTES'].notna() & test['MIN_AVG_LAST_5'].notna() & test['MIN_AVG_LAST_10'].notna()
        test = test.loc[valid].copy()
        if test.empty:
            continue
        result = test[['GAME_ID', 'PLAYER_ID', 'GAME_DATE']].copy()
        result['FOLD'] = fold
        result['ACTUAL'] = test['ACTUAL_MINUTES'].to_numpy()
        result['BASELINE_5'] = test['MIN_AVG_LAST_5'].to_numpy()
        result['BASELINE_10'] = test['MIN_AVG_LAST_10'].to_numpy()
        result['ROLE'] = pd.cut(test['MIN_AVG_LAST_5'], [-np.inf, 10, 20, 30, np.inf], labels=['Under 10', '10-19', '20-29', '30+'], right=False).astype(str).to_numpy()
        for name, columns in GROUPS.items():
            x_train = train[columns].copy()
            x_test = test[columns].copy()
            for c in ('OPPONENT', 'HOME_AWAY'):
                if c in columns:
                    x_test[c] = x_test[c].cat.set_categories(x_train[c].cat.categories)
            model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
            model.fit(x_train, train['ACTUAL_MINUTES'])
            result[name] = np.maximum(model.predict(x_test), 0)
        all_preds.append(result)
        print(f'Fold {fold}: train={len(train):,}, test={len(test):,}', flush=True)
    if not all_preds:
        raise RuntimeError('No eligible validation rows')
    preds = pd.concat(all_preds, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / 'predictions.csv', index=False)
    overall = summarize(preds)
    overall.to_csv(OUT / 'overall.csv', index=False)
    folds_summary = pd.concat([summarize(g).assign(FOLD=f) for f, g in preds.groupby('FOLD')], ignore_index=True)
    folds_summary.to_csv(OUT / 'fold_metrics.csv', index=False)
    roles_summary = pd.concat([summarize(g).assign(ROLE=r) for r, g in preds.groupby('ROLE')], ignore_index=True)
    roles_summary.to_csv(OUT / 'role_metrics.csv', index=False)
    print('\nOVERALL\n', overall.round(4).to_string(index=False))
    print('\nBY ROLE\n', roles_summary.round(4).to_string(index=False))
    print('\nMAE BY FOLD\n', folds_summary.pivot(index='FOLD', columns='MODEL', values='MAE').round(4).to_string())
    print(f'\nSaved to {OUT}')

if __name__ == '__main__':
    main()
