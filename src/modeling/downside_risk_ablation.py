"""Walk-forward downside-risk feature ablation using out-of-fold CatBoost predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRED = Path('reports/modeling/boosting_comparison/predictions.csv')
FEATURES = Path('data/nba_features_2025_26.csv')
OUT = Path('reports/modeling/downside_risk/ablation')
BASE = ['CATBOOST', 'MIN_AVG_LAST_5']
GROUPS = {
    'scoring_volatility': ['DK_STD_LAST_5', 'DK_STD_LAST_10', 'DK_WHIPSAW_5', 'DK_BELOW_70_RATE_LAST_10', 'DK_BELOW_80_RATE_LAST_10', 'DK_BELOW_90_RATE_LAST_10'],
    'minutes_stability': ['MIN_STD_LAST_5', 'MIN_STD_LAST_10', 'MIN_WHIPSAW_5', 'MIN_DIRECTION_CHANGES_5', 'MIN_RATIO_5_10', 'MIN_RATIO_5_7'],
    'minutes_range': ['MIN_MIN_LAST_5', 'MIN_P90_LAST_10', 'MIN_HEADROOM_LAST_10', 'MIN_CEILING_RATIO_LAST_10'],
    'participation': ['APPEARANCE_RATE_LAST_5', 'APPEARANCE_RATE_LAST_10', 'COACH_DNP_LAST_5', 'COACH_DNP_LAST_10', 'MIN_PER_TEAM_GAME_LAST_5', 'MIN_PER_TEAM_GAME_LAST_10'],
    'rotation': ['PLAYER_MINUTES_SHARE_STD_5', 'PLAYER_MINUTES_SHARE_TREND', 'TEAM_ROTATION_CONCENTRATION_AVG_5'],
    'trends': ['DK_EMA_3', 'DK_EMA_10', 'DK_MOMENTUM', 'MIN_TREND_3', 'MIN_TREND_RECENT_3'],
    'team_context': ['TEAM_PACE_LAST_5', 'TEAM_DEF_RATING_LAST_5', 'TEAM_GAMES_LAST_7_DAYS', 'TEAM_TRAVEL_MILES_LAST_7_DAYS'],
}


def metrics(y, p):
    return {
        'N': len(y), 'EVENT_RATE': float(np.mean(y)),
        'BRIER': brier_score_loss(y, p),
        'LOG_LOSS': log_loss(y, p, labels=[0, 1]),
        'AUC': roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        'PR_AUC': average_precision_score(y, p) if np.any(y) else np.nan,
    }


def main():
    pred = pd.read_csv(PRED, dtype={'GAME_ID': str, 'PLAYER_ID': str})
    hist = pd.read_csv(FEATURES, dtype={'GAME_ID': str, 'PLAYER_ID': str})
    feature_names = list(dict.fromkeys(BASE[1:] + sum(GROUPS.values(), [])))
    missing = sorted(set(feature_names) - set(hist.columns))
    if missing:
        raise ValueError(f'Missing features: {missing}')
    if pred.duplicated(['GAME_ID', 'PLAYER_ID']).any() or hist.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game keys')
    df = pred.merge(hist[['GAME_ID', 'PLAYER_ID'] + feature_names], on=['GAME_ID', 'PLAYER_ID'], how='left', validate='one_to_one', indicator=True)
    if not df['_merge'].eq('both').all():
        raise ValueError('Unmatched player-games')
    df = df.loc[(df.MIN_AVG_LAST_5 >= 22) & (df.CATBOOST >= 10)].copy()
    df['FOLD'] = pd.to_numeric(df['FOLD'])
    df['SHORTFALL'] = df['CATBOOST'] - df['ACTUAL']
    df['ROLE'] = np.where(df.MIN_AVG_LAST_5 < 30, '22-29', '30+')
    variants = {'baseline': BASE}
    variants.update({name: list(dict.fromkeys(BASE + cols)) for name, cols in GROUPS.items()})
    all_cols = list(dict.fromkeys(BASE + sum(GROUPS.values(), [])))
    variants['combined'] = all_cols
    rows, predictions = [], []
    for role, role_df in df.groupby('ROLE'):
        for threshold in (10, 15, 20):
            group = role_df.copy()
            group['TARGET'] = (group.SHORTFALL >= threshold).astype(int)
            for fold in sorted(group.FOLD.unique()):
                if fold == group.FOLD.min():
                    continue
                train = group.loc[group.FOLD < fold]
                test = group.loc[group.FOLD == fold]
                if len(train) < 200 or len(test) < 30 or train.TARGET.nunique() < 2:
                    continue
                for variant, columns in variants.items():
                    # Drop columns with no observed training values to avoid version-specific imputer behavior.
                    used = [c for c in columns if train[c].notna().any()]
                    if not used:
                        continue
                    for algorithm in ('logistic', 'lightgbm'):
                        if algorithm == 'logistic':
                            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(max_iter=2000))
                        else:
                            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), LGBMClassifier(n_estimators=150, learning_rate=0.035, num_leaves=15, min_child_samples=60, reg_lambda=5, verbosity=-1, n_jobs=4, random_state=42))
                        model.fit(train[used], train.TARGET)
                        prob = np.clip(model.predict_proba(test[used])[:, 1], 1e-6, 1 - 1e-6)
                        record = {'ROLE': role, 'THRESHOLD': threshold, 'FOLD': int(fold), 'VARIANT': variant, 'ALGORITHM': algorithm, **metrics(test.TARGET.to_numpy(), prob)}
                        rows.append(record)
                        predictions.append(pd.DataFrame({'GAME_DATE': test.GAME_DATE.to_numpy(), 'GAME_ID': test.GAME_ID.to_numpy(), 'PLAYER_ID': test.PLAYER_ID.to_numpy(), 'ROLE': role, 'THRESHOLD': threshold, 'FOLD': int(fold), 'VARIANT': variant, 'ALGORITHM': algorithm, 'TARGET': test.TARGET.to_numpy(), 'PROBABILITY': prob}))
                        print(f'{role:5s} miss>={threshold:2d} F{fold} {variant:20s} {algorithm:8s} AUC={record["AUC"]:.3f} Brier={record["BRIER"]:.3f}', flush=True)
    if not predictions:
        raise RuntimeError('No valid test folds')
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'fold_metrics.csv', index=False)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(OUT / 'predictions.csv', index=False)
    summary = []
    for (role, threshold, variant, algorithm), part in oof.groupby(['ROLE', 'THRESHOLD', 'VARIANT', 'ALGORITHM']):
        summary.append({'ROLE': role, 'THRESHOLD': threshold, 'VARIANT': variant, 'ALGORITHM': algorithm, **metrics(part.TARGET.to_numpy(), part.PROBABILITY.to_numpy())})
    summary = pd.DataFrame(summary).sort_values(['ROLE', 'THRESHOLD', 'BRIER'])
    summary.to_csv(OUT / 'summary.csv', index=False)
    print('\nBEST BY BRIER WITHIN ROLE/THRESHOLD:\n')
    print(summary.groupby(['ROLE', 'THRESHOLD']).head(5).round(4).to_string(index=False))
    print(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
