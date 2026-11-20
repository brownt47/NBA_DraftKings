"""Compare existing stat CatBoost against opponent-position impact features.
Run: python -m src.modeling.opponent_stat_benchmark
Does not modify production models or source feature files.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.modeling.model_registry import get_model_adapter

DATA = [Path('data/nba_features_2025_26.csv'), Path('data/nba_features_2026_27.csv')]
IMPACT = Path('reports/modeling/opponent_stat_impact/features.csv')
OUTPUT = Path('reports/modeling/opponent_stat_impact/benchmark')
CUTOFF = pd.Timestamp('2026-11-01')
TEST_END = pd.Timestamp('2026-11-19')
TARGETS = ['MIN','PTS','REB','AST','FG3M','STL','BLK','TOV','DK_BONUS']
WEIGHTS = {'PTS':1,'REB':1.25,'AST':1.5,'FG3M':0.5,'STL':2,'BLK':2,'TOV':-0.5,'DK_BONUS':1}

def main():
    df = pd.concat([pd.read_csv(p, dtype={'GAME_ID':str,'PLAYER_ID':str}) for p in DATA], ignore_index=True)
    impact = pd.read_csv(IMPACT, dtype={'GAME_ID':str,'PLAYER_ID':str})
    cols = [c for c in impact if c.startswith('OPP_POS_')]
    if df.duplicated(['GAME_ID','PLAYER_ID']).any() or impact.duplicated(['GAME_ID','PLAYER_ID']).any():
        raise ValueError('Duplicate player-game keys')
    df = df.merge(impact[['GAME_ID','PLAYER_ID']+cols], on=['GAME_ID','PLAYER_ID'], how='left', validate='one_to_one', indicator=True)
    if not df['_merge'].eq('both').all():
        raise ValueError(f"Missing opponent features: {(df['_merge'] != 'both').sum()}")
    df.drop(columns='_merge', inplace=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    train = df.loc[df.GAME_DATE < CUTOFF].copy()
    test = df.loc[(df.GAME_DATE >= CUTOFF) & (df.GAME_DATE < TEST_END)].copy()
    if train.empty or test.empty:
        raise ValueError('Empty train or test data')
    adapter = get_model_adapter('catboost')
    # Explicitly remove the experimental columns before preparing the baseline.
    Xtrain, prep = adapter.prepare_training_features(train.drop(columns=cols))
    Xtest, _ = adapter.prepare_training_features(test.drop(columns=cols))
    Xtest = Xtest.reindex(columns=Xtrain.columns)
    if any(c in Xtrain for c in cols):
        raise ValueError('Opponent impact feature leaked into baseline')
    Xtrain_plus = Xtrain.join(train[cols].set_axis(Xtrain.index))
    Xtest_plus = Xtest.join(test[cols].set_axis(Xtest.index))
    for col in cols:
        Xtrain_plus[col] = pd.to_numeric(Xtrain_plus[col], errors='coerce').fillna(0.0)
        Xtest_plus[col] = pd.to_numeric(Xtest_plus[col], errors='coerce').fillna(0.0)
    print(f'Train {len(train):,}; test {len(test):,}; baseline features {Xtrain.shape[1]}; extra {len(cols)}', flush=True)
    print('Train dates:', train.GAME_DATE.min().date(), 'through', train.GAME_DATE.max().date(), flush=True)
    print('Test dates:', test.GAME_DATE.min().date(), 'through', test.GAME_DATE.max().date(), flush=True)
    output = test[['GAME_DATE','GAME_ID','PLAYER_ID','PLAYER_NAME','TEAM','DK_POINTS']].copy()
    rows = []
    for target in TARGETS:
        ytrain = pd.to_numeric(train[target], errors='coerce')
        ytest = pd.to_numeric(test[target], errors='coerce')
        oktrain, oktest = ytrain.notna(), ytest.notna()
        for label, tr, te in [('BASELINE',Xtrain,Xtest), ('OPP_IMPACT',Xtrain_plus,Xtest_plus)]:
            model = adapter.create_model()
            model.fit(tr.loc[oktrain], ytrain.loc[oktrain], cat_features=prep['categories'])
            pred = np.maximum(model.predict(te), 0)
            output[f'{label}_{target}'] = pred
            err = ytest.loc[oktest].to_numpy() - pred[oktest.to_numpy()]
            rows.append({'TARGET':target,'MODEL':label,'N':len(err),'MAE':np.abs(err).mean(),'RMSE':np.sqrt(np.mean(err**2)),'BIAS':err.mean()})
            print(f'{target:8} {label:10} MAE={np.abs(err).mean():.4f}', flush=True)
    actual = pd.to_numeric(output.DK_POINTS, errors='coerce')
    valid = actual.notna()
    for label in ('BASELINE','OPP_IMPACT'):
        pred = sum(output[f'{label}_{s}']*w for s,w in WEIGHTS.items()).clip(lower=0)
        output[f'{label}_DK'] = pred
        error = actual.loc[valid].to_numpy() - pred.loc[valid].to_numpy()
        rows.append({'TARGET':'DERIVED_DK','MODEL':label,'N':len(error),'MAE':np.abs(error).mean(),'RMSE':np.sqrt(np.mean(error**2)),'BIAS':error.mean(),
                     'BELOW_70_RATE':np.mean(actual.loc[valid].to_numpy() < .7*pred.loc[valid].to_numpy()),
                     'UNDER_10_RATE':np.mean(error < -10)})
    metrics = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(OUTPUT/'metrics.csv',index=False)
    output.to_csv(OUTPUT/'predictions.csv',index=False)
    print('\n=== COMPARISON ===')
    pivot = metrics.pivot(index='TARGET',columns='MODEL',values='MAE')
    pivot['DELTA_OPP_MINUS_BASE'] = pivot['OPP_IMPACT']-pivot['BASELINE']
    print(pivot.round(4).to_string())
    print('\n=== DK DOWNSIDE ===')
    print(metrics.loc[metrics.TARGET=='DERIVED_DK'].round(4).to_string(index=False))
    print('\nSaved:',OUTPUT)

if __name__=='__main__':
    main()
