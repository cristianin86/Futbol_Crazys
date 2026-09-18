import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from config.settings import resolve_path, RAW_DATA_DIR, PROCESSED_DATA_DIR
from processor import ROLLING_MAP
from scipy.stats import poisson

def dixon_coles_matrix(hg, ag, rho=-0.08, max_goals=11):
    p_A = np.array([poisson.pmf(i, hg) for i in range(max_goals)])
    p_B = np.array([poisson.pmf(j, ag) for j in range(max_goals)])
    matrix = np.outer(p_A, p_B)
    matrix[0, 0] *= 1.0 - (hg * ag * rho)
    matrix[0, 1] *= 1.0 + (hg * rho)
    matrix[1, 0] *= 1.0 + (ag * rho)
    matrix[1, 1] *= 1.0 - rho
    matrix = np.maximum(matrix, 0)
    matrix /= matrix.sum()
    return matrix

def matrix_probs(hg, ag):
    m = dixon_coles_matrix(hg, ag)
    return (float(np.sum(np.tril(m, -1))), float(np.sum(np.diag(m))), float(np.sum(np.triu(m, 1))))

def test_k_for_league(raw_csv, processed_csv, k_val, custom_prior=None):
    # Process
    in_path = resolve_path(raw_csv, RAW_DATA_DIR)
    df = pd.read_csv(in_path).sort_values('timestamp').reset_index(drop=True)
    for col in ['home_team_shots_on_target', 'away_team_shots_on_target',
                'home_team_yellow_cards', 'away_team_yellow_cards']:
        if col not in df.columns:
            df[col] = np.nan

    df['h_pts'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'],
                             df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)
    df['a_pts'] = np.select([df['away_team_goal_count'] > df['home_team_goal_count'],
                             df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)

    stats_cols = list(ROLLING_MAP.keys()) + ['ppg']
    for col in [f'l_{c}' for c in stats_cols] + [f'v_{c}' for c in stats_cols]:
        df[col] = 0.0
    df['l_gp'] = 0
    df['v_gp'] = 0

    history = {}
    league_accum = {
        'Local': {k: [0.0, 0] for k in list(ROLLING_MAP.keys()) + ['ppg']},
        'Visita': {k: [0.0, 0] for k in list(ROLLING_MAP.keys()) + ['ppg']}
    }
    
    prior_league_means = custom_prior or {
        'Local':  {'gf': 1.45, 'gc': 1.15, 'ppg': 1.60, 'pos': 51.0, 'xg': 1.35, 'yc': 2.30, 'sot': 4.50},
        'Visita': {'gf': 1.15, 'gc': 1.45, 'ppg': 1.15, 'pos': 49.0, 'xg': 1.15, 'yc': 2.50, 'sot': 3.80},
    }

    def get_league_mean(cond, key):
        acc = league_accum[cond][key]
        return (acc[0] / acc[1]) if acc[1] > 0 else prior_league_means[cond].get(key, 1.0)

    def snapshot(row, side):
        idx = 0 if side == 'home' else 1
        snap = {'pts': row['h_pts'] if side == 'home' else row['a_pts']}
        for key, cols in ROLLING_MAP.items():
            val = row[cols[idx]]
            snap[key] = float(val) if pd.notna(val) else np.nan
        return snap

    def fill_features(i, prefix, last_5, total_played, n_cond, cond):
        df.at[i, f'{prefix}_gp'] = total_played
        if not last_5:
            df.at[i, f'{prefix}_ppg'] = float(get_league_mean(cond, 'ppg'))
            for key in ROLLING_MAP:
                df.at[i, f'{prefix}_{key}'] = float(get_league_mean(cond, key))
            return

        w = float(n_cond / (n_cond + k_val))
        team_ppg = float(np.mean([m['pts'] for m in last_5]))
        mu_ppg = float(get_league_mean(cond, 'ppg'))
        df.at[i, f'{prefix}_ppg'] = float(w * team_ppg + (1.0 - w) * mu_ppg)

        for key in ROLLING_MAP:
            vals = [m[key] for m in last_5 if pd.notna(m[key])]
            mu_key = float(get_league_mean(cond, key))
            if vals:
                team_val = float(np.mean(vals))
                df.at[i, f'{prefix}_{key}'] = float(w * team_val + (1.0 - w) * mu_key)
            else:
                df.at[i, f'{prefix}_{key}'] = float(mu_key)

    for i, row in df.iterrows():
        h_team = row['home_team_name']
        a_team = row['away_team_name']
        if h_team not in history: history[h_team] = {'Local': [], 'Visita': []}
        if a_team not in history: history[a_team] = {'Local': [], 'Visita': []}
        h_hist = history[h_team]
        a_hist = history[a_team]

        n_h_cond = len(h_hist['Local'])
        last_5_h = h_hist['Local'][-5:] if h_hist['Local'] else (h_hist['Local'] + h_hist['Visita'])[-5:]
        fill_features(i, 'l', last_5_h, len(h_hist['Local']) + len(h_hist['Visita']), n_h_cond, 'Local')

        n_a_cond = len(a_hist['Visita'])
        last_5_a = a_hist['Visita'][-5:] if a_hist['Visita'] else (a_hist['Local'] + a_hist['Visita'])[-5:]
        fill_features(i, 'v', last_5_a, len(a_hist['Local']) + len(a_hist['Visita']), n_a_cond, 'Visita')

        es_jugado = pd.notna(row['home_team_goal_count']) and pd.notna(row['away_team_goal_count'])
        if str(row.get('status', 'complete')) == 'complete' and es_jugado:
            h_snap = snapshot(row, 'home')
            a_snap = snapshot(row, 'away')
            h_hist['Local'].append(h_snap)
            a_hist['Visita'].append(a_snap)
            league_accum['Local']['ppg'][0] += h_snap['pts']
            league_accum['Local']['ppg'][1] += 1
            league_accum['Visita']['ppg'][0] += a_snap['pts']
            league_accum['Visita']['ppg'][1] += 1
            for key in ROLLING_MAP:
                if pd.notna(h_snap[key]):
                    league_accum['Local'][key][0] += h_snap[key]
                    league_accum['Local'][key][1] += 1
                if pd.notna(a_snap[key]):
                    league_accum['Visita'][key][0] += a_snap[key]
                    league_accum['Visita'][key][1] += 1

    df = df.dropna(subset=['home_team_goal_count', 'away_team_goal_count'])
    if 'status' in df.columns:
        df = df[df['status'] == 'complete']
    df['target_1x2'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'],
                                  df['home_team_goal_count'] == df['away_team_goal_count']], [0, 1], 2)

    base_feats = ['gf', 'gc', 'ppg', 'pos', 'xg', 'yc', 'sot']
    features = [f'l_{c}' for c in base_feats] + [f'v_{c}' for c in base_feats]
    mask_historia = (df['l_gp'] >= 1) & (df['v_gp'] >= 1)
    df_agnostic = df[mask_historia][features + ['target_1x2']].copy().dropna()

    # Train / Val
    n_val = max(int(len(df_agnostic) * 0.2), 30)
    split = len(df_agnostic) - n_val
    X_tr, X_val = df_agnostic[features].iloc[:split], df_agnostic[features].iloc[split:]
    y_tr, y_val = df_agnostic['target_1x2'].iloc[:split], df_agnostic['target_1x2'].iloc[split:]
    
    # Train 1x2 with early stopping
    params = {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4,
              'learning_rate': 0.07, 'subsample': 0.9, 'colsample_bytree': 0.9,
              'eval_metric': 'mlogloss'}
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(params, dtr, num_boost_round=300, evals=[(dval, 'val')],
                        early_stopping_rounds=30, verbose_eval=False)
    best_n = booster.best_iteration + 1
    m_1x2 = xgb.train(params, xgb.DMatrix(df_agnostic[features], label=df_agnostic['target_1x2']), num_boost_round=best_n)

    probs_val = booster.predict(dval)
    # Calibrate T
    grid = np.linspace(0.6, 4.0, 69)
    def loss(T):
        p = np.power(np.clip(probs_val, 1e-12, 1.0), 1.0 / T)
        p = p / p.sum(axis=1, keepdims=True)
        idx = np.arange(len(y_val))
        return float(-np.mean(np.log(np.clip(p[idx, y_val], 1e-12, 1.0))))
    losses = [loss(t) for t in grid]
    temp = float(grid[int(np.argmin(losses))])

    p_cal = np.power(np.clip(probs_val, 1e-12, 1.0), 1.0 / temp)
    p_cal /= p_cal.sum(axis=1, keepdims=True)

    actual_home = float((y_val == 0).mean())
    pred_home_clf = float(p_cal[:, 0].mean())
    bias_clf = pred_home_clf - actual_home
    spread = float(p_cal[:, 0].max() - p_cal[:, 0].min())
    acc = float((probs_val.argmax(axis=1) == y_val).mean())

    return {
        'k': k_val,
        'temp': temp,
        'bias_clf': bias_clf,
        'pred_home_clf': pred_home_clf,
        'actual_home': actual_home,
        'spread': spread,
        'accuracy': acc,
        'rounds': best_n
    }

if __name__ == '__main__':
    print('Testing K on ESP:')
    for k in [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]:
        res = test_k_for_league('espana_api_raw.csv', 'espana_ml_ready.csv', k)
        b = res['bias_clf']
        t = res['temp']
        s = res['spread']
        a = res['accuracy']
        print(f'ESP K={k:.1f} -> Bias: {b:+.3f}, Temp: {t:.2f}, Spread: {s:.3f}, Acc: {a:.3f}')

    print('\nTesting K on ARG:')
    for k in [1.0, 2.0, 3.0, 5.0, 8.0]:
        res = test_k_for_league('argentina_api_raw.csv', 'argentina_ml_ready.csv', k)
        b = res['bias_clf']
        t = res['temp']
        s = res['spread']
        a = res['accuracy']
        print(f'ARG K={k:.1f} -> Bias: {b:+.3f}, Temp: {t:.2f}, Spread: {s:.3f}, Acc: {a:.3f}')
