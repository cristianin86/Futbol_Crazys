"""
audits/heldout_models.py
========================
Módulo reutilizable para entrenamiento y carga de modelos XGBoost genuinamente HELD-OUT:
- Entrena sobre el 80% inicial (train split) con early stopping sobre el 20% final (val split).
- NUNCA ejecuta _refit_full sobre el 100% de los datos.
- Fija semilla SEED = 42 para reproducibilidad estricta.
- Guarda los artefactos en `models_saved/heldout/` con sufijo `_heldout.json`.
- Diseñado para responder a la Ronda 3 de ANTIGRAVITY_COLAB.md para eliminar
  completamente la contaminación por data leakage en auditorías.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE / "data" / "processed"
HELDOUT_MODELS_DIR = BASE / "models_saved" / "heldout"

SEED = 42
MIN_SAMPLES = 100
BASE_FEATS = ['gf', 'gc', 'ppg', 'pos', 'xg', 'yc', 'sot']
CONTEXT_FEATS = ['is_cup', 'home_team_lsi', 'away_team_lsi']

LIGAS = {
    "CHI": "chile_ml_ready_v8.csv",
    "B":   "chile_b_ml_ready.csv",
    "ENG": "premier_ml_ready_v1.csv",
    "ARG": "argentina_ml_ready.csv",
    "PER": "peru_ml_ready.csv",
    "ESP": "espana_ml_ready.csv",
}


def _time_weights(df, xi=0.0035):
    """Decaimiento exponencial por días reales; fallback lineal por posición."""
    if 'timestamp' in df.columns and df['timestamp'].notna().all():
        ts = pd.to_numeric(df['timestamp'], errors='coerce').values.astype(float)
        if np.isfinite(ts).all() and ts.max() > 1e8:
            days = (ts.max() - ts) / 86400.0
            return np.exp(-xi * np.maximum(days, 0))
    return np.linspace(0.5, 1.0, len(df))


def _log_loss_multi(y_true, probs, eps=1e-12):
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))


def _rps_mean(probs, y_true):
    total = 0.0
    for p, y in zip(probs, y_true):
        e = np.zeros(3); e[y] = 1.0
        total += 0.5 * np.sum((np.cumsum(p)[:2] - np.cumsum(e)[:2]) ** 2)
    return float(total / max(len(y_true), 1))


def _calibrate_temperature(probs, y_true):
    def loss(T):
        p = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / T)
        p = p / p.sum(axis=1, keepdims=True)
        return _log_loss_multi(y_true, p)
    grid = np.linspace(0.6, 4.0, 69)
    losses = [loss(t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def _train_with_es(params, X_tr, y_tr, w_tr, X_val, y_val, num_rounds=300, es_rounds=30):
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(params, dtr, num_boost_round=num_rounds,
                        evals=[(dval, 'val')], early_stopping_rounds=es_rounds,
                        verbose_eval=False)
    best_n = getattr(booster, 'best_iteration', None)
    return booster, (best_n + 1 if best_n is not None else num_rounds)


def train_heldout_committee(suf):
    """Entrena y persiste el comité de modelos HELD-OUT (solo 80% train) para una liga."""
    if suf not in LIGAS:
        raise ValueError(f"Liga desconocida: '{suf}'. Opciones: {list(LIGAS.keys())}")

    dataset = LIGAS[suf]
    ds_path = PROCESSED_DATA_DIR / dataset
    if not ds_path.exists():
        raise FileNotFoundError(f"No existe el archivo {ds_path}")

    df = pd.read_csv(ds_path, encoding='utf-8', encoding_errors='replace')
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    features = [f'{p}_{c}' for p in ('l', 'v') for c in BASE_FEATS if f'{p}_{c}' in df.columns]
    features += [c for c in CONTEXT_FEATS if c in df.columns]

    if len(df) < MIN_SAMPLES:
        raise ValueError(f"Insuficientes muestras en {dataset} ({len(df)} < {MIN_SAMPLES})")

    X = df[features].astype(float)
    pesos = _time_weights(df)

    n_val = max(int(len(df) * 0.2), 30)
    split = len(df) - n_val
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    w_tr = pesos[:split]
    df_val = df.iloc[split:].copy()
    df_tr = df.iloc[:split].copy()

    HELDOUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_multi = {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4,
                    'learning_rate': 0.07, 'subsample': 0.9, 'colsample_bytree': 0.9,
                    'eval_metric': 'mlogloss', 'random_state': SEED, 'seed': SEED}
    params_poisson = {'objective': 'count:poisson', 'max_depth': 3, 'learning_rate': 0.05,
                      'subsample': 0.9, 'eval_metric': 'poisson-nloglik',
                      'random_state': SEED, 'seed': SEED}

    # 1. 1X2 (Held-Out)
    y = df['target_1x2'].astype(int)
    y_tr, y_val = y.iloc[:split], y.iloc[split:]
    b_1x2, n_best_1x2 = _train_with_es(params_multi, X_tr, y_tr, w_tr, X_val, y_val)
    probs_val = b_1x2.predict(xgb.DMatrix(X_val))
    y_val_arr = y_val.values
    temp = _calibrate_temperature(probs_val, y_val_arr)
    p_cal = np.power(np.clip(probs_val, 1e-12, 1.0), 1.0 / temp)
    p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)

    val_acc = float((probs_val.argmax(axis=1) == y_val_arr).mean())
    val_loss = _log_loss_multi(y_val_arr, probs_val)
    val_loss_cal = _log_loss_multi(y_val_arr, p_cal)
    val_rps = _rps_mean(p_cal, y_val_arr)

    b_1x2.save_model(str(HELDOUT_MODELS_DIR / f"model_1x2_v5_{suf}_heldout.json"))

    # 2. HG (Goles Local Held-Out)
    y_hg = df['home_team_goal_count'].astype(float)
    b_hg, n_best_hg = _train_with_es(params_poisson, X_tr, y_hg.iloc[:split], w_tr, X_val, y_hg.iloc[split:])
    pred_val_hg = b_hg.predict(xgb.DMatrix(X_val))
    mae_hg = float(np.mean(np.abs(pred_val_hg - y_hg.iloc[split:].values)))
    b_hg.save_model(str(HELDOUT_MODELS_DIR / f"model_hg_v5_{suf}_heldout.json"))

    # 3. AG (Goles Visitante Held-Out)
    y_ag = df['away_team_goal_count'].astype(float)
    b_ag, n_best_ag = _train_with_es(params_poisson, X_tr, y_ag.iloc[:split], w_tr, X_val, y_ag.iloc[split:])
    pred_val_ag = b_ag.predict(xgb.DMatrix(X_val))
    mae_ag = float(np.mean(np.abs(pred_val_ag - y_ag.iloc[split:].values)))
    b_ag.save_model(str(HELDOUT_MODELS_DIR / f"model_ag_v5_{suf}_heldout.json"))

    metadata = {
        'version': 'V10_HELDOUT_AUDIT',
        'is_heldout_only': True,
        'seed': SEED,
        'liga': suf,
        'dataset': dataset,
        'n_total': int(len(df)),
        'n_train': int(split),
        'n_val': int(n_val),
        'features': features,
        'feature_means': {c: float(X[c].mean()) for c in features},
        'train_feature_means': {c: float(X_tr[c].mean()) for c in features},
        'models': {
            '1x2': {
                'rounds': n_best_1x2,
                'temperature': temp,
                'val_log_loss': val_loss,
                'val_log_loss_calibrado': val_loss_cal,
                'val_accuracy': val_acc,
                'val_rps': val_rps,
            },
            'hg': {'rounds': n_best_hg, 'val_mae': mae_hg},
            'ag': {'rounds': n_best_ag, 'val_mae': mae_ag},
        }
    }

    meta_path = HELDOUT_MODELS_DIR / f"metadata_{suf}_heldout.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        'b_1x2': b_1x2,
        'b_hg': b_hg,
        'b_ag': b_ag,
        'temp': temp,
        'metadata': metadata,
        'df_tr': df_tr,
        'df_val': df_val,
        'X_tr': X_tr,
        'X_val': X_val,
        'y_val': y_val_arr,
    }


def get_heldout_committee(suf, force_retrain=False):
    """Obtiene el comité held-out para `suf`. Si ya existe en disco y no se fuerza reentreno, lo carga."""
    m1x2_path = HELDOUT_MODELS_DIR / f"model_1x2_v5_{suf}_heldout.json"
    mhg_path = HELDOUT_MODELS_DIR / f"model_hg_v5_{suf}_heldout.json"
    mag_path = HELDOUT_MODELS_DIR / f"model_ag_v5_{suf}_heldout.json"
    meta_path = HELDOUT_MODELS_DIR / f"metadata_{suf}_heldout.json"

    if not force_retrain and m1x2_path.exists() and mhg_path.exists() and mag_path.exists() and meta_path.exists():
        b_1x2 = xgb.Booster(); b_1x2.load_model(str(m1x2_path))
        b_hg = xgb.Booster(); b_hg.load_model(str(mhg_path))
        b_ag = xgb.Booster(); b_ag.load_model(str(mag_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        temp = meta['models']['1x2']['temperature']

        # Cargar dataframe para split
        dataset = LIGAS[suf]
        df = pd.read_csv(PROCESSED_DATA_DIR / dataset, encoding='utf-8', encoding_errors='replace')
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        features = meta['features']
        n_val = meta['n_val']
        split = len(df) - n_val
        df_tr = df.iloc[:split].copy()
        df_val = df.iloc[split:].copy()
        X_tr = df_tr[features].astype(float)
        X_val = df_val[features].astype(float)
        y_val = df_val['target_1x2'].astype(int).values

        return {
            'b_1x2': b_1x2,
            'b_hg': b_hg,
            'b_ag': b_ag,
            'temp': temp,
            'metadata': meta,
            'df_tr': df_tr,
            'df_val': df_val,
            'X_tr': X_tr,
            'X_val': X_val,
            'y_val': y_val,
        }

    return train_heldout_committee(suf)


def train_all_heldout():
    print("=" * 80)
    print("ENTRENANDO COMITÉS HELD-OUT (80% TRAIN PURO, SIN REFIT)")
    print(f"Directorio de guardado: {HELDOUT_MODELS_DIR}")
    print(f"Semilla fija: {SEED}")
    print("=" * 80)

    summary = []
    for suf in LIGAS.keys():
        print(f"\n--- Entrenando Held-Out para {suf} ---")
        comm = train_heldout_committee(suf)
        m1x2 = comm['metadata']['models']['1x2']
        summary.append({
            'liga': suf,
            'n_tr': comm['metadata']['n_train'],
            'n_val': comm['metadata']['n_val'],
            'acc': m1x2['val_accuracy'],
            'log_loss': m1x2['val_log_loss'],
            't': m1x2['temperature'],
            'hg_mae': comm['metadata']['models']['hg']['val_mae'],
            'ag_mae': comm['metadata']['models']['ag']['val_mae'],
        })
        print(f"   [OK] {suf}: Val Acc = {m1x2['val_accuracy']:.3f} | LogLoss = {m1x2['val_log_loss']:.4f} | T = {m1x2['temperature']:.2f}")

    print("\n" + "=" * 80)
    print(f"{'Liga':<6}{'n_tr':>7}{'n_val':>7}{'Val Acc':>12}{'Val LogLoss':>14}{'T':>8}{'HG MAE':>10}{'AG MAE':>10}")
    print("=" * 80)
    for r in summary:
        print(f"{r['liga']:<6}{r['n_tr']:>7}{r['n_val']:>7}{r['acc']:>12.3f}{r['log_loss']:>14.4f}{r['t']:>8.2f}{r['hg_mae']:>10.3f}{r['ag_mae']:>10.3f}")


if __name__ == "__main__":
    train_all_heldout()
