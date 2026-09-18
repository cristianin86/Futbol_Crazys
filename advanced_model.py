"""
advanced_model.py
=================
Entrenamiento y sincronización del comité XGBoost (V10) para una liga dada:
1. 1X2 (multi:softprob calibrado por temperatura)
2. Goles Local / 3. Goles Visitante (count:poisson)
4. Córners Local / 5. Córners Visitante (count:poisson)
6. Tarjetas Local / 7. Tarjetas Visitante (count:poisson, solo con datos reales)

V10:
- Split temporal (80/20) con early stopping: las métricas reportadas son out-of-sample.
- Pesos de decaimiento exponencial por fecha real (xi=0.0035/día) si hay timestamp.
- Calibración de temperatura del 1X2 sobre el set de validación.
- Guardia de muestras mínimas (no se entrena con datasets raquíticos).
- metadata_<SUF>.json con features, medias (para imputación en inferencia) y métricas.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

try:
    from config.settings import resolve_path, PROCESSED_DATA_DIR, MODELS_DIR
except ImportError:
    BASE = Path(__file__).resolve().parent
    PROCESSED_DATA_DIR = BASE / "data" / "processed"
    MODELS_DIR = BASE / "models_saved"
    def resolve_path(f, d=None):
        if d and (d / f).exists(): return d / f
        if (BASE / f).exists(): return BASE / f
        return (d / f) if d else (BASE / f)

MIN_SAMPLES = 100
SEED = 42  # Semilla fija para reproducibilidad determinista estricta en entrenamientos
BASE_FEATS = ['gf', 'gc', 'ppg', 'pos', 'xg', 'yc', 'sot']
CONTEXT_FEATS = ['is_cup', 'home_team_lsi', 'away_team_lsi']


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
    """Ranked Probability Score promedio (0 = perfecto)."""
    total = 0.0
    for p, y in zip(probs, y_true):
        e = np.zeros(3); e[y] = 1.0
        total += 0.5 * np.sum((np.cumsum(p)[:2] - np.cumsum(e)[:2]) ** 2)
    return float(total / max(len(y_true), 1))


def _calibrate_temperature(probs, y_true):
    """Busca T que minimiza el log loss: p_cal ∝ p^(1/T)."""
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


def _refit_full(params, X, y, w, n_rounds):
    return xgb.train(params, xgb.DMatrix(X, label=y, weight=w), num_boost_round=n_rounds)


def entrenar_comite_v5(dataset="chile_ml_ready_v8.csv", suffix=""):
    """Entrena el comité de modelos XGBoost a partir de un dataset procesado."""
    print(f"--- Entrenando Comite V10 (Dataset: {dataset}, Sufijo: '{suffix}') ---")

    ds_path = resolve_path(dataset, PROCESSED_DATA_DIR)
    if not ds_path.exists():
        print(f"Error: Falta el archivo {dataset} (buscado en {ds_path}). Ejecuta processor.py primero.")
        return False

    df = pd.read_csv(ds_path, encoding='utf-8', encoding_errors='replace')
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    features = [f'{p}_{c}' for p in ('l', 'v') for c in BASE_FEATS if f'{p}_{c}' in df.columns]
    features += [c for c in CONTEXT_FEATS if c in df.columns]

    if len(df) < MIN_SAMPLES:
        print(f"ABORTADO: solo {len(df)} muestras (mínimo {MIN_SAMPLES}). "
              f"Un modelo entrenado así produce predicciones inservibles. "
              f"Descarga más historial antes de entrenar '{suffix}'.")
        return False

    X = df[features].astype(float)
    pesos = _time_weights(df)

    # Split temporal 80/20 (el dataset ya viene ordenado cronológicamente)
    n_val = max(int(len(df) * 0.2), 30)
    split = len(df) - n_val
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    w_tr = pesos[:split]

    s = f"_{suffix}" if suffix else ""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        'version': 'V10',
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'dataset': dataset,
        'n_samples': int(len(df)),
        'n_val': int(n_val),
        'features': features,
        'feature_means': {c: float(X[c].mean()) for c in features},
        'pct_stats_reales': float(df['stats_real'].mean()) if 'stats_real' in df.columns else None,
        'models': {},
    }

    # Regularizacion fuerte para el 1X2: con depth=4 sin L2 el booster salia
    # sobreconfiado y la calibracion por temperatura tenia que aplanarlo mucho
    # (ej. Argentina necesitaba T~4.0, comprimiendo las probabilidades a una
    # banda casi uniforme). Con menos profundidad y mas L2/min_child_weight el
    # modelo llega mejor calibrado de fabrica: menor log-loss y mejor accuracy
    # de validacion en las 6 ligas, con una T final mucho mas baja.
    params_multi = {'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 2,
                    'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'reg_lambda': 5.0, 'min_child_weight': 10,
                    'eval_metric': 'mlogloss', 'random_state': SEED, 'seed': SEED}
    params_poisson = {'objective': 'count:poisson', 'max_depth': 3, 'learning_rate': 0.05,
                      'subsample': 0.9, 'eval_metric': 'poisson-nloglik',
                      'random_state': SEED, 'seed': SEED}

    # ------------------ 1X2 ------------------
    print("1/7: Entrenando 1X2 (con validación temporal)...")
    y = df['target_1x2'].astype(int)
    b_val, n_best = _train_with_es(params_multi, X_tr, y.iloc[:split], w_tr, X_val, y.iloc[split:])
    probs_val = b_val.predict(xgb.DMatrix(X_val))
    y_val = y.iloc[split:].values
    temp = _calibrate_temperature(probs_val, y_val)
    p_cal = np.power(np.clip(probs_val, 1e-12, 1.0), 1.0 / temp)
    p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)

    metadata['models']['1x2'] = {
        'rounds': n_best,
        'temperature': temp,
        'val_log_loss': _log_loss_multi(y_val, probs_val),
        'val_log_loss_calibrado': _log_loss_multi(y_val, p_cal),
        'val_accuracy': float((probs_val.argmax(axis=1) == y_val).mean()),
        'val_rps': _rps_mean(p_cal, y_val),
    }
    m_1x2 = _refit_full(params_multi, X, y, pesos, n_best)
    m_1x2.save_model(str(MODELS_DIR / f"model_1x2_v5{s}.json"))
    print(f"   LogLoss val: {metadata['models']['1x2']['val_log_loss']:.4f} | "
          f"Acc: {metadata['models']['1x2']['val_accuracy']:.3f} | "
          f"RPS: {metadata['models']['1x2']['val_rps']:.4f} | T={temp:.2f}")

    # ------------------ Modelos Poisson ------------------
    def entrenar_poisson(nombre, target_col, filename, requiere_varianza=False):
        if target_col not in df.columns:
            print(f"   Skipping {nombre}: falta columna {target_col}.")
            return
        y_p = df[target_col].astype(float)
        if requiere_varianza and y_p.std() < 0.35:
            print(f"   Skipping {nombre}: varianza insuficiente en {target_col} "
                  f"(std={y_p.std():.2f}) — probablemente datos sintéticos.")
            return
        b, n = _train_with_es(params_poisson, X_tr, y_p.iloc[:split], w_tr, X_val, y_p.iloc[split:])
        pred_val = b.predict(xgb.DMatrix(X_val))
        mae = float(np.mean(np.abs(pred_val - y_p.iloc[split:].values)))
        metadata['models'][nombre] = {'rounds': n, 'val_mae': mae}
        m = _refit_full(params_poisson, X, y_p, pesos, n)
        m.save_model(str(MODELS_DIR / filename))
        print(f"   {nombre}: MAE val = {mae:.3f} ({n} rondas)")

    print("2/7: Goles Local...");      entrenar_poisson('hg', 'home_team_goal_count', f"model_hg_v5{s}.json")
    print("3/7: Goles Visitante...");  entrenar_poisson('ag', 'away_team_goal_count', f"model_ag_v5{s}.json")
    print("4/7: Córners Local...");    entrenar_poisson('hc', 'home_team_corner_count', f"model_hc_v5{s}.json", requiere_varianza=True)
    print("5/7: Córners Visitante..."); entrenar_poisson('ac', 'away_team_corner_count', f"model_ac_v5{s}.json", requiere_varianza=True)
    print("6/7: Tarjetas Local...");   entrenar_poisson('hy', 'home_team_yellow_cards', f"model_hy_v5{s}.json", requiere_varianza=True)
    print("7/7: Tarjetas Visitante..."); entrenar_poisson('ay', 'away_team_yellow_cards', f"model_ay_v5{s}.json", requiere_varianza=True)

    with open(MODELS_DIR / f"metadata{s}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"COMPLETO: Comité {s or '(base)'} sincronizado en {MODELS_DIR} (metadata{s}.json incluida).")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2:
        entrenar_comite_v5(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        entrenar_comite_v5(sys.argv[1])
    else:
        entrenar_comite_v5()
