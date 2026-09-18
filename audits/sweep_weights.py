import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models_saved"
PROCESSED_DIR = BASE / "data" / "processed"

LIGAS = {
    "CHI": "chile_ml_ready_v8.csv",
    "B":   "chile_b_ml_ready.csv",
    "ENG": "premier_ml_ready_v1.csv",
    "ARG": "argentina_ml_ready.csv",
    "PER": "peru_ml_ready.csv",
    "ESP": "espana_ml_ready.csv",
}

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

def aplicar_temperatura(probs, T):
    p = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / T)
    return p / p.sum(axis=1, keepdims=True)

def run_sweep():
    val_data = {}
    for suf, dataset in LIGAS.items():
        meta = json.loads((MODELS_DIR / f"metadata_{suf}.json").read_text(encoding="utf-8"))
        features = meta["features"]
        temp = meta["models"]["1x2"]["temperature"]
        df = pd.read_csv(PROCESSED_DIR / dataset).sort_values("timestamp").reset_index(drop=True)
        n_val = max(int(len(df) * 0.2), 30)
        df_val = df.iloc[len(df)-n_val:].copy()
        X_val = df_val[features].astype(float)
        y_val = df_val["target_1x2"].astype(int).values
        actual_home = float((y_val == 0).mean())

        b_1x2 = xgb.Booster(); b_1x2.load_model(str(MODELS_DIR / f"model_1x2_v5_{suf}.json"))
        probs_cal = aplicar_temperatura(b_1x2.predict(xgb.DMatrix(X_val)), temp)

        b_hg = xgb.Booster(); b_hg.load_model(str(MODELS_DIR / f"model_hg_v5_{suf}.json"))
        b_ag = xgb.Booster(); b_ag.load_model(str(MODELS_DIR / f"model_ag_v5_{suf}.json"))
        hg_pred = np.maximum(b_hg.predict(xgb.DMatrix(X_val)), 0.05)
        ag_pred = np.maximum(b_ag.predict(xgb.DMatrix(X_val)), 0.05)
        p_matrix = np.array([matrix_probs(h, a) for h, a in zip(hg_pred, ag_pred)])

        val_data[suf] = {
            "actual_home": actual_home,
            "probs_cal": probs_cal,
            "p_matrix": p_matrix
        }

    combos = [
        (0.55, 0.45),
        (0.50, 0.50),
        (0.40, 0.60),
        (0.35, 0.65),
        (0.30, 0.70),
        (0.20, 0.80),
        (0.00, 1.00),
    ]

    header_ligas = "".join([f"{s:>8}" for s in LIGAS.keys()])
    print(f"{'Combo (W_clf, W_mat)':<25}{header_ligas}{'Mean|Bias|':>12}")
    print("-" * 85)

    for wc, wm in combos:
        biases = []
        line = f"({wc:.2f}, {wm:.2f})"
        line_parts = [f"{line:<25}"]
        for suf in LIGAS.keys():
            d = val_data[suf]
            p_b = wc * d["probs_cal"] + wm * d["p_matrix"]
            p_b /= p_b.sum(axis=1, keepdims=True)
            pred_home = float(p_b[:, 0].mean())
            bias = pred_home - d["actual_home"]
            biases.append(abs(bias))
            line_parts.append(f"{bias:>+8.3f}")
        line_parts.append(f"{np.mean(biases):>12.4f}")
        print("".join(line_parts))

if __name__ == "__main__":
    run_sweep()
