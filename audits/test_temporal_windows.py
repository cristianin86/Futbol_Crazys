"""
audits/test_temporal_windows.py
===============================
Evaluación cuantitativa del blend (Clasificador 1X2 vs Matriz Poisson/Dixon-Coles)
a través de 3 ventanas temporales independientes evaluado sobre modelos HELD-OUT:
- Ventana 1 (Reciente): 80% al 100% (split de validación genuinamente held-out).
- Ventana 2 (Media): 60% al 80%.
- Ventana 3 (Antigua): 40% al 60%.

Responde a la Ronda 3 de ANTIGRAVITY_COLAB.md:
Garantiza que la evaluación multiventana utiliza el modelo held-out (entrenado solo
al 80% con seed=42) para que la ventana reciente 80-100% esté 100% libre de contaminación.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from audits.heldout_models import HELDOUT_MODELS_DIR, get_heldout_committee, LIGAS

PROCESSED_DIR = BASE / "data" / "processed"


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


def auditar_ventanas_temporales():
    print("=" * 95)
    print("EVALUACIÓN TEMPORAL MULTI-VENTANA: SESGO CLF VS MATRIZ POISSON (MODELOS HELD-OUT)")
    print("=" * 95)

    combos = [
        ("Clasificador Puro (1.0, 0.0)", 1.00, 0.00),
        ("Fallback Prod Ronda 4 (0.55, 0.45)",  0.55, 0.45),
        ("Blend Ronda 2/3, obsoleto (0.417, 0.583)", 0.4167, 0.5833),
        ("Matriz Ponderada (0.30, 0.70)", 0.30, 0.70),
        ("Matriz Pura (0.0, 1.0)",        0.00, 1.00),
    ]

    ventanas = [
        ("Ventana 1: 80%-100% (Genuinamente Held-Out)", 0.80, 1.00),
        ("Ventana 2: 60%-80% (Media)",                  0.60, 0.80),
        ("Ventana 3: 40%-60% (Antigua)",                0.40, 0.60),
    ]

    # Pre-cargar modelos held-out
    heldout_comms = {suf: get_heldout_committee(suf) for suf in LIGAS.keys()}

    resultados_ventanas = {}

    for v_nombre, p_start, p_end in ventanas:
        print(f"\n>>> {v_nombre}")
        val_data = {}
        for suf, dataset in LIGAS.items():
            comm = heldout_comms[suf]
            features = comm['metadata']["features"]
            temp = comm['temp']
            df = pd.read_csv(PROCESSED_DIR / dataset).sort_values("timestamp").reset_index(drop=True)

            i_start = int(len(df) * p_start)
            i_end = int(len(df) * p_end)
            df_sub = df.iloc[i_start:i_end].copy()

            for f in features:
                if f not in df_sub.columns:
                    df_sub[f] = comm['metadata']["feature_means"].get(f, 0.0)
            X_sub = df_sub[features].astype(float)
            y_sub = df_sub["target_1x2"].astype(int).values
            actual_home = float((y_sub == 0).mean())

            probs_cal = aplicar_temperatura(comm['b_1x2'].predict(xgb.DMatrix(X_sub)), temp)

            hg_pred = np.maximum(comm['b_hg'].predict(xgb.DMatrix(X_sub)), 0.05)
            ag_pred = np.maximum(comm['b_ag'].predict(xgb.DMatrix(X_sub)), 0.05)
            p_matrix = np.array([matrix_probs(h, a) for h, a in zip(hg_pred, ag_pred)])

            val_data[suf] = {
                "n": len(df_sub),
                "actual_home": actual_home,
                "probs_cal": probs_cal,
                "p_matrix": p_matrix
            }

        header_ligas = "".join([f"{s:>8}" for s in LIGAS.keys()])
        print(f"{'Configuración Blend':<32}{header_ligas}{'Media |Bias|':>14}")
        print("-" * 95)

        res_v = []
        for c_nombre, wc, wm in combos:
            biases = []
            parts = [f"{c_nombre:<32}"]
            for suf in LIGAS.keys():
                d = val_data[suf]
                p_b = wc * d["probs_cal"] + wm * d["p_matrix"]
                p_b /= p_b.sum(axis=1, keepdims=True)
                pred_home = float(p_b[:, 0].mean())
                bias = pred_home - d["actual_home"]
                biases.append(abs(bias))
                parts.append(f"{bias:>+8.3f}")
            mean_abs_bias = float(np.mean(biases))
            parts.append(f"{mean_abs_bias:>14.4f}")
            print("".join(parts))
            res_v.append({"config": c_nombre, "wc": wc, "wm": wm, "mean_abs_bias": mean_abs_bias})
        resultados_ventanas[v_nombre] = res_v

    return resultados_ventanas


if __name__ == "__main__":
    auditar_ventanas_temporales()
