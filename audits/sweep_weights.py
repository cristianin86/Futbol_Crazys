"""
audits/sweep_weights.py
=======================
Barrido cuantitativo de ponderaciones del blend (Clasificador 1X2 vs Matriz Poisson)
evaluado estrictamente sobre los modelos HELD-OUT (80% train, sin data leakage).
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

from audits.heldout_models import get_heldout_committee, LIGAS


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
    for suf in LIGAS.keys():
        comm = get_heldout_committee(suf)
        b_1x2 = comm['b_1x2']
        b_hg = comm['b_hg']
        b_ag = comm['b_ag']
        temp = comm['temp']
        X_val = comm['X_val']
        y_val = comm['y_val']
        actual_home = float((y_val == 0).mean())

        probs_cal = aplicar_temperatura(b_1x2.predict(xgb.DMatrix(X_val)), temp)
        hg_pred = np.maximum(b_hg.predict(xgb.DMatrix(X_val)), 0.05)
        ag_pred = np.maximum(b_ag.predict(xgb.DMatrix(X_val)), 0.05)
        p_matrix = np.array([matrix_probs(h, a) for h, a in zip(hg_pred, ag_pred)])

        val_data[suf] = {
            "actual_home": actual_home,
            "probs_cal": probs_cal,
            "p_matrix": p_matrix
        }

    combos = [
        (1.00, 0.00),
        (0.55, 0.45),
        (0.4167, 0.5833),
        (0.50, 0.50),
        (0.40, 0.60),
        (0.35, 0.65),
        (0.30, 0.70),
        (0.20, 0.80),
        (0.00, 1.00),
    ]

    header_ligas = "".join([f"{s:>8}" for s in LIGAS.keys()])
    print("=" * 90)
    print("BARRIDO DE PESOS CLF VS MATRIZ POISSON (EVALUACIÓN HELD-OUT PURA)")
    print("=" * 90)
    print(f"{'Combo (W_clf, W_mat)':<28}{header_ligas}{'Media |Bias|':>14}")
    print("-" * 90)

    for wc, wm in combos:
        biases = []
        if wc == 1.0:
            nombre = "Solo Clasificador (1.0, 0.0)"
        elif wc == 0.0:
            nombre = "Solo Matriz (0.0, 1.0)"
        elif abs(wc - 0.4167) < 0.001:
            nombre = "Fallback Prod (0.417, 0.583)"
        elif abs(wc - 0.55) < 0.001:
            nombre = "Blend Anterior (0.55, 0.45)"
        else:
            nombre = f"Blend ({wc:.2f}, {wm:.2f})"

        line_parts = [f"{nombre:<28}"]
        for suf in LIGAS.keys():
            d = val_data[suf]
            p_b = wc * d["probs_cal"] + wm * d["p_matrix"]
            p_b /= p_b.sum(axis=1, keepdims=True)
            pred_home = float(p_b[:, 0].mean())
            bias = pred_home - d["actual_home"]
            biases.append(abs(bias))
            line_parts.append(f"{bias:>+8.3f}")
        mean_abs_bias = float(np.mean(biases))
        line_parts.append(f"{mean_abs_bias:>14.4f}")
        print("".join(line_parts))


if __name__ == "__main__":
    run_sweep()
