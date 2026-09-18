"""
Auditoria cuantitativa de sesgo de localia (home advantage) en el pipeline V10.

Ronda 3 (Separacion estricta Held-Out vs Produccion):
- Evalua de forma primaria los modelos HELD-OUT (entrenados estrictamente sobre
  el 80% de train, SIN el paso _refit_full sobre el 100%), garantizando que las
  metricas out-of-sample son genuinas y libres de data leakage / memorizacion.
- De forma secundaria y con fines diagnosticos comparativos, evalua el modelo
  guardado de PRODUCCION (refit 100%) sobre esas mismas filas para documentar
  la brecha de memorizacion.

Uso:
    python audits/audit_home_advantage.py
"""
import sys
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from audits.heldout_models import get_heldout_committee, HELDOUT_MODELS_DIR, SEED

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
MODELS_DIR = BASE / "models_saved"
PROCESSED_DIR = BASE / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent

LIGAS = {
    "CHI": "chile_ml_ready_v8.csv",
    "B":   "chile_b_ml_ready.csv",
    "ENG": "premier_ml_ready_v1.csv",
    "ARG": "argentina_ml_ready.csv",
    "PER": "peru_ml_ready.csv",
    "ESP": "espana_ml_ready.csv",
}


def aplicar_temperatura(probs, T):
    p = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / T)
    return p / p.sum(axis=1, keepdims=True)


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


def run_audit():
    results = []

    for suf, dataset in LIGAS.items():
        # 1. Obtener comite genuinamente HELD-OUT (solo 80% train, sin refit)
        heldout = get_heldout_committee(suf)
        b_1x2_heldout = heldout['b_1x2']
        b_hg_heldout = heldout['b_hg']
        b_ag_heldout = heldout['b_ag']
        meta_heldout = heldout['metadata']
        temp_heldout = heldout['temp']
        features = meta_heldout["features"]
        df_val = heldout['df_val']
        X_val = heldout['X_val']
        y_val = heldout['y_val']

        actual_home = float((y_val == 0).mean())
        actual_draw = float((y_val == 1).mean())
        actual_away = float((y_val == 2).mean())

        # Predicciones 1X2 Held-Out (Genuinas Out-of-Sample)
        probs_raw_h = b_1x2_heldout.predict(xgb.DMatrix(X_val))
        probs_cal_h = aplicar_temperatura(probs_raw_h, temp_heldout)
        heldout_acc = float((probs_cal_h.argmax(axis=1) == y_val).mean())

        pred_home_clf_h = float(probs_cal_h[:, 0].mean())
        pred_draw_clf_h = float(probs_cal_h[:, 1].mean())
        pred_away_clf_h = float(probs_cal_h[:, 2].mean())
        bias_home_clf_h = pred_home_clf_h - actual_home

        # Goles Poisson Held-Out
        hg_pred_h = np.maximum(b_hg_heldout.predict(xgb.DMatrix(X_val)), 0.05)
        ag_pred_h = np.maximum(b_ag_heldout.predict(xgb.DMatrix(X_val)), 0.05)
        p_matrix_h = np.array([matrix_probs(h, a) for h, a in zip(hg_pred_h, ag_pred_h)])
        pred_home_matrix_only_h = float(p_matrix_h[:, 0].mean())
        bias_home_matrix_h = pred_home_matrix_only_h - actual_home

        # Blend anterior (55% clf + 45% matrix) con componentes held-out
        p_blend_h = 0.55 * probs_cal_h + 0.45 * p_matrix_h
        p_blend_h = p_blend_h / p_blend_h.sum(axis=1, keepdims=True)
        pred_home_blend_h = float(p_blend_h[:, 0].mean())
        bias_home_blend_h = pred_home_blend_h - actual_home

        # Blend fallback actual (41.7% clf + 58.3% matrix) con componentes held-out
        p_blend_fb_h = (0.25 / 0.60) * probs_cal_h + (0.35 / 0.60) * p_matrix_h
        p_blend_fb_h = p_blend_fb_h / p_blend_fb_h.sum(axis=1, keepdims=True)
        pred_home_blend_fb_h = float(p_blend_fb_h[:, 0].mean())
        bias_home_blend_fb_h = pred_home_blend_fb_h - actual_home

        # 2. Diagnóstico comparativo: Modelo de PRODUCCIÓN (refit 100% sobre todo el dataset)
        m1x2_prod_path = MODELS_DIR / f"model_1x2_v5_{suf}.json"
        meta_prod_path = MODELS_DIR / f"metadata_{suf}.json"
        prod_acc = None
        prod_bias_clf = None
        prod_pred_home_clf = None
        if m1x2_prod_path.exists() and meta_prod_path.exists():
            meta_prod = json.loads(meta_prod_path.read_text(encoding="utf-8"))
            t_prod = meta_prod["models"]["1x2"]["temperature"]
            b_prod = xgb.Booster(); b_prod.load_model(str(m1x2_prod_path))
            probs_prod_raw = b_prod.predict(xgb.DMatrix(X_val))
            probs_prod_cal = aplicar_temperatura(probs_prod_raw, t_prod)
            prod_acc = float((probs_prod_cal.argmax(axis=1) == y_val).mean())
            prod_pred_home_clf = float(probs_prod_cal[:, 0].mean())
            prod_bias_clf = prod_pred_home_clf - actual_home

        # Tasa historica completa (para contexto)
        df_full = pd.read_csv(PROCESSED_DIR / dataset)
        full_home_rate = float((df_full["target_1x2"] == 0).mean())

        # Reliability diagram clase Home sobre el CLASIFICADOR HELD-OUT calibrado
        p_home = probs_cal_h[:, 0]
        order = np.argsort(p_home)
        bins = np.array_split(order, min(5, max(1, len(order) // 15))) if len(order) >= 15 else [order]
        reliab_heldout = []
        for b in bins:
            if len(b) == 0:
                continue
            reliab_heldout.append((float(p_home[b].mean()), float((y_val[b] == 0).mean()), len(b)))

        results.append({
            "liga": suf,
            "n_val": len(df_val),
            "n_train": len(heldout['df_tr']),
            "n_total": len(df_full),
            "home_rate_val": actual_home,
            "home_rate_full_hist": full_home_rate,
            "draw_rate_val": actual_draw,
            "away_rate_val": actual_away,
            # Métricas Held-Out (Genuinas Out-of-Sample)
            "heldout_acc": heldout_acc,
            "heldout_temp": temp_heldout,
            "pred_home_clf_heldout": pred_home_clf_h,
            "bias_home_clf_heldout": bias_home_clf_h,
            "pred_home_blend_heldout": pred_home_blend_h,
            "bias_home_blend_heldout": bias_home_blend_h,
            "pred_home_blend_fb_heldout": pred_home_blend_fb_h,
            "bias_home_blend_fb_heldout": bias_home_blend_fb_h,
            "pred_home_matrix_heldout": pred_home_matrix_only_h,
            "bias_home_matrix_heldout": bias_home_matrix_h,
            "hg_mean_pred_heldout": float(hg_pred_h.mean()),
            "ag_mean_pred_heldout": float(ag_pred_h.mean()),
            "reliability_home_heldout": reliab_heldout,
            # Métricas Modelo Producción Refit-100% (Contaminado sobre val)
            "prod_acc": prod_acc,
            "prod_pred_home_clf": prod_pred_home_clf,
            "prod_bias_clf": prod_bias_clf,
            "acc_leakage_gap": (prod_acc - heldout_acc) if prod_acc is not None else None,
            "l_gf_mean": meta_heldout["feature_means"].get("l_gf"),
            "v_gf_mean": meta_heldout["feature_means"].get("v_gf"),
            "l_gc_mean": meta_heldout["feature_means"].get("l_gc"),
            "v_gc_mean": meta_heldout["feature_means"].get("v_gc"),
        })

    print("\n" + "=" * 115)
    print("AUDITORÍA DE SESGO DE LOCALÍA — EVALUACIÓN GENUINAMENTE HELD-OUT (80% TRAIN PURO, SIN DATA LEAKAGE)")
    print("=" * 115)
    print(f"{'Liga':<5}{'n_val':>7}{'HomeReal(val)':>15}{'HeldOut Acc':>13}{'PredHome(clf)':>15}{'Bias(clf)':>12}{'Bias(blend fb)':>16}{'Bias(Matriz)':>14}{'Prod Acc (refit)':>18}")
    print("-" * 115)
    for r in results:
        p_acc = f"{r['prod_acc']:.3f}" if r['prod_acc'] is not None else "N/A"
        print(f"{r['liga']:<5}{r['n_val']:>7}{r['home_rate_val']:>15.3f}{r['heldout_acc']:>13.3f}"
              f"{r['pred_home_clf_heldout']:>15.3f}{r['bias_home_clf_heldout']:>+12.3f}"
              f"{r['bias_home_blend_fb_heldout']:>+16.3f}{r['bias_home_matrix_heldout']:>+14.3f}{p_acc:>18}")

    print("\n--- DIAGNÓSTICO DE DATA LEAKAGE: ACCURACY HELD-OUT (80%) VS MODELO DE PRODUCCIÓN (REFIT 100%) ---")
    print(f"{'Liga':<6}{'HeldOut Acc (80%)':>20}{'Prod Acc (Refit 100%)':>24}{'Brecha de Memorización':>26}")
    print("-" * 78)
    for r in results:
        gap = f"{r['acc_leakage_gap']:>+24.1%}" if r['acc_leakage_gap'] is not None else "N/A"
        p_acc = f"{r['prod_acc']:>22.1%}" if r['prod_acc'] is not None else "N/A"
        print(f"{r['liga']:<6}{r['heldout_acc']:>18.1%}{p_acc}{gap}")

    print("\n--- COMPARACIÓN DE BLEND SOBRE DATOS HELD-OUT: 55/45 VS NUEVO FALLBACK (41.7/58.3) ---")
    for r in results:
        bb_orig = f"{r['bias_home_blend_heldout']:+.3f}"
        bb_fb = f"{r['bias_home_blend_fb_heldout']:+.3f}"
        p_fb = f"{r['pred_home_blend_fb_heldout']:.3f}"
        print(f"{r['liga']}: Blend 55/45 Bias={bb_orig}  -->  Nuevo Fallback (41.7/58.3) Pred={p_fb} Bias={bb_fb}")

    print("\n--- COMPONENTE DIXON-COLES/POISSON HELD-OUT SOLO: MEDIA PREDICHA VS REAL ---")
    for r in results:
        print(f"{r['liga']}: p_home matriz={r['pred_home_matrix_heldout']:.3f}  real={r['home_rate_val']:.3f}  "
              f"bias={r['bias_home_matrix_heldout']:+.3f}  | hg_pred_medio={r['hg_mean_pred_heldout']:.2f}  ag_pred_medio={r['ag_mean_pred_heldout']:.2f}")

    print("\n--- RELIABILITY DIAGRAM CLASE HOME (CLASIFICADOR HELD-OUT CALIBRADO) ---")
    for r in results:
        print(f"\n{r['liga']}:")
        for p_pred, p_real, n in r["reliability_home_heldout"]:
            marca = "  <-- SOBRESTIMA" if p_pred - p_real > 0.07 else ("  <-- subestima" if p_real - p_pred > 0.07 else "")
            print(f"   pred={p_pred:.3f}  real={p_real:.3f}  n={n}{marca}")

    out_path = OUT_DIR / "audit_home_advantage_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {out_path}")
    return results


if __name__ == "__main__":
    run_audit()
