"""
Auditoria cuantitativa de sesgo de localia (home advantage) en el pipeline V10.

Herramienta de referencia OBLIGATORIA para el protocolo de colaboracion descrito
en ANTIGRAVITY_COLAB.md. Cualquier cambio a processor.py / advanced_model.py /
app.py que toque features de localia (l_*, v_*), el blend 1X2 o los modelos
hg/ag debe re-ejecutar este script ANTES y DESPUES del cambio y adjuntar
ambos outputs (o el JSON) en el reporte.

Para cada liga:
 1. Reconstruye el split temporal 80/20 EXACTO usado en advanced_model.py.
 2. Calcula la calibracion agregada del clasificador 1X2 (prob media predicha
    vs frecuencia real observada) para las 3 clases, en el set de VALIDACION
    (out-of-sample, igual que reporta metadata).
 3. Reconstruye el blend productivo real de app.py (55% clasificador + 45%
    matriz Poisson/Dixon-Coles con hg/ag) y repite la comparacion.
 4. Reliability diagram (deciles) para la clase Home.

Uso:
    python audits/audit_home_advantage.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

BASE = Path(__file__).resolve().parent.parent
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
        meta_path = MODELS_DIR / f"metadata_{suf}.json"
        m1x2_path = MODELS_DIR / f"model_1x2_v5_{suf}.json"
        ds_path = PROCESSED_DIR / dataset
        if not (meta_path.exists() and m1x2_path.exists() and ds_path.exists()):
            print(f"[SKIP] {suf}: falta metadata/modelo/dataset")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        features = meta["features"]
        temp = meta["models"]["1x2"]["temperature"]

        df = pd.read_csv(ds_path, encoding="utf-8", encoding_errors="replace")
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        n_val = max(int(len(df) * 0.2), 30)
        split = len(df) - n_val
        df_val = df.iloc[split:].copy()

        for f in features:
            if f not in df_val.columns:
                df_val[f] = meta["feature_means"].get(f, 0.0)
        X_val = df_val[features].astype(float)

        booster = xgb.Booster()
        booster.load_model(str(m1x2_path))
        probs_raw = booster.predict(xgb.DMatrix(X_val))
        probs_cal = aplicar_temperatura(probs_raw, temp)

        y_val = df_val["target_1x2"].astype(int).values
        actual_home = float((y_val == 0).mean())
        actual_draw = float((y_val == 1).mean())
        actual_away = float((y_val == 2).mean())

        pred_home_clf = float(probs_cal[:, 0].mean())
        pred_draw_clf = float(probs_cal[:, 1].mean())
        pred_away_clf = float(probs_cal[:, 2].mean())

        # --- Blend productivo real (55% clf + 45% Dixon-Coles/Poisson con hg/ag) ---
        hg_path = MODELS_DIR / f"model_hg_v5_{suf}.json"
        ag_path = MODELS_DIR / f"model_ag_v5_{suf}.json"
        pred_home_blend = pred_draw_blend = pred_away_blend = None
        pred_home_matrix_only = hg_mean = ag_mean = None
        if hg_path.exists() and ag_path.exists():
            b_hg = xgb.Booster(); b_hg.load_model(str(hg_path))
            b_ag = xgb.Booster(); b_ag.load_model(str(ag_path))
            hg_pred = b_hg.predict(xgb.DMatrix(X_val))
            ag_pred = b_ag.predict(xgb.DMatrix(X_val))
            hg_pred = np.maximum(hg_pred, 0.05)
            ag_pred = np.maximum(ag_pred, 0.05)
            p_matrix = np.array([matrix_probs(h, a) for h, a in zip(hg_pred, ag_pred)])
            p_blend = 0.55 * probs_cal + 0.45 * p_matrix
            p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)
            pred_home_blend = float(p_blend[:, 0].mean())
            pred_draw_blend = float(p_blend[:, 1].mean())
            pred_away_blend = float(p_blend[:, 2].mean())

            # Nuevo blend fallback de producción (Ronda 2: 25/60 clf + 35/60 matrix)
            p_blend_fb = (0.25 / 0.60) * probs_cal + (0.35 / 0.60) * p_matrix
            p_blend_fb = p_blend_fb / p_blend_fb.sum(axis=1, keepdims=True)
            pred_home_blend_fb = float(p_blend_fb[:, 0].mean())

            pred_home_matrix_only = float(p_matrix[:, 0].mean())
            hg_mean = float(hg_pred.mean())
            ag_mean = float(ag_pred.mean())

        # Home win rate REAL sobre TODO el historial crudo (no solo validacion), para contexto
        full_home_rate = float((df["target_1x2"] == 0).mean())

        # Reliability diagram clase Home (deciles) sobre el CLASIFICADOR calibrado
        p_home = probs_cal[:, 0]
        order = np.argsort(p_home)
        bins = np.array_split(order, min(5, max(1, len(order) // 15))) if len(order) >= 15 else [order]
        reliab = []
        for b in bins:
            if len(b) == 0:
                continue
            reliab.append((float(p_home[b].mean()), float((y_val[b] == 0).mean()), len(b)))

        results.append({
            "liga": suf,
            "n_val": len(df_val),
            "n_total": len(df),
            "home_rate_val": actual_home,
            "home_rate_full_hist": full_home_rate,
            "draw_rate_val": actual_draw,
            "away_rate_val": actual_away,
            "pred_home_clf": pred_home_clf,
            "pred_draw_clf": pred_draw_clf,
            "pred_away_clf": pred_away_clf,
            "bias_home_clf": pred_home_clf - actual_home,
            "pred_home_blend": pred_home_blend,
            "pred_draw_blend": pred_draw_blend,
            "pred_away_blend": pred_away_blend,
            "bias_home_blend": (pred_home_blend - actual_home) if pred_home_blend is not None else None,
            "pred_home_blend_fb": pred_home_blend_fb if pred_home_blend_fb is not None else None,
            "bias_home_blend_fb": (pred_home_blend_fb - actual_home) if pred_home_blend_fb is not None else None,
            "pred_home_matrix_only": pred_home_matrix_only,
            "bias_home_matrix_only": (pred_home_matrix_only - actual_home) if pred_home_matrix_only is not None else None,
            "hg_mean_pred": hg_mean,
            "ag_mean_pred": ag_mean,
            "reliability_home": reliab,
            "l_gf_mean": meta["feature_means"].get("l_gf"),
            "v_gf_mean": meta["feature_means"].get("v_gf"),
            "l_gc_mean": meta["feature_means"].get("l_gc"),
            "v_gc_mean": meta["feature_means"].get("v_gc"),
        })

    print("\n" + "=" * 100)
    print(f"{'Liga':<5}{'n_val':>7}{'HomeReal(val)':>16}{'HomeReal(hist)':>16}{'PredHome(clf)':>16}{'Bias(clf)':>12}{'PredHome(blend)':>18}{'Bias(blend)':>14}")
    print("=" * 100)
    for r in results:
        pb = f"{r['pred_home_blend']:.3f}" if r["pred_home_blend"] is not None else "N/A"
        bb = f"{r['bias_home_blend']:+.3f}" if r["bias_home_blend"] is not None else "N/A"
        print(f"{r['liga']:<5}{r['n_val']:>7}{r['home_rate_val']:>16.3f}{r['home_rate_full_hist']:>16.3f}"
              f"{r['pred_home_clf']:>16.3f}{r['bias_home_clf']:>+12.3f}{pb:>18}{bb:>14}")

    print("\n--- Comparación: Blend Original (55/45) vs Nuevo Fallback Rebalanceado (41.7/58.3) ---")
    for r in results:
        bb_orig = f"{r['bias_home_blend']:+.3f}" if r["bias_home_blend"] is not None else "N/A"
        bb_fb = f"{r['bias_home_blend_fb']:+.3f}" if r["bias_home_blend_fb"] is not None else "N/A"
        p_fb = f"{r['pred_home_blend_fb']:.3f}" if r["pred_home_blend_fb"] is not None else "N/A"
        print(f"{r['liga']}: Blend 55/45 Bias={bb_orig}  -->  Nuevo Fallback (41.7/58.3) Pred={p_fb} Bias={bb_fb}")

    print("\n--- Componente Dixon-Coles/Poisson SOLO (45% del blend): media predicha vs real ---")
    for r in results:
        if r["pred_home_matrix_only"] is None:
            continue
        print(f"{r['liga']}: p_home matriz={r['pred_home_matrix_only']:.3f}  real={r['home_rate_val']:.3f}  "
              f"bias={r['bias_home_matrix_only']:+.3f}  | hg_pred_medio={r['hg_mean_pred']:.2f}  ag_pred_medio={r['ag_mean_pred']:.2f}")

    print("\n--- Sesgo de goles esperados (feature_means, ultimos 5 partidos aislados por condicion) ---")
    for r in results:
        print(f"{r['liga']}: l_gf(local, en casa)={r['l_gf_mean']:.2f}  v_gf(visita, fuera)={r['v_gf_mean']:.2f}  "
              f"| l_gc(local recibe en casa)={r['l_gc_mean']:.2f}  v_gc(visita recibe fuera)={r['v_gc_mean']:.2f}")

    print("\n--- Reliability diagram clase HOME (clasificador calibrado, deciles por prob. predicha) ---")
    for r in results:
        print(f"\n{r['liga']}:")
        for p_pred, p_real, n in r["reliability_home"]:
            marca = "  <-- SOBRESTIMA" if p_pred - p_real > 0.07 else ("  <-- subestima" if p_real - p_pred > 0.07 else "")
            print(f"   pred={p_pred:.3f}  real={p_real:.3f}  n={n}{marca}")

    out_path = OUT_DIR / "audit_home_advantage_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {out_path}")
    return results


if __name__ == "__main__":
    run_audit()
