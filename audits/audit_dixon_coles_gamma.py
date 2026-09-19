"""
audits/audit_dixon_coles_gamma.py
=================================
Auditoría diagnóstica rigurosa de localía pura (gamma) y dependencia (rho)
estimados por Máxima Verosimilitud (MLE) mediante DixonColesModel para las 6 ligas.

V2 (Ronda 2):
- Reemplaza el test de permutación conmutativa por:
  1. BOOTSTRAP CON REEMPLAZO (N=30) paralelizado con ProcessPoolExecutor para medir
     el error estándar empírico real (std_gamma) e intervalo de confianza al 95%.
  2. PRUEBA DE INICIALIZACIONES MÚLTIPLES (init_params: estándar, cero, alto) para
     descartar convergencia a mínimos locales.
  3. Contraste frente a la brecha empírica de goles en producción (l_gf vs v_gf).
"""

import json
import math
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from dixon_coles_model import DixonColesModel

RAW_DIR = BASE / "data" / "raw"
MODELS_DIR = BASE / "models_saved"
OUT_DIR = Path(__file__).resolve().parent

LIGAS_CONFIG = {
    "CHI": {"raw": "chile_api_raw.csv", "meta": "metadata_CHI.json", "name": "Chile (Primera)"},
    "B":   {"raw": "chile_b_api_raw.csv", "meta": "metadata_B.json", "name": "Chile (Primera B)"},
    "ENG": {"raw": "premier_api_raw.csv", "meta": "metadata_ENG.json", "name": "Premier League"},
    "ARG": {"raw": "argentina_api_raw.csv", "meta": "metadata_ARG.json", "name": "Argentina"},
    "PER": {"raw": "peru_api_raw.csv", "meta": "metadata_PER.json", "name": "Perú"},
    "ESP": {"raw": "espana_api_raw.csv", "meta": "metadata_ESP.json", "name": "España (LaLiga)"},
}


def _fit_bootstrap_single(args):
    raw_path_str, seed = args
    df = pd.read_csv(raw_path_str, encoding="utf-8", encoding_errors="replace")
    if "status" in df.columns:
        df = df[df["status"] == "complete"]
    else:
        df = df.dropna(subset=["home_team_goal_count", "away_team_goal_count"])
    df_boot = df.sample(frac=1.0, replace=True, random_state=seed).reset_index(drop=True)
    m = DixonColesModel()
    m.fit(df_boot)
    return float(m.gamma), float(m.rho)


def _fit_custom_init(df, gamma_0, rho_0):
    """Ajusta Dixon-Coles modificando los puntos de inicio para verificar mínimos locales."""
    m = DixonColesModel()
    h_col = 'home_team_name' if 'home_team_name' in df.columns else 'home_team'
    a_col = 'away_team_name' if 'away_team_name' in df.columns else 'away_team'
    hg_col = 'home_team_goal_count' if 'home_team_goal_count' in df.columns else 'home_goals'
    ag_col = 'away_team_goal_count' if 'away_team_goal_count' in df.columns else 'away_goals'
    df_clean = df.dropna(subset=[h_col, a_col, hg_col, ag_col]).copy()
    m.teams = sorted(list(set(df_clean[h_col].unique().tolist() + df_clean[a_col].unique().tolist())))
    m.team_indices = {t: i for i, t in enumerate(m.teams)}
    n_teams = len(m.teams)
    home_idx = np.array([m.team_indices[t] for t in df_clean[h_col]])
    away_idx = np.array([m.team_indices[t] for t in df_clean[a_col]])
    home_goals = df_clean[hg_col].values.astype(int)
    away_goals = df_clean[ag_col].values.astype(int)
    from dixon_coles_model import calculate_time_weights
    weights = calculate_time_weights(df_clean['timestamp'].values, xi=m.xi) if 'timestamp' in df_clean.columns else np.ones(len(df_clean))
    from scipy.optimize import minimize
    init_params = np.concatenate([np.zeros(n_teams), np.zeros(n_teams), [gamma_0, rho_0]])
    res = minimize(m._log_likelihood, init_params, args=(home_idx, away_idx, home_goals, away_goals, weights), method='BFGS', options={'maxiter': 100, 'disp': False})
    return float(res.x[2*n_teams]), float(res.x[2*n_teams + 1])


def auditar_gamma_ligas(n_boot=30):
    print("=" * 115)
    print(f"AUDITORÍA DIXON-COLES V2 (BOOTSTRAP N={n_boot} + MULTI-INIT + BRECHA EMPÍRICA)")
    print("=" * 115)

    results = []

    for suf, cfg in LIGAS_CONFIG.items():
        raw_path = RAW_DIR / cfg["raw"]
        meta_path = MODELS_DIR / cfg["meta"]

        if not raw_path.exists():
            print(f"[SKIP] No existe raw: {raw_path}")
            continue

        df = pd.read_csv(raw_path, encoding="utf-8", encoding_errors="replace")
        if "status" in df.columns:
            df = df[df["status"] == "complete"]
        else:
            df = df.dropna(subset=["home_team_goal_count", "away_team_goal_count"])

        n_partidos = len(df)

        # 1. Ajuste estándar por MLE
        m_base = DixonColesModel()
        m_base.fit(df)
        gamma_mle = float(m_base.gamma)
        rho_mle = float(m_base.rho)
        n_teams = len(m_base.teams)

        # 2. Prueba de Sensibilidad a Inicializaciones (descarta mínimos locales)
        # Init 1: Estándar (0.25, -0.05) -> gamma_mle
        # Init 2: Cero (0.00, 0.00)
        # Init 3: Alto (0.50, +0.10)
        g_zero, _ = _fit_custom_init(df, 0.00, 0.00)
        g_high, _ = _fit_custom_init(df, 0.50, 0.10)
        max_init_diff = max(abs(g_zero - gamma_mle), abs(g_high - gamma_mle))

        # 3. Bootstrap con Reemplazo (N=30) paralelizado
        boot_args = [(str(raw_path), seed) for seed in range(n_boot)]
        with ProcessPoolExecutor(max_workers=min(n_boot, 16)) as executor:
            boot_res = list(executor.map(_fit_bootstrap_single, boot_args))

        boot_gammas = [b[0] for b in boot_res]
        gamma_boot_mean = float(np.mean(boot_gammas))
        gamma_boot_std = float(np.std(boot_gammas))
        ci_95_low = float(np.percentile(boot_gammas, 2.5))
        ci_95_high = float(np.percentile(boot_gammas, 97.5))

        # 4. Leer feature_means de metadata para comparar
        l_gf_mean = v_gf_mean = l_gc_mean = v_gc_mean = None
        ln_ratio_gf = None
        pct_stats_reales = None

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fmeans = meta.get("feature_means", {})
                l_gf_mean = fmeans.get("l_gf")
                v_gf_mean = fmeans.get("v_gf")
                l_gc_mean = fmeans.get("l_gc")
                v_gc_mean = fmeans.get("v_gc")
                pct_stats_reales = meta.get("pct_stats_reales")
                if l_gf_mean and v_gf_mean and v_gf_mean > 0:
                    ln_ratio_gf = math.log(l_gf_mean / v_gf_mean)
            except Exception as e:
                print(f"[WARN] Error leyendo metadata {meta_path}: {e}")

        ratio_gf = (l_gf_mean / v_gf_mean) if (l_gf_mean and v_gf_mean) else None

        row = {
            "liga": suf,
            "nombre": cfg["name"],
            "partidos": n_partidos,
            "equipos": n_teams,
            "gamma_mle": gamma_mle,
            "exp_gamma": math.exp(gamma_mle),
            "rho_mle": rho_mle,
            "bootstrap_mean": gamma_boot_mean,
            "bootstrap_std": gamma_boot_std,
            "ci_95": [ci_95_low, ci_95_high],
            "max_init_diff": max_init_diff,
            "l_gf": l_gf_mean,
            "v_gf": v_gf_mean,
            "ratio_gf": ratio_gf,
            "ln_ratio_gf": ln_ratio_gf,
            "pct_stats_reales": pct_stats_reales,
        }
        results.append(row)

    print(f"\n{'Liga':<5}{'Part':>6}{'gamma(MLE)':>11}{'Boot Mean':>11}{'Boot Std':>10}{'IC 95% (gamma)':>18}{'InitDiff':>10}{'ratio(l/v)':>12}{'ln(l/v)':>10}{'rho(MLE)':>10}")
    print("-" * 115)
    for r in results:
        r_gf = f"{r['ratio_gf']:.3f}" if r["ratio_gf"] else "N/A"
        ln_r = f"{r['ln_ratio_gf']:.3f}" if r["ln_ratio_gf"] else "N/A"
        ci_str = f"[{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}]"
        print(f"{r['liga']:<5}{r['partidos']:>6}{r['gamma_mle']:>11.4f}{r['bootstrap_mean']:>11.4f}{r['bootstrap_std']:>10.4f}{ci_str:>18}{r['max_init_diff']:>10.5f}{r_gf:>12}{ln_r:>10}{r['rho_mle']:>10.4f}")

    out_file = OUT_DIR / "audit_dixon_coles_gamma_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados rigurosos guardados en: {out_file}")
    return results


if __name__ == "__main__":
    auditar_gamma_ligas(n_boot=30)
