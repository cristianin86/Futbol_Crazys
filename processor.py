"""
processor.py
============
Transformación agnóstica de datos brutos a variables móviles (Rolling 5)
para alimentar a los modelos predictivos sin lookahead bias.

V10:
- El historial solo se alimenta con partidos FINALIZADOS (antes entraban upcoming).
- Filtro real de historial mínimo (antes la condición l_ppg >= 0 era siempre True).
- Nuevas features: tarjetas amarillas y tiros al arco (rolling 5 por condición).
- Se conservan timestamp y nombres de equipo en el dataset final para
  validación temporal, auditoría y backtesting (no son features).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Resolver paths con config o fallback local
try:
    from config.settings import resolve_path, RAW_DATA_DIR, PROCESSED_DATA_DIR
except ImportError:
    BASE = Path(__file__).resolve().parent
    RAW_DATA_DIR = BASE / "data" / "raw"
    PROCESSED_DATA_DIR = BASE / "data" / "processed"
    def resolve_path(f, d=None):
        if d and (d / f).exists(): return d / f
        if (BASE / f).exists(): return BASE / f
        return (d / f) if d else (BASE / f)

# Stats por partido que se promedian en ventana móvil.
# clave rolling -> (columna home, columna away)
ROLLING_MAP = {
    'gf':   ('home_team_goal_count', 'away_team_goal_count'),
    'gc':   ('away_team_goal_count', 'home_team_goal_count'),
    'pos':  ('home_team_possession', 'away_team_possession'),
    'xg':   ('home_team_pre_match_xg', 'away_team_pre_match_xg'),
    'yc':   ('home_team_yellow_cards', 'away_team_yellow_cards'),
    'sot':  ('home_team_shots_on_target', 'away_team_shots_on_target'),
}


def procesar_modelo_agnostico_v9(input_csv=None, output_csv="chile_ml_ready_v8.csv"):
    print("Iniciando Procesador V10 (Rolling 5 por condicion, sin lookahead)...")

    in_path = None
    if input_csv:
        in_path = resolve_path(input_csv, RAW_DATA_DIR)

    if not in_path or not in_path.exists():
        candidatos = list(RAW_DATA_DIR.glob('*api_raw.csv')) + list(Path('.').glob('*api_raw.csv'))
        if not candidatos:
            print("Error: No se encontro ningun archivo CSV crudo de partidos.")
            return False
        in_path = max(candidatos, key=os.path.getctime)

    print(f"Leyendo: {in_path}")
    df = pd.read_csv(in_path, encoding='utf-8', encoding_errors='replace').sort_values('timestamp').reset_index(drop=True)

    # Columnas opcionales que pueden faltar en históricos antiguos
    for col in ['home_team_shots_on_target', 'away_team_shots_on_target',
                'home_team_yellow_cards', 'away_team_yellow_cards']:
        if col not in df.columns:
            df[col] = np.nan

    # 1. Puntos por partido
    df['h_pts'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'],
                             df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)
    df['a_pts'] = np.select([df['away_team_goal_count'] > df['home_team_goal_count'],
                             df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)

    # 2. ROLLING STATS (memoria viva por equipo, separada por condición)
    # Constante de encogimiento Bayesiano (Bayes empirical shrinkage):
    # w = n / (n + k). Con k=2.0: 1 partido -> peso 0.33; 2 partidos -> peso 0.50; 5 partidos -> peso 0.71; 15 partidos -> peso 0.88.
    # Calibrado en Ronda 2 para evitar sobre-encogimiento y preservar la discriminación en ligas asimétricas (ESP, CHI, PER).
    K_SHRINKAGE = 2.0
    print("Calculando Ventanas Moviles (Rolling 5 con Shrinkage Bayesiano K=2.0)...")

    stats_cols = list(ROLLING_MAP.keys()) + ['ppg']
    for col in [f'l_{c}' for c in stats_cols] + [f'v_{c}' for c in stats_cols]:
        df[col] = 0.0
    df['l_gp'] = 0
    df['v_gp'] = 0

    history = {}

    # Acumuladores de liga en tiempo real (estrictamente anti-lookahead) por condición
    league_accum = {
        'Local': {k: [0.0, 0] for k in list(ROLLING_MAP.keys()) + ['ppg']},
        'Visita': {k: [0.0, 0] for k in list(ROLLING_MAP.keys()) + ['ppg']}
    }
    prior_league_means = {
        'Local':  {'gf': 1.45, 'gc': 1.15, 'ppg': 1.60, 'pos': 51.0, 'xg': 1.35, 'yc': 2.30, 'sot': 4.50},
        'Visita': {'gf': 1.15, 'gc': 1.45, 'ppg': 1.15, 'pos': 49.0, 'xg': 1.15, 'yc': 2.50, 'sot': 3.80},
    }

    def get_league_mean(cond, key):
        acc = league_accum[cond][key]
        return (acc[0] / acc[1]) if acc[1] > 0 else prior_league_means[cond].get(key, 1.0)

    def snapshot(row, side):
        """Extrae las stats de un partido jugado, desde la perspectiva home/away."""
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

        w = float(n_cond / (n_cond + K_SHRINKAGE))

        # PPG con shrinkage bayesiano hacia la media acumulada de la liga
        team_ppg = float(np.mean([m['pts'] for m in last_5]))
        mu_ppg = float(get_league_mean(cond, 'ppg'))
        df.at[i, f'{prefix}_ppg'] = float(w * team_ppg + (1.0 - w) * mu_ppg)

        # Métricas de ROLLING_MAP con shrinkage bayesiano hacia media de liga
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

        # Racha previa del local (prioridad: partidos en casa; fallback: mixto)
        n_h_cond = len(h_hist['Local'])
        last_5_h = h_hist['Local'][-5:] if h_hist['Local'] else (h_hist['Local'] + h_hist['Visita'])[-5:]
        fill_features(i, 'l', last_5_h, len(h_hist['Local']) + len(h_hist['Visita']), n_h_cond, 'Local')

        # Racha previa del visitante (prioridad: partidos fuera; fallback: mixto)
        n_a_cond = len(a_hist['Visita'])
        last_5_a = a_hist['Visita'][-5:] if a_hist['Visita'] else (a_hist['Local'] + a_hist['Visita'])[-5:]
        fill_features(i, 'v', last_5_a, len(a_hist['Local']) + len(a_hist['Visita']), n_a_cond, 'Visita')

        # Actualizar historia SOLO con partidos ya jugados (los upcoming contaminaban el tracker)
        es_jugado = pd.notna(row['home_team_goal_count']) and pd.notna(row['away_team_goal_count'])
        if str(row.get('status', 'complete')) == 'complete' and es_jugado:
            h_snap = snapshot(row, 'home')
            a_snap = snapshot(row, 'away')
            h_hist['Local'].append(h_snap)
            a_hist['Visita'].append(a_snap)

            # Acumulador de liga sin lookahead (solo partidos jugados hasta el momento)
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

    # 3. Targets (solo partidos finalizados)
    df = df.dropna(subset=['home_team_goal_count', 'away_team_goal_count'])
    if 'status' in df.columns:
        df = df[df['status'] == 'complete']
    df['target_1x2'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'],
                                  df['home_team_goal_count'] == df['away_team_goal_count']], [0, 1], 2)

    # 4. Exportar Dataset
    base_feats = ['gf', 'gc', 'ppg', 'pos', 'xg', 'yc', 'sot']
    features = [f'l_{c}' for c in base_feats] + [f'v_{c}' for c in base_feats]
    if 'is_cup' in df.columns:
        features += ['is_cup', 'home_team_lsi', 'away_team_lsi']
    targets = ['target_1x2', 'home_team_goal_count', 'away_team_goal_count',
               'home_team_corner_count', 'away_team_corner_count',
               'home_team_yellow_cards', 'away_team_yellow_cards']
    targets = [t for t in targets if t in df.columns]
    meta = [c for c in ['timestamp', 'home_team_name', 'away_team_name', 'stats_real'] if c in df.columns]

    # Filtro REAL de historial mínimo: ambos equipos con al menos 1 partido previo.
    # (La versión anterior usaba l_ppg >= 0, que era siempre True.)
    mask_historia = (df['l_gp'] >= 1) & (df['v_gp'] >= 1)
    n_descartadas = int((~mask_historia).sum())
    df_agnostic = df[mask_historia][meta + features + targets].copy()
    df_agnostic = df_agnostic.dropna(subset=features + targets)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_target = PROCESSED_DATA_DIR / output_csv
    df_agnostic.to_csv(out_target, index=False, encoding='utf-8')

    print(f"Dataset V10 guardado ({len(df_agnostic)} muestras, {n_descartadas} sin historial descartadas) en {out_target}.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2:
        procesar_modelo_agnostico_v9(sys.argv[1], sys.argv[2])
    else:
        procesar_modelo_agnostico_v9()
