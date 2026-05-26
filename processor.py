import pandas as pd
import numpy as np
import glob
import os
import sys
import io


def procesar_modelo_agnostico_v9(input_csv=None, output_csv="chile_ml_ready_v9.csv"):
    print(f"Iniciando Procesador V9 (Modelo Agnostico & Momentum Real)...")
    
    if not input_csv:
        archivos_csv = glob.glob('*api_raw.csv') + glob.glob('*chile*.csv')
        archivos_csv = [f for f in archivos_csv if 'ml_ready' not in f and 'agnostic' not in f]
        
        if not archivos_csv:
            print("Error: No se encontro ningun archivo CSV de partidos.")
            return
        
        input_csv = max(archivos_csv, key=os.path.getctime)
    
    print(f"Leyendo: {input_csv}")
    df = pd.read_csv(input_csv).sort_values('timestamp')
    
    # 1. Puntos y Goles por partido
    df['h_pts'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'], df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)
    df['a_pts'] = np.select([df['away_team_goal_count'] > df['home_team_goal_count'], df['home_team_goal_count'] == df['away_team_goal_count']], [3, 1], 0)
    
    # 2. CALCULO DE ROLLING STATS (MEMORIA VIVA)
    print("Calculando Ventanas Moviles (Rolling 5)...")
    
    stats_cols = ['gf', 'gc', 'ppg', 'pos', 'xg']
    for col in [f'l_{c}' for c in stats_cols] + [f'v_{c}' for c in stats_cols]:
        df[col] = 0.0

    # Diccionario para trackear la historia de cada equipo, separada por condición
    # history = {team_name: {'Local': [matches_data], 'Visita': [matches_data]}}
    history = {} 

    for i, row in df.iterrows():
        h_team = row['home_team_name']
        a_team = row['away_team_name']
        
        # Inicializar en el tracker si es primera vez
        if h_team not in history: history[h_team] = {'Local': [], 'Visita': []}
        if a_team not in history: history[a_team] = {'Local': [], 'Visita': []}
        
        # --- OBTENER RACHA PREVIA (MATCH STRICT CONDITION) ---
        # 1. Racha Home
        if len(history[h_team]['Local']) > 0:
            last_5_h = history[h_team]['Local'][-5:]
        elif len(history[h_team]['Visita']) > 0:
            last_5_h = (history[h_team]['Local'] + history[h_team]['Visita'])[-5:] # Fallback
        else:
            last_5_h = []
            
        if last_5_h:
            df.at[i, 'l_gf'] = np.mean([m['gf'] for m in last_5_h])
            df.at[i, 'l_gc'] = np.mean([m['gc'] for m in last_5_h])
            df.at[i, 'l_ppg'] = np.mean([m['pts'] for m in last_5_h])
            df.at[i, 'l_pos'] = np.mean([m['pos'] for m in last_5_h])
            df.at[i, 'l_xg'] = np.mean([m['xg'] for m in last_5_h])
        
        # 2. Racha Away
        if len(history[a_team]['Visita']) > 0:
            last_5_a = history[a_team]['Visita'][-5:]
        elif len(history[a_team]['Local']) > 0:
            last_5_a = (history[a_team]['Local'] + history[a_team]['Visita'])[-5:] # Fallback
        else:
            last_5_a = []
            
        if last_5_a:
            df.at[i, 'v_gf'] = np.mean([m['gf'] for m in last_5_a])
            df.at[i, 'v_gc'] = np.mean([m['gc'] for m in last_5_a])
            df.at[i, 'v_ppg'] = np.mean([m['pts'] for m in last_5_a])
            df.at[i, 'v_pos'] = np.mean([m['pos'] for m in last_5_a])
            df.at[i, 'v_xg'] = np.mean([m['xg'] for m in last_5_a])

        # --- ACTUALIZAR HISTORIA CON ESTE PARTIDO (PARA EL FUTURO) ---
        history[h_team]['Local'].append({'gf': row['home_team_goal_count'], 'gc': row['away_team_goal_count'], 'pts': row['h_pts'], 'pos': row['home_team_possession'], 'xg': row['home_team_pre_match_xg']})
        history[a_team]['Visita'].append({'gf': row['away_team_goal_count'], 'gc': row['home_team_goal_count'], 'pts': row['a_pts'], 'pos': row['away_team_possession'], 'xg': row['away_team_pre_match_xg']})

    # 3. Targets (Solo partidos finalizados para entrenamiento)
    df = df.dropna(subset=['home_team_goal_count', 'away_team_goal_count'])
    df['target_1x2'] = np.select([df['home_team_goal_count'] > df['away_team_goal_count'], df['home_team_goal_count'] == df['away_team_goal_count']], [0, 1], 2)
    
    # 4. Exportar Dataset Agnostico
    features = [f'l_{c}' for c in stats_cols] + [f'v_{c}' for c in stats_cols]
    if 'is_cup' in df.columns:
        features += ['is_cup', 'home_team_lsi', 'away_team_lsi']
    targets = ['target_1x2', 'home_team_goal_count', 'away_team_goal_count', 'home_team_corner_count', 'away_team_corner_count']
    
    # Solo filas con algo de historia (minimo 1 partido previo para no ser puro 0)
    df_agnostic = df[(df['l_ppg'] >= 0) | (df['v_ppg'] >= 0)][features + targets].copy()
    df_agnostic = df_agnostic.dropna()
    
    df_agnostic.to_csv(output_csv, index=False)
    print(f"Dataset Agnostico V9 guardado ({len(df_agnostic)} muestras) en {output_csv}.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        procesar_modelo_agnostico_v9(sys.argv[1], sys.argv[2])
    else:
        procesar_modelo_agnostico_v9()

