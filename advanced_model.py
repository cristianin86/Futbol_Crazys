import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import io


def entrenar_comite_v5(dataset="chile_ml_ready_v9.csv", suffix=""):
    print(f"--- Entrenando Comite V5 (Dataset: {dataset}, Sufijo: '{suffix}') ---")
    
    if not os.path.exists(dataset):
        print(f"Error: Falta el archivo {dataset}. Ejecuta processor.py primero.")
        return
        
    df = pd.read_csv(dataset)
    features = ['l_gf', 'l_gc', 'l_ppg', 'l_pos', 'l_xg', 'v_gf', 'v_gc', 'v_ppg', 'v_pos', 'v_xg']
    if 'is_cup' in df.columns:
        features += ['is_cup', 'home_team_lsi', 'away_team_lsi']
    X = df[features]
    
    pesos = np.linspace(0.5, 1.0, len(df))
    params_poisson = {'objective': 'count:poisson', 'max_depth': 3, 'learning_rate': 0.05}

    s = f"_{suffix}" if suffix else ""

    print("1/5: Entrenando IA de Resultado 1X2...")
    xgb.train({'objective': 'multi:softprob', 'num_class': 3, 'max_depth': 4}, xgb.DMatrix(X, label=df['target_1x2'], weight=pesos), 100).save_model(f"model_1x2_v5{s}.json")
    
    print("2/5: Entrenando IA Goles Local...")
    xgb.train(params_poisson, xgb.DMatrix(X, label=df['home_team_goal_count'], weight=pesos), 100).save_model(f"model_hg_v5{s}.json")
    
    print("3/5: Entrenando IA Goles Visitante...")
    xgb.train(params_poisson, xgb.DMatrix(X, label=df['away_team_goal_count'], weight=pesos), 100).save_model(f"model_ag_v5{s}.json")
    
    if 'home_team_corner_count' in df.columns:
        print("4/5: Entrenando IA Córners Local...")
        xgb.train(params_poisson, xgb.DMatrix(X, label=df['home_team_corner_count'], weight=pesos), 100).save_model(f"model_hc_v5{s}.json")
        
        print("5/5: Entrenando IA Córners Visitante...")
        xgb.train(params_poisson, xgb.DMatrix(X, label=df['away_team_corner_count'], weight=pesos), 100).save_model(f"model_ac_v5{s}.json")
    else:
        print("Skipping corners: Columnas no encontradas.")

    print(f"COMPLETO: Los 5 cerebros {s} han sido sincronizados!")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        entrenar_comite_v5(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        entrenar_comite_v5(sys.argv[1])
    else:
        entrenar_comite_v5()

