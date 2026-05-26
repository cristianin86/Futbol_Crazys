import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os

print("Iniciando Entrenamiento del Cerebro Nativo Chile (V31.0)")

# 1. Carga de datos base
df_raw = pd.read_csv("chile_api_raw.csv")

# Filtrar completados y limpiar nulos
if 'status' in df_raw.columns:
    df_raw = df_raw[df_raw['status'] == 'complete']
else:
    df_raw = df_raw.dropna(subset=['home_team_goal_count'])

# Ordenar por tiempo
df_raw = df_raw.sort_values('timestamp')
print(f"Total de partidos completados para entrenar: {len(df_raw)}")

# 2. Reconstrucción del Historial (Viaje en el tiempo)
data_rows = []

for idx, match in df_raw.iterrows():
    # Solo usar partidos anteriores a este para evitar Data Leakage
    df_history = df_raw[df_raw['timestamp'] < match['timestamp']].copy()
    
    home_team = match['home_team_name']
    away_team = match['away_team_name']
    
    # HOME STATS (Local condition)
    df_h = df_history[df_history['home_team_name'] == home_team].copy()
    df_h = df_h.sort_values('timestamp', ascending=False).head(5)
    n_h = len(df_h)
    
    # AWAY STATS (Visita condition)
    df_a = df_history[df_history['away_team_name'] == away_team].copy()
    df_a = df_a.sort_values('timestamp', ascending=False).head(5)
    n_a = len(df_a)
    
    # Si no hay suficiente historia para alguno, nos saltamos para no ensuciar el modelo (mínimo 1)
    if n_h == 0 or n_a == 0:
        continue
        
    l_gf = df_h['home_team_goal_count'].sum() / n_h
    l_gc = df_h['away_team_goal_count'].sum() / n_h
    l_pos = df_h['home_team_possession'].sum() / n_h
    l_xg = df_h['home_team_pre_match_xg'].sum() / n_h
    
    h_wins = sum(df_h['home_team_goal_count'] > df_h['away_team_goal_count'])
    h_draws = sum(df_h['home_team_goal_count'] == df_h['away_team_goal_count'])
    l_ppg = ((h_wins * 3) + h_draws) / n_h

    v_gf = df_a['away_team_goal_count'].sum() / n_a
    v_gc = df_a['home_team_goal_count'].sum() / n_a
    v_pos = df_a['away_team_possession'].sum() / n_a
    v_xg = df_a['away_team_pre_match_xg'].sum() / n_a
    
    a_wins = sum(df_a['away_team_goal_count'] > df_a['home_team_goal_count'])
    a_draws = sum(df_a['away_team_goal_count'] == df_a['home_team_goal_count'])
    v_ppg = ((a_wins * 3) + a_draws) / n_a

    # Determine targets
    h_goals = match['home_team_goal_count']
    a_goals = match['away_team_goal_count']
    
    if h_goals > a_goals:
        target_1x2 = 0
    elif h_goals == a_goals:
        target_1x2 = 1
    else:
        target_1x2 = 2
        
    data_rows.append({
        'l_gf': l_gf, 'l_gc': l_gc, 'l_ppg': l_ppg, 'l_pos': l_pos, 'l_xg': l_xg,
        'v_gf': v_gf, 'v_gc': v_gc, 'v_ppg': v_ppg, 'v_pos': v_pos, 'v_xg': v_xg,
        'target_1x2': target_1x2,
        'hg': h_goals, 'ag': a_goals,
        'hc': match['home_team_corner_count'],
        'ac': match['away_team_corner_count']
    })

dataset = pd.DataFrame(data_rows)
# Fillna in targets if any API missing data (especially corners)
dataset = dataset.fillna(0)

print(f"Dataset generado exitosamente con {len(dataset)} registros utiles.")

features = ['l_gf', 'l_gc', 'l_ppg', 'l_pos', 'l_xg', 'v_gf', 'v_gc', 'v_ppg', 'v_pos', 'v_xg']
X = dataset[features]

# --- ENTRENAMIENTO DE LA SUITE COMPLETA ---
print("\nENTRENANDO MODELOS...")

# 1. Modelo 1X2 (Clasificador)
y_1x2 = dataset['target_1x2']
dtrain_1x2 = xgb.DMatrix(X, label=y_1x2)
params_1x2 = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 2,
    'eta': 0.01,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}
model_1x2 = xgb.train(params_1x2, dtrain_1x2, num_boost_round=100)
preds_1x2 = np.argmax(model_1x2.predict(dtrain_1x2), axis=1)
acc = accuracy_score(y_1x2, preds_1x2)
model_1x2.save_model('modelo_chile_1x2_v1.json')
print(f"Modelo 1X2 Entrenado | Train Accuracy: {acc*100:.1f}%")

# 2. Modelo HG (Home Goals Regressor)
y_hg = dataset['hg']
dtrain_hg = xgb.DMatrix(X, label=y_hg)
params_reg = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'eta': 0.01,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}
model_hg = xgb.train(params_reg, dtrain_hg, num_boost_round=100)
rmse_hg = np.sqrt(mean_squared_error(y_hg, model_hg.predict(dtrain_hg)))
model_hg.save_model('modelo_chile_hg_v1.json')
print(f"Modelo HG Entrenado | Train RMSE: {rmse_hg:.2f}")

# 3. Modelo AG (Away Goals Regressor)
y_ag = dataset['ag']
dtrain_ag = xgb.DMatrix(X, label=y_ag)
model_ag = xgb.train(params_reg, dtrain_ag, num_boost_round=100)
rmse_ag = np.sqrt(mean_squared_error(y_ag, model_ag.predict(dtrain_ag)))
model_ag.save_model('modelo_chile_ag_v1.json')
print(f"Modelo AG Entrenado | Train RMSE: {rmse_ag:.2f}")

# 4. Modelo HC (Home Corners Regressor)
y_hc = dataset['hc']
dtrain_hc = xgb.DMatrix(X, label=y_hc)
model_hc = xgb.train(params_reg, dtrain_hc, num_boost_round=100)
rmse_hc = np.sqrt(mean_squared_error(y_hc, model_hc.predict(dtrain_hc)))
model_hc.save_model('modelo_chile_hc_v1.json')
print(f"Modelo HC Entrenado | Train RMSE: {rmse_hc:.2f}")

# 5. Modelo AC (Away Corners Regressor)
y_ac = dataset['ac']
dtrain_ac = xgb.DMatrix(X, label=y_ac)
model_ac = xgb.train(params_reg, dtrain_ac, num_boost_round=100)
rmse_ac = np.sqrt(mean_squared_error(y_ac, model_ac.predict(dtrain_ac)))
model_ac.save_model('modelo_chile_ac_v1.json')
print(f"Modelo AC Entrenado | Train RMSE: {rmse_ac:.2f}")

print("\nTodos los modelos de Chile exportados exitosamente.")
