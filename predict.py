import pandas as pd
import numpy as np
import xgboost as xgb
import os
import glob
import sys
import io


def obtener_features_partido(equipo_local, equipo_visita, ruta_csv):
    """Busca las últimas métricas disponibles para ambos equipos en el CSV."""
    df = pd.read_csv(ruta_csv)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # 1. Buscar la última aparición del equipo Local en casa
    df_local = df[df['home_team_name'].str.contains(equipo_local, case=False, na=False)].copy()
    if df_local.empty:
        print(f"⚠️ Aviso: No se encontraron datos recientes como Local para '{equipo_local}'")
        return None
    
    # Ordenar por fecha (asumiendo que las filas finales son las más recientes) y tomar la última
    ultimo_partido_local = df_local.iloc[-1:]
    
    # 2. Buscar la última aparición del equipo Visitante fuera
    df_visita = df[df['away_team_name'].str.contains(equipo_visita, case=False, na=False)].copy()
    if df_visita.empty:
        print(f"⚠️ Aviso: No se encontraron datos recientes como Visitante para '{equipo_visita}'")
        return None
    
    ultimo_partido_visita = df_visita.iloc[-1:]

    # 3. Construir el vector de características "sintético" para el enfrentamiento
    features_comb = ultimo_partido_local.copy()
    
    # Reemplazamos las métricas 'away' con las del equipo visitante real
    cols_away = [c for c in df.columns if 'away' in c or 'team_b' in c]
    for col in cols_away:
        features_comb[col] = ultimo_partido_visita[col].values[0]

    # Recalcular métricas compuestas (Momentum, Severidad)
    if 'team_a_xg_pre_match' in features_comb.columns and 'team_b_xg_pre_match' in features_comb.columns:
        features_comb['xg_momentum_diff'] = features_comb['team_a_xg_pre_match'] - features_comb['team_b_xg_pre_match']
    
    # Añadir un árbitro neutral por defecto si no lo especificamos
    features_comb['referee_strictness'] = 4.5 

    return features_comb

def realizar_prediccion_maestra(equipo_local, equipo_visita):
    print(f"\n🔮 Consultando al Comité de IAs para: {equipo_local} vs {equipo_visita}")
    
    # 1. Cargar el CSV base (necesitamos la estructura de columnas exacta del entrenamiento)
    if not os.path.exists("chile_ml_ready_multitarget.csv"):
         print("❌ Error: No se encuentra 'chile_ml_ready_multitarget.csv'.")
         return
    
    df_modelo = pd.read_csv("chile_ml_ready_multitarget.csv")
    
    # Targets que hay que excluir de los features de entrada
    targets = ['target_1x2', 'target_goals', 'target_corners', 'target_cards']
    features_cols = [col for col in df_modelo.columns if col not in targets]

    # 2. Cargar el CSV crudo para buscar las estadísticas reales de los equipos
    # Excluimos los archivos procesados 'ml_ready' para no confundir al lector de estadísticas
    archivos_csv_crudos = [f for f in (glob.glob('*matches*.csv') + glob.glob('*chile*.csv')) if 'ml_ready' not in f]
    if not archivos_csv_crudos:
        print("❌ Error: No se encontró el CSV original de estadísticas.")
        return
    ruta_csv_crudo = max(archivos_csv_crudos, key=os.path.getctime)

    # 3. Importar y utilizar la función de extracción estricta (aislada) desde app_master.py
    # Para asegurar consistencia absoluta entre panel web e inferencia local
    try:
        from app_master import obtener_stats_aisladas
    except ImportError:
        print("❌ Error: No se pudo importar 'obtener_stats_aisladas' desde app_master.py.")
        return
        
    l_s = obtener_stats_aisladas(equipo_local, ruta_csv_crudo, 'Local')
    v_s = obtener_stats_aisladas(equipo_visita, ruta_csv_crudo, 'Visita')
    
    # Ensamblar exactamente como en el DataFrame de entrenamiento (chile_ml_ready_multitarget)
    vals = {
        'l_gf': l_s['gf'], 'l_gc': l_s['gc'], 'l_ppg': l_s['ppg'], 'l_pos': l_s['pos'], 'l_xg': l_s['xg'],
        'v_gf': v_s['gf'], 'v_gc': v_s['gc'], 'v_ppg': v_s['ppg'], 'v_pos': v_s['pos'], 'v_xg': v_s['xg']
    }
    
    # 4. Construir el tensor exacto
    X_infer = pd.DataFrame([vals])
    
    # Rellenar features extrañas/nuevas (como is_cup o lsi) que puedan existir en features_cols con 1.0 (neutro) o 0
    for col in features_cols:
        if col not in X_infer.columns:
            if 'lsi' in col:
                X_infer[col] = 1.0
            else:
                X_infer[col] = 0
                
    # Asegurar el orden exacto de columnas que el modelo entrenado exige
    X_infer = X_infer[features_cols]
    
    X_infer = X_infer.astype(float).fillna(0)
    dmatrix_infer = xgb.DMatrix(X_infer)

    # 4. Inferencia: Consultar los 4 modelos
    resultados = {}
    
    try:
        # Modelo 1X2
        modelo_1x2 = xgb.Booster()
        modelo_1x2.load_model("deep_soccer_1x2.model")
        probs_1x2 = modelo_1x2.predict(dmatrix_infer)[0]
        resultados['Victoria Local'] = f"{probs_1x2[0]:.2%}"
        resultados['Empate'] = f"{probs_1x2[1]:.2%}"
        resultados['Victoria Visita'] = f"{probs_1x2[2]:.2%}"

        # Modelo Goles
        modelo_goles = xgb.Booster()
        modelo_goles.load_model("deep_soccer_goals.model")
        resultados['Goles Esperados'] = round(float(modelo_goles.predict(dmatrix_infer)[0]), 2)

        # Modelo Córners
        if os.path.exists("deep_soccer_corners.model"):
            modelo_corners = xgb.Booster()
            modelo_corners.load_model("deep_soccer_corners.model")
            resultados['Córners Esperados'] = round(float(modelo_corners.predict(dmatrix_infer)[0]), 2)

        # Modelo Tarjetas
        if os.path.exists("deep_soccer_cards.model"):
            modelo_cards = xgb.Booster()
            modelo_cards.load_model("deep_soccer_cards.model")
            resultados['Tarjetas Esperadas'] = round(float(modelo_cards.predict(dmatrix_infer)[0]), 2)

        # 5. Imprimir el Reporte
        print("-" * 40)
        print("📊 REPORTE DEL COMITÉ DE IAs")
        print("-" * 40)
        for key, value in resultados.items():
            print(f" > {key}: {value}")
        print("-" * 40)

    except xgb.core.XGBoostError as e:
         print(f"❌ Error al cargar los modelos. Asegúrate de haber ejecutado 'advanced_model.py'. Detalle: {e}")

if __name__ == "__main__":
    # ¡Prueba con dos equipos que existan en tu CSV de la Primera A de Chile!
    equipo_a = "Colo-Colo" 
    equipo_b = "Universidad Chile"
    
    realizar_prediccion_maestra(equipo_a, equipo_b)
