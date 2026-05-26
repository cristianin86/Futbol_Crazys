import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import os
import json
import requests
import datetime
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

class CopaLigaModel:
    """
    Entorno aislado para el modelo de la Copa de la Liga de Chile.
    Implementa reglas exactas: Data Pooling, Penalización LSI por rotación y Flag de Motivación (Dead Rubbers).
    """

    def __init__(self, copa_weight=1.5, liga_weight=1.0):
        # 1. DATA POOLING (Fusión Ponderada)
        # Diferenciamos el peso para contexto (Liga) vs tendencia específica (Copa)
        self.copa_weight = copa_weight
        self.liga_weight = liga_weight
        self.model_1x2 = xgb.XGBClassifier(eval_metric='mlogloss')
        self.model_xg = xgb.XGBRegressor(eval_metric='rmse')
        self.is_trained = False

    def data_pooling(self, df_liga, df_copa):
        """
        Fusión ponderada de datos.
        Aplica el peso diferenciado en la columna 'weight_pool' para el entrenamiento.
        """
        print("=> Ejecutando Data Pooling (Fusión Ponderada)...")
        df_l = df_liga.copy()
        df_c = df_copa.copy()
        
        df_l['weight_pool'] = self.liga_weight
        df_c['weight_pool'] = self.copa_weight
        
        # Concatenar historial
        df_master = pd.concat([df_l, df_c], ignore_index=True)
        return df_master

    def calculate_lsi_penalty(self, ideal_xi_minutes, current_xi_minutes):
        """
        2. PENALIZACIÓN POR ROTACIÓN (Lineup Strength Index)
        Basado en volumen de minutos reales.
        
        ideal_xi_minutes: Sumatoria de minutos jugados en el año por los 11 más regulares de la Liga.
        current_xi_minutes: Sumatoria de minutos reales de los 11 titulares presentados en la Copa.
        
        Retorna el coeficiente LSI.
        """
        if ideal_xi_minutes == 0:
            return 1.0 # Evitar división por 0, no penalizar si no hay historial
            
        lsi_ratio = current_xi_minutes / ideal_xi_minutes
        
        # Ajuste cap para no sobreestimar si superan minutos por refuerzos (tope en 1.0)
        lsi_ratio = min(lsi_ratio, 1.0)
        
        return lsi_ratio

    def apply_dead_rubber_flag(self, xg_expected, is_eliminated):
        """
        3. FLAG DE MOTIVACIÓN (Dead Rubbers)
        Castiga el rendimiento esperado de equipos matemáticamente eliminados.
        """
        if is_eliminated:
            # Penalización del 25% en Expected Goals por desmotivación
            return xg_expected * 0.75 
        return xg_expected

    def adjust_predictions(self, team_name, native_xg, native_prob_win, is_eliminated):
        """
        Ajusta las predicciones nativas (que ya incluyen el LSI del XGBoost)
        aplicando únicamente el flag de Dead Rubbers.
        """
        # Aplicar Flag de Eliminación (Dead Rubbers)
        final_xg = self.apply_dead_rubber_flag(native_xg, is_eliminated)
        
        adjusted_prob_win = native_prob_win
        # Si xG cae, ajustamos la probabilidad de victoria proporcionalmente
        if is_eliminated:
            xg_drop_ratio = final_xg / native_xg if native_xg > 0 else 1.0
            adjusted_prob_win = native_prob_win * xg_drop_ratio
            
        print(f"[{team_name}] Ajuste Copa de la Liga (Dead Rubber):")
        print(f"   -> Eliminado: {is_eliminated}")
        print(f"   -> xG Nativo (LSI): {native_xg:.2f} => Final: {final_xg:.2f}")
        
        return final_xg, adjusted_prob_win

    def train_pool_model(self, df_liga, df_copa, features, target_1x2, target_xg):
        """
        Entrena el modelo usando la fusión ponderada (Weight Pooling).
        """
        df_train = self.data_pooling(df_liga, df_copa)
        
        X = df_train[features]
        y_1x2 = df_train[target_1x2]
        y_xg = df_train[target_xg]
        weights = df_train['weight_pool']
        
        print("\n=> Entrenando Modelos (Aislado para Copa de la Liga)...")
        # Entrenar modelo 1X2
        self.model_1x2.fit(X, y_1x2, sample_weight=weights)
        # Entrenar modelo xG
        self.model_xg.fit(X, y_xg, sample_weight=weights)
        
        self.is_trained = True
        print("=> Modelos entrenados correctamente bajo Data Pooling.")

class CopaDataPipeline:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv("APISPORTS_KEY")
        self.headers = {
            'x-apisports-key': self.api_key,
            'x-rapidapi-host': "v3.football.api-sports.io"
        }
        self.cache_file = "players_minutes_cache.json"

    def get_ideal_xi_minutes(self, team_id, league_id="265", season="2026"):
        """
        1. PIPELINE DE MINUTOS (El Baseline):
        Extrae y almacena estadísticas acumuladas de jugadores en la Liga.
        Retorna el Ideal_XI_Minutes para el equipo (suma de los 11 más regulares).
        Implementa caché para no saturar llamadas a la API.
        """
        # Intentar cargar caché
        cache_data = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                
                # Revisar si el caché es reciente (menos de 7 días)
                last_updated_str = cache_data.get('last_updated', '2000-01-01T00:00:00')
                last_updated = datetime.datetime.fromisoformat(last_updated_str)
                if (datetime.datetime.now() - last_updated).days < 7:
                    team_str = str(team_id)
                    if team_str in cache_data.get('teams', {}):
                        print(f"[Caché] Cargando Ideal_XI_Minutes para equipo {team_id}")
                        return cache_data['teams'][team_str]['ideal_xi_minutes'], cache_data['teams'][team_str]['players']
            except Exception as e:
                print(f"Error leyendo caché: {e}. Se regenerará.")

        print(f"Obteniendo datos de jugadores para el equipo {team_id} desde API...")
        url = "https://v3.football.api-sports.io/players"
        page = 1
        all_players = []
        
        while True:
            params = {"team": team_id, "league": league_id, "season": season, "page": page}
            res = requests.get(url, headers=self.headers, params=params)
            data = res.json()
            
            if not data or not data.get('response'):
                break
                
            for p_data in data['response']:
                player = p_data['player']
                minutes = 0
                for stat in p_data['statistics']:
                    # Verificar que la estadística corresponda a la liga y temporada solicitadas
                    if str(stat['league']['id']) == str(league_id) and str(stat['league']['season']) == str(season):
                        games = stat.get('games', {})
                        minutes = games.get('minutes') or 0
                
                all_players.append({
                    "id": player['id'],
                    "name": player['name'],
                    "minutes": minutes
                })
            
            paging = data.get('paging', {})
            if paging.get('current', 1) >= paging.get('total', 1):
                break
            page += 1

        # Ordenar por minutos jugados y tomar los 11 primeros (el Baseline)
        top_11 = sorted(all_players, key=lambda x: x['minutes'], reverse=True)[:11]
        ideal_xi_minutes = sum([p['minutes'] for p in top_11])
        
        player_dict = {str(p['id']): p['minutes'] for p in all_players}

        # Actualizar caché
        if 'teams' not in cache_data:
            cache_data['teams'] = {}
            
        cache_data['teams'][str(team_id)] = {
            'ideal_xi_minutes': ideal_xi_minutes,
            'players': player_dict # mapeo de player_id a minutos jugados en la liga
        }
        cache_data['last_updated'] = datetime.datetime.now().isoformat()
        
        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)
            
        return ideal_xi_minutes, player_dict

    def get_current_xi_minutes(self, fixture_id, team_id, player_minutes_dict):
        """
        2. PIPELINE DE FORMACIONES (El Match Day):
        Extrae alineación para el partido específico de la Copa.
        Cruza los IDs de los 11 titulares confirmados con la base de datos de la Liga
        para calcular el Current_XI_Minutes real.
        """
        print(f"Consultando formaciones para fixture {fixture_id}, equipo {team_id}...")
        url = "https://v3.football.api-sports.io/fixtures/lineups"
        params = {"fixture": fixture_id, "team": team_id}
        res = requests.get(url, headers=self.headers, params=params)
        data = res.json()
        
        if not data or not data.get('response'):
            print("No hay alineaciones disponibles para este partido todavía. Aplicando SEGURO PRE-MATCH (LSI = 1.0)")
            # Retornamos los mismos minutos ideales para forzar LSI 1.0 de forma segura.
            return sum(player_minutes_dict.values()) # Fallback ultra seguro si no se pasó el valor ideal, aunque matemáticamente es mejor un flag
            # El fix en app_master.py intercepta el -1 si existe, pero para mayor pureza a nivel módulo, podemos emitir -1 aquí
            # para que la clase madre CopaLigaModel lo intercepte. Lo dejaremos como -1 y el orquestador master lo purificará.
            return -1
            
        lineup = data['response'][0]
        startXI = lineup.get('startXI', [])
        
        if not startXI:
            print("Alineaciones recibidas pero vacías. Aplicando SEGURO PRE-MATCH (LSI = 1.0)")
            return -1
            
        current_xi_minutes = 0
        print(f"--- Titulares Encontrados (Fixture {fixture_id}) ---")
        for player_obj in startXI:
            pid = str(player_obj['player']['id'])
            pname = player_obj['player']['name']
            # Sumar minutos de este jugador (usando el diccionario de la Liga)
            mins_in_liga = player_minutes_dict.get(pid, 0)
            current_xi_minutes += mins_in_liga
            print(f"  - {pname} (Liga mins: {mins_in_liga})")
            
        return current_xi_minutes

# Ejemplo de uso/Testing stub
if __name__ == "__main__":
    print("Iniciando Módulo: Copa de la Liga de Chile (Aislado)")
    
    # 1. Instanciar Modelos y Pipeline
    modelo_copa = CopaLigaModel(copa_weight=1.5, liga_weight=1.0)
    pipeline = CopaDataPipeline()
    
    # 2. Prueba con un equipo y partido simulado (ej: U de Chile en un fixture real si conocemos los IDs)
    # Por temas de test local, podemos usar variables de simulación si no pasamos IDs válidos.
    # U. de Chile ID en API-Sports suele ser 228 (por ejemplo). Fixture inventado.
    team_id_test = "228" # U. de Chile (ejemplo)
    
    # Intento real de extraer Ideal_XI_Minutes
    if pipeline.api_key:
        print("\n--- Ejecutando Prueba Real del Data Pipeline ---")
        try:
            ideal_min, player_dict = pipeline.get_ideal_xi_minutes(team_id_test, league_id="265", season="2026")
            print(f"-> Ideal_XI_Minutes (Baseline) calculado: {ideal_min} min")
            
            # Para la simulación del Current_XI_Minutes simulamos que no hay fixture ID disponible aún o usamos 0
            # Si no hay lineup disponible devolverá 0, simulamos una caída usando un valor ficticio:
            current_min_simulado = ideal_min * 0.45 # Simulamos que juegan suplentes (LSI al 45%)
            
            print(f"\nAplicando LSI Penalty Model:")
            modelo_copa.adjust_predictions(
                team_name="U. de Chile (Prueba Pipeline Real)",
                raw_xg=1.80,
                raw_prob_win=0.55,
                ideal_xi_minutes=ideal_min,
                current_xi_minutes=current_min_simulado,
                is_eliminated=False
            )
        except Exception as e:
            print(f"Error en prueba real: {e}")
    else:
        print("Sin API Key disponible en .env para prueba de integración real.")
