import requests
import pandas as pd
import os
import time
import datetime
from dotenv import load_dotenv
from copa_liga_model import CopaDataPipeline

def calcular_ventana_jornada():
    """
    Calcula automaticamente el rango Viernes-Lunes de la semana actual.
    Sirve para filtrar el radar en la jornada vigente.
    """
    hoy = datetime.date.today()
    # Buscamos el viernes de esta semana (dia 4: viernes)
    distancia_al_viernes = hoy.weekday() - 4
    viernes = hoy - datetime.timedelta(days=distancia_al_viernes)
    # El cierre es el lunes (viernes + 3 dias)
    lunes = viernes + datetime.timedelta(days=3)
    return viernes.isoformat(), lunes.isoformat()

# Cargar variables de entorno
load_dotenv(override=True)
API_KEY = os.getenv("APISPORTS_KEY")

def detectar_jornada_actual(api_key, league_id="265", season="2026"):
    """
    Consulta a la API cual es la jornada marcada como 'current'.
    """
    url = "https://v3.football.api-sports.io/fixtures/rounds"
    headers = {'x-apisports-key': api_key, 'x-rapidapi-host': "v3.football.api-sports.io"}
    try:
        res = requests.get(url, headers=headers, params={"league": league_id, "season": season, "current": "true"}, timeout=10)
        rounds = res.json().get('response', [])
        return rounds[0] if rounds else "Regular Season - 1"
    except:
        return "Regular Season - 1"

def descargar_historial_api(api_key, league_id="265", seasons=["2024", "2025", "2026"], output_file="chile_api_raw.csv", is_cup=0):
    """
    Descarga el inventario de la temporada para una liga específica (soporta múltiples temporadas).
    """
    if not api_key:
        print("ERROR: No se encontro la APISPORTS_KEY", flush=True)
        return None

    pipeline = CopaDataPipeline()
    headers = {'x-apisports-key': api_key, 'x-rapidapi-host': "v3.football.api-sports.io"}
    url_fixtures = "https://v3.football.api-sports.io/fixtures"
    
    data_list = []
    
    # Asegurar que seasons sea una lista
    if isinstance(seasons, str):
        seasons = [seasons]

    for season in seasons:
        print(f"--- Descargando temporada {season} (Liga {league_id}) ---", flush=True)
        querystring = {"league": league_id, "season": season}
        
        try:
            res = requests.get(url_fixtures, headers=headers, params=querystring, timeout=15)
            partidos_raw = res.json().get('response', [])
            print(f"Se extrajeron {len(partidos_raw)} partidos.", flush=True)
            
            for i, match in enumerate(partidos_raw):
                f_status = match['fixture']['status']['short']
                f_id = match['fixture']['id']
                
                local_name = match['teams']['home']['name']
                visita_name = match['teams']['away']['name']
                local_id = match['teams']['home']['id']
                visita_id = match['teams']['away']['id']
                
                s_home, s_away = {}, {}
                home_lsi = 1.0
                away_lsi = 1.0
                
                if f_status == "FT":
                    try:
                        stats_url = "https://v3.football.api-sports.io/fixtures/statistics"
                        s_res = requests.get(stats_url, headers=headers, params={"fixture": f_id}, timeout=10)
                        s_data = s_res.json().get('response', [])
                        if len(s_data) >= 2:
                            s_home = {item['type']: item['value'] for item in s_data[0]['statistics']}
                            s_away = {item['type']: item['value'] for item in s_data[1]['statistics']}
                            
                        if is_cup == 1:
                            ideal_L, dict_L = pipeline.get_ideal_xi_minutes(local_id, league_id, season)
                            ideal_V, dict_V = pipeline.get_ideal_xi_minutes(visita_id, league_id, season)
                            
                            curr_L = pipeline.get_current_xi_minutes(f_id, local_id, dict_L)
                            curr_V = pipeline.get_current_xi_minutes(f_id, visita_id, dict_V)
                            
                            home_lsi = min(curr_L / ideal_L if ideal_L > 0 else 1.0, 1.0)
                            away_lsi = min(curr_V / ideal_V if ideal_V > 0 else 1.0, 1.0)
                    except: pass
                    # time.sleep(0.01) # Muy pequeño delay para evitar rate limits pero ser rápido

                def clean_stat(val, default=0):
                    if val is None: return default
                    if isinstance(val, str) and "%" in val: return float(val.replace("%", ""))
                    try: return float(val)
                    except: return default

                row = {
                    "home_team_name": local_name,
                    "away_team_name": visita_name,
                    "home_team_goal_count": match['goals']['home'],
                    "away_team_goal_count": match['goals']['away'],
                    "status": "complete" if f_status == "FT" else "upcoming",
                    "timestamp": match['fixture']['timestamp'],
                    "referee": match['fixture'].get('referee', 'PENDIENTE'),
                    "home_team_possession": clean_stat(s_home.get("Ball Possession"), 50),
                    "away_team_possession": clean_stat(s_away.get("Ball Possession"), 50),
                    "home_team_corner_count": clean_stat(s_home.get("Corner Kicks")),
                    "away_team_corner_count": clean_stat(s_away.get("Corner Kicks")),
                    "home_team_yellow_cards": clean_stat(s_home.get("Yellow Cards")),
                    "away_team_yellow_cards": clean_stat(s_away.get("Yellow Cards")),
                    "home_team_pre_match_xg": clean_stat(s_home.get("expected_goals"), 1.0),
                    "away_team_pre_match_xg": clean_stat(s_away.get("expected_goals"), 1.0),
                    "is_cup": is_cup,
                    "home_team_lsi": home_lsi,
                    "away_team_lsi": away_lsi
                }
                data_list.append(row)
                if i % 50 == 0: print(f"Procesando {i}/{len(partidos_raw)} de {season}...", flush=True)
                
        except Exception as e:
            print(f"Error procesando liga {league_id} temporada {season}: {e}", flush=True)

    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)
    print(f"Sincronizacion completa en: {output_file} ({len(df)} registros)", flush=True)
    return df

if __name__ == "__main__":
    import sys
    
    l_id = sys.argv[1] if len(sys.argv) > 1 else "265"
    out_f = sys.argv[2] if len(sys.argv) > 2 else "chile_api_raw.csv"
    cup_flag = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    seasons_to_get = ["2024", "2025", "2026"]
    
    descargar_historial_api(API_KEY, league_id=l_id, seasons=seasons_to_get, output_file=out_f, is_cup=cup_flag)
