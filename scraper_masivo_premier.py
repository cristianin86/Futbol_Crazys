import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# 1. GESTIÓN DE CREDENCIALES
load_dotenv(override=True)
API_KEY = os.getenv("APISPORTS_KEY")

def descargar_masivo_premier(seasons=["2024", "2025"]):
    """
    Descarga el historial masivo de la Premier League (ID 39) para las temporadas indicadas.
    Guarda los datos con la estructura exacta para processor.py.
    """
    if not API_KEY:
        print("ERROR: No se encontró la APISPORTS_KEY en el archivo .env")
        return

    league_id = "39"
    headers = {
        'x-apisports-key': API_KEY,
        'x-rapidapi-host': "v3.football.api-sports.io"
    }

    all_data = []

    for season in seasons:
        print(f"\n--- Iniciando Extracción Masiva: Premier League {season} ---")
        url_fixtures = "https://v3.football.api-sports.io/fixtures"
        params_fixtures = {"league": league_id, "season": season, "status": "FT"} # Solo terminados

        try:
            response = requests.get(url_fixtures, headers=headers, params=params_fixtures, timeout=15)
            datos = response.json()
            
            if datos.get('errors'):
                print(f"ERROR API ({season}): {datos['errors']}")
                continue

            partidos_raw = datos.get('response', [])
            print(f"Se detectaron {len(partidos_raw)} partidos finalizados.")

            for i, match in enumerate(partidos_raw):
                f_id = match['fixture']['id']
                local_name = match['teams']['home']['name']
                visita_name = match['teams']['away']['name']
                
                # Obtener estadísticas detalladas del partido
                stats_url = "https://v3.football.api-sports.io/fixtures/statistics"
                s_home, s_away = {}, {}
                
                try:
                    s_res = requests.get(stats_url, headers=headers, params={"fixture": f_id}, timeout=10)
                    s_data = s_res.json().get('response', [])
                    if len(s_data) >= 2:
                        s_home = {item['type']: item['value'] for item in s_data[0]['statistics']}
                        s_away = {item['type']: item['value'] for item in s_data[1]['statistics']}
                except Exception as e:
                    print(f"  Warning: No se pudieron obtener stats para fixture {f_id}: {e}")

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
                    "status": "complete",
                    "timestamp": match['fixture']['timestamp'],
                    "referee": match['fixture'].get('referee', "N/A"),
                    "home_team_possession": clean_stat(s_home.get("Ball Possession"), 50),
                    "away_team_possession": clean_stat(s_away.get("Ball Possession"), 50),
                    "home_team_corner_count": clean_stat(s_home.get("Corner Kicks")),
                    "away_team_corner_count": clean_stat(s_away.get("Corner Kicks")),
                    "home_team_yellow_cards": clean_stat(s_home.get("Yellow Cards")),
                    "away_team_yellow_cards": clean_stat(s_away.get("Yellow Cards")),
                    "home_team_pre_match_xg": clean_stat(s_home.get("expected_goals"), 1.0),
                    "away_team_pre_match_xg": clean_stat(s_away.get("expected_goals"), 1.0),
                }
                all_data.append(row)
                
                if (i + 1) % 20 == 0:
                    print(f"  Progreso {season}: {i+1}/{len(partidos_raw)} partidos procesados...")
                
                # Delay para respetar Rate Limits (ajustar segun plan)
                time.sleep(0.1)

        except Exception as e:
            print(f"ERROR CRÍTICO en temporada {season}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        output_file = "premier_api_raw.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ PROCESO COMPLETADO")
        print(f"Archivo generado: {output_file}")
        print(f"Total de registros: {len(df)}")
    else:
        print("\n❌ No se extrajo ninguna información.")

if __name__ == "__main__":
    descargar_masivo_premier()
