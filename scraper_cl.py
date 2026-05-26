import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import re
import unicodedata
import time
import os
from datetime import datetime, timedelta, timezone, date
import datetime as dt_module
from dotenv import load_dotenv

# Cargar variables de entorno localmente
load_dotenv(override=True)

def get_api_sports_key():
    """Obtiene la llave de forma dinámica para evitar desincronizaciones en Cloud."""
    try:
        import streamlit as st
        if "APISPORTS_KEY" in st.secrets:
            return st.secrets["APISPORTS_KEY"]
        elif "API_SPORTS_KEY" in st.secrets:
            return st.secrets["API_SPORTS_KEY"]
    except:
        pass
    
    return os.getenv("APISPORTS_KEY") or os.getenv("API_SPORTS_KEY")

# Asignación inicial (fallback)
API_KEY_SPORTS = get_api_sports_key()
# --- DICCIONARIO DINÁMICO DE ÁRBITROS CHILENOS (V31.0) ---
def get_referee_metrics(name):
    """
    Busca el nombre del árbitro en el diccionario dinámico y calcula el delta.
    Si no existe, retorna valores por defecto (5.0).
    """
    # Base de datos expandida V32.0 (Incluye formatos API y Nombres Completos)
    referee_db = {
        # Chile
        "Piero Maza": 6.1, "José Cabero": 5.4, "Fernando Véjar": 5.8, 
        "Francisco Gilabert": 4.5, "Héctor Jona": 5.2, "Cristián Garay": 4.9, 
        "Felipe González": 5.5, "Juan Lara": 5.1,
        
        # Premier League (Nombres Completos)
        "Michael Oliver": 3.8, "Anthony Taylor": 4.2, "Stuart Attwell": 4.8,
        "Craig Pawson": 4.5, "Simon Hooper": 4.0, "Chris Kavanagh": 3.9,
        "Jarred Gillett": 4.1, "Robert Jones": 3.7, "Tim Robinson": 4.3,
        
        # Premier League (Formatos API S. Attwell)
        "M. Oliver": 3.8, "A. Taylor": 4.2, "S. Attwell": 4.8,
        "C. Pawson": 4.5, "S. Hooper": 4.0, "C. Kavanagh": 3.9,
        "J. Gillett": 4.1, "R. Jones": 3.7, "T. Robinson": 4.3
    }
    
    if not name or name == "Por Designar":
        return {"media": 5.0, "tendencia": 0.0, "real": False}
        
    norm_target = normalize_text(name)
    
    # Búsqueda exacta normalizada
    for ref_name, val in referee_db.items():
        if normalize_text(ref_name) == norm_target:
            return {
                "media": val, 
                "tendencia": val - 5.0,
                "real": True
            }
            
    # Búsqueda por coincidencia parcial (Ej: "Stuart Attwell, England" -> "Stuart Attwell")
    for ref_name, val in referee_db.items():
        if normalize_text(ref_name) in norm_target or norm_target in normalize_text(ref_name):
            return {
                "media": val,
                "tendencia": val - 5.0,
                "real": True
            }

    return {"media": 5.0, "tendencia": 0.0, "real": False}

def normalize_text(text, translator=None):
    """Normaliza texto eliminando acentos y pasando a minúsculas, usando un traductor opcional."""
    if not text: return ""
    text = text.lower().strip()
    
    # Aplicar traducciones específicas si existen
    if translator:
        for key, val in translator.items():
            if key in text:
                return val
    
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def get_active_season(league_id):
    """
    Asigna automáticamente el año de la temporada según la liga.
    Chile (265): Calendario Anual (2026).
    Premier (39): Calendario Europeo (2025 para la 25/26).
    """
    hoy = date.today()
    año = hoy.year
    mes = hoy.month
    
    # Ligas Sudamericanas (Chile, Brasil, etc.)
    if str(league_id) in ["265"]:
        return str(año)
    
    # Ligas Europeas (Premier, LaLiga, etc.)
    # Si es Enero-Junio, la temporada activa empezó el año pasado.
    if str(league_id) in ["39"]:
        return str(año - 1) if mes <= 6 else str(año)
        
    return str(año)

def radar_api_sports(api_key, league_id="265", season=None):
    """
    Radar Táctico conectado a API-Sports v3.
    """
    if not api_key: return []
    url = "https://v3.football.api-sports.io/fixtures"
    
    # Auditoría Automática de Temporada
    if not season or season == "auto":
        season = get_active_season(league_id)
    
    # Barrido de Tiempo Estricto: Hoy (T-0) hasta T+15
    hoy_dt = datetime.now()
    fecha_inicio = hoy_dt.strftime("%Y-%m-%d")
    fecha_fin = (hoy_dt + timedelta(days=15)).strftime("%Y-%m-%d")
    
    print(f"📡 API CALL: League={league_id} | Season={season} | Window={fecha_inicio} to {fecha_fin}")

    querystring = {
        "league": str(league_id),
        "season": str(season),
        "from": fecha_inicio,
        "to": fecha_fin
    }
    headers = {
        'x-apisports-key': api_key,
        'x-rapidapi-host': "v3.football.api-sports.io"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        datos = response.json()
        if datos.get('errors'): return []
        
        partidos_reales = datos.get('response', [])
        if not partidos_reales:
            print(f"DEBUG: La API no devolvió la llave 'response' o está vacía para liga {league_id}.")
            return []
            
        fixture_live = []
        for match in partidos_reales:
            round_raw = match.get('league', {}).get('round', 'Unknown')
            jornada = round_raw.replace("Regular Season - ", "Fecha ")
            
            fixture_live.append({
                "id": match['fixture']['id'],
                "date": match['fixture']['date'],
                "league_logo": match['league'].get('logo', ""),
                "local": match['teams']['home']['name'],
                "local_id": match['teams']['home']['id'],
                "local_logo": match['teams']['home'].get('logo', ""),
                "visita": match['teams']['away']['name'],
                "visita_id": match['teams']['away']['id'],
                "visita_logo": match['teams']['away'].get('logo', ""),
                "score_h": match['goals']['home'],
                "score_a": match['goals']['away'],
                "status_short": match['fixture']['status']['short'],
                "minuto": match['fixture']['status']['elapsed'],
                "jornada": jornada,
                "referee": match['fixture'].get('referee', "Por Designar"),
                "venue_name": match['fixture'].get('venue', {}).get('name', "Sede Oficial"),
                "venue_city": match['fixture'].get('venue', {}).get('city', "Ciudad TBD")
            })
        return fixture_live
    except:
        return []

def get_live_fixtures(league_id="265", season=None, translator=None):
    """
    Motor V30.0: Sincronización dinámica multi-liga.
    """
    try:
        # 1. Obtener base de partidos desde la ventana de la API
        api_matches = radar_api_sports(API_KEY_SPORTS, league_id, season)
        if not api_matches: return pd.DataFrame()

        # 2. Enriquecimiento Condicional (Solo Chile por ahora)
        html_map = {}
        if league_id == "265":
            try:
                base_url = "https://www.campeonatochileno.cl/ligas/liga-de-primera-mercado-libre/"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                r = requests.get(base_url, headers=headers, timeout=10)
                soup = BeautifulSoup(r.text, 'html.parser')
                html_items = soup.select('.match-slim')
                for item in html_items:
                    try:
                        h = item.select_one('.match-slim__team-home-title').get_text().strip().title()
                        a = item.select_one('.match-slim__team-away-title').get_text().strip().title()
                        key = f"{normalize_text(h, translator)}-{normalize_text(a, translator)}"
                        html_map[key] = item
                    except: continue
            except: pass
        
        partidos_finales = []
        for m in api_matches:
            local = m['local']
            visita = m['visita']
            loc_norm = normalize_text(local, translator)
            vis_norm = normalize_text(visita, translator)
            match_key = f"{loc_norm}-{vis_norm}"

            # Datos base API
            # Estandarización de Status (V32.0)
            status_api = m['status_short']
            if status_api == "NS":
                status = "upcoming"
            elif status_api in ["FT", "AET", "PEN"]:
                status = "finished"
            elif status_api in ["1H", "HT", "2H", "ET", "P", "LIVE"]:
                status = "live"
            else:
                status = "upcoming"
            score_h, score_a = m['score_h'], m['score_a']
            hora_label = f"DIRECTO {m['minuto']}'" if status == "live" else "Programado"

            # Enriquecimiento (Prioridad API, fallback HTML)
            arbitro = m['referee']
            estadio = m['venue_name']
            
            if match_key in html_map:
                item = html_map[match_key]
                arbitro = item.select_one('.match-slim__referee').get_text().strip() if item.select_one('.match-slim__referee') else arbitro
                estadio = item.select_one('.match-slim__stadium').get_text().strip() if item.select_one('.match-slim__stadium') else estadio
                if status == "upcoming":
                    time_elem = item.select_one('.match-slim__time')
                    hora_label = time_elem.get_text().strip() if time_elem else hora_label
            
            # --- FILTRO DE LÍNEA DE TIEMPO RADICAL (Simplificación Date-Only con soporte UTC Cloud) ---
            raw_date_str = m.get('date', '')
            if not raw_date_str: continue
            
            # 1. Cortar YYYY-MM-DD e ignorar horas/timezones
            try:
                match_day_str = raw_date_str[:10]
                match_date_obj = datetime.strptime(match_day_str, "%Y-%m-%d").date()
            except:
                continue

            # 2. Límites locales (Restamos 1 día para absorber el desfase de servidores UTC)
            today_local = date.today() - timedelta(days=1)
            limit_date = today_local + timedelta(days=16)
            
            status_api = m['status_short']
            
            # 3. Condición de Oro Simplificada
            # Permitir si está en el rango de fechas (Ayer hasta +16 días) y no es estado finalizado/cancelado
            # También permitimos estados in-play aunque la fecha sea hoy
            if today_local <= match_date_obj <= limit_date:
                if status_api in ['FT', 'AET', 'PEN', 'CANC', 'PST', 'ABD']:
                    continue
            else:
                # Si está fuera del rango de fechas, solo lo permitimos si está LIVE ahora mismo
                if status_api not in ["1H", "HT", "2H", "ET", "P", "LIVE"]:
                    continue

            fecha_cal = match_date_obj.strftime("%d/%m")

            jornada = m.get('jornada', 'N/A')
            partidos_finales.append({
                "id": m['id'],
                "League_Logo": m.get('league_logo', ""),
                "Local": local, "Visita": visita, 
                "Local_ID": m['local_id'], "Visita_ID": m['visita_id'],
                "Local_Logo": m.get('local_logo', ""), "Visita_Logo": m.get('visita_logo', ""),
                "Arbitro": arbitro, 
                "Estadio": estadio,
                "Ciudad": m.get('venue_city', "Ciudad TBD"),
                "Status": status, "Status_Label": "EN VIVO" if status == "live" else "PROGRAMADO",
                "Marcador": f"{score_h} - {score_a}", "Health_Score": 100,
                "minuto": m.get('minuto', 0),
                "Fecha": fecha_cal,
                "Partido_String": f"[{jornada}] {fecha_cal} | {local} vs {visita} ({hora_label})"
            })

        return pd.DataFrame(partidos_finales)
    except Exception as e:
        print(f"Error Scraper V30.0: {e}")
        return pd.DataFrame()

import streamlit as st

@st.cache_data(ttl=300)
def get_odds(fixture_id):
    """Obtiene cuotas multi-mercado comparativas (Betano, Bet365, Pinnacle)."""
    if not API_KEY_SPORTS: return {}
    headers = {
        'x-apisports-key': API_KEY_SPORTS,
        'x-rapidapi-host': "v3.football.api-sports.io"
    }
    url = "https://v3.football.api-sports.io/odds"
    
    # Bookmaker IDs prioritarios: 11=Betano, 8=Bet365, 3=Pinnacle
    bookmaker_ids = {"11": "Betano", "8": "Bet365", "3": "Pinnacle"}
    
    # Estructura comparativa
    resultado = {
        '1x2': {},        # {'Home': {'Betano': 1.85, ...}, 'Draw': {...}, 'Away': {...}}
        'dc': {},         # {'Home/Draw': {'Betano': 1.25, ...}, ...}
        'ou_goals': {},   # {'2.5': {'over': {'Betano': 1.90, ...}, 'under': {...}}}
        'ou_corners': {}, # {'9.5': {'over': {...}, 'under': {...}}}
        'ou_cards': {},   # {'4.5': {'over': {...}, 'under': {...}}}
    }
    
    for bk_id, bk_name in bookmaker_ids.items():
        try:
            querystring = {"fixture": str(fixture_id), "bookmaker": bk_id}
            response = requests.get(url, headers=headers, params=querystring, timeout=10)
            data = response.json()
            
            if not data.get('response') or len(data['response']) == 0:
                continue
            
            bookmakers = data['response'][0].get('bookmakers', [])
            if not bookmakers:
                continue
                
            bets = bookmakers[0].get('bets', [])
            
            for bet in bets:
                bet_name = bet.get('name', '').lower()
                values = bet.get('values', [])
                
                # 1X2 (Match Winner)
                if 'match winner' in bet_name:
                    for v in values:
                        outcome = v['value'] # 'Home', 'Draw', 'Away'
                        if outcome not in resultado['1x2']: resultado['1x2'][outcome] = {}
                        resultado['1x2'][outcome][bk_name] = float(v['odd'])

                # Double Chance
                if 'double chance' in bet_name:
                    for v in values:
                        outcome = v['value'] # 'Home/Draw', 'Home/Away', 'Draw/Away'
                        # Estandarizar nombres
                        outcome = outcome.replace(' or ', '/').replace(' ', '')
                        if outcome not in resultado['dc']: resultado['dc'][outcome] = {}
                        resultado['dc'][outcome][bk_name] = float(v['odd'])
                
                # Over/Under (Goals, Corners, Cards)
                is_goals = 'over/under' in bet_name and 'corner' not in bet_name and 'card' not in bet_name
                is_corners = 'corner' in bet_name
                is_cards = 'card' in bet_name or 'booking' in bet_name
                
                if is_goals or is_corners or is_cards:
                    cat = 'ou_goals' if is_goals else ('ou_corners' if is_corners else 'ou_cards')
                    lines = {}
                    for v in values:
                        val_str = v['value']  # "Over 2.5"
                        parts = val_str.split(' ')
                        if len(parts) == 2:
                            direction, line = parts[0].lower(), parts[1]
                            # FILTRO ESTRICTO .5 (Solo líneas estándar)
                            if not line.endswith('.5'):
                                continue
                                
                            if line not in resultado[cat]:
                                resultado[cat][line] = {'over': {}, 'under': {}}
                            resultado[cat][line][direction][bk_name] = float(v['odd'])
                
        except Exception:
            continue
    
    return resultado

def get_live_odds(fixture_id):
    """
    Obtiene cuotas en vivo (In-Play) desde API-Sports.
    """
    if not API_KEY_SPORTS: return {}
    headers = {
        'x-apisports-key': API_KEY_SPORTS,
        'x-rapidapi-host': "v3.football.api-sports.io"
    }
    url = "https://v3.football.api-sports.io/odds/live"
    querystring = {"fixture": str(fixture_id)}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        data = response.json()
        if not data.get('response') or len(data['response']) == 0: 
            return {}
        
        # Extraer cuotas del primer (y usualmente único) registro de respuesta
        odds_raw = data['response'][0].get('odds', [])
        resultado = {}
        for bet in odds_raw:
            bet_name = bet.get('name')
            resultado[bet_name] = {v['value']: float(v['odd']) for v in bet['values']}
        return resultado
    except Exception as e:
        print(f"Error Live Odds: {e}")
        return {}

def recalculate_live_ev(prob_inicial, minuto_actual):
    """
    Recálculo Time-Decay: Ajusta la probabilidad base según el tiempo restante.
    Usa un decaimiento lineal simple para el MVP.
    """
    if minuto_actual >= 90: return 0.0
    
    # Factor de tiempo restante (de 1.0 a 0.0)
    tiempo_restante_pct = (90 - minuto_actual) / 90
    
    # Ajuste: La probabilidad de que ocurra un evento (ej. ganar) 
    # suele decaer a medida que queda menos tiempo si no se ha cumplido.
    # Para el 1X2, esto es más complejo, pero para el MVP usaremos decaimiento lineal.
    prob_ajustada = prob_inicial * tiempo_restante_pct
    
    return prob_ajustada

def calculate_kelly(prob, odds, bankroll, fraction=4):
    """
    Calculadora de Gestión de Capital (Criterio de Kelly).
    f* = (p * b - q) / b
    """
    if odds <= 1: return 0
    b = odds - 1
    p = prob
    q = 1 - p
    f_star = (p * b - q) / b
    return max(0, (f_star / fraction) * bankroll)

def get_combined_probs(p_win):
    """
    Calcula las probabilidades combinadas de nuestra IA: 
    1X (Local+Empate), X2 (Visita+Empate), 12 (Local+Visita).
    """
    if len(p_win) < 3: return {"1X": 0.5, "X2": 0.5, "12": 0.5}
    p_l, p_e, p_v = p_win[0], p_win[1], p_win[2]
    return {
        "1X": p_l + p_e,
        "X2": p_e + p_v,
        "12": p_l + p_v
    }

def get_injuries(fixture_id):
    """Obtiene las bajas (lesiones/suspensiones) para un fixture."""
    if not API_KEY_SPORTS: return []
    url = "https://v3.football.api-sports.io/injuries"
    querystring = {"fixture": fixture_id}
    headers = {
        'x-apisports-key': API_KEY_SPORTS,
        'x-rapidapi-host': "v3.football.api-sports.io"
    }
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        data = response.json()
        return data.get('response', [])
    except:
        return []

def check_lineups_status(fixture_id):
    """Verifica si ya hay formaciones confirmadas en la API-Sports."""
    if not API_KEY_SPORTS: return False
    url = "https://v3.football.api-sports.io/fixtures/lineups"
    headers = {'x-apisports-key': API_KEY_SPORTS, 'x-rapidapi-host': "v3.football.api-sports.io"}
    params = {"fixture": str(fixture_id)}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()
        if data.get('response') and len(data['response']) >= 2:
            return True
        return False
    except:
        return False

def get_absence_impact(fixture_id, team_id):
    """
    Escáner de Ausencias: Identifica jugadores clave ausentes.
    """
    injuries = get_injuries(fixture_id)
    if not injuries: return []
    
    bajas_clave = []
    for injury in injuries:
        if injury['team']['id'] == team_id:
            # Marcamos como clave si es delantero o mediocentro ofensivo (simplificado)
            player = injury['player']
            reason = injury['fixture'].get('comment', 'Lesión')
            is_key = any(word in player.get('type', '').lower() for word in ['attacker', 'midfielder']) or player.get('name') in ['Vidal', 'Sanchez', 'Zampedri']
            
            bajas_clave.append({
                "name": player['name'],
                "reason": reason,
                "key": is_key
            })
    return bajas_clave

def obtener_dossier_360(fixture_id, local_id, visita_id, league_id, season):
    """
    Radiografía 360°: Extrae contexto táctico y estadístico profundo de API-Sports.
    """
    if not API_KEY_SPORTS: return {}
    headers = {'x-apisports-key': API_KEY_SPORTS, 'x-rapidapi-host': "v3.football.api-sports.io"}
    base_url = "https://v3.football.api-sports.io"
    
    dossier = {
        'posicion_local': 'N/A', 'posicion_visita': 'N/A',
        'forma_local': 'N/A', 'forma_visita': 'N/A',
        'formacion_local': 'TBD', 'formacion_visita': 'TBD',
        'posesion_local': 50, 'posesion_visita': 50,
        'bajas_local': 'Sin reportes', 'bajas_visita': 'Sin reportes',
        'resumen_h2h': 'Sin historial reciente'
    }

    try:
        # 1. Tabla de Posiciones
        st_params = {'league': league_id, 'season': season}
        res_st = requests.get(f"{base_url}/standings", headers=headers, params=st_params, timeout=5)
        
        try:
            st_res = res_st.json()
        except Exception as e:
            st_res = {}

        for team in st_res.get('response', [{}])[0].get('league', {}).get('standings', [[]])[0]:
            if team['team']['id'] == local_id:
                dossier['posicion_local'] = f"{team['rank']}º ({team['points']} pts, GD: {team['goalsDiff']})"
            if team['team']['id'] == visita_id:
                dossier['posicion_visita'] = f"{team['rank']}º ({team['points']} pts, GD: {team['goalsDiff']})"

        # 2. Estadísticas y Forma
        for tid, key_prefix in [(local_id, 'local'), (visita_id, 'visita')]:
            stat_params = {'league': league_id, 'season': season, 'team': tid}
            res_stat = requests.get(f"{base_url}/teams/statistics", headers=headers, params=stat_params, timeout=5)
            
            try:
                stat_res = res_stat.json()
            except Exception as e:
                stat_res = {}

            if stat_res.get('response'):
                r = stat_res['response']
                dossier[f'forma_{key_prefix}'] = r.get('form', 'N/A')
                dossier[f'posesion_{key_prefix}'] = r.get('lineups', [{}])[0].get('formation', '4-4-2')

        # 3. Alineaciones y Bajas
        lineup_res = requests.get(f"{base_url}/fixtures/lineups", headers=headers, params={'fixture': fixture_id}, timeout=5).json()
        if lineup_res.get('response') and len(lineup_res['response']) >= 2:
            dossier['formacion_local'] = lineup_res['response'][0].get('formation', 'TBD')
            dossier['formacion_visita'] = lineup_res['response'][1].get('formation', 'TBD')

        # 4. H2H
        h2h_res = requests.get(f"{base_url}/fixtures/headtohead", headers=headers, params={'h2h': f"{local_id}-{visita_id}", 'last': 3}, timeout=5).json()
        if h2h_res.get('response'):
            resumen = []
            for h in h2h_res['response']:
                winner = h['teams']['home']['name'] if h['teams']['home']['winner'] else h['teams']['away']['name'] if h['teams']['away']['winner'] else "Empate"
                resumen.append(f"{h['teams']['home']['name']} {h['goals']['home']}-{h['goals']['away']} {h['teams']['away']['name']} ({winner})")
            dossier['resumen_h2h'] = " | ".join(resumen)

    except Exception as e:
        print(f"Error Dossier 360: {e}")
        
    return dossier

def obtener_noticias_tacticas(liga_activa):
    """
    Interceptor RSS Táctico: Extrae noticias de lesiones, suspensiones y táctica.
    Usa Google News RSS con operadores booleanos.
    """
    # 1. Configuración de Queries
    queries = {
        "🇨🇱 Campeonato Nacional": '"Campeonato Nacional" AND ("lesión" OR "castigo" OR "táctica" OR "árbitro" OR "baja")',
        "🇬🇧 Premier League": '"Premier League" AND ("injury" OR "suspension" OR "tactics" OR "referee" OR "ruled out")'
    }
    
    # 2. Ventana de Tiempo (Filtro desde el lunes de la semana actual)
    from datetime import date, timedelta
    hoy = date.today()
    lunes_semana_actual = hoy - timedelta(days=hoy.weekday())
    fecha_filtro = lunes_semana_actual.strftime("%Y-%m-%d")
    
    query = queries.get(liga_activa, "futbol")
    query += f" after:{fecha_filtro}"
    
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    
    # URL de Google News RSS (Ajustada a la región para Chile/Global)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=es-419&gl=CL&ceid=CL:es-419"
    
    try:
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        items = root.findall('.//item')
        
        noticias = []
        for item in items[:15]: # Últimos 15 artículos
            titulo = item.find('title').text
            link = item.find('link').text
            fecha = item.find('pubDate').text
            
            # Limpieza básica de título (Google News suele añadir el medio al final)
            if " - " in titulo:
                titulo = " - ".join(titulo.split(" - ")[:-1])
                
            noticias.append({
                "titulo": titulo,
                "link": link,
                "fecha_publicacion": fecha
            })
        return noticias
    except Exception as e:
        print(f"Error RSS News: {e}")
        return []

def get_live_news(league_id="265"):
    """Consumo de Noticias vía WP-REST (Solo Chile por ahora)."""
    if league_id != "265":
        return [{"text": "Feed de noticias internacionales próximamente...", "status": "INFO"}]
        
    url = "https://www.campeonatochileno.cl/wp-json/wp/v2/posts?per_page=3"
    try:
        r = requests.get(url, timeout=5)
        posts = r.json()
        news = []
        for p in posts:
            title = p['title']['rendered']
            status = "ALERT" if any(w in title.upper() for w in ["BAJA", "LESIÓN", "SUSPENDIDO"]) else "INFO"
            news.append({"text": title, "status": status})
        return news
    except:
        return [{"text": "Sincronizando feed oficial de noticias...", "status": "INFO"}]

if __name__ == "__main__":
    print(f"Iniciando Ingesta Satelital V30.0 (Multi-League Mode)...")
    df = get_live_fixtures(league_id="265", season="2026")
    if not df.empty:
        print(df[['Local', 'Visita', 'Status', 'Marcador']])
    else:
        print("ALERTA: No se detectaron partidos.")
    
    print("\n--- News Feed API ---")
    for n in get_live_news():
        print(f"[{n['status']}] {n['text']}")
