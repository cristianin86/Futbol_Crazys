import pandas as pd
import requests
from datetime import datetime
import time
import glob
import os

# --- MAPA GEOLOCALIZADO DE CHILE PRIMERA A (Definitivo V13) ---
ESTADIOS_CHILE = {
    "Colo-Colo": {"estadio": "Monumental (Santiago)", "lat": -33.5065, "lon": -70.6059},
    "Universidad de Chile": {"estadio": "Nacional (Santiago)", "lat": -33.4600, "lon": -70.6105},
    "Universidad Chile": {"estadio": "Nacional (Santiago)", "lat": -33.4600, "lon": -70.6105},
    "Universidad Catolica": {"estadio": "San Carlos (Santiago)", "lat": -33.3950, "lon": -70.5005},
    "Everton": {"estadio": "Sausalito (Viña del Mar)", "lat": -33.0138, "lon": -71.5367},
    "Cobreloa": {"estadio": "Zorros del Desierto (Calama)", "lat": -22.4646, "lon": -68.9248},
    "Palestino": {"estadio": "La Cisterna (Santiago)", "lat": -33.5350, "lon": -70.6653},
    "Iquique": {"estadio": "Tierra de Campeones (Iquique)", "lat": -20.2446, "lon": -70.1340},
    "Deportes Iquique": {"estadio": "Tierra de Campeones (Iquique)", "lat": -20.2446, "lon": -70.1340},
    "O'Higgins": {"estadio": "El Teniente (Rancagua)", "lat": -34.1755, "lon": -70.7397},
    "Cobresal": {"estadio": "El Cobre (El Salvador)", "lat": -26.2443, "lon": -69.6277},
    "Coquimbo Unido": {"estadio": "Sánchez Rumoroso (Coquimbo)", "lat": -29.9672, "lon": -71.3414},
    "Huachipato": {"estadio": "Estadio CAP (Talcahuano)", "lat": -36.7454, "lon": -73.1097},
    "Nublense": {"estadio": "Nelson Oyarzún (Chillán)", "lat": -36.6111, "lon": -72.1027},
    "Union La Calera": {"estadio": "Nicolás Chahuán (La Calera)", "lat": -32.7844, "lon": -71.2158},
    "Union Espanola": {"estadio": "Santa Laura (Santiago)", "lat": -33.4072, "lon": -70.6558},
    "Audax Italiano": {"estadio": "La Florida (Santiago)", "lat": -33.5222, "lon": -70.5964},
    "Copiapo": {"estadio": "Luis Valenzuela (Copiapó)", "lat": -27.3758, "lon": -70.3292}
}

def obtener_coordenadas_estadio(equipo_local):
    """Busca el estadio y devuelve coordenadas. Fallback: Santiago."""
    for key, data in ESTADIOS_CHILE.items():
        if key.lower() in equipo_local.lower() or equipo_local.lower() in key.lower():
            return data
    return {"estadio": "Estadio Local (Default)", "lat": -33.4489, "lon": -70.6693}

def calcular_fatiga_y_clima(equipo_local, equipo_visita):
    archivos = glob.glob('*matches*.csv') + glob.glob('*chile*.csv')
    if not archivos: return None
    
    # Filtrar archivos procesados por el pipeline
    archivos = [f for f in archivos if 'ml_ready' not in f]
        
    ruta_csv = max(archivos, key=os.path.getctime)
    df = pd.read_csv(ruta_csv)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    hoy = datetime.now()
    now_unix = time.time() # Tiempo real de hoy
    
    # 1. CÁLCULO DE FATIGA (CORREGIDO: Solo partidos pasados)
    def obtener_dias_descanso(equipo):
        if 'timestamp' in df.columns:
            partidos_equipo = df[
                (df['home_team_name'].str.contains(equipo, case=False, na=False) | 
                 df['away_team_name'].str.contains(equipo, case=False, na=False))
            ]
            
            # FILTRO CLAVE: Solo partidos que ya se jugaron (timestamp menor a hoy)
            partidos_jugados = partidos_equipo[partidos_equipo['timestamp'] < now_unix]
            
            if not partidos_jugados.empty:
                ultimo_timestamp = partidos_jugados['timestamp'].max()
                dias = (hoy - datetime.fromtimestamp(ultimo_timestamp)).days
                return max(0, dias)
        
        return 7 # Fallback si no hay historial
        
    descanso_local = obtener_dias_descanso(equipo_local)
    descanso_visita = obtener_dias_descanso(equipo_visita)

    # 2. CLIMA GEOLOCALIZADO
    datos_estadio = obtener_coordenadas_estadio(equipo_local)
    url_clima = f"https://api.open-meteo.com/v1/forecast?latitude={datos_estadio['lat']}&longitude={datos_estadio['lon']}&current_weather=true"
    
    clima_str = "Normal"
    es_lluvia = False
    
    try:
        res = requests.get(url_clima, timeout=3)
        if res.status_code == 200:
            datos = res.json()['current_weather']
            codigo, temp = datos['weathercode'], datos['temperature']
            if codigo >= 51:
                clima_str = f"Lluvia detectada ({temp}°C)"
                es_lluvia = True
            elif temp > 30:
                clima_str = f"Calor Extremo ({temp}°C)"
            else:
                clima_str = f"Despejado/Nublado ({temp}°C)"
    except:
        clima_str = "Datos meteorológicos no disponibles"

    return {
        "estadio": datos_estadio['estadio'],
        "descanso_local": descanso_local,
        "descanso_visita": descanso_visita,
        "clima_str": clima_str,
        "es_lluvia": es_lluvia,
        "arbitro": "Cristian Garay",
        "arbitro_tarjetas": 6.2
    }
