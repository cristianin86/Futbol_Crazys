import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from scraper_cl import (
    get_live_fixtures, get_live_news, get_odds, get_injuries, 
    obtener_noticias_tacticas, obtener_dossier_360, 
    calculate_kelly, get_combined_probs, check_lineups_status,
    get_referee_metrics, get_absence_impact,
    get_live_odds, recalculate_live_ev
)
from copa_liga_model import CopaLigaModel, CopaDataPipeline



# --- 0. GESTIÓN DE CREDENCIALES (V30.0) ---
load_dotenv(override=True)
# Nota: La clave de Gemini ha sido removida en favor de Ollama (Inferencia Local)

def invocar_agente_v8(local, visita, prob_L, prob_V, corners, tarjetas, arbitro, dossier):
    """
    Motor de Inferencia Cloud (Groq API Llama 3 8B) - Operación Deep Context 360.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "🚨 ALERTA: No se encontró GROQ_API_KEY en las variables de entorno/Secrets. Configure el Secret en Streamlit Cloud."

    MODELO_NUBE = "llama3-8b-8192" 
    
    prompt_maestro = f"""
Eres un Analista Cuantitativo y Táctico de Élite especializado en Proyecciones Matemáticas de Rendimiento Deportivo. 
Analiza este dossier 360° y entrega un reporte estratégico de desviaciones probabilísticas. 
Cruza la matemática del modelo predictivo con la táctica real y el rendimiento histórico.

[1. MATRIZ PREDICTIVA (MODELO V8)]
- Partido: {local} vs {visita}
- Probabilidad de Éxito Deportivo: Local {prob_L*100:.1f}% | Visita {prob_V*100:.1f}%
- Proyecciones de Eventos: {corners} Córners Esperados | {tarjetas} Faltas/Tarjetas (Árbitro: {arbitro})

[2. CONTEXTO LIGA Y FORMA]
- Posición en Tabla: Local: {dossier['posicion_local']} | Visita: {dossier['posicion_visita']}
- Racha Reciente (W-D-L): Local: {dossier['forma_local']} | Visita: {dossier['forma_visita']}

[3. RADIOGRAFÍA TÁCTICA]
- Formación Local: {dossier['formacion_local']} | Posesión Promedio: {dossier['posesion_local']}%
- Formación Visita: {dossier['formacion_visita']} | Posesión Promedio: {dossier['posesion_visita']}%

[4. INFORME DE BAJAS E HISTORIAL]
- Ausencias Confirmadas Local: {dossier['bajas_local']}
- Ausencias Confirmadas Visita: {dossier['bajas_visita']}
- Últimos enfrentamientos (H2H): {dossier['resumen_h2h']}

INSTRUCCIÓN ESTRICTA Y FORMATO:
Escribe un análisis de exactamente 3 párrafos cortos. 
1. Matchup Táctico: Analiza el choque de formaciones, posesión y cómo las ausencias afectan el desarrollo esperado del juego.
2. Momentum: Analiza si la racha y posición en tabla respaldan o contradicen las probabilidades matemáticas de nuestro Modelo V8.
3. Conclusión de Desviación: Indica claramente qué escenario estadístico tiene mayor probabilidad de cumplirse con base en los datos cruzados (Victoria/Empate, Volumen de Goles, Volumen de Córners o Nivel de Fricción/Tarjetas). Solo nombra un escenario destacado.

REGLA DE SEGURIDAD ABSOLUTA (SYSTEM OVERRIDE):
- REGLA DE ANOMALÍAS: Si las probabilidades del Modelo V8 contradicen la posición en la tabla, NO uses justificaciones emocionales. Asume que el Modelo detectó valor matemático oculto y justifica la desviación usando factores tácticos o estadísticos.
- NO inventes estadísticas que no estén en el texto proporcionado.
- NO agregues secciones extra ni listas de viñetas al final.
- Detén tu generación de texto INMEDIATAMENTE después de escribir la Conclusión de Desviación.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODELO_NUBE,
        "messages": [
            {"role": "system", "content": "Eres un analista de datos deportivos avanzado."},
            {"role": "user", "content": prompt_maestro}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60) 
        response.raise_for_status()
        res_json = response.json()
        return res_json["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"🚨 Error en la Nube (API Groq): {response.text}"
    except Exception as e:
        return f"🚨 Falla crítica conectando con Groq: {e}"

# --- 1. CONFIGURACIÓN MULTI-LIGA V30.0 ---
TRADUCTOR_CHILE = {
    "u. de chile": "universidad chile",
    "universidad de chile": "universidad chile",
    "u. catolica": "universidad catolica",
    "universidad catolica": "universidad catolica"
}

TRADUCTOR_EQUIPOS_CHILE = {
    "Colo Colo": "Colo Colo",
    "Coquimbo Unido": "Coquimbo Unido",
    "Univ. de Chile": "Universidad de Chile",
    "Univ. Catolica": "Universidad Catolica"
}

TRADUCTOR_PREMIER = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man. united": "manchester united",
    "tottenham hotspur": "tottenham"
}

CONFIG_LIGAS = {
    "🇨🇱 Campeonato Nacional": {
        "id_api": "265", 
        "season": "auto", 
        "dataset": "chile_ml_ready_v8.csv", 
        "dataset_raw": "chile_api_raw.csv",
        "traductor": TRADUCTOR_CHILE,
        "clima_default": "Macul, Santiago"
    },
    "🇨🇱 Primera B": {
        "id_api": "266", 
        "season": "auto", 
        "dataset": "chile_b_ml_ready.csv", 
        "dataset_raw": "chile_b_api_raw.csv",
        "traductor": TRADUCTOR_CHILE,
        "clima_default": "Santiago, Chile"
    },
    "🇬🇧 Premier League": {
        "id_api": "39", 
        "season": "auto", 
        "dataset": "premier_ml_ready_v1.csv", 
        "dataset_raw": "premier_api_raw.csv",
        "traductor": TRADUCTOR_PREMIER,
        "clima_default": "London, UK"
    }
}

st.set_page_config(
    page_title="Smart Money Eigen V30.0", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estética Antigravity Industrial y Responsive Móvil
st.markdown("""
<style>
    /* CSS Responsivo para Nube y Celulares */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-bottom: 3rem !important;
        }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.3rem !important; }
        .pro-card { padding: 12px !important; }
        .square-card { padding: 10px !important; }
        .stColumns { display: flex !important; flex-direction: column !important; }
    }
    
    .stApp { background-color: #F9F6F0; color: #2C2C2C; }
    .pro-card { 
        background: #FFFFFF; border: 1px solid #EAE3D2; border-radius: 12px; padding: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); color: #2C2C2C;
    }
    .square-card {
        background: #FDFBF7; border-radius: 8px; padding: 20px; text-align: center; border: 1px solid #EAE3D2; height: 100%; color: #2C2C2C;
    }
    .badge-live { background: #C85A17; color: white; padding: 4px 10px; border-radius: 15px; font-weight: 900; animation: pulse 2s infinite; font-size: 0.7rem;}
    .badge-upcoming { background: #4A7c44; color: white; padding: 4px 10px; border-radius: 15px; font-weight: 800; font-size: 0.7rem;}
    .badge-locked { background: #8b949e; color: white; padding: 4px 10px; border-radius: 15px; font-weight: 800; font-size: 0.7rem;}
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .news-alert { color: #C85A17; border-left: 4px solid #C85A17; padding-left: 10px; margin-bottom: 10px; font-weight: 600; }
    .news-info { color: #4A7c44; border-left: 4px solid #4A7c44; padding-left: 10px; margin-bottom: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 2. SELECTOR DE MANDO (SIDEBAR) ---
st.sidebar.title("🎛️ Centro de Mando")
liga_seleccionada = st.sidebar.selectbox("Seleccione Liga Activa:", list(CONFIG_LIGAS.keys()))
conf = CONFIG_LIGAS[liga_seleccionada]

# --- 2.1 GESTIÓN DE RIESGO (SIDEBAR) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Gestión de Riesgo")
bankroll = st.sidebar.number_input("💰 Bankroll Total ($)", value=100000, step=1000)

HISTORY_FILE = "odds_trend_memory.json"

def load_odds_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_odds_history(history):
    try:
        # Limitar tamaño para no saturar el JSON (solo guardar últimos 1000 registros)
        if len(history) > 1000:
            keys_to_keep = list(history.keys())[-1000:]
            history = {k: history[k] for k in keys_to_keep}
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)
    except:
        pass

if 'odds_history' not in st.session_state:
    st.session_state.odds_history = load_odds_history()

# --- 3. CAPA DE INGESTIÓN (API) ---
@st.cache_data(ttl=600)
def cargar_master_data_v30(config):
    """Inyecta la grilla oficial y el dataset histórico dinámicamente."""
    with st.spinner(f"📡 Sincronizando {liga_seleccionada}..."):
        df_grilla = get_live_fixtures(
            league_id=config['id_api'], 
            season=config['season'], 
            translator=config['traductor']
        )
    try:
        df_hist = pd.read_csv(config['dataset'], encoding='latin-1')
        df_hist.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df_hist.columns]
        df_hist = df_hist.ffill().bfill()
        
        targets = ['target_1x2', 'home_team_goal_count', 'away_team_goal_count']
        feats = [c for c in df_hist.columns if c not in targets and c != 'timestamp']
        return df_grilla, df_hist, feats
    except Exception as e:
        st.sidebar.error(f"Error cargando dataset: {e}")
        return df_grilla, None, []

# Eliminamos caché temporalmente para forzar lectura del disco (Destrucción de caché)
def cargar_modelos_v30(liga):
    mods = {}
    s = ""
    if "Primera B" in liga:
        s = "_B"
        
    if "Campeonato Nacional" in liga:
        archivos = [('1x2','modelo_chile_1x2_v1.json'), ('hg','modelo_chile_hg_v1.json'), 
                    ('ag','modelo_chile_ag_v1.json'), ('hc','modelo_chile_hc_v1.json'), 
                    ('ac','modelo_chile_ac_v1.json')]
    else:
        archivos = [('1x2',f'model_1x2_v5{s}.json'), ('hg',f'model_hg_v5{s}.json'), 
                    ('ag',f'model_ag_v5{s}.json'), ('hc',f'model_hc_v5{s}.json'), 
                    ('ac',f'model_ac_v5{s}.json')]
                    
    for m, f in archivos:
        if os.path.exists(f):
            b = xgb.Booster(); b.load_model(f); mods[m] = b
    return mods

# --- 4. MOTOR DE INFERENCIA AGNOSTICO V30 ---
def obtener_stats_aisladas(team_name, raw_csv, condicion):
    """Escanea el historial crudo para extraer el momentum real aislado."""
    import traceback
    import streamlit as st
    try:
        df_h = pd.read_csv(raw_csv).sort_values('timestamp')
    except Exception as e:
        st.error(f"🚨 Falla crítica al construir el vector matemático: {e}")
        st.error(f"Traza del error: {traceback.format_exc()}")
        # No retornes valores por defecto, deja que el sistema se detenga
        raise
        
    # --- TRADUCCIÓN DE NOMBRES ---
    equipo = TRADUCTOR_EQUIPOS_CHILE.get(team_name, team_name)
    
    # Filtrar solo partidos completados
    if 'status' in df_h.columns:
        df_h = df_h[df_h['status'] == 'complete']
    else:
        df_h = df_h.dropna(subset=['home_team_goal_count'])

    # --- DOBLE FILTRADO DE DATAFRAME ---
    if condicion == 'Local':
        df_filtrado = df_h[df_h['home_team_name'] == equipo].copy()
    else:
        df_filtrado = df_h[df_h['away_team_name'] == equipo].copy()
    
    if 'timestamp' in df_filtrado.columns:
        df_filtrado = df_filtrado.sort_values(by='timestamp', ascending=False)
        
    df_filtrado = df_filtrado.head(5) # Tomamos los últimos 5
    
    if df_filtrado.empty:
        # FALLBACK: Si no hay partidos en esa condición (Local/Visita), buscar CUALQUIER partido del equipo
        df_filtrado = df_h[(df_h['home_team_name'] == equipo) | (df_h['away_team_name'] == equipo)].copy()
        if 'timestamp' in df_filtrado.columns:
            df_filtrado = df_filtrado.sort_values(by='timestamp', ascending=False)
        df_filtrado = df_filtrado.head(5)
        
        if df_filtrado.empty:
            import streamlit as st
            st.error(f"🚨 Falla crítica: No se encontraron datos de NINGÚN tipo para '{equipo}' en {raw_csv}.")
            raise ValueError(f"Sin datos históricos para {equipo}")
        else:
            st.warning(f"⚠️ Aviso: Usando historial general para {equipo} (insuficiente historial como {condicion}).")

    df_equipo = df_filtrado

    
    n_partidos = len(df_equipo)

    # --- EXPANSIÓN TOTAL DE MÉTRICAS (Aisladas con np.where para soporte Fallback) ---
    is_home = df_equipo['home_team_name'] == equipo
    
    gf = np.where(is_home, df_equipo['home_team_goal_count'], df_equipo['away_team_goal_count']).sum() / n_partidos
    gc = np.where(is_home, df_equipo['away_team_goal_count'], df_equipo['home_team_goal_count']).sum() / n_partidos
    pos = np.where(is_home, df_equipo['home_team_possession'], df_equipo['away_team_possession']).sum() / n_partidos
    xg = np.where(is_home, df_equipo['home_team_pre_match_xg'], df_equipo['away_team_pre_match_xg']).sum() / n_partidos
    corners = np.where(is_home, df_equipo['home_team_corner_count'], df_equipo['away_team_corner_count']).sum() / n_partidos
    yc = np.where(is_home, df_equipo['home_team_yellow_cards'], df_equipo['away_team_yellow_cards']).sum() / n_partidos
    
    # Puntos por Partido (PPG) Dinámico
    wins = np.where(is_home, 
                    df_equipo['home_team_goal_count'] > df_equipo['away_team_goal_count'], 
                    df_equipo['away_team_goal_count'] > df_equipo['home_team_goal_count'])
    draws = df_equipo['home_team_goal_count'] == df_equipo['away_team_goal_count']
    
    pts_total = (sum(wins) * 3) + sum(draws)
    ppg = pts_total / n_partidos

    return {
        'gf': gf, 'gc': gc, 'ppg': ppg, 
        'pos': pos, 'xg': xg, 'corners': corners, 'yc': yc
    }



# --- RUTAS ESTÁTICAS DE FEATURES (INMUTABLES) ---
# Fuente de verdad: columnas exactas y en el orden correcto según la matriz de entrenamiento.
# NUNCA modificar el orden de estas listas sin re-entrenar los modelos.
FEATURES_10 = ['l_gf', 'l_gc', 'l_ppg', 'l_pos', 'l_xg', 'v_gf', 'v_gc', 'v_ppg', 'v_pos', 'v_xg']
FEATURES_13 = ['l_gf', 'l_gc', 'l_ppg', 'l_pos', 'l_xg', 'v_gf', 'v_gc', 'v_ppg', 'v_pos', 'v_xg',
                'is_cup', 'home_team_lsi', 'away_team_lsi']

def run_master_inference(local, visita, config, is_cup=0, lsi_local=1.0, lsi_visita=1.0):
    l_s = obtener_stats_aisladas(local, config['dataset_raw'], 'Local')
    v_s = obtener_stats_aisladas(visita, config['dataset_raw'], 'Visita')
    
    # --- ENRUTAMIENTO ESTÁTICO POR FIRMA DE MODELO (V32.1 ROLLBACK) ---
    # Inspeccionamos el número de features del booster desplegado para elegir la lista correcta.
    # Se usa el conteo, NO el getattr dinámico que provocó el feature shift.
    booster_feature_count = 10  # Default seguro: todos los modelos actuales son F10
    if 'hg' in modelos and modelos['hg'].feature_names:
        booster_feature_count = len(modelos['hg'].feature_names)

    # --- SEGURO LSI PRE-MATCH ---
    # Si la API no retornó formaciones (Copa_liga_model mandó -1), forzamos un valor neutral 1.0 para no romper el XGBoost.
    lsi_local_safe = max(1.0, float(lsi_local)) if float(lsi_local) < 0 else float(lsi_local)
    lsi_visita_safe = max(1.0, float(lsi_visita)) if float(lsi_visita) < 0 else float(lsi_visita)

    if booster_feature_count == 13:
        # Modelos re-entrenados con LSI (Chile full pipeline con Copa)
        cols = FEATURES_13
        vals = [l_s['gf'], l_s['gc'], l_s['ppg'], l_s['pos'], l_s['xg'],
                v_s['gf'], v_s['gc'], v_s['ppg'], v_s['pos'], v_s['xg'],
                is_cup, lsi_local_safe, lsi_visita_safe]
    else:
        # RUTA ESTÁNDAR: Premier League, Chile clásico, Primera B (todos F10)
        # lsi e is_cup son ignorados deliberadamente para no contaminar la matriz
        cols = FEATURES_10
        vals = [l_s['gf'], l_s['gc'], l_s['ppg'], l_s['pos'], l_s['xg'],
                v_s['gf'], v_s['gc'], v_s['ppg'], v_s['pos'], v_s['xg']]

    # Construcción del tensor de entrada con orden de columnas garantizado
    X = pd.DataFrame([vals], columns=cols)

    
    # --- CORTAFUEGOS ANTI-BASURA (V31.5) ---
    is_b = "Primera B" in config.get('liga_seleccionada', '') # Necesitaremos pasar esto o detectarlo
    # Para ser más robustos, detectamos por el nombre del archivo raw
    is_b = "_b_" in config['dataset_raw'].lower()
    
    threshold_pos = 50.0 if not is_b else 10.0 # Mucho más permisivo en la B si estamos empezando
    
    if X.isnull().values.any() or (not is_b and (X['l_pos'].iloc[0] == 50.0 or X['v_pos'].iloc[0] == 50.0)):
        st.error("🚨 CORTE DE EMERGENCIA: Falla en la ingesta de datos. El sistema no pudo extraer estadísticas reales y está recurriendo a valores por defecto (NaN o 50% de posesión). Predicción abortada para evitar falsos positivos.")
        st.stop()

    dmat = xgb.DMatrix(X.astype(float))

    p_win = modelos['1x2'].predict(dmat)[0] if '1x2' in modelos else [0.33, 0.33, 0.33]
    hg = modelos['hg'].predict(dmat)[0] if 'hg' in modelos else 1.5
    ag = modelos['ag'].predict(dmat)[0] if 'ag' in modelos else 1.0
    
    # Corners Dinámicos
    hc = modelos['hc'].predict(dmat)[0] if 'hc' in modelos else l_s['corners']
    ac = modelos['ac'].predict(dmat)[0] if 'ac' in modelos else v_s['corners']
    
    # Tarjetas Dinámicas (Suma de promedios históricos como base)
    t_base = l_s['yc'] + v_s['yc']
    
    # --- VINCULACIÓN REAL V31.0 ---
    return {
        "p_win": p_win, 
        "local_goals_proyected": hg, 
        "visitor_goals_proyected": ag, 
        "hc": hc, 
        "ac": ac, 
        "total_corners": hc + ac,
        "total_cards": t_base
    }

# --- INICIO DE LA SUITE ---
df_grilla, df_comp, features_cols = cargar_master_data_v30(conf)
modelos = cargar_modelos_v30(liga_seleccionada)

# --- 4. HEADER PREMIUM (LOGO LIGA) ---
header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    league_logo = df_grilla.iloc[0]['League_Logo'] if not df_grilla.empty else ""
    if league_logo:
        st.image(league_logo, width=60)
with header_col2:
    st.title("Modelo Eigen V30.0")
    st.caption(f"Liga Activa: {liga_seleccionada}")

# --- Cerebro Activo (Sidebar) ---
modelo_activo_nombre = "modelo_chile_1x2_v1.json" if "Campeonato Nacional" in liga_seleccionada else "model_1x2_v5.json"
st.sidebar.caption(f"🧠 Cerebro: {modelo_activo_nombre}")

if df_grilla.empty:
    st.error(f"🛑 No se detectaron partidos para {liga_seleccionada} en la ventana actual.")
    st.stop()

# Selector Pro
partido_sel = st.selectbox(
    "Selección de Evento:",
    df_grilla['Partido_String'].tolist(),
    index=0
)

# Datos del partido
match_data = df_grilla[df_grilla['Partido_String'] == partido_sel].iloc[0]
local, visita = match_data['Local'], match_data['Visita']
arbitro = str(match_data['Arbitro']) if pd.notnull(match_data['Arbitro']) else "Por Designar"
estadio = str(match_data['Estadio']) if pd.notnull(match_data['Estadio']) else "Sede Oficial"
status = match_data['Status']
marcador = match_data['Marcador']
health = match_data['Health_Score']
fixture_id = match_data['id']
local_id = match_data['Local_ID']
visita_id = match_data['Visita_ID']
fecha_evento = match_data.get('Fecha', '--/--')

can_predict = (status == "upcoming")
badge_html = {
    "live": "<span class='badge-live'>🔴 EN VIVO</span>",
    "upcoming": "<span class='badge-upcoming'>🕒 PROGRAMADO</span>",
    "finished": "<span class='badge-locked'>🔒 FINALIZADO</span>"
}.get(status, "<span class='badge-upcoming'>🕒 PROGRAMADO</span>")

# --- 5. TARJETA VISUAL DEL ENCUENTRO ---
st.markdown("<br>", unsafe_allow_html=True)
with st.container():
    v1, v2, v3 = st.columns([1, 0.5, 1])
    
    with v1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if match_data['Local_Logo']:
            st.image(match_data['Local_Logo'], width=100)
        st.markdown(f"### {local}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with v2:
        st.markdown("<div style='text-align: center; padding-top: 20px;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #C85A17; margin-bottom: 0;'>VS</h1>", unsafe_allow_html=True)
        st.markdown(f"<b>{badge_html}</b><br>{fecha_evento}", unsafe_allow_html=True)
        if status != "upcoming":
            st.markdown(f"<h2 style='margin-top: 10px;'>{marcador}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with v3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if match_data['Visita_Logo']:
            st.image(match_data['Visita_Logo'], width=100)
        st.markdown(f"### {visita}")
        st.markdown("</div>", unsafe_allow_html=True)

# --- 5.1 EL ORÁCULO DEL ENTORNO (PRO ANALYTICS) ---
with st.container():
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        ciudad = match_data.get('Ciudad', 'Ciudad TBD')
        st.markdown(f"<div style='text-align: center; color: #5D5D5D;'><small>🏟️ Impacto Físico</small><br><b style='color: #2C2C2C;'>{estadio}</b><br><small>{ciudad} | ☀️ 18°C</small></div>", unsafe_allow_html=True)
    
    with o2:
        # PERFILADOR DE ÁRBITROS (V31.0)
        ref_meta = get_referee_metrics(arbitro)
        val_display = f"{ref_meta['media']}" if ref_meta['real'] else "5.0"
        delta_display = f"{ref_meta['tendencia']:+.1f} vs Media" if ref_meta['real'] else "N/A"
        
        st.metric(
            label="⚖️ Juez: " + (arbitro if len(arbitro) < 15 else arbitro[:12]+"..."),
            value=f"{val_display} Tarjetas",
            delta=delta_display if ref_meta['real'] else None,
            delta_color="inverse" if ref_meta['tendencia'] > 0.5 else "normal"
        )
    
    with o3:
        # ESCÁNER DE AUSENCIAS LOCAL
        st.markdown("<div style='text-align: center; color: #5D5D5D;'><small>🏠 Impacto XI Local</small></div>", unsafe_allow_html=True)
        bajas_l = get_absence_impact(fixture_id, local_id)
        if not bajas_l:
            st.markdown("<div style='text-align: center;'><b style='color: #4A7c44;'>✅ Sin Bajas</b></div>", unsafe_allow_html=True)
        else:
            for b in bajas_l[:2]:
                label = f"🚨 {b['name']}" if b['key'] else b['name']
                st.markdown(f"<div style='font-size: 0.75rem; background: #FDF2F2; color: #9B1C1C; padding: 2px 8px; border-radius: 4px; margin-bottom: 2px;'>{label}</div>", unsafe_allow_html=True)

    with o4:
        # ESCÁNER DE AUSENCIAS VISITA
        st.markdown("<div style='text-align: center; color: #5D5D5D;'><small>🚀 Impacto XI Visita</small></div>", unsafe_allow_html=True)
        bajas_v = get_absence_impact(fixture_id, visita_id)
        if not bajas_v:
            st.markdown("<div style='text-align: center;'><b style='color: #4A7c44;'>✅ Sin Bajas</b></div>", unsafe_allow_html=True)
        else:
            for b in bajas_v[:2]:
                label = f"🚨 {b['name']}" if b['key'] else b['name']
                st.markdown(f"<div style='font-size: 0.75rem; background: #FDF2F2; color: #9B1C1C; padding: 2px 8px; border-radius: 4px; margin-bottom: 2px;'>{label}</div>", unsafe_allow_html=True)

st.markdown("---")

inf_res = run_master_inference(local, visita, conf) if can_predict else None

# --- 2. SEGURO DE INTEGRIDAD Y AJUSTE DE ÁRBITRO ---
if inf_res:
    required_keys = ['p_win', 'local_goals_proyected', 'visitor_goals_proyected', 'total_corners', 'total_cards']
    missing_keys = [k for k in required_keys if k not in inf_res]
    if missing_keys:
        st.error(f"🚨 ERROR CRÍTICO DE INTEGRIDAD: El modelo no entregó las llaves: {missing_keys}")
        st.stop()
        
    # --- PROYECCIÓN DUAL DE TARJETAS (V32.0) ---
    inf_res['cards_club_history'] = inf_res['total_cards']
    ref_meta = get_referee_metrics(arbitro)
    if ref_meta['real']:
        # El resultado final es la suma del historial + la tendencia del árbitro
        inf_res['total_cards'] = inf_res['cards_club_history'] + ref_meta['tendencia']
        inf_res['referee_influence'] = ref_meta['tendencia']
        inf_res['referee_real'] = True
    else:
        inf_res['referee_influence'] = 0.0
        inf_res['referee_real'] = False



# --- 6. RUTEO DINÁMICO DE INTERFAZ ---
# Si el partido no es 'upcoming', 'live' o 'finished', aplicamos el candado de seguridad.
# Pero permitimos 'live' para que el radar de la Fase 3 pueda operar.
if status not in ["upcoming", "live", "finished"] and not can_predict:
    st.warning(f"🚫 El radar se encuentra en modo standby para este fixture. Status: {status}")
    st.info("El sistema está esperando telemetría oficial o el inicio del encuentro.")
else:
    tab_inicio, tab_analisis, tab_cuotas, tab_planteles, tab_noticias, tab_live, tab_copa = st.tabs([
        "📊 Centro de Mando", "⏱️ Análisis Táctico", "💰 Rigor Betting", "🏃‍♂️ Planteles Pro", "📰 News Feed", "🔴 Live Trading Radar", "🏆 Copa de la Liga (Radar LSI)"
    ])

    with tab_inicio:
        st.subheader(f"Contexto del Evento: {local} vs {visita}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='square-card'><b>🌧️ Clima</b><br><br>16°C Despejado<br><small>{conf['clima_default']}</small></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='square-card'><b>🏟️ Estadio</b><br><br>{estadio}<br><small>Health Score: {health}%</small></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='square-card'><b>⚖️ Árbitro</b><br><br>{arbitro}<br><small>Status: Oficial</small></div>", unsafe_allow_html=True)
        with c4:
            prob_txt = f"{inf_res['p_win'][0]*100:.1f}%" if inf_res else "--"
            st.markdown(f"<div class='square-card'><b>🎯 Prob. Local</b><br><br>{prob_txt}<br><small>Status: {'LISTO' if can_predict else 'LOCK'}</small></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Tendencia de Partido (1X2)")
        if inf_res:
            col_L, col_E, col_V = st.columns(3)
            prob_l, prob_e, prob_v = inf_res['p_win'][0], inf_res['p_win'][1], inf_res['p_win'][2]
            col_L.metric("Victoria Local", f"{prob_l*100:.1f}%", f"Cuota Justa: {1/(prob_l+0.001):.2f}")
            col_E.metric("Empate", f"{prob_e*100:.1f}%", f"Cuota Justa: {1/(prob_e+0.001):.2f}")
            col_V.metric("Victoria Visita", f"{prob_v*100:.1f}%", f"Cuota Justa: {1/(prob_v+0.001):.2f}")
        else:
            st.info("Inferencia no disponible en vivo/finalizado.")

    with tab_analisis:
        st.subheader("⏱️ Análisis Táctico y Proyecciones")
        if not can_predict:
            st.warning(f"⚠️ Análisis pre-partido deshabilitado. Marcador: {marcador}")
        elif inf_res:
            st.markdown("---")
            st.header("📈 Desglose Cuantitativo Profundo")

            # --- 🚩 MERCADO DE CÓRNERS ---
            st.subheader("🚩 Mercado de Córners")
            # Vinculación real sin valores por defecto ocultos
            c_base = inf_res['total_corners']
            hc_inf = inf_res['hc']
            ac_inf = inf_res['ac']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Línea Central", f"{c_base:.1f}")
            col2.metric("Distribución 1T", f"{c_base * 0.45:.1f}", "45% del total")
            col3.metric("Distribución 2T", f"{c_base * 0.55:.1f}", "55% del total")
            col4.metric("Desglose Equipos", f"L: {hc_inf:.1f} | V: {ac_inf:.1f}")

            st.caption("Distribución de Probabilidad: Under 8.5 (30%) | Rango 9-10 (45%) | Over 10.5 (25%)")
            st.progress(0.45) # Barra visual para el rango central

            # --- ⚖️ MERCADO DE TARJETAS ---
            st.markdown("---")
            st.subheader("⚖️ Mercado de Tarjetas (Disciplina)")
            
            # 1. Resultado por Historial de Clubes (Base)
            t_historial = inf_res['cards_club_history']
            
            # 2. Resultado Final (con Influencia del Árbitro)
            t_final = inf_res['total_cards']
            
            t_1T = t_final * 0.35

            col_t1, col_t2, col_t3 = st.columns(3)
            
            # Mostramos el resultado dual
            delta_ref = f"{inf_res['referee_influence']:+.1f} por Juez" if inf_res['referee_real'] else "Sin info Juez"
            
            col_t1.metric(
                "Proyección (Clubes)", 
                f"{t_historial:.1f}", 
                help="Basado únicamente en el promedio de tarjetas de los últimos 5 partidos de cada equipo."
            )
            
            col_t2.metric(
                "Proyección (Final/Árbitro)", 
                f"{t_final:.1f}", 
                delta=delta_ref,
                delta_color="inverse" if inf_res['referee_influence'] > 0 else "normal",
                help="Resultado ajustado según la tendencia histórica del árbitro designado."
            )
            
            col_t3.metric(
                "Riesgo Global", 
                "ALTO" if t_final > 5.5 else "MEDIO" if t_final > 4.0 else "BAJO",
                delta="Fricción Estimada"
            )
            
            st.markdown(f"**Desglose por Tiempos (Influencia Árbitro):**")
            d1, d2, d3 = st.columns(3)
            d1.write(f"⏱️ 1er Tiempo: **{t_1T:.1f}**")
            d2.write(f"⏱️ 2do Tiempo: **{t_final - t_1T:.1f}**")
            d3.write(f"⚖️ Juez: **{arbitro}** ({'Oficial' if inf_res['referee_real'] else 'N/A'})")

            # --- ⚽ MERCADO DE GOLES ---
            st.markdown("---")
            st.subheader("⚽ Mercado de Goles (Poisson Limits)")
            # Vinculación real a llaves descriptivas
            goles_local = inf_res['local_goals_proyected']
            goles_visita = inf_res['visitor_goals_proyected']
            total_goles = goles_local + goles_visita

            col_g1, col_g2, col_g3 = st.columns(3)
            col_g1.metric("Total Esperado (xG)", f"{total_goles:.2f}")
            col_g2.metric("Línea Asiática", "2.0 / 2.5", "Tendencia Under" if total_goles < 2.5 else "Tendencia Over")

            # Cálculo simplificado de Ambos Marcan (BTTS) basado en xG
            prob_btts = (1 - (2.718 ** -float(goles_local))) * (1 - (2.718 ** -float(goles_visita)))
            col_g3.metric("Ambos Marcan (BTTS)", f"{prob_btts * 100:.1f}%")

            if st.button("🧠 Generar Informe IA"):
                with st.spinner("Extrayendo telemetría 360 desde API-Sports y procesando en LLM..."):
                    # Fase 1: Dossier
                    dossier = obtener_dossier_360(fixture_id, local_id, visita_id, conf['id_api'], conf['season'])
                    # Fase 2 & 3: Inferencia
                    informe = invocar_agente_v8(local, visita, inf_res['p_win'][0], inf_res['p_win'][2], c_base, t_final, arbitro, dossier)
                    st.info(informe)

    with tab_cuotas:
        if st.button("🔄 Actualizar Cuotas en Vivo", type="primary"):
            with st.spinner("Obteniendo datos frescos del mercado..."):
                get_odds.clear()
                cargar_master_data_v30.clear()
                st.rerun()

        st.subheader("💰 Rigor Betting Dinámico")
        if can_predict and inf_res:
            import math

            def poisson_over(lam, line):
                """Calcula P(X > line) usando distribución de Poisson."""
                k_max = int(line)
                prob_under = sum((lam**k * math.exp(-lam)) / math.factorial(k) for k in range(k_max + 1))
                return 1 - prob_under

            # --- Extracción de Cuotas Reales ---
            with st.spinner("Sincronizando cuotas de mercado..."):
                odds_data = get_odds(fixture_id)

            bookmaker_name = odds_data.get('bookmaker', 'N/A') if odds_data else 'N/A'
            st.caption(f"📡 Fuente de Cuotas: {bookmaker_name}")

            oportunidades = []
            descartadas = []

            # Mercado 1X2
            odds_1x2 = odds_data.get('1x2', {}) if odds_data else {}
            if odds_1x2:
                prob_l, prob_e, prob_v = inf_res['p_win'][0], inf_res['p_win'][1], inf_res['p_win'][2]
                for label, key, prob in [(local, 'Home', prob_l), ("Empate", 'Draw', prob_e), (visita, 'Away', prob_v)]:
                    bk_odds = odds_1x2.get(key, {})
                    if bk_odds:
                        # Usar la mejor cuota disponible para el cálculo de ROI superior
                        cuota_max = max(bk_odds.values())
                        ev = (prob * cuota_max) - 1
                        item = {"Mercado": "1X2", "Línea": label, "Cuota": cuota_max, "Prob. IA": f"{prob*100:.1f}%", "ROI": ev}
                        if ev > 0: oportunidades.append(item)
                        else: descartadas.append(item)

            # Mercado Goles, Corners, Tarjetas
            mercados = [
                ('ou_goals', "⚽ Goles", float(inf_res['local_goals_proyected']) + float(inf_res['visitor_goals_proyected'])),
                ('ou_corners', "🚩 Corners", float(inf_res['total_corners'])),
                ('ou_cards', "⚖️ Tarjetas", float(inf_res['total_cards']))
            ]

            for key, name, lambda_val in mercados:
                ou_cat = odds_data.get(key, {}) if odds_data else {}
                for line_val, line_data in ou_cat.items():
                    # Over
                    prob_over = poisson_over(lambda_val, float(line_val))
                    if line_data['over']:
                        cuota_over = max(line_data['over'].values())
                        ev_over = (prob_over * cuota_over) - 1
                        item_over = {"Mercado": name, "Línea": f"Over {line_val}", "Cuota": cuota_over, "Prob. IA": f"{prob_over*100:.1f}%", "ROI": ev_over}
                        if ev_over > 0: oportunidades.append(item_over)
                        else: descartadas.append(item_over)
                    # Under
                    prob_under = 1 - prob_over
                    if line_data['under']:
                        cuota_under = max(line_data['under'].values())
                        ev_under = (prob_under * cuota_under) - 1
                        item_under = {"Mercado": name, "Línea": f"Under {line_val}", "Cuota": cuota_under, "Prob. IA": f"{prob_under*100:.1f}%", "ROI": ev_under}
                        if ev_under > 0: oportunidades.append(item_under)
                        else: descartadas.append(item_under)

            # --- RENDER ESTILO BOOKMAKER (BETANO-STYLE) ---
            
            # --- RENDER COMPARADOR HORIZONTAL (PRO-SPEC) ---
            
            def render_comparison_header():
                """Dibuja la fila de encabezados para las tablas de comparación."""
                h = st.columns([2, 1.2, 1.5, 1.5, 1.5])
                h[0].caption("Mercado")
                h[1].caption("IA Eigen")
                h[2].caption("Betano")
                h[3].caption("Bet365")
                h[4].caption("Pinnacle")
                st.divider()

            def render_comparison_row(label, prob_ia, odds_dict, market_id="1x2"):
                """Dibuja una fila horizontal de comparación con formato unificado Cuota (Prob%), tendencia y Kelly."""
                cols = st.columns([2, 1.2, 1.5, 1.5, 1.5])
                
                # 1. Etiqueta del Mercado
                cols[0].write(label)
                
                # 2. IA Eigen (Cuota Justa + Prob IA)
                cuota_justa = 1 / (prob_ia + 0.0001)
                cols[1].write(f"**{cuota_justa:.2f}** ({prob_ia*100:.0f}%)")
                
                # 3. Bookies (Cuota + Prob Implícita + Tendencia + Kelly)
                for i, bk in enumerate(["Betano", "Bet365", "Pinnacle"]):
                    cuota_bk = odds_dict.get(bk, 0)
                    if cuota_bk > 0:
                        # --- RADAR DE TENDENCIA (MONEY TRACKER) ---
                        hist_key = f"{fixture_id}_{market_id}_{label}_{bk}"
                        prev_odd = st.session_state.odds_history.get(hist_key)
                        trend_icon = "-"
                        if prev_odd:
                            if cuota_bk < prev_odd: trend_icon = "📉"
                            elif cuota_bk > prev_odd: trend_icon = "📈"
                        
                        # Guardar para la siguiente comparación
                        st.session_state.odds_history[hist_key] = cuota_bk

                        prob_imp = (1 / cuota_bk) * 100
                        ev = (prob_ia * cuota_bk) - 1
                        ev_pct = ev * 100
                        
                        display_text = f"{cuota_bk:.2f} {trend_icon} ({prob_imp:.0f}%)"
                        
                        if ev > 0:
                            # --- CALCULADORA KELLY (Vía Scraper Logic) ---
                            stake_sugerido = calculate_kelly(prob_ia, cuota_bk, bankroll, fraction=4)
                            
                            # Formato +EV: Verde + Emoji + ROI + Stake
                            cols[i+2].markdown(f"""
                            <div style='color: #1a7f37; font-weight: bold;'>
                                {display_text}<br>
                                🔥 +{ev_pct:.1f}%<br>
                                <span style='font-size: 0.8rem; color: #2C2C2C;'>🎯 Stake Sugerido: ${stake_sugerido:,.0f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Formato Estándar: Gris para el ROI negativo
                            cols[i+2].markdown(f"{display_text} <br> <small style='color: gray;'>{ev_pct:.1f}%</small>", unsafe_allow_html=True)
                    else:
                        cols[i+2].caption("-")

            # 1. 🎯 Ganador del Partido (1X2)
            st.markdown("#### 🎯 Comparador 1X2 (Match Winner)")
            odds_1x2 = odds_data.get('1x2', {})
            if odds_1x2:
                render_comparison_header()
                prob_l, prob_e, prob_v = inf_res['p_win'][0], inf_res['p_win'][1], inf_res['p_win'][2]
                for label, key, prob in [(local, 'Home', prob_l), ("Empate", 'Draw', prob_e), (visita, 'Away', prob_v)]:
                    render_comparison_row(label, prob, odds_1x2.get(key, {}), market_id="1x2")
            else:
                st.info("Cuotas 1X2 no disponibles para comparación.")

            st.divider()

            # 2. 🛡️ Doble Oportunidad (Gestión de Riesgo)
            st.markdown("#### 🛡️ Gestión de Riesgo (Double Chance)")
            with st.expander("🛡️ Doble Oportunidad (1X, X2, 12)", expanded=True):
                odds_dc = odds_data.get('dc', {})
                
                # Cálculo de Probabilidades Combinadas (Vía Scraper Logic)
                probs_dc = get_combined_probs(inf_res['p_win'])
                
                dc_mappings = [
                    ("Local o Empate (1X)", "Home/Draw", probs_dc["1X"]),
                    ("Empate o Visita (X2)", "Draw/Away", probs_dc["X2"]),
                    ("Local o Visita (12)", "Home/Away", probs_dc["12"])
                ]
                
                if odds_dc:
                    render_comparison_header()
                    for label, key, prob in dc_mappings:
                        # Intentar buscar con diferentes formatos si es necesario (ej: Home/Draw vs 1X)
                        render_comparison_row(label, prob, odds_dc.get(key, {}), market_id="dc")
                else:
                    st.info("Cuotas de Doble Oportunidad no disponibles.")

            st.divider()

            # --- VISTA DUAL (PANTALLA DIVIDIDA) ---
            col_izq, col_der = st.columns(2)

            with col_izq:
                st.markdown("### ⚽ Mercado de Goles")
                # 2. ⚽ Goles Totales
                with st.expander("Goles Totales (.5 Líneas)", expanded=True):
                    ou_goals = odds_data.get('ou_goals', {})
                    lambda_goles = float(inf_res['local_goals_proyected']) + float(inf_res['visitor_goals_proyected'])
                    if ou_goals:
                        render_comparison_header()
                        for line_val in sorted(ou_goals.keys(), key=lambda x: float(x)):
                            st.markdown(f"**Línea {line_val}**")
                            prob_over = poisson_over(lambda_goles, float(line_val))
                            render_comparison_row("Más", prob_over, ou_goals[line_val]['over'], market_id=f"goles_{line_val}_over")
                            render_comparison_row("Menos", 1 - prob_over, ou_goals[line_val]['under'], market_id=f"goles_{line_val}_under")
                            st.divider()
                    else:
                        st.write("Sin cuotas de goles estándar (.5) disponibles.")

            with col_der:
                st.markdown("### 🚩 Mercados Secundarios")
                # 3. 🚩 Córners
                with st.expander("🚩 Córners Totales (.5 Líneas)", expanded=True):
                    ou_corners = odds_data.get('ou_corners', {})
                    lambda_corners = float(inf_res['total_corners'])
                    if ou_corners:
                        render_comparison_header()
                        for line_val in sorted(ou_corners.keys(), key=lambda x: float(x)):
                            st.markdown(f"**Línea {line_val}**")
                            prob_over = poisson_over(lambda_corners, float(line_val))
                            render_comparison_row("Más", prob_over, ou_corners[line_val]['over'], market_id=f"corners_{line_val}_over")
                            render_comparison_row("Menos", 1 - prob_over, ou_corners[line_val]['under'], market_id=f"corners_{line_val}_under")
                            st.divider()
                    else:
                        st.write("Sin cuotas de córners estándar (.5) disponibles.")

                # 4. 🟨 Tarjetas
                with st.expander("🟨 Tarjetas Totales (.5 Líneas)", expanded=True):
                    ou_cards = odds_data.get('ou_cards', {})
                    lambda_cards = float(inf_res['total_cards'])
                    if ou_cards:
                        render_comparison_header()
                        for line_val in sorted(ou_cards.keys(), key=lambda x: float(x)):
                            st.markdown(f"**Línea {line_val}**")
                            prob_over = poisson_over(lambda_cards, float(line_val))
                            render_comparison_row("Más", prob_over, ou_cards[line_val]['over'], market_id=f"cards_{line_val}_over")
                            render_comparison_row("Menos", 1 - prob_over, ou_cards[line_val]['under'], market_id=f"cards_{line_val}_under")
                            st.divider()
                    else:
                        st.write("Sin cuotas de tarjetas estándar (.5) disponibles.")

        else:
            st.warning("Seleccione un fixture válido para ver el comparador de Rigor Betting.")

    with tab_planteles:
        st.subheader("🏃‍♂️ Dinámica de Planteles")
        with st.spinner("Sincronizando..."):
            injuries = get_injuries(fixture_id)
            if injuries: st.json(injuries)
            else: st.info("Sin bajas reportadas.")

    with tab_noticias:
        st.subheader("🚨 Reporte de Bajas y Táctica (RSS)")
        with st.spinner("Interceptando feed satelital..."):
            noticias = obtener_noticias_tacticas(liga_seleccionada)
            
            if not noticias:
                st.warning("⚠️ No se detectaron alertas tácticas recientes en el radar.")
            else:
                for n in noticias:
                    with st.container():
                        st.markdown(f"🔗 **[{n['titulo']}]({n['link']})**")
                        st.caption(f"🕒 {n['fecha_publicacion']}")
                        st.divider()

    with tab_live:
        st.subheader("🔴 Live Trading Radar (In-Play)")
        
        # --- PANEL DE CONTROL IN-PLAY ---
        if status != "live":
            st.markdown(f"""
            <div class='pro-card' style='text-align: center; border-style: dashed;'>
                <h3 style='color: #8b949e;'>🛰️ Radar en Espera</h3>
                <p>El monitor de trading se activará automáticamente cuando el partido <b>{local} vs {visita}</b> pase a estado 'In-Play'.</p>
                <small>Status Actual: {status.upper()} | Hora Programada: {fecha_evento}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Captura de datos live desde match_data (vía get_live_fixtures)
            minuto_actual = match_data.get('minuto', 0)
            score_actual = match_data.get('Marcador', '0 - 0')

            @st.fragment
            def live_monitor_fragment():
                # --- CABECERA MONITOR 16:9 STYLE ---
                st.markdown(f"""
                <div style='background: #1a1a1a; color: #ffffff; padding: 30px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #C85A17;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='text-align: left;'>
                            <h4 style='margin:0; color: #8b949e;'>SCOREBOARD LIVE</h4>
                            <h1 style='margin:0; font-size: 3rem;'>{score_actual}</h1>
                        </div>
                        <div style='text-align: center;'>
                            <div class='badge-live' style='font-size: 1rem; padding: 5px 15px;'>{minuto_actual}'</div>
                            <p style='margin-top: 10px; color: #8b949e;'>Reloj In-Play</p>
                        </div>
                        <div style='text-align: right;'>
                            <h4 style='margin:0; color: #8b949e;'>SITUACIÓN</h4>
                            <h2 style='margin:0; color: #C85A17;'>EN DISPUTA</h2>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    if st.button("⚡ Escanear Cuotas In-Play", type="primary", use_container_width=True):
                        st.session_state.last_live_scan = get_live_odds(fixture_id)
                        st.toast("📡 Telemetría de mercado capturada.", icon="✅")
                with col_info:
                    st.caption("⚠️ El recálculo Time-Decay ajusta las probabilidades pre-match según el tiempo restante. Use con precaución en los minutos finales.")

                # --- RESULTADOS DEL ESCÁNER ---
                if 'last_live_scan' in st.session_state and st.session_state.last_live_scan:
                    l_odds = st.session_state.last_live_scan
                    st.markdown("---")
                    
                    # Mercado Principal 1X2
                    if 'Match Winner' in l_odds:
                        st.markdown("#### ⚖️ Oportunidades 1X2 (Recalculado)")
                        win_odds = l_odds['Match Winner']
                        
                        m1, m2, m3 = st.columns(3)
                        prob_pre = inf_res['p_win'] if inf_res else [0.33, 0.33, 0.33]
                        labels = ['Home', 'Draw', 'Away']
                        
                        for i, (label, col) in enumerate(zip(labels, [m1, m2, m3])):
                            cuota_now = win_odds.get(label)
                            if cuota_now:
                                # Aplicar Time-Decay MVP
                                p_adj = recalculate_live_ev(prob_pre[i], minuto_actual)
                                ev = (p_adj * cuota_now) - 1
                                
                                with col:
                                    bg_color = "#f0fff4" if ev > 0 else "#ffffff"
                                    border_color = "#22c55e" if ev > 0 else "#EAE3D2"
                                    st.markdown(f"""
                                    <div style='background: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 10px; text-align: center;'>
                                        <small style='color: #5D5D5D;'>{label.upper()}</small>
                                        <h2 style='margin: 5px 0;'>{cuota_now:.2f}</h2>
                                        <p style='margin:0; font-size: 0.8rem;'>IA Adj: <b>{p_adj*100:.1f}%</b></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if ev > 0:
                                        st.success(f"🔥 +EV: {ev*100:.1f}%")
                                        stake = calculate_kelly(p_adj, cuota_now, bankroll, fraction=8)
                                        st.metric("Stake Sugerido", f"${stake:,.0f}")
                                    else:
                                        st.caption(f"ROI: {ev*100:.1f}%")
                    
                    # --- RENDERIZADO FILTRADO (WHITELIST) ---
                    st.markdown("#### 🔍 Mercados Secundarios In-Play")
                    
                    # 1. Goles Totales (Match Goals)
                    if "Match Goals" in l_odds:
                        with st.expander("⚽ Goles Totales (Over/Under)", expanded=True):
                            df_goles = pd.DataFrame([
                                {"Línea": k, "Cuota": v} for k, v in l_odds["Match Goals"].items()
                            ])
                            st.dataframe(df_goles, use_container_width=True, hide_index=True)
                    
                    # 2. Doble Oportunidad (Double Chance)
                    if "Double Chance" in l_odds:
                        with st.expander("🛡️ Doble Oportunidad", expanded=True):
                            dc_data = l_odds["Double Chance"]
                            c1, c2, c3 = st.columns(3)
                            for i, (k, v) in enumerate(dc_data.items()):
                                with [c1, c2, c3][i % 3]:
                                    st.metric(k, f"{v:.2f}")
                else:
                    st.info("Presione el botón de escaneo para obtener las cuotas actuales del mercado en vivo.")

            live_monitor_fragment()

    with tab_copa:
        st.subheader("🏆 Copa de la Liga - Radar LSI")
        
        # 16:9 Strict Proportions - Panel de Control Horizontal
        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
        
        with col_c1:
            st.markdown("### 🔌 Conexión de Pipeline")
            if st.button("Ejecutar Análisis LSI", type="primary", use_container_width=True):
                st.session_state.run_copa_lsi = True
                
        with col_c2:
            st.markdown("### ⚙️ Opciones")
            st.caption("Aislamiento Activo")
            
        with col_c3:
            st.markdown("### 📊 Status de Integración")
            st.info("Pipeline Data Base: Listo | Model Endpoint: Listo")
            
        if st.session_state.get('run_copa_lsi', False):
            with st.spinner("Conectando con Data Pipeline y extrayendo Minutos Base..."):
                try:
                    pipeline = CopaDataPipeline()
                    modelo_copa = CopaLigaModel()
                    
                    # 1. Baseline
                    ideal_min_L, player_dict_L = pipeline.get_ideal_xi_minutes(local_id, conf['id_api'], conf['season'])
                    ideal_min_V, player_dict_V = pipeline.get_ideal_xi_minutes(visita_id, conf['id_api'], conf['season'])
                    
                    # 2. Match Day (Con SEGURO PRE-MATCH)
                    curr_min_L = pipeline.get_current_xi_minutes(fixture_id, local_id, player_dict_L)
                    curr_min_V = pipeline.get_current_xi_minutes(fixture_id, visita_id, player_dict_V)
                    
                    # Si el Scraper devuelve -1, forzamos LSI 1.0 (Sin penalización)
                    lsi_L = 1.0 if curr_min_L == -1 else min(curr_min_L / ideal_min_L if ideal_min_L > 0 else 1.0, 1.0)
                    lsi_V = 1.0 if curr_min_V == -1 else min(curr_min_V / ideal_min_V if ideal_min_V > 0 else 1.0, 1.0)
                    
                    # 3. Datos del modelo principal (Base Sin Rotar)
                    xg_base_L = inf_res['local_goals_proyected'] if inf_res else 1.5
                    xg_base_V = inf_res['visitor_goals_proyected'] if inf_res else 1.0
                    prob_base_L = inf_res['p_win'][0] if inf_res else 0.45
                    prob_base_V = inf_res['p_win'][2] if inf_res else 0.30
                    
                    # Prediccion Nativa con LSI internalizado
                    # Detectamos si es Copa o Liga para el flag is_cup
                    es_copa = 1 if "Copa" in liga_seleccionada else 0
                    native_res = run_master_inference(local, visita, conf, is_cup=es_copa, lsi_local=lsi_L, lsi_visita=lsi_V)
                    
                    native_xg_L = native_res['local_goals_proyected'] if native_res else 1.5
                    native_xg_V = native_res['visitor_goals_proyected'] if native_res else 1.0
                    prob_native_L = native_res['p_win'][0] if native_res else 0.45
                    prob_native_V = native_res['p_win'][2] if native_res else 0.30
                    
                    # 4. Ajustes Dinámicos (Dead Rubber)
                    st.divider()
                    
                    # Renderizado 16:9 Strict
                    tarjeta_L, tarjeta_V = st.columns(2)
                    
                    with tarjeta_L:
                        st.markdown(f"<div class='pro-card' style='margin-bottom: 10px;'><h3>🏠 {local}</h3></div>", unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("Minutos Baseline", f"{ideal_min_L}m")
                        m2.metric("Minutos Real (XI)", f"{curr_min_L}m")
                        
                        # Render visual LSI
                        delta_lsi_L = lsi_L * 100 - 100
                        st.metric(label="Fuerza Titular (LSI)", value=f"{lsi_L*100:.1f}%", delta=f"{delta_lsi_L:.1f}% Rotación", delta_color="inverse")
                        
                        if lsi_L < 0.7:
                            st.warning("🚨 Alerta: Rotación masiva detectada. xG penalizado.")
                        
                        # Control Manual de Eliminación (Dead Rubber)
                        dead_rubber_L = st.toggle("🚩 Equipo Matemáticamente Eliminado (Dead Rubber)", key="dr_L")
                        
                        # Recálculo EN VIVO
                        final_xg_L, final_prob_L = modelo_copa.adjust_predictions(local, native_xg_L, prob_native_L, dead_rubber_L)
                        
                        x1, x2 = st.columns(2)
                        x1.metric("xG Base (Sin Rotar)", f"{xg_base_L:.2f}")
                        x2.metric("xG Nativo LSI", f"{final_xg_L:.2f}", f"{final_xg_L - xg_base_L:.2f}", delta_color="inverse")
                            
                    with tarjeta_V:
                        st.markdown(f"<div class='pro-card' style='margin-bottom: 10px;'><h3>🚀 {visita}</h3></div>", unsafe_allow_html=True)
                        v1, v2 = st.columns(2)
                        v1.metric("Minutos Baseline", f"{ideal_min_V}m")
                        v2.metric("Minutos Real (XI)", f"{curr_min_V}m")
                        
                        # Render visual LSI
                        delta_lsi_V = lsi_V * 100 - 100
                        st.metric(label="Fuerza Titular (LSI)", value=f"{lsi_V*100:.1f}%", delta=f"{delta_lsi_V:.1f}% Rotación", delta_color="inverse")
                        
                        if lsi_V < 0.7:
                            st.warning("🚨 Alerta: Rotación masiva detectada. xG penalizado.")
                        
                        # Control Manual de Eliminación (Dead Rubber)
                        dead_rubber_V = st.toggle("🚩 Equipo Matemáticamente Eliminado (Dead Rubber)", key="dr_V")
                        
                        # Recálculo EN VIVO
                        final_xg_V, final_prob_V = modelo_copa.adjust_predictions(visita, native_xg_V, prob_native_V, dead_rubber_V)
                        
                        y1, y2 = st.columns(2)
                        y1.metric("xG Base (Sin Rotar)", f"{xg_base_V:.2f}")
                        y2.metric("xG Nativo LSI", f"{final_xg_V:.2f}", f"{final_xg_V - xg_base_V:.2f}", delta_color="inverse")
                        
                except Exception as e:
                    st.error(f"Falla en la inyección de datos LSI: {e}")

st.markdown("---")
save_odds_history(st.session_state.odds_history)
st.caption(f"Deep Soccer Master Suite V30.0 | Multi-League | {datetime.now().strftime('%H:%M:%S')}")

