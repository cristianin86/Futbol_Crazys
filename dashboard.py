import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import glob
import os
from live_context import calcular_fatiga_y_clima
import time

# --- 1. CONFIGURACIÓN PREMIUM DE LA PÁGINA ---
st.set_page_config(page_title="Deep Soccer Pro | Centro de Mando", layout="wide", page_icon="⚡")

# --- 2. INYECCIÓN DE CSS (ESTÉTICA TERMINAL FINANCIERA) ---
st.markdown("""
<style>
    /* Fondo y texto general */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Cajas de Datos (Pro-Cards) */
    .pro-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .pro-title { color: #8b949e; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 5px; }
    .pro-value { font-size: 1.6rem; font-weight: 800; color: #ffffff; }
    
    /* Semáforos de EV (Trading Style) */
    .ev-good { color: #3fb950; background: rgba(46,160,67,0.1); border: 1px solid rgba(46,160,67,0.3); padding: 5px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; margin-top: 8px; display: inline-block;}
    .ev-bad { color: #f85149; background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3); padding: 5px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; margin-top: 8px; display: inline-block;}
    
    /* Caja del Reporte IA */
    .report-box {
        background: #1f2937;
        border-left: 4px solid #58a6ff;
        padding: 20px;
        border-radius: 5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ DEEP SOCCER PRO: TERMINAL ANALÍTICA")
st.caption("Motor Quant-Táctico V16.2 | Modos: Pre-Match & Live Steam")
st.divider()

# --- 3. CARGA DE MOTORES (JSON) ---
@st.cache_data
def cargar_datos():
    archivos = glob.glob('*matches*.csv') + glob.glob('*chile*.csv')
    archivos = [f for f in archivos if 'ml_ready' not in f]
    if not archivos: return [], None, None, None
    ruta = max(archivos, key=os.path.getctime)
    try:
        df = pd.read_csv(ruta, encoding='latin-1')
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        equipos = sorted(df['home_team_name'].dropna().unique().tolist())
        
        df_mod = pd.read_csv("chile_ml_ready_v8.csv")
        targets = ['target_1x2', 'target_home_goals', 'target_away_goals', 'target_home_corners', 'target_away_corners']
        feats = [c for c in df_mod.columns if c not in targets]
        return equipos, ruta, feats, df
    except: return [], ruta, None, None

@st.cache_resource
def cargar_modelos():
    mods = {}
    for m, f in [('1x2','model_1x2_v5.json'), ('hg','model_hg_v5.json'), ('ag','model_ag_v5.json'), ('hc','model_hc_v5.json'), ('ac','model_ac_v5.json')]:
        if os.path.exists(f):
            b = xgb.Booster()
            b.load_model(f)
            mods[m] = b
    return mods

equipos, ruta_csv, features_cols, df_comp = cargar_datos()
modelos = cargar_modelos()

if not features_cols or '1x2' not in modelos:
    st.error("⚠️ Error: Modelos JSON no detectados. Ejecuta 'python advanced_model.py' primero.")
    st.stop()

# --- 4. PANEL DE CONFIGURACIÓN ---
st.markdown("<h4 style='color: #58a6ff;'>1. Definición del Escenario</h4>", unsafe_allow_html=True)

# Filtros principales
col_team1, col_team2, col_arb = st.columns([2, 2, 1])
# Ajuste de nombres flexible
nombre_local_match = "Everton" if "Everton" in equipos else equipos[0] if equipos else ""
nombre_visita_match = "Universidad Chile" if "Universidad Chile" in equipos else "Universidad de Chile" if "Universidad de Chile" in equipos else equipos[1] if len(equipos)>1 else equipos[0]
local = col_team1.selectbox("🏠 Equipo Local", equipos, index=equipos.index(nombre_local_match) if nombre_local_match in equipos else 0)
visita = col_team2.selectbox("✈️ Equipo Visitante", equipos, index=equipos.index(nombre_visita_match) if nombre_visita_match in equipos else 0)

# Árbitro Manual para Control Total sobre Tarjetas
arbitro_nombre = col_arb.text_input("Nombre Árbitro", value="Cristian Garay")
arbitro_tarjetas_avg = col_arb.number_input("Promedio Tarjetas Árbitro", min_value=1.0, value=6.2, step=0.1)

# Cuotas y Control Manual de Fatiga (Por si la DB falla)
st.markdown("##### 💹 Cuotas de Mercado y Ajuste de Fatiga")
col_q1, col_qx, col_q2, col_f1, col_f2 = st.columns([1.5, 1.5, 1.5, 1, 1])
cuota_1 = col_q1.number_input(f"Cuota Gana {local}", 1.01, 20.0, 2.80)
cuota_X = col_qx.number_input("Cuota Empate", 1.01, 20.0, 3.30)
cuota_2 = col_q2.number_input(f"Cuota Gana {visita}", 1.01, 20.0, 2.45)

# Detectar fatiga real primero para sugerir el valor manual
ctx_prev = calcular_fatiga_y_clima(local, visita)
descanso_base_l = ctx_prev['descanso_local'] if ctx_prev else 7
descanso_base_v = ctx_prev['descanso_visita'] if ctx_prev else 7

descanso_manual_local = col_f1.number_input(f"Días Descanso {local}", 0, 30, int(descanso_base_l))
descanso_manual_visita = col_f2.number_input(f"Días Descanso {visita}", 0, 30, int(descanso_base_v))

st.write("") # Espaciador

if st.button("🔥 PROCESAR INFERENCIA IA", use_container_width=True, type="primary"):
    with st.spinner("Compilando telemetría, calculando EV y generando reporte..."):
        time.sleep(0.5) 
        
        # --- EXTRACCIÓN DE DATOS Y CONTEXTO ---
        ctx = calcular_fatiga_y_clima(local, visita) # Usamos clima de la API
        
        df_local_matches = df_comp[df_comp['home_team_name'] == local]
        df_visita_matches = df_comp[df_comp['away_team_name'] == visita]
        
        if df_visita_matches.empty: df_visita_matches = df_comp[df_comp['home_team_name'] == visita]
        if df_local_matches.empty: df_local_matches = df_comp[df_comp['away_team_name'] == local]

        if not df_local_matches.empty and not df_visita_matches.empty:
            f_loc = df_local_matches.iloc[-1:].copy()
            f_vis = df_visita_matches.iloc[-1:].copy()
            df_infer = f_loc.copy()
            cols_away = [c for c in df_comp.columns if 'away' in c or 'team_b' in c]
            for col in cols_away:
                if col in f_loc.columns and col in f_vis.columns: df_infer[col] = f_vis[col].values[0]

            # Penalización por Lluvia
            if ctx['es_lluvia']:
                if 'team_a_xg_pre_match' in df_infer: df_infer['team_a_xg_pre_match'] *= 0.85
                if 'team_b_xg_pre_match' in df_infer: df_infer['team_b_xg_pre_match'] *= 0.85
                
            # Penalización por Fatiga Manual
            if descanso_manual_visita < 4 and 'pre_match_ppg_(away)' in df_infer: 
                df_infer['pre_match_ppg_(away)'] *= 0.85

            # Matchup Metrics Base
            xg_a_base = float(df_infer['team_a_xg_pre_match'].values[0]) if 'team_a_xg_pre_match' in df_infer else 1.0
            xg_b_base = float(df_infer['team_b_xg_pre_match'].values[0]) if 'team_b_xg_pre_match' in df_infer else 1.0
            pos_a_base = float(df_infer.get('home_team_possession', pd.Series([50])).values[0])
            pos_b_base = float(df_infer.get('away_team_possession', pd.Series([50])).values[0])
            
            df_infer['matchup_xg_ratio'] = xg_a_base / (xg_a_base + xg_b_base + 0.01)
            df_infer['matchup_possession_diff'] = pos_a_base - pos_b_base
            
            # --- PREDICCIONES XGBOOST ---
            X_test = pd.DataFrame(columns=features_cols)
            for col in features_cols: X_test.loc[0, col] = df_infer[col].values[0] if col in df_infer.columns else 0
            X_test = X_test.astype(float).fillna(0)
            dmat = xgb.DMatrix(X_test)
            
            p_1x2 = modelos['1x2'].predict(dmat)[0]
            hg = float(modelos['hg'].predict(dmat)[0])
            ag = float(modelos['ag'].predict(dmat)[0])
            hc = float(modelos['hc'].predict(dmat)[0]) if 'hc' in modelos else 4.5
            ac = float(modelos['ac'].predict(dmat)[0]) if 'ac' in modelos else 4.0

            # --- MATEMÁTICA DE TARJETAS (Árbitro + Equipos) ---
            avg_cards_loc = float(f_loc.get('home_team_yellow_cards', pd.Series([2.5])).values[0])
            avg_cards_vis = float(f_vis.get('away_team_yellow_cards', pd.Series([2.5])).values[0])
            base_agresividad = avg_cards_loc + avg_cards_vis
            # Si el árbitro promedia > 4.5 (promedio liga), infla la cantidad
            factor_arbitro = arbitro_tarjetas_avg / 4.5 
            tarjetas_proyectadas = (base_agresividad * 0.4) + (arbitro_tarjetas_avg * 0.6) # Mayor peso al árbitro

            st.divider()

            # --- RESULTADOS: SECCIÓN 2 (ESTADÍSTICAS PROFUNDAS) ---
            st.markdown("<h4 style='color: #58a6ff;'>2. Estadísticas Base y Contexto (Pre-Inferencia)</h4>", unsafe_allow_html=True)
            c_k1, c_k2, c_k3, c_k4 = st.columns(4)
            c_k1.markdown(f"<div class='pro-card'><div class='pro-title'>🌤️ CLIMA Y SEDE</div><div class='pro-value' style='font-size:1.2rem;'>{ctx['clima_str']}</div><div class='pro-subtext'>{ctx['estadio']}</div></div>", unsafe_allow_html=True)
            c_k2.markdown(f"<div class='pro-card'><div class='pro-title'>🎯 POSESIÓN HISTÓRICA</div><div class='pro-value' style='font-size:1.2rem;'>{local}: {pos_a_base:.1f}%</div><div class='pro-subtext'>{visita}: {pos_b_base:.1f}%</div></div>", unsafe_allow_html=True)
            c_k3.markdown(f"<div class='pro-card'><div class='pro-title'>📈 xG PURO (Ataque)</div><div class='pro-value' style='font-size:1.2rem;'>{local}: {xg_a_base:.2f}</div><div class='pro-subtext'>{visita}: {xg_b_base:.2f}</div></div>", unsafe_allow_html=True)
            c_k4.markdown(f"<div class='pro-card'><div class='pro-title'>🟨 PROMEDIO TARJETAS EQ.</div><div class='pro-value' style='font-size:1.2rem;'>{local}: {avg_cards_loc:.1f}</div><div class='pro-subtext'>{visita}: {avg_cards_vis:.1f}</div></div>", unsafe_allow_html=True)

            st.write("")

            # --- RESULTADOS: SECCIÓN 3 (MERCADOS AUXILIARES) ---
            st.markdown("<h4 style='color: #58a6ff;'>3. Proyecciones de Mercados Auxiliares (Poisson)</h4>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='pro-card'><div class='pro-title'>⚽ GOLES ESPERADOS</div><div class='pro-value'>{hg:.2f} - {ag:.2f}</div><div class='pro-subtext'>Total proyectado: {hg+ag:.2f}</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='pro-card'><div class='pro-title'>⛳ CÓRNERS</div><div class='pro-value'>{local}: {hc:.1f} | {visita}: {ac:.1f}</div><div class='pro-subtext'>Total: {hc+ac:.1f}</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='pro-card'><div class='pro-title'>🟨 TARJETAS TOTALES</div><div class='pro-value'>{tarjetas_proyectadas:.1f}</div><div class='pro-subtext'>Ajuste severidad: {arbitro_nombre} ({arbitro_tarjetas_avg})</div></div>", unsafe_allow_html=True)

            st.write("")

            # --- RESULTADOS: SECCIÓN 4 (RADAR EV 1X2) ---
            st.markdown("<h4 style='color: #58a6ff;'>4. Análisis Financiero 1X2 (Valor vs Bookmakers)</h4>", unsafe_allow_html=True)
            r1, rx, r2 = st.columns(3)
            
            def render_fin_ev(col, titulo, prob, cuota_mercado):
                ev = (prob * cuota_mercado) - 1
                cuota_justa = (1/max(0.001, prob)) if prob > 0 else 0
                
                html = f"<div class='pro-card'>"
                html += f"<div class='pro-title'>{titulo}</div>"
                html += f"<div class='pro-value'>{prob*100:.1f}%</div>"
                
                if ev >= 0.05:
                    html += f"<div class='ev-good'>🟢 VALOR (+{ev*100:.1f}% EV)</div>"
                    html += f"<div class='pro-subtext' style='color:#3fb950;'>Cuota Casa: {cuota_mercado} (Justa: {cuota_justa:.2f})</div>"
                elif 0 <= ev < 0.05:
                    html += f"<div class='ev-good' style='background:rgba(210,153,34,0.15); color:#d29922; border-color:#d29922;'>🟡 MARGEN ESTRECHO</div>"
                    html += f"<div class='pro-subtext'>Cuota Casa: {cuota_mercado}</div>"
                else:
                    html += f"<div class='ev-bad'>🔴 NO APOSTAR (EV Negativo)</div>"
                    html += f"<div class='pro-subtext' style='color:#f85149;'>Piden cuota {cuota_justa:.2f}, ofrecen {cuota_mercado}.</div>"
                
                html += "</div>"
                col.markdown(html, unsafe_allow_html=True)

            render_fin_ev(r1, f"Victoria {local}", p_1x2[0], cuota_1)
            render_fin_ev(rx, "Empate", p_1x2[1], cuota_X)
            render_fin_ev(r2, f"Victoria {visita}", p_1x2[2], cuota_2)

            st.divider()

            # --- RESULTADOS: SECCIÓN 5 (REPORTE LLM INTEGRADO) ---
            st.markdown("<h4 style='color: #58a6ff;'>5. Informe del Agente Táctico (Auto-Generado)</h4>", unsafe_allow_html=True)
            
            # Lógica narrativa
            dominio_txt = local if pos_a_base > pos_b_base else visita
            ataque_txt = local if hg > ag else visita
            
            fatiga_txt = "Ambos equipos llegan con tiempo estándar de preparación."
            if descanso_manual_local > descanso_manual_visita + 3: 
                fatiga_txt = f"**{local}** tiene una clara ventaja física tras descansar {descanso_manual_local} días frente a los cortos {descanso_manual_visita} días de {visita}."
            elif descanso_manual_visita > descanso_manual_local + 3: 
                fatiga_txt = f"**{visita}** llega más fresco ({descanso_manual_visita} días de descanso), lo cual podría ser clave en los últimos 20 minutos."

            veredicto_ev = ""
            ev_local = (p_1x2[0] * cuota_1) - 1
            ev_visita = (p_1x2[2] * cuota_2) - 1
            if ev_local >= 0.05:
                veredicto_ev = f"La ineficiencia del mercado es notable: la casa paga {cuota_1} por {local}, pero según nuestro modelo la cuota real debería ser {(1/max(0.001, p_1x2[0])):.2f}. <strong>Fuerte recomendación de inversión a favor del local.</strong>"
            elif ev_visita >= 0.05:
                veredicto_ev = f"Existe Valor Esperado (+EV) claro en la visita. La cuota de {cuota_2} no refleja el {p_1x2[2]*100:.1f}% de probabilidad que estima nuestra IA."
            else:
                veredicto_ev = f"Las cuotas están perfectamente ajustadas por las casas de apuestas (mercado eficiente). <strong>Se recomienda NO invertir en ganador del partido (1X2)</strong> y buscar oportunidades en mercados secundarios como Córners o Tarjetas."

            reporte_html = f"""
            <div class='report-box'>
                <strong>🏟️ Análisis de Contexto:</strong><br>
                El encuentro se disputa en {ctx['estadio']} bajo un clima pronosticado de {ctx['clima_str']}. En el aspecto físico, {fatiga_txt} Este factor es determinante para la intensidad táctica que veremos.<br><br>
                
                <strong>⚔️ Desarrollo Táctico y Goles:</strong><br>
                Las métricas puras sugieren que <strong>{dominio_txt}</strong> buscará controlar el ritmo del partido mediante la posesión. En cuanto a peligro real, la balanza se inclina hacia <strong>{ataque_txt}</strong>. El modelo de Poisson proyecta un marcador ajustado ({hg:.2f} a {ag:.2f}), respaldado por un xG base de {xg_a_base:.2f} vs {xg_b_base:.2f}. Adicionalmente, el volumen ofensivo generará cerca de {hc+ac:.1f} saques de esquina totales.<br><br>
                
                <strong>🟨 Mercado de Disciplina (Tarjetas):</strong><br>
                El factor crítico aquí es la designación de {arbitro_nombre} (promedio severo de {arbitro_tarjetas_avg} amarillas). Cruzando su tendencia tarjetera con la agresividad histórica de ambos planteles, el algoritmo proyecta <strong>{tarjetas_proyectadas:.1f} tarjetas</strong>, lo que sugiere apuntar al mercado de OVER (Más de) en amonestaciones.<br><br>
                
                <strong>💡 Veredicto de Mercado:</strong><br>
                {veredicto_ev}
            </div>
            """
            st.markdown(reporte_html, unsafe_allow_html=True)
        else:
            st.error("No hay datos históricos suficientes para los equipos seleccionados.")
