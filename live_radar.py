import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Auto-Refresh de Streamlit
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.error("❌ Falta la dependencia 'streamlit-autorefresh'. Por favor corre: pip install streamlit-autorefresh")
    st.stop()

st.set_page_config(page_title="Deep Soccer | Live Steam Radar", layout="wide", page_icon="📡")

# Refrescar la página cada 5 segundos (Simulando web socket de The-Odds-API)
count = st_autorefresh(interval=5000, limit=100, key="fizzbuzzcounter")

st.markdown("""
<style>
    .alert-box { background-color: #3b0000; border-left: 5px solid #ff0000; padding: 20px; border-radius: 5px; margin-bottom: 20px;}
    .alert-title { color: #ff4b4b; font-size: 24px; font-weight: bold; margin-bottom: 5px; animation: blinker 1.5s linear infinite;}
    .agent-action { background-color: #002200; border-left: 5px solid #00fa9a; padding: 20px; border-radius: 5px;}
    .agent-title { color: #00fa9a; font-size: 20px; font-weight: bold; margin-bottom: 5px;}
    @keyframes blinker { 50% { opacity: 0; } }
    div[data-testid="stMetricValue"] { font-size: 32px !important; }
</style>
""", unsafe_allow_html=True)

st.title("📡 Deep Soccer: Live Steam Radar")
st.caption(f"🟢 Estado: Monitoreando 14 Casas de Apuestas (Ping: 1.2s) | Último refresco: {datetime.now().strftime('%H:%M:%S')}")
st.divider()

# Simulador Dinámico: La cuota cae progresivamente cada 5 segundos
cuota_apertura = 2.80
# La cuota baja 0.05 por cada "Tick" del refresco automático
cuota_actual = max(2.10, 2.80 - (count * 0.05)) 
caida_porcentaje = ((cuota_actual - cuota_apertura) / cuota_apertura) * 100

def generar_datos_caida(ticks):
    tiempos = [(datetime.now() - timedelta(minutes=(15-i))).strftime("%H:%M") for i in range(16)]
    # Genera una caída dramática al final basada en el contador
    cuotas_base = [2.80, 2.80, 2.81, 2.79, 2.80, 2.80, 2.81, 2.80, 2.78, 2.75, 2.65, 2.58, 2.50, 2.48, 2.46]
    cuotas_base.append(max(2.10, 2.80 - (ticks * 0.05)))
    df = pd.DataFrame({"Cuota Everton": cuotas_base, "Tiempo": tiempos})
    return df.set_index("Tiempo")

datos_mercado = generar_datos_caida(count)

col_alerta, col_grafico = st.columns([1, 2])

with col_alerta:
    st.markdown("### 🚨 Panel de Alertas en Vivo")
    
    if count > 3: # Empieza a alarmarse al 4to refresco
        st.markdown(f"""
        <div class='alert-box'>
            <div class='alert-title'>🔥 CAÍDA DETECTADA ({caida_porcentaje:.1f}%)</div>
            <strong>Partido:</strong> Everton vs Universidad de Chile<br>
            <strong>Mercado:</strong> Victoria Local (1)<br>
            <strong>Origen del flujo:</strong> Pinnacle / SBOBET (Asia)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Buscando anomalías de mercado...")
    
    c_met1, c_met2 = st.columns(2)
    c_met1.metric(label="Apertura (Open)", value=f"{cuota_apertura:.2f}")
    c_met2.metric(label="Actual (Live)", value=f"{cuota_actual:.2f}", delta=f"{cuota_actual - cuota_apertura:.2f}", delta_color="inverse")

with col_grafico:
    st.markdown("### 📉 Evolución de la Cuota (Live)")
    st.line_chart(datos_mercado, color="#ff4b4b", height=250)

st.divider()

if count > 5:
    st.markdown("### 🤖 Acción del Agente IA de Arbitraje")
    st.markdown(f"""
    <div class='agent-action'>
        <div class='agent-title'>⚠️ Movimiento masivo de Sharp Money detectado</div>
        El sistema ha detectado una inyección de capital superior a 400K USD en mercados asiáticos a favor de Everton en los últimos {count} minutos. 
        Esto sugiere información interna no pública (posible lesión clave confirmada en U. de Chile).<br><br>
        <strong>🟢 Acción Recomendada:</strong> Revisar inmediatamente casas de apuestas locales (ej. Betano, Coolbet). Si aún mantienen la cuota original por encima de 2.65, <strong>TOMAR POSICIÓN INMEDIATAMENTE</strong> antes de que sus bots de riesgo copien la caída asiática.
    </div>
    """, unsafe_allow_html=True)
