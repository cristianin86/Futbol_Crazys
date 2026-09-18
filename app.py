"""
app.py / app_master.py
======================
Deep Soccer AI - Smart Money Eigen V30.0 (Master Suite Multi-Liga).
Centro de mando y terminal analítica cuantitativa para fútbol profesional.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson

# Configuración del proyecto y rutas
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    resolve_path,
    get_secret,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    MODELS_DIR
)
from config.leagues import (
    CONFIG_LIGAS,
    get_league_config_by_name
)
from scraper_cl import (
    get_live_fixtures, get_odds, obtener_dossier_360,
    calculate_kelly, get_combined_probs,
    get_referee_metrics, get_absence_impact,
    get_live_odds, live_probabilities, devig_proportional,
    get_api_sports_key, normalize_text
)
from copa_liga_model import CopaDataPipeline, CopaLigaModel


# --- 0. GESTIÓN DE CREDENCIALES Y LLM ---
def get_ollama_host():
    return get_secret("OLLAMA_HOST") or os.getenv("OLLAMA_HOST", "http://localhost:11434")


MODELOS_EXCLUIDOS = ["orcarouter", "27b"]


@st.cache_data(ttl=120)
def listar_modelos_ollama():
    """Modelos disponibles en el servidor Ollama local (cacheado 2 min)."""
    try:
        r = requests.get(f"{get_ollama_host()}/api/tags", timeout=3)
        if r.status_code == 200:
            todos = [m["name"] for m in r.json().get("models", [])]
            return [m for m in todos if not any(excl in m.lower() for excl in MODELOS_EXCLUIDOS)]
    except Exception:
        pass
    return []


# Orden de preferencia para análisis LLM:
# Modelos prioritarios de alta capacidad y razonamiento:
PRIORIDAD_MODELOS = [
    "hf.co/mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-GGUF:Q5_K_M",
    "mradermacher/Qwen3.8-9B-Distill-uncensored-heretic-GGUF",
    "Qwen3.8-9B-Distill-uncensored-heretic",
    "heretic",
    "huihui_ai/qwen3.5-abliterated:9b",
    "huihui_ai/qwen3.5-abliterated",
    "qwen3:8b",
    "gemma3:12b",
    "gemma3:12b-it-qat",
    "llama3.1:8b"
]


def _modelo_coincide(candidato, disponible):
    """Verifica si el nombre solicitado coincide con el disponible (exacto, sin tag, substring o case-insensitive)."""
    if not candidato or not disponible:
        return False
    cand_lower = candidato.lower().strip()
    disp_lower = disponible.lower().strip()
    if cand_lower == disp_lower:
        return True
    if cand_lower in disp_lower or disp_lower in cand_lower:
        return True
    if ':' not in cand_lower and disp_lower.startswith(cand_lower + ':'):
        return True
    if ':' not in disp_lower and cand_lower.startswith(disp_lower + ':'):
        return True
    return False


def seleccionar_modelo_ollama():
    """Modelo elegido por el usuario en la UI, o el mejor disponible según prioridad."""
    disponibles = listar_modelos_ollama()
    if not disponibles:
        return None

    eleccion = st.session_state.get('ollama_model_choice')
    if eleccion:
        for d in disponibles:
            if _modelo_coincide(eleccion, d):
                return d

    env_model = get_secret("OLLAMA_MODEL")
    if env_model:
        for d in disponibles:
            if _modelo_coincide(env_model, d):
                return d

    for pref in PRIORIDAD_MODELOS:
        for d in disponibles:
            if _modelo_coincide(pref, d):
                return d

    return disponibles[0]


@st.cache_data(ttl=600)
def buscar_noticias_y_contexto_web(query, max_items=4):
    """
    Búsqueda web en vivo (Google News RSS + DuckDuckGo Lite) para inyectar
    noticias de última hora, lesiones de último minuto, polémica arbitral y clima al LLM.
    """
    resultados = []
    import urllib.parse
    import xml.etree.ElementTree as ET

    # 1. Google News RSS (tiempo real, noticias deportivas)
    try:
        q_encoded = urllib.parse.quote(query)
        url_rss = f"https://news.google.com/rss/search?q={q_encoded}&hl=es-419&gl=CL&ceid=CL:es-419"
        r = requests.get(url_rss, timeout=4)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            items = root.findall('.//item')
            for item in items[:max_items]:
                t = item.find('title')
                d = item.find('pubDate')
                titulo = t.text if t is not None and t.text else ""
                fecha = d.text if d is not None and d.text else ""
                if titulo:
                    clean_t = " - ".join(titulo.split(" - ")[:-1]) if " - " in titulo else titulo
                    f_corta = fecha[:16] if fecha else "Reciente"
                    resultados.append(f"📰 {clean_t} ({f_corta})")
    except Exception:
        pass

    # 2. DuckDuckGo Lite (snippets web si faltan resultados)
    if len(resultados) < max_items:
        try:
            import bs4
            r_ddg = requests.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=4
            )
            if r_ddg.status_code == 200:
                soup = bs4.BeautifulSoup(r_ddg.text, "html.parser")
                snippets = soup.find_all("td", class_="result-snippet")
                for s in snippets[:max_items - len(resultados)]:
                    txt = s.text.strip()
                    if txt and len(txt) > 25:
                        resultados.append(f"🌐 {txt[:180]}...")
        except Exception:
            pass

    if not resultados:
        return "No se detectaron reportes web recientes de última hora."
    return "\n".join(resultados)


def extraer_thinking_y_contenido(texto):
    """
    Separa de manera robusta el razonamiento analítico (thinking) del informe o respuesta final,
    soportando etiquetas </think>, 'Thinking Process:', 'Analyze the Request:' o encabezados de sección.
    """
    if not texto:
        return "", ""
    texto = texto.strip()
    thinking = ""
    contenido = texto

    # Caso A: Etiqueta de cierre </think>
    if "</think>" in contenido:
        partes = contenido.split("</think>", 1)
        thinking = partes[0].replace("<think>", "").strip()
        contenido = partes[1].strip()
    # Caso B: Encabezados de razonamiento de modelos destilados
    elif any(tag in contenido for tag in ["Thinking Process:", "Analyze the Request:", "Proceso de Pensamiento:"]):
        for tag in ["Thinking Process:", "Analyze the Request:", "Proceso de Pensamiento:"]:
            if tag in contenido:
                partes = contenido.split(tag, 1)
                resto = partes[1] if len(partes) > 1 else partes[0]
                corte_encontrado = False
                for sep in ["### 1.", "### 1", "# 1.", "1. ⚔️", "1. ", "## 1", "## ", "\n# "]:
                    if sep in resto:
                        p_rep = resto.split(sep, 1)
                        thinking = p_rep[0].strip()
                        contenido = sep + (" " if not sep.endswith(" ") else "") + p_rep[1].strip()
                        corte_encontrado = True
                        break
                if not corte_encontrado:
                    contenido = resto.strip()
                break

    return thinking, contenido


def chat_llm(messages, temperatura=0.4):
    """
    Chat conversacional híbrido: Ollama local (/api/chat) con fallback a Groq.
    messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
    Devuelve (texto, etiqueta_motor) o (None, None).
    """
    modelo = seleccionar_modelo_ollama()
    if modelo:
        try:
            r = requests.post(f"{get_ollama_host()}/api/chat", json={
                "model": modelo,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperatura, "top_p": 0.9,
                            "num_predict": 3500, "num_ctx": 8192}
            }, timeout=180)
            if r.status_code == 200:
                res_json = r.json()
                msg = res_json.get("message", {})
                raw_contenido = msg.get("content", "").strip()
                thinking = msg.get("thinking", "").strip()

                if not thinking:
                    thinking, contenido = extraer_thinking_y_contenido(raw_contenido)
                else:
                    contenido = raw_contenido

                if not contenido and thinking:
                    contenido = thinking
                elif thinking and contenido:
                    contenido = (
                        f"<details style='margin-bottom: 10px; background: rgba(0, 242, 254, 0.04); padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(0, 242, 254, 0.2);'>"
                        f"<summary style='cursor: pointer; font-weight: 600; font-size: 0.88rem; color: #00f2fe;'>🧠 Ver Razonamiento y Cálculos del Analista (Pensamiento en Español)</summary>\n\n"
                        f"<div style='margin-top: 8px; font-size: 0.83rem; color: #cbd5e1; line-height: 1.5; white-space: pre-wrap;'>{thinking}</div>"
                        f"</details>\n\n{contenido}"
                    )
                if contenido:
                    return contenido, f"Ollama · {modelo}"
        except Exception:
            pass

    api_key = get_secret("GROQ_API_KEY")
    if api_key:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {api_key}",
                                       "Content-Type": "application/json"},
                              json={"model": "llama-3.3-70b-versatile",
                                    "messages": messages, "temperature": temperatura},
                              timeout=45)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], "Groq · Llama 3.3 70B"
        except Exception:
            pass
    return None, None


def construir_contexto_chat(local, visita, inf_res, dossier, odds_data, arbitro, liga):
    """Contexto compacto del partido para el chatbot con directrices de razonamiento cuantitativo sin censura y búsqueda web."""
    p = inf_res['p_win'] if inf_res else [0.33, 0.34, 0.33]
    hg = inf_res['local_goals_proyected'] if inf_res else 1.3
    ag = inf_res['visitor_goals_proyected'] if inf_res else 1.1
    odds_1x2 = odds_data.get('1x2', {}) if isinstance(odds_data, dict) else {}
    mejor = {k: v.get('Best') for k, v in odds_1x2.items()} if odds_1x2 else {}

    # Búsqueda web en vivo de noticias de última hora
    noticias_web = buscar_noticias_y_contexto_web(f"{local} vs {visita} futbol noticias bajas lesiones polemica", max_items=3)

    return f"""Eres el Analista Cuantitativo y Táctico Sin Filtros de Deep Soccer AI. Operas con máxima inteligencia, sin censura moralista, sin eufemismos y con rigor matemático absoluto. Tu único compromiso es la verdad estadística despiadada y la detección de valor (+EV).

IDIOMA ESTRICTAMENTE OBLIGATORIO:
TODO tu proceso de pensamiento y razonamiento analítico dentro de <think>...</think> y todas tus respuestas DEBEN estar redactadas 100% en ESPAÑOL. Queda terminantemente prohibido pensar o redactar en inglés o usar expresiones como 'Thinking Process:'.

REGLA DE ORO DE PRECISIÓN CUANTITATIVA (CERO VAGUEDADES):
Prohibido dar opiniones o comentarios vagos sin respaldo numérico. Justifica SIEMPRE cada veredicto citando las cifras numéricas del modelo: probabilidades %, goles esperados xG, córners, tarjetas, cuotas de mercado y cálculo explícito de EV = (Probabilidad * Cuota) - 1.

DATOS MAESTROS DEL PARTIDO ({liga}):
- Duelo: {local} (local) vs {visita} (visita) | Árbitro designado: {arbitro}
- Modelo Matemático V10 (probabilidades calibradas):
  * Local ({local}): {p[0]*100:.1f}%
  * Empate: {p[1]*100:.1f}%
  * Visitante ({visita}): {p[2]*100:.1f}%
- Proyecciones de Eventos:
  * Goles Esperados: {local} {hg:.2f} vs {ag:.2f} {visita} (Ambos Anotan / BTTS: {(inf_res.get('p_btts', 0.5)*100 if inf_res else 50):.0f}%)
  * Córners Totales Proyectados: {(inf_res['total_corners'] if inf_res else 9.5):.1f}
  * Tarjetas/Fricción Estimada: {(inf_res['total_cards'] if inf_res else 4.5):.1f}
- Mercado de Cuotas 1X2 (Mejor cuota disponible): {mejor if mejor else 'Sin cuotas de mercado activas en este momento'}
- Contexto de Clasificación: {local}: {dossier.get('posicion_local', 'N/A')} | {visita}: {dossier.get('posicion_visita', 'N/A')}
- Dinámica Reciente (ORDEN CRONOLÓGICO: MÁS RECIENTE AL MÁS ANTIGUO):
  * {local}: {dossier.get('ultimos_partidos_local', dossier.get('forma_local', 'N/A'))}
  * {visita}: {dossier.get('ultimos_partidos_visita', dossier.get('forma_visita', 'N/A'))}
  * Micro-Inercia: Local: {dossier.get('micro_inercia_local', 'N/A')} | Visita: {dossier.get('micro_inercia_visita', 'N/A')}
- Bajas Confirmadas: Local: {dossier.get('bajas_local', 'Sin bajas')} | Visita: {dossier.get('bajas_visita', 'Sin bajas')}
- Historial H2H: {dossier.get('resumen_h2h', 'Sin registros recientes')}

NOTICIAS Y REÚMENES WEB EN VIVO (INTELIGENCIA EN TIEMPO REAL):
{noticias_web}

DIRECTRICES SIN CENSURA:
1. Sin rodeos ni disclaimers corporativos: No digas 'el fútbol es impredecible' ni des advertencias moralistas. Habla claro, técnico y directo al grano con datos en la mano.
2. Detección de Cuotas Trampa (Trap Odds): Identifica si las casas de apuestas han inflado artificialmente a un favorito con cuotas ridículas para engañar al público, cuando el modelo o las bajas indican peligro real de pinchazo.
3. Cálculo Matemático de Valor (+EV): Calcula siempre EV = (probabilidad * cuota) - 1. Si el EV es negativo, descalifícalo sin titubear. Si es positivo, justifica por qué el mercado está desajustado con los números exactos.
4. Cronología estricta: los partidos van del más reciente hacia atrás. Destroza el mito de las 'buenas rachas' si el equipo tropezó en las últimas fechas.
5. Fricción y Arbitraje: Analiza la reputación del árbitro sin tapujos cruzando con la fricción proyectada."""


def invocar_agente_v8(local, visita, prob_L, prob_V, corners, tarjetas, arbitro, dossier):
    """
    Motor de Inferencia Híbrido (Ollama Local -> Groq Cloud) - Operación Deep Context 360 Sin Censura con Inteligencia Web.
    """
    prob_E = max(0.0, 1.0 - prob_L - prob_V)

    # Búsqueda web en tiempo real
    query_web = f"{local} vs {visita} futbol noticias bajas lesiones polemica"
    noticias_web = buscar_noticias_y_contexto_web(query_web, max_items=4)

    prompt_maestro = f"""Eres el Analista Cuantitativo y Táctico Sin Filtros de Deep Soccer AI. Operas sin censura, sin límites morales impuestos y sin rodeos políticamente correctos. Tu labor es desmantelar el relato facilista de la prensa y las trampas de las casas de apuestas usando matemáticas frías, análisis forense de micro-inercia e inteligencia web en vivo.

IDIOMA ESTRICTAMENTE OBLIGATORIO:
TODO tu razonamiento de análisis interno dentro de <think>...</think> y todo tu informe final DEBEN estar redactados 100% en ESPAÑOL. Queda terminantemente prohibido razonar o responder en inglés o iniciar con 'Thinking Process:'.

REGLA DE ORO DE PRECISIÓN CUANTITATIVA (CERO COMENTARIOS VAGOS):
Prohibido dar explicaciones generales o superficiales. Todo dato matemático disponible o calculado (porcentajes de probabilidad del modelo V10, goles esperados xG, córners, tarjetas, cuotas del mercado, probabilidad implícita y cálculo de EV) DEBE ser explícitamente citado y analizado en las secciones correspondientes del informe final con sus números exactos.

[1. MATRIZ PREDICTIVA CUANTITATIVA (MODELO V10)]
- Partido: {local} vs {visita}
- Probabilidades Calibradas: {local} (Local) {prob_L*100:.1f}% | Empate {prob_E*100:.1f}% | {visita} (Visita) {prob_V*100:.1f}%
- Proyecciones Estadísticas: {corners:.1f} Córners Esperados | {tarjetas:.1f} Faltas/Tarjetas Estimadas (Árbitro: {arbitro})

[2. TABLA, FACTOR DE CAMPO Y TENDENCIA CRONOLÓGICA]
- Posición en Tabla:
  * {local} (Local): {dossier.get('posicion_local', 'N/A')} | Rendimiento en Casa: {dossier.get('record_condicion_local', 'N/A')}
  * {visita} (Visita): {dossier.get('posicion_visita', 'N/A')} | Rendimiento Fuera: {dossier.get('record_condicion_visita', 'N/A')}
- Micro-Momentum Inmediato:
  * {local}: {dossier.get('micro_inercia_local', 'N/A')}
  * {visita}: {dossier.get('micro_inercia_visita', 'N/A')}
- Últimos 5 Partidos (CRONOLÓGICO: DEL MÁS RECIENTE AL MÁS ANTIGUO):
  * {local}: {dossier.get('ultimos_partidos_local', dossier.get('forma_local', 'N/A'))}
  * {visita}: {dossier.get('ultimos_partidos_visita', dossier.get('forma_visita', 'N/A'))}
- Acumulado Temporada:
  * {local}: {dossier.get('pg_local', 0)}V-{dossier.get('pe_local', 0)}E-{dossier.get('pp_local', 0)}D ({dossier.get('gf_local', 0)} GF, {dossier.get('gc_local', 0)} GC)
  * {visita}: {dossier.get('pg_visita', 0)}V-{dossier.get('pe_visita', 0)}E-{dossier.get('pp_visita', 0)}D ({dossier.get('gf_visita', 0)} GF, {dossier.get('gc_visita', 0)} GC)
- Producción Goleadora: Local: {dossier.get('goles_avg_local', 1.0):.1f} GF/p ({dossier.get('clean_sheet_local', 0)} vallas invictas) | Visita: {dossier.get('goles_avg_visita', 1.0):.1f} GF/p ({dossier.get('clean_sheet_visita', 0)} vallas invictas)

[3. RADIOGRAFÍA TÁCTICA Y DISCIPLINA]
- Formación Local: {dossier.get('formacion_local', '4-3-3')} | Posesión Promedio: {dossier.get('posesion_local', 50.0)}%
- Formación Visita: {dossier.get('formacion_visita', '4-4-2')} | Posesión Promedio: {dossier.get('posesion_visita', 50.0)}%
- Disciplina: Local: {dossier.get('tarjetas_amarillas_local', 0)} Amarillas, {dossier.get('tarjetas_rojas_local', 0)} Rojas | Visita: {dossier.get('tarjetas_amarillas_visita', 0)} Amarillas, {dossier.get('tarjetas_rojas_visita', 0)} Rojas

[4. PARTE MÉDICO Y ANTECEDENTES DIRECTOS]
- Ausencias Confirmadas Local: {dossier.get('bajas_local', 'Sin bajas confirmadas')}
- Ausencias Confirmadas Visita: {dossier.get('bajas_visita', 'Sin bajas confirmadas')}
- Historial H2H Reciente: {dossier.get('resumen_h2h', 'Sin historial reciente')}

[5. INTELIGENCIA WEB EN VIVO (ÚLTIMAS NOTICIAS Y REPORTES)]
{noticias_web}

INSTRUCCIONES DE EJECUCIÓN (PASO A PASO OBLIGATORIO):
1. PENSAMIENTO Y ANÁLISIS INTERNO (DENTRO DE <think>...</think> 100% EN ESPAÑOL):
   - Abre la etiqueta <think> e inicia tu análisis forense en español.
   - Analiza las probabilidades del modelo V10 ({local} {prob_L*100:.1f}%, Empate {prob_E*100:.1f}%, {visita} {prob_V*100:.1f}%) y calcula las cuotas justas teóricas (1 / Probabilidad).
   - Compara con las cuotas reales del mercado y calcula explícitamente el Valor Esperado: EV = (Probabilidad * Cuota) - 1.
   - Pesa con frialdad las bajas confirmadas y las noticias web de última hora.
   - Cierra con </think>.

2. INFORME FINAL ESTRUCTURADO (DESPUÉS DE </think>, 100% EN ESPAÑOL):
   - Redacta las 3 secciones obligatorias.
   - REGLA CRÍTICA: Debes VOLCAR obligatoriamente los datos, porcentajes y cálculos numéricos realizados en el paso previo. No resumas con opiniones vacías; cada conclusión debe tener el soporte de las cifras numéricas del modelo.

FORMATO DEL INFORME (3 SECCIONES ESTRUCTURADAS EN ESPAÑOL):

### 1. ⚔️ Choque Táctico, Factor Campo e Inteligencia de Última Hora
Analiza el emparejamiento táctico (formaciones, posesión {dossier.get('posesion_local', 50.0):.1f}% vs {dossier.get('posesion_visita', 50.0):.1f}%), el impacto real de las ausencias confirmadas y las noticias web de última hora. Cita los datos de producción goleadora ({dossier.get('goles_avg_local', 1.0):.1f} vs {dossier.get('goles_avg_visita', 1.0):.1f} GF/p) y vallas invictas. ¿Quién llega realmente condicionado?

### 2. 📉 Radiografía Forense de Momentum, Fricción y Arbitraje
Desglosa la inercia real partido a partido (en estricto orden cronológico: del más reciente al más antiguo). Señala baches ocultos, desajustes defensivos y si las estadísticas de temporada ocultan un declive reciente. Cruza las proyecciones de fricción ({tarjetas:.1f} faltas/tarjetas estimadas) y córners ({corners:.1f} esperados) con el perfil del árbitro ({arbitro}) y las tarjetas acumuladas.

### 3. 🎯 Veredicto Cuantitativo: Desmontando Cuotas y Escenarios de Alto Valor (+EV)
Cruza la matemática fría del Modelo V10 ({local} {prob_L*100:.1f}% | Empate {prob_E*100:.1f}% | {visita} {prob_V*100:.1f}%) contra el mercado.
- Presenta el cálculo explícito de cuota justa teórica vs cuota real y el cálculo numérico de EV = (Probabilidad * Cuota) - 1 para las opciones evaluadas.
- Identifica cuotas trampa (Trap Odds) si las casas sobrevaloran a un equipo sin justificación estadística.
- Dictamina con honestidad brutal y justificación cuantitativa el mejor escenario de inversión de alto valor (+EV): ya sea en 1X2 / Doble Oportunidad, Over/Under de Goles o Córners/Tarjetas."""

    # --- INTENTO 1: OLLAMA LOCAL ---
    modelo_a_usar = seleccionar_modelo_ollama()
    ollama_host = get_ollama_host()

    if modelo_a_usar:
        try:
            payload_local = {
                "model": modelo_a_usar,
                "prompt": prompt_maestro,
                "stream": False,
                "options": {
                    "temperature": 0.35,
                    "top_p": 0.9,
                    "num_predict": 3500,
                    "num_ctx": 8192
                }
            }
            response_local = requests.post(f"{ollama_host}/api/generate", json=payload_local, timeout=180)
            if response_local.status_code == 200:
                res_json = response_local.json()
                raw_text = res_json.get("response", "").strip()
                thinking = res_json.get("thinking", "").strip()

                if not thinking:
                    thinking, informe = extraer_thinking_y_contenido(raw_text)
                else:
                    informe = raw_text

                if not informe and thinking:
                    informe = thinking
                elif thinking and informe:
                    informe = (
                        f"<details style='margin-bottom: 14px; background: rgba(0, 242, 254, 0.05); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(0, 242, 254, 0.25);'>"
                        f"<summary style='cursor: pointer; font-weight: 600; font-size: 0.95rem; color: #00f2fe;'>🧠 Ver Análisis Cuantitativo y Razonamiento Forense (Thinking en Español)</summary>\n\n"
                        f"<div style='margin-top: 10px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; white-space: pre-wrap;'>{thinking}</div>"
                        f"</details>\n\n{informe}"
                    )
                if informe:
                    return f"🤖 **IA Local ({modelo_a_usar}):**\n\n" + informe
        except Exception:
            pass

    # --- INTENTO 2: GROQ CLOUD ---
    api_key = get_secret("GROQ_API_KEY")
    if api_key:
        MODELO_NUBE = "llama-3.3-70b-versatile"
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
            response = requests.post(url, headers=headers, json=payload, timeout=30) 
            response.raise_for_status()
            res_json = response.json()
            return f"⚡ **IA Cloud (Groq Llama 3):**\n\n" + res_json["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            return f"🚨 Error en la Nube (API Groq): {response.text}"
        except Exception as e:
            return f"🚨 Falla conectando con Groq: {e}"

    return "🚨 ALERTA: No se pudo conectar con Ollama (Local) ni se encontró GROQ_API_KEY en las variables de entorno/Secrets. Por favor inicie Ollama o configure la API Key."


def poisson_over(lam, line):
    """Calcula la probabilidad de Superar (Over) una línea dada bajo distribución Poisson."""
    k = int(np.floor(line))
    return float(1.0 - poisson.cdf(k, float(lam)))


# --- GESTIÓN DE CACHÉ Y RUTAS ---
HISTORY_FILE = resolve_path("odds_trend_memory.json", CACHE_DIR)

def load_odds_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_odds_history(history):
    try:
        if len(history) > 1000:
            keys_to_keep = list(history.keys())[-1000:]
            history = {k: history[k] for k in keys_to_keep}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding='utf-8') as f:
            json.dump(history, f)
    except Exception:
        pass


@st.cache_data(ttl=600)
def cargar_master_data_v30(config):
    """Inyecta la grilla oficial y el dataset histórico dinámicamente."""
    df_grilla = get_live_fixtures(
        league_id=config['id_api'], 
        season=config['season'], 
        translator=config['traductor']
    )
    
    # Estandarizar columnas de df_grilla para compatibilidad absoluta
    if not df_grilla.empty:
        if 'Partido_String' in df_grilla.columns and 'Match' not in df_grilla.columns:
            df_grilla['Match'] = df_grilla['Partido_String']
        elif 'Match' in df_grilla.columns and 'Partido_String' not in df_grilla.columns:
            df_grilla['Partido_String'] = df_grilla['Match']
            
        if 'Local' in df_grilla.columns and 'Home_Team' not in df_grilla.columns:
            df_grilla['Home_Team'] = df_grilla['Local']
        if 'Visita' in df_grilla.columns and 'Away_Team' not in df_grilla.columns:
            df_grilla['Away_Team'] = df_grilla['Visita']
        if 'id' in df_grilla.columns and 'Fixture_ID' not in df_grilla.columns:
            df_grilla['Fixture_ID'] = df_grilla['id']
        if 'Local_ID' in df_grilla.columns and 'Home_ID' not in df_grilla.columns:
            df_grilla['Home_ID'] = df_grilla['Local_ID']
        if 'Visita_ID' in df_grilla.columns and 'Away_ID' not in df_grilla.columns:
            df_grilla['Away_ID'] = df_grilla['Visita_ID']
            
    try:
        ds_path = resolve_path(config['dataset'], PROCESSED_DATA_DIR)
        df_hist = pd.read_csv(ds_path, encoding='utf-8', encoding_errors='replace')
        df_hist.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df_hist.columns]
        df_hist = df_hist.ffill().bfill()
        
        targets = ['target_1x2', 'home_team_goal_count', 'away_team_goal_count']
        feats = [c for c in df_hist.columns if c not in targets and c != 'timestamp']
        return df_grilla, df_hist, feats
    except Exception as e:
        return df_grilla, None, []


@st.cache_resource(ttl=600)
def cargar_modelos_v30(liga):
    """
    Carga el comité XGBoost + metadata de entrenamiento de la liga.
    V10: un solo pipeline de modelos para TODAS las ligas (antes Chile usaba
    un comité v1 paralelo con otro motor de probabilidad). Cacheado como recurso
    (antes se deserializaban los boosters en cada rerun).
    """
    mods = {}
    cfg = get_league_config_by_name(liga)
    s = f"_{cfg['suffix']}" if cfg.get('suffix') else ""

    nombres = [('1x2', f'model_1x2_v5{s}.json'), ('hg', f'model_hg_v5{s}.json'),
               ('ag', f'model_ag_v5{s}.json'), ('hc', f'model_hc_v5{s}.json'),
               ('ac', f'model_ac_v5{s}.json'), ('hy', f'model_hy_v5{s}.json'),
               ('ay', f'model_ay_v5{s}.json')]

    for m, f in nombres:
        f_path = resolve_path(f, MODELS_DIR)
        if f_path.exists():
            try:
                b = xgb.Booster()
                b.load_model(str(f_path))
                mods[m] = b
            except Exception:
                pass

    # Metadata de entrenamiento (features, medias para imputación, temperatura, métricas)
    metadata = {}
    meta_path = resolve_path(f'metadata{s}.json', MODELS_DIR)
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return mods, metadata


def obtener_stats_aisladas(team_name, raw_csv, condicion, translator=None):
    """Escanea el historial crudo para extraer el momentum real aislado (últimos 5
    en la condición local/visita). Usa matching normalizado + traductor de la liga
    activa (antes se usaba siempre el traductor chileno para las 6 ligas)."""
    raw_path = resolve_path(raw_csv, RAW_DATA_DIR)
    try:
        df_h = pd.read_csv(raw_path, encoding='utf-8', encoding_errors='replace').sort_values('timestamp')
    except Exception as e:
        raise ValueError(f"No se pudo cargar el historial {raw_csv}: {e}")

    if 'status' in df_h.columns:
        df_h = df_h[df_h['status'] == 'complete']
    else:
        df_h = df_h.dropna(subset=['home_team_goal_count'])

    # Matching robusto: exacto -> normalizado con traductor de liga
    target_norm = normalize_text(team_name, translator)
    equipos_csv = pd.unique(pd.concat([df_h['home_team_name'], df_h['away_team_name']]))
    equipo = team_name
    if team_name not in equipos_csv:
        for cand in equipos_csv:
            if normalize_text(cand, translator) == target_norm:
                equipo = cand
                break

    col_cond = 'home_team_name' if condicion == 'Local' else 'away_team_name'
    df_cond_all = df_h[df_h[col_cond] == equipo]
    n_cond = len(df_cond_all)
    df_filtrado = df_cond_all.sort_values('timestamp', ascending=False).head(5)

    fallback = False
    if df_filtrado.empty:
        df_filtrado = df_h[(df_h['home_team_name'] == equipo) | (df_h['away_team_name'] == equipo)]
        df_filtrado = df_filtrado.sort_values('timestamp', ascending=False).head(5)

        if df_filtrado.empty:
            # Equipo sin historial (recién ascendido): media de la liga POR CONDICION,
            # marcada como fallback. Antes usaba siempre las columnas home_*, lo que
            # inflaba 'gf' y deflacionaba 'gc' de equipos nuevos jugando de visita
            # (sobreestimando la localia del rival en ese partido).
            col_f, col_c = ('home_team_goal_count', 'away_team_goal_count') if condicion == 'Local' \
                else ('away_team_goal_count', 'home_team_goal_count')
            col_pos = 'home_team_possession' if condicion == 'Local' else 'away_team_possession'
            col_xg = 'home_team_pre_match_xg' if condicion == 'Local' else 'away_team_pre_match_xg'
            col_corners = 'home_team_corner_count' if condicion == 'Local' else 'away_team_corner_count'
            col_yc = 'home_team_yellow_cards' if condicion == 'Local' else 'away_team_yellow_cards'
            col_sot = 'home_team_shots_on_target' if condicion == 'Local' else 'away_team_shots_on_target'

            def col_mean(c, d):
                return float(df_h[c].mean()) if c in df_h.columns and df_h[c].notna().any() else d
            return {
                'gf': col_mean(col_f, 1.2 if condicion == 'Local' else 1.1),
                'gc': col_mean(col_c, 1.1 if condicion == 'Local' else 1.2),
                'ppg': 1.0, 'pos': col_mean(col_pos, 50.0),
                'xg': col_mean(col_xg, 1.3), 'corners': col_mean(col_corners, 4.8),
                'yc': col_mean(col_yc, 2.4), 'sot': col_mean(col_sot, 4.2),
                'fallback': True, 'n': 0
            }

    # Constante de encogimiento Bayesiano hacia la media de liga (empírico de Bayes)
    # Coherente al 100% con K_SHRINKAGE = 2.0 en processor.py (Ronda 2)
    K_SHRINKAGE = 2.0
    w = float(n_cond / (n_cond + K_SHRINKAGE))

    df_equipo = df_filtrado
    n_partidos = len(df_equipo)
    is_home = df_equipo['home_team_name'] == equipo

    def avg_pair(col_h, col_a, default):
        if col_h not in df_equipo.columns or col_a not in df_equipo.columns:
            return default
        vals = np.where(is_home, df_equipo[col_h], df_equipo[col_a]).astype(float)
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if len(vals) else default

    # Medias de liga para esa condición específica sobre todo el historial disponible
    col_f, col_c = ('home_team_goal_count', 'away_team_goal_count') if condicion == 'Local' \
        else ('away_team_goal_count', 'home_team_goal_count')
    col_pos = 'home_team_possession' if condicion == 'Local' else 'away_team_possession'
    col_xg = 'home_team_pre_match_xg' if condicion == 'Local' else 'away_team_pre_match_xg'
    col_corners = 'home_team_corner_count' if condicion == 'Local' else 'away_team_corner_count'
    col_yc = 'home_team_yellow_cards' if condicion == 'Local' else 'away_team_yellow_cards'
    col_sot = 'home_team_shots_on_target' if condicion == 'Local' else 'away_team_shots_on_target'

    def col_mean_league(c, d):
        return float(df_h[c].dropna().mean()) if c in df_h.columns and df_h[c].notna().any() else d

    # Medias recientes del equipo (últimos 5 partidos)
    raw_gf = avg_pair('home_team_goal_count', 'away_team_goal_count', 1.2)
    raw_gc = avg_pair('away_team_goal_count', 'home_team_goal_count', 1.1)
    raw_pos = avg_pair('home_team_possession', 'away_team_possession', 50.0)
    raw_xg = avg_pair('home_team_pre_match_xg', 'away_team_pre_match_xg', 1.3)
    raw_corners = avg_pair('home_team_corner_count', 'away_team_corner_count', 4.8)
    raw_yc = avg_pair('home_team_yellow_cards', 'away_team_yellow_cards', 2.4)
    raw_sot = avg_pair('home_team_shots_on_target', 'away_team_shots_on_target', 4.2)

    wins = np.where(is_home,
                    df_equipo['home_team_goal_count'] > df_equipo['away_team_goal_count'],
                    df_equipo['away_team_goal_count'] > df_equipo['home_team_goal_count'])
    draws = df_equipo['home_team_goal_count'] == df_equipo['away_team_goal_count']
    raw_ppg = ((sum(wins) * 3) + sum(draws)) / n_partidos

    # League PPG en esa condición
    h_wins = (df_h['home_team_goal_count'] > df_h['away_team_goal_count']).astype(int)
    a_wins = (df_h['away_team_goal_count'] > df_h['home_team_goal_count']).astype(int)
    drws = (df_h['home_team_goal_count'] == df_h['away_team_goal_count']).astype(int)
    league_ppg = float(((h_wins * 3 + drws).mean()) if condicion == 'Local' else ((a_wins * 3 + drws).mean()))

    # Aplicar Shrinkage Bayesiano hacia la media de la liga
    gf = float(w * raw_gf + (1.0 - w) * col_mean_league(col_f, 1.2 if condicion == 'Local' else 1.1))
    gc = float(w * raw_gc + (1.0 - w) * col_mean_league(col_c, 1.1 if condicion == 'Local' else 1.2))
    pos = float(w * raw_pos + (1.0 - w) * col_mean_league(col_pos, 50.0))
    xg = float(w * raw_xg + (1.0 - w) * col_mean_league(col_xg, 1.3))
    corners = float(w * raw_corners + (1.0 - w) * col_mean_league(col_corners, 4.8))
    yc = float(w * raw_yc + (1.0 - w) * col_mean_league(col_yc, 2.4))
    sot = float(w * raw_sot + (1.0 - w) * col_mean_league(col_sot, 4.2))
    ppg = float(w * raw_ppg + (1.0 - w) * league_ppg)

    return {'gf': gf, 'gc': gc, 'ppg': ppg, 'pos': pos, 'xg': xg,
            'corners': corners, 'yc': yc, 'sot': sot, 'fallback': fallback, 'n': n_cond}


def dixon_coles_matrix(hg, ag, rho=-0.08, max_goals=11):
    """Matriz de marcadores Poisson con corrección tau de Dixon-Coles (1997)
    para la dependencia en marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    p_A = np.array([poisson.pmf(i, hg) for i in range(max_goals)])
    p_B = np.array([poisson.pmf(j, ag) for j in range(max_goals)])
    matrix = np.outer(p_A, p_B)
    # Corrección tau
    matrix[0, 0] *= 1.0 - (hg * ag * rho)
    matrix[0, 1] *= 1.0 + (hg * rho)
    matrix[1, 0] *= 1.0 + (ag * rho)
    matrix[1, 1] *= 1.0 - rho
    matrix = np.maximum(matrix, 0)
    matrix /= matrix.sum()
    return matrix


def aplicar_temperatura(probs, temperatura):
    """Calibración por temperatura: p_cal ∝ p^(1/T)."""
    if not temperatura or temperatura <= 0:
        return probs
    p = np.power(np.clip(np.array(probs, dtype=float), 1e-12, 1.0), 1.0 / temperatura)
    return (p / p.sum()).tolist()


def run_master_inference(local, visita, config, is_cup=0, lsi_local=1.0, lsi_visita=1.0,
                         modelos_dict=None, metadata=None, ref_media=None, odds_data=None):
    """
    Motor de inferencia V10 unificado para todas las ligas:
    1. Vector de features guiado por los nombres del booster (imputación con medias de entrenamiento).
    2. Clasificador 1X2 calibrado por temperatura.
    3. Matriz Poisson Dixon-Coles sobre los goles proyectados (con LSI aplicado y clamp).
    4. Blend 1X2 fusionado (clasificador + matriz + probabilidad implícita de mercado).
    5. Tarjetas: modelos hy/ay si existen (datos reales), si no rolling + ajuste por árbitro.
    """
    modelos = modelos_dict or {}
    metadata = metadata or {}
    translator = config.get('traductor')
    l_s = obtener_stats_aisladas(local, config['dataset_raw'], 'Local', translator)
    v_s = obtener_stats_aisladas(visita, config['dataset_raw'], 'Visita', translator)

    # LSI con clamp: un XI debilitado baja la proyección; nunca la infla más de 8%
    lsi_local_safe = float(np.clip(lsi_local if lsi_local > 0 else 1.0, 0.75, 1.08))
    lsi_visita_safe = float(np.clip(lsi_visita if lsi_visita > 0 else 1.0, 0.75, 1.08))

    # Diccionario completo de features candidatas
    feats_all = {
        'l_gf': l_s['gf'], 'l_gc': l_s['gc'], 'l_ppg': l_s['ppg'], 'l_pos': l_s['pos'],
        'l_xg': l_s['xg'], 'l_yc': l_s['yc'], 'l_sot': l_s['sot'],
        'v_gf': v_s['gf'], 'v_gc': v_s['gc'], 'v_ppg': v_s['ppg'], 'v_pos': v_s['pos'],
        'v_xg': v_s['xg'], 'v_yc': v_s['yc'], 'v_sot': v_s['sot'],
        'is_cup': is_cup, 'home_team_lsi': lsi_local_safe, 'away_team_lsi': lsi_visita_safe,
    }
    means = metadata.get('feature_means', {})

    def dmatrix_para(booster):
        nombres = booster.feature_names or list(feats_all.keys())
        vals = [feats_all.get(n, means.get(n, 0.0)) for n in nombres]
        X = pd.DataFrame([vals], columns=nombres).astype(float)
        return xgb.DMatrix(X)

    # --- Goles esperados ---
    hg = float(modelos['hg'].predict(dmatrix_para(modelos['hg']))[0]) if 'hg' in modelos else max(l_s['gf'], 0.4)
    ag = float(modelos['ag'].predict(dmatrix_para(modelos['ag']))[0]) if 'ag' in modelos else max(v_s['gf'], 0.3)
    hg = max(hg * lsi_local_safe, 0.05)
    ag = max(ag * lsi_visita_safe, 0.05)

    # --- 1X2: clasificador calibrado + matriz Dixon-Coles ---
    if '1x2' in modelos:
        p_clf = modelos['1x2'].predict(dmatrix_para(modelos['1x2']))[0].tolist()
        temp = metadata.get('models', {}).get('1x2', {}).get('temperature')
        p_clf = aplicar_temperatura(p_clf, temp)
    else:
        p_clf = [0.33, 0.34, 0.33]

    matrix = dixon_coles_matrix(hg, ag)
    p_matrix = [float(np.sum(np.tril(matrix, -1))), float(np.sum(np.diag(matrix))), float(np.sum(np.triu(matrix, 1)))]

    # --- Fusión del blend 1X2 (Ronda 2: rebalanceo coherente) ---
    # Pesos configurables expuestos como constantes (deben sumar 1.0):
    # W_CLF: clasificador XGBoost calibrado
    # W_MATRIX: matriz Poisson / Dixon-Coles
    # W_MARKET: probabilidad implícita del mercado (cuotas desvigadas)
    # En ausencia de mercado, fallback normaliza a 25/60 (~41.7%) CLF y 35/60 (~58.3%) MATRIX
    W_CLF = 0.25
    W_MATRIX = 0.35
    W_MARKET = 0.40

    p_market = None
    market_used = False
    if odds_data and isinstance(odds_data, dict):
        odds_1x2 = odds_data.get('1x2', {})
        if isinstance(odds_1x2, dict):
            # Prioridad 1: Pinnacle (línea sharp)
            odds_map = {k: odds_1x2[k].get('Pinnacle') for k in ['Home', 'Draw', 'Away'] if k in odds_1x2 and odds_1x2[k].get('Pinnacle')}
            # Prioridad 2: Mejor cuota disponible ('Best')
            if len(odds_map) < 3:
                odds_map = {k: odds_1x2[k].get('Best') for k in ['Home', 'Draw', 'Away'] if k in odds_1x2 and odds_1x2[k].get('Best')}
            # Prioridad 3: Cualquier cuota válida disponible por resultado
            if len(odds_map) < 3:
                odds_map = {}
                for k in ['Home', 'Draw', 'Away']:
                    for bk, val in odds_1x2.get(k, {}).items():
                        if val and float(val) > 1.0:
                            odds_map[k] = val
                            break
            if len(odds_map) == 3:
                fair_odds = devig_proportional(odds_map, target_sum=1.0)
                if 'Home' in fair_odds and 'Draw' in fair_odds and 'Away' in fair_odds:
                    p_market = [float(fair_odds['Home']), float(fair_odds['Draw']), float(fair_odds['Away'])]
                    market_used = True

    if market_used and p_market is not None:
        p_win = [W_CLF * c + W_MATRIX * m + W_MARKET * mk for c, m, mk in zip(p_clf, p_matrix, p_market)]
        blend_weights = {'clf': W_CLF, 'matrix': W_MATRIX, 'market': W_MARKET}
    else:
        # Fallback dinámico normalizado cuando no hay cuotas de mercado disponibles
        w_sub = W_CLF + W_MATRIX
        w_c = (W_CLF / w_sub) if w_sub > 0 else 0.55
        w_m = (W_MATRIX / w_sub) if w_sub > 0 else 0.45
        p_win = [w_c * c + w_m * m for c, m in zip(p_clf, p_matrix)]
        blend_weights = {'clf': w_c, 'matrix': w_m, 'market': 0.0}

    s_total = sum(p_win)
    p_win = [p / s_total for p in p_win]

    # --- Mercados derivados de la matriz (BTTS, marcadores) ---
    p_btts = float(matrix[1:, 1:].sum())

    # --- Córners ---
    hc = float(modelos['hc'].predict(dmatrix_para(modelos['hc']))[0]) if 'hc' in modelos else l_s['corners']
    ac = float(modelos['ac'].predict(dmatrix_para(modelos['ac']))[0]) if 'ac' in modelos else v_s['corners']

    # --- Tarjetas: modelo real si existe; ajuste aditivo moderado por severidad del árbitro ---
    if 'hy' in modelos and 'ay' in modelos:
        t_base = float(modelos['hy'].predict(dmatrix_para(modelos['hy']))[0]) + \
                 float(modelos['ay'].predict(dmatrix_para(modelos['ay']))[0])
        cards_modelo_real = True
    else:
        t_base = l_s['yc'] + v_s['yc']
        cards_modelo_real = False
    if ref_media is not None:
        t_base += (float(ref_media) - 4.6) * 0.4

    # Marcadores más probables (de la matriz corregida)
    flat = [((i, j), float(matrix[i, j])) for i in range(7) for j in range(7)]
    top_scores = sorted(flat, key=lambda t: -t[1])[:4]

    return {
        "p_win": p_win,
        "p_win_clf": p_clf,
        "p_win_matrix": p_matrix,
        "p_btts": p_btts,
        "top_scorelines": [(f"{i}-{j}", p) for (i, j), p in top_scores],
        "local_goals_proyected": hg,
        "visitor_goals_proyected": ag,
        "hc": hc,
        "ac": ac,
        "total_corners": hc + ac,
        "total_cards": max(t_base, 0.5),
        "cards_modelo_real": cards_modelo_real,
        "market_used": market_used,
        "p_market": p_market,
        "blend_weights": blend_weights,
        "stats_fallback": bool(l_s.get('fallback') or v_s.get('fallback')),
        "n_hist": (l_s.get('n', 0), v_s.get('n', 0)),
    }


def arbitro_designado(nombre):
    """True solo si hay un árbitro real asignado (la API devuelve 'None'/'PENDIENTE' como texto)."""
    if nombre is None:
        return False
    limpio = str(nombre).strip().lower()
    return limpio not in ("", "none", "nan", "null", "por designar", "pendiente", "tbd")


def mejor_apuesta(inf_res, odds_data, local, visita):
    """
    Busca la mejor apuesta recomendable del partido (EV >= 3%, cuota 1.30-8.00,
    EV < 30%) usando la mejor cuota del mercado. Devuelve dict o None.
    """
    if not inf_res or not isinstance(odds_data, dict) or inf_res.get('stats_fallback'):
        return None
    p_l, p_e, p_v = inf_res['p_win']
    dc = get_combined_probs(inf_res['p_win'])
    candidatos = [
        (f"{local} (1)", inf_res and odds_data.get('1x2', {}).get('Home', {}).get('Best'), p_l),
        ("Empate (X)", odds_data.get('1x2', {}).get('Draw', {}).get('Best'), p_e),
        (f"{visita} (2)", odds_data.get('1x2', {}).get('Away', {}).get('Best'), p_v),
        ("1X", odds_data.get('dc', {}).get('Home/Draw', {}).get('Best'), dc.get('1X', 0)),
        ("X2", odds_data.get('dc', {}).get('Draw/Away', {}).get('Best'), dc.get('X2', 0)),
        ("BTTS Sí", odds_data.get('btts', {}).get('Yes', {}).get('Best'), inf_res.get('p_btts', 0)),
        ("BTTS No", odds_data.get('btts', {}).get('No', {}).get('Best'), 1 - inf_res.get('p_btts', 1)),
    ]
    mejor = None
    for etiqueta, cuota, prob in candidatos:
        if not cuota or not (1.30 <= cuota <= 8.00):
            continue
        ev = prob * cuota - 1
        if 0.03 <= ev < 0.30 and (mejor is None or ev > mejor['ev']):
            mejor = {'label': etiqueta, 'cuota': cuota, 'prob': prob, 'ev': ev}
    return mejor


# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================
def main():
    # --- 1. CONFIGURACIÓN DE PÁGINA STREAMLIT (ZERO SIDEBAR / SQUIRCLES) ---
    st.set_page_config(
        page_title="Futbol IA · Smart Money Eigen V10", 
        page_icon="⚽", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Inyección de estilos CSS Vanguardistas (Propuesta 2: Top Command Studio & Cards Modulares)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg: #0B1017;
            --bg-raised: #101724;
            --bg-card: #131E2C;
            --bg-card-hover: #18273A;
            --bg-card-glass: rgba(19, 30, 44, 0.88);
            --border: rgba(255, 255, 255, 0.09);
            --border-strong: rgba(6, 182, 212, 0.45);
            --text: #F8FAFC;
            --text-dim: #94A3B8;
            --accent-cyan: #06B6D4;
            --accent-cyan-soft: rgba(6, 182, 212, 0.12);
            --accent: #06B6D4;
            --accent-soft: rgba(6, 182, 212, 0.12);
            --positive: #10B981;
            --positive-soft: rgba(16, 185, 129, 0.12);
            --negative: #F87171;
            --info: #60A5FA;
            --warn: #FBBF24;
            --radius: 18px;
            --radius-lg: 20px;
            --radius-pill: 9999px;
            --shadow-card: 0 12px 30px -6px rgba(0, 0, 0, 0.6), 0 3px 10px -2px rgba(0, 0, 0, 0.35);
        }

        /* ELIMINACIÓN TOTAL DEL SIDEBAR CLÁSICO */
        [data-testid="stSidebar"], [data-testid="collapsedControl"], header[data-testid="stHeader"] {
            display: none !important;
        }
        #MainMenu, footer, .stDeployButton { display: none !important; }

        .block-container {
            padding: 0.8rem 2rem 4rem 2rem !important;
            max-width: 1480px !important;
            margin: 0 auto !important;
        }

        .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Instrument Sans', -apple-system, 'Segoe UI', sans-serif !important;
        }
        h1, h2, h3, h4, .stMarkdown h3 {
            font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
        }

        /* Top Command Header (Estilo Apple Pro / Vercel Studio) */
        .studio-top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card-glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 12px 22px;
            margin-bottom: 14px;
            box-shadow: var(--shadow-card);
        }

        .studio-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .studio-logo {
            font-size: 22px;
            background: var(--accent-cyan-soft);
            border: 1px solid var(--border-strong);
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .studio-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: -0.02em;
            color: var(--text);
        }

        .studio-sub {
            font-size: 0.82rem;
            color: var(--text-dim);
            font-weight: 500;
            margin-left: 6px;
        }

        .studio-mode-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 5px 14px;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent-cyan);
            letter-spacing: 0.02em;
        }

        .studio-tools {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .studio-search-pill {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 5px 14px;
            font-size: 0.76rem;
            color: var(--text-dim);
        }

        .studio-online-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--positive);
            background: var(--positive-soft);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 10px;
            border-radius: var(--radius-pill);
        }

        .pulse-dot-cyan {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--positive);
            box-shadow: 0 0 8px var(--positive);
            animation: pulse-glow-cyan 2s infinite ease-in-out;
        }

        @keyframes pulse-glow-cyan {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.6; }
        }

        /* Cinta Horizontal de Ligas con Pills */
        div[data-testid="stRadio"] > div {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }
        div[data-testid="stRadio"] label {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 6px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 140ms ease-out;
            color: var(--text-dim);
        }
        div[data-testid="stRadio"] label:hover {
            border-color: var(--accent-cyan);
            color: var(--text);
            background: var(--bg-card-hover);
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            background: var(--accent-cyan-soft);
            border-color: var(--accent-cyan);
            color: var(--text);
            box-shadow: 0 2px 10px rgba(6, 182, 212, 0.25);
        }

        /* Barra de Estado del Cerebro */
        .studio-status-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 6px 16px;
            font-size: 0.78rem;
            color: var(--text-dim);
            margin-bottom: 18px;
        }

        /* Tarjetas de Módulo Studio (Esquinas 18px) */
        .studio-card-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: var(--shadow-card);
            position: relative;
            transition: border-color 160ms ease-out, transform 160ms ease-out;
        }
        .studio-card-box:hover {
            border-color: var(--border-strong);
            transform: translateY(-2px);
        }

        .studio-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }
        .studio-card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-dim);
        }
        .studio-card-more {
            font-size: 14px;
            color: var(--text-dim);
            opacity: 0.7;
        }

        /* Cuotas y métricas con números tabulares */
        [data-testid="stMetricValue"], .metric-value-huge, .odds-num {
            font-variant-numeric: tabular-nums;
        }

        /* Hero del Partido (Pro Studio) */
        .hero {
            background: linear-gradient(165deg, #132236 0%, #0E1624 80%);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg) !important;
            padding: 28px 32px 22px 32px;
            margin: 8px 0 20px 0;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-card);
        }
        .hero::before {
            content: '';
            position: absolute; inset: 0;
            background: radial-gradient(700px 220px at 50% -80px, rgba(6, 182, 212, 0.12), transparent);
            pointer-events: none;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 24px;
        }
        .hero-team { display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center; }
        .hero-team img { width: 72px; height: 72px; object-fit: contain;
                         filter: drop-shadow(0 6px 14px rgba(0,0,0,0.6)); }
        .hero-team .cond { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
                           color: var(--text-dim); font-weight: 600; }
        .hero-center { text-align: center; display: flex; flex-direction: column; gap: 8px; align-items: center; }
        .hero-vs { font-family: 'Space Grotesk'; font-size: 1.8rem; font-weight: 700; color: var(--text-dim); }
        .hero-score { font-family: 'Space Grotesk'; font-size: 2.8rem; font-weight: 800;
                      font-variant-numeric: tabular-nums; }
        .hero-meta {
            display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
            margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border);
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 4px 12px;
            font-size: 0.78rem;
            color: var(--text);
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .chip.dim { color: var(--text-dim); }
        .chip.ok { border-color: rgba(16,185,129,0.35); color: var(--positive); }
        .chip.warn { border-color: rgba(251,191,36,0.35); color: var(--warn); }

        .prob-bar {
            display: flex;
            height: 30px;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 18px;
            background: var(--bg-raised);
            border: 1px solid var(--border);
        }
        .prob-bar .seg {
            display: flex; align-items: center; justify-content: center;
            font-family: 'Space Grotesk'; font-size: 0.78rem; font-weight: 700;
            color: #04241B; transition: width 300ms ease;
        }
        .prob-bar .seg-l { background: #06B6D4; color: #02252C; }
        .prob-bar .seg-e { background: #94A3B8; color: #0E1624; }
        .prob-bar .seg-v { background: #F43F5E; color: #2C050D; }
        .prob-legend {
            display: flex; justify-content: space-between;
            font-size: 0.74rem; color: var(--text-dim); margin-top: 6px; padding: 0 4px;
        }

        /* Veredicto Pro Studio */
        .verdict {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            margin-bottom: 20px;
            display: flex; flex-wrap: wrap; gap: 20px; align-items: center;
            box-shadow: var(--shadow-card);
        }
        .verdict .v-block { min-width: 150px; }
        .verdict .v-label { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
                            color: var(--text-dim); font-weight: 600; margin-bottom: 4px; }
        .verdict .v-value { font-family: 'Space Grotesk'; font-size: 1.35rem; font-weight: 700;
                            font-variant-numeric: tabular-nums; }
        .verdict .v-sub { font-size: 0.78rem; color: var(--text-dim); }

        /* Stat tiles de proyección (Squircles 18px) */
        .tile {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px 20px;
            position: relative;
            transition: transform 160ms ease-out, border-color 160ms ease-out, box-shadow 160ms ease-out;
            height: 100%;
            box-shadow: var(--shadow-card);
        }
        .tile:hover { 
            transform: translateY(-2px); 
            border-color: var(--border-strong); 
            box-shadow: 0 16px 36px -6px rgba(0,0,0,0.6);
        }
        .tile .t-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
                         font-weight: 600; margin-bottom: 6px; }
        .tile .t-value { font-family: 'Space Grotesk'; font-size: 2.1rem; font-weight: 700;
                         line-height: 1.05; font-variant-numeric: tabular-nums; }
        .tile .t-sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 6px; }
        .tile .t-track { height: 5px; background: var(--bg-raised); border-radius: 3px; margin-top: 14px; overflow: hidden; }
        .tile .t-fill { height: 100%; border-radius: 3px; }

        /* Chip flotante de veredicto */
        .float-pick {
            position: fixed; right: 24px; bottom: 24px; z-index: 9999;
            background: rgba(16, 24, 38, 0.94);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-strong);
            border-radius: var(--radius-pill);
            padding: 12px 20px;
            font-size: 0.84rem;
            box-shadow: 0 12px 32px rgba(0,0,0,0.6);
            max-width: 360px;
            font-variant-numeric: tabular-nums;
        }
        .float-pick .fp-title { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
                                color: var(--text-dim); font-weight: 700; margin-bottom: 4px; }
        @media (max-width: 900px) { .float-pick { display: none; } }

        /* Botones: feedback táctil elástico */
        .stButton > button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: transform 120ms ease-out, opacity 120ms ease-out !important;
        }
        .stButton > button:active { transform: scale(0.96) !important; }

        /* Pestañas estilo Studio */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(16, 24, 36, 0.9);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 5px;
            width: fit-content;
            max-width: 100%;
            overflow-x: auto;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            color: var(--text-dim);
            border-radius: 8px !important;
            padding: 8px 18px !important;
            transition: color 150ms ease-out, background 150ms ease-out;
        }
        .stTabs [data-baseweb="tab"]:hover { 
            color: var(--text) !important; 
            background: rgba(255, 255, 255, 0.05) !important; 
        }
        .stTabs [aria-selected="true"] {
            color: #02252C !important;
            background: var(--accent-cyan) !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 10px rgba(6, 182, 212, 0.35);
        }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

        /* Inputs y Selectboxes suavizados */
        div[data-testid="stSelectbox"] > div {
            border-radius: 12px !important;
            border-color: var(--border) !important;
        }
        div[data-testid="stNumberInput"] input {
            border-radius: 12px !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background: var(--bg-card);
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
        }

        hr { border-color: var(--border) !important; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
    </style>
    """, unsafe_allow_html=True)

    # --- 2. TOP COMMAND HEADER (PROPUESTA 2: STUDIO & CARDS MODULARES) ---
    st.markdown("""
    <div class='studio-top-bar'>
        <div class='studio-brand'>
            <div class='studio-logo'>⚽</div>
            <div>
                <span class='studio-title'>FUTBOL IA</span>
                <span class='studio-sub'>- Deep Soccer Pro Studio</span>
            </div>
        </div>
        <div class='studio-tools'>
            <div class='studio-mode-tag'>Live Analysis ▾</div>
            <div class='studio-search-pill'>🔍 Search matches, teams, stats... (⌘K)</div>
            <div class='studio-online-badge'>
                <span class='pulse-dot-cyan'></span> PRO STUDIO ONLINE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controles de Liga y Gestión en Toolbar Compacto
    col_tools_liga, col_tools_bank, col_tools_ia = st.columns([4.2, 2.8, 3.0])
    with col_tools_liga:
        liga_seleccionada = st.selectbox("🏆 Torneo Activo:", list(CONFIG_LIGAS.keys()), index=0)
        conf = CONFIG_LIGAS[liga_seleccionada]
    with col_tools_bank:
        bankroll = st.number_input("⛃ Bankroll Studio ($):", value=100000, step=1000, min_value=1000)
    with col_tools_ia:
        modelos_ollama = listar_modelos_ollama()
        if modelos_ollama:
            defecto = seleccionar_modelo_ollama()
            idx = modelos_ollama.index(defecto) if defecto in modelos_ollama else 0
            st.session_state['ollama_model_choice'] = st.selectbox(
                "🤖 Motor Analítico Local:", modelos_ollama, index=idx,
                help="Modelos locales detectados (ej. Qwen 3.8 Heretic Uncensored 9B, Qwen 3.5 Abliterated 9B).")
        else:
            st.selectbox("🤖 Motor IA:", ["Groq Cloud / Fallback"], disabled=True)

    if 'odds_history' not in st.session_state:
        st.session_state.odds_history = load_odds_history()

    # --- 3. INGESTIÓN Y MODELOS ---
    df_grilla, df_comp, features_cols = cargar_master_data_v30(conf)
    modelos, meta_modelos = cargar_modelos_v30(liga_seleccionada)

    # Tira de Estado del Cerebro Activo (Pro Studio)
    cfg_curr = get_league_config_by_name(liga_seleccionada)
    s_code = f"_{cfg_curr['suffix']}" if cfg_curr.get('suffix') else ""
    brain_title = f"🧠 <b>Modelo Predictivo Activo:</b> <code>model_1x2_v5{s_code}.json</code>"
    brain_stats = ""
    if meta_modelos:
        m1x2 = meta_modelos.get('models', {}).get('1x2', {})
        n_samp = meta_modelos.get('n_samples', '?')
        if m1x2.get('val_accuracy') is not None:
            brain_stats = f" · <b>Acc Val:</b> {m1x2['val_accuracy']*100:.0f}% · <b>RPS:</b> {m1x2.get('val_rps', 0):.3f} · {n_samp} partidos en dataset"
    st.markdown(f"""
    <div class='studio-status-strip'>
        <span>{brain_title}{brain_stats}</span>
        <span style='color: var(--accent-cyan); font-weight: 600;'>Módulo: Operativo</span>
    </div>
    """, unsafe_allow_html=True)

    if df_grilla.empty:
        st.error(f"⚠️ No se detectaron partidos para {liga_seleccionada} en la ventana actual.")
        key = get_api_sports_key()
        key_masked = f"{key[:4]}...{key[-4:]}" if key and len(key)>8 else str(key)
        st.info(f"**🛠️ DIAGNÓSTICO EN SERVIDOR:**\n"
                f"- API-Sports Key leída: `{key_masked}`\n"
                f"- Hora Servidor UTC: `{datetime.now(timezone.utc)}`")
        st.stop()

    # Selector de Partidos robusto
    selector_col = 'Partido_String' if 'Partido_String' in df_grilla.columns else ('Match' if 'Match' in df_grilla.columns else df_grilla.columns[0])
    partido_sel = st.selectbox(
        "Selecciona un partido para análisis profundo:", 
        df_grilla[selector_col].tolist(),
        index=0
    )

    # Extracción de fila seleccionada
    match_data = df_grilla[df_grilla[selector_col] == partido_sel].iloc[0]
    local = str(match_data.get('Local') or match_data.get('Home_Team') or 'Local')
    visita = str(match_data.get('Visita') or match_data.get('Away_Team') or 'Visita')
    fecha_evento = str(match_data.get('Fecha', '--/--'))
    arbitro = str(match_data.get('Arbitro', 'Por Designar'))
    estadio = str(match_data.get('Estadio', 'Estadio Principal'))
    ciudad = str(match_data.get('Ciudad', conf.get('clima_default', 'Santiago, Chile')))
    fixture_id = match_data.get('id') or match_data.get('Fixture_ID') or 0
    home_id = match_data.get('Local_ID') or match_data.get('Home_ID') or 0
    away_id = match_data.get('Visita_ID') or match_data.get('Away_ID') or 0
    status = str(match_data.get('Status', 'upcoming')).lower()
    marcador = str(match_data.get('Marcador', '0 - 0'))
    minuto = match_data.get('minuto', 0)
    local_logo = match_data.get('Local_Logo', '')
    visita_logo = match_data.get('Visita_Logo', '')

    # Extracción de cuotas y alineaciones
    odds_data = get_odds(fixture_id, conf['id_api'])
    pipeline_copa = CopaDataPipeline()

    ideal_home, pdict_home = pipeline_copa.get_ideal_xi_minutes(home_id, league_id=conf['id_api']) if home_id else (0, {})
    ideal_away, pdict_away = pipeline_copa.get_ideal_xi_minutes(away_id, league_id=conf['id_api']) if away_id else (0, {})

    curr_home = pipeline_copa.get_current_xi_minutes(fixture_id, home_id, pdict_home) if (fixture_id and home_id) else -1
    curr_away = pipeline_copa.get_current_xi_minutes(fixture_id, away_id, pdict_away) if (fixture_id and away_id) else -1

    lsi_h = (curr_home / ideal_home) if (ideal_home > 0 and curr_home > 0) else 1.0
    lsi_v = (curr_away / ideal_away) if (ideal_away > 0 and curr_away > 0) else 1.0

    # Inferencia principal
    try:
        ref_meta_inf = get_referee_metrics(arbitro)
        inf_res = run_master_inference(
            local, visita, conf,
            is_cup=0, lsi_local=lsi_h, lsi_visita=lsi_v,
            modelos_dict=modelos, metadata=meta_modelos,
            ref_media=ref_meta_inf['media'] if ref_meta_inf['real'] else None,
            odds_data=odds_data
        )
        if inf_res.get('stats_fallback'):
            st.warning("⚠️ Uno de los equipos no tiene historial en el dataset: la proyección usa "
                       "medias de liga y es menos confiable. Evita apostar este partido.")
    except Exception as e:
        st.error(f"Falla en la inferencia matemática: {e}")
        inf_res = None

    # --- HERO DEL PARTIDO (banner + contexto integrado, sin datos placebo) ---
    bajas_l = get_absence_impact(fixture_id, home_id) if (fixture_id and home_id) else []
    bajas_v = get_absence_impact(fixture_id, away_id) if (fixture_id and away_id) else []
    hora_evento = str(match_data.get('Hora_Chile', '') or '')
    jornada_evento = str(match_data.get('Jornada', '') or '')

    badge_cls = "badge-live" if status == "live" else "badge-upcoming"
    badge_txt = "● EN VIVO" if status == "live" else "PROGRAMADO"
    centro_html = f"<div class='hero-score'>{marcador}</div>" if status != "upcoming" \
        else "<div class='hero-vs'>VS</div>"
    logo_l = f"<img src='{local_logo}' alt=''>" if local_logo else ""
    logo_v = f"<img src='{visita_logo}' alt=''>" if visita_logo else ""

    # Chips de contexto: solo información REAL
    chips = [f"<span class='chip'>🏟️ <b>{estadio}</b> · {ciudad}</span>"]
    if jornada_evento and jornada_evento != 'N/A':
        chips.append(f"<span class='chip dim'>{jornada_evento}</span>")
    if arbitro_designado(arbitro):
        ref_meta = get_referee_metrics(arbitro)
        ref_extra = f" · {ref_meta['media']:.1f} 🟨/partido" if ref_meta['real'] else ""
        chips.append(f"<span class='chip'>⚖️ <b>{arbitro}</b>{ref_extra}</span>")
    else:
        chips.append("<span class='chip dim'>⚖️ Árbitro por designar</span>")
    if not bajas_l and not bajas_v:
        chips.append("<span class='chip ok'>✓ Sin bajas reportadas</span>")
    else:
        for b in bajas_l[:2]:
            marca = "🚨 " if b.get('key') else ""
            chips.append(f"<span class='chip warn'>{marca}{b['name']} · baja {local}</span>")
        for b in bajas_v[:2]:
            marca = "🚨 " if b.get('key') else ""
            chips.append(f"<span class='chip warn'>{marca}{b['name']} · baja {visita}</span>")

    # Barra de probabilidad 1X2 dentro del hero (HTML en una línea: Markdown
    # convierte las líneas indentadas en bloque de código)
    prob_bar_html = ""
    if inf_res:
        pl, pe, pv = [max(p * 100, 3) for p in inf_res['p_win']]  # mínimo 3% de ancho para legibilidad
        rl, re_, rv = [p * 100 for p in inf_res['p_win']]
        prob_bar_html = (
            f"<div class='prob-bar'>"
            f"<div class='seg seg-l' style='width:{pl:.1f}%'><span>1 · {rl:.0f}%</span></div>"
            f"<div class='seg seg-e' style='width:{pe:.1f}%'><span>X · {re_:.0f}%</span></div>"
            f"<div class='seg seg-v' style='width:{pv:.1f}%'><span>2 · {rv:.0f}%</span></div>"
            f"</div>"
            f"<div class='prob-legend'><span>{local}</span><span>Empate</span><span>{visita}</span></div>")

    st.markdown(f"""
    <div class='hero'>
        <div class='hero-grid'>
            <div class='hero-team'>{logo_l}<div class='team-name'>{local}</div><div class='cond'>Local</div></div>
            <div class='hero-center'>
                <span class='{badge_cls}'>{badge_txt}</span>
                {centro_html}
                <div style='font-size:0.85rem; color:var(--text-dim);'>{fecha_evento} {('· ' + hora_evento + ' h') if hora_evento else ''}</div>
            </div>
            <div class='hero-team'>{logo_v}<div class='team-name'>{visita}</div><div class='cond'>Visitante</div></div>
        </div>
        {prob_bar_html}
        <div class='hero-meta'>{''.join(chips)}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- VEREDICTO DEL MODELO + CHIP FLOTANTE ---
    if inf_res:
        p = inf_res['p_win']
        picks = [(f"{local}", p[0]), ("Empate", p[1]), (f"{visita}", p[2])]
        pick_txt, pick_p = max(picks, key=lambda t: t[1])
        scoreline_chips = " ".join(
            f"<span class='chip'>{s} · {pr*100:.0f}%</span>" for s, pr in inf_res.get('top_scorelines', []))
        top_bet = mejor_apuesta(inf_res, odds_data, local, visita)
        if top_bet:
            bet_html = (f"<div class='v-value' style='color: var(--positive);'>{top_bet['label']}</div>"
                        f"<div class='v-sub'>cuota {top_bet['cuota']:.2f} · EV +{top_bet['ev']*100:.1f}%</div>")
        else:
            bet_html = ("<div class='v-value' style='color: var(--text-dim); font-size: 1rem;'>Sin valor claro</div>"
                        "<div class='v-sub'>ningún mercado supera el umbral de EV</div>")

        st.markdown(f"""
        <div class='verdict'>
            <div class='v-block'>
                <div class='v-label'>Escenario más probable</div>
                <div class='v-value'>{pick_txt}</div>
                <div class='v-sub'>{pick_p*100:.1f}% de probabilidad calibrada</div>
            </div>
            <div class='v-block'>
                <div class='v-label'>Goles esperados</div>
                <div class='v-value'>{inf_res['local_goals_proyected']:.2f} — {inf_res['visitor_goals_proyected']:.2f}</div>
                <div class='v-sub'>BTTS {inf_res.get('p_btts', 0)*100:.0f}% · {inf_res['total_corners']:.1f} córners · {inf_res['total_cards']:.1f} 🟨</div>
            </div>
            <div class='v-block'>
                <div class='v-label'>Mejor valor detectado</div>
                {bet_html}
            </div>
            <div class='v-block' style='flex: 1; min-width: 220px;'>
                <div class='v-label'>Marcadores más probables</div>
                <div style='display:flex; gap:6px; flex-wrap:wrap; margin-top: 4px;'>{scoreline_chips}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chip flotante visible desde cualquier tab
        fp_bet = (f"💎 {top_bet['label']} @ {top_bet['cuota']:.2f} (EV +{top_bet['ev']*100:.1f}%)"
                  if top_bet else "Sin valor claro en el mercado")
        st.markdown(f"""
        <div class='float-pick'>
            <div class='fp-title'>Veredicto rápido</div>
            <div>🎯 <b>{pick_txt}</b> · {pick_p*100:.0f}%</div>
            <div style='color: var(--text-dim); margin-top: 2px;'>{fp_bet}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs de navegación (Pro Studio)
    tab_pre, tab_dossier, tab_smart, tab_ai, tab_chat, tab_live, tab_copa = st.tabs([
        "⚡ Match Studio",
        "📁 Dossier 360°",
        "🎯 Value Detector (EV+)",
        "🤖 Reporte Cuantitativo",
        "💬 Chat Analista IA",
        "📡 Live Radar & xG",
        "🏆 Radar LSI Multi-Torneo"
    ])

    # --- TAB 1: PROYECCIONES ---
    with tab_pre:
        if inf_res:
            p_l, p_e, p_v = inf_res['p_win']
            hg = inf_res['local_goals_proyected']
            ag = inf_res['visitor_goals_proyected']
            corners_tot = inf_res['total_corners']
            cards_tot = inf_res['total_cards']

            def tile(label, color, valor, sub, pct=None):
                track = ""
                if pct is not None:
                    track = (f"<div class='t-track'><div class='t-fill' "
                             f"style='width:{pct*100:.0f}%; background:{color};'></div></div>")
                return (f"<div class='tile'><div class='t-label' style='color:{color};'>{label}</div>"
                        f"<div class='t-value'>{valor}</div><div class='t-sub'>{sub}</div>{track}</div>")

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(tile(f"Victoria {local} (1)", "#7DD3FC", f"{p_l*100:.1f}%",
                             f"Goles esperados: <b>{hg:.2f}</b>", p_l), unsafe_allow_html=True)
            c2.markdown(tile("Empate (X)", "#CBD5E1", f"{p_e*100:.1f}%",
                             f"Cuota justa: <b>{1/max(p_e,1e-4):.2f}</b>", p_e), unsafe_allow_html=True)
            c3.markdown(tile(f"Victoria {visita} (2)", "#FDA4AF", f"{p_v*100:.1f}%",
                             f"Goles esperados: <b>{ag:.2f}</b>", p_v), unsafe_allow_html=True)
            c4.markdown(tile("Totales proyectados", "#2DD4A7", f"{hg+ag:.2f} goles",
                             f"⛳ {corners_tot:.1f} córners · 🟨 {cards_tot:.1f} tarjetas"), unsafe_allow_html=True)

            # Desglose de los dos motores que componen la probabilidad final
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### ⚙️ Composición del pronóstico (55% clasificador calibrado · 45% matriz Dixon-Coles)")
            p_clf = inf_res.get('p_win_clf', inf_res['p_win'])
            p_mat = inf_res.get('p_win_matrix', inf_res['p_win'])

            def mini_bar(nombre, probs):
                l, e, v = [max(x * 100, 3) for x in probs]
                rl, re_, rv = [x * 100 for x in probs]
                return (
                    f"<div style='margin-bottom: 10px;'>"
                    f"<div style='font-size:0.75rem; color:var(--text-dim); margin-bottom:4px;'>{nombre}</div>"
                    f"<div class='prob-bar' style='height: 26px; margin: 0;'>"
                    f"<div class='seg seg-l' style='width:{l:.1f}%'><span>{rl:.0f}%</span></div>"
                    f"<div class='seg seg-e' style='width:{e:.1f}%'><span>{re_:.0f}%</span></div>"
                    f"<div class='seg seg-v' style='width:{v:.1f}%'><span>{rv:.0f}%</span></div>"
                    f"</div></div>")

            st.markdown(
                mini_bar("Clasificador XGBoost (calibrado por temperatura)", p_clf)
                + mini_bar("Matriz de marcadores Poisson + corrección Dixon-Coles", p_mat)
                + mini_bar("→ Pronóstico final (blend)", inf_res['p_win']),
                unsafe_allow_html=True)

            n_l, n_v = inf_res.get('n_hist', (0, 0))
            st.caption(f"Base: últimos {n_l} partidos de {local} como local · últimos {n_v} de {visita} como visita. "
                       f"Tarjetas {'con modelo dedicado' if inf_res.get('cards_modelo_real') else 'sin modelo dedicado (promedio móvil)'}.")

    # --- TAB 2: DOSSIER 360° ---
    dossier_data = obtener_dossier_360(
        fixture_id=fixture_id,
        local_id=home_id,
        visita_id=away_id,
        league_id=conf['id_api'],
        season=conf['season'],
        raw_csv=conf['dataset_raw'],
        local_name=local,
        visita_name=visita
    )

    with tab_dossier:
        st.markdown("### 📁 Dossier de Inteligencia Táctica y Contexto 360°")
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.markdown(f"#### 🔵 {local}")
            st.write(f"- **Posición en Tabla:** {dossier_data.get('posicion_local', 'N/A')}")
            st.write(f"- **Récord en Casa:** `{dossier_data.get('record_condicion_local', 'N/A')}`")
            st.write(f"- **Micro-Inercia:** `{dossier_data.get('micro_inercia_local', 'N/A')}`")
            st.write(f"- **Últimos Partidos:** {dossier_data.get('ultimos_partidos_local', dossier_data.get('forma_local', 'N/A'))}")
            st.write(f"- **Balance Anual:** {dossier_data.get('pg_local',0)}V - {dossier_data.get('pe_local',0)}E - {dossier_data.get('pp_local',0)}D ({dossier_data.get('gf_local',0)} GF / {dossier_data.get('gc_local',0)} GC)")
            st.write(f"- **Bajas/Dudas:** {dossier_data.get('bajas_local', 'Sin bajas reportadas')}")
            
        with d_col2:
            st.markdown(f"#### 🔴 {visita}")
            st.write(f"- **Posición en Tabla:** {dossier_data.get('posicion_visita', 'N/A')}")
            st.write(f"- **Récord de Visita:** `{dossier_data.get('record_condicion_visita', 'N/A')}`")
            st.write(f"- **Micro-Inercia:** `{dossier_data.get('micro_inercia_visita', 'N/A')}`")
            st.write(f"- **Últimos Partidos:** {dossier_data.get('ultimos_partidos_visita', dossier_data.get('forma_visita', 'N/A'))}")
            st.write(f"- **Balance Anual:** {dossier_data.get('pg_visita',0)}V - {dossier_data.get('pe_visita',0)}E - {dossier_data.get('pp_visita',0)}D ({dossier_data.get('gf_visita',0)} GF / {dossier_data.get('gc_visita',0)} GC)")
            st.write(f"- **Bajas/Dudas:** {dossier_data.get('bajas_visita', 'Sin bajas reportadas')}")

        st.markdown("---")
        st.markdown(f"**Historial Directo (H2H):** {dossier_data.get('resumen_h2h', 'Sin historial reciente')}")
        if arbitro_designado(arbitro):
            ref_d = get_referee_metrics(arbitro)
            severidad = f" (severidad histórica: `{ref_d['media']:.1f}` amarillas/partido)" if ref_d['real'] else " (sin historial de severidad en la base)"
            st.markdown(f"**Árbitro Asignado:** {arbitro}{severidad}")
        else:
            st.markdown("**Árbitro:** por designar")

    # --- TAB 3: SMART MONEY & EV ARBITRAGE ---
    with tab_smart:
        st.markdown("### 💎 Detección de Valor Esperado (+EV) y Gestión Kelly")
        st.caption("Las probabilidades de las casas se muestran **sin margen** (desvigorizadas). "
                   f"Solo se recomienda una apuesta con EV ≥ {3}% y cuota entre 1.30 y 8.00.")

        # Umbrales de la política de apuestas
        EV_MIN = 0.03          # edge mínimo para recomendar
        EV_SOSPECHOSO = 0.30   # por encima de esto, probablemente hay error de datos
        ODDS_MIN, ODDS_MAX = 1.30, 8.00
        STAKE_CAP = 0.025 * bankroll  # nunca más del 2.5% del bankroll en una apuesta

        BOOKIES = ["Betano", "Bet365", "Pinnacle", "Best"]

        def render_market_block(rows, market_odds, market_id, target_sum=1.0):
            """
            rows: [(label, outcome_key, prob_modelo)]
            market_odds: {outcome_key: {bookie: cuota}}
            Desvigoriza por casa usando el mercado completo y calcula EV/stake.
            target_sum=2.0 para Double Chance (cada resultado cubre 2 de 3 salidas).
            """
            h = st.columns([2, 1.2, 1.4, 1.4, 1.4, 1.4])
            h[0].caption("Mercado"); h[1].caption("IA Eigen")
            for i, bk in enumerate(BOOKIES):
                h[i+2].caption("⭐ Mejor" if bk == "Best" else bk)
            st.divider()

            # Probabilidades justas por casa (mercado completo, sin margen)
            fair_por_casa = {}
            for bk in BOOKIES:
                odds_bk = {k: v.get(bk) for k, v in market_odds.items() if v.get(bk)}
                fair_por_casa[bk] = devig_proportional(odds_bk, target_sum=target_sum)

            for label, key, prob_ia in rows:
                cols = st.columns([2, 1.2, 1.4, 1.4, 1.4, 1.4])
                cols[0].write(label)
                cuota_justa = 1 / (prob_ia + 1e-4)
                cols[1].write(f"**{cuota_justa:.2f}** ({prob_ia*100:.0f}%)")
                odds_dict = market_odds.get(key, {})

                for i, bk in enumerate(BOOKIES):
                    cuota_bk = odds_dict.get(bk, 0)
                    if not cuota_bk or cuota_bk <= 1.0:
                        cols[i+2].caption("-")
                        continue

                    hist_key = f"{fixture_id}_{market_id}_{label}_{bk}"
                    prev_odd = st.session_state.odds_history.get(hist_key)
                    trend_icon = ""
                    if prev_odd and abs(prev_odd - cuota_bk) > 1e-9:
                        trend_icon = "📉" if cuota_bk < prev_odd else "📈"
                    st.session_state.odds_history[hist_key] = cuota_bk

                    fair_prob = fair_por_casa.get(bk, {}).get(key)
                    fair_txt = f"{fair_prob*100:.0f}%" if fair_prob else f"{(1/cuota_bk)*100:.0f}%*"
                    ev = (prob_ia * cuota_bk) - 1
                    ev_pct = ev * 100
                    display_text = f"{cuota_bk:.2f} {trend_icon} ({fair_txt})"

                    recomendable = (ev >= EV_MIN and ODDS_MIN <= cuota_bk <= ODDS_MAX
                                    and ev < EV_SOSPECHOSO
                                    and not (inf_res and inf_res.get('stats_fallback')))
                    if ev >= EV_SOSPECHOSO:
                        cols[i+2].markdown(
                            f"{display_text}<br><small style='color:#F59E0B;'>⚠️ EV {ev_pct:+.0f}% — "
                            f"verificar datos</small>", unsafe_allow_html=True)
                    elif recomendable:
                        stake = min(calculate_kelly(prob_ia, cuota_bk, bankroll, fraction=4), STAKE_CAP)
                        cols[i+2].markdown(f"""
                        <div style='color: var(--positive); font-weight: 600;'>
                            {display_text}<br>
                            ▲ +{ev_pct:.1f}% EV<br>
                            <span style='font-size: 0.8rem; color: var(--text-dim);'>Stake: ${stake:,.0f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        cols[i+2].markdown(
                            f"{display_text}<br><small style='color: var(--text-dim);'>{ev_pct:+.1f}%</small>",
                            unsafe_allow_html=True)

        # 1. Ganador del Partido 1X2
        st.markdown("#### 🎯 Comparador 1X2 (Match Winner)")
        odds_1x2 = odds_data.get('1x2', {}) if isinstance(odds_data, dict) else {}
        if odds_1x2 and inf_res:
            prob_l, prob_e, prob_v = inf_res['p_win']
            render_market_block(
                [(local, 'Home', prob_l), ("Empate", 'Draw', prob_e), (visita, 'Away', prob_v)],
                odds_1x2, market_id="1x2")
        else:
            st.info("Cuotas 1X2 en proceso de captura.")

        # 2. Doble Oportunidad
        st.markdown("---")
        st.markdown("#### 🛡️ Gestión de Riesgo (Double Chance)")
        odds_dc = odds_data.get('dc', {}) if isinstance(odds_data, dict) else {}
        if inf_res and odds_dc:
            probs_dc = get_combined_probs(inf_res['p_win'])
            render_market_block(
                [("Local o Empate (1X)", "Home/Draw", probs_dc.get("1X", 0.5)),
                 ("Empate o Visita (X2)", "Draw/Away", probs_dc.get("X2", 0.5)),
                 ("Local o Visita (12)", "Home/Away", probs_dc.get("12", 0.5))],
                odds_dc, market_id="dc", target_sum=2.0)
        elif inf_res:
            st.caption("Sin cuotas de Doble Oportunidad disponibles.")

        # 3. Ambos Anotan (BTTS) — derivado de la matriz Dixon-Coles
        st.markdown("---")
        st.markdown("#### 🥅 Ambos Equipos Anotan (BTTS)")
        odds_btts = odds_data.get('btts', {}) if isinstance(odds_data, dict) else {}
        if inf_res and odds_btts:
            p_btts = inf_res.get('p_btts', 0.5)
            render_market_block(
                [("Sí (ambos anotan)", "Yes", p_btts), ("No", "No", 1.0 - p_btts)],
                odds_btts, market_id="btts")
        elif inf_res:
            st.caption(f"Proyección BTTS del modelo: {inf_res.get('p_btts', 0.5)*100:.0f}% (sin cuotas activas).")

        # 4. Mercados Over/Under (Goles, Córners, Tarjetas)
        st.markdown("---")
        st.markdown("#### ⚽ Mercados de Totales (Over / Under)")
        if inf_res:
            aviso_cards = "" if inf_res.get('cards_modelo_real') else " · ⚠️ proyección sin modelo dedicado"
            mercados = [
                ('ou_goals', "⚽ Goles Totales", float(inf_res['local_goals_proyected']) + float(inf_res['visitor_goals_proyected']), ""),
                ('ou_corners', "🚩 Córners Totales", float(inf_res['total_corners']), ""),
                ('ou_cards', "🟨 Tarjetas Totales", float(inf_res['total_cards']), aviso_cards)
            ]
            for cat_key, cat_name, lambda_val, aviso in mercados:
                with st.expander(f"{cat_name} (Proyección: {lambda_val:.2f}{aviso})", expanded=False):
                    ou_cat = odds_data.get(cat_key, {}) if isinstance(odds_data, dict) else {}
                    if ou_cat:
                        for line_val, line_data in ou_cat.items():
                            p_over = poisson_over(lambda_val, float(line_val))
                            render_market_block(
                                [(f"Over {line_val}", 'over', p_over), (f"Under {line_val}", 'under', 1.0 - p_over)],
                                {'over': line_data.get('over', {}), 'under': line_data.get('under', {})},
                                market_id=f"{cat_key}_{line_val}")
                    else:
                        st.caption(f"Sin cotizaciones activas de casas de apuestas para {cat_name}.")

    # --- TAB 4: REPORTE IA TÁCTICO ---
    with tab_ai:
        st.markdown("### 🤖 Diagnóstico Táctico-Cuantitativo con IA")
        st.caption("🌐 Inteligencia Web en Vivo Conectada · Modelo Sin Censura · Detección de Cuotas Trampa & Asimetrías")
        if st.button("🚀 Generar Informe de Inteligencia Táctica", type="primary"):
            with st.spinner("Analizando matrices cruzadas, noticias web y redactando informe..."):
                p_l, p_e, p_v = inf_res['p_win'] if inf_res else (0.33, 0.33, 0.33)
                corners_tot = inf_res['total_corners'] if inf_res else 9.5
                cards_tot = inf_res['total_cards'] if inf_res else 4.5
                
                informe_texto = invocar_agente_v8(
                    local=local,
                    visita=visita,
                    prob_L=p_l,
                    prob_V=p_v,
                    corners=corners_tot,
                    tarjetas=cards_tot,
                    arbitro=arbitro,
                    dossier=dossier_data
                )
                st.session_state.ultimo_informe = informe_texto

        if 'ultimo_informe' in st.session_state:
            st.markdown(f"<div class='report-box'>\n\n{st.session_state.ultimo_informe}\n\n</div>", unsafe_allow_html=True)

    # --- TAB 5: CHAT ANALISTA (LLM CONVERSACIONAL) ---
    with tab_chat:
        st.markdown("### 💬 Chat con el Analista Cuantitativo")
        motor_activo = seleccionar_modelo_ollama()
        st.caption(f"Motor: {'🟢 Ollama local · ' + motor_activo if motor_activo else '🟡 Groq Cloud (Ollama no detectado)'}"
                   f" · Contexto: {local} vs {visita} · 🌐 Live Web Intel: Conectada")

        # El historial de chat se reinicia al cambiar de partido
        chat_key = f"chat_{fixture_id}"
        if st.session_state.get('chat_partido_activo') != chat_key:
            st.session_state.chat_partido_activo = chat_key
            st.session_state.chat_historial = []

        if not st.session_state.get('chat_historial'):
            st.info("Pregunta lo que quieras sobre este partido: lectura del momento de forma, "
                    "por qué el modelo favorece a un equipo, qué mercados tienen valor, "
                    "o pide un escenario (ej: *'¿y si llueve y el local rota 3 titulares?'*).")

        for msg in st.session_state.get('chat_historial', []):
            with st.chat_message(msg['role'], avatar="⚽" if msg['role'] == 'assistant' else None):
                st.markdown(msg['content'], unsafe_allow_html=True)

        pregunta = st.chat_input("Pregúntale al analista sobre este partido...")
        if pregunta:
            st.session_state.chat_historial.append({'role': 'user', 'content': pregunta})
            with st.chat_message('user'):
                st.markdown(pregunta)

            system_prompt = construir_contexto_chat(local, visita, inf_res, dossier_data,
                                                    odds_data, arbitro, liga_seleccionada)
            mensajes = [{'role': 'system', 'content': system_prompt}]
            mensajes += st.session_state.chat_historial[-10:]  # ventana de memoria

            with st.chat_message('assistant', avatar="⚽"):
                with st.spinner("Analizando..."):
                    respuesta, motor = chat_llm(mensajes)
                if respuesta:
                    st.markdown(respuesta, unsafe_allow_html=True)
                    st.caption(f"— {motor}")
                    st.session_state.chat_historial.append({'role': 'assistant', 'content': respuesta})
                else:
                    st.error("No hay motor LLM disponible: inicia Ollama (`ollama serve`) "
                             "o configura GROQ_API_KEY en el .env.")

        if st.session_state.get('chat_historial'):
            if st.button("🗑️ Limpiar conversación"):
                st.session_state.chat_historial = []
                st.rerun()

    # --- TAB 6: LIVE STEAM RADAR ---
    with tab_live:
        st.markdown("### 📡 Live Steam In-Play Radar (Telemetría de Mercado)")
        if status != "live":
            st.markdown(f"""
            <div class='pro-card' style='text-align: center; border-style: dashed;'>
                <h3 style='color: #94A3B8;'>🛰️ Radar In-Play en Espera</h3>
                <p>El monitor de trading se activará automáticamente cuando el partido <b>{local} vs {visita}</b> comience.</p>
                <small>Estado: {status.upper()} | Hora Programada: {fecha_evento}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: #1E293B; border-left: 5px solid #10B981; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h4 style='margin:0; color: #94A3B8;'>MARCADOR EN VIVO</h4>
                        <h1 style='margin:0; font-size: 2.5rem;'>{marcador}</h1>
                    </div>
                    <div style='text-align: center;'>
                        <span class='badge-live' style='font-size: 1rem; padding: 6px 14px;'>{minuto}'</span>
                    </div>
                    <div style='text-align: right;'>
                        <h4 style='margin:0; color: #94A3B8;'>ESTADO</h4>
                        <h2 style='margin:0; color: #10B981;'>EN JUEGO</h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Escanear Cuotas In-Play", type="primary"):
                st.session_state.last_live_scan = get_live_odds(fixture_id)
                st.toast("Telemetría en vivo actualizada.", icon="✅")

            if 'last_live_scan' in st.session_state and st.session_state.last_live_scan:
                l_odds = st.session_state.last_live_scan
                if 'Match Winner' in l_odds:
                    st.markdown("#### ⚖️ Oportunidades 1X2 en Vivo (condicionadas al marcador)")
                    win_odds = l_odds['Match Winner']
                    m1, m2, m3 = st.columns(3)

                    # Probabilidades vivas: Poisson del tiempo restante + marcador actual
                    try:
                        sc_h, sc_a = [int(x.strip()) for x in str(marcador).split('-')]
                    except Exception:
                        sc_h, sc_a = 0, 0
                    hg_pre = inf_res['local_goals_proyected'] if inf_res else 1.3
                    ag_pre = inf_res['visitor_goals_proyected'] if inf_res else 1.1
                    p_live = live_probabilities(hg_pre, ag_pre, sc_h, sc_a, minuto)
                    labels = ['Home', 'Draw', 'Away']

                    for i, (lbl, col) in enumerate(zip(labels, [m1, m2, m3])):
                        c_now = win_odds.get(lbl)
                        if c_now:
                            p_adj = p_live[i]
                            ev = (p_adj * c_now) - 1
                            with col:
                                cls_ev = "ev-good" if ev > 0.03 else "ev-bad"
                                st.markdown(f"""
                                <div class='pro-card' style='text-align: center;'>
                                    <small>{lbl.upper()}</small>
                                    <div class='metric-value-huge'>{c_now:.2f}</div>
                                    <p>P(live): <b>{p_adj*100:.1f}%</b></p>
                                    <div class='{cls_ev}'>EV: {ev*100:+.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                                if ev > 0.03:
                                    stake = min(calculate_kelly(p_adj, c_now, bankroll, fraction=8),
                                                0.015 * bankroll)
                                    st.metric("Stake Sugerido", f"${stake:,.0f}")

    # --- TAB 6: RADAR LSI / ROTACIÓN ---
    with tab_copa:
        st.subheader("🏆 Radar LSI (Lineup Strength Index) & Rotación")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"#### 🏠 {local}")
            st.metric("Minutos Baseline XI Ideal", f"{ideal_home} m")
            st.metric("Minutos Titulares Hoy", f"{curr_home} m" if curr_home > 0 else "Sin XI confirmado")
            st.metric("Fuerza Titular (LSI)", f"{lsi_h*100:.1f}%")
        with col_c2:
            st.markdown(f"#### 🚀 {visita}")
            st.metric("Minutos Baseline XI Ideal", f"{ideal_away} m")
            st.metric("Minutos Titulares Hoy", f"{curr_away} m" if curr_away > 0 else "Sin XI confirmado")
            st.metric("Fuerza Titular (LSI)", f"{lsi_v*100:.1f}%")

    st.markdown("---")
    save_odds_history(st.session_state.odds_history)
    st.caption(f"Deep Soccer Master Suite V30.0 | Servidor Conectado | {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
