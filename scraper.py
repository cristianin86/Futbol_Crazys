import os
import time
import glob
import sys
import io
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Cargar credenciales de forma segura
load_dotenv(override=True)
EMAIL = os.getenv("FOOTYSTATS_EMAIL")
PASSWORD = os.getenv("FOOTYSTATS_PASSWORD")


def descargar_h2h_datasets_v3():
    print("🚀 Iniciando Scraper V3 (Datasets Page Target)...")
    
    if not EMAIL or not PASSWORD:
        print("❌ ERROR CRÍTICO: No se encontraron las credenciales en el archivo .env")
        return

    # Configurar directorio de descarga al directorio actual del proyecto
    directorio_actual = os.getcwd()
    prefs = {
        "download.default_directory": directorio_actual,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", prefs)
    options.add_argument('--headless') # Ejecución en modo fantasma activada
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 15)
    
    try:
        # 2. Login robusto
        print("🌐 Accediendo a login y verificando credenciales Premium...")
        driver.get("https://footystats.org/login")
        
        email_input = wait.until(EC.presence_of_element_located((By.ID, "username")))
        password_input = driver.find_element(By.ID, "password")
        
        email_input.clear()
        email_input.send_keys(EMAIL)
        password_input.clear()
        password_input.send_keys(PASSWORD)
        
        login_button = wait.until(EC.element_to_be_clickable((By.ID, "register_submit"))) 
        driver.execute_script("arguments[0].click();", login_button)
        
        print("✅ Credenciales enviadas. Esperando validación del servidor...")
        time.sleep(4) 
        
        # 3. Navegar directamente a la página de Datasets (NUEVO PLAN)
        url_datasets = "https://footystats.org/chile/primera-division/datasets"
        print(f"⚽ Navegando a la central de datos: {url_datasets}")
        driver.get(url_datasets)
        time.sleep(4) # Dar tiempo a que carguen las tablas Premium
        
        print("📥 Buscando el botón de descarga del CSV de Partidos (Matches H2H)...")
        try:
            # Estrategia A: Buscar el botón específico de la temporada 2026
            # Usamos un selector que asegure que estamos en la fila de 2026 para evitar los samples de la Premier League
            xpath_selector = "//tr[td[contains(., '2026/2026')]]//a[contains(@href, 'type=matches')]"
            csv_match_button = wait.until(EC.presence_of_element_located((By.XPATH, xpath_selector)))
            
            driver.execute_script("arguments[0].scrollIntoView(true);", csv_match_button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", csv_match_button)
            print("✅ ¡Botón localizado y clicado con éxito!")
            
        except Exception as e_btn:
            print("⚠️ El selector principal falló. Activando Fallback de URL directa...")
            # Estrategia B (Fallback): Extraer todas las URLs de la página y forzar la descarga de la que contenga los datos de partidos
            enlaces = driver.find_elements(By.XPATH, "//a[@href]")
            url_directa = None
            
            for enlace in enlaces:
                href = enlace.get_attribute("href").lower()
                if 'csv' in href and ('match' in href or 'h2h' in href):
                    url_directa = href
                    break
                    
            if url_directa:
                print(f"🔗 URL de descarga directa interceptada: {url_directa}")
                driver.get(url_directa) # Navegar directamente al archivo fuerza la descarga
            else:
                raise Exception("Fallo en el Fallback: No se pudo localizar ningún enlace de descarga CSV de partidos en esta página.")

        print("⏳ Descarga en progreso. Esperando 15 segundos a que el archivo caiga en el disco...")
        time.sleep(15)
        print(f"✅ Extracción finalizada. Revisa tu carpeta: {directorio_actual}")

    except Exception as e:
        print(f"❌ Error durante el scraping: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    descargar_h2h_datasets_v3()
