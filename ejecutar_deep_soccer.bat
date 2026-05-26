@echo off
setlocal
title Deep Soccer AI - Master Suite V32.1 (Multi-League)

:menu
cls
echo =======================================================
echo           DEEP SOCCER AI - MASTER SUITE V32.1 (Multi-League)
echo =======================================================
echo.
echo [1] ACTUALIZAR DATASET CHILE (Campeonato Nacional)
echo [2] ACTUALIZAR DATASET CHILE (Primera B)
echo [3] ACTUALIZAR DATASET PREMIER (Inglaterra)
echo [4] ABRIR DASHBOARD MULTI-LIGA
echo [5] VERIFICAR CONEXION Y LIBRERIAS
echo [6] SALIR
echo [7] RE-ENTRENAR MODELOS (Sin bajar datos)
echo.
echo =======================================================
set /p opt="Selecciona una opcion [1-7]: "

if "%opt%"=="1" goto update_chile
if "%opt%"=="2" goto update_chile_b
if "%opt%"=="3" goto update_premier
if "%opt%"=="4" goto dashboard
if "%opt%"=="5" goto verify
if "%opt%"=="6" goto exit
if "%opt%"=="7" goto retrain_all
goto menu

:verify
cls
echo [1/3] Verificando Librerias...
python -c "import requests, bs4, pandas, xgboost, google.generativeai, streamlit; print('Librerias OK')" || (echo Instalando librerias faltantes... && pip install -r requirements.txt)

echo [2/3] Verificando Conexion a API-Sports...
python -c "import os, requests; from dotenv import load_dotenv; load_dotenv(); r=requests.get('https://v3.football.api-sports.io/status', headers={'x-apisports-key': os.getenv('APISPORTS_KEY')}); print('API-Sports Status:', r.status_code)"

echo [3/3] Probando Scraper Engine...
python scraper_cl.py
echo.
pause
goto menu

:update_chile
cls
echo [1/3] Sincronizando Chile (scraper_api.py)...
python scraper_api.py 265 chile_api_raw.csv 0
echo [2/3] Procesando Data Chile...
python processor.py chile_api_raw.csv chile_ml_ready_v8.csv
echo [3/3] Entrenando Modelos Chile...
python advanced_model.py chile_ml_ready_v8.csv
echo.
echo Sincronización Chile completa.
pause
goto menu

:update_chile_b
cls
echo [1/3] Sincronizando Chile B (scraper_api.py)...
python scraper_api.py 266 chile_b_api_raw.csv 0
echo [2/3] Procesando Data Chile B...
python processor.py chile_b_api_raw.csv chile_b_ml_ready.csv
echo [3/3] Entrenando Modelos Chile B...
python advanced_model.py chile_b_ml_ready.csv B
echo.
echo Sincronización Primera B completa.
pause
goto menu

:update_premier
cls
echo [1/3] Sincronizando Premier (scraper_masivo_premier.py)...
python scraper_masivo_premier.py
echo [2/3] Procesando Data Premier...
python processor.py premier_api_raw.csv premier_ml_ready_v1.csv
echo [3/3] Entrenando Modelos Premier...
python advanced_model.py premier_ml_ready_v1.csv
echo.
echo Sincronización Premier completa.
pause
goto menu


:retrain_all
cls
echo =====================================================
echo   RE-ENTRENAMIENTO FORZADO - TODOS LOS MODELOS
echo   (Los CSVs deben estar actualizados previamente)
echo =====================================================
echo.
echo [1/3] Re-entrenando modelos Chile (Campeonato Nacional)...
python advanced_model.py chile_ml_ready_v8.csv
if %errorlevel% neq 0 (echo ERROR en Chile. Abortando. && pause && goto menu)
echo.
echo [2/3] Re-entrenando modelos Chile B (Primera B)...
python advanced_model.py chile_b_ml_ready.csv B
if %errorlevel% neq 0 (echo ERROR en Chile B. Abortando. && pause && goto menu)
echo.
echo [3/3] Re-entrenando modelos Premier League...
python advanced_model.py premier_ml_ready_v1.csv
if %errorlevel% neq 0 (echo ERROR en Premier. Abortando. && pause && goto menu)
echo.
echo =====================================================
echo  TODOS LOS MODELOS RE-ENTRENADOS CORRECTAMENTE
echo  Los nuevos .json estan listos para inferencia.
echo =====================================================
echo.
pause
goto menu

:dashboard
cls
echo Lanzando Dashboard V30.0 Multi-League...
python -m streamlit run app_master.py
if %errorlevel% neq 0 pause
goto menu

:exit
exit

