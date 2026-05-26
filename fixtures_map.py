# fixtures_map.py

# Diccionario geolocalizado de los equipos de Primera A de Chile (Latitud, Longitud)
ESTADIOS_CHILE = {
    "Colo-Colo": {"estadio": "Estadio Monumental, Santiago", "lat": -33.5065, "lon": -70.6059},
    "Universidad de Chile": {"estadio": "Estadio Nacional, Santiago", "lat": -33.4600, "lon": -70.6105},
    "Universidad Chile": {"estadio": "Estadio Nacional, Santiago", "lat": -33.4600, "lon": -70.6105},
    "Universidad Catolica": {"estadio": "San Carlos de Apoquindo, Santiago", "lat": -33.3950, "lon": -70.5005},
    "Everton": {"estadio": "Estadio Sausalito, Viña del Mar", "lat": -33.0138, "lon": -71.5367},
    "Santiago Wanderers": {"estadio": "Elias Figueroa, Valparaíso", "lat": -33.0560, "lon": -71.6253},
    "Cobreloa": {"estadio": "Zorros del Desierto, Calama", "lat": -22.4646, "lon": -68.9248},
    "Cobresal": {"estadio": "Estadio El Cobre, El Salvador", "lat": -26.2443, "lon": -69.6277},
    "Coquimbo Unido": {"estadio": "Estadio Francisco Sánchez Rumoroso, Coquimbo", "lat": -29.9672, "lon": -71.3414},
    "Huachipato": {"estadio": "Estadio CAP, Talcahuano", "lat": -36.7454, "lon": -73.1097},
    "Deportes Iquique": {"estadio": "Tierra de Campeones, Iquique", "lat": -20.2446, "lon": -70.1340},
    "Deportes Copiapo": {"estadio": "Luis Valenzuela Hermosilla, Copiapó", "lat": -27.3758, "lon": -70.3292},
    "Palestino": {"estadio": "Estadio Municipal de La Cisterna, Santiago", "lat": -33.5350, "lon": -70.6653},
    "Union Espanola": {"estadio": "Estadio Santa Laura, Santiago", "lat": -33.4072, "lon": -70.6558},
    "Audax Italiano": {"estadio": "Bicentenario de La Florida, Santiago", "lat": -33.5222, "lon": -70.5964},
    "O'Higgins": {"estadio": "El Teniente, Rancagua", "lat": -34.1755, "lon": -70.7397},
    "Nublense": {"estadio": "Nelson Oyarzún, Chillán", "lat": -36.6111, "lon": -72.1027},
    "Union La Calera": {"estadio": "Nicolás Chahuán, La Calera", "lat": -32.7844, "lon": -71.2158}
}

def obtener_coordenadas_estadio(equipo_local):
    """Busca las coordenadas del equipo local. Si no lo encuentra, usa Santiago por defecto."""
    # Búsqueda flexible
    for key, data in ESTADIOS_CHILE.items():
        if key.lower() in equipo_local.lower() or equipo_local.lower() in key.lower():
            return data
    
    # Fallback: Santiago Centro
    print(f"⚠️ Estadio no mapeado para '{equipo_local}'. Usando Santiago por defecto.")
    return {"estadio": "Estadio Nacional (Default)", "lat": -33.4489, "lon": -70.6693}

def obtener_partidos_jornada():
    """
    En producción, esto leería el CSV de 'Upcoming Matches' de FootyStats.
    Por ahora, devolvemos un fixture de prueba para el Dashboard.
    """
    return [
        "Everton vs Universidad Chile",
        "Cobreloa vs Colo-Colo",
        "Huachipato vs Universidad Catolica",
        "O'Higgins vs Palestino"
    ]
