import pandas as pd

def limpiar_datos_gnss(archivo_entrada, archivo_salida):
    """
    Función para limpiar el dataset de deformación GNSS, conservando solo las columnas requeridas.
    
    Args:
        archivo_entrada (str): Ruta del archivo CSV de entrada
        archivo_salida (str): Ruta del archivo CSV de salida con los datos limpios
    """
    # Cargar el dataset
    df = pd.read_csv(archivo_entrada)
    
    # Seleccionar solo las columnas requeridas
    columnas_a_mantener = ['FECHA_UTC', 'ESTE', 'NORTE', 'VERTICAL']
    df_limpio = df[columnas_a_mantener].copy()
    
    # Verificar y limpiar valores nulos o inconsistentes
    # Reemplazar ceros por NaN en las columnas de medición (excepto posiblemente en la fecha inicial)
    columnas_medicion = ['ESTE', 'NORTE', 'VERTICAL']
    for col in columnas_medicion:
        # Verificar si hay valores no numéricos
        df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
    
    # Guardar el dataset limpio
    df_limpio.to_csv(archivo_salida, index=False)
    print(f"Datos limpios guardados en: {archivo_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    archivo_entrada = "catdedeformacionGNSS.csv"
    archivo_salida = "catdedeformacionGNSS_limpio.csv"
    limpiar_datos_gnss(archivo_entrada, archivo_salida)