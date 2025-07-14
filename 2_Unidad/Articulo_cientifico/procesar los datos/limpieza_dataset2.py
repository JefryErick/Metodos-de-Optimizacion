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
    columnas_a_mantener = ['FECHA_UTC']
    df_limpio = df[columnas_a_mantener].copy()
    
    
    # Guardar el dataset limpio
    df_limpio.to_csv(archivo_salida, index=False)
    print(f"Datos limpios guardados en: {archivo_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    archivo_entrada = "IGP_Explosiones.csv"
    archivo_salida = "IGP_Explosiones_limpio.csv"
    limpiar_datos_gnss(archivo_entrada, archivo_salida)