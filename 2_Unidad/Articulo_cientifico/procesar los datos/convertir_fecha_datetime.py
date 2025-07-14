import pandas as pd

def convertir_fecha_datetime():
    """
    Convierte la columna FECHA_UTC a formato DATETIME en el archivo
    catdedeformacionGNSS_con_erupciones.csv
    """
    
    # Cargar el archivo
    print("Cargando archivo CSV...")
    df = pd.read_csv('data/catdedeformacionGNSS_con_erupciones.csv')
    
    print(f"Archivo cargado: {len(df)} registros")
    print(f"Columnas actuales: {list(df.columns)}")
    
    # Mostrar el formato actual de FECHA_UTC
    print(f"\nFormato actual de FECHA_UTC (primeras 5 filas):")
    print(df['FECHA_UTC'].head())
    print(f"Tipo de dato actual: {df['FECHA_UTC'].dtype}")
    
    # Convertir FECHA_UTC a datetime
    # El formato actual es YYYYMMDD (ej: 20231001)
    df['FECHA_UTC'] = pd.to_datetime(df['FECHA_UTC'], format='%Y%m%d')
    
    print(f"\nFormato convertido de FECHA_UTC (primeras 5 filas):")
    print(df['FECHA_UTC'].head())
    print(f"Tipo de dato convertido: {df['FECHA_UTC'].dtype}")
    
    # Guardar el archivo con el nuevo formato
    output_file = 'data/catdedeformacionGNSS_con_erupciones.csv'
    df.to_csv(output_file, index=False)
    print(f"\nArchivo guardado como: {output_file}")
    
    # Mostrar información adicional
    print(f"\nRango de fechas:")
    print(f"Fecha más antigua: {df['FECHA_UTC'].min()}")
    print(f"Fecha más reciente: {df['FECHA_UTC'].max()}")
    
    return df

if __name__ == "__main__":
    resultado = convertir_fecha_datetime() 