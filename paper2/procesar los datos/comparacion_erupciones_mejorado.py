import pandas as pd

def comparar_erupciones():
    """
    Lee los archivos CSV de explosiones y deformación GNSS,
    compara las fechas UTC y agrega una columna ERUPCION
    """
    
    # Cargar los datasets
    print("Cargando archivos CSV...")
    explosiones_df = pd.read_csv('IGP_Explosiones_limpio.csv')
    deformacion_df = pd.read_csv('catdedeformacionGNSS_limpio.csv')
    
    print(f"Dataset de explosiones: {len(explosiones_df)} registros")
    print(f"Dataset de deformación: {len(deformacion_df)} registros")
    
    # Obtener las fechas únicas de explosiones
    fechas_explosiones = set(explosiones_df['FECHA_UTC'].astype(str))
    print(f"Fechas únicas con explosiones: {len(fechas_explosiones)}")
    
    # Mostrar rango de fechas en cada dataset
    print(f"Rango de fechas en explosiones: {min(explosiones_df['FECHA_UTC'])} - {max(explosiones_df['FECHA_UTC'])}")
    print(f"Rango de fechas en deformación: {min(deformacion_df['FECHA_UTC'])} - {max(deformacion_df['FECHA_UTC'])}")
    
    # Crear la nueva columna ERUPCION
    # Convertir FECHA_UTC a string para comparación
    deformacion_df['FECHA_UTC_str'] = deformacion_df['FECHA_UTC'].astype(str)
    
    # Aplicar la lógica: 1 si la fecha está en explosiones, 0 si no
    deformacion_df['ERUPCION'] = deformacion_df['FECHA_UTC_str'].apply(
        lambda x: 1 if x in fechas_explosiones else 0
    )
    
    # Eliminar la columna temporal
    deformacion_df = deformacion_df.drop('FECHA_UTC_str', axis=1)
    
    # Mostrar estadísticas detalladas
    total_erupciones = deformacion_df['ERUPCION'].sum()
    total_sin_erupcion = len(deformacion_df) - total_erupciones
    
    print(f"\n=== RESULTADOS ===")
    print(f"Registros con erupción (1): {total_erupciones}")
    print(f"Registros sin erupción (0): {total_sin_erupcion}")
    print(f"Porcentaje con erupción: {(total_erupciones/len(deformacion_df)*100):.2f}%")
    
    # Guardar el resultado
    output_file = 'catdedeformacionGNSS_con_erupciones.csv'
    deformacion_df.to_csv(output_file, index=False)
    print(f"\nArchivo guardado como: {output_file}")
    
    # Mostrar las primeras filas con la nueva columna
    print("\nPrimeras 10 filas del resultado:")
    print(deformacion_df[['FECHA_UTC', 'ERUPCION']].head(10))
    
    # Mostrar algunas fechas con erupciones para verificar
    fechas_con_erupcion = deformacion_df[deformacion_df['ERUPCION'] == 1]['FECHA_UTC'].head(10)
    print(f"\nPrimeras 10 fechas con erupción: {list(fechas_con_erupcion)}")
    
    # Mostrar algunas fechas sin erupciones
    fechas_sin_erupcion = deformacion_df[deformacion_df['ERUPCION'] == 0]['FECHA_UTC'].head(10)
    print(f"Primeras 10 fechas sin erupción: {list(fechas_sin_erupcion)}")
    
    return deformacion_df

if __name__ == "__main__":
    resultado = comparar_erupciones() 