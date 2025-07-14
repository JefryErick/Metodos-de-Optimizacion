import pandas as pd
"""
# Cargar datos
df_gnss = pd.read_csv('GNSS.csv')

# Convertir FECHA_UTC a datetime (formato YYYYMMDD)
df_gnss['FECHA_UTC'] = pd.to_datetime(df_gnss['FECHA_UTC'], format='%Y%m%d')

# Limpieza básica
df_gnss_clean = df_gnss[['FECHA_UTC', 'ESTE', 'NORTE', 'VERTICAL', 'VOLCAN']].copy()

# Eliminar filas con deformación = 0 en todas las componentes (posible error)
df_gnss_clean = df_gnss_clean[~((df_gnss_clean['ESTE'] == 0) & 
                               (df_gnss_clean['NORTE'] == 0) & 
                               (df_gnss_clean['VERTICAL'] == 0))]  # Se añadió este paréntesis

# Guardar
df_gnss_clean.to_csv('gnss_clean.csv', index=False)

#-------
#explosiones
df_explosiones = pd.read_csv('explosiones.csv')

# Convertir FECHA_UTC y limpiar
df_explosiones['FECHA_UTC'] = pd.to_datetime(df_explosiones['FECHA_UTC'], format='%Y%m%d')
df_explosiones_clean = df_explosiones[['FECHA_UTC', 'ALTURA', 'VOLCAN']].dropna()

# Guardar
df_explosiones_clean.to_csv('explosiones_clean.csv', index=False)

"""
#sismos
df_sismos = pd.read_csv('sismos.csv')

# Convertir FECHA_UTC y filtrar
df_sismos['FECHA_UTC'] = pd.to_datetime(df_sismos['FECHA_UTC'], format='%Y%m%d')
df_sismos_clean = df_sismos[['FECHA_UTC', 'TIPO', 'ENERGIA', 'VOLCAN']]

# Agregar por día y tipo sísmico
sismos_daily = df_sismos_clean.groupby(['FECHA_UTC', 'TIPO']).agg(
    conteo=('ENERGIA', 'count'),
    energia_total=('ENERGIA', 'sum')
).unstack(fill_value=0)

# Aplanar columnas
sismos_daily.columns = [f'{col[0]}_{col[1]}' for col in sismos_daily.columns]
sismos_daily.reset_index(inplace=True)

# Guardar
sismos_daily.to_csv('sismos_daily_clean.csv', index=False)