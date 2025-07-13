# 🌋 Predicción de Erupciones Volcánicas con LSTM y Optimizadores

## 📋 Descripción

Este proyecto implementa un modelo de Deep Learning basado en LSTM (Long Short-Term Memory) para predecir erupciones volcánicas utilizando datos de deformación GNSS de volcanes en Perú. Se comparan tres algoritmos de optimización diferentes: Adam, SGD y RMSprop.

## 📄 Artículo Científico

- **📄 PDF del Artículo**: [ArtCientifico3.pdf](LATEX/ArtCientifico3.pdf)
- **📝 Código LaTeX**: [main.tex](LATEX/main.tex)
- **🖼️ Figuras**: [Carpeta de imágenes](LATEX/img/)

## 🎯 Objetivos

- Desarrollar un modelo LSTM para predicción de erupciones volcánicas
- Comparar el rendimiento de diferentes optimizadores
- Evaluar la efectividad del modelo para sistemas de alerta temprana
- Proporcionar visualizaciones detalladas del proceso de entrenamiento

## 📊 Resultados Principales

- **Mejor Optimizador**: RMSprop
- **Accuracy**: 92.03%
- **Recall**: 100% (detección perfecta de erupciones)
- **F1-Score**: 95.79%
- **Precision**: 91.91%

## 🏗️ Estructura del Proyecto

```
📁 proyecto/
├── 📁 LATEX/                   # Artículo científico
│   ├── 📄 main.tex            # Código fuente LaTeX
│   ├── 📄 ArtCientifico3.pdf  # PDF del artículo
│   └── 📁 img/                # Figuras del artículo
├── 📁 data/                   # Datos del proyecto
│   ├── 📁 data_clean/         # Datos limpios
│   ├── 📁 data_no_clean/      # Datos originales
│   └── catdedeformacionGNSS_con_erupciones.csv
├── 📁 imagenes/               # Gráficas adicionales
├── 📁 para_graficas_separadas/ # Scripts para generar gráficas
├── 📁 procesar los datos/     # Scripts de preprocesamiento
├── 📁 models/                 # Modelos entrenados (.h5)
├── 📄 modelo_lstm_erupciones.py  # Modelo principal
├── 📄 ejecutar_analisis.py    # Script de ejecución
└── 📄 requirements.txt        # Dependencias
```

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.8+
- pip

### Instalación

1. **Clonar el repositorio:**
```bash
git https://github.com/JefryErick/PREDICCION-DE-ERUPCIONES-VOLCANICAS.git
cd prediccion-erupciones-volcanicas
```

2. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

3. **Ejecutar el análisis:**
```bash
python ejecutar_analisis.py
```

## 📈 Características del Modelo

### Arquitectura LSTM
- **Capas LSTM**: 2 capas con 50 y 25 unidades
- **Dropout**: 0.2 para prevenir overfitting
- **Dense**: Capa final con activación sigmoid
- **Secuencia**: 10 pasos de tiempo

### Optimizadores Evaluados
1. **Adam**: Optimizador adaptativo con momentum
2. **SGD**: Descenso por gradiente estocástico con momentum
3. **RMSprop**: Optimizador adaptativo basado en RMS

## 📊 Métricas de Rendimiento

| Optimizador | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| Adam        | 91.30%   | 92.48%    | 98.40% | 95.35%   |
| SGD         | 90.58%   | 90.58%    | 100%   | 95.06%   |
| **RMSprop** | **92.03%** | **91.91%** | **100%** | **95.79%** |

## 🎨 Visualizaciones

El proyecto incluye múltiples visualizaciones:

### Métricas Individuales
- Comparación de accuracy por optimizador
- Análisis de precision y recall
- F1-Score comparativo

### Historial de Entrenamiento
- Curvas de pérdida (train/validation)
- Curvas de accuracy (train/validation)
- Análisis de convergencia

### Matrices de Confusión
- Visualización detallada de predicciones
- Análisis de falsos positivos/negativos
- Estadísticas por optimizador

### Visualización 3D
- Descenso por gradiente en espacio 3D
- Trayectorias de optimización
- Comparación de estrategias

## 🔬 Metodología

### Preprocesamiento de Datos
1. **Limpieza**: Eliminación de valores faltantes y outliers
2. **Normalización**: Escalado de características
3. **Secuenciación**: Creación de secuencias temporales
4. **División**: Train/Validation/Test (70/15/15)

### Entrenamiento
- **Épocas**: 100
- **Batch Size**: 32
- **Early Stopping**: Patience = 10
- **Callbacks**: ModelCheckpoint, EarlyStopping

### Evaluación
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Validación**: K-fold cross-validation
- **Análisis**: Matrices de confusión detalladas

## 📝 Archivos Principales

- `modelo_lstm_erupciones.py`: Implementación del modelo LSTM
- `ejecutar_analisis.py`: Script principal de ejecución
- `generar_graficas_separadas.py`: Generación de visualizaciones
- `LATEX/main.tex`: Código fuente del artículo científico
- `LATEX/ArtCientifico3.pdf`: Artículo científico en PDF
- `README.md`: Documentación del proyecto

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Jefry Erick Quispe Ramos** - *Estudiante* - [@JefryErick](https://github.com/JefryErick)

## 🙏 Agradecimientos

- Instituto Geofísico del Perú (IGP) por los datos de erupciones
- Comunidad científica por el desarrollo de herramientas de ML
- Contribuidores del proyecto

## 📞 Contacto

- **Email**: jefryerickq@gmail.com
- **GitHub**: [@JefryErick](https://github.com/JefryErick)
---

⭐ **Si este proyecto te fue útil, por favor dale una estrella en GitHub!** 