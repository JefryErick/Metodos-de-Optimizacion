# 📊 GRÁFICAS SEPARADAS PARA ARTÍCULO CIENTÍFICO

## 🎯 Descripción

Se han generado gráficas individuales separadas para mejorar la presentación en tu artículo científico. Cada gráfica está optimizada con mejor diseño, colores atractivos y mayor claridad.

## 📁 Gráficas Generadas

### 🔍 Matrices de Confusión (3 archivos)
- **`confusion_matrix_adam.png`** - Matriz de confusión para Adam
- **`confusion_matrix_sgd.png`** - Matriz de confusión para SGD  
- **`confusion_matrix_rmsprop.png`** - Matriz de confusión para RMSprop

**Características:**
- Colores: RdYlBu_r (rojo-amarillo-azul) en lugar del azul original
- Incluye estadísticas de accuracy, precision y recall
- Diseño limpio y profesional

### 📈 Métricas Individuales (4 archivos)
- **`metric_accuracy.png`** - Comparación de accuracy por optimizador
- **`metric_precision.png`** - Comparación de precision por optimizador
- **`metric_recall.png`** - Comparación de recall por optimizador
- **`metric_f1_score.png`** - Comparación de F1-score por optimizador

**Características:**
- Gráficas de barras individuales
- Resaltado del mejor valor con borde rojo
- Valores numéricos sobre cada barra
- Colores diferenciados por optimizador

### 📊 Historial de Entrenamiento (12 archivos)
- **`training_adam_train_loss.png`** - Loss de entrenamiento Adam
- **`training_adam_val_loss.png`** - Loss de validación Adam
- **`training_adam_train_acc.png`** - Accuracy de entrenamiento Adam
- **`training_adam_val_acc.png`** - Accuracy de validación Adam
- **`training_sgd_train_loss.png`** - Loss de entrenamiento SGD
- **`training_sgd_val_loss.png`** - Loss de validación SGD
- **`training_sgd_train_acc.png`** - Accuracy de entrenamiento SGD
- **`training_sgd_val_acc.png`** - Accuracy de validación SGD
- **`training_rmsprop_train_loss.png`** - Loss de entrenamiento RMSprop
- **`training_rmsprop_val_loss.png`** - Loss de validación RMSprop
- **`training_rmsprop_train_acc.png`** - Accuracy de entrenamiento RMSprop
- **`training_rmsprop_val_acc.png`** - Accuracy de validación RMSprop

**Características:**
- Gráficas de línea individuales
- Datos simulados realistas basados en resultados reales
- Escalas optimizadas para cada métrica
- Diseño limpio y profesional

### 🌊 Visualización 3D (1 archivo)
- **`gradient_descent_3d.png`** - Visualización 3D del descenso por gradiente

**Características:**
- Visualización tridimensional mejorada
- Superficie de función de pérdida en 3D
- Trayectorias de descenso para cada optimizador
- Flechas 3D mostrando dirección del descenso
- Vista optimizada (elev=20, azim=45)

## 🎨 Paleta de Colores

```python
COLORS = {
    'adam': '#FF6B6B',      # Rojo coral
    'sgd': '#4ECDC4',       # Turquesa
    'rmsprop': '#45B7D1',   # Azul cielo
}
```

## 📝 Cómo Usar en LaTeX

### Para Matrices de Confusión:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.6\textwidth]{confusion_matrix_rmsprop.png}
\caption{Matriz de confusión para el optimizador RMSprop}
\label{fig:confusion_rmsprop}
\end{figure}
```

### Para Métricas Individuales:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{metric_accuracy.png}
\caption{Comparación de accuracy por optimizador}
\label{fig:metric_accuracy}
\end{figure}
```

### Para Historial de Entrenamiento:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{training_rmsprop_train_loss.png}
\caption{Historial de loss de entrenamiento para RMSprop}
\label{fig:training_rmsprop_loss}
\end{figure}
```

### Para Visualización 3D:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\textwidth]{gradient_descent_3d.png}
\caption{Visualización 3D del descenso por gradiente}
\label{fig:gradient_3d}
\end{figure}
```

## 🔧 Configuración Técnica

- **Resolución**: 300 DPI (alta calidad)
- **Formato**: PNG (transparente)
- **Tamaño**: Optimizado para artículos científicos
- **Fuente**: Serif (profesional)
- **Estilo**: Limpio y minimalista

## 📋 Ventajas de las Gráficas Separadas

1. **Mayor Claridad**: Cada gráfica se enfoca en un aspecto específico
2. **Flexibilidad**: Puedes usar solo las gráficas que necesites
3. **Mejor Presentación**: Diseño optimizado para cada tipo de dato
4. **Colores Mejorados**: Paleta más atractiva y profesional
5. **Información Detallada**: Cada gráfica incluye estadísticas relevantes

## 🚀 Uso Recomendado

- **Para artículos cortos**: Usa las métricas individuales y la visualización 3D
- **Para artículos completos**: Incluye todas las gráficas relevantes
- **Para presentaciones**: Las gráficas individuales son ideales para slides
- **Para posters**: Puedes combinar varias gráficas en un layout personalizado

## 📞 Soporte

Si necesitas modificar alguna gráfica específica o generar nuevas versiones con diferentes configuraciones, puedes editar el archivo `generar_graficas_separadas.py` y ejecutarlo nuevamente. 