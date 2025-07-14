"""
MODELO LSTM PARA PREDICCIÓN DE ERUPCIONES VOLCÁNICAS
====================================================

Autor: Sistema de Análisis Científico
Fecha: 2024
Objetivo: Predicción de erupciones volcánicas usando datos GNSS y redes neuronales LSTM

Este código implementa un modelo de red neuronal LSTM para predecir erupciones
volcánicas basándose en datos de deformación GNSS (ESTE, NORTE, VERTICAL).
Se comparan tres optimizadores: Adam, SGD y RMSprop.

Referencias científicas:
- Hochreiter & Schmidhuber (1997) - LSTM
- Kingma & Ba (2014) - Adam optimizer
- Tieleman & Hinton (2012) - RMSprop
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configuración para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

class ModeloErupcionesVolcanicas:
    """
    Clase principal para el modelo de predicción de erupciones volcánicas
    """
    
    def __init__(self, csv_path='data/catdedeformacionGNSS_con_erupciones.csv'):
        """
        Inicialización del modelo
        
        Args:
            csv_path (str): Ruta al archivo CSV con los datos
        """
        self.csv_path = csv_path
        self.data = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 7
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.models = {}
        self.histories = {}
        self.metrics = {}
        
    def cargar_datos(self):
        """
        Carga y preprocesa los datos del CSV
        """
        print("=== CARGA Y PREPROCESAMIENTO DE DATOS ===")
        
        # Cargar datos
        self.data = pd.read_csv(self.csv_path)
        print(f"Datos cargados: {len(self.data)} registros")
        print(f"Columnas: {list(self.data.columns)}")
        
        # Convertir FECHA_UTC a datetime si no lo está
        if self.data['FECHA_UTC'].dtype == 'object':
            self.data['FECHA_UTC'] = pd.to_datetime(self.data['FECHA_UTC'])
        
        # Verificar datos faltantes
        print(f"\nDatos faltantes:")
        print(self.data.isnull().sum())
        
        # Estadísticas descriptivas
        print(f"\nEstadísticas descriptivas:")
        print(self.data[['ESTE', 'NORTE', 'VERTICAL']].describe())
        
        # Distribución de la variable objetivo
        print(f"\nDistribución de ERUPCION:")
        print(self.data['ERUPCION'].value_counts())
        print(f"Porcentaje de erupciones: {self.data['ERUPCION'].mean()*100:.2f}%")
        
        return self.data
    
    def normalizar_datos(self):
        """
        Normaliza las columnas ESTE, NORTE, VERTICAL
        """
        print("\n=== NORMALIZACIÓN DE DATOS ===")
        
        # Normalizar solo las columnas numéricas (excluyendo FECHA_UTC y ERUPCION)
        features = ['ESTE', 'NORTE', 'VERTICAL']
        self.data[features] = self.scaler.fit_transform(self.data[features])
        
        print("Datos normalizados con MinMaxScaler")
        print("Rango de valores normalizados: [0, 1]")
        
        return self.data
    
    def crear_secuencias(self):
        """
        Crea secuencias de 7 días para el modelo LSTM
        """
        print("\n=== CREACIÓN DE SECUENCIAS LSTM ===")
        
        # Preparar datos
        features = ['ESTE', 'NORTE', 'VERTICAL']
        X = self.data[features].values
        y = self.data['ERUPCION'].values
        
        # Crear secuencias
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Secuencias creadas: {len(X_sequences)}")
        print(f"Forma de X: {X_sequences.shape}")
        print(f"Forma de y: {y_sequences.shape}")
        
        # Dividir en train, validation y test (70%, 15%, 15%)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.15, random_state=42, stratify=y_sequences
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"\nDivisión de datos:")
        print(f"Train: {len(self.X_train)} ({len(self.X_train)/len(X_sequences)*100:.1f}%)")
        print(f"Validation: {len(self.X_val)} ({len(self.X_val)/len(X_sequences)*100:.1f}%)")
        print(f"Test: {len(self.X_test)} ({len(self.X_test)/len(X_sequences)*100:.1f}%)")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def crear_modelo(self, optimizer_name='adam'):
        """
        Crea el modelo LSTM
        
        Args:
            optimizer_name (str): Nombre del optimizador ('adam', 'sgd', 'rmsprop')
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 3)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Configurar optimizador
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=0.001)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=0.01, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def entrenar_modelo(self, optimizer_name, epochs=100, batch_size=32):
        """
        Entrena el modelo con el optimizador especificado
        
        Args:
            optimizer_name (str): Nombre del optimizador
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
        """
        print(f"\n=== ENTRENAMIENTO CON {optimizer_name.upper()} ===")
        
        # Crear modelo
        model = self.crear_modelo(optimizer_name)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Entrenar modelo
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Guardar modelo y historial
        self.models[optimizer_name] = model
        self.histories[optimizer_name] = history
        
        return model, history
    
    def evaluar_modelo(self, optimizer_name):
        """
        Evalúa el modelo y calcula métricas
        
        Args:
            optimizer_name (str): Nombre del optimizador
        """
        print(f"\n=== EVALUACIÓN CON {optimizer_name.upper()} ===")
        
        model = self.models[optimizer_name]
        
        # Predicciones
        y_pred_proba = model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Guardar métricas
        self.metrics[optimizer_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nMatriz de confusión:")
        print(cm)
        
        return self.metrics[optimizer_name]
    
    def crear_visualizaciones(self):
        """
        Crea todas las visualizaciones necesarias
        """
        print("\n=== CREACIÓN DE VISUALIZACIONES ===")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Gráficas de loss y accuracy para cada optimizador
        self.plot_training_history()
        
        # 2. Matrices de confusión
        self.plot_confusion_matrices()
        
        # 3. Comparación de métricas
        self.plot_metrics_comparison()
        
        # 4. Visualización del descenso por gradiente
        self.plot_gradient_descent_concept()
        
        print("Visualizaciones guardadas en el directorio actual")
    
    def plot_training_history(self):
        """
        Gráficas de loss y accuracy durante el entrenamiento
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Historial de Entrenamiento por Optimizador', fontsize=16, fontweight='bold')
        
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for i, optimizer in enumerate(optimizers):
            if optimizer in self.histories:
                history = self.histories[optimizer]
                
                # Loss
                axes[0, i].plot(history.history['loss'], label='Train Loss', linewidth=2)
                axes[0, i].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
                axes[0, i].set_title(f'{optimizer.upper()} - Loss')
                axes[0, i].set_xlabel('Época')
                axes[0, i].set_ylabel('Loss')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # Accuracy
                axes[1, i].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
                axes[1, i].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                axes[1, i].set_title(f'{optimizer.upper()} - Accuracy')
                axes[1, i].set_xlabel('Época')
                axes[1, i].set_ylabel('Accuracy')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """
        Matrices de confusión para cada optimizador
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Matrices de Confusión por Optimizador', fontsize=16, fontweight='bold')
        
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for i, optimizer in enumerate(optimizers):
            if optimizer in self.metrics:
                cm = self.metrics[optimizer]['confusion_matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Erupción', 'Erupción'],
                           yticklabels=['No Erupción', 'Erupción'],
                           ax=axes[i])
                axes[i].set_title(f'{optimizer.upper()}')
                axes[i].set_xlabel('Predicción')
                axes[i].set_ylabel('Valor Real')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self):
        """
        Comparación de métricas entre optimizadores
        """
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        # Preparar datos
        data = []
        for optimizer in optimizers:
            if optimizer in self.metrics:
                row = [optimizer.upper()]
                for metric in metrics_names:
                    row.append(self.metrics[optimizer][metric])
                data.append(row)
        
        df_metrics = pd.DataFrame(data, columns=['Optimizador'] + metrics_names)
        
        # Gráfica de barras
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparación de Métricas por Optimizador', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics_names):
            ax = axes[i//2, i%2]
            bars = ax.bar(df_metrics['Optimizador'], df_metrics[metric], 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Valor')
            ax.set_ylim(0, 1)
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Guardar métricas en CSV
        df_metrics.to_csv('metrics_comparison.csv', index=False)
        print("Métricas guardadas en 'metrics_comparison.csv'")
    
    def plot_gradient_descent_concept(self):
        """
        Visualización conceptual del descenso por gradiente
        """
        # Crear función de ejemplo (parábola)
        x = np.linspace(-3, 3, 100)
        y = x**2 + 1
        
        # Puntos de inicio para cada optimizador
        start_points = {
            'Adam': (-2.5, 7.25),
            'SGD': (-2.0, 5.0),
            'RMSprop': (-1.5, 3.25)
        }
        
        # Colores
        colors = {'Adam': '#FF6B6B', 'SGD': '#4ECDC4', 'RMSprop': '#45B7D1'}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Graficar función
        ax.plot(x, y, 'k-', linewidth=3, label='Función de pérdida')
        
        # Graficar puntos de inicio y trayectorias
        for optimizer, (start_x, start_y) in start_points.items():
            # Simular trayectoria de descenso
            x_traj = [start_x]
            y_traj = [start_y]
            
            current_x = start_x
            current_y = start_y
            
            for _ in range(10):
                # Derivada: dy/dx = 2x
                gradient = 2 * current_x
                
                # Diferentes tasas de aprendizaje para cada optimizador
                if optimizer == 'Adam':
                    lr = 0.1
                elif optimizer == 'SGD':
                    lr = 0.05
                else:  # RMSprop
                    lr = 0.08
                
                # Actualizar posición
                current_x = current_x - lr * gradient
                current_y = current_x**2 + 1
                
                x_traj.append(current_x)
                y_traj.append(current_y)
            
            # Graficar trayectoria
            ax.plot(x_traj, y_traj, 'o-', color=colors[optimizer], 
                   linewidth=2, markersize=8, label=f'{optimizer}')
            
            # Agregar flechas
            for i in range(len(x_traj)-1):
                ax.annotate('', xy=(x_traj[i+1], y_traj[i+1]), 
                           xytext=(x_traj[i], y_traj[i]),
                           arrowprops=dict(arrowstyle='->', color=colors[optimizer], lw=2))
        
        ax.set_xlabel('Parámetros del modelo', fontsize=12)
        ax.set_ylabel('Función de pérdida', fontsize=12)
        ax.set_title('Visualización Conceptual del Descenso por Gradiente\nComparación de Optimizadores', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig('gradient_descent_visual.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def guardar_modelos(self):
        """
        Guarda los modelos entrenados
        """
        print("\n=== GUARDADO DE MODELOS ===")
        
        for optimizer_name, model in self.models.items():
            filename = f'modelo_lstm_{optimizer_name}.h5'
            model.save(filename)
            print(f"Modelo {optimizer_name} guardado como: {filename}")
    
    def ejecutar_analisis_completo(self):
        """
        Ejecuta el análisis completo
        """
        print("=" * 60)
        print("MODELO LSTM PARA PREDICCIÓN DE ERUPCIONES VOLCÁNICAS")
        print("=" * 60)
        
        # 1. Carga y preprocesamiento
        self.cargar_datos()
        self.normalizar_datos()
        
        # 2. Crear secuencias
        self.crear_secuencias()
        
        # 3. Entrenar modelos con diferentes optimizadores
        optimizers = ['adam', 'sgd', 'rmsprop']
        
        for optimizer in optimizers:
            self.entrenar_modelo(optimizer, epochs=100, batch_size=32)
            self.evaluar_modelo(optimizer)
        
        # 4. Crear visualizaciones
        self.crear_visualizaciones()
        
        # 5. Guardar modelos
        self.guardar_modelos()
        
        print("\n" + "=" * 60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return self.metrics

# Ejecutar el análisis
if __name__ == "__main__":
    modelo = ModeloErupcionesVolcanicas()
    resultados = modelo.ejecutar_analisis_completo() 