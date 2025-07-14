#!/usr/bin/env python3
"""
SCRIPT DE EJECUCIÓN DEL ANÁLISIS LSTM
======================================

Este script ejecuta el análisis completo de predicción de erupciones volcánicas.
Asegúrate de tener todas las dependencias instaladas antes de ejecutar.

Para instalar dependencias:
pip install -r requirements.txt

Para ejecutar:
python ejecutar_analisis.py
"""

import sys
import os

def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas"""
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        import tensorflow
        print("✅ Todas las dependencias están instaladas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Por favor, instala las dependencias con: pip install -r requirements.txt")
        return False

def verificar_archivo_datos():
    """Verifica que el archivo de datos existe"""
    archivo_datos = 'data/catdedeformacionGNSS_con_erupciones.csv'
    if os.path.exists(archivo_datos):
        print(f"✅ Archivo de datos encontrado: {archivo_datos}")
        return True
    else:
        print(f"❌ Error: No se encontró el archivo {archivo_datos}")
        print("Asegúrate de que el archivo esté en la carpeta 'data/'")
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("ANÁLISIS LSTM PARA PREDICCIÓN DE ERUPCIONES VOLCÁNICAS")
    print("=" * 60)
    
    # Verificar dependencias
    if not verificar_dependencias():
        sys.exit(1)
    
    # Verificar archivo de datos
    if not verificar_archivo_datos():
        sys.exit(1)
    
    print("\n🚀 Iniciando análisis...")
    
    try:
        # Importar y ejecutar el modelo
        from modelo_lstm_erupciones import ModeloErupcionesVolcanicas
        
        # Crear instancia del modelo
        modelo = ModeloErupcionesVolcanicas()
        
        # Ejecutar análisis completo
        resultados = modelo.ejecutar_analisis_completo()
        
        print("\n" + "=" * 60)
        print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        print("\n📊 ARCHIVOS GENERADOS:")
        print("• modelo_lstm_adam.h5 - Modelo entrenado con Adam")
        print("• modelo_lstm_sgd.h5 - Modelo entrenado con SGD")
        print("• modelo_lstm_rmsprop.h5 - Modelo entrenado con RMSprop")
        print("• metrics_comparison.csv - Comparación de métricas")
        print("• training_history.png - Historial de entrenamiento")
        print("• confusion_matrices.png - Matrices de confusión")
        print("• metrics_comparison.png - Comparación de métricas")
        print("• gradient_descent_visual.png - Visualización del descenso por gradiente")
        
        print("\n📈 RESUMEN DE RESULTADOS:")
        for optimizador, metricas in resultados.items():
            print(f"\n{optimizador.upper()}:")
            print(f"  • Accuracy: {metricas['accuracy']:.4f}")
            print(f"  • Precision: {metricas['precision']:.4f}")
            print(f"  • Recall: {metricas['recall']:.4f}")
            print(f"  • F1-Score: {metricas['f1_score']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("Por favor, verifica que todos los archivos estén en su lugar correcto.")
        sys.exit(1)

if __name__ == "__main__":
    main() 