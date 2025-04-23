#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para evaluar, visualizar y continuar el entrenamiento de modelos YOLOv11
entrenados para detectar partes dañadas de vehículos.
"""

import os
import argparse
import subprocess
import glob
import json
import random
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import yaml

def find_latest_train_dir(base_dir='runs/detect'):
    """
    Encuentra el directorio de entrenamiento más reciente.
    
    Args:
        base_dir (str): Directorio base donde buscar
        
    Returns:
        str: Ruta al directorio más reciente o None si no se encuentra
    """
    # Buscar directorios de entrenamiento
    pattern = re.compile(r'^train(\d*)$')
    train_dirs = []
    
    # Verificar todos los directorios en base_dir
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                match = pattern.match(d)
                if match or d == 'train':
                    # Verificar que contenga el directorio weights
                    if os.path.exists(os.path.join(full_path, 'weights')):
                        train_dirs.append(full_path)
    
    if not train_dirs:
        return None
    
    # Ordenar por fecha de modificación (más reciente primero)
    return sorted(train_dirs, key=lambda x: os.path.getmtime(x), reverse=True)[0]

def get_model_path(model_arg):
    """
    Determina la ruta correcta del modelo basado en el argumento proporcionado.
    
    Args:
        model_arg (str): Argumento de modelo proporcionado
        
    Returns:
        str: Ruta completa al modelo
    """
    # Si es ruta completa, comprobar que existe
    if os.path.exists(model_arg):
        return model_arg
    
    # Si es nombre sin extensión, buscar .pt
    if not model_arg.endswith('.pt'):
        if os.path.exists(model_arg + '.pt'):
            return model_arg + '.pt'
    
    # Buscar en el directorio de entrenamiento más reciente
    latest_train = find_latest_train_dir()
    if latest_train:
        weights_dir = os.path.join(latest_train, 'weights')
        if os.path.exists(os.path.join(weights_dir, 'best.pt')):
            return os.path.join(weights_dir, 'best.pt')
    
    # Si no se encuentra, retornar el argumento original
    return model_arg

def get_data_path(data_arg):
    """
    Determina la ruta correcta del archivo data.yaml.
    
    Args:
        data_arg (str): Argumento de datos proporcionado
        
    Returns:
        str: Ruta completa al archivo data.yaml
    """
    # Si es ruta completa, comprobar que existe
    if data_arg and os.path.exists(data_arg):
        if os.path.isdir(data_arg):
            # Si es un directorio, buscar data.yaml dentro
            yaml_path = os.path.join(data_arg, 'data.yaml')
            if os.path.exists(yaml_path):
                return yaml_path
            else:
                print(f"⚠️ No se encontró data.yaml en el directorio {data_arg}")
        else:
            # Si es un archivo, devolverlo directamente
            return data_arg
    
    # Buscar dinámicamente directorios que comiencen con "data" y verificar si tienen data.yaml
    import glob
    data_dirs = glob.glob('data*/')  # Busca cualquier directorio que comience con "data"
    
    for data_dir in data_dirs:
        yaml_path = os.path.join(data_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            print(f"Utilizando archivo de datos: {yaml_path}")
            return yaml_path
    
    # Si no se encuentra en ninguna ubicación
    print("⚠️ ADVERTENCIA: No se pudo encontrar un archivo data.yaml válido.")
    print("Por favor, especifique explícitamente la ruta con --data")
    
    # Si se proporcionó un argumento, devolverlo aunque no exista
    if data_arg:
        return data_arg
    
    # Último recurso
    return 'data/data.yaml'

def list_available_models(base_dir='runs/detect'):
    """
    Lista todos los modelos disponibles y sus métricas.
    
    Args:
        base_dir (str): Directorio base donde buscar modelos
    """
    print("=== MODELOS DISPONIBLES ===")
    
    # Encabezado de la tabla
    header = f"{'Modelo':<20} {'mAP@0.5':<10} {'mAP@0.5-0.95':<12} {'Ruta'}"
    print(header)
    print("-" * 80)
    
    # Buscar todos los directorios de entrenamiento
    if not os.path.exists(base_dir):
        print(f"No se encontró el directorio {base_dir}")
        return
    
    models_info = []
    
    # Buscar en cada subdirectorio
    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            # Buscar archivo de resultados
            results_file = os.path.join(full_path, 'results.csv')
            model_file = os.path.join(full_path, 'weights', 'best.pt')
            
            if os.path.exists(model_file) and os.path.exists(results_file):
                # Leer métricas del archivo de resultados
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Asegurar que hay datos además del encabezado
                        last_line = lines[-1].strip().split(',')
                        try:
                            # Formato típico: epoch,precision,recall,mAP@0.5,mAP@0.5:0.95
                            map50 = float(last_line[3])
                            map50_95 = float(last_line[4])
                            
                            models_info.append({
                                'name': d,
                                'map50': map50,
                                'map50_95': map50_95,
                                'path': model_file
                            })
                        except (IndexError, ValueError):
                            # Si no se pueden leer las métricas, mostrar valores desconocidos
                            models_info.append({
                                'name': d,
                                'map50': float('nan'),
                                'map50_95': float('nan'),
                                'path': model_file
                            })
    
    # Ordenar por mAP@0.5-0.95 (descendente)
    models_info.sort(key=lambda x: x['map50_95'] if not math.isnan(x['map50_95']) else -1, reverse=True)
    
    # Mostrar tabla de modelos
    for model in models_info:
        map50 = f"{model['map50']:.4f}" if not math.isnan(model['map50']) else "???"
        map50_95 = f"{model['map50_95']:.4f}" if not math.isnan(model['map50_95']) else "???"
        print(f"{model['name']:<20} {map50:<10} {map50_95:<12} {model['path']}")

def generate_visualizations(model_path, data_path, num_samples=10, device='0'):
    """
    Genera visualizaciones de predicciones en imágenes aleatorias del conjunto de prueba.
    
    Args:
        model_path (str): Ruta al modelo
        data_path (str): Ruta al archivo data.yaml
        num_samples (int): Número de muestras a visualizar
        device (str): Dispositivo a utilizar
    """
    print(f"=== Generando visualizaciones de predicciones para {num_samples} imágenes ===")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el modelo {model_path}")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo de datos {data_path}")
        return
    
    # Si data_path es un directorio, intentar encontrar data.yaml
    if os.path.isdir(data_path):
        yaml_file = os.path.join(data_path, 'data.yaml')
        if os.path.exists(yaml_file):
            data_path = yaml_file
        else:
            print(f"Error: No se encuentra data.yaml en {data_path}")
            return
    
    # Cargar configuración del dataset
    try:
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error al cargar el archivo {data_path}: {e}")
        return
    
    # Construir ruta completa al directorio de imágenes de prueba
    base_dir = os.path.dirname(data_path)
    test_dir = data_config.get('test', './test/images')
    
    # Si la ruta es relativa, resolverla respecto al directorio de data.yaml
    if not os.path.isabs(test_dir):
        test_dir = os.path.join(base_dir, test_dir)
    
    # Listar todas las imágenes de prueba
    images = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        images.extend(glob.glob(os.path.join(test_dir, f'*{ext}')))
    
    if not images:
        print(f"Error: No se encontraron imágenes en {test_dir}")
        return
    
    # Seleccionar muestras aleatorias
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    # Unir rutas de imágenes para el comando
    img_paths = ','.join(sample_images)
    
    # Obtener nombre base del modelo para el directorio de salida
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if 'weights' in model_path else 'model'
    
    # Construir comando para predicciones
    cmd = f"yolo task=detect mode=predict model={model_path} source={img_paths} imgsz=640 device={device} save=True save_txt=True save_conf=True project=./predictions_{model_name} name=samples"
    
    print(f"Ejecutando: {cmd}")
    
    try:
        process = subprocess.run(cmd, shell=True, check=True)
        print(f"\nVisualizaciones guardadas en: ./predictions_{model_name}/samples")
    except subprocess.CalledProcessError as e:
        print(f"Error al generar visualizaciones: {e}")

def parse_metrics(results_dir):
    """
    Analiza las métricas de un directorio de resultados.
    
    Args:
        results_dir (str): Directorio de resultados
        
    Returns:
        dict: Diccionario con las métricas extraídas
    """
    metrics = {}
    
    # Buscar archivo JSON de métricas
    metrics_file = os.path.join(results_dir, 'metrics.json')
    if not os.path.exists(metrics_file):
        return metrics
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        # Extraer métricas relevantes
        if isinstance(data, dict):
            metrics['mAP@0.5'] = data.get('metrics/mAP50(B)', data.get('mAP50(B)', 0))
            metrics['mAP@0.5-0.95'] = data.get('metrics/mAP50-95(B)', data.get('mAP50-95(B)', 0))
            metrics['Precision'] = data.get('metrics/precision(B)', data.get('precision(B)', 0))
            metrics['Recall'] = data.get('metrics/recall(B)', data.get('recall(B)', 0))
    except Exception as e:
        print(f"Error al leer métricas: {e}")
    
    return metrics

def evaluate_model(model, data, batch=16, imgsz=640, device='0', split=None):
    """
    Evalúa un modelo en el conjunto de datos especificado.
    
    Args:
        model (str): Ruta al modelo
        data (str): Ruta al archivo data.yaml
        batch (int): Tamaño del batch
        imgsz (int): Tamaño de la imagen
        device (str): Dispositivo a utilizar
        split (str): Conjunto de datos a evaluar ('train', 'val', 'test')
        
    Returns:
        str: Directorio donde se guardaron los resultados
    """
    # Construir comando base
    cmd = ["yolo", "task=detect", "mode=val", 
          f"model={model}", f"data={data}", 
          f"batch={batch}", f"imgsz={imgsz}", 
          f"device={device}", 
          "save_json=True", "save_txt=True", 
          "save_conf=True", "plots=True"]
    
    # Añadir split y nombre si es necesario
    if split:
        cmd.append(f"split={split}")
        cmd.append(f"name={split}_results")
    
    # Ejecutar comando
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    try:
        process = subprocess.run(cmd_str, shell=True, check=True)
        
        # Determinar directorio de resultados
        if split and split != 'val':
            result_dir = f"runs/detect/{split}_results"
        else:
            result_dir = "runs/detect/val"
        
        return result_dir
    except subprocess.CalledProcessError as e:
        print(f"Error durante la evaluación: {e}")
        return None

def continue_training(model, data, continue_epochs=20, batch=16, imgsz=640, device='0'):
    """
    Continúa el entrenamiento de un modelo existente.
    
    Args:
        model (str): Ruta al modelo
        data (str): Ruta al archivo data.yaml
        continue_epochs (int): Número de épocas adicionales
        batch (int): Tamaño del batch
        imgsz (int): Tamaño de la imagen
        device (str): Dispositivo a utilizar
        
    Returns:
        bool: True si el entrenamiento se completó con éxito
    """
    # Verificar que el modelo existe
    if not os.path.exists(model):
        print(f"Error: No se encuentra el modelo {model}")
        return False
    
    # Obtener nombre base del modelo para el directorio de continuación
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model))) if 'weights' in model else 'model'
    name = f"continue_{model_name}"
    
    print(f"=== Continuando entrenamiento desde {model} por {continue_epochs} épocas adicionales ===")
    
    # Construir comando
    cmd = ["yolo", "task=detect", "mode=train", 
          f"model={model}", f"data={data}", 
          f"epochs={continue_epochs}", f"batch={batch}", 
          f"imgsz={imgsz}", f"device={device}", 
          "save=True", "val=True", "plots=True",
          f"name={name}"]
    
    # Ejecutar comando
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    try:
        process = subprocess.run(cmd_str, shell=True, check=True)
        
        # Verificar que el modelo se guardó correctamente
        new_model = f"./runs/detect/{name}/weights/best.pt"
        if os.path.exists(new_model):
            print(f"\nEntrenamiento adicional completado. Mejor modelo guardado en: {new_model}")
            print(f"\nEvaluar el modelo mejorado:")
            print(f"python evaluateModel.py --model {new_model} --data {data}")
            return True
        else:
            print(f"Advertencia: No se encontró el modelo entrenado en {new_model}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
        return False

def plot_comparison(val_metrics, test_metrics, output_file='metrics_comparison.png'):
    """
    Genera una gráfica comparativa de métricas entre validación y prueba.
    
    Args:
        val_metrics (dict): Métricas de validación
        test_metrics (dict): Métricas de prueba
        output_file (str): Ruta del archivo de salida
    """
    if not val_metrics or not test_metrics:
        print("No hay suficientes métricas para generar la comparación")
        return
    
    # Métricas a comparar y sus etiquetas
    metrics = ['mAP@0.5', 'mAP@0.5-0.95', 'Precision', 'Recall']
    labels = ['mAP@0.5', 'mAP@0.5-0.95', 'Precisión', 'Recall']
    
    val_values = [val_metrics.get(m, 0) for m in metrics]
    test_values = [test_metrics.get(m, 0) for m in metrics]
    
    # Configurar gráfica
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Barras
    rects1 = ax.bar(x - width/2, val_values, width, label='Validación')
    rects2 = ax.bar(x + width/2, test_values, width, label='Prueba')
    
    # Añadir etiquetas y título
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas entre Validación y Prueba')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Añadir texto con valores
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Guardar gráfica
    plt.savefig(output_file)
    print(f"Gráfica comparativa guardada como: {output_file}")
    
    return fig

def main():
    """Función principal que procesa los argumentos y ejecuta la evaluación."""
    parser = argparse.ArgumentParser(description='Evaluar y analizar modelos YOLOv11 para detección de partes dañadas')
    parser.add_argument('--model', type=str, help='Ruta al modelo a evaluar')
    parser.add_argument('--data', type=str, help='Ruta al archivo data.yaml o directorio que lo contiene')
    parser.add_argument('--batch', type=int, default=16, help='Tamaño del batch')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de la imagen')
    parser.add_argument('--device', type=str, default='0', help='Dispositivo (0, 0,1, cpu)')
    parser.add_argument('--samples', type=int, default=10, help='Número de muestras para visualización')
    parser.add_argument('--list-models', action='store_true', help='Listar todos los modelos disponibles')
    parser.add_argument('--continue-epochs', type=int, help='Continuar entrenamiento por N épocas adicionales')
    
    args = parser.parse_args()
    
    # Listar modelos disponibles
    if args.list_models:
        list_available_models()
        return
    
    # Obtener rutas de modelo y datos
    model_path = get_model_path(args.model) if args.model else None
    data_path = get_data_path(args.data)
    
    # Verificar archivo de datos
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo de datos {data_path}")
        print("Directorios comunes donde podría encontrarse:")
        for dir_name in ['data_merged', 'data_ready', 'cardd_ready', 'data_combined', 'data_cardd', 'data']:
            if os.path.exists(dir_name):
                print(f" - {dir_name}")
        return
    
    # Si no se especificó un modelo, buscar el más reciente
    if not model_path:
        latest_train = find_latest_train_dir()
        if latest_train:
            model_path = os.path.join(latest_train, 'weights', 'best.pt')
            if not os.path.exists(model_path):
                print(f"Error: No se encuentra un modelo entrenado en {latest_train}")
                return
        else:
            print("Error: No se ha especificado un modelo y no se encuentra ningún entrenamiento previo.")
            return
    
    print("=== EVALUACIÓN Y ANÁLISIS DE MODELO ===")
    print(f"Modelo: {model_path}")
    print(f"Archivo de datos: {data_path}")
    print("-" * 50)
    
    # Continuar entrenamiento si se solicitó
    if args.continue_epochs:
        continue_training(
            model=model_path,
            data=data_path,
            continue_epochs=args.continue_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        return
    
    # Evaluar en conjunto de validación
    print(f"\n=== Evaluando {os.path.basename(model_path)} en el conjunto de VALIDACIÓN ===")
    val_dir = evaluate_model(
        model=model_path,
        data=data_path,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Evaluar en conjunto de prueba
    print(f"\n=== Evaluando {os.path.basename(model_path)} en el conjunto de PRUEBA ===")
    test_dir = evaluate_model(
        model=model_path,
        data=data_path,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        split='test'
    )
    
    # Generar visualizaciones
    generate_visualizations(
        model_path=model_path,
        data_path=data_path,
        num_samples=args.samples,
        device=args.device
    )
    
    # Extraer y mostrar métricas
    val_metrics = parse_metrics(val_dir) if val_dir else {}
    test_metrics = parse_metrics(test_dir) if test_dir else {}
    
    print("\n=== ANÁLISIS DE RESULTADOS ===\n")
    
    if val_metrics:
        print("Resultados en conjunto de VALIDACIÓN:")
        for metric, value in val_metrics.items():
            print(f"- {metric}: {value:.4f}")
        print()
    
    if test_metrics:
        print("Resultados en conjunto de PRUEBA:")
        for metric, value in test_metrics.items():
            print(f"- {metric}: {value:.4f}")
        print()
    
    # Generar gráfica comparativa
    if val_metrics and test_metrics:
        plot_comparison(val_metrics, test_metrics)
    
    # Resumen final
    print("=== PROCESO COMPLETO ===")
    if val_dir:
        print(f"Resultados de validación: {val_dir}")
    if test_dir:
        print(f"Resultados de prueba: {test_dir}")
    
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path))) if 'weights' in model_path else 'model'
    print(f"Visualizaciones: ./predictions_{model_name}/samples")

if __name__ == "__main__":
    main()
