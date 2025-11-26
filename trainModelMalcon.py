#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar un modelo YOLOv11 para la detección de partes dañadas de vehículos.
Este script configura y ejecuta el entrenamiento con los parámetros especificados.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import datetime
import yaml

def get_model_path(model_name):
    """
    Obtiene la ruta del modelo pre-entrenado o el archivo de configuración.
    
    Args:
        model_name (str): Nombre del modelo (p.ej., yolo11s, yolo11m, etc.)
    
    Returns:
        str: Ruta al modelo o archivo de configuración
    """
    # Si el modelo ya incluye extensión, usar directamente
    if model_name.endswith('.pt') or model_name.endswith('.yaml'):
        if os.path.exists(model_name):
            return model_name
    
    # Buscar en el directorio actual
    for ext in ['.pt', '.yaml']:
        if os.path.exists(model_name + ext):
            return model_name + ext
    
    # Si no se encuentra, asumir que es un modelo de Ultralytics
    return model_name

def train_yolov11(
    model='yolo11s',
    data='data/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device='0',
    workers=8,
    name=None,
    project='runs/detect',
    hyps='hyps/Malcon_config.yaml'
):
    """
    Entrena un modelo YOLOv11 para la detección de partes dañadas de vehículos.
    
    Args:
        model (str): Modelo a utilizar (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
        data (str): Ruta al archivo data.yaml
        epochs (int): Número de épocas de entrenamiento
        batch (int): Tamaño del batch
        imgsz (int): Tamaño de la imagen
        device (str): Dispositivo a utilizar ('0', '0,1', 'cpu')
        workers (int): Número de workers para la carga de datos
        name (str): Nombre para el directorio de resultados
        project (str): Directorio de proyecto para guardar resultados
    """
    # Normalizar la ruta del archivo de datos
    data_path = os.path.abspath(data)
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo de datos {data_path}")
        return False
    
    # Obtener ruta del modelo
    model_path = get_model_path(model)
    
    # Configurar nombre de ejecución si no se especificó
    if name is None:
        name = f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Construir comando de entrenamiento
    cmd = [
        "yolo", "task=detect", "mode=train",
        f"model={model_path}",
        #f"model=/home/insolitetao/Desktop/Damaged-Car-Parts-Recognition-System/runs/detect/train_20250428_105915/weights/best.pt",
        f"data={data_path}",
        f"epochs={epochs}",
        f"batch={batch}",
        #f"imgsz={imgsz}",
        f"device={device}",
        f"workers={workers}",
        f"name={name}",
        f"project={project}",
        #f"optimizer=AdamW",
        #f"patience=60",
        "val=True",
        "plots=True",
        "save=True",
        "save_period=20",
        # "pretrained=True",
        "single_cls=True", # Just identify damages
        
    ]

    with open(hyps, 'r') as f:
        data = yaml.safe_load(f)

        for key,value in data.items():
            cmd.append(f"{key}={value}")
    
    print("Iniciando entrenamiento de YOLOv11 para detección de partes dañadas de vehículos")
    print(f"Archivo de datos: {data_path}")
    print(f"Modelo: {model_path}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Tamaño de imagen: {imgsz}x{imgsz}")
    print(f"Dispositivo: {device}")
    print("-" * 50)
    
    # Ejecutar comando
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    try:
        process = subprocess.run(cmd_str, shell=True, check=True)
        print("\nEntrenamiento completado.")
        
        # Obtener ruta del modelo entrenado
        trained_model = f"./{project}/{name}/weights/best.pt"
        if os.path.exists(trained_model):
            print(f"Mejor modelo guardado en: {trained_model}")
            
            # Ejecutar evaluación del modelo
            print("\nEvaluando modelo: ", end="")
            eval_cmd = f"yolo task=detect mode=val model={trained_model} data={data_path} save_json=True plots=True"
            print(eval_cmd)
            subprocess.run(eval_cmd, shell=True, check=True)
            
            print("\nProceso completo. Revisa los resultados en la carpeta '{project}'.")
            return True
        else:
            print(f"Advertencia: No se encontró el modelo entrenado en {trained_model}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False

def main():
    """Función principal que procesa los argumentos y ejecuta el entrenamiento."""
    parser = argparse.ArgumentParser(description='Entrenar YOLOv11 para detección de partes dañadas de vehículos')
    parser.add_argument('--model', type=str, default='yolo11s', help='Modelo a utilizar (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='Ruta al archivo data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch', type=int, default=16, help='Tamaño del batch')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de la imagen')
    parser.add_argument('--device', type=str, default='0', help='Dispositivo (0, 0,1, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Número de workers para carga de datos')
    parser.add_argument('--name', type=str, default=None, help='Nombre para el directorio de resultados')
    parser.add_argument('--project', type=str, default='runs/detect', help='Directorio de proyecto para guardar resultados')
    parser.add_argument('--hypsyaml', type=str, default='hyps/Malcon_config.yaml', help='Configuracion de hiperparámetros')
    
    args = parser.parse_args()
    
    # Ejecutar entrenamiento
    success = train_yolov11(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        hyps=args.hypsyaml
    )
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
