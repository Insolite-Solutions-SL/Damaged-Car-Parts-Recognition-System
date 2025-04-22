# Sistema de Entrenamiento para Detección de Daños en Vehículos (YOLOv11)

Este proyecto proporciona un entorno completo para el entrenamiento y evaluación de modelos YOLOv11 para la detección de daños en vehículos. El sistema está optimizado para el desarrollo, entrenamiento y análisis de modelos, sin componentes innecesarios de despliegue.

## Características principales

- Entrenamiento con modelos YOLOv11 (n, s, m, l, x) para detección precisa de daños
- Preparación automática de datos y combinación de múltiples fuentes
- Soporte para dataset CarDD (4,000+ imágenes especializadas en daños)
- Herramientas para conversión entre formatos (COCO a YOLO)
- Análisis detallado de resultados con métricas y visualizaciones
- Flujo de trabajo unificado para automatizar todo el proceso

## Clases de daños soportadas

El sistema detecta 8 tipos de daños en vehículos:
- damaged door (puerta dañada)
- damaged window (ventana dañada)
- damaged headlight (faro dañado)
- damaged mirror (espejo dañado)
- dent (abolladura)
- damaged hood (capó dañado)
- damaged bumper (parachoques dañado)
- damaged wind shield (parabrisas dañado)

## Estructura del proyecto

```
Damaged-Car-Parts-Recognition-System/
├── combineDatasets.py        # Combinar múltiples datasets
├── convertCarDD.py           # Convertir dataset CarDD a formato YOLO
├── damageDetectionWorkflow.py # Flujo de trabajo unificado
├── prepareLocalData.py       # Preparar estructura de datos local
├── trainYolov11s.py          # Entrenar modelos YOLOv11
├── evaluateModel.py          # Evaluar y analizar modelos
├── best.pt                   # Modelo de referencia entrenado (PyTorch)
├── data/                     # Datos de entrenamiento originales
├── data_cardd/               # Datos convertidos de CarDD
└── CarDD_release/            # Dataset CarDD original
```

## Guía rápida de uso

### 1. Preparación de datos

```bash
# Preparar estructura local mínima
python3 prepareLocalData.py

# O preparar con datos de muestra
python3 prepareLocalData.py --import-from /ruta/a/datos/existentes

# Convertir dataset CarDD a formato YOLO (solo necesario una vez)
python3 convertCarDD.py
```

### 2. Entrenamiento

```bash
# Entrenar con dataset original
python3 trainYolov11s.py --data $(pwd)/data/data.yaml --epochs 100 --batch 16 --device 0

# Entrenar con dataset CarDD convertido
python3 trainYolov11s.py --data $(pwd)/data_cardd/data.yaml --epochs 100 --batch 16 --device 0

# Combinar datasets y entrenar
python3 combineDatasets.py --sources data data_cardd --output data_combined
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --epochs 100 --batch 16 --device 0
```

### 3. Evaluación

```bash
# Evaluar modelo entrenado
python3 evaluateModel.py --model runs/detect/train/weights/best.pt

# Listar todos los modelos disponibles
python3 evaluateModel.py --list-models

# Continuar entrenamiento
python3 evaluateModel.py --model runs/detect/train/weights/best.pt --continue-epochs 20
```

### 4. Flujo de trabajo completo

```bash
# Ver todas las opciones
python3 damageDetectionWorkflow.py --help

# Ejecutar flujo completo (local, CPU)
python3 damageDetectionWorkflow.py workflow --local --device cpu

# Flujo completo con GPU
python3 damageDetectionWorkflow.py workflow --sources data data_cardd --epochs 100 --device 0
```

## Requisitos

- Python 3.8 o superior
- PyTorch 1.7 o superior
- Ultralytics YOLO (v8.0.0+)
- Matplotlib, NumPy, PyYAML
