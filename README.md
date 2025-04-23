# Sistema de Entrenamiento para Detección de Daños en Vehículos (YOLOv11)

Este proyecto proporciona un entorno completo para el entrenamiento y evaluación de modelos YOLOv11 para la detección de daños en vehículos. El sistema está optimizado para el desarrollo, entrenamiento y análisis de modelos, sin componentes innecesarios de despliegue.

## Características principales

- Entrenamiento con modelos YOLOv11 (n, s, m, l, x) para detección precisa de daños
- Preparación automática de datos y combinación de múltiples fuentes
- Soporte para diversos formatos de datos (YOLO estándar, invertido, COCO, etc.)
- Conversión automática entre formatos de datos (COCO a YOLO, estructuras invertidas)
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
├── CarDD_COCO/               # Dataset CarDD en formato COCO
├── data_ready/               # Dataset /data preparado
├── cardd_ready/              # Dataset CarDD_COCO preparado
└── data_merged/              # Datasets combinados
```

## Formatos de datos soportados

El sistema ahora soporta múltiples formatos de datos de entrada:

1. **Formato YOLO estándar**: 
   - Estructura: `/train/images/`, `/train/labels/`, `/valid/images/`, etc.
   - Ejemplo: Carpetas organizadas por split (train, valid, test)

2. **Formato invertido**: 
   - Estructura: `/images/train/`, `/labels/train/`
   - Ejemplo: El dataset `/data` original

3. **Formato COCO**: 
   - Estructura: `/annotations/instances_*.json`, `/train*/`, `/val*/`, `/test*/`
   - Ejemplos: Dataset `CarDD_COCO` (con cualquier sufijo, como 2017, 2023, etc.)

4. **Formato plano**: 
   - Estructura: `/images/`, `/labels/` sin subdirectorios
   - O imágenes y etiquetas en el mismo directorio

## Guía rápida de uso

### 1. Preparación de datos

```bash
# Preparar estructura local mínima
python3 prepareLocalData.py

# Preparar dataset con estructura invertida (como /data)
python3 prepareLocalData.py --import-from data --data-dir data_ready

# Preparar dataset en formato COCO (como CarDD_COCO)
python3 prepareLocalData.py --import-from CarDD_COCO --data-dir cardd_ready

# Control del tamaño de muestra (0 = usar todo el dataset)
python3 prepareLocalData.py --import-from data --data-dir data_sample --sample-size 100
```

### 2. Combinación de datasets

```bash
# Combinar múltiples datasets
python3 combineDatasets.py --sources data_ready cardd_ready --output data_merged

# Combinar 3 o más datasets
python3 combineDatasets.py --sources data_ready cardd_ready data_cardd --output data_mega
```

### 3. Entrenamiento

```bash
# Entrenar con dataset original
python3 trainYolov11s.py --data $(pwd)/data_ready/data.yaml --epochs 100 --batch 16 --device 0

# Entrenar con dataset CarDD convertido
python3 trainYolov11s.py --data $(pwd)/cardd_ready/data.yaml --epochs 100 --batch 16 --device 0

# Entrenar con dataset combinado (recomendado)
python3 trainYolov11s.py --data $(pwd)/data_merged/data.yaml --epochs 100 --batch 16 --device 0
```

### 4. Evaluación

```bash
# Evaluar modelo entrenado
python3 evaluateModel.py --model runs/detect/train/weights/best.pt

# Listar todos los modelos disponibles
python3 evaluateModel.py --list-models

# Continuar entrenamiento
python3 evaluateModel.py --model runs/detect/train/weights/best.pt --continue-epochs 20
```

### 5. Flujo de trabajo completo

```bash
# Ver todas las opciones
python3 damageDetectionWorkflow.py --help

# Ejecutar flujo completo (local, CPU)
python3 damageDetectionWorkflow.py workflow --local --device cpu

# Flujo completo con GPU
python3 damageDetectionWorkflow.py workflow --sources data_ready cardd_ready --epochs 100 --device 0
```

## Estadísticas de conjuntos de datos

| Dataset      | Train | Valid | Test | Total | Formato            |
|--------------|-------|-------|------|-------|-------------------|
| data         | 362   | 87    | 36   | 485   | Invertido         |
| CarDD_COCO   | 2816  | 810   | 374  | 4000  | COCO              |
| data_merged  | 3588  | 672   | 225  | 4485  | YOLO estándar     |

## Requisitos

- Python 3.8 o superior
- PyTorch 1.7 o superior
- Ultralytics YOLO (v8.0.0+)
- Matplotlib, NumPy, PyYAML

## Resolución de problemas

### Estructura de datos incorrecta

Si `prepareLocalData.py` muestra advertencias sobre no encontrar imágenes con etiquetas correspondientes:

1. Verifica que tu directorio de origen tenga uno de los formatos soportados
2. Para estructuras COCO, asegúrate de que las carpetas de imágenes comiencen con 'train', 'val' o 'test'
3. Para formatos invertidos, verifica que existan `/images/train/` y `/labels/train/`

### Consejos de optimización

- Para datasets grandes, es recomendable usar una GPU para el entrenamiento
- Para transferencia de aprendizaje rápida, use `--epochs 20 --patience 10`
- Para detección de alta precisión, use `--epochs 100 --patience 25`
