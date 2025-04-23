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
├── configs/                  # Directorio de archivos de configuración
│   └── extended_mapping.yaml # Mapeo extenso de clases para todos los datasets
├── datasetProcessorTool.py   # Herramienta avanzada para procesamiento de datasets
├── evaluateModel.py          # Evaluar modelos entrenados
├── trainYolov11s.py          # Entrenar modelos YOLOv11
├── best.pt                   # Modelo de referencia entrenado (PyTorch)
├── data/                     # Datos de entrenamiento originales
├── data_*/                   # Datos procesados con diferentes configuraciones
└── results/                  # Resultados de evaluación de modelos
```

## Preparación del Entorno

### Requisitos

- Python 3.8 o superior
- PyTorch 1.8 o superior
- Ultralytics (YOLOv11)
- Otras dependencias en `requirements.txt`

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/Damaged-Car-Parts-Recognition-System.git
cd Damaged-Car-Parts-Recognition-System
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar datasets (opcional):
```bash
# Descargar datasets desde fuentes externas si es necesario
python downloadDatasets.py
```

## Procesamiento de Datasets

La herramienta `datasetProcessorTool.py` facilita el procesamiento de datasets de diferentes fuentes y formatos, permitiendo:

1. **Convertir datasets de formato COCO a formato YOLO**
2. **Preparar datasets con estructura no estándar a formato YOLO**
3. **Remapear y unificar clases entre diferentes datasets**

### Formatos de datos soportados

El sistema soporta múltiples formatos de datos de entrada con detección automática:

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
   - Estructura: Imágenes y etiquetas en el mismo directorio sin subdirectorios

### Procesamiento Unificado de Datasets

Todo el procesamiento de datos se realiza a través de un único script con detección automática del formato de entrada:

```bash
python datasetProcessorTool.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR [opciones]
```

#### Opciones Principales:

- `--input-dir`: Directorio con el dataset original
- `--output-dir`: Directorio donde se guardará el dataset procesado
- `--config`: Archivo de configuración YAML con el mapeo de clases (por defecto: configs/extended_mapping.yaml)
- `--input-format`: Formato del dataset de entrada (auto, coco, yolo, raw)
- `--use-remapping`: Aplicar remapeo de clases según el archivo de configuración
- `--sample-size`: Número de imágenes a incluir por split (0=todas)

#### Ejemplos de Uso:

1. **Procesar dataset COCO con remapeo** (como el dataset chino):
```bash
python datasetProcessorTool.py --input-dir data_cardd_chino_merge_classes --output-dir data_cardd_chino_processed --use-remapping
```

2. **Procesar dataset YOLO que solo necesita remapeo**:
```bash
python datasetProcessorTool.py --input-dir data_cardd_roboflow_need_merge_classes --output-dir data_cardd_roboflow_processed --use-remapping
```

3. **Procesar dataset con estructura no estándar** (como el dataset raw original):
```bash
python datasetProcessorTool.py --input-dir data --output-dir data_ready --input-format raw
```

### Mapeo de Clases

El mapeo de clases se define en el archivo `configs/extended_mapping.yaml` y permite unificar las diferentes nomenclaturas de los datasets a un conjunto común de clases:

```yaml
target_classes:
  - damaged door
  - damaged window
  # ...otras clases

class_mapping:
  crack: damaged window
  collapse: damaged door
  # ...otros mapeos
```

## Entrenamiento de Modelos

Para entrenar un modelo con un dataset procesado:

```bash
python trainModel.py --data data_merged/data.yaml --model yolov11-s.pt --epochs 100 --batch-size 16
```

Opciones principales:
- `--data`: Ruta al archivo data.yaml del dataset procesado
- `--model`: Modelo base a utilizar (s, m, l, x)
- `--epochs`: Número de épocas de entrenamiento
- `--batch-size`: Tamaño del batch

## Evaluación de Modelos

Para evaluar un modelo entrenado:

```bash
python evaluateModel.py --model runs/train/exp/weights/best.pt --data data_merged/data.yaml
```

Para generar un reporte completo con análisis por clase y matriz de confusión:

```bash
python evaluateModel.py --model runs/train/exp/weights/best.pt --data data_merged/data.yaml --full-report
```

### Consejos para la evaluación

- Verifique los falsos positivos más comunes para identificar áreas de mejora
- Compare siempre el rendimiento entre validación y prueba para detectar overfitting
- Para modelos destinados a dispositivos con recursos limitados, priorice el benchmark de velocidad
- Los modelos con mAP50 > 0.85 son generalmente adecuados para implementación en producción

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir los cambios propuestos o envíe un pull request directamente.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.
