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

## Datasets y proceso de unificación

Para mejorar la precisión del sistema, se han integrado tres datasets especializados en detección de daños en vehículos con un total de **10,959 imágenes**. Este proceso ha requerido un sofisticado sistema de remapeo que convierte más de 50 clases originales en un conjunto unificado de 8 categorías estándar.

### Datasets originales

| Dataset | Fuente | Imágenes | Clases originales |
|---------|--------|----------|-------------------|
| **Surya Remanan** | [GitHub](https://github.com/suryaremanan/Damaged-Car-parts-prediction-using-YOLOv8) | 400 | 8 clases: damaged door, damaged window, damaged headlight, damaged mirror, dent, damaged hood, damaged bumper, damaged windshield |
| **RoboFlow** | [Car Damage Detection](https://universe.roboflow.com/capstone-nh0nc/car-damage-detection-t0g92) | 6,559 | 46 clases: Bodypanel-Dent, Front-Windscreen-Damage, door, headlight, back-bumper, front-bumper, bonnet-dent, etc. |
| **CarDD USTC** | [CarDD-USTC](https://cardd-ustc.github.io/) | 4,000 | 6 clases: dent, scratch, crack, glass shatter, lamp broken, tire flat |

**Distribución total**: Train (7,658 - 69.88%), Validation (2,222 - 20.28%), Test (1,079 - 9.84%)

### Clases unificadas (target)

Todas las clases originales han sido remapeadas a este conjunto unificado de 8 categorías:

- **damaged door** (puerta dañada): Incluye door, damaged-door, collapse, etc.
- **damaged window** (ventana dañada): Incluye window, damaged-window, glass shatter, shattered_glass, crack, etc.
- **damaged headlight** (faro dañado): Incluye headlight, Headlight-Damage, damaged-head-light, lamp broken, light_damage, etc.
- **damaged mirror** (espejo dañado): Incluye mirror, Sidemirror-Damage, mirror_damage, etc.
- **dent** (abolladura): Incluye dent, Bodypanel-Dent, scratch, depression, bonnet-dent, etc.
- **damaged hood** (capó dañado): Incluye hood, damaged-hood, hood_damage, trunk, etc.
- **damaged bumper** (parachoques dañado): Incluye back-bumper, front-bumper, bumper_damage, tire flat, etc.
- **damaged wind shield** (parabrisas dañado): Incluye damaged-windscreen, Front-Windscreen-Damage, windshield_damage, etc.

### Proceso de unificación

El sistema utiliza un mapa de equivalencias detallado en `configs/extended_mapping.yaml` que:

1. **Estandariza nomenclaturas** entre datasets
2. **Gestiona variaciones de formato** (guiones, guiones bajos, espacios)
3. **Preserva la semántica** agrupando daños similares

Este enfoque multiplica por 27 veces la cantidad de ejemplos disponibles (de 400 a 10,959), manteniendo una consistencia en la distribución train/validation/test (~70/20/10) para crear un modelo más robusto y generalizable.

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

### Combinación de Datasets

El script `datasetProcessorTool.py` también permite combinar múltiples datasets en uno solo, lo que es especialmente útil para unificar datos de diferentes fuentes.

#### Fusión de Datasets con datasetProcessorTool.py:

```bash
python datasetProcessorTool.py --merge-datasets --input-dirs dir1 dir2 dir3 --output-dir data_merged --use-remapping
```

Opciones para la fusión de datasets:
- `--merge-datasets`: Activa el modo de fusión de datasets
- `--input-dirs`: Lista de directorios con datasets a combinar (formato YOLO)
- `--output-dir`: Directorio donde se guardará el dataset combinado
- `--use-remapping`: Aplica remapeo de clases durante la fusión (recomendado)
- `--train-ratio`, `--val-ratio`, `--test-ratio`: Proporciones para redistribuir las imágenes (por defecto: 0.7/0.2/0.1)

Ejemplo de flujo de trabajo para combinar datasets:

1. **Preparar cada dataset por separado**:
   ```bash
   python datasetProcessorTool.py --input-dir dataset1 --output-dir dataset1_processed --use-remapping
   python datasetProcessorTool.py --input-dir dataset2 --output-dir dataset2_processed --use-remapping
   ```

2. **Combinar los datasets procesados**:
   ```bash
   python datasetProcessorTool.py --merge-datasets --input-dirs dataset1_processed dataset2_processed --output-dir data_merged
   ```

Este enfoque garantiza que todos los datasets tengan una estructura y nomenclatura de clases consistente antes de fusionarlos.

## Entrenamiento de Modelos

Para entrenar un modelo YOLOv11 con el dataset procesado, use el script `trainYolov11s.py`:

```bash
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --epochs 100 --batch 16 --device 0
```

Este script utiliza la CLI de Ultralytics YOLOv11 para entrenar y evaluar automáticamente el modelo. Al finalizar el entrenamiento, también realiza una validación y genera gráficos de rendimiento.

### Opciones de entrenamiento:

- `--data`: Ruta al archivo data.yaml del dataset procesado
- `--model`: Modelo base a utilizar (por defecto: yolo11s). Opciones: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
- `--epochs`: Número de épocas para el entrenamiento (predeterminado: 100)
- `--batch`: Tamaño del lote de entrenamiento (predeterminado: 16)
- `--imgsz`: Tamaño de imagen para entrenamiento (predeterminado: 640)
- `--device`: Dispositivo para entrenamiento (0 para GPU, cpu para CPU)
- `--workers`: Número de procesos para carga de datos (predeterminado: 8)
- `--name`: Nombre del experimento (por defecto se genera automáticamente)
- `--project`: Directorio para guardar los resultados (predeterminado: runs/detect)

### Ejemplos prácticos:

```bash
# Entrenamiento básico con GPU
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --epochs 100 --batch 16 --device 0

# Entrenamiento con el modelo más ligero
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --model yolo11n --batch 32

# Entrenamiento con mayor resolución de imagen
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --imgsz 800 --batch 8

# Entrenamiento en CPU (más lento)
python3 trainYolov11s.py --data $(pwd)/data_combined/data.yaml --device cpu
```

Los resultados del entrenamiento, incluyendo el mejor modelo, las gráficas de pérdida y las métricas de evaluación, se guardarán en la carpeta especificada por `--project` y `--name`.

## Evaluación de Modelos

La herramienta `evaluateModel.py` permite realizar análisis detallados de los modelos entrenados, ofreciendo múltiples opciones para evaluación, visualización y optimización:

```bash
python evaluateModel.py --model runs/train/exp/weights/best.pt --data data_merged/data.yaml
```

### Opciones de evaluación

| Opción | Descripción |
|--------|-------------|
| `--model PATH` | Ruta al modelo a evaluar |
| `--data PATH` | Ruta al archivo data.yaml o directorio que lo contiene |
| `--batch INT` | Tamaño del batch (default: 16) |
| `--imgsz INT` | Tamaño de la imagen (default: 640) |
| `--device STR` | Dispositivo (0, 0,1, cpu) (default: 0) |
| `--samples INT` | Número de muestras para visualización (default: 10) |
| `--list-models` | Listar todos los modelos disponibles |
| `--continue-epochs INT` | Continuar entrenamiento por N épocas adicionales |
| `--benchmark` | Realizar benchmark de velocidad del modelo |
| `--analyze-classes` | Analizar rendimiento por clase |
| `--confusion-matrix` | Generar matriz de confusión |
| `--analyze-errors` | Analizar falsos positivos/negativos |
| `--compare-models MODEL1 [MODEL2 ...]` | Comparar varios modelos (proporcionar múltiples rutas) |
| `--export {onnx,torchscript,openvino}` | Exportar modelo a formato optimizado |
| `--export-dir PATH` | Directorio para modelos exportados (default: ./exported_models) |
| `--full-report` | Generar informe completo con todas las métricas |

### Ejemplos de uso

```bash
# Evaluación básica del modelo
python evaluateModel.py --model runs/train/exp/weights/best.pt --data data_merged/data.yaml

# Generar informe completo con todas las métricas y visualizaciones
python evaluateModel.py --model runs/train/exp/weights/best.pt --data data_merged/data.yaml --full-report

# Benchmark de velocidad del modelo
python evaluateModel.py --model runs/train/exp/weights/best.pt --benchmark

# Análisis de rendimiento por clase
python evaluateModel.py --model runs/train/exp/weights/best.pt --analyze-classes

# Comparar múltiples modelos entrenados
python evaluateModel.py --compare-models runs/train/exp1/weights/best.pt runs/train/exp2/weights/best.pt --data data_merged/data.yaml

# Exportar modelo a formato ONNX
python evaluateModel.py --model runs/train/exp/weights/best.pt --export onnx
```

> **Nota**: La funcionalidad de análisis por clase (`--analyze-classes`) está actualmente en desarrollo y puede no proporcionar resultados precisos para todos los modelos. Se recomienda usar esta opción con precaución y verificar los resultados manualmente.

### Consejos para la evaluación

- Verifique los falsos positivos más comunes para identificar áreas de mejora
- Compare siempre el rendimiento entre validación y prueba para detectar overfitting
- Para modelos destinados a dispositivos con recursos limitados, priorice el benchmark de velocidad
- Los modelos con mAP50 > 0.85 son generalmente adecuados para implementación en producción

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir los cambios propuestos o envíe un pull request directamente.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.
