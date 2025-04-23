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
├── configs/                  # Directorio de archivos de configuración
│   └── extended_mapping.yaml # Mapeo extenso de clases para todos los datasets
├── datasetProcessorTool.py   # Herramienta avanzada para procesamiento de datasets
├── damageDetectionWorkflow.py # Flujo de trabajo unificado
├── prepareLocalData.py       # Preparar estructura de datos local
├── trainYolov11s.py          # Entrenar modelos YOLOv11
├── evaluateModel.py          # Evaluar modelos entrenados
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

## Procesamiento avanzado de datasets

La herramienta `datasetProcessorTool.py` proporciona todas las funcionalidades necesarias para la preparación de datasets:

```bash
# Convertir un dataset COCO a formato YOLO con mapeo personalizado de clases
python datasetProcessorTool.py --coco-dir dataset_original --output-dir dataset_procesado --config configs/extended_mapping.yaml

# Convertir y fusionar múltiples datasets
python datasetProcessorTool.py --coco-dirs dataset1 dataset2 dataset3 --output-dir datasets_combinados --merge

# Procesar y remapear un dataset existente en formato YOLO
python datasetProcessorTool.py --yolo-dir dataset_yolo_existente --output-dir dataset_remapeado --use-remapping --config configs/extended_mapping.yaml
```

#### Características principales:

- **Conversión flexible**: Maneja múltiples estructuras de datasets COCO, adaptándose automáticamente.
- **Mapeo de clases**: Unifica diferentes nomenclaturas de etiquetas bajo un esquema consistente.
- **Fusión de datasets**: Combina varios datasets en uno solo con distribuciones personalizables.
- **Remapeo de clases**: Aplica transformaciones de clases incluso a datasets ya en formato YOLO.

#### Configuración para el mapeo de clases:

La herramienta utiliza el archivo `configs/extended_mapping.yaml` que contiene el mapeo completo para unificar más de 50 variantes de etiquetas en las 8 clases estándar:

```bash
# Ejemplo: Remapear un dataset con múltiples clases a las 8 clases estándar
python datasetProcessorTool.py --yolo-dir data_con_muchas_clases --output-dir data_estandarizado --use-remapping --config configs/extended_mapping.yaml
```

#### Opciones avanzadas:

```bash
# Incluir clases adicionales no definidas en el mapeo por defecto
python datasetProcessorTool.py --coco-dir dataset_original --output-dir dataset_procesado --allow-new-classes

# Personalizar ratios de splits para la fusión
python datasetProcessorTool.py --coco-dirs dataset1 dataset2 --output-dir resultado --merge --split-ratios 0.8,0.1,0.1
```

## Guía rápida: Flujo completo de trabajo

Este es el flujo de trabajo recomendado para utilizar el sistema completo:

### 1. Preparación y procesamiento de datos

```bash
# Preparar datos y convertir a formato YOLO con clases unificadas
python datasetProcessorTool.py --coco-dir dataset_original --output-dir dataset_procesado --config configs/extended_mapping.yaml
```

### 2. Entrenamiento de modelos

```bash
# Entrenar con parámetros básicos
python trainYolov11s.py --data dataset_procesado/data.yaml --epochs 100 --batch 16 --device 0
```

### 3. Evaluación y análisis

```bash
# Evaluar modelo entrenado
python evaluateModel.py --model runs/detect/train/weights/best.pt --data dataset_procesado/data.yaml --full-report
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

## Integración con YOLO Detection

Para integrar un modelo evaluado en el sistema principal YOLO Detection:

1. Copie el modelo (`.pt`) a la carpeta `/models/` en la raíz del proyecto principal
2. Actualice la configuración del modo 'Defects' para usar su modelo:
   ```python
   # En src/modes/defects/config.py
   MODEL_WEIGHTS = "nombre_de_su_modelo.pt"  # El archivo en /models/
   ```
3. El sistema utilizará automáticamente `get_model_path()` para cargar su modelo

## Consejos para la Evaluación

- Utilice `--full-report` para obtener un análisis completo en un solo paso
- Verifique los falsos positivos más comunes para identificar áreas de mejora
- Compare siempre el rendimiento entre validación y prueba para detectar overfitting
- Para modelos destinados a dispositivos con recursos limitados, priorice el benchmark de velocidad
- Los modelos con mAP50 > 0.85 son generalmente adecuados para implementación en producción
