#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Processor Tool - Herramienta avanzada para procesamiento de datasets
===============================================================================

Esta herramienta permite:
1. Convertir datasets en formato COCO a formato YOLO
2. Preparar datasets con estructura no estándar a formato YOLO
3. Fusionar múltiples datasets en uno solo
4. Remapear y unificar clases entre diferentes datasets
5. Distribuir imágenes en splits train/valid/test

Es especialmente útil para trabajar con datasets de daños en vehículos
que utilizan diferentes esquemas de etiquetado.

Autor: Insolite Solutions SL
Versión: 1.1.0
"""

import os
import json
import shutil
import argparse
import yaml
import re
import random
from tqdm import tqdm
from collections import defaultdict, Counter

# Caminos por defecto
DEFAULT_CONFIG_PATH = 'configs/extended_mapping.yaml'

# Esta es la configuración mínima por defecto que se utilizará solo si no se encuentra el archivo de configuración
# La configuración principal debe estar en el archivo YAML
DEFAULT_TARGET_CLASSES = [
    "damaged door", "damaged window", "damaged headlight",
    "damaged mirror", "dent", "damaged hood",
    "damaged bumper", "damaged wind shield"
]

# Mapeo mínimo por defecto (solo se usa si no hay archivo de configuración)
DEFAULT_CLASS_MAPPING = {
    "scratch": "dent",
    "crack": "damaged window",
    "collapse": "damaged door",
    "breakage": "damaged bumper",
    "depression": "dent",
    "part_off": "damaged bumper",
    "damage": "dent",
    "door_damage": "damaged door",
    "window_damage": "damaged window",
    "light_damage": "damaged headlight",
    "mirror_damage": "damaged mirror",
    "hood_damage": "damaged hood",
    "bumper_damage": "damaged bumper",
    "windshield_damage": "damaged wind shield",
}

def convert_bbox_coco_to_yolo(img_width, img_height, bbox):
    """
    Convierte un bounding box del formato COCO [x,y,width,height] al formato YOLO [x_center,y_center,width,height].
    
    Args:
        img_width (int): Ancho de la imagen
        img_height (int): Alto de la imagen
        bbox (list): Bounding box en formato COCO [x,y,width,height]
        
    Returns:
        list: Bounding box en formato YOLO [x_center,y_center,width,height] normalizado de 0 a 1
    """
    x, y, width, height = bbox
    
    # Convertir a coordenadas relativas centrales (formato YOLO)
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width = width / img_width
    height = height / img_height
    
    return [x_center, y_center, width, height]

def get_class_id(category_id, categories, class_mapping, target_classes, allow_new_classes=False):
    """
    Obtiene el ID de clase según el sistema objetivo.
    
    Args:
        category_id (int): ID de categoría en COCO
        categories (list): Lista de categorías del dataset COCO
        class_mapping (dict): Diccionario de mapeo de clases originales a clases objetivo
        target_classes (list): Lista de clases objetivo
        allow_new_classes (bool): Si permitir clases nuevas no definidas en target_classes
        
    Returns:
        int: ID de clase en el sistema objetivo, o None si no se debe incluir
    """
    # Obtener el nombre de la categoría original
    category_name = next((cat["name"] for cat in categories if cat["id"] == category_id), None)
    
    if category_name is None:
        return None
    
    # Usar mapeo personalizado para convertir a nuestras clases
    mapped_class = class_mapping.get(category_name, category_name)
    
    # Si la clase mapeada está en nuestras clases objetivo, usarla
    if mapped_class in target_classes:
        return target_classes.index(mapped_class)
    
    # Si permitimos clases nuevas, agregar al final
    if allow_new_classes:
        if mapped_class not in target_classes:
            target_classes.append(mapped_class)
        return target_classes.index(mapped_class)
        
    return None  # Ignorar esta anotación

def load_mapping_config(config_path):
    """
    Carga la configuración de mapeo desde un archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        tuple: (class_mapping, target_classes)
    """
    if not os.path.exists(config_path):
        print(f"Archivo de configuración no encontrado: {config_path}")
        print("Usando configuración por defecto.")
        return DEFAULT_CLASS_MAPPING, DEFAULT_TARGET_CLASSES
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_mapping = config.get('class_mapping', DEFAULT_CLASS_MAPPING)
        target_classes = config.get('target_classes', DEFAULT_TARGET_CLASSES)
        
        print(f"Configuración cargada de {config_path}")
        return class_mapping, target_classes
    
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        print("Usando configuración por defecto.")
        return DEFAULT_CLASS_MAPPING, DEFAULT_TARGET_CLASSES

def convert_dataset(coco_dir, output_dir, class_mapping, target_classes, split_config=None, allow_new_classes=False, include_empty=False):
    """
    Convierte un dataset en formato COCO a formato YOLO.
    
    Args:
        coco_dir (str): Directorio con el dataset en formato COCO
        output_dir (str): Directorio donde se guardará el dataset en formato YOLO
        class_mapping (dict): Diccionario de mapeo de clases
        target_classes (list): Lista de clases objetivo
        split_config (dict): Configuración de splits (opcional)
        allow_new_classes (bool): Si permitir clases no definidas en target_classes
        include_empty (bool): Si incluir imágenes sin anotaciones
    
    Returns:
        dict: Estadísticas de la conversión
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear estructura de directorios para YOLO
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Detectar los directorios en COCO y mapearlos a splits YOLO
    coco_dirs = {}
    
    # Verificar si es un dataset estándar COCO con subdirectorios train/val/test
    for item in os.listdir(coco_dir):
        item_path = os.path.join(coco_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        if item.startswith('train'):
            coco_dirs['train'] = item
        elif item.startswith('val'):
            coco_dirs['valid'] = item
        elif item.startswith('test'):
            coco_dirs['test'] = item
    
    # Verificar si existe una carpeta de anotaciones (formato COCO estándar)
    anno_folder = os.path.join(coco_dir, 'annotations')
    if os.path.isdir(anno_folder):
        # Buscar archivos de anotaciones en la carpeta annotations
        anno_files = {}
        for f in os.listdir(anno_folder):
            if f.endswith('.json'):
                if 'train' in f:
                    anno_files['train'] = os.path.join(anno_folder, f)
                elif 'val' in f:
                    anno_files['valid'] = os.path.join(anno_folder, f)
                elif 'test' in f:
                    anno_files['test'] = os.path.join(anno_folder, f)
        
        # Si se encontraron archivos de anotaciones estándar, usarlos
        if anno_files:
            print(f"Encontrada estructura COCO estándar con carpeta annotations")
            for split, anno_file in anno_files.items():
                # Intentar determinar el directorio correspondiente
                img_dir = None
                for d in coco_dirs:
                    if d.startswith(split):
                        img_dir = coco_dirs[d]
                        break
                
                if img_dir:
                    print(f"  - Split {split}: usando {anno_file} con imágenes en {img_dir}")
                else:
                    print(f"  - Split {split}: usando {anno_file} (no se encontró un directorio de imágenes específico)")
    
    # Si no se encontraron directorios estándar ni anotaciones, usar el directorio principal
    if not coco_dirs and not (os.path.isdir(anno_folder) and os.listdir(anno_folder)):
        # Buscar archivos de anotaciones en el directorio principal
        anno_files = [f for f in os.listdir(coco_dir) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(coco_dir, f))]
        
        if not anno_files:
            print(f"No se encontraron archivos de anotaciones COCO en {coco_dir}")
            return {}
        
        # Usar el directorio principal para todos los splits si se especifica custom_split
        if split_config:
            for split in ['train', 'valid', 'test']:
                if split in split_config:
                    coco_dirs[split] = ''
        else:
            # Si no hay configuración de split, asignar todo a train
            coco_dirs['train'] = ''
    
    # Estadísticas
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_counts': defaultdict(int),
        'splits': {}
    }
    
    # Procesar cada split
    for yolo_split, coco_split in coco_dirs.items():
        coco_split_path = os.path.join(coco_dir, coco_split)
        
        # Buscar archivo de anotaciones
        anno_file = None
        
        # Primero verificar si hay un archivo en la carpeta annotations
        if os.path.isdir(anno_folder):
            for f in os.listdir(anno_folder):
                if f.endswith('.json') and (
                    (yolo_split == 'train' and 'train' in f) or
                    (yolo_split == 'valid' and ('val' in f or 'valid' in f)) or
                    (yolo_split == 'test' and 'test' in f)
                ):
                    anno_file = os.path.join(anno_folder, f)
                    break
        
        # Si no se encontró en annotations, buscar en el directorio del split
        if not anno_file and os.path.isdir(coco_split_path):
            anno_files = [f for f in os.listdir(coco_split_path) 
                         if f.endswith('.json') and os.path.isfile(os.path.join(coco_split_path, f))]
            if anno_files:
                anno_file = os.path.join(coco_split_path, anno_files[0])
        
        # Si aún no se encuentra, buscar en el directorio principal
        if not anno_file:
            anno_files = [f for f in os.listdir(coco_dir) 
                         if f.startswith(coco_split) and f.endswith('.json')]
            if anno_files:
                anno_file = os.path.join(coco_dir, anno_files[0])
            elif coco_split == '':
                # Si es el directorio principal, usar cualquier archivo de anotaciones
                anno_files = [f for f in os.listdir(coco_dir) 
                             if f.endswith('.json') and os.path.isfile(os.path.join(coco_dir, f))]
                if anno_files:
                    anno_file = os.path.join(coco_dir, anno_files[0])
        
        if not anno_file:
            print(f"No se encontró archivo de anotaciones para split {yolo_split}")
            continue
        
        print(f"Procesando split {yolo_split} con anotaciones de {anno_file}")
        
        # Cargar anotaciones COCO
        with open(anno_file, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data['images']
        annotations = coco_data['annotations']
        categories = coco_data['categories']
        
        # Organizar anotaciones por imagen
        annotations_by_image = defaultdict(list)
        for ann in annotations:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Estadísticas del split
        split_images = 0
        split_annotations = 0
        
        # Procesar cada imagen
        for img in tqdm(images, desc=f"Convirtiendo {yolo_split}"):
            img_id = img['id']
            img_file = img['file_name']
            img_width = img['width']
            img_height = img['height']
            
            # Si la imagen no tiene anotaciones, puede ser opcional incluirla
            if img_id not in annotations_by_image and not include_empty:
                continue
            
            # Determinar ruta de la imagen original
            if os.path.isdir(coco_split_path):
                # Buscar la imagen en el directorio del split
                img_dirs = ['images', 'JPEGImages', '']
                img_found = False
                for img_dir in img_dirs:
                    src_img_path = os.path.join(coco_split_path, img_dir, img_file)
                    if os.path.exists(src_img_path):
                        img_found = True
                        break
                
                if not img_found:
                    print(f"No se encontró la imagen {img_file} en {coco_split_path}")
                    continue
            else:
                # Buscar la imagen en el directorio principal o en subdirectorios comunes
                img_dirs = ['images', 'JPEGImages', '']
                img_found = False
                for img_dir in img_dirs:
                    src_img_path = os.path.join(coco_dir, img_dir, img_file)
                    if os.path.exists(src_img_path):
                        img_found = True
                        break
                
                if not img_found:
                    print(f"No se encontró la imagen {img_file} en {coco_dir}")
                    continue
            
            # Determinar destino de la imagen
            dst_img_path = os.path.join(output_dir, yolo_split, 'images', os.path.basename(img_file))
            
            # Copiar imagen
            shutil.copy2(src_img_path, dst_img_path)
            
            # Incrementar contadores
            split_images += 1
            stats['total_images'] += 1
            
            # Preparar archivo de etiquetas
            label_file = os.path.splitext(os.path.basename(img_file))[0] + '.txt'
            dst_label_path = os.path.join(output_dir, yolo_split, 'labels', label_file)
            
            # Convertir anotaciones a formato YOLO
            yolo_annotations = []
            for ann in annotations_by_image[img_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']  # Formato COCO: [x,y,width,height]
                
                # Obtener ID de clase según nuestro sistema
                class_id = get_class_id(category_id, categories, class_mapping, target_classes, allow_new_classes)
                if class_id is None:
                    continue  # Ignorar clases que no están en nuestro sistema
                
                # Convertir bbox a formato YOLO
                yolo_bbox = convert_bbox_coco_to_yolo(img_width, img_height, bbox)
                
                # Guardar anotación YOLO: class_id, x_center, y_center, width, height
                yolo_annotation = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_annotations.append(yolo_annotation)
                
                # Contar clases
                class_name = target_classes[class_id]
                stats['class_counts'][class_name] += 1
                
                stats['total_annotations'] += 1
                split_annotations += 1
            
            # Escribir anotaciones YOLO a archivo
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"Split {yolo_split}: {split_images} imágenes, {split_annotations} anotaciones")
        stats['splits'][yolo_split] = {'images': split_images, 'annotations': split_annotations}
    
    # Crear archivo data.yaml
    yaml_content = f"""train: {os.path.join(output_dir, 'train', 'images')}
val: {os.path.join(output_dir, 'valid', 'images')}
test: {os.path.join(output_dir, 'test', 'images')}
nc: {len(target_classes)}
names: {target_classes}
"""
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # Imprimir resumen
    print("\nResumen de la conversión:")
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Total de anotaciones: {stats['total_annotations']}")
    print("\nConteo de clases:")
    for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {cls}: {count}")
    
    print(f"\nDataset convertido guardado en: {output_dir}")
    
    return stats

def convert_coco_to_yolo(coco_dir, output_dir, class_mapping=None, target_classes=None, sample_size=0):
    """
    Convierte un dataset en formato COCO a formato YOLO.
    
    Args:
        coco_dir (str): Directorio con el dataset en formato COCO
        output_dir (str): Directorio donde se guardará el dataset en formato YOLO
        class_mapping (dict): Mapeo de clases COCO a clases unificadas
        target_classes (list): Lista de clases en el formato unificado
        sample_size (int): Número de imágenes a incluir por split (0 = todas)
    
    Returns:
        dict: Estadísticas del proceso
    """
    print(f"\n=== Convirtiendo dataset COCO a YOLO ===")
    print(f"Directorio COCO: {coco_dir}")
    print(f"Directorio de salida: {output_dir}")
    
    # Verificar que existe el directorio de anotaciones
    annotations_dir = os.path.join(coco_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        print(f"⚠️ No se encontró el directorio de anotaciones: {annotations_dir}")
        return None
    
    # Buscar archivos de anotaciones
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    if not annotation_files:
        print(f"⚠️ No se encontraron archivos de anotaciones en: {annotations_dir}")
        return None
    
    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Detectar qué splits están disponibles
    available_splits = []
    split_info = {}  # {split: {'annotation_file': file, 'images_dir': dir}}
    
    # Paso 1: Buscar archivos de anotaciones para cada split
    for ann_file in annotation_files:
        # Buscar patrones como "instances_train2017.json"
        if 'train' in ann_file:
            if 'train' not in split_info:
                split_info['train'] = {'annotation_file': ann_file}
                if 'train' not in available_splits:
                    available_splits.append('train')
        elif 'val' in ann_file:
            if 'valid' not in split_info:
                split_info['valid'] = {'annotation_file': ann_file}
                if 'valid' not in available_splits:
                    available_splits.append('valid')
        elif 'test' in ann_file:
            if 'test' not in split_info:
                split_info['test'] = {'annotation_file': ann_file}
                if 'test' not in available_splits:
                    available_splits.append('test')
    
    # Paso 2: Buscar directorios de imágenes para cada split - formatos comunes
    for d in os.listdir(coco_dir):
        dir_path = os.path.join(coco_dir, d)
        if os.path.isdir(dir_path) and d != 'annotations':
            # Formatos comunes: train2017, val2017, test2017, train, val, valid, test
            if 'train' in d.lower():
                split_info.setdefault('train', {})['images_dir'] = dir_path
                if 'train' not in available_splits:
                    available_splits.append('train')
            elif 'val' in d.lower():
                split_info.setdefault('valid', {})['images_dir'] = dir_path
                if 'valid' not in available_splits:
                    available_splits.append('valid')
            elif 'test' in d.lower():
                split_info.setdefault('test', {})['images_dir'] = dir_path
                if 'test' not in available_splits:
                    available_splits.append('test')
    
    # Si no encontramos splits específicos, usar defaults
    if not available_splits:
        available_splits.append('train')
        split_info['train'] = {'annotation_file': annotation_files[0]}
    
    # Cargar categorías COCO para crear el mapeo
    coco_categories = []
    coco_to_yolo_mapping = {}
    
    # Intentar cargar de cualquier archivo de anotaciones disponible
    for ann_file in annotation_files:
        try:
            with open(os.path.join(annotations_dir, ann_file), 'r') as f:
                coco_data = json.load(f)
                if 'categories' in coco_data:
                    coco_categories = coco_data['categories']
                    break
        except Exception as e:
            print(f"Error al cargar {ann_file}: {e}")
            continue
    
    # Si no se encontraron categorías, usar valores por defecto
    if not coco_categories:
        print("⚠️ No se encontraron categorías en los archivos COCO")
        if target_classes:
            print(f"Usando clases objetivo proporcionadas: {target_classes}")
        else:
            print("No hay información de clases disponible")
            return None
    
    # Crear mapeo de categorías COCO a clases YOLO
    coco_class_names = []
    for cat in coco_categories:
        cat_id = cat['id']
        cat_name = cat['name']
        coco_class_names.append(cat_name)
        
        # Si tenemos mapeo de clases, aplicarlo
        if class_mapping and target_classes:
            mapped_name = class_mapping.get(cat_name, cat_name)
            if mapped_name in target_classes:
                coco_to_yolo_mapping[cat_id] = target_classes.index(mapped_name)
        else:
            # Mapeo 1:1 (el índice en la lista será el ID en YOLO)
            coco_to_yolo_mapping[cat_id] = len(coco_to_yolo_mapping)
    
    # Si no hay target_classes, usar las categorías COCO
    if not target_classes:
        target_classes = coco_class_names
    
    total_images = 0
    class_counts = Counter()
    
    # Procesar cada split
    for split in available_splits:
        if split not in split_info:
            print(f"Información incompleta para split {split}, omitiendo")
            continue
            
        split_data = split_info[split]
        
        # Verificar si tenemos archivo de anotaciones
        if 'annotation_file' not in split_data:
            print(f"No se encontró archivo de anotaciones para {split}, omitiendo")
            continue
            
        # Cargar anotaciones
        ann_file = split_data['annotation_file']
        try:
            with open(os.path.join(annotations_dir, ann_file), 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"Error al cargar {ann_file}: {e}")
            continue
        
        # Verificar estructura de datos COCO
        if 'images' not in coco_data or 'annotations' not in coco_data:
            print(f"Estructura COCO inválida en {ann_file}")
            continue
        
        # Verificar directorio de imágenes
        if 'images_dir' not in split_data:
            # Buscar imágenes en directorio con el mismo nombre que el split
            candidate_dirs = []
            for name_pattern in [f"{split}", f"{split}2017", f"{split}2019", f"{split}2020", f"{split}2021", f"{split}2022", f"{split}2023"]:
                if split == 'valid':  # También probar con "val"
                    name_pattern = name_pattern.replace('valid', 'val')
                img_dir = os.path.join(coco_dir, name_pattern)
                if os.path.exists(img_dir) and os.path.isdir(img_dir):
                    candidate_dirs.append(img_dir)
            
            if candidate_dirs:
                split_data['images_dir'] = candidate_dirs[0]
            else:
                print(f"No se encontró directorio de imágenes para {split}")
                continue
        
        images_dir = split_data['images_dir']
        print(f"Procesando split: {split}")
        print(f"  Directorio de imágenes: {images_dir}")
        print(f"  Archivo de anotaciones: {os.path.join(annotations_dir, ann_file)}")
        
        # Crear mapeo de ID a imagen
        image_id_to_file = {}
        for img_info in coco_data['images']:
            image_id_to_file[img_info['id']] = {
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height']
            }
        
        # Organizar anotaciones por ID de imagen
        image_annotations = defaultdict(list)
        for anno in coco_data['annotations']:
            image_id = anno['image_id']
            image_annotations[image_id].append(anno)
        
        # Procesar cada imagen
        image_ids = list(image_id_to_file.keys())
        
        # Limitar número de imágenes si es necesario
        if sample_size > 0 and len(image_ids) > sample_size:
            image_ids = random.sample(image_ids, sample_size)
        
        # Mostrar progreso
        for image_id in tqdm(image_ids, desc=f"Procesando {split}", unit="img"):
            img_info = image_id_to_file[image_id]
            img_file = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Rutas de origen
            img_src_path = os.path.join(images_dir, img_file)
            
            # En caso de que la imagen esté en un subdirectorio
            if not os.path.exists(img_src_path):
                # Buscar la imagen en subdirectorios
                for root, _, files in os.walk(images_dir):
                    if img_file in files:
                        img_src_path = os.path.join(root, img_file)
                        break
            
            if not os.path.exists(img_src_path):
                continue  # Saltar esta imagen si no se encuentra
                
            # Obtener nombre base para etiqueta
            img_basename = os.path.splitext(os.path.basename(img_file))[0]
            label_file = f"{img_basename}.txt"
            
            # Rutas de destino
            img_dst_path = os.path.join(output_dir, split, 'images', os.path.basename(img_file))
            label_dst_path = os.path.join(output_dir, split, 'labels', label_file)
            
            # Copiar imagen
            shutil.copy2(img_src_path, img_dst_path)
            
            # Procesar anotaciones
            annos = image_annotations.get(image_id, [])
            with open(label_dst_path, 'w') as f:
                for anno in annos:
                    # Solo procesar anotaciones de tipo bbox
                    if 'bbox' not in anno:
                        continue
                        
                    # Obtener categoría y verificar mapeo
                    cat_id = anno['category_id']
                    if cat_id not in coco_to_yolo_mapping:
                        continue
                        
                    # Mapear a clase YOLO
                    yolo_class_id = coco_to_yolo_mapping[cat_id]
                    
                    # Convertir bbox COCO a formato YOLO
                    x, y, width, height = anno['bbox']
                    x_center = (x + width / 2) / img_width
                    y_center = (y + height / 2) / img_height
                    width = width / img_width
                    height = height / img_height
                    
                    # Escribir línea en formato YOLO
                    f.write(f"{yolo_class_id} {x_center} {y_center} {width} {height}\n")
                    
                    # Actualizar conteo de clases
                    if target_classes and yolo_class_id < len(target_classes):
                        class_name = target_classes[yolo_class_id]
                        class_counts[class_name] += 1
            
            total_images += 1
    
    # Mostrar resumen
    print("\nConversión completada:")
    print(f"Total de imágenes: {total_images}")
    print("Conteo de clases:")
    for cls, count in class_counts.items():
        print(f"- {cls}: {count}")
    
    # Crear archivo data.yaml
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml_content = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',
            'nc': len(target_classes),
            'names': target_classes
        }
        yaml.dump(yaml_content, f, sort_keys=False)
    
    return {
        'total_images': total_images,
        'total_annotations': sum(class_counts.values()),
        'class_counts': dict(class_counts)
    }

def merge_datasets(input_dirs, output_dir, class_mapping, target_classes, split_config=None, use_remapping=False):
    """
    Fusiona múltiples datasets YOLO en uno solo.
    
    Args:
        input_dirs (list): Lista de directorios con datasets YOLO
        output_dir (str): Directorio donde se guardará el dataset fusionado
        class_mapping (dict): Diccionario de mapeo de clases
        target_classes (list): Lista de clases objetivo
        split_config (dict): Configuración de splits (opcional)
        use_remapping (bool): Si aplicar remapeo a los IDs de clase
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear estructura de directorios para YOLO
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Estadísticas
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_counts': defaultdict(int),
        'splits': defaultdict(lambda: {'images': 0, 'annotations': 0})
    }
    
    # Recopilar todas las imágenes y etiquetas
    all_image_paths = []
    all_label_paths = []
    
    for input_dir in input_dirs:
        # Cargar clases originales si están disponibles
        orig_target_classes = []
        yaml_file = os.path.join(input_dir, 'data.yaml')
        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        orig_target_classes = yaml_data['names']
            except:
                pass
        
        for split in ['train', 'valid', 'test']:
            image_dir = os.path.join(input_dir, split, 'images')
            label_dir = os.path.join(input_dir, split, 'labels')
            
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                continue
            
            # Encontrar todas las imágenes con etiquetas
            for img_file in os.listdir(image_dir):
                if not (img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png')):
                    continue
                
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(label_dir, label_file)
                
                if os.path.exists(label_path):
                    all_image_paths.append((os.path.join(image_dir, img_file), orig_target_classes))
                    all_label_paths.append(label_path)
    
    print(f"Encontradas {len(all_image_paths)} imágenes con etiquetas en total")
    
    # Determinar la distribución de splits
    if split_config is None:
        split_config = {'train': 0.7, 'valid': 0.2, 'test': 0.1}
    
    # Calcular límites de índices para cada split
    n_samples = len(all_image_paths)
    train_idx = int(n_samples * split_config['train'])
    valid_idx = train_idx + int(n_samples * split_config['valid'])
    
    # Distribuir las imágenes en splits
    split_indices = {
        'train': list(range(0, train_idx)),
        'valid': list(range(train_idx, valid_idx)),
        'test': list(range(valid_idx, n_samples))
    }
    
    # Procesar las imágenes según la asignación de splits
    for split, indices in split_indices.items():
        print(f"Procesando split {split}: {len(indices)} imágenes")
        
        for idx in tqdm(indices, desc=f"Copiando {split}"):
            img_info = all_image_paths[idx]
            img_path = img_info[0]
            orig_classes = img_info[1]
            label_path = all_label_paths[idx]
            
            # Copiar imagen
            dst_img_path = os.path.join(output_dir, split, 'images', os.path.basename(img_path))
            shutil.copy2(img_path, dst_img_path)
            
            # Leer y procesar etiquetas
            annotations = []
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        # Obtener nombre original y mapear a clase objetivo
                        if orig_classes and class_id < len(orig_classes):
                            class_name = orig_classes[class_id]
                            mapped_name = class_mapping.get(class_name, class_name)
                            
                            if mapped_name in target_classes:
                                new_class_id = target_classes.index(mapped_name)
                                parts[0] = str(new_class_id)
                                annotations.append(' '.join(parts) + '\n')
                                stats['class_counts'][mapped_name] += 1
                            else:
                                # Ignorar clases que no están en nuestro target
                                continue
                        
                        # Si no tenemos nombres, copiar directamente
                        annotations.append(line)
                        stats['class_counts'][f"class_{class_id}"] += 1
                    
                    except ValueError:
                        continue
            
            # Guardar anotaciones remapeadas
            dst_label_path = os.path.join(output_dir, split, 'labels', os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            stats['total_images'] += 1
    
    # Crear archivo data.yaml
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(target_classes),
        'names': target_classes
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    # Imprimir resumen
    print("\nResumen de la fusión:")
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Total de anotaciones: {stats['total_annotations']}")
    
    print("\nDistribución por split:")
    for split, split_stats in stats['splits'].items():
        print(f"- {split}: {split_stats['images']} imágenes, {split_stats['annotations']} anotaciones")
    
    print("\nConteo de clases:")
    for cls, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {cls}: {count}")
    
    print(f"\nDataset fusionado guardado en: {output_dir}")
    
    return stats

def process_yolo_dataset(input_dir, output_dir, split, current_classes, class_mapping, target_classes, allow_new_classes=False, include_empty=False, sample_size=0):
    """
    Procesa un split de un dataset YOLO aplicando remapeo de clases.
    
    Args:
        input_dir (str): Directorio con el dataset YOLO original
        output_dir (str): Directorio donde se guardará el dataset procesado
        split (str): Split a procesar (train, valid, test)
        current_classes (list): Lista de clases actuales en el dataset
        class_mapping (dict): Mapeo de clases (old_class -> new_class)
        target_classes (list): Lista de clases objetivo
        allow_new_classes (bool): Permitir clases nuevas no definidas en target_classes
        include_empty (bool): Incluir imágenes sin anotaciones
        sample_size (int): Número de imágenes a incluir (0 = todas)
    
    Returns:
        dict: Estadísticas del proceso
    """
    # Rutas de origen
    src_img_dir = os.path.join(input_dir, split, 'images')
    src_lbl_dir = os.path.join(input_dir, split, 'labels')
    
    # Verificar si existe este split
    if not os.path.exists(src_img_dir) or not os.path.isdir(src_img_dir):
        return None
    
    # Si no hay clase objetivo, usar las clases actuales
    if target_classes is None:
        target_classes = current_classes
        print(f"Usando clases actuales como clases objetivo: {target_classes}")
    
    # Si aún no tenemos clases objetivo y tenemos información de clases en el dataset, usarlas
    if target_classes is None and 'classes' in detect_dataset_structure(input_dir).get('structure', {}):
        target_classes = detect_dataset_structure(input_dir)['structure']['classes']
        print(f"Usando clases del dataset: {target_classes}")
    
    # Si aún no tenemos clases objetivo, usar una lista predeterminada
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES
        print(f"Usando clases predeterminadas: {target_classes}")
    
    # Rutas de destino
    dst_img_dir = os.path.join(output_dir, split, 'images')
    dst_lbl_dir = os.path.join(output_dir, split, 'labels')
    
    # Crear directorios de destino
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    # Estadísticas
    stats = {
        'total_images': 0,
        'total_annotations': 0,
        'class_counts': defaultdict(int)
    }
    
    # Procesar cada imagen
    for img_file in os.listdir(src_img_dir):
        if not (img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png')):
            continue
        
        img_path = os.path.join(src_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(src_lbl_dir, label_file)
        
        # Copiar imagen
        dst_img_path = os.path.join(dst_img_dir, img_file)
        shutil.copy2(img_path, dst_img_path)
        
        # Si existe la etiqueta, procesarla
        if os.path.exists(label_path):
            # Leer el archivo de etiquetas
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            remapped_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # formato YOLO: class_id x_center y_center width height
                    try:
                        class_id = int(parts[0])
                        # Obtener nombre original y mapear a clase objetivo
                        if current_classes and class_id < len(current_classes):
                            class_name = current_classes[class_id]
                            mapped_name = class_mapping.get(class_name, class_name)
                            
                            if mapped_name in target_classes:
                                new_class_id = target_classes.index(mapped_name)
                                parts[0] = str(new_class_id)
                                remapped_lines.append(' '.join(parts) + '\n')
                                stats['class_counts'][mapped_name] += 1
                            else:
                                # Ignorar clases que no están en nuestro target
                                continue
                        else:
                            # Si no tenemos nombres para las clases originales, mantener el ID tal cual
                            remapped_lines.append(line)
                            stats['class_counts'][f"class_{class_id}"] += 1
                    
                    except ValueError:
                        continue
            
            # Guardar anotaciones remapeadas
            dst_label_path = os.path.join(dst_lbl_dir, label_file)
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(remapped_lines))
        else:
            # Copiar etiqueta vacía
            dst_label_path = os.path.join(dst_lbl_dir, label_file)
            with open(dst_label_path, 'w') as f:
                pass
        
        stats['total_images'] += 1
    
    return stats

def detect_dataset_structure(input_dir):
    """
    Detecta automáticamente el tipo y estructura del dataset.
    
    Args:
        input_dir (str): Directorio a analizar
        
    Returns:
        dict: Información sobre el tipo y estructura del dataset
    """
    info = {
        'type': 'unknown',  # 'coco', 'yolo', 'raw'
        'structure': {},
        'has_annotations': False
    }
    
    # Verificar si es estructura COCO
    ann_dir = os.path.join(input_dir, 'annotations')
    has_coco_annotations = False
    
    if os.path.exists(ann_dir) and os.path.isdir(ann_dir):
        annotation_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        if annotation_files:
            has_coco_annotations = True
            info['type'] = 'coco'
            info['structure']['annotations_dir'] = ann_dir
            info['has_annotations'] = True
            
            # Detectar subdirectorios de imágenes (train2017, val2017, etc.)
            img_dirs = {}
            possible_splits = []
            
            # Buscar patrones como "train2017", "val2020", etc.
            pattern = re.compile(r'(train|val|test|valid)(?:\d+|_v\d+)?')
            
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path) and pattern.match(item):
                    split = 'train' if item.startswith('train') else 'valid' if item.startswith('val') else 'test'
                    img_dirs[split] = item
                    possible_splits.append(item)
            
            if img_dirs:
                info['structure']['image_dirs'] = img_dirs
    
    # Verificar si es estructura YOLO
    if not has_coco_annotations:
        # Buscar estructura YOLO clásica (train/images, train/labels, etc.)
        train_images = os.path.join(input_dir, 'train', 'images')
        train_labels = os.path.join(input_dir, 'train', 'labels')
        
        valid_images = os.path.join(input_dir, 'valid', 'images')
        valid_labels = os.path.join(input_dir, 'valid', 'labels')
        
        if (os.path.exists(train_images) and os.path.isdir(train_images) and 
            os.path.exists(train_labels) and os.path.isdir(train_labels)):
            info['type'] = 'yolo'
            info['structure']['format'] = 'standard'  # train/images, train/labels
            info['has_annotations'] = True
            
            # Buscar data.yaml
            data_yaml_path = os.path.join(input_dir, 'data.yaml')
            if os.path.exists(data_yaml_path):
                info['structure']['data_yaml'] = data_yaml_path
                
                # Cargar clases de data.yaml
                try:
                    with open(data_yaml_path, 'r') as f:
                        data_yaml = yaml.safe_load(f)
                        if 'names' in data_yaml:
                            info['structure']['classes'] = data_yaml['names']
                except Exception as e:
                    print(f"Error al cargar data.yaml: {e}")
        
        # Revisar estructura invertida (images/train, labels/train)
        elif os.path.exists(os.path.join(input_dir, 'images')) and os.path.exists(os.path.join(input_dir, 'labels')):
            info['type'] = 'yolo'
            info['structure']['format'] = 'inverted'  # images/train, labels/train
            info['has_annotations'] = True
            
            # Buscar data.yaml
            data_yaml_path = os.path.join(input_dir, 'data.yaml')
            if os.path.exists(data_yaml_path):
                info['structure']['data_yaml'] = data_yaml_path
    
    # Si no es COCO ni YOLO, puede ser estructura RAW
    if info['type'] == 'unknown':
        # Verificar si hay imágenes y anotaciones en el directorio raíz
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
        
        if image_files:
            info['type'] = 'raw'
            info['structure']['format'] = 'flat'  # imágenes y .txt en el directorio raíz
            
            # Verificar si hay archivos .txt con el mismo nombre que las imágenes
            image_bases = [os.path.splitext(img)[0] for img in image_files]
            txt_bases = [os.path.splitext(txt)[0] for txt in txt_files]
            
            common_bases = set(image_bases).intersection(set(txt_bases))
            if common_bases:
                info['has_annotations'] = True
    
    return info

def prepare_standard_structure(input_dir, output_dir, class_mapping, target_classes, sample_size=0):
    """
    Prepara un dataset con estructura no estándar (raw) para convertirlo al formato YOLO estándar.
    
    Args:
        input_dir (str): Directorio de entrada
        output_dir (str): Directorio de salida
        class_mapping (dict): Mapeo de clases
        target_classes (list): Lista de clases objetivo
        sample_size (int): Número de imágenes a incluir por split (0 = todas)
        
    Returns:
        dict: Estadísticas del proceso
    """
    print(f"\n=== Preparando dataset {input_dir} a estructura YOLO estándar ===")
    
    # Detectar la estructura del dataset
    structure_info = detect_dataset_structure(input_dir)
    
    if structure_info['type'] == 'unknown':
        print(f"⚠️ No se pudo determinar la estructura del dataset {input_dir}")
        return None
    
    print(f"Tipo de dataset detectado: {structure_info['type']}")
    
    if structure_info['type'] == 'yolo' and structure_info['structure'].get('format') == 'standard':
        print("El dataset ya tiene estructura YOLO estándar, solo se realizará remapeo de clases")
        return None
    
    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    total_images = 0
    class_counts = Counter()
    
    # Si no tenemos clases objetivo y tenemos información de clases en el dataset, usarlas
    if target_classes is None and 'classes' in structure_info.get('structure', {}):
        target_classes = structure_info['structure']['classes']
        print(f"Usando clases del dataset: {target_classes}")
    
    # Si aún no tenemos clases objetivo, usar una lista predeterminada
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES
        print(f"Usando clases predeterminadas: {target_classes}")
    
    # Procesar según el tipo de dataset
    if structure_info['type'] == 'coco':
        # Ya existe una función para procesar COCO, solo con el nuevo mapeo
        # Esto se manejará con el llamado existente a convert_dataset
        return None
    
    elif structure_info['type'] == 'raw':
        # Procesar dataset con estructura plana
        print("Procesando dataset con estructura plana (imágenes y anotaciones en directorio raíz)")
        
        images_paths = []
        
        # Recopilar todos los pares imagen-etiqueta
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(input_dir, file)
                base_name = os.path.splitext(file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(input_dir, label_file)
                
                if os.path.exists(label_path):
                    # Distribuir en train, valid, test según ratios 70/20/10
                    rand = random.random()
                    if rand < 0.7:
                        split = 'train'
                    elif rand < 0.9:
                        split = 'valid'
                    else:
                        split = 'test'
                    images_paths.append((img_path, label_path, split))
        
        if not images_paths:
            print("⚠️ No se encontraron pares imagen-etiqueta válidos")
            return None
        
        # Mezclar aleatoriamente las imágenes
        random.shuffle(images_paths)
        
        # Limitar por sample_size si es necesario
        if sample_size > 0 and len(images_paths) > sample_size:
            images_paths = images_paths[:sample_size]
        
        # Procesar y copiar los archivos
        print("Copiando archivos...")
        for img_path, label_path, split in tqdm(images_paths, desc=f"Procesando", unit="archivos"):
            img_filename = os.path.basename(img_path)
            label_filename = os.path.basename(label_path)
            
            # Destinos
            img_dest = os.path.join(output_dir, split, 'images', img_filename)
            label_dest = os.path.join(output_dir, split, 'labels', label_filename)
            
            # Copiar imagen
            shutil.copy2(img_path, img_dest)
            
            # Si queremos remapear las clases al copiar
            if class_mapping and os.path.exists(label_path):
                # Leer el archivo de etiquetas
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # formato YOLO: class_id x_center y_center width height
                        try:
                            class_id = int(parts[0])
                            # Si queremos remapear la clase
                            if class_mapping:
                                # Si tenemos información de clases en el dataset
                                if structure_info.get('structure', {}).get('classes') and class_id < len(structure_info['structure']['classes']):
                                    class_name = structure_info['structure']['classes'][class_id]
                                    mapped_name = class_mapping.get(class_name, class_name)
                                    
                                    if mapped_name in target_classes:
                                        new_class_id = target_classes.index(mapped_name)
                                        parts[0] = str(new_class_id)
                                        remapped_lines.append(' '.join(parts) + '\n')
                                        class_counts[mapped_name] += 1
                                    else:
                                        # Ignorar clases que no están en nuestro target
                                        continue
                                else:
                                    # No tenemos información de clases, mantener el ID y contar genéricamente
                                    mapped_name = f"class_{class_id}"
                                    if class_id < len(target_classes):
                                        mapped_name = target_classes[class_id]
                                    class_counts[mapped_name] += 1
                                    remapped_lines.append(line)
                            else:
                                # Si no queremos remapear, mantener la clase original y contar
                                mapped_name = f"class_{class_id}"
                                if class_id < len(target_classes):
                                    mapped_name = target_classes[class_id]
                                class_counts[mapped_name] += 1
                                remapped_lines.append(line)
                        except ValueError:
                            continue
                
                # Guardar anotaciones remapeadas
                with open(label_dest, 'w') as f:
                    f.writelines(remapped_lines)
            else:
                # Copiar etiqueta sin modificar
                shutil.copy2(label_path, label_dest)
                
                # Contar las clases en el archivo copiado aunque no las remapeemos
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # formato YOLO: class_id x_center y_center width height
                        try:
                            class_id = int(parts[0])
                            # Usar nombre de clase si está dentro del rango, o "class_X" si no
                            if class_id < len(target_classes):
                                class_name = target_classes[class_id]
                            else:
                                class_name = f"class_{class_id}"
                            
                            class_counts[class_name] += 1
                        except ValueError:
                            continue
            
            total_images += 1
    
    elif structure_info['type'] == 'yolo' and structure_info['structure'].get('format') == 'inverted':
        # Procesar dataset YOLO con estructura invertida (images/train, labels/train)
        print("Procesando dataset YOLO con estructura invertida (images/train, labels/train)")
        
        images_dir = os.path.join(input_dir, 'images')
        labels_dir = os.path.join(input_dir, 'labels')
        
        # Revisar todos los directorios de split disponibles
        available_splits = []
        for possible_split in ['train', 'valid', 'val', 'test']:
            split_images = os.path.join(images_dir, possible_split)
            if os.path.exists(split_images) and os.path.isdir(split_images):
                if possible_split == 'val':
                    # Estandarizar 'val' como 'valid'
                    available_splits.append('valid')
                else:
                    available_splits.append(possible_split)
        
        # Si no se encuentran splits explícitos, tratar como si todo fuera 'train'
        if not available_splits:
            available_splits = ['train']
            if os.path.exists(images_dir) and os.path.isdir(images_dir):
                # Asumimos que las imágenes están directamente en el directorio 'images'
                split_to_dir = {'train': images_dir}  
            else:
                print(f"⚠️ No se encontraron directorios de imágenes válidos")
                return None
        else:
            # Mapear cada split a su directorio correspondiente
            split_to_dir = {}
            for split in available_splits:
                split_name = 'val' if split == 'valid' else split
                split_to_dir[split] = os.path.join(images_dir, split_name)
        
        print(f"Splits encontrados: {available_splits}")
        
        # Verificar redistribución de splits
        do_auto_split = False
        if len(available_splits) == 1 and available_splits[0] == 'train':
            # Verificar cuántas imágenes hay en train
            train_images_dir = split_to_dir['train']
            all_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(all_images) > 100:  # Solo redistribuir si hay suficientes imágenes
                print("Todas las imágenes están en 'train'. Redistribuyendo automáticamente en train/valid/test (70%/20%/10%)")
                do_auto_split = True
                
                # Obtener todas las imágenes y labels
                image_label_pairs = []
                for img_file in all_images:
                    img_path = os.path.join(train_images_dir, img_file)
                    base_name = os.path.splitext(img_file)[0]
                    label_file = f"{base_name}.txt"
                    label_path = os.path.join(labels_dir, 'train', label_file)
                    
                    if os.path.exists(label_path):
                        image_label_pairs.append((img_path, label_path))
                
                # Mezclar aleatoriamente
                random.shuffle(image_label_pairs)
                
                # Dividir según proporciones
                total = len(image_label_pairs)
                train_count = int(total * 0.7)
                valid_count = int(total * 0.2)
                
                train_pairs = image_label_pairs[:train_count]
                valid_pairs = image_label_pairs[train_count:train_count+valid_count]
                test_pairs = image_label_pairs[train_count+valid_count:]
                
                # Crear diccionario de pares por split
                split_pairs = {
                    'train': train_pairs,
                    'valid': valid_pairs,
                    'test': test_pairs
                }
                
                # Procesar cada split con la distribución automática
                for split, pairs in split_pairs.items():
                    print(f"Procesando split: {split} ({len(pairs)} imágenes)")
                    
                    for img_path, label_path in tqdm(pairs, desc=f"Copiando {split}"):
                        img_file = os.path.basename(img_path)
                        label_file = os.path.basename(label_path)
                        
                        img_dest = os.path.join(output_dir, split, 'images', img_file)
                        label_dest = os.path.join(output_dir, split, 'labels', label_file)
                        
                        # Copiar imagen
                        shutil.copy2(img_path, img_dest)
                        
                        # Si existe la etiqueta, procesarla
                        if os.path.exists(label_path):
                            # Similar al procesamiento de etiquetas anterior, aplicando remapeo si es necesario
                            if class_mapping:
                                with open(label_path, 'r') as f:
                                    lines = f.readlines()
                                
                                remapped_lines = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            class_id = int(parts[0])
                                            if structure_info.get('structure', {}).get('classes') and class_id < len(structure_info['structure']['classes']):
                                                class_name = structure_info['structure']['classes'][class_id]
                                                mapped_name = class_mapping.get(class_name, class_name)
                                                
                                                if mapped_name in target_classes:
                                                    new_class_id = target_classes.index(mapped_name)
                                                    parts[0] = str(new_class_id)
                                                    remapped_lines.append(' '.join(parts) + '\n')
                                                    class_counts[mapped_name] += 1
                                                else:
                                                    # Ignorar clases que no están en nuestro target
                                                    continue
                                            else:
                                                remapped_lines.append(line)
                                                class_counts[f"class_{class_id}"] += 1
                                        except ValueError:
                                            continue
                                
                                with open(label_dest, 'w') as f:
                                    f.writelines(remapped_lines)
                            else:
                                shutil.copy2(label_path, label_dest)
                                # Contar las clases en el archivo copiado aunque no las remapeemos
                                with open(label_path, 'r') as f:
                                    lines = f.readlines()
                                
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:  # formato YOLO: class_id x_center y_center width height
                                        try:
                                            class_id = int(parts[0])
                                            # Usar nombre de clase si está dentro del rango, o "class_X" si no
                                            if class_id < len(target_classes):
                                                class_name = target_classes[class_id]
                                            else:
                                                class_name = f"class_{class_id}"
                                            
                                            class_counts[class_name] += 1
                                        except ValueError:
                                            continue
                        else:
                            # Crear archivo vacío
                            with open(label_dest, 'w') as f:
                                pass
                        
                        total_images += 1
        
        # Si no hacemos redistribución automática, procesar normalmente cada split encontrado
        if not do_auto_split:
            # Procesar cada split
            for split in available_splits:
                split_images_dir = split_to_dir[split]
                split_name = 'val' if split == 'valid' else split
                split_labels_dir = os.path.join(labels_dir, split_name)
                
                if not os.path.exists(split_labels_dir):
                    # Intentar inferir el directorio de etiquetas si no existe exactamente con ese nombre
                    candidates = [
                        os.path.join(labels_dir, split_name),
                        os.path.join(labels_dir, 'train' if split_name == 'val' else 'val' if split_name == 'train' else split_name),
                        labels_dir  # Como último recurso, usar el directorio principal
                    ]
                    
                    for candidate in candidates:
                        if os.path.exists(candidate) and os.path.isdir(candidate):
                            split_labels_dir = candidate
                            break
                
                print(f"Procesando split: {split}")
                print(f"  Directorio de imágenes: {split_images_dir}")
                print(f"  Directorio de etiquetas: {split_labels_dir}")
                
                # Obtener todas las imágenes
                img_files = [f for f in os.listdir(split_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Limitar si es necesario
                if sample_size > 0 and len(img_files) > sample_size:
                    img_files = random.sample(img_files, sample_size)
                
                print(f"  Procesando {len(img_files)} imágenes")
                
                for img_file in tqdm(img_files, desc=f"Procesando {split}", unit="img"):
                    img_path = os.path.join(split_images_dir, img_file)
                    base_name = os.path.splitext(img_file)[0]
                    label_file = f"{base_name}.txt"
                    label_path = os.path.join(split_labels_dir, label_file)
                    
                    img_dest = os.path.join(output_dir, split, 'images', img_file)
                    label_dest = os.path.join(output_dir, split, 'labels', label_file)
                    
                    # Copiar imagen
                    shutil.copy2(img_path, img_dest)
                    
                    # Si existe la etiqueta, procesarla
                    if os.path.exists(label_path):
                        # Similar al procesamiento de etiquetas de arriba, aplicando remapeo si es necesario
                        if class_mapping:
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                            
                            remapped_lines = []
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        class_id = int(parts[0])
                                        if structure_info.get('structure', {}).get('classes') and class_id < len(structure_info['structure']['classes']):
                                            class_name = structure_info['structure']['classes'][class_id]
                                            mapped_name = class_mapping.get(class_name, class_name)
                                            
                                            if mapped_name in target_classes:
                                                new_class_id = target_classes.index(mapped_name)
                                                parts[0] = str(new_class_id)
                                                remapped_lines.append(' '.join(parts) + '\n')
                                                class_counts[mapped_name] += 1
                                            else:
                                                # Ignorar clases que no están en nuestro target
                                                continue
                                        else:
                                            remapped_lines.append(line)
                                            class_counts[f"class_{class_id}"] += 1
                                    except ValueError:
                                        continue
                            
                            with open(label_dest, 'w') as f:
                                f.writelines(remapped_lines)
                        else:
                            shutil.copy2(label_path, label_dest)
                            # Contar las clases en el archivo copiado aunque no las remapeemos
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                            
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:  # formato YOLO: class_id x_center y_center width height
                                    try:
                                        class_id = int(parts[0])
                                        # Usar nombre de clase si está dentro del rango, o "class_X" si no
                                        if class_id < len(target_classes):
                                            class_name = target_classes[class_id]
                                        else:
                                            class_name = f"class_{class_id}"
                                        
                                        class_counts[class_name] += 1
                                    except ValueError:
                                        continue
                    else:
                        # Crear archivo vacío
                        with open(label_dest, 'w') as f:
                            pass
                    
                    total_images += 1
    
    # Crear archivo data.yaml
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(target_classes),
        'names': target_classes
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nEstructura de datos preparada en: {output_dir}")
    print(f"Total de imágenes: {total_images}")
    print("Conteo de clases:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {cls}: {count}")
    
    return {'total_images': total_images, 'class_counts': class_counts}

def main():
    parser = argparse.ArgumentParser(description="Dataset Processor Tool - Conversión y procesamiento avanzado de datasets")
    
    # Argumentos básicos
    parser.add_argument('--input-dir', type=str, required=True, help='Directorio de entrada con el dataset original')
    parser.add_argument('--output-dir', type=str, required=True, help='Directorio de salida para guardar el dataset procesado')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Archivo de configuración YAML con el mapeo de clases')
    
    # Opciones de procesamiento
    parser.add_argument('--input-format', type=str, choices=['auto', 'coco', 'yolo', 'raw'], default='auto', 
                        help='Formato del dataset de entrada (auto=detección automática)')
    parser.add_argument('--use-remapping', action='store_true', help='Aplicar remapeo de clases según el archivo de configuración')
    parser.add_argument('--allow-new-classes', action='store_true', help='Permitir clases nuevas no definidas en el mapeo')
    parser.add_argument('--include-empty', action='store_true', help='Incluir imágenes sin anotaciones')
    parser.add_argument('--sample-size', type=int, default=0, help='Número de imágenes a incluir por split (0=todas)')
    
    args = parser.parse_args()
    
    # Crear directorios de salida si no existen
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar configuración de mapeo de clases
    if args.use_remapping:
        class_mapping, target_classes = load_mapping_config(args.config)
        print("Mapeo de clases cargado desde:", args.config)
        print("Clases objetivo:", target_classes)
    else:
        class_mapping, target_classes = None, None
    
    # Detectar automáticamente el formato del dataset si es 'auto'
    if args.input_format == 'auto':
        structure_info = detect_dataset_structure(args.input_dir)
        format_type = structure_info['type']
        print(f"Formato de dataset detectado: {format_type}")
    else:
        format_type = args.input_format
    
    # Procesar según el formato
    if format_type == 'coco':
        # Convertir de COCO a YOLO con remapeo integrado
        stats = convert_coco_to_yolo(
            args.input_dir, 
            args.output_dir, 
            class_mapping if args.use_remapping else None,
            target_classes if args.use_remapping else None,
            args.sample_size
        )
        
    elif format_type == 'raw':
        # Preparar dataset con estructura no estándar
        stats = prepare_standard_structure(
            args.input_dir,
            args.output_dir,
            class_mapping if args.use_remapping else None,
            target_classes if args.use_remapping else None,
            args.sample_size
        )
        
    elif format_type == 'yolo':
        # Si es YOLO con estructura estándar, solo aplicar remapeo
        data_yaml_path = os.path.join(args.input_dir, 'data.yaml')
        
        # Cargar clases actuales del dataset
        current_classes = []
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                    if 'names' in data_yaml:
                        current_classes = data_yaml['names']
            except Exception as e:
                print(f"Error al cargar data.yaml: {e}")
                current_classes = [f"class_{i}" for i in range(100)]  # Fallback
        
        # Verificar que tenemos clases objetivo 
        if args.use_remapping and not target_classes:
            print("Error: Se solicitó remapeo pero no se pudieron cargar las clases objetivo")
            return
        
        # Si no queremos remapear, mantener las clases originales
        if not args.use_remapping:
            target_classes = current_classes
        
        # Procesamiento por splits
        total_images = 0
        total_annotations = 0
        class_counts = Counter()
        
        # Procesar cada split (train, valid, test)
        for split in ['train', 'valid', 'test']:
            stats = process_yolo_dataset(
                args.input_dir, 
                args.output_dir, 
                split, 
                current_classes, 
                class_mapping if args.use_remapping else None, 
                target_classes, 
                args.allow_new_classes, 
                args.include_empty,
                args.sample_size
            )
            
            if stats:
                total_images += stats['total_images']
                total_annotations += stats['total_annotations']
                for cls, count in stats['class_counts'].items():
                    class_counts[cls] += count
        
        # Crear archivo data.yaml en el directorio de salida
        data_yaml = {
            'train': './train/images',
            'val': './valid/images',
            'test': './test/images',
            'nc': len(target_classes),
            'names': target_classes
        }
        
        with open(os.path.join(args.output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\nProcesamiento de dataset YOLO completado:")
        print(f"Total de imágenes: {total_images}")
        print(f"Total de anotaciones: {total_annotations}")
        print(f"Conteo de clases:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {cls}: {count}")
    
    else:
        print(f"⚠️ No se pudo determinar el formato del dataset o no es compatible")
        return
    
    print("\n✅ Dataset procesado correctamente!")
    print(f"Dataset de salida: {args.output_dir}")
    print(f"Archivo de configuración: {args.output_dir}/data.yaml")

if __name__ == "__main__":
    main()
