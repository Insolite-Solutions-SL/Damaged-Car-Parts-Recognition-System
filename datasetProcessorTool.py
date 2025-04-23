#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Processor Tool - Herramienta avanzada para procesamiento de datasets
===============================================================================

Esta herramienta permite:
1. Convertir datasets en formato COCO a formato YOLO
2. Fusionar múltiples datasets en uno solo
3. Remapear y unificar clases entre diferentes datasets
4. Distribuir imágenes en splits train/valid/test

Es especialmente útil para trabajar con datasets de daños en vehículos
que utilizan diferentes esquemas de etiquetado.

Autor: Insolite Solutions SL
Versión: 1.0.0
"""

import os
import json
import shutil
import argparse
import yaml
from tqdm import tqdm
from collections import defaultdict, Counter

# Mapeo de clases predefinido (se puede sobrescribir con un archivo de configuración)
DEFAULT_CLASS_MAPPING = {
    # Mapeo del dataset CarDD
    "scratch": "dent",                  # Rascado → abolladura
    "crack": "damaged window",          # Grieta → ventana dañada
    "collapse": "damaged door",         # Colapso → puerta dañada
    "breakage": "damaged bumper",       # Rotura → parachoques dañado
    "depression": "dent",               # Depresión → abolladura
    "part_off": "damaged bumper",       # Parte desprendida → parachoques dañado
    
    # Mapeos adicionales para otros datasets
    "dent": "dent",                     # Mantener dent como está
    "damage": "dent",                   # Daño genérico → abolladura 
    "door_damage": "damaged door",      # Daño en puerta → puerta dañada
    "window_damage": "damaged window",  # Daño en ventana → ventana dañada
    "light_damage": "damaged headlight",# Daño en luz → faro dañado
    "mirror_damage": "damaged mirror",  # Daño en espejo → espejo dañado
    "hood_damage": "damaged hood",      # Daño en capó → capó dañado
    "bumper_damage": "damaged bumper",  # Daño en parachoques → parachoques dañado
    "windshield_damage": "damaged wind shield", # Daño en parabrisas → parabrisas dañado
}

# Lista de clases en nuestro modelo final
DEFAULT_TARGET_CLASSES = [
    "damaged door", "damaged window", "damaged headlight",
    "damaged mirror", "dent", "damaged hood",
    "damaged bumper", "damaged wind shield"
]

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
                bbox = ann['bbox']  # Formato COCO: [x, y, width, height]
                
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
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convertir clase si es necesario
                        if use_remapping and orig_classes:
                            if class_id < len(orig_classes):
                                # Obtener nombre de clase original
                                orig_class_name = orig_classes[class_id]
                                # Mapear a nueva clase
                                mapped_class = class_mapping.get(orig_class_name, orig_class_name)
                                if mapped_class in target_classes:
                                    class_id = target_classes.index(mapped_class)
                                else:
                                    # Ignorar clases que no están en nuestro target
                                    continue
                        
                        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        
                        # Actualizar estadísticas
                        if class_id < len(target_classes):
                            class_name = target_classes[class_id]
                            stats['class_counts'][class_name] += 1
                        
                        stats['total_annotations'] += 1
                        stats['splits'][split]['annotations'] += 1
                    
                    except Exception as e:
                        print(f"Error al procesar línea '{line}' en {label_path}: {e}")
            
            # Escribir etiquetas procesadas
            dst_label_path = os.path.join(output_dir, split, 'labels', os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            # Actualizar estadísticas
            stats['total_images'] += 1
            stats['splits'][split]['images'] += 1
    
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

def main():
    parser = argparse.ArgumentParser(description="Dataset Processor Tool - Conversión y fusión de datasets")
    
    # Argumentos principales
    parser.add_argument("--coco-dirs", type=str, nargs='+',
                        help="Directorios con datasets en formato COCO a procesar")
    parser.add_argument("--coco-dir", type=str, default="",
                        help="Directorio con dataset en formato COCO (para compatibilidad)")
    parser.add_argument("--yolo-dir", type=str, default="",
                        help="Directorio con dataset en formato YOLO para remapeo de clases")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directorio donde se guardará el dataset procesado")
    
    # Argumentos de configuración
    parser.add_argument("--config", type=str, default="",
                        help="Archivo de configuración de mapeo de clases (YAML)")
    parser.add_argument("--merge", action="store_true",
                        help="Fusionar múltiples datasets")
    parser.add_argument("--split-ratios", type=str, default="0.7,0.2,0.1",
                        help="Ratios para train,valid,test al fusionar (por defecto: 0.7,0.2,0.1)")
    
    # Opciones avanzadas
    parser.add_argument("--use-remapping", action="store_true",
                        help="Aplicar remapeo de clases a datasets ya en formato YOLO")
    parser.add_argument("--allow-new-classes", action="store_true",
                        help="Permitir clases no definidas en target_classes")
    parser.add_argument("--include-empty", action="store_true",
                        help="Incluir imágenes sin anotaciones")
    
    args = parser.parse_args()
    
    # Cargar configuración de mapeo
    class_mapping, target_classes = load_mapping_config(args.config) if args.config else (DEFAULT_CLASS_MAPPING, DEFAULT_TARGET_CLASSES)
    
    # Mostrar configuración
    print("\nConfiguración de clases:")
    print("Clases objetivo:")
    for i, cls in enumerate(target_classes):
        print(f"{i}: {cls}")
    
    print("\nMapeo de clases:")
    for src, dst in sorted(class_mapping.items()):
        print(f"{src} → {dst}")
    
    # Opción para procesar directamente un dataset en formato YOLO
    if args.yolo_dir:
        print(f"\n=== Procesando dataset YOLO: {args.yolo_dir} ===")
        
        # Verificar que el directorio contiene un dataset YOLO válido
        if not os.path.exists(os.path.join(args.yolo_dir, 'data.yaml')):
            print(f"Error: No se encontró data.yaml en {args.yolo_dir}")
            return
        
        # Cargar las clases actuales del dataset
        current_classes = []
        try:
            with open(os.path.join(args.yolo_dir, 'data.yaml'), 'r') as f:
                yaml_data = yaml.safe_load(f)
                if 'names' in yaml_data:
                    current_classes = yaml_data['names']
                    print(f"Clases actuales en el dataset ({len(current_classes)}):")
                    for i, cls in enumerate(current_classes):
                        print(f"{i}: {cls}")
        except Exception as e:
            print(f"Error al leer data.yaml: {e}")
            return
        
        # Crear directorio de salida
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Crear estructura de directorios
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(args.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, split, 'labels'), exist_ok=True)
        
        # Procesar cada split
        total_images = 0
        total_annotations = 0
        class_counts = defaultdict(int)
        
        for split in ['train', 'valid', 'test']:
            src_img_dir = os.path.join(args.yolo_dir, split, 'images')
            src_lbl_dir = os.path.join(args.yolo_dir, split, 'labels')
            
            dst_img_dir = os.path.join(args.output_dir, split, 'images')
            dst_lbl_dir = os.path.join(args.output_dir, split, 'labels')
            
            if not os.path.exists(src_img_dir) or not os.path.exists(src_lbl_dir):
                print(f"Advertencia: No se encontró el directorio {split} en {args.yolo_dir}")
                continue
                
            images = [f for f in os.listdir(src_img_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Procesando split {split}: {len(images)} imágenes")
            
            for img_file in tqdm(images, desc=f"Procesando {split}"):
                # Ruta del archivo de imagen y etiqueta
                img_path = os.path.join(src_img_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(src_lbl_dir, label_file)
                
                # Verificar si existe el archivo de etiquetas
                if not os.path.exists(label_path):
                    if args.include_empty:
                        # Copiar solo la imagen
                        shutil.copy2(img_path, os.path.join(dst_img_dir, img_file))
                        # Crear archivo de etiquetas vacío
                        with open(os.path.join(dst_lbl_dir, label_file), 'w') as f:
                            pass
                        total_images += 1
                    continue
                
                # Copiar imagen
                shutil.copy2(img_path, os.path.join(dst_img_dir, img_file))
                total_images += 1
                
                # Procesar etiquetas
                new_annotations = []
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
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Remapear clases si se solicitó
                            if args.use_remapping and class_id < len(current_classes):
                                # Obtener nombre de clase actual
                                current_class = current_classes[class_id]
                                # Mapear a nueva clase
                                mapped_class = class_mapping.get(current_class, current_class)
                                
                                if mapped_class in target_classes:
                                    # Usar el ID de la clase mapeada
                                    new_class_id = target_classes.index(mapped_class)
                                    new_annotations.append(f"{new_class_id} {x_center} {y_center} {width} {height}")
                                    class_counts[mapped_class] += 1
                                    total_annotations += 1
                                elif args.allow_new_classes:
                                    # Agregar la clase si no existe
                                    if mapped_class not in target_classes:
                                        target_classes.append(mapped_class)
                                    new_class_id = target_classes.index(mapped_class)
                                    new_annotations.append(f"{new_class_id} {x_center} {y_center} {width} {height}")
                                    class_counts[mapped_class] += 1
                                    total_annotations += 1
                            else:
                                # Mantener la clase original
                                new_annotations.append(line)
                                if class_id < len(current_classes):
                                    class_counts[current_classes[class_id]] += 1
                                else:
                                    class_counts[f"class_{class_id}"] += 1
                                total_annotations += 1
                                
                        except Exception as e:
                            print(f"Error en línea '{line}' de {label_path}: {e}")
                
                # Escribir nuevas etiquetas
                with open(os.path.join(dst_lbl_dir, label_file), 'w') as f:
                    f.write('\n'.join(new_annotations))
        
        # Crear archivo data.yaml
        yaml_content = f"""train: {os.path.join(args.output_dir, 'train', 'images')}
val: {os.path.join(args.output_dir, 'valid', 'images')}
test: {os.path.join(args.output_dir, 'test', 'images')}
nc: {len(target_classes)}
names: {target_classes}
"""
        
        with open(os.path.join(args.output_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        
        # Mostrar resumen
        print("\n=== RESUMEN DE PROCESAMIENTO ===")
        print(f"Total de imágenes: {total_images}")
        print(f"Total de anotaciones: {total_annotations}")
        print("\nConteo de clases:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {cls}: {count}")
        
        print(f"\nDataset procesado guardado en: {args.output_dir}")
        print(f"Archivo de configuración: {os.path.join(args.output_dir, 'data.yaml')}")
        return
    
    # Convertir una lista de datasets COCO
    if args.coco_dirs or args.coco_dir:
        dirs_to_process = args.coco_dirs or [args.coco_dir]
        all_stats = []
        
        for coco_dir in dirs_to_process:
            print(f"\nProcesando dataset: {coco_dir}")
            
            # Determinar directorio de salida
            if len(dirs_to_process) > 1:
                dir_name = os.path.basename(os.path.normpath(coco_dir))
                out_dir = os.path.join(args.output_dir, dir_name)
            else:
                out_dir = args.output_dir
            
            # Convertir el dataset
            stats = convert_dataset(
                coco_dir, 
                out_dir, 
                class_mapping, 
                target_classes,
                allow_new_classes=args.allow_new_classes,
                include_empty=args.include_empty
            )
            
            if stats:
                all_stats.append((coco_dir, stats))
    
        # Si se procesaron múltiples datasets y se solicitó fusión
        if len(all_stats) > 1 and args.merge:
            print("\n=== Fusionando datasets ===")
            
            # Obtener lista de directorios convertidos
            dirs_to_merge = [os.path.join(args.output_dir, os.path.basename(os.path.normpath(d))) for d in dirs_to_process]
            
            # Convertir string de ratios a diccionario
            split_ratios = list(map(float, args.split_ratios.split(',')))
            split_dict = {'train': split_ratios[0], 'valid': split_ratios[1], 'test': split_ratios[2]}
            
            # Fusionar datasets
            merge_output = os.path.join(args.output_dir, "merged")
            merge_datasets(
                dirs_to_merge, 
                merge_output, 
                class_mapping, 
                target_classes, 
                split_dict,
                use_remapping=args.use_remapping
            )
    
    # Fusionar datasets existentes en formato YOLO
    elif args.merge:
        print("\n=== Fusionando datasets existentes ===")
        
        # Buscar subdirectorios que contengan datasets YOLO
        dirs_to_merge = []
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'data.yaml')):
                dirs_to_merge.append(item_path)
        
        if not dirs_to_merge:
            print("No se encontraron datasets YOLO para fusionar")
            return
        
        print(f"Datasets encontrados para fusionar: {len(dirs_to_merge)}")
        for d in dirs_to_merge:
            print(f"- {d}")
        
        # Convertir string de ratios a diccionario
        split_ratios = list(map(float, args.split_ratios.split(',')))
        split_dict = {'train': split_ratios[0], 'valid': split_ratios[1], 'test': split_ratios[2]}
        
        # Fusionar datasets
        merge_output = os.path.join(args.output_dir, "merged")
        merge_datasets(
            dirs_to_merge, 
            merge_output, 
            class_mapping, 
            target_classes, 
            split_dict,
            use_remapping=args.use_remapping
        )
    
    else:
        print("No se especificaron datasets para procesar")
        parser.print_help()

if __name__ == "__main__":
    main()
