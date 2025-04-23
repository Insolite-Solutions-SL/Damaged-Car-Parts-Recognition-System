#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para preparar una estructura local de datasets para el sistema de detección de partes dañadas de coches.
Permite crear una estructura mínima para evaluación o importar datos de un directorio existente.
Soporta múltiples formatos, incluyendo YOLO y COCO.
"""

import os
import shutil
import argparse
import yaml
import json
import random


def create_directory_structure(data_dir, import_from=None, sample_size=0):
    """
    Crea una estructura de directorios para el dataset de detección de daños.
    
    Args:
        data_dir (str): Ruta donde se creará la estructura
        import_from (str, opcional): Ruta desde donde importar datos existentes
        sample_size (int, opcional): Número de imágenes a importar para cada conjunto (0 = todas)
    """
    # Evitar sobreescribir directorios existentes
    original_data_dir = data_dir
    counter = 1
    while os.path.exists(data_dir):
        if data_dir == "data" or data_dir == "data_fixed":
            # Si es data o data_fixed, crear data_v2, data_v3, etc.
            data_dir = f"data_v{counter}"
        else:
            # Si ya tiene un sufijo, incrementarlo
            if "_v" in data_dir:
                base = data_dir.split("_v")[0]
                data_dir = f"{base}_v{counter}"
            else:
                data_dir = f"{data_dir}_v{counter}"
        counter += 1
    
    if data_dir != original_data_dir:
        print(f"El directorio {original_data_dir} ya existe. Se usará {data_dir} en su lugar.")
    
    print(f"Creando estructura de dataset en: {data_dir}")
    
    # Crear directorios principales
    subdirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images", "test/labels"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    
    # Crear archivo data.yaml
    class_names = [
        'damaged door', 'damaged window', 'damaged headlight', 
        'damaged mirror', 'dent', 'damaged hood', 
        'damaged bumper', 'damaged wind shield'
    ]
    
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(data_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("\nEstructura de directorios creada:")
    print(f"- {data_dir}/")
    for subdir in subdirs:
        print(f"  - {subdir}/")
    print(f"- data.yaml")
    
    # Importar datos si se ha especificado
    if import_from and os.path.exists(import_from):
        import_data(data_dir, import_from, sample_size)
    else:
        print("\nLa estructura está vacía (sin imágenes ni etiquetas).")
        print("Esta estructura permite que los scripts de entrenamiento y evaluación se ejecuten sin errores,")
        print("aunque no es posible generar visualizaciones ni resultados reales sin datos.")
        
        print("\nPróximos pasos:")
        print("1. Para evaluación real, coloque sus imágenes y etiquetas en los directorios correspondientes")
        print("2. O utilice el script combineDatasets.py para preparar el conjunto de datos completo")
        print("3. Para entrenar el modelo:")
        print(f"   python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/{data_dir}/data.yaml")


def coco_to_yolo(coco_annotation, image_width, image_height, category_mapping):
    """
    Convierte anotaciones COCO a formato YOLO.
    
    Args:
        coco_annotation (dict): Anotación COCO
        image_width (int): Ancho de la imagen
        image_height (int): Alto de la imagen
        category_mapping (dict): Mapeo de categoría COCO a índice YOLO
        
    Returns:
        str: Línea de anotación en formato YOLO
    """
    # COCO bbox: [x_min, y_min, width, height]
    x_min, y_min, width, height = coco_annotation['bbox']
    
    # Convertir a formato YOLO: [x_center, y_center, width, height] normalizado
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height
    
    # Obtener clase YOLO desde el category_id de COCO
    category_id = coco_annotation['category_id']
    if category_id not in category_mapping:
        return None  # Ignorar categorías no mapeadas
        
    yolo_class = category_mapping[category_id]
    
    # Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
    return f"{yolo_class} {x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f}"


def process_coco_dataset(data_dir, import_from, sample_size=0):
    """
    Procesa un dataset en formato COCO y lo convierte a formato YOLO.
    Detecta automáticamente cualquier formato de año o número como sufijo (ej: train2017, train2023, train_v1, etc.)
    
    Args:
        data_dir (str): Directorio destino
        import_from (str): Directorio origen con formato COCO
        sample_size (int): Número de imágenes a importar (0 = todas)
    """
    print(f"Detectado formato COCO en: {import_from}")
    
    # Buscar directorios de imágenes (train*, val*, test*)
    img_dirs = {}
    for item in os.listdir(import_from):
        item_path = os.path.join(import_from, item)
        if not os.path.isdir(item_path):
            continue
            
        if item.startswith('train'):
            img_dirs['train'] = item
        elif item.startswith('val'):
            img_dirs['valid'] = item
        elif item.startswith('test'):
            img_dirs['test'] = item
    
    if not img_dirs:
        print("⚠️ No se encontraron directorios train*, val* o test* en el dataset COCO")
        return False
        
    print(f"Directorios de imágenes encontrados: {img_dirs}")
    
    # Buscar archivos de anotaciones (instances_train*.json, instances_val*.json, etc.)
    ann_files = {}
    ann_dir = os.path.join(import_from, 'annotations')
    
    if not os.path.exists(ann_dir):
        print("⚠️ No se encontró el directorio 'annotations'")
        return False
        
    for file in os.listdir(ann_dir):
        if not file.endswith('.json'):
            continue
            
        if file.startswith('instances_train'):
            ann_files['train'] = file
        elif file.startswith('instances_val'):
            ann_files['valid'] = file
        elif file.startswith('instances_test'):
            ann_files['test'] = file
    
    if not ann_files:
        print("⚠️ No se encontraron archivos de anotaciones instances_*.json en el directorio annotations")
        # Esto no es un error fatal, tal vez solo tengamos imágenes de test sin anotaciones
    else:
        print(f"Archivos de anotaciones encontrados: {ann_files}")
    
    # Crear mapeo de categorías COCO a índices YOLO
    category_mapping = {}
    class_names = []
    
    # Intentar cargar categorías de cualquier archivo de anotaciones disponible
    for split in ['train', 'valid', 'test']:
        if split in ann_files:
            annotations_file = os.path.join(ann_dir, ann_files[split])
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                    
                # Crear mapeo de categorías (COCO category_id -> índice YOLO)
                for i, category in enumerate(coco_data['categories']):
                    category_mapping[category['id']] = i
                    
                class_names = [category['name'] for category in coco_data['categories']]
                break  # Una vez que tenemos las categorías, no necesitamos revisar más archivos
            except Exception as e:
                print(f"Error al cargar archivo de anotaciones {annotations_file}: {str(e)}")
                continue
    
    if not class_names and 'train' in img_dirs:
        print("⚠️ No se pudieron cargar las categorías del dataset COCO")
        print("Se usarán las categorías genéricas por defecto")
        class_names = [
            'damaged door', 'damaged window', 'damaged headlight', 
            'damaged mirror', 'dent', 'damaged hood', 
            'damaged bumper', 'damaged wind shield'
        ]
    
    # Crear un nuevo data.yaml con las categorías del dataset COCO
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(os.path.join(data_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    total_images = 0
    
    # Procesar cada split (train, valid, test)
    for yolo_split, img_dir_name in img_dirs.items():
        # Directorios de imágenes
        img_dir = os.path.join(import_from, img_dir_name)
        if not os.path.exists(img_dir):
            continue
            
        # Archivo de anotaciones
        if yolo_split in ann_files:
            ann_file = os.path.join(ann_dir, ann_files[yolo_split])
        else:
            ann_file = None
            
        if not ann_file or not os.path.exists(ann_file):
            # Si no hay anotaciones, copiar solo las imágenes
            print(f"Procesando imágenes de {yolo_split} sin anotaciones")
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if sample_size > 0 and len(img_files) > sample_size:
                img_files = random.sample(img_files, sample_size)
            
            for img_file in img_files:
                shutil.copy2(
                    os.path.join(img_dir, img_file),
                    os.path.join(data_dir, yolo_split, 'images', img_file)
                )
                total_images += 1
            continue
            
        # Cargar anotaciones COCO
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"Error al procesar {ann_file}: {str(e)}")
            continue
        
        # Crear mapeo de imagen_id -> anotaciones
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Crear mapeo de image_id -> file_name
        image_files = {}
        for img in coco_data['images']:
            image_files[img['id']] = {
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height']
            }
        
        # Obtener lista de IDs de imágenes con anotaciones
        image_ids = list(image_annotations.keys())
        if sample_size > 0 and len(image_ids) > sample_size:
            image_ids = random.sample(image_ids, sample_size)
        
        for i, image_id in enumerate(image_ids):
            if i % 20 == 0:
                print(f"Procesando {yolo_split}: {i+1}/{len(image_ids)}")
                
            if image_id not in image_files:
                continue
                
            img_info = image_files[image_id]
            img_file = img_info['file_name']
            img_path = os.path.join(img_dir, img_file)
            
            if not os.path.exists(img_path):
                continue
                
            # Copiar imagen
            shutil.copy2(
                img_path,
                os.path.join(data_dir, yolo_split, 'images', img_file)
            )
            
            # Convertir anotaciones a formato YOLO
            anns = image_annotations[image_id]
            yolo_annotations = []
            
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                    
                yolo_line = coco_to_yolo(
                    ann, 
                    img_info['width'], 
                    img_info['height'],
                    category_mapping
                )
                if yolo_line:
                    yolo_annotations.append(yolo_line)
            
            # Guardar anotaciones YOLO
            if yolo_annotations:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                with open(os.path.join(data_dir, yolo_split, 'labels', label_file), 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                    
            total_images += 1
    
    print(f"\nSe procesaron {total_images} imágenes del dataset COCO")
    return True


def import_data(data_dir, import_from, sample_size=0):
    """
    Importa una muestra de datos desde un directorio existente.
    Optimizado para manejar varias estructuras de directorios.
    
    Args:
        data_dir (str): Directorio destino
        import_from (str): Directorio origen
        sample_size (int): Número de imágenes a importar para cada conjunto (0 = todas)
    """
    print(f"\nImportando datos desde: {import_from}")
    
    # Comprobar si es un dataset COCO
    if os.path.exists(os.path.join(import_from, 'annotations')):
        # Buscar directorios que comiencen con train, val o test
        has_coco_structure = False
        for item in os.listdir(import_from):
            if os.path.isdir(os.path.join(import_from, item)) and (
                item.startswith('train') or item.startswith('val') or item.startswith('test')):
                has_coco_structure = True
                break
                
        if has_coco_structure:
            if process_coco_dataset(data_dir, import_from, sample_size):
                return  # Si se procesó correctamente, salir
    
    # Buscar imágenes y etiquetas en el directorio origen
    images_paths = []
    
    # Variante 1: Estructura tradicional de YOLOv (train/valid/test)
    # Ejemplo: /import_from/train/images/* y /import_from/train/labels/*
    if os.path.exists(os.path.join(import_from, 'train')):
        print("Detectada estructura estándar de YOLOv (train/valid/test)")
        for split in ['train', 'valid', 'test']:
            split_images_dir = os.path.join(import_from, split, 'images')
            split_labels_dir = os.path.join(import_from, split, 'labels')
            
            if os.path.exists(split_images_dir) and os.path.exists(split_labels_dir):
                for img_file in os.listdir(split_images_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(split_images_dir, img_file)
                        base_name = os.path.splitext(img_file)[0]
                        label_file = f"{base_name}.txt"
                        label_path = os.path.join(split_labels_dir, label_file)
                        
                        if os.path.exists(label_path):
                            images_paths.append((img_path, label_path, split))
    
    # Variante 2: Datos invertidos (images/train y labels/train)
    # Ejemplo: /import_from/images/train/* y /import_from/labels/train/*
    elif os.path.exists(os.path.join(import_from, 'images', 'train')) and os.path.exists(os.path.join(import_from, 'labels', 'train')):
        print("Detectada estructura invertida (images/train y labels/train)")
        for img_file in os.listdir(os.path.join(import_from, 'images', 'train')):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(import_from, 'images', 'train', img_file)
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(import_from, 'labels', 'train', label_file)
                
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
    
    # Variante 3: Estructura plana con imágenes y etiquetas en un mismo nivel
    # Ejemplo: /import_from/images/* y /import_from/labels/*
    elif os.path.exists(os.path.join(import_from, 'images')) and os.path.exists(os.path.join(import_from, 'labels')):
        print("Detectada estructura plana (images/ y labels/)")
        for img_file in os.listdir(os.path.join(import_from, 'images')):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(import_from, 'images', img_file)
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(import_from, 'labels', label_file)
                
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
    
    # Variante 4: Estructura completamente plana
    # Ejemplo: /import_from/* con imágenes y *.txt correspondientes
    else:
        print("Buscando archivos en el directorio raíz...")
        # Buscar todas las imágenes y comprobar si tienen etiquetas correspondientes
        for file in os.listdir(import_from):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(import_from, file)
                base_name = os.path.splitext(file)[0]
                label_file = f"{base_name}.txt"
                label_path = os.path.join(import_from, label_file)
                
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
        print("⚠️ ADVERTENCIA: No se encontraron imágenes con etiquetas correspondientes.")
        print("Verifica que la estructura del directorio sea una de las siguientes:")
        print("1. /train/images/ y /train/labels/")
        print("2. /images/train/ y /labels/train/")
        print("3. /images/ y /labels/")
        print("4. Archivos planos: imagen.jpg y imagen.txt")
        print("5. Formato COCO: /annotations y /train2017, /val2017, etc.")
        return
    
    # Mezclar aleatoriamente los datos
    random.shuffle(images_paths)
    
    # Contar cuántos archivos tenemos por split
    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    for _, _, split in images_paths:
        split_counts[split] += 1
    
    print(f"\nEncontrados {len(images_paths)} pares de imágenes/etiquetas:")
    for split, count in split_counts.items():
        print(f"- {split}: {count} archivos")
    
    # Limitar a sample_size por split si se especifica
    max_counts = {}
    for split in ['train', 'valid', 'test']:
        max_counts[split] = min(sample_size, split_counts[split]) if sample_size > 0 else split_counts[split]
    
    counts = {'train': 0, 'valid': 0, 'test': 0}
    total_copied = 0
    
    print("\nCopiando archivos...")
    # Copiar archivos
    for img_path, label_path, split in images_paths:
        if counts[split] >= max_counts[split]:
            continue
            
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        
        # Destinos
        img_dest = os.path.join(data_dir, split, 'images', img_filename)
        label_dest = os.path.join(data_dir, split, 'labels', label_filename)
        
        # Copiar archivos
        shutil.copy2(img_path, img_dest)
        shutil.copy2(label_path, label_dest)
        
        counts[split] += 1
        total_copied += 1
        
        # Mostrar progreso
        if total_copied % 20 == 0:
            print(f"Copiados {total_copied} archivos...")
    
    print(f"\nImportación completada. Total copiado: {total_copied} pares de imágenes/etiquetas")
    print(f"- train: {counts['train']} imágenes")
    print(f"- valid: {counts['valid']} imágenes")
    print(f"- test: {counts['test']} imágenes")
    
    print("\nPróximos pasos:")
    print("1. Para entrenar con este conjunto:")
    print(f"   python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/{data_dir}/data.yaml")
    print("2. Para combinar con otros datasets:")
    print(f"   python combineDatasets.py --sources {data_dir} data_cardd --output data_combined")


def main():
    parser = argparse.ArgumentParser(description="Prepara una estructura local de datos para el sistema de detección de daños")
    parser.add_argument("--import-from", type=str, help="Directorio desde el que importar datos existentes")
    parser.add_argument("--data-dir", type=str, default="data_fixed", help="Directorio donde crear la estructura")
    parser.add_argument("--sample-size", type=int, default=0, help="Número de imágenes a importar para cada conjunto (0 = todas)")
    
    args = parser.parse_args()
    create_directory_structure(args.data_dir, args.import_from, args.sample_size)


if __name__ == "__main__":
    main()
