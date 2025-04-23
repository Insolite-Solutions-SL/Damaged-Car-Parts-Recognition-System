#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para convertir el dataset CarDD desde formato COCO a formato YOLO.
Este script convierte las anotaciones de formato JSON (COCO) a formato TXT (YOLO)
y reorganiza las imágenes y etiquetas según la estructura requerida.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm

# Mapeo de clases de CarDD a nuestras clases
CLASS_MAPPING = {
    "scratch": "dent",  # Rascado → abolladura
    "crack": "damaged window",  # Grieta → ventana dañada
    "collapse": "damaged door",  # Colapso → puerta dañada
    "breakage": "damaged bumper",  # Rotura → parachoques dañado
    "depression": "dent",  # Depresión → abolladura
    "part_off": "damaged bumper"  # Parte desprendida → parachoques dañado
}

# Lista de clases en nuestro modelo
OUR_CLASSES = [
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

def get_class_id(category_id, categories, use_custom_mapping=True):
    """
    Obtiene el ID de clase según nuestro sistema.
    
    Args:
        category_id (int): ID de categoría en COCO
        categories (list): Lista de categorías del dataset COCO
        use_custom_mapping (bool): Si se debe usar el mapeo personalizado de clases
        
    Returns:
        int: ID de clase en nuestro sistema
    """
    # Obtener el nombre de la categoría original
    category_name = next((cat["name"] for cat in categories if cat["id"] == category_id), None)
    
    if category_name is None:
        return None
    
    if use_custom_mapping:
        # Usar mapeo personalizado para convertir a nuestras clases
        mapped_class = CLASS_MAPPING.get(category_name, category_name)
        # Obtener el índice en nuestras clases
        if mapped_class in OUR_CLASSES:
            return OUR_CLASSES.index(mapped_class)
        return None
    else:
        # Solo usar el índice de la categoría original
        return category_id - 1  # Restar 1 porque YOLO empieza en 0

def convert_coco_to_yolo(coco_dir, output_dir, use_custom_mapping=True):
    """
    Convierte un dataset en formato COCO a formato YOLO.
    
    Args:
        coco_dir (str): Directorio con el dataset en formato COCO
        output_dir (str): Directorio donde se guardará el dataset en formato YOLO
        use_custom_mapping (bool): Si se debe usar el mapeo personalizado de clases
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear estructura de directorios para YOLO
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Detectar los directorios en COCO y mapearlos a splits YOLO
    coco_dirs = {}
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
    
    if not coco_dirs:
        print("⚠️ No se encontraron directorios train*, val* o test* en el dataset COCO")
        return
        
    print(f"Directorios de imágenes encontrados: {coco_dirs}")
    
    # Buscar archivos de anotaciones (instances_train*.json, instances_val*.json, etc.)
    ann_files = {}
    ann_dir = os.path.join(coco_dir, 'annotations')
    
    if not os.path.exists(ann_dir):
        print("⚠️ No se encontró el directorio 'annotations'")
        return
        
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
        return
        
    print(f"Archivos de anotaciones encontrados: {ann_files}")
    
    # Contador total de imágenes y etiquetas
    total_images = 0
    total_annotations = 0
    classes_count = {cls: 0 for cls in OUR_CLASSES}
    
    # Procesar cada conjunto (train, val, test)
    for coco_split, yolo_split in [('train', 'train'), ('valid', 'valid'), ('test', 'test')]:
        # Verificar que exista el directorio para este split
        if coco_split not in coco_dirs or coco_split not in ann_files:
            print(f"Omitiendo split {coco_split} por falta de datos")
            continue
        
        coco_img_dir = os.path.join(coco_dir, coco_dirs[coco_split])
        ann_file = os.path.join(ann_dir, ann_files[coco_split])
        
        print(f"\nProcesando split {coco_split} ({coco_img_dir})")
        
        # Cargar archivo de anotaciones COCO
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Extraer información relevante
        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Crear mapeo de ID de categoría COCO a índice de clase YOLO
        if use_custom_mapping:
            category_map = create_category_mapping(categories)
        else:
            # Si no usamos mapeo personalizado, usar índices directos
            category_map = {cat_id: idx for idx, (cat_id, _) in enumerate(categories.items())}
        
        # Organizar anotaciones por imagen
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Contador para este split
        split_images = 0
        split_annotations = 0
        
        # Procesar cada imagen
        for img_id, img_info in images.items():
            img_file = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Comprobar si la imagen tiene anotaciones
            if img_id not in annotations_by_image:
                continue
            
            # Path a la imagen original
            src_img_path = os.path.join(coco_img_dir, img_file)
            if not os.path.exists(src_img_path):
                print(f"Advertencia: No se encontró la imagen {src_img_path}")
                continue
            
            # Path para guardar la imagen
            dst_img_path = os.path.join(output_dir, yolo_split, 'images', img_file)
            
            # Copiar imagen
            shutil.copy2(src_img_path, dst_img_path)
            total_images += 1
            split_images += 1
            
            # Path para guardar etiquetas YOLO
            label_file = os.path.splitext(img_file)[0] + '.txt'
            dst_label_path = os.path.join(output_dir, yolo_split, 'labels', label_file)
            
            # Convertir anotaciones a formato YOLO
            yolo_annotations = []
            for ann in annotations_by_image[img_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']  # Formato COCO: [x, y, width, height]
                
                # Obtener ID de clase según nuestro sistema
                class_id = get_class_id(category_id, list(categories.items()), use_custom_mapping)
                if class_id is None:
                    continue  # Ignorar clases que no están en nuestro sistema
                
                # Convertir bbox a formato YOLO
                yolo_bbox = convert_bbox_coco_to_yolo(img_width, img_height, bbox)
                
                # Guardar anotación YOLO: class_id, x_center, y_center, width, height
                yolo_annotation = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                yolo_annotations.append(yolo_annotation)
                
                # Contar clases
                if use_custom_mapping:
                    class_name = OUR_CLASSES[class_id]
                    classes_count[class_name] += 1
                
                total_annotations += 1
                split_annotations += 1
            
            # Escribir anotaciones YOLO a archivo
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"Split {coco_split}: {split_images} imágenes, {split_annotations} anotaciones")
    
    # Crear archivo data.yaml
    yaml_content = f"""train: {os.path.join(output_dir, 'train', 'images')}
val: {os.path.join(output_dir, 'valid', 'images')}
test: {os.path.join(output_dir, 'test', 'images')}
nc: {len(OUR_CLASSES)}
names: {OUR_CLASSES}
"""
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # Imprimir resumen
    print("\nResumen de la conversión:")
    print(f"Total de imágenes: {total_images}")
    print(f"Total de anotaciones: {total_annotations}")
    print("\nConteo de clases:")
    for cls, count in classes_count.items():
        print(f"- {cls}: {count}")
    
    print(f"\nDataset convertido guardado en: {output_dir}")
    print("\nPróximos pasos:")
    print("1. Entrenar modelo con el nuevo dataset:")
    print(f"   python trainYolov11s.py --data {os.path.join(output_dir, 'data.yaml')} --epochs 100 --batch 16 --device 0")
    print("2. O ejecutar el flujo de trabajo completo:")
    print(f"   python damageDetectionWorkflow.py workflow --data-dir {output_dir} --epochs 100 --device 0")

def create_category_mapping(categories):
    category_map = {}
    for cat_id, cat_name in categories.items():
        if cat_name in CLASS_MAPPING:
            category_map[cat_id] = OUR_CLASSES.index(CLASS_MAPPING[cat_name])
        else:
            category_map[cat_id] = len(OUR_CLASSES)
            OUR_CLASSES.append(cat_name)
    return category_map

def main():
    parser = argparse.ArgumentParser(description="Convierte dataset CarDD de formato COCO a formato YOLO")
    parser.add_argument("--coco-dir", type=str, default="/Users/tastafur/workspace/detection-yolov-with-security/Damaged-Car-Parts-Recognition-System/CarDD_COCO", 
                        help="Directorio con el dataset en formato COCO")
    parser.add_argument("--output-dir", type=str, default="/Users/tastafur/workspace/detection-yolov-with-security/Damaged-Car-Parts-Recognition-System/data_cardd", 
                        help="Directorio donde se guardará el dataset en formato YOLO")
    parser.add_argument("--custom-mapping", action="store_true", 
                        help="Usar mapeo personalizado de clases")
    
    args = parser.parse_args()
    convert_coco_to_yolo(args.coco_dir, args.output_dir, args.custom_mapping)

if __name__ == "__main__":
    main()
