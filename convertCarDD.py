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
    
    # Mapeo de nombres de conjuntos COCO a YOLO
    split_mapping = {
        'train2017': 'train',
        'val2017': 'valid',
        'test2017': 'test'
    }
    
    # Contador total de imágenes y etiquetas
    total_images = 0
    total_annotations = 0
    classes_count = {cls: 0 for cls in OUR_CLASSES}
    
    # Procesar cada conjunto (train, val, test)
    for coco_split, yolo_split in split_mapping.items():
        # Path al archivo de anotaciones COCO
        ann_file = os.path.join(coco_dir, 'annotations', f'instances_{coco_split}.json')
        if not os.path.exists(ann_file):
            print(f"Advertencia: No se encontró el archivo de anotaciones {ann_file}")
            continue
        
        # Cargar anotaciones COCO
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Organizar anotaciones por ID de imagen
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        # Directorio de imágenes de origen
        src_img_dir = os.path.join(coco_dir, coco_split)
        if not os.path.exists(src_img_dir):
            print(f"Advertencia: No se encontró el directorio de imágenes {src_img_dir}")
            continue
        
        # Procesar cada imagen
        print(f"Procesando conjunto {coco_split} → {yolo_split}...")
        for img_info in tqdm(coco_data['images']):
            img_id = img_info['id']
            img_file = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Comprobar si la imagen tiene anotaciones
            if img_id not in image_annotations:
                continue
            
            # Path a la imagen original
            src_img_path = os.path.join(src_img_dir, img_file)
            if not os.path.exists(src_img_path):
                print(f"Advertencia: No se encontró la imagen {src_img_path}")
                continue
            
            # Path para guardar la imagen
            dst_img_path = os.path.join(output_dir, yolo_split, 'images', img_file)
            
            # Copiar imagen
            shutil.copy2(src_img_path, dst_img_path)
            total_images += 1
            
            # Path para guardar etiquetas YOLO
            label_file = os.path.splitext(img_file)[0] + '.txt'
            dst_label_path = os.path.join(output_dir, yolo_split, 'labels', label_file)
            
            # Convertir anotaciones a formato YOLO
            yolo_annotations = []
            for ann in image_annotations[img_id]:
                category_id = ann['category_id']
                bbox = ann['bbox']  # Formato COCO: [x, y, width, height]
                
                # Obtener ID de clase según nuestro sistema
                class_id = get_class_id(category_id, coco_data['categories'], use_custom_mapping)
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
            
            # Escribir anotaciones YOLO a archivo
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
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

def main():
    parser = argparse.ArgumentParser(description="Convierte dataset CarDD de formato COCO a formato YOLO")
    parser.add_argument("--coco-dir", type=str, default="/Users/tastafur/workspace/detection-yolov-with-security/Damaged-Car-Parts-Recognition-System/CarDD_release/CarDD_COCO", 
                        help="Directorio con el dataset en formato COCO")
    parser.add_argument("--output-dir", type=str, default="/Users/tastafur/workspace/detection-yolov-with-security/Damaged-Car-Parts-Recognition-System/data_cardd", 
                        help="Directorio donde se guardará el dataset en formato YOLO")
    parser.add_argument("--use-mapping", action="store_true", default=True,
                        help="Usar mapeo personalizado de clases (CarDD → Nuestro sistema)")
    
    args = parser.parse_args()
    convert_coco_to_yolo(args.coco_dir, args.output_dir, args.use_mapping)

if __name__ == "__main__":
    main()
