#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para combinar múltiples datasets de detección de daños en vehículos en una estructura unificada.
Este script organiza los datos en conjuntos de entrenamiento, validación y prueba,
siguiendo la estructura requerida por YOLOv11.
"""

import os
import shutil
import yaml
import argparse
import random
from tqdm import tqdm

def combine_datasets(data_sources=None, output_dir="data", train_ratio=0.80, val_ratio=0.15, test_ratio=0.05):
    """
    Combina múltiples datasets en una estructura unificada para entrenamiento.
    
    Args:
        data_sources (list): Lista de directorios con los datasets a combinar. Si es None,
                             se usa la carpeta 'data' actual.
        output_dir (str): Directorio donde se creará el dataset combinado
        train_ratio (float): Proporción de imágenes para entrenamiento
        val_ratio (float): Proporción de imágenes para validación
        test_ratio (float): Proporción de imágenes para prueba
    """
    # Si no se especifican fuentes, usar el directorio data/
    if data_sources is None:
        # Verificar si la estructura de datos ya existe
        if os.path.exists('data/images'):
            data_sources = ['data']
        else:
            print("Error: No se han especificado fuentes de datos y no existe una estructura en 'data/'")
            print("Por favor, especifique al menos una fuente de datos con --sources")
            return

    # Crear directorios de salida
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Contadores para estadísticas
    total_images = 0
    train_count, val_count, test_count = 0, 0, 0
    
    # Clases disponibles
    class_names = [
        'damaged door', 'damaged window', 'damaged headlight', 
        'damaged mirror', 'dent', 'damaged hood', 
        'damaged bumper', 'damaged wind shield'
    ]
    
    print("Iniciando la combinación de datasets...")
    
    # Procesar cada fuente de datos
    for source_dir in data_sources:
        print(f"Procesando dataset: {source_dir}")
        
        # Buscar imágenes en el directorio
        images = []
        
        # Caso 1: Estructura YOLOv (imágenes en train/valid/test)
        if os.path.exists(os.path.join(source_dir, 'train', 'images')):
            for split in ['train', 'valid', 'test']:
                split_dir = os.path.join(source_dir, split, 'images')
                if os.path.exists(split_dir):
                    for img in os.listdir(split_dir):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(split_dir, img)
                            label_file = os.path.join(source_dir, split, 'labels', os.path.splitext(img)[0] + '.txt')
                            
                            if os.path.exists(label_file):
                                images.append((img_path, label_file, None))  # El split original no importa
        
        # Caso 2: Estructura plana (todas las imágenes en un directorio)
        elif os.path.exists(os.path.join(source_dir, 'images')):
            img_dir = os.path.join(source_dir, 'images')
            label_dir = os.path.join(source_dir, 'labels')
            
            if os.path.exists(img_dir) and os.path.exists(label_dir):
                for img in os.listdir(img_dir):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(img_dir, img)
                        label_file = os.path.join(label_dir, os.path.splitext(img)[0] + '.txt')
                        
                        if os.path.exists(label_file):
                            images.append((img_path, label_file, None))
        
        # Caso 3: Directamente imágenes en el directorio raíz
        else:
            img_dir = source_dir
            for img in os.listdir(img_dir):
                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(img_dir, img)
                    # Intentar encontrar etiqueta con el mismo nombre pero extensión .txt
                    label_file = os.path.join(img_dir, os.path.splitext(img)[0] + '.txt')
                    
                    if os.path.exists(label_file):
                        images.append((img_path, label_file, None))
        
        # Mezclar aleatoriamente las imágenes para tener una distribución equilibrada
        random.shuffle(images)
        
        # Calcular división de los conjuntos
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        # El resto va a test
        
        # Asignar imágenes a los conjuntos
        splits = ['train'] * n_train + ['valid'] * n_val + ['test'] * (n_images - n_train - n_val)
        assert len(splits) == n_images, "Error en la división de conjuntos"
        
        # Copiar imágenes y etiquetas a la estructura correspondiente
        for i, (img_path, label_path, _) in enumerate(tqdm(images, desc="Copiando archivos")):
            split = splits[i]
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            
            # Destinos
            img_dest = os.path.join(output_dir, split, 'images', img_name)
            label_dest = os.path.join(output_dir, split, 'labels', label_name)
            
            # Copiar archivos
            shutil.copy2(img_path, img_dest)
            shutil.copy2(label_path, label_dest)
            
            # Actualizar contadores
            if split == 'train':
                train_count += 1
            elif split == 'valid':
                val_count += 1
            else:
                test_count += 1
            
            total_images += 1
    
    # Crear archivo data.yaml
    yaml_content = {
        'train': './train/images',
        'val': './valid/images',
        'test': './test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("\nResumen de imágenes copiadas:")
    print(f"- Train: {train_count} imágenes")
    print(f"- Validation: {val_count} imágenes")
    print(f"- Test: {test_count} imágenes")
    print(f"Total: {total_images} imágenes")
    
    print(f"\nSe ha creado el archivo data.yaml en {yaml_path}")
    print("\nProceso completado. Ahora puedes entrenar YOLOv11 con el dataset combinado.")

def main():
    parser = argparse.ArgumentParser(description="Combina múltiples datasets de detección de daños en vehículos")
    parser.add_argument("--sources", nargs='+', help="Directorios con datasets a combinar")
    parser.add_argument("--output", type=str, default="data", help="Directorio de salida para el dataset combinado")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proporción de imágenes para entrenamiento")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proporción de imágenes para validación")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Proporción de imágenes para prueba")
    
    args = parser.parse_args()
    
    # Verificar que las proporciones sumen 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Las proporciones deben sumar 1.0, pero suman {total_ratio}")
        print(f"Train: {args.train_ratio}, Validación: {args.val_ratio}, Test: {args.test_ratio}")
        return
    
    combine_datasets(args.sources, args.output, args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main()
