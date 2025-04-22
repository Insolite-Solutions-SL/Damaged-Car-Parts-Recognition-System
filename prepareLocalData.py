#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para preparar una estructura local de datasets para el sistema de detección de partes dañadas de coches.
Permite crear una estructura mínima para evaluación o importar datos de un directorio existente.
"""

import os
import shutil
import argparse
import yaml
from pathlib import Path
import random

def create_directory_structure(data_dir, import_from=None, sample_size=50):
    """
    Crea una estructura de directorios para el dataset de detección de daños.
    
    Args:
        data_dir (str): Ruta donde se creará la estructura
        import_from (str, opcional): Ruta desde donde importar datos existentes
        sample_size (int, opcional): Número de imágenes a importar para cada conjunto
    """
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
        print("   python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/data/data.yaml")

def import_data(data_dir, import_from, sample_size=50):
    """
    Importa una muestra de datos desde un directorio existente.
    
    Args:
        data_dir (str): Directorio destino
        import_from (str): Directorio origen
        sample_size (int): Número de imágenes a importar para cada conjunto
    """
    print(f"\nImportando datos desde: {import_from}")
    
    # Buscar imágenes y etiquetas en el directorio origen
    source_images = []
    source_labels = []
    
    # Si el directorio origen tiene estructura train/valid/test
    if os.path.exists(os.path.join(import_from, 'train')):
        # Estructura organizada
        for split in ['train', 'valid', 'test']:
            if os.path.exists(os.path.join(import_from, split, 'images')):
                img_dir = os.path.join(import_from, split, 'images')
                lbl_dir = os.path.join(import_from, split, 'labels')
                
                # Encontrar imágenes disponibles
                imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(imgs) == 0:
                    continue
                    
                # Seleccionar muestra aleatoria
                sample = min(sample_size, len(imgs))
                selected = random.sample(imgs, sample)
                
                # Copiar imágenes y etiquetas correspondientes
                for img in selected:
                    img_src = os.path.join(img_dir, img)
                    img_dst = os.path.join(data_dir, split, 'images', img)
                    shutil.copy2(img_src, img_dst)
                    
                    # Buscar etiqueta correspondiente
                    base = os.path.splitext(img)[0]
                    lbl = f"{base}.txt"
                    lbl_src = os.path.join(lbl_dir, lbl)
                    if os.path.exists(lbl_src):
                        lbl_dst = os.path.join(data_dir, split, 'labels', lbl)
                        shutil.copy2(lbl_src, lbl_dst)
                
                print(f"- Importadas {sample} imágenes para {split}")
    else:
        # Si el directorio tiene estructura plana, importar solo para test
        img_files = [f for f in os.listdir(import_from) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) == 0:
            print("No se encontraron imágenes en el directorio origen.")
            return
        
        # Seleccionar muestra aleatoria
        sample = min(sample_size, len(img_files))
        selected = random.sample(img_files, sample)
        
        # Copiar imágenes para test
        for img in selected:
            img_src = os.path.join(import_from, img)
            img_dst = os.path.join(data_dir, 'test', 'images', img)
            shutil.copy2(img_src, img_dst)
        
        print(f"- Importadas {sample} imágenes para test (sin etiquetas)")
    
    # Contar total de archivos importados
    total_images = sum(len(os.listdir(os.path.join(data_dir, split, 'images'))) for split in ['train', 'valid', 'test'])
    total_labels = sum(len(os.listdir(os.path.join(data_dir, split, 'labels'))) for split in ['train', 'valid', 'test'])
    
    print(f"\nTotal importado: {total_images} imágenes y {total_labels} etiquetas")
    
    if total_images == 0:
        print("\nADVERTENCIA: No se importaron imágenes. Verifique la estructura del directorio origen.")
    
    print("\nPróximos pasos:")
    print("1. Para entrenar con este conjunto de muestra:")
    print("   python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/data/data.yaml")
    print("2. Para ampliar el conjunto de datos:")
    print("   Copie más imágenes y etiquetas manualmente o use combineDatasets.py")

def main():
    parser = argparse.ArgumentParser(description="Prepara una estructura local de datos para el sistema de detección de daños")
    parser.add_argument("--import-from", type=str, help="Directorio desde el que importar datos existentes")
    parser.add_argument("--data-dir", type=str, default="data", help="Directorio donde crear la estructura")
    parser.add_argument("--sample-size", type=int, default=50, help="Número de imágenes a importar para cada conjunto")
    
    args = parser.parse_args()
    create_directory_structure(args.data_dir, args.import_from, args.sample_size)

if __name__ == "__main__":
    main()
