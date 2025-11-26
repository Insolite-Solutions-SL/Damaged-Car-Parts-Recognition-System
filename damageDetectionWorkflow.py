#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script unificado para el flujo de trabajo completo de detección de partes dañadas de vehículos.
Este script integra todas las etapas: preparación de datos, entrenamiento, evaluación y visualización.
"""

import os
import argparse
import subprocess

def check_prerequisites():
    """
    Verifica que los requisitos previos estén instalados.
    
    Returns:
        bool: True si todos los requisitos están instalados
    """
    print("Verificando requisitos previos...")
    
    try:
        import ultralytics
        print(f"✓ Ultralytics instalado (versión {ultralytics.__version__})")
    except ImportError:
        print("✗ Ultralytics no está instalado. Instálalo con 'pip install ultralytics'")
        return False
    
    try:
        import yaml
        print("✓ PyYAML instalado")
    except ImportError:
        print("✗ PyYAML no está instalado. Instálalo con 'pip install pyyaml'")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib instalado")
    except ImportError:
        print("✗ Matplotlib no está instalado. Instálalo con 'pip install matplotlib'")
        return False
    
    try:
        import numpy
        print("✓ NumPy instalado")
    except ImportError:
        print("✗ NumPy no está instalado. Instálalo con 'pip install numpy'")
        return False
    
    try:
        import tqdm
        print("✓ tqdm instalado")
    except ImportError:
        print("✗ tqdm no está instalado. Instálalo con 'pip install tqdm'")
        return False
    
    # Verificar que los scripts necesarios existen
    required_scripts = [
        "prepareLocalData.py",
        "combineDatasets.py",
        "trainYolov11s.py",
        "evaluateModel.py"
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"✗ No se encontró el script {script}")
            return False
        print(f"✓ {script} encontrado")
    
    print("Todos los requisitos previos están instalados.")
    return True

def get_abs_path(path):
    """
    Obtiene la ruta absoluta de un archivo o directorio.
    
    Args:
        path (str): Ruta a convertir
        
    Returns:
        str: Ruta absoluta
    """
    return os.path.abspath(path)

def run_prepare_local(args):
    """
    Prepara una estructura local para entrenamiento o evaluación.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    cmd = ["python", "prepareLocalData.py"]
    
    if args.import_from:
        cmd.extend(["--import-from", get_abs_path(args.import_from)])
    
    if args.data_dir:
        cmd.extend(["--data-dir", get_abs_path(args.data_dir)])
    
    if args.sample_size:
        cmd.extend(["--sample-size", str(args.sample_size)])
    
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True)

def run_combine_datasets(args):
    """
    Combina múltiples datasets en una estructura unificada.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    cmd = ["python", "combineDatasets.py"]
    
    if args.sources:
        sources = [get_abs_path(src) for src in args.sources]
        cmd.extend(["--sources"] + sources)
    
    if args.output:
        cmd.extend(["--output", get_abs_path(args.output)])
    
    if args.train_ratio:
        cmd.extend(["--train-ratio", str(args.train_ratio)])
    
    if args.val_ratio:
        cmd.extend(["--val-ratio", str(args.val_ratio)])
    
    if args.test_ratio:
        cmd.extend(["--test-ratio", str(args.test_ratio)])
    
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True)

def run_train(args):
    """
    Entrena un modelo YOLOv11.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    cmd = ["python", "trainYolov11s.py"]
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.data:
        # Usar ruta absoluta para evitar problemas con rutas relativas
        data_path = get_abs_path(args.data)
        cmd.extend(["--data", data_path])
    
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    
    if args.imgsz:
        cmd.extend(["--imgsz", str(args.imgsz)])
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    
    if args.name:
        cmd.extend(["--name", args.name])
    
    if args.project:
        cmd.extend(["--project", args.project])
    
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True)

def run_evaluate(args):
    """
    Evalúa un modelo entrenado.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    cmd = ["python", "evaluateModel.py"]
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.data:
        # Usar ruta absoluta para evitar problemas con rutas relativas
        data_path = get_abs_path(args.data)
        cmd.extend(["--data", data_path])
    
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    
    if args.imgsz:
        cmd.extend(["--imgsz", str(args.imgsz)])
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    if args.samples:
        cmd.extend(["--samples", str(args.samples)])
    
    if args.list_models:
        cmd.append("--list-models")
    
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True)

def run_continue_training(args):
    """
    Continúa el entrenamiento de un modelo existente.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    cmd = ["python", "evaluateModel.py"]
    
    if args.model:
        cmd.extend(["--model", args.model])
    
    if args.data:
        # Usar ruta absoluta para evitar problemas con rutas relativas
        data_path = get_abs_path(args.data)
        cmd.extend(["--data", data_path])
    
    if args.continue_epochs:
        cmd.extend(["--continue-epochs", str(args.continue_epochs)])
    
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    
    if args.imgsz:
        cmd.extend(["--imgsz", str(args.imgsz)])
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    cmd_str = " ".join(cmd)
    print(f"Ejecutando: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True)

def run_workflow(args):
    """
    Ejecuta el flujo de trabajo completo: preparación, entrenamiento y evaluación.
    
    Args:
        args (Namespace): Argumentos del comando
    """
    print("=== INICIANDO FLUJO DE TRABAJO COMPLETO ===")
    
    # Modo local o completo
    if args.local:
        print("\n[1/4] Preparando entorno local...")
        prepare_args = argparse.Namespace(
            import_from=args.import_from,
            data_dir=args.data_dir or "data",
            sample_size=args.sample_size or 50
        )
        run_prepare_local(prepare_args)
    else:
        print("\n[1/4] Combinando datasets...")
        combine_args = argparse.Namespace(
            sources=args.sources,
            output=args.data_dir or "data",
            train_ratio=args.train_ratio or 0.8,
            val_ratio=args.val_ratio or 0.15,
            test_ratio=args.test_ratio or 0.05
        )
        run_combine_datasets(combine_args)
    
    # Entrenamiento
    print("\n[2/4] Entrenando modelo...")
    train_args = argparse.Namespace(
        model=args.model or "yolo11s",
        data=os.path.join(args.data_dir or "data", "data.yaml"),
        epochs=args.epochs or (20 if args.local else 100),
        batch=args.batch or 16,
        imgsz=args.imgsz or 640,
        device=args.device or ("cpu" if args.local else "0"),
        workers=args.workers or 8,
        name=args.name,
        project=args.project or "runs/detect"
    )
    run_train(train_args)
    
    # Evaluación
    print("\n[3/4] Evaluando modelo...")
    evaluate_args = argparse.Namespace(
        model=None,  # Usar el modelo más reciente
        data=os.path.join(args.data_dir or "data", "data.yaml"),
        batch=args.batch or 16,
        imgsz=args.imgsz or 640,
        device=args.device or ("cpu" if args.local else "0"),
        samples=args.samples or 10,
        list_models=False
    )
    run_evaluate(evaluate_args)
    
    # Continuación opcional
    if args.continue_epochs:
        print("\n[4/4] Continuando entrenamiento...")
        continue_args = argparse.Namespace(
            model=None,  # Usar el modelo más reciente
            data=os.path.join(args.data_dir or "data", "data.yaml"),
            continue_epochs=args.continue_epochs,
            batch=args.batch or 16,
            imgsz=args.imgsz or 640,
            device=args.device or ("cpu" if args.local else "0")
        )
        run_continue_training(continue_args)
    else:
        print("\n[4/4] Omitiendo continuación de entrenamiento (use --continue-epochs para activarla)")
    
    print("\n=== FLUJO DE TRABAJO COMPLETO FINALIZADO ===")
    print("Puedes ver los resultados en las carpetas:")
    print(f"- Entrenamiento: {args.project or 'runs/detect'}")
    print("- Visualizaciones: predictions_*/samples")

def main():
    """Función principal que procesa los argumentos y ejecuta los comandos."""
    # Crear parser principal
    parser = argparse.ArgumentParser(
        description="Flujo de trabajo unificado para detección de partes dañadas de vehículos",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Crear subparsers para cada comando
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # 1. Verificar requisitos
    parser_check = subparsers.add_parser("check", help="Verificar requisitos previos")
    
    # 2. Preparar entorno local
    parser_prepare = subparsers.add_parser("prepare-local", help="Preparar estructura local para evaluación")
    parser_prepare.add_argument("--import-from", type=str, help="Directorio desde el que importar datos existentes")
    parser_prepare.add_argument("--data-dir", type=str, help="Directorio donde crear la estructura")
    parser_prepare.add_argument("--sample-size", type=int, help="Número de imágenes a importar para cada conjunto")
    
    # 3. Combinar datasets
    parser_combine = subparsers.add_parser("combine-datasets", help="Combinar múltiples datasets")
    parser_combine.add_argument("--sources", nargs='+', help="Directorios con datasets a combinar")
    parser_combine.add_argument("--output", type=str, help="Directorio de salida para el dataset combinado")
    parser_combine.add_argument("--train-ratio", type=float, help="Proporción de imágenes para entrenamiento")
    parser_combine.add_argument("--val-ratio", type=float, help="Proporción de imágenes para validación")
    parser_combine.add_argument("--test-ratio", type=float, help="Proporción de imágenes para prueba")
    
    # 4. Entrenar modelo
    parser_train = subparsers.add_parser("train", help="Entrenar modelo YOLOv11")
    parser_train.add_argument("--model", type=str, help="Modelo a utilizar (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)")
    parser_train.add_argument("--data", type=str, help="Ruta al archivo data.yaml")
    parser_train.add_argument("--epochs", type=int, help="Número de épocas")
    parser_train.add_argument("--batch", type=int, help="Tamaño del batch")
    parser_train.add_argument("--imgsz", type=int, help="Tamaño de la imagen")
    parser_train.add_argument("--device", type=str, help="Dispositivo (0, 0,1, cpu)")
    parser_train.add_argument("--workers", type=int, help="Número de workers para carga de datos")
    parser_train.add_argument("--name", type=str, help="Nombre para el directorio de resultados")
    parser_train.add_argument("--project", type=str, help="Directorio de proyecto para guardar resultados")
    
    # 5. Evaluar modelo
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluar modelo entrenado")
    parser_evaluate.add_argument("--model", type=str, help="Ruta al modelo a evaluar")
    parser_evaluate.add_argument("--data", type=str, help="Ruta al archivo data.yaml")
    parser_evaluate.add_argument("--batch", type=int, help="Tamaño del batch")
    parser_evaluate.add_argument("--imgsz", type=int, help="Tamaño de la imagen")
    parser_evaluate.add_argument("--device", type=str, help="Dispositivo (0, 0,1, cpu)")
    parser_evaluate.add_argument("--samples", type=int, help="Número de muestras para visualización")
    parser_evaluate.add_argument("--list-models", action="store_true", help="Listar todos los modelos disponibles")
    
    # 6. Continuar entrenamiento
    parser_continue = subparsers.add_parser("continue-training", help="Continuar entrenamiento de un modelo existente")
    parser_continue.add_argument("model", type=str, help="Ruta al modelo a continuar entrenando")
    parser_continue.add_argument("--data", type=str, help="Ruta al archivo data.yaml")
    parser_continue.add_argument("--continue-epochs", type=int, default=20, help="Número de épocas adicionales")
    parser_continue.add_argument("--batch", type=int, help="Tamaño del batch")
    parser_continue.add_argument("--imgsz", type=int, help="Tamaño de la imagen")
    parser_continue.add_argument("--device", type=str, help="Dispositivo (0, 0,1, cpu)")
    
    # 7. Flujo de trabajo completo
    parser_workflow = subparsers.add_parser("workflow", help="Ejecutar flujo de trabajo completo")
    parser_workflow.add_argument("--local", action="store_true", help="Modo local (preparación local en lugar de combinación)")
    parser_workflow.add_argument("--import-from", type=str, help="Directorio desde el que importar datos (modo local)")
    parser_workflow.add_argument("--sources", nargs='+', help="Directorios con datasets a combinar (modo completo)")
    parser_workflow.add_argument("--data-dir", type=str, help="Directorio de datos")
    parser_workflow.add_argument("--model", type=str, help="Modelo a utilizar")
    parser_workflow.add_argument("--epochs", type=int, help="Número de épocas iniciales")
    parser_workflow.add_argument("--continue-epochs", type=int, help="Número de épocas adicionales")
    parser_workflow.add_argument("--batch", type=int, help="Tamaño del batch")
    parser_workflow.add_argument("--imgsz", type=int, help="Tamaño de la imagen")
    parser_workflow.add_argument("--device", type=str, help="Dispositivo (0, 0,1, cpu)")
    parser_workflow.add_argument("--workers", type=int, help="Número de workers para carga de datos")
    parser_workflow.add_argument("--name", type=str, help="Nombre para el directorio de resultados")
    parser_workflow.add_argument("--project", type=str, help="Directorio de proyecto para guardar resultados")
    parser_workflow.add_argument("--sample-size", type=int, help="Número de imágenes a importar (modo local)")
    parser_workflow.add_argument("--train-ratio", type=float, help="Proporción de imágenes para entrenamiento")
    parser_workflow.add_argument("--val-ratio", type=float, help="Proporción de imágenes para validación")
    parser_workflow.add_argument("--test-ratio", type=float, help="Proporción de imágenes para prueba")
    parser_workflow.add_argument("--samples", type=int, help="Número de muestras para visualización")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Si no se especificó un comando, mostrar ayuda
    if not args.command:
        parser.print_help()
        return
    
    # Ejecutar el comando correspondiente
    if args.command == "check":
        check_prerequisites()
    elif args.command == "prepare-local":
        run_prepare_local(args)
    elif args.command == "combine-datasets":
        run_combine_datasets(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "continue-training":
        run_continue_training(args)
    elif args.command == "workflow":
        run_workflow(args)

if __name__ == "__main__":
    main()
