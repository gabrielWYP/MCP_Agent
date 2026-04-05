#!/usr/bin/env python3
"""
MLOps Orchestrator Main Entry Point
Ejecuta el agente de orquestación o calibración según los argumentos proporcionados.
"""

import sys
import argparse
import logging
from pathlib import Path

# Agregar el directorio padre al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.process import process as main_process
from utils.logger import logger_singleton as logger

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def run_mlops_orchestrator():
    """
    Ejecuta el orquestador MLOps principal.
    Chequea nuevos datos en S3 y ejecuta training si es necesario.
    """
    try:
        logger.info("="*60)
        logger.info("INICIANDO ORQUESTADOR MLOPS")
        logger.info("="*60)
        
        main_process()
        
        logger.info("="*60)
        logger.info("✅ ORQUESTADOR MLOPS COMPLETADO")
        logger.info("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error en orquestador MLOps: {str(e)}", exc_info=True)
        return 1


def run_homography_calibration(args):
    """
    Ejecuta la calibración de homografía para imágenes RGB-NIR.
    """
    try:
        from notebooks.homografia_script import (
            run_homography_calibration,
            resolve_glob_pattern,
            resolve_output_path,
        )
        
        logger.info("="*60)
        logger.info("INICIANDO CALIBRACIÓN DE HOMOGRAFÍA")
        logger.info("="*60)

        dir_rgb = resolve_glob_pattern(args.dir_rgb)
        dir_nir = resolve_glob_pattern(args.dir_nir)
        output_path = resolve_output_path(args.output_path)

        logger.info(f"Directorio RGB: {dir_rgb}")
        logger.info(f"Directorio NIR: {dir_nir}")
        logger.info(f"Salida matriz: {output_path}")
        
        H = run_homography_calibration(
            dir_rgb=dir_rgb,
            dir_nir=dir_nir,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            output_path=output_path,
            ransac_threshold=args.ransac_threshold
        )
        
        if H is not None:
            logger.info("="*60)
            logger.info("✅ CALIBRACIÓN COMPLETADA EXITOSAMENTE")
            logger.info("="*60)
            return 0
        else:
            logger.error("❌ CALIBRACIÓN FALLÓ")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error en calibración de homografía: {str(e)}", exc_info=True)
        return 1


def main():
    """
    Función principal con CLI para múltiples modos de ejecución.
    """
    parser = argparse.ArgumentParser(
        description='MLOps Orchestrator - Mango Ripeness Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ejecutar orquestador MLOps principal
  python3 main.py
  python3 main.py mlops
  
  # Ejecutar calibración de homografía
  python3 main.py homography
  python3 main.py homography --dir-rgb calibracion/rgb/*/*.jpg
  python3 main.py homography --squares-x 12 --squares-y 9
  python3 main.py homography --verbose
  
  # Mostrar ayuda para cada comando
  python3 main.py -h
  python3 main.py homography -h
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    subparsers.required = False
    
    # Comando MLOPS (por defecto)
    mlops_parser = subparsers.add_parser(
        'mlops',
        help='Ejecutar orquestador MLOps principal'
    )
    mlops_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Activar logging detallado'
    )
    
    # Comando HOMOGRAPHY
    homography_parser = subparsers.add_parser(
        'homography',
        help='Ejecutar calibración de homografía RGB-NIR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    homography_parser.add_argument(
        '--dir-rgb',
        type=str,
        default='calibracion/rgb/news/rgb/*.jpg',
        help='Ruta glob para imágenes RGB (default: calibracion/rgb/news/rgb/*.jpg)'
    )
    homography_parser.add_argument(
        '--dir-nir',
        type=str,
        default='calibracion/nir/news/nir/*.jpg',
        help='Ruta glob para imágenes NIR (default: calibracion/nir/news/nir/*.jpg)'
    )
    homography_parser.add_argument(
        '--squares-x',
        type=int,
        default=11,
        help='Columnas del tablero Charuco (default: 11)'
    )
    homography_parser.add_argument(
        '--squares-y',
        type=int,
        default=8,
        help='Filas del tablero Charuco (default: 8)'
    )
    homography_parser.add_argument(
        '--square-length',
        type=float,
        default=0.023,
        help='Lado del cuadrado negro en metros (default: 0.023)'
    )
    homography_parser.add_argument(
        '--marker-length',
        type=float,
        default=0.016,
        help='Lado del marcador en metros (default: 0.016)'
    )
    homography_parser.add_argument(
        '--output-path',
        type=str,
        default='matriz_homografia_charuco.npy',
        help='Ruta de salida para la matriz (default: matriz_homografia_charuco.npy)'
    )
    homography_parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=5.0,
        help='Umbral RANSAC en píxeles (default: 5.0)'
    )
    homography_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Activar logging detallado'
    )
    
    # Parse y ejecutar
    args = parser.parse_args()
    
    # Ajustar nivel de logging si es necesario
    if hasattr(args, 'verbose') and args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Ejecutar el comando correspondiente
    if args.command == 'homography':
        return run_homography_calibration(args)
    elif args.command == 'mlops' or args.command is None:
        # Default: ejecutar MLOps
        return run_mlops_orchestrator()
    else:
        logger.error(f"Comando desconocido: {args.command}")
        parser.print_help()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)