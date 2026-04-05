import cv2
import numpy as np
import glob
import os
import argparse
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 0. DETECCIÓN AUTOMÁTICA DE RUTAS BASE
# ==========================================
# Detectar si estamos ejecutando desde notebooks/ o desde raíz del proyecto
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'notebooks' else SCRIPT_DIR

logger.debug(f"Script dir: {SCRIPT_DIR}")
logger.debug(f"Project root: {PROJECT_ROOT}")

# ==========================================
# 1. CONFIGURACIÓN DEL TABLERO CHARUCO
# ==========================================
# ⚠️ OBLIGATORIO: Agarra una regla y mide tu hoja impresa físicamente.
# Reemplaza estos valores en metros (ej. si mide 35mm, pon 0.035).
SQUARES_X = 11   # Columnas de tu PDF
SQUARES_Y = 8   # Filas de tu PDF
SQUARE_LENGTH = 0.023  # Lado del cuadrado negro (EN METROS)
MARKER_LENGTH = 0.016  # Lado del código QR interior (EN METROS)

# Rutas por defecto (relativas a la raíz del proyecto)
DEFAULT_DIR_RGB_REL = 'calibracion/rgb/news/rgb/*.jpg'
DEFAULT_DIR_NIR_REL = 'calibracion/nir/news/nir/*.jpg'

# Opciones fallback por si la estructura cambia
POSSIBLE_RGB_PATHS = [
    DEFAULT_DIR_RGB_REL,
    'calibracion/rgb/news/nir/*.jpg',
    'calibracion/rgb/**/*.jpg',
]

POSSIBLE_NIR_PATHS = [
    DEFAULT_DIR_NIR_REL,
    'calibracion/nir/news/rgb/*.jpg',
    'calibracion/nir/**/*.jpg',
]


def resolve_glob_pattern(path_pattern):
    """
    Resuelve patrones glob en este orden:
    1) Ruta absoluta
    2) Relativa a PROJECT_ROOT
    3) Relativa al directorio actual (cwd)
    """
    path = Path(path_pattern)
    if path.is_absolute():
        return str(path)

    candidate_project = PROJECT_ROOT / path_pattern
    if glob.glob(str(candidate_project), recursive=True):
        return str(candidate_project)

    candidate_cwd = Path.cwd() / path_pattern
    if glob.glob(str(candidate_cwd), recursive=True):
        return str(candidate_cwd)

    # Mantener comportamiento predecible aunque no haya match
    return str(candidate_project)


def find_first_existing_pattern(possible_patterns):
    """Encuentra el primer patrón que tenga archivos."""
    for pattern in possible_patterns:
        resolved = resolve_glob_pattern(pattern)
        if glob.glob(resolved, recursive=True):
            return resolved
    return resolve_glob_pattern(possible_patterns[0])


DIR_RGB = find_first_existing_pattern(POSSIBLE_RGB_PATHS)
DIR_NIR = find_first_existing_pattern(POSSIBLE_NIR_PATHS)

def run_homography_calibration(
    dir_rgb,
    dir_nir,
    squares_x=SQUARES_X,
    squares_y=SQUARES_Y,
    square_length=SQUARE_LENGTH,
    marker_length=MARKER_LENGTH,
    output_path='matriz_homografia_charuco.npy',
    ransac_threshold=5.0
):
    """
    Ejecuta el calibrado de homografía para imágenes RGB y NIR usando tableros Charuco.
    
    Args:
        dir_rgb (str): Ruta glob para imágenes RGB
        dir_nir (str): Ruta glob para imágenes NIR
        squares_x (int): Columnas del tablero Charuco
        squares_y (int): Filas del tablero Charuco
        square_length (float): Lado del cuadrado en metros
        marker_length (float): Lado del marcador en metros
        output_path (str): Ruta de salida para la matriz
        ransac_threshold (float): Umbral RANSAC en píxeles
    
    Returns:
        np.ndarray: Matriz de homografía calculada, o None si falla
    """
    try:
        # Setup estricto para OpenCV 4.7+
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), 
            square_length, 
            marker_length, 
            aruco_dict
        )
        
        detector_params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
        
        # ==========================================
        # 2. PROCESAMIENTO MLOPS: EXTRACCIÓN Y EMPAREJAMIENTO
        # ==========================================
        rutas_rgb = sorted(glob.glob(dir_rgb, recursive=True))
        rutas_nir = sorted(glob.glob(dir_nir, recursive=True))
        
        if not rutas_rgb or not rutas_nir:
            logger.error(f"No se encontraron imágenes. RGB: {len(rutas_rgb)}, NIR: {len(rutas_nir)}")
            return None
        
        logger.info(f"Iniciando pipeline estéreo en {len(rutas_rgb)} pares...")
        
        puntos_comunes_rgb = []
        puntos_comunes_nir = []
        estadisticas_pares = []
        
        for ruta_rgb, ruta_nir in zip(rutas_rgb, rutas_nir):
            img_rgb = cv2.imread(ruta_rgb, cv2.IMREAD_GRAYSCALE)
            img_nir = cv2.imread(ruta_nir, cv2.IMREAD_GRAYSCALE)
            
            if img_rgb is None or img_nir is None:
                logger.warning(f"❌ Error al leer: {os.path.basename(ruta_rgb)}")
                continue
            
            # Detección del tablero en ambos espectros
            charucoCorners_rgb, charucoIds_rgb, _, _ = charuco_detector.detectBoard(img_rgb)
            charucoCorners_nir, charucoIds_nir, _, _ = charuco_detector.detectBoard(img_nir)
            
            if charucoCorners_rgb is not None and charucoCorners_nir is not None:
                puntos_rgb_filtrados = []
                puntos_nir_filtrados = []
                
                # Filtro de Correspondencia 1:1 (Solo los IDs que ambas cámaras ven)
                for i, id_rgb in enumerate(charucoIds_rgb):
                    if id_rgb in charucoIds_nir:
                        idx_nir = np.where(charucoIds_nir == id_rgb)[0][0]
                        puntos_rgb_filtrados.append(charucoCorners_rgb[i])
                        puntos_nir_filtrados.append(charucoCorners_nir[idx_nir])
                
                if len(puntos_rgb_filtrados) >= 4:
                    puntos_comunes_rgb.extend(puntos_rgb_filtrados)
                    puntos_comunes_nir.extend(puntos_nir_filtrados)
                    estadisticas_pares.append(len(puntos_rgb_filtrados))
                    logger.info(f"✅ {os.path.basename(ruta_rgb)}: {len(puntos_rgb_filtrados)} IDs emparejados.")
                else:
                    logger.warning(f"⚠️ {os.path.basename(ruta_rgb)}: Descartado (<4 IDs en común).")
            else:
                logger.warning(f"❌ {os.path.basename(ruta_rgb)}: No se detectó tablero.")
        
        # ==========================================
        # 3. CÁLCULO ESTADÍSTICO DE LA HOMOGRAFÍA
        # ==========================================
        if len(puntos_comunes_rgb) >= 4:
            pts_rgb = np.array(puntos_comunes_rgb).reshape(-1, 2)
            pts_nir = np.array(puntos_comunes_nir).reshape(-1, 2)
            
            # RANSAC con umbral configurable
            H, mascara = cv2.findHomography(pts_rgb, pts_nir, cv2.RANSAC, ransac_threshold)
            
            # Cálculos estadísticos para el reporte
            inliers = np.sum(mascara)
            total_puntos = len(pts_rgb)
            porcentaje_inliers = (inliers / total_puntos) * 100
            
            report = "\n" + "="*50 + "\n"
            report += "📊 REPORTE ESTADÍSTICO DE CALIBRACIÓN ÓPTICA\n"
            report += "="*50 + "\n"
            report += f"Pares procesados exitosamente : {len(estadisticas_pares)}\n"
            report += f"Promedio de marcadores por par: {np.mean(estadisticas_pares):.1f}\n"
            report += f"Total de coordenadas cruzadas : {total_puntos}\n"
            report += f"Inliers (Válidos RANSAC)      : {inliers} ({porcentaje_inliers:.2f}%)\n"
            report += f"Outliers (Ruido descartado)   : {total_puntos - inliers}\n"
            report += "-" * 50 + "\n"
            report += "Matriz de Homografía Resultante (H):\n"
            report += str(H) + "\n"
            report += "="*50
            
            logger.info(report)
            
            # Crear directorio si es necesario
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            np.save(output_path, H)
            logger.info(f"💾 Matriz guardada exitosamente como '{output_path}'")
            
            return H
        else:
            logger.error("⚠️ ERROR CRÍTICO: No hay suficientes puntos para calcular la matriz. Revisa las fotos.")
            return None
            
    except Exception as e:
        logger.error(f"Error durante calibración de homografía: {str(e)}", exc_info=True)
        return None


def resolve_output_path(path_str):
    """
    Resuelve ruta de salida:
    - Absoluta: se usa tal cual
    - Relativa: se interpreta desde PROJECT_ROOT
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def main():
    """
    Función principal para ejecutar el calibrado de homografía desde línea de comandos.
    
    Soporta rutas glob relativas y absolutas para imágenes.
    """
    parser = argparse.ArgumentParser(
        description='Calibración óptica de homografía RGB-NIR usando tableros Charuco',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Desde raíz del proyecto (detecta automáticamente)
  python3 -m notebooks.homografia_script
  
  # Desde carpeta notebooks (detecta automáticamente)
  cd notebooks && python3 homografia_script.py
  
  # Con rutas relativas a PROJECT_ROOT
  python3 notebooks/homografia_script.py --dir-rgb calibracion/rgb/news/rgb/*.jpg
  
  # Con rutas absolutas
  python3 notebooks/homografia_script.py --dir-rgb /home/user/project/calibracion/rgb/**/*.jpg
  
  # Con parámetros personalizados del tablero
  python3 notebooks/homografia_script.py --squares-x 12 --squares-y 9 --square-length 0.025
  
  # Especificar ruta de salida
  python3 notebooks/homografia_script.py --output-path ./outputs/homografia.npy --ransac-threshold 3.0
  
  # Verbose mode
  python3 notebooks/homografia_script.py --verbose
        """
    )
    
    parser.add_argument(
        '--dir-rgb',
        type=str,
        default=DIR_RGB,
        help=f'''Ruta glob para imágenes RGB. 
                Soporta:
                - Relativas desde PROJECT_ROOT: "calibracion/rgb/news/rgb/*.jpg"
                - Relativas desde cwd: "../calibracion/rgb/news/rgb/*.jpg"
                - Absolutas: "/home/user/project/calibracion/rgb/**/*.jpg"
                (default: autodetectado = {DIR_RGB})'''
    )
    parser.add_argument(
        '--dir-nir',
        type=str,
        default=DIR_NIR,
        help=f'''Ruta glob para imágenes NIR.
                Soporta:
                - Relativas desde PROJECT_ROOT: "calibracion/nir/news/nir/*.jpg"
                - Relativas desde cwd: "../calibracion/nir/news/nir/*.jpg"
                - Absolutas: "/home/user/project/calibracion/nir/**/*.jpg"
                (default: autodetectado = {DIR_NIR})'''
    )
    parser.add_argument(
        '--squares-x',
        type=int,
        default=SQUARES_X,
        help=f'Columnas del tablero Charuco (default: {SQUARES_X})'
    )
    parser.add_argument(
        '--squares-y',
        type=int,
        default=SQUARES_Y,
        help=f'Filas del tablero Charuco (default: {SQUARES_Y})'
    )
    parser.add_argument(
        '--square-length',
        type=float,
        default=SQUARE_LENGTH,
        help=f'Lado del cuadrado negro en metros (default: {SQUARE_LENGTH})'
    )
    parser.add_argument(
        '--marker-length',
        type=float,
        default=MARKER_LENGTH,
        help=f'Lado del marcador en metros (default: {MARKER_LENGTH})'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='matriz_homografia_charuco.npy',
        help='Ruta de salida para la matriz de homografía (default: matriz_homografia_charuco.npy)'
    )
    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=5.0,
        help='Umbral RANSAC en píxeles (default: 5.0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Activar logging detallado'
    )
    
    args = parser.parse_args()
    
    # Ajustar nivel de logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Resolver rutas
    dir_rgb = resolve_glob_pattern(args.dir_rgb)
    dir_nir = resolve_glob_pattern(args.dir_nir)
    
    logger.info("="*60)
    logger.info("CALIBRACIÓN ÓPTICA - HOMOGRAFÍA RGB-NIR")
    logger.info("="*60)
    logger.info(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"📁 SCRIPT_DIR: {SCRIPT_DIR}")
    logger.info(f"Directorio RGB: {dir_rgb}")
    logger.info(f"Directorio NIR: {dir_nir}")
    logger.info(f"Tablero Charuco: {args.squares_x}x{args.squares_y}")
    logger.info(f"Tamaño cuadrado: {args.square_length}m | Tamaño marcador: {args.marker_length}m")
    logger.info(f"Archivo de salida: {args.output_path}")
    logger.info("="*60)
    
    # Ejecutar calibración
    H = run_homography_calibration(
        dir_rgb=dir_rgb,
        dir_nir=dir_nir,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        output_path=resolve_output_path(args.output_path),
        ransac_threshold=args.ransac_threshold
    )
    
    if H is not None:
        logger.info("✅ Calibración completada exitosamente")
        return 0
    else:
        logger.error("❌ Calibración falló")
        return 1


if __name__ == '__main__':
    exit(main())