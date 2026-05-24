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
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'notebooks' else SCRIPT_DIR

logger.debug(f"Script dir: {SCRIPT_DIR}")
logger.debug(f"Project root: {PROJECT_ROOT}")

# ==========================================
# 1. CONFIGURACIÓN DEL TABLERO
# ==========================================
SQUARES_X = 11  
SQUARES_Y = 8 
SQUARE_LENGTH = 0.012  
MARKER_LENGTH = 0.009  

DEFAULT_DIR_RGB_REL = 'calibracion/rgb/news/rgb/*.jpg'
DEFAULT_DIR_NIR_REL = 'calibracion/nir/news/nir/*.jpg'

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
    path = Path(path_pattern)
    if path.is_absolute():
        return str(path)

    candidate_project = PROJECT_ROOT / path_pattern
    if glob.glob(str(candidate_project), recursive=True):
        return str(candidate_project)

    candidate_cwd = Path.cwd() / path_pattern
    if glob.glob(str(candidate_cwd), recursive=True):
        return str(candidate_cwd)

    return str(candidate_project)

def find_first_existing_pattern(possible_patterns):
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
    output_path='matriz_homografia_aruco.npy',
    ransac_threshold=5.0
):
    """
    Ejecuta el calibrado de homografía usando únicamente los centros de los marcadores ArUco.
    Inmune a rotaciones de cámara y falta de márgenes blancos.
    """
    try:
        # 1. Configuración del Detector ArUco Puro (OpenCV 4.7+)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        
        # ==========================================
        # 2. PROCESAMIENTO MLOPS: EXTRACCIÓN Y EMPAREJAMIENTO
        # ==========================================
        rutas_rgb = sorted(glob.glob(dir_rgb, recursive=True))
        rutas_nir = sorted(glob.glob(dir_nir, recursive=True))
        
        if not rutas_rgb or not rutas_nir:
            logger.error(f"No se encontraron imágenes. RGB: {len(rutas_rgb)}, NIR: {len(rutas_nir)}")
            return None
        
        logger.info(f"Iniciando pipeline estéreo invariante a rotación en {len(rutas_rgb)} pares...")
        
        puntos_comunes_rgb = []
        puntos_comunes_nir = []
        estadisticas_pares = []
        
        for ruta_rgb, ruta_nir in zip(rutas_rgb, rutas_nir):
            img_rgb = cv2.imread(ruta_rgb, cv2.IMREAD_GRAYSCALE)
            img_nir = cv2.imread(ruta_nir, cv2.IMREAD_GRAYSCALE)
            
            if img_rgb is None or img_nir is None:
                logger.warning(f"❌ Error al leer: {os.path.basename(ruta_rgb)}")
                continue
            
            # Detección de marcadores (No importa la orientación de las imágenes)
            corners_rgb, ids_rgb, _ = aruco_detector.detectMarkers(img_rgb)
            corners_nir, ids_nir, _ = aruco_detector.detectMarkers(img_nir)
            
            num_markers_rgb = len(ids_rgb) if ids_rgb is not None else 0
            num_markers_nir = len(ids_nir) if ids_nir is not None else 0
            
            if num_markers_rgb == 0 or num_markers_nir == 0:
                logger.warning(f"Descartando {os.path.basename(ruta_rgb)}: RGB vio {num_markers_rgb} QRs, NIR vio {num_markers_nir} QRs.")
                continue

            # Mapear ID -> Centro geométrico para la imagen NIR
            centros_nir = {}
            for i, id_nir in enumerate(ids_nir.flatten()):
                # corners_nir[i][0] tiene la forma (4, 2) con las 4 esquinas del QR
                esquinas = corners_nir[i][0]
                cx = np.mean(esquinas[:, 0])
                cy = np.mean(esquinas[:, 1])
                centros_nir[id_nir] = (cx, cy)
            
            puntos_rgb_filtrados = []
            puntos_nir_filtrados = []
            
            # Emparejar con los encontrados en RGB
            for i, id_rgb in enumerate(ids_rgb.flatten()):
                if id_rgb in centros_nir:
                    esquinas_rgb = corners_rgb[i][0]
                    cx_rgb = np.mean(esquinas_rgb[:, 0])
                    cy_rgb = np.mean(esquinas_rgb[:, 1])
                    
                    puntos_rgb_filtrados.append([cx_rgb, cy_rgb])
                    puntos_nir_filtrados.append([centros_nir[id_rgb][0], centros_nir[id_rgb][1]])
            
            if len(puntos_rgb_filtrados) >= 4:
                puntos_comunes_rgb.extend(puntos_rgb_filtrados)
                puntos_comunes_nir.extend(puntos_nir_filtrados)
                estadisticas_pares.append(len(puntos_rgb_filtrados))
                logger.info(f"✅ {os.path.basename(ruta_rgb)}: {len(puntos_rgb_filtrados)} QRs emparejados exactamente.")
            else:
                logger.warning(f"⚠️ {os.path.basename(ruta_rgb)}: Descartado (<4 marcadores en común).")
        
        # ==========================================
        # 3. CÁLCULO ESTADÍSTICO DE LA HOMOGRAFÍA
        # ==========================================
        if len(puntos_comunes_rgb) >= 4:
            pts_rgb = np.array(puntos_comunes_rgb, dtype=np.float32).reshape(-1, 2)
            pts_nir = np.array(puntos_comunes_nir, dtype=np.float32).reshape(-1, 2)
            
            # RANSAC para descartar outliers
            H, mascara = cv2.findHomography(pts_rgb, pts_nir, cv2.RANSAC, ransac_threshold)
            
            inliers = np.sum(mascara)
            total_puntos = len(pts_rgb)
            porcentaje_inliers = (inliers / total_puntos) * 100
            
            report = "\n" + "="*50 + "\n"
            report += "📊 REPORTE ESTADÍSTICO DE CALIBRACIÓN (ARUCO CENTERS)\n"
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
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            np.save(output_path, H)
            logger.info(f"💾 Matriz guardada exitosamente como '{output_path}'")
            
            return H
        else:
            logger.error("⚠️ ERROR CRÍTICO: No hay suficientes puntos en común para calcular la matriz.")
            return None
            
    except Exception as e:
        logger.error(f"Error durante calibración de homografía: {str(e)}", exc_info=True)
        return None

def resolve_output_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)

def main():
    parser = argparse.ArgumentParser(
        description='Calibración de homografía RGB-NIR usando centros ArUco',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dir-rgb', type=str, default=DIR_RGB)
    parser.add_argument('--dir-nir', type=str, default=DIR_NIR)
    # Se mantienen por compatibilidad en la CLI, aunque el algoritmo ahora no depende de ellos
    parser.add_argument('--squares-x', type=int, default=SQUARES_X)
    parser.add_argument('--squares-y', type=int, default=SQUARES_Y)
    parser.add_argument('--square-length', type=float, default=SQUARE_LENGTH)
    parser.add_argument('--marker-length', type=float, default=MARKER_LENGTH)
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='matriz_homografia_aruco.npy',
        help='Ruta de salida para la matriz'
    )
    parser.add_argument('--ransac-threshold', type=float, default=5.0)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    dir_rgb = resolve_glob_pattern(args.dir_rgb)
    dir_nir = resolve_glob_pattern(args.dir_nir)
    
    logger.info("="*60)
    logger.info("CALIBRACIÓN ÓPTICA - HOMOGRAFÍA INVARIANTE")
    logger.info("="*60)
    logger.info(f"Directorio RGB: {dir_rgb}")
    logger.info(f"Directorio NIR: {dir_nir}")
    logger.info("="*60)
    
    H = run_homography_calibration(
        dir_rgb=dir_rgb,
        dir_nir=dir_nir,
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