"""
Annotation Pipeline — orquesta el flujo completo de anotación semi-automática.

Flujo:
    Para cada par (RGB, NIR):
        1. Cargar imágenes
        2. NIRSegmenter → detectar regiones de daño en NIR
        3. BBoxProjector → proyectar bboxes NIR → RGB via homografía
        4. Detectar bbox del mango completo en RGB (contorno más grande)
        5. AnnotationGenerator → guardar en COCO + YOLO + Label Studio
        6. Guardar imágenes de debug para revisión visual

Uso:
    python3 -m src.annotation.pipeline \
        --rgb-dir data/raw/rgb \
        --nir-dir data/raw/nir \
        --homography matriz_homografia_charuco.npy \
        --output-dir data/annotations \
        --debug
"""

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path

from .nir_segmenter import NIRSegmenter
from .bbox_projector import BBoxProjector
from .annotation_generator import AnnotationGenerator, ImageAnnotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def detect_mango_bbox_rgb(rgb_image: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detecta el bbox del mango completo en la imagen RGB.
    Usa segmentación por color HSV (mango verde/amarillo/naranja).

    Returns:
        (x1, y1, x2, y2) o None si no se detecta.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Rango HSV para mango (verde, amarillo, naranja)
    # Verde
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Amarillo-naranja
    mask_yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
    # Naranja-rojo
    mask_orange = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255))

    mango_mask = cv2.bitwise_or(mask_green, mask_yellow)
    mango_mask = cv2.bitwise_or(mango_mask, mask_orange)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mango_mask = cv2.morphologyEx(mango_mask, cv2.MORPH_CLOSE, kernel)
    mango_mask = cv2.morphologyEx(mango_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mango_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    H_img, W_img = rgb_image.shape[:2]
    return (
        max(0, x),
        max(0, y),
        min(W_img, x + w),
        min(H_img, y + h),
    )


def save_debug_image(
    rgb_image: np.ndarray,
    nir_image: np.ndarray,
    nir_debug: dict,
    projected_bboxes: list[dict],
    mango_bbox: tuple | None,
    output_path: Path,
):
    """Guarda imagen de debug con 4 paneles: RGB, NIR, NIR overlay, RGB con bboxes."""
    H, W = rgb_image.shape[:2]
    target_h = 400
    scale = target_h / H

    def resize(img):
        return cv2.resize(img, (int(W * scale), target_h))

    # Panel 1: RGB con bboxes proyectados
    rgb_ann = rgb_image.copy()
    if mango_bbox:
        x1, y1, x2, y2 = mango_bbox
        cv2.rectangle(rgb_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_ann, "mango", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for proj in projected_bboxes:
        if not proj["valid"]:
            continue
        x1, y1, x2, y2 = proj["bbox_rgb"]
        conf = proj["confidence"]
        color = (0, 0, 255)
        cv2.rectangle(rgb_ann, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb_ann, f"dmg {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Panel 2: NIR original
    nir_bgr = cv2.cvtColor(nir_image, cv2.COLOR_GRAY2BGR) if len(nir_image.shape) == 2 else nir_image

    # Panel 3: NIR overlay con detecciones
    nir_overlay = nir_debug.get("overlay", nir_bgr)

    # Panel 4: NIR damage mask
    damage_clean = nir_debug.get("damage_clean", np.zeros_like(nir_image))
    damage_bgr = cv2.cvtColor(damage_clean, cv2.COLOR_GRAY2BGR)

    panels = [
        resize(rgb_ann),
        resize(nir_bgr),
        resize(nir_overlay),
        resize(damage_bgr),
    ]

    # Asegurar mismo alto
    min_h = min(p.shape[0] for p in panels)
    panels = [p[:min_h] for p in panels]

    combined = np.hstack(panels)

    # Labels
    labels = ["RGB + bboxes", "NIR original", "NIR overlay", "Damage mask"]
    for i, label in enumerate(labels):
        x_offset = i * panels[0].shape[1] + 5
        cv2.putText(combined, label, (x_offset, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imwrite(str(output_path), combined)


class AnnotationPipeline:
    """
    Pipeline completo de anotación semi-automática RGB+NIR.

    Args:
        homography_path:      Ruta a matriz_homografia_charuco.npy
        output_dir:           Directorio de salida para anotaciones
        segmenter_params:     Parámetros para NIRSegmenter (dict)
        confidence_threshold: Confianza mínima para incluir anotación automática
        save_debug:           Guardar imágenes de debug para revisión visual
    """

    def __init__(
        self,
        homography_path: str | Path,
        output_dir: str | Path,
        segmenter_params: dict = None,
        confidence_threshold: float = 0.3,
        save_debug: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.save_debug = save_debug

        self.segmenter = NIRSegmenter(**(segmenter_params or {}))
        self.generator = AnnotationGenerator(
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
        )

        # El projector necesita saber el tamaño RGB — se setea al procesar la primera imagen
        self._homography_path = Path(homography_path)
        self._projector = None

    def _get_projector(self, rgb_size: tuple[int, int]) -> BBoxProjector:
        """Inicializa o actualiza el projector con el tamaño de imagen RGB."""
        if self._projector is None or self._projector.rgb_image_size != rgb_size:
            self._projector = BBoxProjector(
                homography_path=self._homography_path,
                rgb_image_size=rgb_size,
            )
        return self._projector

    def process_pair(
        self,
        rgb_path: str | Path,
        nir_path: str | Path,
        image_id: int,
    ) -> ImageAnnotation | None:
        """
        Procesa un par (RGB, NIR) y retorna su ImageAnnotation.

        Args:
            rgb_path:  Ruta a la imagen RGB
            nir_path:  Ruta a la imagen NIR
            image_id:  ID único para esta imagen

        Returns:
            ImageAnnotation o None si falla
        """
        rgb_path = Path(rgb_path)
        nir_path = Path(nir_path)

        # Cargar imágenes
        rgb_img = cv2.imread(str(rgb_path))
        nir_img = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)

        if rgb_img is None:
            logger.warning(f"No se pudo cargar RGB: {rgb_path}")
            return None
        if nir_img is None:
            logger.warning(f"No se pudo cargar NIR: {nir_path}")
            return None

        H_rgb, W_rgb = rgb_img.shape[:2]
        H_nir, W_nir = nir_img.shape[:2]

        # Inicializar projector con tamaño RGB
        projector = self._get_projector((W_rgb, H_rgb))

        # 1. Segmentar daño en NIR
        regions, debug_imgs = self.segmenter.segment(nir_img, return_debug=self.save_debug)
        logger.info(f"  {rgb_path.name}: {len(regions)} regiones detectadas en NIR")

        # 2. Proyectar bboxes NIR → RGB
        projected = projector.project_regions_to_rgb(
            regions, nir_image_size=(W_nir, H_nir)
        )
        valid_count = sum(1 for p in projected if p["valid"])
        logger.info(f"  {rgb_path.name}: {valid_count}/{len(projected)} bboxes válidos en RGB")

        # 3. Detectar bbox del mango completo en RGB
        mango_bbox = detect_mango_bbox_rgb(rgb_img)

        # 4. Construir anotación
        annotation = AnnotationGenerator.build_annotation(
            image_id=image_id,
            image_path=str(rgb_path),
            width=W_rgb,
            height=H_rgb,
            damage_bboxes_rgb=projected,
            mango_bbox_rgb=mango_bbox,
        )

        # 5. Debug image
        if self.save_debug and debug_imgs is not None:
            debug_path = self.output_dir / "debug" / f"{rgb_path.stem}_debug.jpg"
            save_debug_image(
                rgb_img, nir_img, debug_imgs, projected, mango_bbox, debug_path
            )

        return annotation

    def process_dataset(
        self,
        rgb_dir: str | Path,
        nir_dir: str | Path,
        val_split: float = 0.15,
        test_split: float = 0.10,
        pair_suffix_rgb: str = "_rgb",
        pair_suffix_nir: str = "_nir",
    ) -> dict:
        """
        Procesa un directorio completo de pares RGB+NIR.

        Asume que los archivos están emparejados por nombre:
            mango_001_rgb.jpg  ↔  mango_001_nir.jpg
        O simplemente por orden alfabético si no hay sufijo.

        Args:
            rgb_dir:         Directorio con imágenes RGB
            nir_dir:         Directorio con imágenes NIR
            val_split:       Fracción para validación
            test_split:      Fracción para test
            pair_suffix_rgb: Sufijo en nombre de archivo RGB para emparejar
            pair_suffix_nir: Sufijo en nombre de archivo NIR para emparejar

        Returns:
            Dict con estadísticas del procesamiento
        """
        rgb_dir = Path(rgb_dir)
        nir_dir = Path(nir_dir)

        # Encontrar pares
        rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        nir_files = sorted(nir_dir.glob("*.jpg")) + sorted(nir_dir.glob("*.png"))

        # Intentar emparejar por nombre (reemplazando sufijo)
        pairs = self._match_pairs(rgb_files, nir_files, pair_suffix_rgb, pair_suffix_nir)

        if not pairs:
            logger.error(f"No se encontraron pares en {rgb_dir} y {nir_dir}")
            return {"error": "No pairs found"}

        logger.info(f"Encontrados {len(pairs)} pares RGB+NIR")

        # Procesar todos los pares
        all_annotations = []
        for i, (rgb_path, nir_path) in enumerate(pairs):
            logger.info(f"[{i+1}/{len(pairs)}] Procesando: {rgb_path.name}")
            ann = self.process_pair(rgb_path, nir_path, image_id=i + 1)
            if ann is not None:
                all_annotations.append(ann)

        # Split train/val/test
        n = len(all_annotations)
        n_test = max(1, int(n * test_split))
        n_val  = max(1, int(n * val_split))
        n_train = n - n_val - n_test

        train_anns = all_annotations[:n_train]
        val_anns   = all_annotations[n_train:n_train + n_val]
        test_anns  = all_annotations[n_train + n_val:]

        logger.info(f"Split: train={len(train_anns)}, val={len(val_anns)}, test={len(test_anns)}")

        # Guardar en todos los formatos
        self.generator.save_coco(train_anns, split="train")
        self.generator.save_coco(val_anns,   split="val")
        self.generator.save_coco(test_anns,  split="test")

        self.generator.save_yolo(train_anns, images_dir=rgb_dir, split="train")
        self.generator.save_yolo(val_anns,   images_dir=rgb_dir, split="val")
        self.generator.save_yolo(test_anns,  images_dir=rgb_dir, split="test")

        self.generator.save_label_studio(all_annotations)

        stats = {
            "total_pairs":       len(pairs),
            "processed":         len(all_annotations),
            "train":             len(train_anns),
            "val":               len(val_anns),
            "test":              len(test_anns),
            "total_annotations": sum(len(a.bboxes) for a in all_annotations),
            "damage_annotations": sum(
                sum(1 for b in a.bboxes if b["class_name"] == "mango_damaged")
                for a in all_annotations
            ),
        }

        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETADO")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 50)

        return stats

    @staticmethod
    def _match_pairs(
        rgb_files: list[Path],
        nir_files: list[Path],
        suffix_rgb: str,
        suffix_nir: str,
    ) -> list[tuple[Path, Path]]:
        """
        Empareja archivos RGB y NIR por nombre de stem.
        Estrategia 1: reemplazar sufijo (mango_001_rgb → mango_001_nir)
        Estrategia 2: emparejar por orden si los nombres no tienen sufijo
        """
        # Estrategia 1: por sufijo
        nir_by_stem = {f.stem: f for f in nir_files}
        pairs = []

        for rgb_f in rgb_files:
            # Intentar reemplazar sufijo
            nir_stem = rgb_f.stem.replace(suffix_rgb, suffix_nir)
            if nir_stem in nir_by_stem:
                pairs.append((rgb_f, nir_by_stem[nir_stem]))
                continue

            # Intentar stem idéntico
            if rgb_f.stem in nir_by_stem:
                pairs.append((rgb_f, nir_by_stem[rgb_f.stem]))

        # Estrategia 2: por orden si no se encontraron pares por nombre
        if not pairs and len(rgb_files) == len(nir_files):
            logger.warning("No se encontraron pares por nombre, emparejando por orden alfabético")
            pairs = list(zip(rgb_files, nir_files))

        return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de anotación semi-automática RGB+NIR para detección de daño en mango"
    )
    parser.add_argument("--rgb-dir",      required=True, help="Directorio con imágenes RGB")
    parser.add_argument("--nir-dir",      required=True, help="Directorio con imágenes NIR")
    parser.add_argument("--homography",   default="matriz_homografia_charuco.npy",
                        help="Ruta a la matriz de homografía .npy")
    parser.add_argument("--output-dir",   default="data/annotations",
                        help="Directorio de salida para anotaciones")
    parser.add_argument("--confidence",   type=float, default=0.3,
                        help="Confianza mínima para incluir anotación (default: 0.3)")
    parser.add_argument("--min-area",     type=int, default=200,
                        help="Área mínima de región de daño en px² (default: 200)")
    parser.add_argument("--percentile",   type=int, default=25,
                        help="Percentil de intensidad NIR para umbral de daño (default: 25)")
    parser.add_argument("--val-split",    type=float, default=0.15)
    parser.add_argument("--test-split",   type=float, default=0.10)
    parser.add_argument("--debug",        action="store_true",
                        help="Guardar imágenes de debug para revisión visual")
    parser.add_argument("--suffix-rgb",   default="_rgb",
                        help="Sufijo en nombre de archivo RGB para emparejar con NIR")
    parser.add_argument("--suffix-nir",   default="_nir",
                        help="Sufijo en nombre de archivo NIR")

    args = parser.parse_args()

    pipeline = AnnotationPipeline(
        homography_path=args.homography,
        output_dir=args.output_dir,
        segmenter_params={
            "min_damage_area":  args.min_area,
            "damage_percentile": args.percentile,
        },
        confidence_threshold=args.confidence,
        save_debug=args.debug,
    )

    stats = pipeline.process_dataset(
        rgb_dir=args.rgb_dir,
        nir_dir=args.nir_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        pair_suffix_rgb=args.suffix_rgb,
        pair_suffix_nir=args.suffix_nir,
    )

    return 0 if "error" not in stats else 1


if __name__ == "__main__":
    exit(main())
