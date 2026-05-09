"""
BBox Projector — proyecta bounding boxes del espacio NIR al espacio RGB
usando la matriz de homografía calculada con el tablero Charuco.

La homografía H mapea puntos del espacio RGB → NIR (según cómo fue calculada
en homografia_script.py: pts_rgb → pts_nir). Para proyectar bboxes de NIR a RGB
necesitamos la inversa H_inv.

Transformación de bbox:
    Los 4 vértices del bbox NIR se proyectan individualmente con H_inv,
    y el bbox RGB resultante es el bounding box que los contiene a todos.
    Esto maneja correctamente la distorsión perspectiva (el bbox puede
    rotar/deformarse al proyectar).
"""

import cv2
import numpy as np
from pathlib import Path
from .nir_segmenter import DamageRegion


class BBoxProjector:
    """
    Proyecta bounding boxes del espacio NIR al espacio RGB.

    Args:
        homography_path: Ruta al archivo .npy con la matriz de homografía.
                         Generada por notebooks/homografia_script.py.
        rgb_image_size:  (width, height) de las imágenes RGB.
                         Usado para clampear bboxes fuera de los bordes.
    """

    def __init__(
        self,
        homography_path: str | Path,
        rgb_image_size: tuple[int, int] = None,
    ):
        H = np.load(str(homography_path))
        if H.shape != (3, 3):
            raise ValueError(f"Homografía inválida, shape esperado (3,3), got {H.shape}")

        # H mapea RGB → NIR, necesitamos NIR → RGB
        self.H_nir_to_rgb = np.linalg.inv(H)
        self.H_rgb_to_nir = H
        self.rgb_image_size = rgb_image_size  # (W, H)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project_regions_to_rgb(
        self,
        regions: list[DamageRegion],
        nir_image_size: tuple[int, int] = None,
    ) -> list[dict]:
        """
        Proyecta una lista de DamageRegion del espacio NIR al espacio RGB.

        Args:
            regions:        Lista de DamageRegion del NIRSegmenter.
            nir_image_size: (width, height) de la imagen NIR.
                            Usado para validar que los bboxes estén dentro.

        Returns:
            Lista de dicts con:
                'bbox_nir':  (x1, y1, x2, y2) original en NIR
                'bbox_rgb':  (x1, y1, x2, y2) proyectado en RGB
                'confidence': score del NIRSegmenter
                'area_px':   área original en NIR
                'valid':     bool — bbox proyectado está dentro de la imagen RGB
        """
        results = []
        for region in regions:
            bbox_rgb, valid = self._project_bbox(
                region.bbox_nir,
                nir_image_size=nir_image_size,
            )
            results.append({
                "bbox_nir":   region.bbox_nir,
                "bbox_rgb":   bbox_rgb,
                "confidence": region.confidence,
                "area_px":    region.area_px,
                "valid":      valid,
            })
        return results

    def project_point_to_rgb(self, x: float, y: float) -> tuple[float, float]:
        """Proyecta un punto (x, y) del espacio NIR al espacio RGB."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H_nir_to_rgb)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])

    def project_point_to_nir(self, x: float, y: float) -> tuple[float, float]:
        """Proyecta un punto (x, y) del espacio RGB al espacio NIR."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, self.H_rgb_to_nir)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1])

    def warp_nir_to_rgb(self, nir_image: np.ndarray, rgb_size: tuple[int, int]) -> np.ndarray:
        """
        Transforma la imagen NIR completa al espacio RGB usando la homografía.
        Útil para visualización y verificación de alineación.

        Args:
            nir_image: Imagen NIR (H, W) o (H, W, C)
            rgb_size:  (width, height) de la imagen RGB destino

        Returns:
            nir_warped: Imagen NIR alineada al espacio RGB
        """
        return cv2.warpPerspective(
            nir_image,
            self.H_nir_to_rgb,
            rgb_size,
            flags=cv2.INTER_LINEAR,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _project_bbox(
        self,
        bbox_nir: tuple[int, int, int, int],
        nir_image_size: tuple[int, int] = None,
    ) -> tuple[tuple[int, int, int, int], bool]:
        """
        Proyecta un bbox (x1, y1, x2, y2) de NIR a RGB.

        Proyecta los 4 vértices individualmente y toma el bbox envolvente.
        Esto es correcto bajo transformación perspectiva (los lados del bbox
        pueden no ser paralelos después de la proyección).

        Returns:
            bbox_rgb: (x1, y1, x2, y2) en coords RGB
            valid:    True si el bbox está dentro de la imagen RGB
        """
        x1, y1, x2, y2 = bbox_nir

        # Los 4 vértices del bbox NIR
        corners_nir = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]],
        ], dtype=np.float32)

        # Proyectar al espacio RGB
        corners_rgb = cv2.perspectiveTransform(corners_nir, self.H_nir_to_rgb)
        corners_rgb = corners_rgb.reshape(-1, 2)

        # Bbox envolvente de los 4 vértices proyectados
        rx1 = int(np.floor(corners_rgb[:, 0].min()))
        ry1 = int(np.floor(corners_rgb[:, 1].min()))
        rx2 = int(np.ceil(corners_rgb[:, 0].max()))
        ry2 = int(np.ceil(corners_rgb[:, 1].max()))

        # Clampear a los límites de la imagen RGB
        valid = True
        if self.rgb_image_size is not None:
            W, H = self.rgb_image_size
            rx1_c = max(0, min(rx1, W - 1))
            ry1_c = max(0, min(ry1, H - 1))
            rx2_c = max(0, min(rx2, W - 1))
            ry2_c = max(0, min(ry2, H - 1))

            # Marcar como inválido si el bbox quedó mayormente fuera
            original_area = (rx2 - rx1) * (ry2 - ry1)
            clamped_area = (rx2_c - rx1_c) * (ry2_c - ry1_c)
            if original_area > 0 and clamped_area / original_area < 0.5:
                valid = False

            rx1, ry1, rx2, ry2 = rx1_c, ry1_c, rx2_c, ry2_c

        # Bbox degenerado
        if rx2 <= rx1 or ry2 <= ry1:
            valid = False

        return (rx1, ry1, rx2, ry2), valid
