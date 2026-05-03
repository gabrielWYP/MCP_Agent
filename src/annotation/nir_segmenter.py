"""
NIR Segmenter — detecta regiones de daño mecánico en imágenes NIR.

El daño mecánico temprano en mango produce pardismo interno que se manifiesta
en NIR como regiones de MENOR reflectancia (más oscuras) respecto al tejido sano.
El tejido sano refleja más en NIR; el dañado absorbe más.

Pipeline de segmentación:
    1. Preprocesamiento (CLAHE + blur)
    2. Segmentación del mango (Otsu — separa fondo oscuro del mango claro)
    3. Detección de daño dentro del mango (umbral adaptativo invertido)
    4. Limpieza morfológica (elimina ruido pequeño)
    5. Extracción de bounding boxes

Parámetros clave ajustables según tu setup de cámara NIR.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DamageRegion:
    """Región de daño detectada en imagen NIR."""
    bbox_nir: tuple[int, int, int, int]   # (x1, y1, x2, y2) en coords NIR
    contour: np.ndarray                    # contorno original
    area_px: int                           # área en píxeles
    confidence: float                      # score de confianza (0-1) basado en intensidad


class NIRSegmenter:
    """
    Segmenta regiones de daño mecánico en imágenes NIR de mango.

    Args:
        min_damage_area:    Área mínima en px² para considerar una región como daño.
                            Ajustar según resolución de tu cámara NIR.
        max_damage_area:    Área máxima. Regiones muy grandes suelen ser sombras.
        clahe_clip_limit:   Límite de contraste para CLAHE. Más alto = más contraste.
        clahe_tile_size:    Tamaño de tile para CLAHE (8x8 es estándar).
        blur_kernel:        Kernel para GaussianBlur antes de umbralizar.
        morph_kernel:       Kernel para operaciones morfológicas de limpieza.
        damage_percentile:  Percentil de intensidad dentro del mango para definir
                            el umbral de daño. Default 25 = píxeles en el 25%
                            más oscuro del mango se consideran potencialmente dañados.
    """

    def __init__(
        self,
        min_damage_area: int = 200,
        max_damage_area: int = 50000,
        clahe_clip_limit: float = 3.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        blur_kernel: int = 5,
        morph_kernel: int = 7,
        damage_percentile: int = 25,
    ):
        self.min_damage_area = min_damage_area
        self.max_damage_area = max_damage_area
        self.morph_kernel = morph_kernel
        self.damage_percentile = damage_percentile
        self.blur_kernel = blur_kernel

        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        nir_image: np.ndarray,
        return_debug: bool = False,
    ) -> tuple[list[DamageRegion], dict | None]:
        """
        Detecta regiones de daño en una imagen NIR.

        Args:
            nir_image:    Imagen NIR en escala de grises (H, W) uint8.
                          Si viene en BGR, se convierte automáticamente.
            return_debug: Si True, retorna imágenes intermedias para visualización.

        Returns:
            regions:      Lista de DamageRegion detectadas.
            debug_imgs:   Dict con imágenes intermedias (solo si return_debug=True).
        """
        # Asegurar escala de grises
        if len(nir_image.shape) == 3:
            nir_gray = cv2.cvtColor(nir_image, cv2.COLOR_BGR2GRAY)
        else:
            nir_gray = nir_image.copy()

        # 1. Preprocesamiento
        enhanced = self._preprocess(nir_gray)

        # 2. Máscara del mango (separa fondo)
        mango_mask = self._segment_mango(enhanced)

        # 3. Detección de daño dentro del mango
        damage_mask = self._detect_damage(enhanced, mango_mask)

        # 4. Limpieza morfológica
        damage_clean = self._morphological_cleanup(damage_mask)

        # 5. Extracción de regiones
        regions = self._extract_regions(damage_clean, nir_gray)

        debug_imgs = None
        if return_debug:
            debug_imgs = {
                "original":     nir_gray,
                "enhanced":     enhanced,
                "mango_mask":   mango_mask,
                "damage_mask":  damage_mask,
                "damage_clean": damage_clean,
                "overlay":      self._draw_overlay(nir_gray, regions, mango_mask),
            }

        return regions, debug_imgs

    def segment_from_path(
        self,
        nir_path: str | Path,
        return_debug: bool = False,
    ) -> tuple[list[DamageRegion], dict | None]:
        """Carga imagen NIR desde disco y segmenta."""
        img = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {nir_path}")
        return self.segment(img, return_debug=return_debug)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE + Gaussian blur para mejorar contraste y reducir ruido."""
        enhanced = self.clahe.apply(gray)
        if self.blur_kernel > 1:
            k = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1
            enhanced = cv2.GaussianBlur(enhanced, (k, k), 0)
        return enhanced

    def _segment_mango(self, enhanced: np.ndarray) -> np.ndarray:
        """
        Segmenta el mango del fondo usando Otsu.
        El mango es más brillante en NIR que el fondo oscuro.
        Retorna máscara binaria donde 255 = mango.
        """
        _, mango_mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Limpieza morfológica para cerrar huecos en la máscara del mango
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel * 3, self.morph_kernel * 3)
        )
        mango_mask = cv2.morphologyEx(mango_mask, cv2.MORPH_CLOSE, kernel)
        mango_mask = cv2.morphologyEx(mango_mask, cv2.MORPH_OPEN, kernel)

        # Quedarse solo con el contorno más grande (el mango)
        contours, _ = cv2.findContours(
            mango_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mango_mask = np.zeros_like(mango_mask)
            cv2.drawContours(mango_mask, [largest], -1, 255, thickness=cv2.FILLED)

        return mango_mask

    def _detect_damage(
        self, enhanced: np.ndarray, mango_mask: np.ndarray
    ) -> np.ndarray:
        """
        Detecta regiones oscuras dentro del mango (tejido dañado).

        Estrategia: umbral adaptativo basado en el percentil de intensidad
        dentro de la región del mango. Los píxeles más oscuros que el
        percentil `damage_percentile` se marcan como potencialmente dañados.
        """
        # Solo analizar píxeles dentro del mango
        mango_pixels = enhanced[mango_mask > 0]

        if len(mango_pixels) == 0:
            return np.zeros_like(enhanced)

        # Umbral = percentil de intensidad dentro del mango
        threshold = np.percentile(mango_pixels, self.damage_percentile)

        # Píxeles oscuros dentro del mango = daño potencial
        damage_mask = np.zeros_like(enhanced)
        damage_mask[(enhanced <= threshold) & (mango_mask > 0)] = 255

        return damage_mask

    def _morphological_cleanup(self, damage_mask: np.ndarray) -> np.ndarray:
        """
        Limpieza morfológica:
        - Opening: elimina ruido pequeño (puntos aislados)
        - Closing: cierra huecos pequeños dentro de regiones de daño
        - Dilate: expande ligeramente para capturar bordes del daño
        """
        k = self.morph_kernel
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2, k * 2))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        cleaned = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)

        return cleaned

    def _extract_regions(
        self, damage_mask: np.ndarray, original: np.ndarray
    ) -> list[DamageRegion]:
        """
        Extrae bounding boxes y métricas de cada región de daño.
        Filtra por área mínima/máxima.
        """
        contours, _ = cv2.findContours(
            damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_damage_area or area > self.max_damage_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Confianza: qué tan oscura es la región respecto al mango
            # Más oscura = más probable que sea daño real
            region_pixels = original[y1:y2, x1:x2]
            mean_intensity = float(np.mean(region_pixels))
            # Normalizar: intensidad 0 → confianza 1.0, intensidad 255 → confianza 0.0
            confidence = max(0.0, min(1.0, 1.0 - (mean_intensity / 255.0)))

            regions.append(DamageRegion(
                bbox_nir=(x1, y1, x2, y2),
                contour=contour,
                area_px=int(area),
                confidence=round(confidence, 3),
            ))

        # Ordenar por confianza descendente
        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions

    def _draw_overlay(
        self,
        nir_gray: np.ndarray,
        regions: list[DamageRegion],
        mango_mask: np.ndarray,
    ) -> np.ndarray:
        """Genera imagen de visualización con regiones marcadas."""
        overlay = cv2.cvtColor(nir_gray, cv2.COLOR_GRAY2BGR)

        # Mango mask en verde semitransparente
        green_layer = np.zeros_like(overlay)
        green_layer[mango_mask > 0] = (0, 60, 0)
        overlay = cv2.addWeighted(overlay, 1.0, green_layer, 0.3, 0)

        for region in regions:
            x1, y1, x2, y2 = region.bbox_nir
            # Color según confianza: rojo alto, amarillo medio
            color = (0, int(255 * (1 - region.confidence)), int(255 * region.confidence))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{region.confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return overlay
