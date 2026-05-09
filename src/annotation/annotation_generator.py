"""
Annotation Generator — convierte bboxes proyectados a formatos estándar.

Soporta:
    - COCO JSON  (para Cascade R-CNN / MMDetection)
    - YOLO TXT   (para YOLOv8 estudiante)
    - Label Studio JSON (para revisión y corrección manual)

Clases:
    0: background      (no se anota explícitamente)
    1: mango           (bbox del mango completo — se genera automáticamente)
    2: mango_damaged   (bbox de región dañada — viene del NIRSegmenter)
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime


# Mapeo de clases
CLASS_MAP = {
    "mango":         1,
    "mango_damaged": 2,
}

COCO_CATEGORIES = [
    {"id": 1, "name": "mango",         "supercategory": "fruit"},
    {"id": 2, "name": "mango_damaged", "supercategory": "fruit"},
]


@dataclass
class ImageAnnotation:
    """Anotación completa de una imagen (RGB)."""
    image_id:    int
    image_path:  str                          # ruta relativa a la imagen RGB
    width:       int
    height:      int
    bboxes:      list[dict] = field(default_factory=list)
    # Cada bbox: {'x1', 'y1', 'x2', 'y2', 'class_name', 'class_id', 'confidence', 'source'}
    # source: 'auto_nir' | 'manual' | 'auto_mango'


class AnnotationGenerator:
    """
    Genera archivos de anotación en múltiples formatos.

    Args:
        output_dir:         Directorio donde se guardan las anotaciones.
        confidence_threshold: Confianza mínima para incluir una región automática.
        dataset_name:       Nombre del dataset para metadatos COCO.
    """

    def __init__(
        self,
        output_dir: str | Path,
        confidence_threshold: float = 0.3,
        dataset_name: str = "mango_damage_detection",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.dataset_name = dataset_name

        # Subdirectorios por formato
        (self.output_dir / "coco").mkdir(exist_ok=True)
        (self.output_dir / "yolo" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "label_studio").mkdir(exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_coco(
        self,
        annotations: list[ImageAnnotation],
        split: str = "train",
    ) -> Path:
        """
        Guarda anotaciones en formato COCO JSON.

        Args:
            annotations: Lista de ImageAnnotation.
            split:       'train', 'val', o 'test'.

        Returns:
            Ruta al archivo JSON generado.
        """
        coco_dict = {
            "info": {
                "description": self.dataset_name,
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": COCO_CATEGORIES,
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img_ann in annotations:
            # Imagen
            coco_dict["images"].append({
                "id":        img_ann.image_id,
                "file_name": img_ann.image_path,
                "width":     img_ann.width,
                "height":    img_ann.height,
            })

            # Bboxes filtradas por confianza
            for bbox in img_ann.bboxes:
                if bbox.get("confidence", 1.0) < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                coco_dict["annotations"].append({
                    "id":           ann_id,
                    "image_id":     img_ann.image_id,
                    "category_id":  bbox["class_id"],
                    "bbox":         [x1, y1, w, h],   # COCO: [x, y, width, height]
                    "area":         w * h,
                    "iscrowd":      0,
                    "segmentation": [],
                    # Metadatos extra (no estándar COCO, útil para revisión)
                    "confidence":   bbox.get("confidence", 1.0),
                    "source":       bbox.get("source", "unknown"),
                })
                ann_id += 1

        output_path = self.output_dir / "coco" / f"annotations_{split}.json"
        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=2)

        print(f"[COCO] Guardado: {output_path}")
        print(f"       {len(coco_dict['images'])} imágenes, {len(coco_dict['annotations'])} anotaciones")
        return output_path

    def save_yolo(
        self,
        annotations: list[ImageAnnotation],
        images_dir: str | Path,
        split: str = "train",
    ) -> Path:
        """
        Guarda anotaciones en formato YOLO TXT.
        Un archivo .txt por imagen con líneas: class_id cx cy w h (normalizados 0-1).

        Args:
            annotations: Lista de ImageAnnotation.
            images_dir:  Directorio con las imágenes RGB (para el dataset.yaml).
            split:       'train', 'val', o 'test'.

        Returns:
            Ruta al directorio de labels generado.
        """
        labels_dir = self.output_dir / "yolo" / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_ann in annotations:
            label_lines = []
            for bbox in img_ann.bboxes:
                if bbox.get("confidence", 1.0) < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                W, H = img_ann.width, img_ann.height

                # YOLO format: class_id cx cy w h (normalized)
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w  = (x2 - x1) / W
                h  = (y2 - y1) / H

                # class_id en YOLO es 0-indexed sin background
                # mango=0, mango_damaged=1
                yolo_class_id = bbox["class_id"] - 1

                if w > 0 and h > 0:
                    label_lines.append(
                        f"{yolo_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )

            # Nombre del label = mismo nombre que la imagen
            img_stem = Path(img_ann.image_path).stem
            label_path = labels_dir / f"{img_stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

        # Generar dataset.yaml para YOLOv8
        yaml_path = self.output_dir / "yolo" / "dataset.yaml"
        yaml_content = f"""# YOLOv8 dataset config — {self.dataset_name}
path: {self.output_dir / 'yolo'}
train: images/train
val:   images/val
test:  images/test

nc: 2
names:
  0: mango
  1: mango_damaged
"""
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"[YOLO] Labels guardados en: {labels_dir}")
        print(f"[YOLO] dataset.yaml: {yaml_path}")
        return labels_dir

    def save_label_studio(
        self,
        annotations: list[ImageAnnotation],
        image_base_url: str = "/data/local-files/?d=",
    ) -> Path:
        """
        Guarda anotaciones en formato Label Studio JSON para revisión manual.

        Las anotaciones automáticas aparecen como pre-anotaciones en Label Studio,
        permitiendo al usuario solo corregir las malas en lugar de anotar desde cero.

        Args:
            annotations:     Lista de ImageAnnotation.
            image_base_url:  Prefijo de URL para las imágenes en Label Studio.

        Returns:
            Ruta al archivo JSON generado.
        """
        ls_tasks = []

        for img_ann in annotations:
            predictions = []
            for bbox in img_ann.bboxes:
                if bbox.get("confidence", 1.0) < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                W, H = img_ann.width, img_ann.height

                # Label Studio usa porcentajes
                predictions.append({
                    "from_name": "label",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                    "value": {
                        "x":      (x1 / W) * 100,
                        "y":      (y1 / H) * 100,
                        "width":  ((x2 - x1) / W) * 100,
                        "height": ((y2 - y1) / H) * 100,
                        "rotation": 0,
                        "rectanglelabels": [bbox["class_name"]],
                    },
                    "score": bbox.get("confidence", 1.0),
                })

            ls_tasks.append({
                "data": {
                    "image": f"{image_base_url}{img_ann.image_path}",
                },
                "predictions": [{
                    "model_version": "auto_nir_v1",
                    "score": max((b.get("confidence", 0) for b in img_ann.bboxes), default=0),
                    "result": predictions,
                }],
            })

        output_path = self.output_dir / "label_studio" / "tasks.json"
        with open(output_path, "w") as f:
            json.dump(ls_tasks, f, indent=2)

        print(f"[Label Studio] Guardado: {output_path}")
        print(f"               {len(ls_tasks)} tareas listas para revisión")
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_annotation(
        image_id: int,
        image_path: str,
        width: int,
        height: int,
        damage_bboxes_rgb: list[dict],
        mango_bbox_rgb: tuple[int, int, int, int] = None,
    ) -> ImageAnnotation:
        """
        Construye un ImageAnnotation a partir de los bboxes proyectados.

        Args:
            image_id:          ID único de la imagen.
            image_path:        Ruta relativa a la imagen RGB.
            width, height:     Dimensiones de la imagen RGB.
            damage_bboxes_rgb: Lista de dicts del BBoxProjector.project_regions_to_rgb().
            mango_bbox_rgb:    Bbox del mango completo (x1,y1,x2,y2). Opcional.
                               Si se provee, se agrega como clase 'mango'.

        Returns:
            ImageAnnotation lista para guardar.
        """
        bboxes = []

        # Bbox del mango completo (clase 1)
        if mango_bbox_rgb is not None:
            x1, y1, x2, y2 = mango_bbox_rgb
            bboxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_name": "mango",
                "class_id":   CLASS_MAP["mango"],
                "confidence": 1.0,
                "source":     "auto_mango",
            })

        # Regiones de daño (clase 2)
        for proj in damage_bboxes_rgb:
            if not proj["valid"]:
                continue
            x1, y1, x2, y2 = proj["bbox_rgb"]
            bboxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_name": "mango_damaged",
                "class_id":   CLASS_MAP["mango_damaged"],
                "confidence": proj["confidence"],
                "source":     "auto_nir",
            })

        return ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            width=width,
            height=height,
            bboxes=bboxes,
        )
