"""
Dataset de madurez de mangos con soporte RGB y RGB+NIR.

Estructura esperada en disco:
    data/
      train/
        verde/
          rgb/img001.jpg
          nir/img001.jpg   ← opcional, si use_nir=False se ignora
        pinton/
        maduro/
      val/
      test/

Las imágenes NIR y RGB se emparejan por nombre de archivo.
El warp NIR→RGB se aplica on-the-fly usando la matriz de homografía
calculada con el tablero Charuco.

Nota sobre normalización NIR:
    Mean/std del canal NIR son estimaciones. Para tu dataset específico,
    calculá los valores reales con compute_nir_stats() antes de entrenar.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASSES = ["sano", "danado"]

# Stats ImageNet para RGB + estimación razonable para NIR monocromático
# Calculá las reales con compute_nir_stats() sobre tu dataset
_MEAN_RGB = [0.485, 0.456, 0.406]
_STD_RGB  = [0.229, 0.224, 0.225]
_MEAN_NIR = [0.45]
_STD_NIR  = [0.22]


class MangoRipeness(Dataset):
    """
    Dataset para clasificación de madurez de mangos.

    Args:
        root:             Carpeta raíz del dataset (contiene train/, val/, test/)
        split:            "train" | "val" | "test"
        use_nir:          Si True, carga y apila el canal NIR → entrada 4-canal (RGBN)
        homography_path:  Path al .npy con la matriz H (NIR→RGB warp)
        transform:        Transform de albumentations. Si None se usa resize+normalize.
        image_size:       Tamaño al que se redimensionan las imágenes (cuadradas)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        use_nir: bool = True,
        homography_path: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        image_size: int = 224,
    ):
        self.root = Path(root)
        self.split = split
        self.use_nir = use_nir
        self.image_size = image_size

        self.H: Optional[np.ndarray] = None
        if use_nir and homography_path and Path(homography_path).exists():
            self.H = np.load(homography_path)

        in_channels = 4 if use_nir else 3
        self.transform = transform or get_transforms(image_size, split, in_channels)

        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict]:
        samples = []
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split '{self.split}' no encontrado en {self.root}. "
                f"Asegurate de que exista la carpeta {split_dir}"
            )

        for label_idx, class_name in enumerate(CLASSES):
            rgb_dir = split_dir / class_name / "rgb"
            nir_dir = split_dir / class_name / "nir"

            if not rgb_dir.exists():
                continue

            for img_path in sorted(rgb_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tiff"}:
                    continue

                nir_path = nir_dir / img_path.name if self.use_nir else None

                samples.append({
                    "rgb":   str(img_path),
                    "nir":   str(nir_path) if nir_path else None,
                    "label": label_idx,
                    "class": class_name,
                })

        if not samples:
            raise RuntimeError(
                f"No se encontraron imágenes en {split_dir}. "
                f"Verificá que la estructura de carpetas sea: "
                f"{split_dir}/{{verde,pinton,maduro}}/rgb/*.jpg"
            )

        return samples

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"No se pudo cargar la imagen: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_and_warp_nir(self, nir_path: str, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Carga NIR en escala de grises y lo warpea al espacio RGB usando
        la matriz de homografía calculada con el tablero Charuco.
        """
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if nir is None:
            raise IOError(f"No se pudo cargar la imagen NIR: {nir_path}")

        h, w = target_hw
        if self.H is not None:
            nir = cv2.warpPerspective(nir, self.H, (w, h))
        else:
            nir = cv2.resize(nir, (w, h))

        return nir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        rgb = self._load_rgb(sample["rgb"])  # (H, W, 3) uint8

        if self.use_nir and sample["nir"] and Path(sample["nir"]).exists():
            nir = self._load_and_warp_nir(sample["nir"], rgb.shape[:2])  # (H, W)
            # Stack → (H, W, 4): RGBN
            image = np.concatenate([rgb, nir[..., np.newaxis]], axis=-1)
        else:
            image = rgb  # (H, W, 3)

        augmented = self.transform(image=image)
        tensor = augmented["image"]  # (C, H, W) float32, normalizado

        return tensor, sample["label"]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(
    image_size: int = 224,
    split: str = "train",
    in_channels: int = 4,
) -> A.Compose:
    """
    Pipeline de augmentación con albumentations.
    Funciona correctamente con imágenes de N canales (RGB o RGBN).

    Para el canal NIR se aplican las mismas augmentaciones espaciales que al RGB,
    pero NO ColorJitter (que solo tiene sentido en 3 canales RGB).
    """
    mean = _MEAN_RGB + (_MEAN_NIR if in_channels == 4 else [])
    std  = _STD_RGB  + (_STD_NIR  if in_channels == 4 else [])

    spatial_augs = [
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
    ]

    photometric_augs = [
        # Solo para los primeros 3 canales (RGB). albumentations aplica ColorJitter
        # solo si el input tiene 3 canales. Con RGBN hay que limitarlo manualmente.
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
    ]

    normalize = [
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    if split == "train":
        return A.Compose(spatial_augs + photometric_augs + normalize)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])


# ---------------------------------------------------------------------------
# Factory de DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    homography_path: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_nir: bool = True,
    in_channels: int = 4,
) -> dict[str, DataLoader]:
    """
    Construye DataLoaders para train/val/test.
    Solo incluye los splits que existen en disco.
    """
    loaders = {}
    in_channels = 4 if use_nir else 3

    for split in ("train", "val", "test"):
        split_path = Path(data_root) / split
        if not split_path.exists():
            continue

        ds = MangoRipeness(
            root=data_root,
            split=split,
            use_nir=use_nir,
            homography_path=homography_path,
            transform=get_transforms(image_size, split, in_channels),
            image_size=image_size,
        )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )

        print(f"  [{split}] {len(ds)} muestras | batch_size={batch_size}")

    return loaders


# ---------------------------------------------------------------------------
# Utilidad: calcular stats NIR reales de tu dataset
# ---------------------------------------------------------------------------

def compute_nir_stats(data_root: str, split: str = "train") -> dict:
    """
    Calcula mean/std del canal NIR en tu dataset.
    Usá estos valores para reemplazar _MEAN_NIR y _STD_NIR antes de entrenar.

    Uso:
        stats = compute_nir_stats("data/", split="train")
        print(stats)  # → {"mean": 0.42, "std": 0.19}
    """
    vals = []
    root = Path(data_root) / split

    for class_name in CLASSES:
        nir_dir = root / class_name / "nir"
        if not nir_dir.exists():
            continue
        for img_path in nir_dir.iterdir():
            nir = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if nir is not None:
                vals.append(nir.astype(np.float32) / 255.0)

    if not vals:
        return {"mean": 0.45, "std": 0.22}  # fallback

    all_pixels = np.concatenate([v.ravel() for v in vals])
    return {"mean": float(all_pixels.mean()), "std": float(all_pixels.std())}
