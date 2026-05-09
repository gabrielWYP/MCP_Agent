"""
Inferencia con el ConvNeXtV2 Teacher entrenado.

Soporta imagen única o batch de imágenes.
Aplica la misma pre-procesamiento que el training (incluyendo homografía NIR).
"""
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_convnext.model import ConvNextTeacher
from model_convnext.config import ConvNextConfig
from utils.logger import logger_singleton as logger

CLASSES = ["sano", "danado"]


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_teacher(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> tuple[ConvNextTeacher, ConvNextConfig]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ConvNextConfig(**ckpt["config"])

    model = ConvNextTeacher(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        f"Modelo cargado: {config.variant} | "
        f"epoch={ckpt['epoch']} | "
        f"val_f1={ckpt['val_metrics']['f1_macro']:.4f}"
    )
    return model, config


# ---------------------------------------------------------------------------
# Pre-procesamiento de imagen única
# ---------------------------------------------------------------------------

def _build_inference_transform(image_size: int, in_channels: int) -> A.Compose:
    from ml_pipeline.training.dataset import _MEAN_RGB, _STD_RGB, _MEAN_NIR, _STD_NIR
    mean = _MEAN_RGB + (_MEAN_NIR if in_channels == 4 else [])
    std  = _STD_RGB  + (_STD_NIR  if in_channels == 4 else [])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def prepare_image(
    rgb_path: str,
    nir_path: Optional[str] = None,
    homography_path: Optional[str] = None,
    image_size: int = 224,
) -> tuple[torch.Tensor, int]:
    """
    Carga y prepara una imagen para inferencia.

    Returns:
        tensor: (1, C, H, W) listo para el modelo
        in_channels: 3 o 4
    """
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise IOError(f"No se pudo leer: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if nir_path and Path(nir_path).exists():
        nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

        if homography_path and Path(homography_path).exists():
            H = np.load(homography_path)
            h, w = rgb.shape[:2]
            nir = cv2.warpPerspective(nir, H, (w, h))
        else:
            nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]))

        image = np.concatenate([rgb, nir[..., np.newaxis]], axis=-1)  # (H, W, 4)
        in_channels = 4
    else:
        image = rgb
        in_channels = 3

    transform = _build_inference_transform(image_size, in_channels)
    tensor = transform(image=image)["image"].unsqueeze(0)  # (1, C, H, W)
    return tensor, in_channels


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    rgb_path: str,
    nir_path: Optional[str] = None,
    checkpoint_path: str = "checkpoints/convnext_best.pt",
    homography_path: str = "matriz_homografia_charuco.npy",
    image_size: int = 224,
) -> dict:
    """
    Clasifica la madurez de un mango a partir de una imagen.

    Args:
        rgb_path:         Imagen RGB del mango.
        nir_path:         Imagen NIR (opcional). Si no se provee, usa solo RGB.
        checkpoint_path:  Checkpoint del modelo entrenado.
        homography_path:  Matriz de homografía .npy para alinear NIR→RGB.
        image_size:       Tamaño de entrada del modelo (default: 224).

    Returns:
        {
          "class":         "verde" | "pinton" | "maduro",
          "confidence":    0.0 - 1.0,
          "probabilities": {"verde": 0.x, "pinton": 0.x, "maduro": 0.x}
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_teacher(checkpoint_path, device)

    tensor, in_channels = prepare_image(rgb_path, nir_path, homography_path, image_size)

    if in_channels != config.in_channels:
        logger.warning(
            f"El modelo fue entrenado con in_channels={config.in_channels} "
            f"pero la imagen tiene {in_channels} canales. Verificá las entradas."
        )

    logits = model(tensor.to(device))
    probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())

    result = {
        "class":         CLASSES[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
    }

    logger.info(
        f"Predicción: {result['class']} "
        f"({result['confidence']:.1%}) | "
        f"probs={result['probabilities']}"
    )
    return result


# ---------------------------------------------------------------------------
# Entry point para el agente LangGraph
# ---------------------------------------------------------------------------

def process():
    """
    Llamado por el agente MLOps (Node de predicción).
    En el pipeline completo recibe el path de la imagen desde el estado.
    """
    logger.info("Prediction process placeholder — invocar predict_single() directamente.")
