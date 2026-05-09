"""
Pipeline de fine-tuning del ConvNeXtV2 Teacher para madurez de mangos.

Estrategia de fine-tuning en 2 fases (recomendada para datasets pequeños):
    Fase 1 — Backbone congelado, solo se entrena la cabeza.
              Epochs: ~5-10. LR alto. Evita destruir pesos preentrenados.
    Fase 2 — Backbone completo descongelado.
              Epochs: restantes. LR bajo (10x menor que fase 1).

El proceso() puede ser llamado directamente o desde el agente LangGraph (Node 3).
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

# Paths relativos al src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_convnext.model import ConvNextTeacher
from model_convnext.config import ConvNextConfig
from ml_pipeline.training.dataset import build_dataloaders, CLASSES
from utils.logger import logger_singleton as logger


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(
    model: ConvNextTeacher,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  train", leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss":     total_loss / len(loader),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
    }


@torch.no_grad()
def _eval_epoch(
    model: ConvNextTeacher,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"  {split_name}", leave=False, ncols=90):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1_per_class = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(CLASSES)))
    )

    metrics = {
        "loss":     total_loss / len(loader),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
    }
    for i, name in enumerate(CLASSES):
        metrics[f"f1_{name}"] = float(f1_per_class[i])

    return metrics


def _build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Cosine decay con warmup lineal. Estándar para ConvNeXt."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Entry point principal
# ---------------------------------------------------------------------------

def process(
    data_root: str = "data",
    homography_path: str = "matriz_homografia_charuco.npy",
    checkpoint_dir: str = "checkpoints",
    # Modelo
    variant: str = "convnextv2_small",
    use_nir: bool = True,
    # Entrenamiento
    num_epochs: int = 30,
    warmup_epochs_phase1: int = 0,   # Fase 1: solo cabeza
    phase1_epochs: int = 5,          # Epochs con backbone congelado
    batch_size: int = 32,
    image_size: int = 224,
    lr_phase1: float = 1e-3,         # LR para la cabeza (fase 1)
    lr_phase2: float = 3e-4,         # LR full fine-tuning (fase 2)
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    num_workers: int = 4,
) -> dict:
    """
    Fine-tuning del ConvNeXtV2 Teacher en 2 fases.

    Args:
        data_root:         Carpeta raíz del dataset (con train/, val/, test/)
        homography_path:   Path a la matriz homografía (.npy). Solo si use_nir=True.
        checkpoint_dir:    Dónde guardar checkpoints.
        variant:           Variante timm del modelo.
        use_nir:           Si True, input de 4 canales (RGBN).
        num_epochs:        Epochs totales (phase1 + phase2).
        phase1_epochs:     Cuántos epochs solo entrenar la cabeza (backbone frozen).
        ...

    Returns:
        dict con best_val_f1 e history completa.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Run: {run_id}")

    in_channels = 4 if use_nir else 3

    # ── Modelo ──────────────────────────────────────────────────────────────
    config = ConvNextConfig(
        variant=variant,
        num_classes=len(CLASSES),
        class_names=CLASSES,
        in_channels=in_channels,
        pretrained=True,
        dropout=0.2,
    )
    model = ConvNextTeacher(config).to(device)

    params = model.count_params()
    logger.info(
        f"Modelo: {variant} | "
        f"in_channels={in_channels} | "
        f"params totales={params['total']:,} | entrenables={params['trainable']:,}"
    )

    # ── Datos ────────────────────────────────────────────────────────────────
    logger.info(f"Cargando datos desde: {data_root}")
    loaders = build_dataloaders(
        data_root=data_root,
        homography_path=homography_path if use_nir else None,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_nir=use_nir,
        in_channels=in_channels,
    )

    train_loader = loaders["train"]
    val_loader   = loaders.get("val", loaders["train"])  # fallback a train si no hay val

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ── Checkpoints dir ──────────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_f1 = 0.0
    best_ckpt_path = ckpt_dir / "convnext_best.pt"

    # ════════════════════════════════════════════════════════════════════════
    # FASE 1: backbone congelado, solo se entrena la cabeza
    # ════════════════════════════════════════════════════════════════════════
    if phase1_epochs > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"FASE 1: {phase1_epochs} epochs | backbone CONGELADO | LR={lr_phase1}")
        logger.info(f"{'='*60}")

        model.freeze_backbone(unfreeze_stages=0)
        trainable_after_freeze = model.count_params()["trainable"]
        logger.info(f"Params entrenables en fase 1: {trainable_after_freeze:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_phase1,
            weight_decay=weight_decay,
        )
        total_steps = phase1_epochs * len(train_loader)
        warmup_steps = max(1, total_steps // 10)
        scheduler = _build_scheduler(optimizer, total_steps, warmup_steps)

        for epoch in range(1, phase1_epochs + 1):
            train_m = _train_epoch(model, train_loader, optimizer, criterion, device, scheduler, grad_clip)
            val_m   = _eval_epoch(model, val_loader, criterion, device)

            record = {"phase": 1, "epoch": epoch, "train": train_m, "val": val_m}
            history.append(record)

            logger.info(
                f"[P1] Epoch {epoch:02d}/{phase1_epochs} | "
                f"train_loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} | "
                f"val_f1={val_m['f1_macro']:.3f} acc={val_m['accuracy']:.3f}"
            )

            if val_m["f1_macro"] > best_val_f1:
                best_val_f1 = val_m["f1_macro"]
                _save_checkpoint(model, config, epoch, val_m, best_ckpt_path)

    # ════════════════════════════════════════════════════════════════════════
    # FASE 2: fine-tuning completo
    # ════════════════════════════════════════════════════════════════════════
    phase2_epochs = num_epochs - phase1_epochs
    if phase2_epochs > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"FASE 2: {phase2_epochs} epochs | backbone COMPLETO | LR={lr_phase2}")
        logger.info(f"{'='*60}")

        model.unfreeze_all()

        # Layer-wise LR decay: backbone con LR más bajo que la cabeza
        backbone_params = [p for p in model.backbone.parameters()]
        head_params     = [p for p in model.head.parameters()]

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": lr_phase2 * 0.1},
                {"params": head_params,     "lr": lr_phase2},
            ],
            weight_decay=weight_decay,
        )

        total_steps  = phase2_epochs * len(train_loader)
        warmup_steps = max(1, total_steps // 10)
        scheduler = _build_scheduler(optimizer, total_steps, warmup_steps)

        for epoch in range(1, phase2_epochs + 1):
            train_m = _train_epoch(model, train_loader, optimizer, criterion, device, scheduler, grad_clip)
            val_m   = _eval_epoch(model, val_loader, criterion, device)

            record = {"phase": 2, "epoch": epoch + phase1_epochs, "train": train_m, "val": val_m}
            history.append(record)

            logger.info(
                f"[P2] Epoch {epoch:02d}/{phase2_epochs} | "
                f"train_loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} | "
                f"val_f1={val_m['f1_macro']:.3f} acc={val_m['accuracy']:.3f}"
                + (" ✅ best" if val_m["f1_macro"] > best_val_f1 else "")
            )

            if val_m["f1_macro"] > best_val_f1:
                best_val_f1 = val_m["f1_macro"]
                _save_checkpoint(model, config, epoch + phase1_epochs, val_m, best_ckpt_path)

    # ── Evaluación final en test ─────────────────────────────────────────────
    if "test" in loaders:
        logger.info("\nEvaluando en test set con mejor checkpoint...")
        best_model = _load_model_from_checkpoint(best_ckpt_path, device)
        test_m = _eval_epoch(best_model, loaders["test"], criterion, device, "test")
        logger.info(f"TEST → accuracy={test_m['accuracy']:.4f} | f1_macro={test_m['f1_macro']:.4f}")
        for cls in CLASSES:
            logger.info(f"  f1_{cls}: {test_m.get(f'f1_{cls}', 0):.4f}")

    # ── Guardar historial ────────────────────────────────────────────────────
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nHistorial guardado en: {history_path}")
    logger.info(f"Mejor val F1: {best_val_f1:.4f} → checkpoint: {best_ckpt_path}")

    return {"best_val_f1": best_val_f1, "history": history, "checkpoint": str(best_ckpt_path)}


# ---------------------------------------------------------------------------
# Helpers de checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: ConvNextTeacher,
    config: ConvNextConfig,
    epoch: int,
    val_metrics: dict,
    path: Path,
):
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "config":      config.__dict__,
            "val_metrics": val_metrics,
            "classes":     CLASSES,
        },
        path,
    )
    logger.info(f"  💾 Checkpoint guardado → {path} (f1={val_metrics['f1_macro']:.4f})")


def _load_model_from_checkpoint(path: Path, device: torch.device) -> ConvNextTeacher:
    ckpt = torch.load(path, map_location=device)
    config = ConvNextConfig(**ckpt["config"])
    model = ConvNextTeacher(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
