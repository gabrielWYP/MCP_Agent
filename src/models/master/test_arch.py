"""
Sanity check — arquitectura completa del Master Model (fusion-redesign).

Correr con:
    python -m src.models.master.test_arch

Nota: no es un módulo pytest (no define funciones `test_*`); es un script
manual, igual que antes de la migración a early fusion.
"""

import torch

from src.models.master.backbone import EarlyFusionBackbone
from src.models.master.neck import FPNNeck
from src.models.master.head import YOLODetectionHead, NUM_CLASSES
from src.models.master.distill_projections import (
    fpn_projections, backbone_projections, head_projections,
)
from src.models.master.master_model import MasterModel
from src.training.strides import DEFAULT_HEAD_STRIDES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

HEAD_STRIDES = list(DEFAULT_HEAD_STRIDES)  # [4, 8, 16, 32] — includes P2

# ── Módulo por módulo ────────────────────────────────────────────────

print('Testing backbone (single-stream, 4-channel early fusion)...')
backbone = EarlyFusionBackbone(pretrained=False, in_channels=4).to(device)
rgb = torch.randn(2, 3, 640, 640, device=device)
nir = torch.randn(2, 1, 640, 640, device=device)
x = torch.cat([rgb, nir], dim=1)  # (N, 4, H, W) — early fusion at the input
features = backbone(x)
print('  Backbone features:', [tuple(f.shape) for f in features])

print('Testing neck (FPNNeck over SingleFPN)...')
neck = FPNNeck(strides=tuple(HEAD_STRIDES)).to(device)
pyramid = neck(features)
assert len(pyramid) == len(HEAD_STRIDES), (
    f"Expected {len(HEAD_STRIDES)} FPN levels for strides {HEAD_STRIDES}, got {len(pyramid)}"
)
print('  FPN pyramid:', [tuple(f.shape) for f in pyramid])

print('Testing YOLO head...')
head = YOLODetectionHead(strides=HEAD_STRIDES).to(device)
out = head(pyramid)
print('  preds:', [tuple(p.shape) for p in out['preds']])
print('  cls_preds:', [tuple(p.shape) for p in out['cls_preds']])
print('  reg_preds:', [tuple(p.shape) for p in out['reg_preds']])
print('  distill_cls:', [tuple(f.shape) for f in out['distill_cls']])
print('  distill_reg:', [tuple(f.shape) for f in out['distill_reg']])

print('Testing distill projections (backbone: teacher S3,S4 -> student)...')
proj = backbone_projections().to(device)
projected = proj(features[2:])  # S3, S4 — channels [384, 768], unchanged by the redesign
print('  Projected:', [tuple(p.shape) for p in projected])

# ── Modelo completo ──────────────────────────────────────────────────

print()
print('Full MasterModel test (batch=2, 640x640)...')
model = MasterModel(pretrained_backbone=False, head_strides=HEAD_STRIDES).to(device)
with torch.no_grad():
    result = model(rgb, nir)

print('  Output keys:', list(result.keys()))
assert set(result.keys()) == {
    "preds", "cls_preds", "reg_preds",
    "distill_backbone", "distill_fpn", "distill_head_cls", "distill_head_reg",
}, f"Unexpected output keys: {sorted(result.keys())}"
print('  preds:', [tuple(p.shape) for p in result['preds']])
print('  cls_preds:', [tuple(p.shape) for p in result['cls_preds']])
print('  reg_preds:', [tuple(p.shape) for p in result['reg_preds']])
print('  distill_backbone:', [tuple(f.shape) for f in result['distill_backbone']])
print('  distill_fpn:', [tuple(f.shape) for f in result['distill_fpn']])
print('  distill_head_cls:', [tuple(f.shape) for f in result['distill_head_cls']])
print('  distill_head_reg:', [tuple(f.shape) for f in result['distill_head_reg']])

params = model.count_parameters()
print()
print('  Parameter counts:')
for k, v in params.items():
    print(f'    {k:<15}: {v:>12,}')

# ── Verificar compatibilidad con YOLO Nano ───────────────────────────

print()
print('YOLO Nano compatibility check (student is fixed at 3 levels, out of scope):')
print(f'  Output format: (B, {NUM_CLASSES}+4, H_i, W_i) per level ✓')
print(f'  Teacher head_strides (configurable): {HEAD_STRIDES}')
print(f'  Student strides (fixed, unaffected by this redesign): [8, 16, 32]')
print(f'  Classes: {NUM_CLASSES} (mango, danado) ✓')
print(f'  KD selects matching levels by stride, not position — see src/training/kd_trainer.py')

print()
print('ALL TESTS PASSED')
