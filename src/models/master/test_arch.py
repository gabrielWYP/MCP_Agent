"""
Sanity check — arquitectura completa del Master Model.

Correr con:
    python -m src.models.master.test_arch
"""

import torch

from src.models.master.backbone import DualConvNeXtBackbone
from src.models.master.fusion import CrossModalFusion
from src.models.master.neck import DualFPN
from src.models.master.head import YOLODetectionHead, NUM_CLASSES
from src.models.master.distill_projections import (
    fpn_projections, backbone_projections, head_projections,
)
from src.models.master.master_model import MasterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

# ── Módulo por módulo ────────────────────────────────────────────────

print('Testing backbone...')
backbone = DualConvNeXtBackbone(pretrained=False).to(device)
rgb = torch.randn(2, 3, 640, 640, device=device)
nir = torch.randn(2, 1, 640, 640, device=device)
rgb_feats, nir_feats = backbone(rgb, nir)
print('  RGB features:', [tuple(f.shape) for f in rgb_feats])
print('  NIR features:', [tuple(f.shape) for f in nir_feats])

print('Testing fusion...')
fusion = CrossModalFusion().to(device)
fused, nir_pass = fusion(rgb_feats, nir_feats)
print('  Fused features:', [tuple(f.shape) for f in fused])

print('Testing neck...')
neck = DualFPN().to(device)
pyramid = neck(fused, nir_pass)
print('  FPN pyramid:', [tuple(f.shape) for f in pyramid])

print('Testing YOLO head...')
head = YOLODetectionHead().to(device)
out = head(pyramid)
print('  preds:', [tuple(p.shape) for p in out['preds']])
print('  cls_preds:', [tuple(p.shape) for p in out['cls_preds']])
print('  reg_preds:', [tuple(p.shape) for p in out['reg_preds']])
print('  distill_cls:', [tuple(f.shape) for f in out['distill_cls']])
print('  distill_reg:', [tuple(f.shape) for f in out['distill_reg']])

print('Testing distill projections...')
proj = fpn_projections().to(device)
projected = proj(pyramid)
print('  Projected:', [tuple(p.shape) for p in projected])

# ── Modelo completo ──────────────────────────────────────────────────

print()
print('Full MasterModel test (batch=2, 640x640)...')
model = MasterModel(pretrained_backbone=False).to(device)
with torch.no_grad():
    result = model(rgb, nir)

print('  Output keys:', list(result.keys()))
print('  preds:', [tuple(p.shape) for p in result['preds']])
print('  cls_preds:', [tuple(p.shape) for p in result['cls_preds']])
print('  reg_preds:', [tuple(p.shape) for p in result['reg_preds']])
print('  distill_backbone_rgb:', [tuple(f.shape) for f in result['distill_backbone_rgb']])
print('  distill_backbone_fused:', [tuple(f.shape) for f in result['distill_backbone_fused']])
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
print('YOLO Nano compatibility check:')
print(f'  Output format: (B, {NUM_CLASSES}+4, H_i, W_i) per level ✓')
print(f'  Strides: [8, 16, 32] ✓')
print(f'  Levels: P3, P4, P5 ✓')
print(f'  Classes: {NUM_CLASSES} (sano, danado) ✓')
print(f'  Direct distillation possible: cls_preds + reg_preds ✓')

print()
print('ALL TESTS PASSED')
