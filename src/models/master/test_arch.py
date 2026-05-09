import torch

from src.models.master.backbone import DualConvNeXtBackbone
from src.models.master.fusion import CrossModalFusion
from src.models.master.neck import DualFPN
from src.models.master.head import CascadeRCNNHead
from src.models.master.master_model import MasterModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

print('Testing backbone...')
backbone = DualConvNeXtBackbone(pretrained=False).to(device)
# 640x640 en GPU — resolución real de entrenamiento
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

print('Testing head...')
head = CascadeRCNNHead().to(device)
proposals = [
    torch.tensor([[10.,10.,200.,200.],[50.,50.,300.,300.]], dtype=torch.float32, device=device)
    for _ in range(2)
]
out = head(pyramid, proposals)
print('  cls_scores:', [tuple(s.shape) for s in out['cls_scores']])
print('  bbox_deltas:', [tuple(d.shape) for d in out['bbox_deltas']])
print('  distill_feats:', [tuple(f.shape) for f in out['distill_feats']])

print()
print('Full MasterModel test (batch=2, 640x640, GPU)...')
model = MasterModel(pretrained_backbone=False).to(device)
with torch.no_grad():
    result = model(rgb, nir, proposals)
params = model.count_parameters()
print('  Parameter counts:')
for k, v in params.items():
    print(f'    {k:<15}: {v:>12,}')
print()
print('ALL TESTS PASSED')
