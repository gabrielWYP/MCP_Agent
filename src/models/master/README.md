# Master Model — Multimodal Mango Damage Detector

Arquitectura del modelo maestro para la tesis de **destilamiento de conocimiento cruzado multimodal** aplicado a la detección de daño mecánico temprano en mango (*Mangifera indica*).

> **fusion-redesign**: la arquitectura de dos streams con cross-attention (`DualConvNeXtBackbone` + `CrossModalFusion` + `DualFPN`) fue **reemplazada** por early fusion simple — RGB y NIR se apilan en un tensor de 4 canales antes de un único backbone. Ver `openspec/changes/fusion-redesign/` para la justificación completa (probe RGB+NIR: AUC 0.9171 con solo 3 bloques conv, vs AP50 0.0643 del diseño de dos streams con 9.4M parámetros dedicados a fusión).

---

## Contexto del problema

El daño mecánico temprano en mango se manifiesta como **pardismo interno** antes de ser visible en la superficie. Esto hace que la detección en imágenes RGB convencionales sea extremadamente difícil. Las imágenes **NIR (Near-Infrared)** capturan cambios en la absorción de agua y azúcares en el tejido dañado, revelando la lesión cuando el RGB aún no muestra nada.

El modelo maestro aprovecha **ambas modalidades** durante el entrenamiento. El modelo estudiante (YOLO Nano) aprende a imitar al maestro usando **solo RGB** en inferencia — ese es el núcleo del destilamiento cruzado multimodal.

---

## Arquitectura general

```
RGB (N, 3, H, W) ──┐
                    ├──► cat(dim=1) ──► EarlyFusionBackbone ──► [S1..S4]
NIR (N, 1, H, W) ──┘        (4,96,H,W)      (96,192,384,768)
                                                    │
                                                    ▼
                                    FPNNeck ( SingleFPN, sin cambios )
                                                    │
                                                    ▼
                              [P2, P3, P4, P5]  (head_strides, default)
                                                    │
                                                    ▼
                                        YOLODetectionHead
                              (anchor-free, decoupled, N niveles)
                                                    │
                                                    ▼
      {preds, cls_preds, reg_preds, distill_backbone, distill_fpn,
       distill_head_cls, distill_head_reg}   — 7 keys
```

Un único stem, un único forward pass por los 4 stages de ConvNeXt — no hay un segundo stream que fusionar. `MasterModel.forward(rgb, nir)` conserva la firma de dos tensores (para no tocar `dataset.py`, `Trainer._train_epoch` ni `KDTrainer`); la concatenación en un tensor de 4 canales ocurre dentro del modelo.

---

## Head YOLO-style (compatible con YOLO Nano)

El head es **anchor-free decoupled**, idéntico en formato al de YOLO Nano, permitiendo distilación directa:

```
P_i (256ch) ──► conv 3×3 + SiLU ──► conv 1×1 (nc)  → cls_preds
              └─► conv 3×3 + SiLU ──► conv 1×1 (4)  → reg_preds
```

`cls_stem`/`reg_stem` se computan **una sola vez** por nivel; la misma salida se reutiliza para la predicción y para la feature de distilación (antes se recomputaban, duplicando cómputo y memoria de activaciones — ver fusion-redesign W5).

**Output por nivel:** `(B, nc + 4, H_i, W_i)` donde:
- `nc = 2` (mango, danado)
- `4` = bbox deltas (xywh)

**Compatibilidad con YOLO Nano** (el estudiante está fuera de alcance de este rediseño y sigue fijo en 3 niveles):

| Propiedad | Maestro | YOLO Nano | Distilación |
|---|---|---|---|
| Formato output | `(B, 6, H_i, W_i)` | `(B, 6, H_i, W_i)` | ✅ Directa |
| Strides | `head_strides`, default `[4, 8, 16, 32]` | fijo `[8, 16, 32]` | KD selecciona por stride, no por posición — ver `src/training/kd_trainer.py` |
| Niveles | configurable (`P2, P3, P4, P5` por default) | `P3, P4, P5` | — |
| Clases | 2 (mango, danado) | 2 (mango, danado) | ✅ Idénticas |

---

## Parámetros (fusion-redesign — exactos, hand-counted)

| Módulo | Antes (dos streams) | Después (early fusion) | Δ |
|---|---:|---:|---:|
| Backbone stages | 27,813,696 | 27,813,696 | 0 |
| Stems | 6,720 (rgb+nir) | 6,432 (4-ch) | −288 |
| Fusión (`StageAttentionFusion`/`CrossModalFusion`) | 9,421,920 | 0 (**eliminado**) | −9,421,920 |
| Neck (`DualFPN`) | 5,855,488 | 2,729,984 (`SingleFPN` × 1) | −3,125,504 |
| Head | 3,545,106 (3 niveles) | 4,726,808 (4 niveles, incluye P2) | +1,181,702 |
| **Total** | **≈46.64M** | **≈35.28M** | **≈ −11.37M** |

El presupuesto liberado (fusión + un FPN completo) se reinvierte parcialmente en el 4º `DecoupledHead` (P2, stride 4) — ver `openspec/changes/fusion-redesign/design.md` D-3/D-6 para la justificación de por qué el resto **no** se reinvierte.

Hardware de entrenamiento local: NVIDIA GTX 1660 SUPER (6.44 GB VRAM), `batch_size=2` con `effective_batch=8` vía gradient accumulation.

---

## Módulos

### 1. `EarlyFusionBackbone` — `backbone.py`

Un único encoder ConvNeXt (Tiny o Small) que procesa el tensor RGB+NIR apilado.

```
RGB+NIR (4ch) ──► stem (Conv 4×4, s4) ──► stages [S1, S2, S3, S4]
```

**Inicialización del stem — inflación mean-preserving (no zero-init, no copia naive):**

```python
W_new[:, :3] = W_imagenet * 3/4       # RGB, atenuado
W_new[:, 3]  = mean(W_imagenet, 1) * 3/4   # NIR, prior de bordes/textura de ImageNet
```

Un 4º canal sin reescalar elevaría la pre-activación del stem en ~4/3 y perturbaría las estadísticas que la LayerNorm pretrained y el stage 1 esperan; el factor 3/4 mantiene el input mean-preserving. `in_channels=3` (copia verbatim, sin reescalar) está también soportado como el brazo de control RGB-only de la hipótesis H-D.

**Salidas por stage:**

| Stage | Canales | Resolución (entrada 640×640) |
|-------|---------|------------------------------|
| S1    | 96      | 160 × 160                    |
| S2    | 192     | 80 × 80                      |
| S3    | 384     | 40 × 40                      |
| S4    | 768     | 20 × 20                      |

### 2. `FPNNeck` (sobre `SingleFPN`, sin cambios) — `neck.py`

`SingleFPN` ya construía la pirámide completa `[P2, P3, P4, P5]`; el diseño anterior (`DualFPN`) descartaba P2 y ejecutaba dos FPNs completos (uno por modalidad) solo para fusionarlos con un 1×1 conv por nivel. `FPNNeck` es un wrapper delgado: ejecuta `SingleFPN` una vez y **selecciona** los niveles nombrados por `head_strides`, en orden finest-first.

```
[S1..S4] ──► SingleFPN ──► [P2, P3, P4, P5] (256ch c/u) ──► select(head_strides) ──► pirámide emitida
```

**P2 se reconecta por default.** El daño mediano mide 30px de lado — a stride 8 ocupa ~3.8 celdas, a stride 4 ~7.6. Con `assigner_level_ranges=[32, 64, 128]`, P2 recibe la mediana de las cajas de daño. `head_strides=[8, 16, 32]` reproduce el comportamiento de 3 niveles sin cambios de código (config-only ablation, hipótesis H-E).

| Nivel | Stride | Resolución (640×640) | Canales |
|-------|--------|----------------------|---------|
| P2    | 4      | 160 × 160            | 256     |
| P3    | 8      | 80 × 80              | 256     |
| P4    | 16     | 40 × 40              | 256     |
| P5    | 32     | 20 × 20              | 256     |

### 3. `YOLODetectionHead` — `head.py`

Head de detección **anchor-free decoupled** estilo YOLOv8, generalizado sobre `len(strides)` niveles (uno `DecoupledHead` por nivel de `head_strides`).

```
P_i (256ch)
  ├──► cls_stem: Conv 3×3 + SiLU ──► cls_pred: Conv 1×1 (nc)   ─┐
  └──► reg_stem: Conv 3×3 + SiLU ──► reg_pred: Conv 1×1 (4)     ┴─► reutilizado como distill_cls/distill_reg
```

**Clases de detección:**

| ID | Clase     |
|----|-----------|
| 0  | mango     |
| 1  | danado    |

### 4. `ProjectionLayers` — `distill_projections.py`

Proyecciones 1×1 aprendibles para alinear dimensiones maestro → estudiante en feature-level distillation. `forward()` valida explícitamente `len(teacher_features) == self.num_levels` — un mismatch (p. ej. el maestro emitiendo 4 niveles contra un preset de 3) levanta un error identificando ambos conteos en vez de truncar el `zip` en silencio.

Presets incluidos (sin cambios de canales — fusion-redesign W7):
- `fpn_projections()` — 3 niveles, para distilación a nivel de FPN
- `backbone_projections()` — `[384, 768]` (S3, S4), para distilación a nivel de backbone
- `head_projections()` — 3 niveles, para distilación a nivel de head

`KDTrainer` selecciona los 3 niveles del maestro que coinciden **por stride** (no por posición) con los del estudiante antes de pasarlos a estos presets — ver `src/training/strides.select_by_strides`.

---

## Outputs del forward pass

```python
output = model(rgb, nir)  # Sin proposals — head YOLO es dense; 7 keys
```

| Key                  | Shape                              | Descripción                                  |
|----------------------|-------------------------------------|-----------------------------------------------|
| `preds`              | `[(B, 6, H_i, W_i), ...]`           | Predicciones concatenadas (cls + reg) por nivel |
| `cls_preds`          | `[(B, 2, H_i, W_i), ...]`           | Scores de clasificación por nivel              |
| `reg_preds`          | `[(B, 4, H_i, W_i), ...]`           | Deltas de bbox (xywh) por nivel                |
| `distill_backbone`   | `[S1..S4]`                          | Features del backbone (renombrado desde `distill_backbone_rgb`; ahora genuinamente multimodal) |
| `distill_fpn`        | pirámide emitida (`head_strides`)   | Niveles FPN, finest-first                      |
| `distill_head_cls`   | `[cls_stem por nivel]`              | Features intermedios de clasificación          |
| `distill_head_reg`   | `[reg_stem por nivel]`              | Features intermedios de regresión              |

(`distill_backbone_fused` ya no existe — no hay fusión que producir.)

---

## Uso

```python
from src.models.master import MasterModel
import torch

model = MasterModel(
    num_classes=2,
    pretrained_backbone=True,   # ImageNet-1K weights, inflated to 4ch
    fpn_channels=256,
    head_strides=[4, 8, 16, 32],  # default; pass [8, 16, 32] to ablate P2
).cuda()

# End-to-end desde epoch 0 es el schedule default para este modelo
# (fusion-redesign D-4) — freeze_backbone(0) dejaría todo entrenable.
# freeze_backbone(n) sigue disponible para ablations puntuales:
model.freeze_backbone(freeze_stages=2)

rgb = torch.randn(4, 3, 640, 640).cuda()
nir = torch.randn(4, 1, 640, 640).cuda()

output = model(rgb, nir)  # dict de 7 keys
```

**Checkpoints v1 (dual-stream fusion) no cargan en este modelo** — cada key del `state_dict` cambió de nombre. Los checkpoints nuevos incluyen `arch_version: 2`; `scripts/evaluate_checkpoint.py` y `scripts/visualize_damage_predictions.py` verifican ese tag antes de intentar `load_state_dict` y fallan con un mensaje explícito en vez de un dump de 200 líneas de keys faltantes/inesperadas.

---

## Estructura de archivos

```
src/models/master/
├── __init__.py              # Exports públicos
├── backbone.py              # EarlyFusionBackbone (single-stream, 4-channel)
├── neck.py                  # SingleFPN (sin cambios) + FPNNeck (wrapper configurable)
├── head.py                  # YOLODetectionHead (anchor-free, decoupled, stem-once)
├── distill_projections.py   # Projection layers para KD
├── master_model.py          # MasterModel (integración completa, 7-key output)
├── test_arch.py             # Sanity check (script manual, no pytest)
├── sanity_check.py          # Sanity check con asserts de shape
└── README.md                # Este archivo
```

`fusion.py` (`StageAttentionFusion`, `CrossModalFusion`) fue **eliminado** en su totalidad — no hay módulo de fusión cruzada en esta arquitectura.

---

## Referencias

- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **TOOD**: Task-aligned One-stage Object Detection, ICCV 2021
- **FPN**: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017
- **Localization Distillation**: Zheng et al., *Localization Distillation for Dense Object Detection*, CVPR 2022
