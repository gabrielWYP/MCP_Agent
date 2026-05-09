# Master Model — Multimodal Mango Damage Detector

Arquitectura del modelo maestro para la tesis de **destilamiento de conocimiento cruzado multimodal** aplicado a la detección de daño mecánico temprano en mango (*Mangifera indica*).

---

## Contexto del problema

El daño mecánico temprano en mango se manifiesta como **pardismo interno** antes de ser visible en la superficie. Esto hace que la detección en imágenes RGB convencionales sea extremadamente difícil. Las imágenes **NIR (Near-Infrared)** capturan cambios en la absorción de agua y azúcares en el tejido dañado, revelando la lesión cuando el RGB aún no muestra nada.

El modelo maestro aprovecha **ambas modalidades** durante el entrenamiento. El modelo estudiante (YOLOv8 Nano) aprende a imitar al maestro usando **solo RGB** en inferencia — ese es el núcleo del destilamiento cruzado multimodal.

---


## Arquitectura general

```
RGB (N, 3, H, W) ──┐
                   ├──► DualConvNeXtBackbone ──► [S1..S4] RGB + [S1..S4] NIR
NIR (N, 1, H, W) ──┘         (pesos compartidos,
                               stems separados)
                                      │
                                      ▼
                             CrossModalFusion
                          (cross-attention por stage)
                                      │
                                      ▼
                    ┌─────────────────┴─────────────────┐
                 FPN-RGB                             FPN-NIR
                    └─────────────────┬─────────────────┘
                              Fusión por nivel
                           (concat + 1×1 conv)
                                      │
                                      ▼
                            [P2, P3, P4, P5]
                          (pirámide unificada)
                                      │
                                      ▼
                           CascadeRCNNHead
                     (3 etapas, IoU 0.5 / 0.6 / 0.7)
                                      │
                                      ▼
                    cls_scores + bbox_deltas + distill_feats
```

---

## Parámetros totales

| Módulo        | Parámetros   |
|---------------|-------------|
| Backbone      | 49,454,976  |
| Fusion        |  9,421,920  |
| Neck          |  5,987,328  |
| Head          | 41,708,565  |
| **Total**     | **106,572,789** |

Hardware de entrenamiento: NVIDIA RTX A5000 (24 GB VRAM).

---

## Módulos

### 1. `DualConvNeXtBackbone` — `backbone.py`

Dos encoders ConvNeXt-Small que procesan RGB y NIR en paralelo.

**Decisión de diseño clave: pesos compartidos con stems separados.**

```
RGB (3ch) ──► rgb_stem (Conv 4×4, s4) ──┐
                                         ├──► shared_stages [S1, S2, S3, S4]
NIR (1ch) ──► nir_stem (Conv 4×4, s4) ──┘
```

Los 4 stages de ConvNeXt-Small son **exactamente los mismos pesos** para ambas modalidades. Solo los stems difieren porque RGB tiene 3 canales y NIR tiene 1. Esto actúa como regularización fuerte con un dataset pequeño (~2000 imágenes), forzando al modelo a aprender representaciones modality-agnostic en las capas profundas.

**Salidas por stage:**

| Stage | Canales | Resolución (entrada 640×640) |
|-------|---------|------------------------------|
| S1    | 96      | 160 × 160                    |
| S2    | 192     | 80 × 80                      |
| S3    | 384     | 40 × 40                      |
| S4    | 768     | 20 × 20                      |

**Inicialización:**
- `rgb_stem`: pesos pretrained de ImageNet-1K (ConvNeXt-Small oficial)
- `nir_stem`: Kaiming normal (no hay pesos pretrained para 1 canal NIR)
- `shared_stages`: pesos pretrained de ImageNet-1K

---

### 2. `CrossModalFusion` — `fusion.py`

Aplica **cross-attention** entre las features RGB y NIR en cada uno de los 4 stages del backbone.

**¿Por qué RGB como Query?**

En inferencia, el estudiante solo ve RGB. Al hacer que RGB sea el Query durante el entrenamiento del maestro, forzamos al stream RGB a aprender a *buscar activamente* la información que el NIR revelaría. Esto enriquece las representaciones RGB con conocimiento NIR latente — exactamente lo que el estudiante necesita aprender a imitar.

```
Q = LayerNorm(RGB_features)  ──┐
K = LayerNorm(NIR_features)  ──┼──► MultiheadAttention ──► attn_out
V = LayerNorm(NIR_features)  ──┘

fused = RGB_features + attn_out          # residual
fused = fused + FFN(LayerNorm(fused))    # feed-forward con residual
```

**Pooling espacial adaptativo:**

La atención sobre secuencias largas es O(n²) en memoria. S1 tiene 160×160 = 25,600 tokens → ~20 GB solo para la matriz de atención. La solución es hacer `adaptive_avg_pool2d` a un grid de `max_tokens_side × max_tokens_side` (default: 20×20 = 400 tokens) antes de la atención, y luego `bilinear upsample` de vuelta a la resolución original.

```
S1: 160×160 ──► pool 20×20 ──► attn (400 tokens) ──► upsample 160×160 ──► + RGB residual
S2:  80×80  ──► pool 20×20 ──► attn (400 tokens) ──► upsample  80×80  ──► + RGB residual
S3:  40×40  ──► pool 20×20 ──► attn (400 tokens) ──► upsample  40×40  ──► + RGB residual
S4:  20×20  ──► sin pool   ──► attn (400 tokens) ──► (ya es 20×20)
```

Este enfoque es consistente con CMX (TPAMI 2023) y TokenFusion (CVPR 2022).

**Cabezas de atención por stage:**

| Stage | Canales | Heads |
|-------|---------|-------|
| S1    | 96      | 4     |
| S2    | 192     | 8     |
| S3    | 384     | 8     |
| S4    | 768     | 8     |

---

### 3. `DualFPN` — `neck.py`

Dos FPNs paralelos (uno por modalidad) que construyen pirámides multi-escala independientes antes de fusionarse.

```
fused_features [F1..F4] ──► FPN-RGB ──► [P2..P5] a 256ch ──┐
                                                              ├──► concat + 1×1 conv ──► [P2..P5] unificado
nir_features   [S1..S4] ──► FPN-NIR ──► [P2..P5] a 256ch ──┘
```

**¿Por qué dos FPNs separados?**

Cada modalidad construye su propia representación multi-escala antes de fusionarse. Esto permite que el modelo aprenda patrones multi-escala específicos de cada modalidad (gradientes de textura NIR vs. gradientes de color RGB) antes de combinarlos. La fusión con `1×1 conv` es ligera pero completamente aprendible.

**Top-down pathway (FPN estándar):**

```
P5 ← lateral(S4)
P4 ← lateral(S3) + upsample(P5)
P3 ← lateral(S2) + upsample(P4)
P2 ← lateral(S1) + upsample(P3)
```

Seguido de una `3×3 conv` de suavizado en cada nivel.

**Salidas:**

| Nivel | Stride | Resolución (640×640) | Canales |
|-------|--------|----------------------|---------|
| P2    | 4      | 160 × 160            | 256     |
| P3    | 8      | 80 × 80              | 256     |
| P4    | 16     | 40 × 40              | 256     |
| P5    | 32     | 20 × 20              | 256     |

---

### 4. `CascadeRCNNHead` — `head.py`

Cabeza de detección Cascade R-CNN con 3 etapas de refinamiento progresivo.

**¿Por qué Cascade R-CNN para daño mecánico?**

El daño mecánico temprano produce regiones pequeñas y de bajo contraste. Cascade R-CNN refina iterativamente los bounding boxes con umbrales de IoU crecientes, lo que mejora significativamente la precisión en objetos difíciles comparado con Faster R-CNN estándar.

```
Propuestas ──► RoIAlign ──► Stage 1 (IoU 0.5) ──► cls + reg
                                    │
                                    ▼
                             Stage 2 (IoU 0.6) ──► cls + reg
                                    │
                                    ▼
                             Stage 3 (IoU 0.7) ──► cls + reg
```

**Clases de detección:**

| ID | Clase           |
|----|-----------------|
| 0  | background      |
| 1  | mango (sano)    |
| 2  | mango_damaged   |

**Por stage:**
- `RoIAlign` 7×7 sobre la pirámide FPN (asignación de nivel por área del bbox)
- 2 capas FC (1024 unidades) con ReLU y Dropout 0.5
- Cabeza de clasificación: `Linear(1024, num_classes)`
- Cabeza de regresión: `Linear(1024, 4)` — class-agnostic (estándar en Cascade R-CNN)

**Features para destilamiento:**

Las activaciones de la segunda capa FC (`FC2`) de cada stage se guardan en `distill_feats`. El estudiante puede destilar desde estas representaciones de alto nivel además de los soft labels.

---

## Outputs del forward pass

```python
output = model(rgb, nir, proposals)
```

| Key                       | Shape                        | Descripción                                      |
|---------------------------|------------------------------|--------------------------------------------------|
| `cls_scores`              | `[3 × (N_rois, 3)]`          | Scores de clasificación por stage de cascade     |
| `bbox_deltas`             | `[3 × (N_rois, 4)]`          | Deltas de regresión por stage                    |
| `distill_backbone_rgb`    | `[S1..S4]`                   | Features RGB del backbone (para destilar)        |
| `distill_backbone_fused`  | `[F1..F4]`                   | Features fusionados post cross-attention          |
| `distill_fpn`             | `[P2..P5]`                   | Pirámide FPN unificada (para destilar)           |
| `distill_head`            | `[FC2_s1, FC2_s2, FC2_s3]`   | Activaciones FC2 por stage (para destilar)       |

---

## Estrategia de destilamiento (resumen)

El estudiante (YOLOv8 Nano, solo RGB) recibe supervisión desde múltiples niveles:

```
L_total = L_det + λ1·L_feat_backbone + λ2·L_feat_fpn + λ3·L_LD

L_det           : pérdida de detección estándar del estudiante
L_feat_backbone : MSE entre features del backbone (con projection layers)
L_feat_fpn      : MSE entre outputs del FPN/PAN
L_LD            : Localization Distillation — distribución de bboxes del maestro
```

Los `projection layers` son necesarios porque el estudiante (YOLOv8 Nano) tiene dimensiones de features distintas al maestro. Son `1×1 conv` aprendibles que adaptan las dimensiones antes del MSE.

---

## Uso

```python
from src.models.master import MasterModel
import torch

model = MasterModel(
    num_classes=3,
    pretrained_backbone=True,   # ImageNet-1K weights
    fpn_channels=256,
    fc_channels=1024,
    roi_out_size=7,
    fusion_dropout=0.1,
    fpn_dropout=0.1,
).cuda()

# Freeze primeros 2 stages del backbone (recomendado con dataset pequeño)
model.freeze_backbone(freeze_stages=2)

rgb = torch.randn(4, 3, 640, 640).cuda()
nir = torch.randn(4, 1, 640, 640).cuda()
proposals = [...]  # list of (M_i, 4) tensors

output = model(rgb, nir, proposals)
```

---

## Estructura de archivos

```
src/models/master/
├── __init__.py          # Exports públicos
├── backbone.py          # DualConvNeXtBackbone
├── fusion.py            # CrossModalFusion (cross-attention)
├── neck.py              # DualFPN (dos FPNs + fusión)
├── head.py              # CascadeRCNNHead (3 stages)
├── master_model.py      # MasterModel (integración completa)
├── test_arch.py         # Sanity check — correr con: python3 -m src.models.master.test_arch
└── README.md            # Este archivo
```

---

## Referencias

- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **Cascade R-CNN**: Cai & Vasconcelos, *Cascade R-CNN: Delving into High Quality Object Detection*, CVPR 2018
- **FPN**: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017
- **CMX**: Zhang et al., *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation*, TPAMI 2023
- **TokenFusion**: Wang et al., *Multimodal Token Fusion for Vision Transformers*, CVPR 2022
- **Localization Distillation**: Zheng et al., *Localization Distillation for Dense Object Detection*, CVPR 2022
