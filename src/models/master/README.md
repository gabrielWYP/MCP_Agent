# Master Model — Multimodal Mango Damage Detector

Arquitectura del modelo maestro para la tesis de **destilamiento de conocimiento cruzado multimodal** aplicado a la detección de daño mecánico temprano en mango (*Mangifera indica*).

---

## Contexto del problema

El daño mecánico temprano en mango se manifiesta como **pardismo interno** antes de ser visible en la superficie. Esto hace que la detección en imágenes RGB convencionales sea extremadamente difícil. Las imágenes **NIR (Near-Infrared)** capturan cambios en la absorción de agua y azúcares en el tejido dañado, revelando la lesión cuando el RGB aún no muestra nada.

El modelo maestro aprovecha **ambas modalidades** durante el entrenamiento. El modelo estudiante (YOLO Nano) aprende a imitar al maestro usando **solo RGB** en inferencia — ese es el núcleo del destilamiento cruzado multimodal.

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
                            [P3, P4, P5]
                        (pirámide unificada)
                                      │
                                      ▼
                          YOLODetectionHead
                      (anchor-free, decoupled, 3 niveles)
                                      │
                                      ▼
              preds + cls_preds + reg_preds + distill_feats
```

---

## Head YOLO-style (compatible con YOLO Nano)

El head reemplaza el anterior Cascade R-CNN por un diseño **anchor-free decoupled** idéntico en formato al de YOLO Nano, permitiendo distilación directa:

```
P_i (256ch) ──► conv 3×3 + SiLU ──► conv 1×1 (nc)  → cls_preds
              └─► conv 3×3 + SiLU ──► conv 1×1 (4)  → reg_preds
```

**Output por nivel:** `(B, nc + 4, H_i, W_i)` donde:
- `nc = 2` (sano, danado)
- `4` = bbox deltas (xywh)

**Compatibilidad con YOLO Nano:**

| Propiedad | Maestro | YOLO Nano | Distilación |
|---|---|---|---|
| Formato output | `(B, 6, H_i, W_i)` | `(B, 6, H_i, W_i)` | ✅ Directa |
| Strides | `[8, 16, 32]` | `[8, 16, 32]` | ✅ Idénticos |
| Niveles | P3, P4, P5 | P3, P4, P5 | ✅ Idénticos |
| Clases | 2 (sano, danado) | 2 (sano, danado) | ✅ Idénticas |

---

## Parámetros totales

| Módulo        | Parámetros   |
|---------------|-------------|
| Backbone      | ~49M        |
| Fusion        | ~9M         |
| Neck          | ~6M         |
| Head (YOLO)   | ~1.5M       |
| **Total**     | **~65M**    |

El head YOLO es ~27x más liviano que el anterior Cascade R-CNN (~41M).

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

### 2. `CrossModalFusion` — `fusion.py`

Aplica **cross-attention** entre las features RGB y NIR en cada uno de los 4 stages del backbone.

**¿Por qué RGB como Query?**

En inferencia, el estudiante solo ve RGB. Al hacer que RGB sea el Query durante el entrenamiento del maestro, forzamos al stream RGB a aprender a *buscar activamente* la información que el NIR revelaría. Esto enriquece las representaciones RGB con conocimiento NIR latente — exactamente lo que el estudiante necesita aprender a imitar.

**Pooling espacial adaptativo:**

La atención sobre secuencias largas es O(n²) en memoria. S1 tiene 160×160 = 25,600 tokens → ~20 GB solo para la matriz de atención. La solución es hacer `adaptive_avg_pool2d` a un grid de `max_tokens_side × max_tokens_side` (default: 20×20 = 400 tokens) antes de la atención, y luego `bilinear upsample` de vuelta a la resolución original.

### 3. `DualFPN` — `neck.py`

Dos FPNs paralelos (uno por modalidad) que construyen pirámides multi-escala independientes antes de fusionarse.

```
fused_features [F1..F4] ──► FPN-RGB ──► [P3..P5] a 256ch ──┐
                                                              ├──► concat + 1×1 conv ──► [P3..P5] unificado
nir_features   [S1..S4] ──► FPN-NIR ──► [P3..P5] a 256ch ──┘
```

**Salidas:**

| Nivel | Stride | Resolución (640×640) | Canales |
|-------|--------|----------------------|---------|
| P3    | 8      | 80 × 80              | 256     |
| P4    | 16     | 40 × 40              | 256     |
| P5    | 32     | 20 × 20              | 256     |

### 4. `YOLODetectionHead` — `head.py`

Head de detección **anchor-free decoupled** estilo YOLOv8.

**¿Por qué YOLO-style en vez de Cascade R-CNN?**

1. **Formato idéntico al estudiante** → distilación directa sin projection layers complejos en el head
2. **Features densos** → el FPN del maestro y del estudiante operan sobre la misma estructura espacial
3. **Head liviano** → ~1.5M params vs ~41M del Cascade R-CNN

```
P_i (256ch)
  ├──► cls_stem: Conv 3×3 + SiLU ──► cls_pred: Conv 1×1 (nc)
  └──► reg_stem: Conv 3×3 + SiLU ──► reg_pred: Conv 1×1 (4)
```

**Clases de detección:**

| ID | Clase     |
|----|-----------|
| 0  | sano      |
| 1  | danado    |

**Features para destilamiento:**

Las activaciones de `cls_stem` y `reg_stem` de cada nivel se exponen en `distill_cls` y `distill_reg`. El estudiante puede destilar desde estas representaciones intermedias además de los outputs finales.

### 5. `ProjectionLayers` — `distill_projections.py`

Proyecciones 1×1 aprendibles para alinear dimensiones maestro → estudiante en feature-level distillation.

```
Maestro FPN[P3] (256ch) ──► proj 1×1 ──► (128ch) ──┐
                                                      ├──► MSE loss
Estudiante P3 (128ch) ───────────────────────────────┘
```

Presets incluidos:
- `fpn_projections()` — para distilación a nivel de FPN
- `backbone_projections()` — para distilación a nivel de backbone
- `head_projections()` — para distilación a nivel de head

---

## Outputs del forward pass

```python
output = model(rgb, nir)  # Sin proposals — head YOLO es dense
```

| Key                       | Shape                        | Descripción                                      |
|---------------------------|------------------------------|--------------------------------------------------|
| `preds`                   | `[(B, 6, H3, W3), ...]`      | Predicciones concatenadas (cls + reg) por nivel  |
| `cls_preds`               | `[(B, 2, H3, W3), ...]`      | Scores de clasificación por nivel                |
| `reg_preds`               | `[(B, 4, H3, W3), ...]`      | Deltas de bbox (xywh) por nivel                  |
| `distill_backbone_rgb`    | `[S1..S4]`                   | Features RGB del backbone                        |
| `distill_backbone_fused`  | `[F1..F4]`                   | Features fusionados post cross-attention         |
| `distill_fpn`             | `[P3..P5]`                   | Pirámide FPN unificada                           |
| `distill_head_cls`        | `[cls_stem x3]`              | Features intermedios de clasificación            |
| `distill_head_reg`        | `[reg_stem x3]`              | Features intermedios de regresión                |

---

## Estrategia de destilamiento (resumen)

El estudiante (YOLO Nano, solo RGB) recibe supervisión desde múltiples niveles:

```
L_total = L_det + λ1·L_feat_fpn + λ2·L_feat_head + λ3·L_output

L_det           : pérdida de detección estándar del estudiante
L_feat_fpn      : MSE(FPN_maestro_proyectado, FPN_estudiante)
L_feat_head     : MSE(head_stem_maestro_proyectado, head_stem_estudiante)
L_output        : KL(cls_maestro || cls_estudiante) + IoU loss
```

Los `projection layers` se entrenan junto con el estudiante durante KD.

---

## Uso

```python
from src.models.master import MasterModel
import torch

model = MasterModel(
    num_classes=2,
    pretrained_backbone=True,   # ImageNet-1K weights
    fpn_channels=256,
    fusion_dropout=0.1,
    fpn_dropout=0.1,
).cuda()

# Freeze primeros 2 stages del backbone (recomendado con dataset pequeño)
model.freeze_backbone(freeze_stages=2)

rgb = torch.randn(4, 3, 640, 640).cuda()
nir = torch.randn(4, 1, 640, 640).cuda()

output = model(rgb, nir)
```

---

## Estructura de archivos

```
src/models/master/
├── __init__.py              # Exports públicos
├── backbone.py              # DualConvNeXtBackbone
├── fusion.py                # CrossModalFusion (cross-attention)
├── neck.py                  # DualFPN (dos FPNs + fusión)
├── head.py                  # YOLODetectionHead (anchor-free, decoupled)
├── distill_projections.py   # Projection layers para KD
├── master_model.py          # MasterModel (integración completa)
├── test_arch.py             # Sanity check
└── README.md                # Este archivo
```

---

## Referencias

- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **TOOD**: Task-aligned One-stage Object Detection, ICCV 2021
- **FPN**: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017
- **CMX**: Zhang et al., *CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation*, TPAMI 2023
- **TokenFusion**: Wang et al., *Multimodal Token Fusion for Vision Transformers*, CVPR 2022
- **Localization Distillation**: Zheng et al., *Localization Distillation for Dense Object Detection*, CVPR 2022
