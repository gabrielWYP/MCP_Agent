# Detección de Daño en Mango con Imágenes Multiespectrales y Knowledge Distillation

**Tesis de pregrado — Ingeniería de Sistemas**

Pipeline end-to-end de detección de daño en mango usando imágenes RGB+NIR (cámara MAPIR Survey3W), modelos de visión multimodal con backbone ConvNeXtV2, y conocimiento destilado (KD) hacia un modelo estudiante YOLOv8-Nano para despliegue eficiente.

---

## 1. Visión General

El proyecto construye un sistema de detección de daño en frutos de mango usando:

- **Imágenes multiespectrales** (RGB + NIR) capturadas con cámara MAPIR Survey3W
- **Anotación automática** con Florence-2-large + refinamiento manual en Label Studio
- **Modelo maestro (Teacher)** multimodal: DualConvNeXt + CrossModalFusion + DualFPN + YOLODetectionHead (~50M params)
- **Modelo estudiante (Student)** YOLOv8-Nano from scratch (~6.9M params) para recibir conocimiento destilado en capas intermedias
- **Training loop** YOLOv8 personalizado (TAL assigner, CIoU, Focal Loss) — sin dependencia de ultralytics
- **Knowledge Distillation** a nivel de features (backbone, FPN, head stems), no solo logits finales

### Resultados actuales

| Métrica | Valor |
|---------|-------|
| mAP@0.5 (maestro) | **0.3003** |
| AP mango (clase 0) | 0.57 |
| AP daño (clase 1) | 0.03–0.08 |
| Pares de imágenes | 65 RGB+NIR |
| Bboxes de daño | 186 manuales |
| Épocas entrenadas | 80 (Fase 1, backbone congelado) |

---

## 2. Arquitectura

### 2.1 Modelo Maestro (Teacher) — `src/models/master/`

```
RGB ──→ DualConvNeXtSmall ──→ CrossModalFusion ──→ DualFPN ──→ YOLODetectionHead
NIR ──→ DualConvNeXtSmall ──┘                                    │
                                                                  ├── preds (bboxes + clases)
                                                                  ├── distill_backbone → Proyecciones → KD
                                                                  ├── distill_fpn      → Proyecciones → KD
                                                                  ├── distill_head_cls → Proyecciones → KD
                                                                  └── distill_head_reg → Proyecciones → KD
```

- **DualConvNeXt**: Dos backbones ConvNeXtV2 independientes para RGB y NIR (pesos compartidos en Fase 1)
- **CrossModalFusion**: Fusión cross-modal aprendible (concat + convolución 1×1)
- **DualFPN**: Feature Pyramid Network bidireccional con C2f blocks, salida 3 niveles [P3, P4, P5]
- **YOLODetectionHead**: Head YOLOv8 decoupled (cls + reg) con bias init calibrado
- **ProjectionLayers**: Capas de proyección lineal que adaptan features del maestro a las dimensiones del estudiante para KD

### 2.2 Modelo Estudiante (Student) — `src/models/student/`

```
RGB ──→ CSPDarknetNano ──→ PANet ──→ YOLOStudentHead
                             │              │
                             ├── distill_fpn       → compatible con proyecciones del maestro
                             ├── distill_head_cls  → compatible con proyecciones del maestro
                             └── distill_head_reg  → compatible con proyecciones del maestro
```

El estudiante expone un contrato de **7 claves de output** idéntico al del maestro, permitiendo KD a nivel de features intermedias: `distill_backbone`, `distill_fpn`, `distill_head_cls`, `distill_head_reg`.

| Módulo | Params |
|--------|--------|
| CSPDarknetNano | 1.27M |
| PANet Neck | 2.13M |
| YOLOStudentHead | 3.47M |
| **Total** | **6.87M** |

---

## 3. Pipeline End-to-End

```
OCI Object Storage                    Label Studio
      │                                    │
      ▼                                    ▼
download_oci.py                  Anotación manual
(65 pares RGB+NIR)               (186 bboxes daño en NIR)
      │                                    │
      ▼                                    ▼
annotate_mango_florence.py       convert_nir_labels.py
(Florence-2-large → bboxes)      (Homografía NIR→RGB)
      │                                    │
      └────────────┬───────────────────────┘
                   ▼
           Dataset YOLO (RGB+NIR .txt labels)
                   │
                   ▼
          Training Loop (80 epochs)
          ┌────────┴────────┐
          ▼                 ▼
   MasterModel        YOLOv8-Nano Student
   (Teacher ~50M)     (Student ~6.9M)
          │                 │
          └────────┬────────┘
                   ▼
          Knowledge Distillation
          (features intermedias)
```

### Componentes del pipeline

| Script / Módulo | Función |
|-----------------|---------|
| `scripts/download_oci.py` | Descarga pares RGB+NIR desde bucket OCI |
| `scripts/annotate_mango_florence.py` | Detección de bboxes de mango con Florence-2-large |
| Label Studio (externo) | Anotación manual de daño en imágenes NIR |
| `scripts/convert_nir_labels.py` | Conversión NIR→RGB vía matriz de homografía |
| `src/training/` | Training loop YOLOv8 personalizado |
| `src/data_pipeline/` | OCI client y descubrimiento de pares RGB/NIR para poblar cache |

---

## 4. Bugs Resueltos Durante el Entrenamiento

El pipeline de entrenamiento atravesó 7 bugs críticos que fueron diagnosticados y corregidos:

| # | Bug | Solución |
|---|-----|----------|
| 1 | **AMP NaN** — Loss divergía a NaN con Automatic Mixed Precision | `exp clamp` en `bbox_decode()` |
| 2 | **TAL Soft Targets** — Positivos con target≈0, gradiente nulo | Binary targets directos (positivo=1.0) |
| 3 | **OHEM** — Online Hard Example Mining reforzaba falsos positivos | OHEM desactivado, Focal Loss basta |
| 4 | **NIR Padding** — Padding RGB (114) producía valores erróneos en NIR | Padding separado con `nir_mean * 255 ≈ 14` |
| 5 | **Bias Init** — Pérdida inicial muy alta | `cls_pred` bias init = −4.6 |
| 6 | **Focal Loss γ** — γ=1.5 insuficiente para desbalance extremo | γ=2.0 |
| 7 | **TAL Fallback** — Assertion cuando cero matches válidos | Fallback a asignación por IoU máximo |

---

## 5. Desarrollo con SDD (Spec-Driven Development)

El proyecto se desarrolló siguiendo la metodología SDD con artefactos versionados:

```
openspec/
├── specs/              ← Especificaciones "source of truth"
│   ├── testing-teacher-arch/
│   ├── data-augmentation/
│   ├── data-extraction/
│   ├── data-preprocessing/
│   ├── training-loop/
│   ├── training-metrics/
│   ├── yolo-dataset/
│   ├── yolo-loss/
│   └── yolo-nano-student/
└── changes/archive/    ← Cambios completados
    ├── 2026-05-18-testing-bootstrap/
    ├── 2026-06-01-fix-dualfpn-levels/
    ├── 2026-06-01-data-aug-pipeline/
    ├── 2026-06-02-pipeline-e2e-homography/
    ├── 2026-06-07-mastermodel-training-loop/
    └── 2026-06-07-yolo-nano-student/
```

---

## 6. Estructura del Proyecto

```
.
├── src/
│   ├── agent/              # LangGraph orchestrator (en pausa)
│   ├── annotation/         # Florence-2, bbox projection, NIR segmentation
│   ├── data_pipeline/      # OCI client y pair discovery para cache RGB/NIR
│   ├── models/
│   │   ├── master/         # Teacher: DualConvNeXt, Fusion, DualFPN, YOLODetectionHead
│   │   └── student/        # Student: CSPDarknetNano, PANet, YOLOStudentHead
│   ├── training/           # Training loop: dataset, loss (TAL+BCE+CIoU), metrics, augmentations
│   ├── storage_logic/      # Object storage (S3/OCI)
│   ├── utils/              # Logger, object storage helpers
│   └── variables/          # Config variables
├── scripts/                # Pipeline scripts (download, annotation, conversion)
├── tests/
│   ├── data_pipeline/      # OCI config y pair discovery
│   ├── training/           # 30 tests (letterbox, TAL, mAP, training step, e2e)
│   └── models/student/     # 12 tests (backbone, neck, head, StudentModel integration)
├── configs/                # Training configs (training_mango.yaml)
├── config/                 # OCI credentials (gitignored)
├── calibracion/            # Camera calibration data
├── checkpoints/            # Model checkpoints (best_model.pt)
├── data/                   # Datasets (zips: RGB, NIR, YOLO labels)
├── notebooks/              # Homography scripts, Gemini manager, utils
├── openspec/               # SDD specs and archived changes
└── resumen_sesion_*.md     # Session summaries
```

---

## 7. Setup

### Requisitos

- Python 3.12+ (venv incluido: `myLinuxVenv`)
- PyTorch 2.1+ con CUDA (para entrenamiento)
- 10 GB+ VRAM recomendado (RTX 3080 usado en desarrollo)

### Instalación

```bash
# Activar entorno virtual
source myLinuxVenv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Variables de entorno (credenciales OCI, API keys)
cp .env.example .env  # Completar con valores reales
```

### Tests

```bash
# Tests de modelos (maestro + estudiante)
python -m pytest tests/models/ -v

# Tests de training loop
python -m pytest tests/training/ -v

# Tests de descarga/pair discovery
python -m pytest tests/data_pipeline/ -v

# Todos los tests
python -m pytest tests/ -v
```

---

## 8. Próximos Pasos

1. **Knowledge Distillation training** — Entrenar al estudiante usando las proyecciones del maestro
2. **Más datos** — 100-200+ pares RGB+NIR adicionales para mejorar AP de daño
3. **Fase 2 de entrenamiento** — Descongelar backbone del maestro cuando haya ≥200 imágenes
4. **Exportar modelos** — ONNX / TorchScript para inferencia en producción
5. **Mergear feature-branch-chain** del estudiante a main
6. **Redacción de tesis** — Documentar métricas, arquitectura y resultados

---

## 9. Commits Relevantes

| Commit | Descripción |
|--------|-------------|
| `4d9d98f` | Pipeline v1 completo: OCI download, Florence-2, training, mAP 0.30 |
| `e256856` | Training curves plot |
| `6bd270f` | CSPDarknet-Nano backbone + building blocks |
| `be1868d` | PANet neck + DecoupledHead con stems de KD |
| `803abf7` | StudentModel integración final, contrato 7-key KD |

---

*Repositorio de tesis. Desarrollado con SDD (Spec-Driven Development) + Engram persistent memory.*
