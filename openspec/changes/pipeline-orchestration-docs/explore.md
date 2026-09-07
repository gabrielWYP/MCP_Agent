## Exploration: Pipeline Orchestration Documentation

### Current State

The project has two independent training pipeline orchestrators plus several standalone scripts that together form the complete training workflow. The pipelines have been constructed incrementally across multiple SDD changes (mastermodel-training-loop, student-training-loop, kd-training, yolo-nano-student, data-aug-pipeline, pipeline-e2e-homography) but no single document ties them together. The pipeline currently spans 6 scripts and 3 YAML configs with no integration-level documentation or tests.

**Pipeline A — Annotation-to-Training** (`scripts/run_training_pipeline.py`):
1. `prepare_yolo_splits.py` — Creates deterministic train/val/test split assignments, empty label files, and Label Studio task template
2. `annotate_mango_florence.py` — Runs Florence-2 for class-0 mango bbox annotation
3. `convert_nir_labels.py` — Converts Label Studio NIR damage annotations → YOLO class-1 via homography projection
4. `src.training.train` — Trains MasterModel (2-phase) or StudentModel (1-phase)

**Pipeline B — Three-Stage Final Training** (`scripts/run_final_training_pipeline.py`):
1. `src.training.train` — MasterModel (RGB+NIR teacher, `configs/training_mango.yaml`)
2. `src.training.train` — StudentModel baseline (RGB-only, `configs/training_student.yaml`)
3. `src.training.kd_train` — KD Student (distilled from frozen teacher, `configs/kd_training.yaml`)

Pipeline B produces timestamped atomic output under `checkpoints/final_runs/{run_id}/{maestro,estudiante,destilado}/` with stage_summary.json, metrics CSV, PNG curves, TensorBoard logs, and a root `run_summary.json` + `final_metrics_summary.png`.

### Affected Areas

- `scripts/run_training_pipeline.py` — Annotation-to-training orchestrator; needs docstring enrichment and environment-level documentation
- `scripts/run_final_training_pipeline.py` — Three-stage final training orchestrator; complex dependency chain with `run_artifacts.py`
- `scripts/prepare_yolo_splits.py` — Split preparation; well-documented internally but missing from any top-level README
- `scripts/annotate_mango_florence.py` — Florence-2 annotator; heavy dependency (transformers, 0.9-3.4GB model), requires GPU-reasonable hardware
- `scripts/convert_nir_labels.py` — Homography projection; depends on Label Studio export being available
- `configs/training_mango.yaml` — MasterModel hyperparameters; phase-2 skipped by design
- `configs/training_student.yaml` — Student baseline; single-phase, carries unused NIR fields
- `configs/kd_training.yaml` — KD config; diverges from others on num_workers and warmup_epochs
- `src/training/config.py` — TrainingConfig Python defaults differ from YAML values
- `src/training/kd_config.py` — KDConfig Python defaults differ from YAML values
- `src/training/train.py` — CLI entry point for master/student training
- `src/training/kd_train.py` — CLI entry point for KD training
- `src/training/run_artifacts.py` — Artifact utilities shared by both orchestrators
- `src/training/loop.py` — Two-phase training loop (used by train.py)
- `src/training/kd_trainer.py` — KD trainer (subclass of Trainer)
- `requirements.txt` — Missing `transformers` version pin for Florence-2 compatibility; `einops` and `flash-attn` noted but optional

### Approaches

1. **OpenSpec-spec + README section + inline docstrings** — Create a new `pipeline-orchestration` domain spec defining pipeline stage ordering, configuration contract, and CLI contracts. Add a "Training Pipeline" section to the project README with a visual flow diagram. Enrich existing docstrings where they fall short. This is the minimal approach that closes the documentation gap.
   - Pros: Formal spec survives SDD archive; README visible to all contributors; docstrings help IDE users; no code changes needed
   - Cons: No additional safeguards against config drift; no integration tests; doesn't resolve the config inconsistencies found
   - Effort: Medium (3-5 files of documentation)

2. **Spec + README + config validation script** — Same as above, plus a `scripts/validate_configs.py` that checks config consistency (class_weights alignment, num_workers justification, warmup_epochs consistency) and is run in CI or pre-commit. Fix the inconsistencies found (or document them as intentional).
   - Pros: Prevents config drift; gives script-level validation; fixes real inconsistencies
   - Cons: Requires code changes to scripts/configs; more work; config fixes may need hyperparameter justification
   - Effort: High (docs + validation script + config adjustments)

3. **Full pipeline integration tests + docs** — Write integration tests for the full pipeline chain using mock datasets, then document the pipeline.
   - Pros: Highest confidence; tests document behavior
   - Cons: Very high effort; requires mock data; not aligned with current testing maturity (arch test broken, no CI)
   - Effort: Very High

### Recommendation

**Approach 1** — OpenSpec spec + README section + inline docstring enrichment.

This is the right level of investment for the current project maturity. The config inconsistencies found (`num_workers` 2 vs 4, `warmup_epochs` 5 vs 3) should be surfaced to the maintainer for a decision: either fix them now or document them as intentional with a rationale comment in the YAML files. Adding a config validation script is worth considering but belongs in a follow-up change after the documentation baseline is established.

### Findings — Config Inconsistencies and Risks

| Finding | Severity | Details |
|---------|----------|---------|
| `num_workers` diverges | Low | `training_mango.yaml`=2, `training_student.yaml`=2, `kd_training.yaml`=4. KD config has double the workers — may be intentional (KD uses student config, not master) but undocumented |
| `warmup_epochs` diverges | Low | `training_mango.yaml`=5, `training_student.yaml`=5, `kd_training.yaml`=3. KD config overrides to lower warmup — undocumented |
| `log_interval` diverges | Low | `training_mango.yaml`=5, `kd_training.yaml`=10. Minor difference in logging frequency |
| YAML vs Python defaults diverge | Medium | `TrainingConfig` defaults: batch_size=2, epochs_phase1=50, amp=True, num_workers=4. YAML: batch_size=8, epochs_phase1=80, amp=False, num_workers=2. Python defaults do not match any real config — a developer running without `--config` gets unexpected behavior |
| `nir_mean`/`nir_std` in student config | Very Low | Present in `training_student.yaml` but StudentModel doesn't use NIR. Dataset code still receives these values. Harmless but confusing |
| No pipeline-level tests | Medium | All 5 pipeline scripts (`prepare_yolo_splits.py`, `annotate_mango_florence.py`, `convert_nir_labels.py`, `run_training_pipeline.py`, `run_final_training_pipeline.py`) have zero test coverage |
| `class_weights` duplicated across 3 configs | Medium | [1.27, 0.83] appears in all three YAML files. Any update requires changing 3 files. No single source of truth. **Annotated 2026-08-29 (damage-map-audit, Q2):** `[1.27, 0.83]` is the arithmetically correct inverse-frequency weighting for the current dataset (mango 208 / damage 335, 1:1.61 — recomputed as `[1.305, 0.810]` in the audit, essentially this value). It is NOT the value currently deployed in `configs/*.yaml` or `TrainingConfig`'s default, which is `[0.5, 1.5]` — held constant deliberately for the damage-map-audit experiment ladder so a class-1 AP change can be attributed to the assigner fix rather than a reweighting. This claim is not deleted because it remains mathematically correct; it is simply not in force during that experiment. Revisiting `class_weights` on inverse-frequency grounds is a valid follow-up once the audit's validation completes. See `openspec/changes/damage-map-audit/proposal.md` Q2 and `openspec/specs/yolo-loss/spec.md` for the full rationale. |
| `run_training_pipeline.py` passes `--labels-dir` as both `--labels-dir` and `--splits` to `convert_nir_labels.py` | Low | Lines 120-122: `--output-dir` and `--splits` both point to `args.labels_dir`. This works because `convert_nir_labels.py` uses `--splits` to find train/val/test directories and `--output-dir` to write output. They're the same by design but the intent isn't clear |
| No docstring on `Trainer.__init__` parameter for `config.patience` usage across phases | Low | `TrainingConfig` has one `patience` field used for both phases, but the docstring says "patience=15". KD config overrides to 30. The KD trainer inherits all early stopping from `loop.Trainer` — works but doesn't document that the patience field is shared |
| `requirements.txt` `transformers` version pin | Medium | `transformers==4.39.3` is pinned and may not support the latest Florence-2 model variants. The Florence annotator uses `trust_remote_code=True` and `attn_implementation="eager"` to avoid flash-attn; these workarounds should be documented |

### Pipeline Chain — Data Flow Diagram (Mermaid)

```
┌─────────────────────────────────────────────────────────┐
│               PIPELINE A — Annotation-to-Training        │
│                                                          │
│  [RGB images] ─┬─ prepare_yolo_splits ──► splits.json   │
│  [NIR images] ─┘   │                     empty YOLO txt │
│                    ▼                    Label Studio tmpl│
│              Florence-2 ──► class-0 mango bboxes         │
│                    │                                     │
│                    ▼ (requires Label Studio export)      │
│              convert_nir ──► class-1 damage bboxes       │
│                    │         (homography NIR→RGB)        │
│                    ▼                                     │
│              src.training.train                          │
│              (MasterModel or StudentModel)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│            PIPELINE B — Three-Stage Final Training       │
│                                                          │
│  training_mango.yaml ──► MasterModel  ──► maestro/       │
│                          (RGB+NIR)                        │
│                              │                           │
│                              ▼ (teacher checkpoint)      │
│  training_student.yaml ──► StudentModel ──► estudiante/  │
│                            (RGB-only)                    │
│                              │                           │
│                              ▼ (baseline for reference)  │
│  kd_training.yaml ──► KD Student    ──► destilado/      │
│                       (distilled, uses frozen teacher)   │
│                                                          │
│  Output: checkpoints/final_runs/{run_id}/                │
│    ├── maestro/best_model_{id}_mAP{score}/              │
│    ├── estudiante/best_model_{id}_mAP{score}/           │
│    ├── destilado/best_model_{id}_mAP{score}/            │
│    ├── run_summary.json                                 │
│    └── final_metrics_summary.png                        │
└─────────────────────────────────────────────────────────┘
```

### Risks

- **Config drift risk**: Three YAML configs with overlapping-but-not-identical fields (class_weights duplicated, differing num_workers/warmup) create a maintenance burden. A future hyperparameter sweep could easily leave one config behind.
- **No integration verification**: A label format change in `YOLODataset` would silently break the annotation pipeline because no test verifies the end-to-end chain.
- **Missing env documentation**: The Florence-2 step requires GPU or it will be extremely slow; the homography step requires `matriz_homografia_aruco.npy`. Neither constraint is surfaced in documentation.
- **Python defaults != YAML**: Running `python -m src.training.train` without `--config` gives completely different hyperparameters than intended. This is a footgun for new contributors.

### Ready for Proposal

Yes — proceed to `sdd-propose` for this change. The orchestrator should tell the user that:

1. The exploration confirmed the two pipelines are functional but under-documented.
2. Three config inconsistencies were found (`num_workers`, `warmup_epochs`, `log_interval`) — the user should decide whether to fix them in this change or document them as intentional.
3. Approach 1 (OpenSpec spec + README + docstrings) is recommended.
4. The `class_weights` duplication across three configs is a medium-term risk but can be addressed later.
