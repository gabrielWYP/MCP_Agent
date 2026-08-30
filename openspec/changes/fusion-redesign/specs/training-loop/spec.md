# Delta for training-loop

## MODIFIED Requirements

### Requirement: End-to-End Training With Discriminative Learning Rate

The trainer SHALL train MasterModel end-to-end from epoch 0: `freeze_stages=0` for the
entire run, with no frozen phase and no unfreeze transition. Pretrained backbone stages
MUST train at 0.1x the neck/head learning rate (discriminative LR); neck and head MUST
train at the configured base LR. When model_type="student", the existing single-phase
schedule continues to apply unchanged.
(Previously: "Two-Phase Fine-Tuning" — Phase 1 froze the entire backbone at lr=1e-3 on
fusion/neck/head only; Phase 2 unfroze stages 3-4 at lr=1e-4. The evaluated run never
effectively reached Phase 2 (D2), which motivates removing the two-phase schedule
for this model rather than fixing its trigger condition)

#### Scenario: MasterModel trains end-to-end from epoch 0
- GIVEN MasterModel with model_type="master"
- WHEN training starts
- THEN every backbone stage, neck, and head parameter has requires_grad=True from epoch 0
- AND no freeze/unfreeze transition occurs at any point during the run

#### Scenario: Discriminative learning rate applied to pretrained stages
- GIVEN a training step
- WHEN the optimizer parameter groups are built
- THEN pretrained backbone-stage parameters use 0.1x the neck/head learning rate
- AND both the backbone group and the neck/head group receive gradient updates every step

#### Scenario: Student training skips two-phase entirely
- GIVEN model_type="student"
- WHEN training begins
- THEN no freeze/unfreeze phases SHALL execute
- AND all model parameters SHALL be trainable from epoch 1

## ADDED Requirements

### Requirement: Verifiable Schedule Execution

The trainer MUST log per-module gradient norms (per backbone stage, neck, head) at a
configurable interval, and MUST support an end-of-run init-distance check that flags any
module whose parameters remain bit-exact at initialisation after training, using the same
LayerNorm bit-exactness check (`max|gamma-1| == 0.000e+00`) that exposed the previous
design's silently-unreached Phase 2 and disconnected pyramid level.

#### Scenario: Grad-norm logging covers every trainable module
- GIVEN per-module gradient-norm logging enabled
- WHEN training runs for at least one step
- THEN a gradient-norm value is recorded for every backbone stage, the neck, and the head

#### Scenario: Init-distance check flags an untrained module
- GIVEN a completed training run
- WHEN the end-of-run init-distance check runs on every backbone stage, neck, and head module
- THEN it flags any module whose parameters are bit-exactly at initialisation
- AND a flagged run MUST be treated as a failed run, not a valid result for any hypothesis

### Requirement: Fatal Batch-Skip Guards (D-G)

The trainer MUST count every batch skipped due to a CUDA out-of-memory (OOM)
error or a NaN/Inf loss, and MUST NOT allow an epoch that took zero optimizer
steps to complete as if it had trained. A run in which every batch fails
MUST raise, not report `total_loss=0.0` and save a checkpoint from a model
that never received a gradient.
(Previously: both the OOM handler and the NaN/Inf guard caught the fault,
printed a warning, and `continue`d past it before the step counter
incremented; the divisor was then floored via `n_batches = max(n_batches, 1)`,
so an all-failing epoch silently reported a valid-looking `0.0` loss instead
of failing.)

#### Scenario: An epoch where every batch OOMs raises
- GIVEN a training epoch numbered 2 or later
- WHEN every batch in that epoch raises a CUDA out-of-memory error
- THEN the epoch raises a `RuntimeError` and does not save a checkpoint for that epoch

#### Scenario: OOM is tolerated only on epoch 1
- GIVEN the first training epoch of a phase (allocator warm-up)
- WHEN a batch raises a CUDA out-of-memory error
- THEN the batch is skipped and counted in `oom_skipped`
- AND the epoch continues, raising only if it ends with zero optimizer steps

#### Scenario: OOM is not tolerated from epoch 2 onward
- GIVEN a training epoch numbered 2 or later
- WHEN any single batch raises a CUDA out-of-memory error
- THEN the trainer raises immediately, without waiting for the epoch to end

#### Scenario: An epoch where every batch is NaN raises and records the count
- GIVEN a training epoch
- WHEN every batch produces a NaN or Inf loss
- THEN `nan_skipped` reflects the number of skipped batches
- AND the epoch raises a `RuntimeError` because zero optimizer steps were taken

#### Scenario: Zero optimizer steps is always fatal
- GIVEN a training epoch, regardless of whether the cause is OOM, NaN/Inf, or an empty loader
- WHEN the epoch completes having executed zero optimizer steps
- THEN the trainer raises a `RuntimeError` naming the epoch and the fault counters

#### Scenario: Fault counters are recorded for every completed epoch
- GIVEN a training epoch that completes without raising
- WHEN the epoch's metrics are recorded
- THEN `oom_skipped`, `nan_skipped`, and `steps_taken` are present in `LossHistory.extra_losses`, in TensorBoard, and in the saved checkpoint dict

### Requirement: Gradient Accumulation With a Fixed Effective Batch (D-H)

The trainer MUST support gradient accumulation so that `effective_batch`
(the experimental constant) can be held fixed while `batch_size` (a memory
knob) varies across hardware. `grad_accum_steps` MUST be derived as
`effective_batch // batch_size` and MUST NOT be settable directly; a
non-evenly-divisible pair MUST raise. Gradient clipping MUST be applied once
per effective (accumulated) step, never once per micro-batch.

#### Scenario: Accumulated and full-batch training produce the same update
- GIVEN two training runs processing the same samples in the same order, one at `batch_size=2` with `grad_accum_steps=4` and one at `batch_size=8` with `grad_accum_steps=1` (`effective_batch=8` in both)
- WHEN each runs exactly one effective step
- THEN the resulting model parameters are numerically equivalent (`torch.allclose`) between the two runs

#### Scenario: Gradient clipping runs once per effective step
- GIVEN `grad_accum_steps > 1`
- WHEN a training epoch runs one full accumulation window (multiple micro-batches)
- THEN `clip_grad_norm_` is called exactly once for that window, not once per micro-batch

#### Scenario: A non-divisible effective_batch/batch_size pair raises
- GIVEN `effective_batch` is not evenly divisible by `batch_size`
- WHEN the training configuration is constructed
- THEN a `ValueError` is raised before any training step runs

#### Scenario: A partial accumulation window is flushed at epoch end
- GIVEN an epoch whose number of successful micro-batches is not a multiple of `grad_accum_steps`
- WHEN the epoch's data loader is exhausted
- THEN the remaining accumulated gradient is flushed as one final (smaller) optimizer step, rather than being silently dropped

### Requirement: Explicit Precision Selection (D-I)

Training precision MUST be an explicit `fp32 | fp16 | bf16` selection,
defaulting to `fp32`. A boolean `amp` flag MUST NOT exist as a configuration
field. Requesting `bf16` on a CUDA device that does not support it MUST
raise at configuration-load time, naming the device, rather than silently
downgrading to `fp32`.
(Previously: `amp: bool = True` meant any configuration omitting the key
silently enabled fp16.)

#### Scenario: Default precision is fp32
- GIVEN a training configuration with no explicit precision setting
- WHEN the configuration is constructed
- THEN `precision` is `"fp32"`

#### Scenario: An invalid precision value is rejected
- GIVEN a `precision` value outside `{"fp32", "fp16", "bf16"}`
- WHEN the training configuration is constructed
- THEN a `ValueError` is raised

#### Scenario: bf16 on an unsupported device raises and names the device
- GIVEN `precision="bf16"` and a CUDA device that does not support bf16
- WHEN the training configuration is constructed
- THEN a `ValueError` is raised naming the active device
- AND the configuration does not silently fall back to fp32

#### Scenario: A backbone's forced-fp32 numerical guard is scoped to fp16
- GIVEN a backbone module with an internal forced-fp32 guard originally justified by fp16 autocast fragility
- WHEN training precision is `bf16` or `fp32`
- THEN the guard does not disable autocast (transparent pass-through)
- AND WHEN training precision is `fp16`, the guard disables autocast as before

### Requirement: Machine-Profile / Experiment Configuration Split (D-H)

Hardware-dependent configuration (`device`, `batch_size`, `num_workers`,
`pin_memory`, `precision`) MUST be separable from the experiment being
measured, and a machine profile MUST NOT be able to set any key outside
that whitelist. Every run MUST be able to record a SHA-256 hash of the
experiment configuration file's raw bytes; two runs are comparable only if
this hash and `effective_batch` both match.

#### Scenario: A machine profile setting a non-whitelisted key raises
- GIVEN a machine-profile file or dict containing a key outside `{device, batch_size, num_workers, pin_memory, precision}`
- WHEN the profile is loaded or applied to a training configuration
- THEN a `ValueError` is raised and no config field is modified

#### Scenario: experiment_sha256 is stable across machine profiles
- GIVEN one fixed experiment configuration file
- WHEN `experiment_sha256` is computed under two different machine profiles
- THEN the resulting hash is identical in both cases

#### Scenario: experiment_sha256 changes on any byte-level edit
- GIVEN an experiment configuration file
- WHEN a single byte of that file changes
- THEN `experiment_sha256` computed on the edited file differs from the original

#### Scenario: Device is configurable, not hardcoded
- GIVEN a training configuration with `device` set to an explicit value (e.g. `"cpu"`, `"cuda:0"`)
- WHEN the trainer is constructed
- THEN it resolves to that device rather than the hardcoded CUDA-availability check
- AND WHEN `device="auto"` (the default), it resolves exactly as the previous hardcoded expression did
