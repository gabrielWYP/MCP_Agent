# Design: Early-fusion redesign — 4-channel RGB+NIR backbone

Change id: `fusion-redesign` · Store: hybrid · `delivery_strategy: exception-ok` (NEW `size:exception`, scoped to this change) · `review_budget_lines: 800` (formally exceeded, exception accepted)

Implements the binding decisions in `proposal.md` D-1 … D-7, `## Maintainer Decisions — Round 1`, and
`## Maintainer Decisions — Round 2 (hardware portability)`. Nothing here relitigates them.

**Evidence convention.** `VERIFIED` = read in the repository at the cited `file:line` in this session.
`MEASURED` = benchmarked by the maintainer on the target hardware and recorded in `proposal.md`
Round 2. `INFERRED` = analytical derivation from verified code or calibrated against a `MEASURED`
figure, not executed here. This agent had **no shell access**, so no test run, no GPU measurement, and
no instantiation was performed by me.

**Round 2 status.** Q5 and Q6 are **resolved** and are implemented below. The maintainer's measured
VRAM and wall-clock figures **supersede** the hand-computed activation estimates in the first draft of
§4; the parameter arithmetic is unchanged and was independently confirmed by the measurement (§4.2).
Round 2 adds one workstream (**W9 — hardware portability**) and one hypothesis (**H-BF16**), and
raises **D-G to a blocking prerequisite**.

**Deviation from the 800-word skill budget** is deliberate: the launch brief requires seven detailed
sections with `file:line` citations, a quantified memory budget, and full validation mechanics.

---

## 1. Technical approach

One stream, one backbone, one FPN, a configurable emitted-level list, and a stride list that is
carried by the config rather than by literals.

```
rgb (N,3,H,W) ─┐
               ├─ cat(dim=1) ─→ EarlyFusionBackbone ─→ [S1,S2,S3,S4]  (96,192,384,768)
nir (N,1,H,W) ─┘   Conv2d(4,96,k4,s4) + LN2d              │
                                                          ▼
                                            FPNNeck( SingleFPN ) ─→ [P2,P3,P4,P5] (256 ea.)
                                                          │  emit_levels ← head_strides
                                                          ▼
                                            YOLODetectionHead(strides=head_strides)
                                                          │
                                                          ▼
                            {preds, cls_preds, reg_preds, distill_backbone,
                             distill_fpn, distill_head_cls, distill_head_reg}   ← 7 keys
```

`MasterModel.forward(rgb, nir)` keeps its two-tensor signature (VERIFIED `master_model.py:101-106`),
so `dataset.py`, `loop.py:344-357` and `kd_trainer.py:147` need no call-site edit. The concatenation
is internal to the model. That is a deliberate cost/benefit choice: a `forward(x4)` signature is
cleaner, but it would force edits to five call sites and the KD trainer for no functional gain.

---

## 2. Architecture decisions

### D-A — Backbone module structure

**Choice.** `EarlyFusionBackbone(pretrained, variant, in_channels=4)` in
`src/models/master/backbone.py`, replacing `DualConvNeXtBackbone` (VERIFIED `backbone.py:93-228`).

```python
class EarlyFusionBackbone(nn.Module):
    STAGE_CHANNELS = [96, 192, 384, 768]
    def __init__(self, pretrained=True, variant="tiny", in_channels=4):
        self.in_channels = in_channels
        self.stem   = _build_stem(in_channels)          # Conv2d(in_ch,96,k4,s4) + _LayerNorm2d(96)
        self.stages = _build_convnext_{variant}_body(pretrained)   # unchanged
        if pretrained:
            self._load_pretrained_stem()
    def forward(self, x):                                # x: (N, in_channels, H, W)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            f, out = self.stem(x.float()), []
            for stage in self.stages:
                f = stage(f); out.append(f)
        return out                                       # [S1..S4]
```

- `_build_stem` (VERIFIED `backbone.py:69-78`) and `_LayerNorm2d` (VERIFIED `backbone.py:81-90`) are
  reused verbatim; only the `in_channels` argument varies. `_build_convnext_tiny_body` /
  `_build_convnext_small_body` (VERIFIED `backbone.py:28-66`) are unchanged.
- The forced `autocast(enabled=False)` is **kept** (VERIFIED `backbone.py:160-168`). Its stated reason
  is ConvNeXt stage-4 fragility, which the single-stream move does not remove. Keeping it holds one
  more variable constant across V1/H-D/H-E. See §4 for the `amp` claim audit.
- `variant` validation against `SUPPORTED_VARIANTS` (VERIFIED `backbone.py:25, 121-124`) is kept.

**Alternatives rejected.** (a) Keep `DualConvNeXtBackbone` and add a 4-channel path — leaves the dead
two-stream code alive and contradicts W3. (b) A separate small NIR trunk — reintroduces the
two-branch design the maintainer decision removed.

### D-B — Stem inflation (implements D-1)

`_load_pretrained_stem()` replaces `_load_pretrained_rgb_stem` / `_load_pretrained_nir_stem`
(VERIFIED `backbone.py:192-228`, which already does `rgb_weight.mean(dim=1, keepdim=True)` — the
proven code the proposal cites).

```python
w = pretrained.features[0][0].weight            # (96, 3, 4, 4)
if self.in_channels == 4:
    new = torch.empty(96, 4, 4, 4)
    new[:, :3] = w * 0.75                       # 3/4 mean-preserving
    new[:, 3]  = w.mean(dim=1) * 0.75
elif self.in_channels == 3:
    new = w.clone()                             # verbatim ImageNet — the H-D control
state = {"0.weight": new,
         "0.bias":          pretrained.features[0][0].bias,
         "1.norm.weight":   pretrained.features[0][1].weight,
         "1.norm.bias":     pretrained.features[0][1].bias}
self.stem.load_state_dict(state, strict=False)
```

**Init asymmetry between H-D arms, stated openly.** The 4-channel arm is rescaled by 3/4; the
3-channel control is not. Each arm therefore receives its own best-available init, which is the
correct control for the question H-D asks ("does the extra modality help?"). It is *not* an
init-parity control. Rejected alternative: scale the 3-channel arm by 3/4 too — that would
deliberately mis-initialise the control and bias H-D toward CONFIRM.

`strict=False` is retained from the existing code so a `_LayerNorm2d` key-name drift degrades to a
silent no-op rather than a crash — that is a pre-existing weakness, so the H-A test asserts the
loaded values, not the return code.

### D-C — Checkpoint compatibility: hard break, loud failure

Every `state_dict` key under the old design changes: `backbone.rgb_stem.*` / `backbone.nir_stem.*` /
`backbone.shared_stages.*` → `backbone.stem.*` / `backbone.stages.*`; all `fusion.*` disappear;
`neck.fpn_rgb.*` / `neck.fpn_nir.*` / `neck.fusion_convs.*` → `neck.fpn.*`. Existing checkpoints are
unloadable — accepted risk, already recorded in `proposal.md` Risks.

**Choice: no migration shim, plus an explicit version tag.** `_save_checkpoint` (VERIFIED
`loop.py:557-564`) already writes `config: self.config.__dict__`. Add `"arch_version": 2` beside it.
Loaders (`evaluate_checkpoint.py:120-131`, `visualize_damage_predictions.py`) check it first and raise
`Architecture v1 checkpoint (dual-stream fusion) is not loadable by MasterModel v2 — see
openspec/changes/fusion-redesign` before `load_state_dict`.

**Rationale.** `load_state_dict(strict=True)` (VERIFIED `evaluate_checkpoint.py:131`) already fails on
a v1 checkpoint, but with a 200-line missing/unexpected-key dump that reads like a bug. The version
tag converts that into one actionable sentence for ~4 lines of code. A remapping shim is rejected:
the fusion weights have no target, so any shim would silently drop 9.42M trained parameters.

### D-D — Neck and P2 wiring (implements D-3)

`SingleFPN` is untouched (VERIFIED `neck.py:29-103`; it already returns `[P2,P3,P4,P5]` at
`neck.py:101-103`). `DualFPN` (VERIFIED `neck.py:106-173`) is deleted. New:

```python
STRIDE_TO_LEVEL = {4: 0, 8: 1, 16: 2, 32: 3}          # P2,P3,P4,P5

class FPNNeck(nn.Module):
    def __init__(self, in_channels=None, out_channels=256, strides=(4, 8, 16, 32)):
        self.fpn = SingleFPN(in_channels, out_channels)
        self.emit_levels = tuple(STRIDE_TO_LEVEL[s] for s in strides)   # KeyError on a bad stride
    def forward(self, features):
        pyramid = self.fpn(features)                   # always [P2,P3,P4,P5]
        return [pyramid[i] for i in self.emit_levels]
```

**Pyramid contract (new).** The emitted list is ordered finest-first and index `k` corresponds to
`head_strides[k]`. `head_strides` is the single source of truth: neck `emit_levels`, head level count,
loss `strides`, and decode `strides` are all derived from it. `strides` must be a strictly increasing
subsequence of `(4, 8, 16, 32)`; anything else raises at construction.

**Anchor generation at stride 4.** The brief states `_generate_anchors` assumes `[8,16,32]`. **That is
not accurate and I am correcting it with evidence.** `_generate_anchors` (VERIFIED `loss.py:289-323`)
is fully generic — it iterates `zip(feat_sizes, strides)` at `loss.py:312` and builds
`(arange(h)+0.5)*stride`, with no `[8,16,32]` anywhere. `_level_admissibility` (VERIFIED
`loss.py:244-262`) is likewise generic: `target_level = len(level_ranges)` at `loss.py:257`. The
`[8,16,32]` literals live in **defaults and call sites**, not in the anchor generator:
`loss.py:362`, `loop.py:68`, `loop.py:523`, `decode.py:40`, `master_model.py:94`,
`evaluate_checkpoint.py:149`, `visualize_damage_predictions.py:73` and `:347`. All eight are handled
in §5. No change to `_generate_anchors` is required. `loss.py:455-458` derives `img_h/img_w` from
`feat_sizes[0] * strides[0]`; with finest-first ordering (160×4 = 640) this stays correct — INFERRED
from the code, and covered by a unit test.

**Assigner level-range policy for 4 levels.** `assigner_level_ranges: [32, 64, 128]`. With
`_level_admissibility` semantics (VERIFIED `loss.py:251-262`), `max(w,h) < 32` → P2/stride-4,
`< 64` → P3, `< 128` → P4, else → P5.

**The one genuinely silent failure in this area.** `len(level_ranges)` must equal
`len(head_strides) - 1`. If `head_strides` becomes `[4,8,16,32]` while `assigner_level_ranges` stays
at its current `[64, 128]` (VERIFIED `config.py:102`, `training_mango.yaml:75`), then
`target_level` maxes out at 2 and **the stride-32 level never receives a single positive assignment**
— no exception, no warning, and mango (the large class) loses its natural level. This is the same
species of defect as D3. It is enforced in `TrainingConfig.__post_init__` (VERIFIED the hook exists at
`config.py:138-144`) with a RED test.

### D-E — W5: remove the head's duplicate stem computation

**VERIFIED.** `head.py:155` calls `head(feat)`, whose `DecoupledHead.forward` runs
`self.cls_stem(x)` and `self.reg_stem(x)` (VERIFIED `head.py:92-93`); then `head.py:164-165` call
`head.cls_stem(feat)` and `head.reg_stem(feat)` a second time. Both stems run twice per level, per
forward, and both copies are retained by autograd during training.

**Fix — change `DecoupledHead.forward` to return its stem features, and reuse them.**

```python
# head.py DecoupledHead.forward
def forward(self, x):
    cls_feat = self.cls_stem(x)
    reg_feat = self.reg_stem(x)
    return self.cls_pred(cls_feat), self.reg_pred(reg_feat), cls_feat, reg_feat

# head.py YOLODetectionHead.forward — replaces lines 155 and 164-165
cls_out, reg_out, cls_feat, reg_feat = head(feat)
distill_cls_feats.append(cls_feat)
distill_reg_feats.append(reg_feat)
```

**Alternative rejected:** call the stems once in `YOLODetectionHead` and pass features into
`DecoupledHead`. That splits one module's computation across two classes and breaks
`DecoupledHead`'s standalone use in `tests/models/student/test_head.py`. Changing the return arity is
a 4-line diff contained in one file; the only external consumer is `YOLODetectionHead.forward`
(VERIFIED: `head(feat)` appears once, at `head.py:155`).

The output dict is byte-identical in shape and semantics — `distill_cls`/`distill_reg` still carry the
post-`SiLU` stem features (VERIFIED `head.py:54-57, 61-64`). This is a pure memory/compute fix with no
numerical consequence, so it can be verified by a `torch.allclose` test against the current output.

### D-F — Training schedule (implements D-4)

`Trainer.fit` (VERIFIED `loop.py:107-140`) branches on `model_type` and hardcodes
`freeze_stages=4` / `freeze_stages=2, unfreeze_stages=[2,3]` for master.

**Choice.** Add `schedule: "two_phase" | "end_to_end"` to `TrainingConfig`, defaulting to
`end_to_end` for this model. The `end_to_end` branch runs one `_train_phase(phase=1,
epochs=config.epochs, lr=config.lr, freeze_stages=0)` and builds a **two-group** AdamW instead of
`loop.py:198-202`'s single group:

| group | params | lr |
|---|---|---|
| pretrained | `backbone.stages.*` | `lr * backbone_lr_mult` (0.1) |
| new | `backbone.stem.*`, `neck.*`, `head.*` | `lr` |

The stem is in the **new** group: it is 4-channel and out of ImageNet distribution by construction
(the D-1 argument applied consistently), even though 3 of its 4 channels are pretrained.

`freeze_backbone` / `unfreeze_backbone_stages` (VERIFIED `master_model.py:170-232`) lose the
`rgb_stem`/`nir_stem` asymmetry: `freeze_backbone(n)` freezes `stem` plus the first `n` stages;
`unfreeze_backbone_stages(idxs, unfreeze_stem=False)`. Retained because H-E/H-F and any future
ablation may need them, and `tests/models/master/test_freeze_policy.py` (VERIFIED, 55 lines) is
rewritten against the new names rather than deleted.

**Grad-norm logging (H-F).** Generalise `_rgb_stem_grad_norm` (VERIFIED `loop.py:414-427`) into
`_module_grad_norms() -> dict[str, float]` over `backbone.stem`, each `backbone.stages[i]`,
`neck.fpn`, and each `head.heads[i]`, aggregated per epoch into `LossHistory.extra_losses` and
TensorBoard. The end-of-run init-distance check is a separate script assertion, not a training-loop
concern.

### D-G — Silent batch-skipping (**BLOCKING PREREQUISITE for every training rung**)

Raised from "new scope to weigh" to a blocking prerequisite by the coordinator, on the strength of the
Round 2 measurements. **No training result is trustworthy until this lands.**

**VERIFIED `loop.py:391-396`:** a CUDA OOM inside the training step is caught, printed, the batch is
skipped, and the loop continues. **VERIFIED `loop.py:364-366`:** a NaN/Inf loss is caught, printed,
and the batch is skipped by the same `continue`. **VERIFIED `loop.py:404`:**
`n_batches = max(n_batches, 1)`.

**The concrete failure, with the Round 2 numbers.** The repository's configured `batch_size: 8`
(VERIFIED `training_mango.yaml:19`) needs ≈17 GB and OOMs at 13.54 GB on a 6.44 GB card (MEASURED).
Every batch therefore OOMs, every one is caught and skipped, `n_batches` reaches 0,
`max(n_batches, 1)` forces it to 1, and `total_loss / 1` is `0.0`. The run completes, prints
`total_loss: 0.0000`, saves `best_model.pt`, and produces a validation mAP from a model that never
received a single gradient. It fails silently and looks successful. This is the same species as D2 and
D3 — a run that provably did nothing while reporting that it did something.

**The NaN path is the same defect and matters for H-BF16.** If bf16 produced non-finite losses, the
guard at `loop.py:364-366` would skip those batches and the epoch mean would be computed over the
survivors — converting an H-BF16 **REFUTE into a CONFIRM**. The bar says "REFUTE on any NaN", so the
NaN counter is a precondition for H-BF16 being testable at all, not a nicety.

**Choice.** One counter mechanism covering both paths:

| Field | Meaning | Policy |
|---|---|---|
| `oom_skipped` | batches skipped by the OOM handler | epoch 1 tolerates skips (allocator warm-up); **raise** on any skip from epoch 2 |
| `nan_skipped` | batches skipped by the NaN/Inf guard | **always recorded**; H-BF16 REFUTEs on `nan_skipped > 0` |
| `steps_taken` | optimizer steps actually executed | **raise** if 0 in any epoch |

`n_batches = max(n_batches, 1)` is removed and replaced by an explicit `if n_batches == 0: raise
RuntimeError(...)`. Both counters go into `LossHistory.extra_losses`, TensorBoard, and the checkpoint.

**Alternative rejected:** keep the skip and just log louder. A log line does not stop the run from
writing a `best_model.pt` that a later reader will treat as a result. The failure has to be fatal.

### D-H — Hardware portability (**W9, new — implements Round 2 Q6**)

The experiment must be movable between the GTX 1660 SUPER (6.44 GB, Turing), an RTX 3080 (10 GB,
Ampere) and an RTX A5000 (24 GB, Ampere) without changing what is being measured.

**Choice: split the configuration in two, and make the split enforceable rather than merely
documented.**

```
configs/experiment/fusion.yaml     ← WHAT is measured. Byte-identical on every machine.
                                     architecture, head_strides, assigner_level_ranges, schedule,
                                     epochs, lr, backbone_lr_mult, loss, decode, seeds,
                                     effective_batch, split_manifest
configs/machines/{gtx1660s,rtx3080,rtx_a5000,cpu}.yaml
                                   ← HOW it runs. device, batch_size, num_workers,
                                     pin_memory, precision
```

Invoked as `--config configs/experiment/fusion.yaml --machine configs/machines/gtx1660s.yaml`.

Four enforcement mechanisms, because "must stay identical" without a mechanism is how D2 happened:

1. **Whitelist.** A machine profile may set only
   `{device, batch_size, num_workers, pin_memory, precision}`. Any other key raises at load. A machine
   profile therefore *cannot* alter the experiment, even by accident.
2. **`grad_accum_steps` is derived, never set.** `accum = effective_batch // batch_size`, and
   `effective_batch % batch_size != 0` raises. An operator cannot silently change the effective batch
   by editing a machine profile.
3. **`experiment_sha256`** — SHA-256 of the experiment YAML bytes, computed at load, printed at
   startup, and written into the checkpoint dict (beside `arch_version`) and into
   `stage_summary.json` via `run_artifacts.write_json` (VERIFIED `run_artifacts.py:72-83, 224`).
4. **Comparability rule, stated in the validation report:** two runs are comparable iff their
   `experiment_sha256` **and** `effective_batch` match. Anything else is a different experiment.

**Gradient accumulation is the load-bearing piece.** `effective_batch: 8` is the experimental
constant; physical `batch_size` is a memory knob.

| Machine | `batch_size` | `grad_accum_steps` | `effective_batch` |
|---|---:|---:|---:|
| GTX 1660 SUPER | 2 | 4 | 8 |
| RTX 3080 | 4 | 2 | 8 |
| RTX A5000 | 8 | 1 | 8 |

Implementation in `_train_epoch` (`loop.py:344-402`): scale the loss by `1/accum`, call `backward()`
every micro-batch, and run `clip_grad_norm_` + `optimizer.step()` + `zero_grad()` only every `accum`
micro-batches, flushing any remainder at epoch end. `grad_clip` (VERIFIED `loop.py:383`) must move
inside the accumulation boundary — clipping per micro-batch would clip a partial gradient and change
the optimisation, which is exactly the attribution leak accumulation exists to prevent.

**Alternative rejected: run each machine at its own batch size with an LR rescale** (linear or
sqrt scaling). It avoids the accumulation code, but H-D compares two arms against a pre-registered
bar denominated in `σ_d`; if the arms ran under different optimisations, a Δ of 2·`σ_d` is
uninterpretable. Accumulation costs ~25 lines and buys exact comparability.

**Device selection.** `loop.py:58` hardcodes `torch.device("cuda" if torch.cuda.is_available() else
"cpu")` (VERIFIED). Replace with `torch.device(config.device)` defaulting to `"auto"` (the current
expression). `scripts/evaluate_checkpoint.py` already exposes `--device`, so this closes the gap
between the two entrypoints rather than inventing a new convention.

**`pin_memory` is hardcoded `True`** at `train.py:176, 186` and `kd_train.py:147, 157` (VERIFIED) —
wire it to `config.pin_memory`. `build_dataloader` already accepts it (VERIFIED `dataset.py:630, 667`)
and already degrades `num_workers` to 0 when worker IPC fails (VERIFIED `dataset.py:673-692`), so the
CPU profile needs no special case.

### D-I — Precision as an enum, and what the backbone's forced fp32 still means

**Choice.** Replace `amp: bool = True` (VERIFIED `config.py:116`) with
`precision: str = "fp32"`, accepting `fp32 | fp16 | bf16`, validated in `__post_init__`.
`amp` is removed, not deprecated — a silently-defaulting boolean that flips a run to fp16 has no safe
migration path, and the repo has exactly one config that sets it (`training_mango.yaml:42`).

A single `src/training/precision.py` owns the policy:

```python
def autocast_ctx(device_type, precision):        # used by loop.py:353 and :466
    if precision == "fp32":  return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device_type, dtype=dtype)

def make_scaler(precision):                      # replaces loop.py:75
    return GradScaler("cuda", enabled=(precision == "fp16"))   # bf16 needs no scaler
```

`bf16` additionally requires `torch.cuda.is_bf16_supported()`; on the Turing GTX 1660 SUPER this is
`False` and the config load raises with a message naming the card. Fail loud, do not silently
downgrade to fp32 — a silent downgrade would make an H-BF16 CONFIRM meaningless.

**Restating the backbone's forced-fp32 block after the redesign.** `backbone.py:160-168` (VERIFIED)
disables autocast, and its comment attributes this to *"ConvNeXt stage 4 is numerically fragile under
CUDA autocast **in this dual-stream setup**"*. After W1 there is no dual-stream setup, so that
specific justification is gone. What survives, and what does not:

| Claim | Status after the redesign |
|---|---|
| "the dual-stream setup is fragile" | **Void** — there is no second stream |
| "ConvNeXt stage 4 is fragile under autocast" | **Unverified but plausible for fp16.** Stage 4 is the same module; the two-stream framing was probably incidental. Never measured, in this repo or in the comment |
| "`YOLOv8Loss` NaNs under fp16" (`training_mango.yaml:42`) | **Independent of the backbone** and unaffected by W1. Also never measured — it is an author's comment |
| bf16 overflow risk | **Different in kind.** bf16 carries fp32's exponent range, so the overflow mechanism the fp16 comments describe does not apply |

**Choice: keep a stage-4 guard, but scope it to fp16 only.** The new backbone wraps its stages in
`autocast(enabled=False)` **when `precision == "fp16"`**, and is transparent under `fp32` (no-op) and
`bf16` (the whole point of H-BF16 is to let bf16 reach the stages). This preserves the existing hedge
where it was claimed to matter, and removes it where the claimed mechanism does not exist.
The alternative — deleting the block outright — was rejected because it would change two variables at
once and would let an H-BF16 REFUTE be blamed on either.

---

## 3. Data flow — the level contract

```
head_strides = [4, 8, 16, 32]         ← configs/experiment/fusion.yaml (single source of truth)
   │
   ├─→ FPNNeck.emit_levels   = [0,1,2,3]      → pyramid [P2,P3,P4,P5]
   ├─→ YOLODetectionHead.strides              → 4 DecoupledHeads, preds finest-first
   ├─→ YOLOv8Loss.strides                     → _generate_anchors → 34,000 anchors @640
   ├─→ decode_detections(strides=...)         → boxes in pixel space
   └─→ assigner_level_ranges (len == 3)       → [32, 64, 128]      (invariant-checked)
```

Anchor count at 640: 160² + 80² + 40² + 20² = 25,600 + 6,400 + 1,600 + 400 = **34,000**, against
8,400 today (INFERRED, arithmetic from `loss.py:312-319`). The assigner loops per-batch-item and
per-GT (VERIFIED `loss.py:136-177`) with a `box_iou(pred_xyxy, gt_xyxy)` of shape
`(34,000, N_gt)` — a 4× increase in assignment cost that is real but small next to the backbone.

---

## 4. Memory budget

Parameter counts are exact, hand-counted from module definitions. VRAM and wall clock for the
*current* model are **MEASURED** (Round 2); projections for the *redesign* are **INFERRED** but now
calibrated against that measurement. `image_size=640`, fp32, AdamW throughout.
**V0 supersedes every projected number here.**

### Parameters (exact, hand-counted from source)

| Module | Now | After | Δ |
|---|---:|---:|---:|
| Backbone stages (`shared_stages`) | 27,813,696 | 27,813,696 | 0 |
| Stems (`rgb_stem` 4,896 + `nir_stem` 1,824) | 6,720 | 6,432 (4-ch stem) | −288 |
| `StageAttentionFusion` / `CrossModalFusion` | 9,421,920 | 0 | −9,421,920 |
| Neck (`fpn_rgb` + `fpn_nir` + `fusion_convs`) | 5,855,488 | 2,729,984 | −3,125,504 |
| Head | 3,545,106 (3×) | 4,726,808 (4×) | **+1,181,702** |
| **Total** | **≈46.64M** | **≈35.28M** | **≈ −11.37M** |

Derivations: `SingleFPN` = laterals (24,576+256)+(49,152+256)+(98,304+256)+(196,608+256) = 369,664,
plus 4 output convs × (256·256·9+256) = 2,360,320 → **2,729,984** (from `neck.py:58-67`).
`fusion_convs` = 3 × (131,072+256+512 BN) = **395,520** (from `neck.py:135-143`).
One `DecoupledHead` = 2×(589,824+256) + (512+2) + (1024+4) = **1,181,702** (from `head.py:54-65`).

**Correction to the proposal.** D-3 and D-6 state the P2 head costs "≈2.4M parameters". By the
arithmetic above it is **1.18M** — 2.4M is the cost of *two* heads. The net reduction is therefore
≈11.4M, not ≈10M. Direction and conclusion unchanged; D-6's "do not reinvest" stands.

Optimizer footprint (params+grads+2 Adam moments = 16 B/param): 0.75 GB → 0.56 GB. Activations
dominate, so this is not where the budget is decided.

### Measured baseline (Round 2 — supersedes the first draft's estimate)

MEASURED, GTX 1660 SUPER 6.44 GB, current master, fp32, synthetic batches, forward+backward+step:

| batch | peak VRAM | step time | throughput |
|---:|---:|---:|---:|
| 1 | 2.32 GB | 430 ms | 2.32 img/s |
| 2 | 4.42 GB | 888 ms | 2.25 img/s |
| 4 | 8.64 GB | — | exceeds VRAM; the WSL driver spills to host RAM |
| 8 | OOM at 13.54 GB | — | — |

Fit: **≈2.10 GB per sample + ≈0.22 GB intercept**. Extrapolated to batch 8: ≈17 GB.
**Master epoch (148 images) ≈ 65 s; 80 epochs ≈ 87 minutes.** Throughput is flat between batch 1 and
2 (2.32 vs 2.25 img/s), so **the card is already saturated at batch 1** — a larger batch buys no
wall clock, only OOM risk. Any "4-6 minutes per epoch" figure came from the cloud instance and from
the much smaller student model; it does not apply here and is not carried forward anywhere in this
document.

### Calibrating the activation model against the measurement

The first draft modelled ≈`12·C·H·W` retained elements per ConvNeXt block (dwconv input, LN input,
Linear1 input, GELU input at 4C, Linear2 input at 4C, gamma-scale input). At batch 8, fp32, 640²,
depths `[3,3,9,3]`, that gives per stream: stage 1 ≈2.83 GB, stage 2 ≈1.42, stage 3 ≈2.12,
stage 4 ≈0.35 → ≈6.7 GB; plus neck ≈1.2 GB and head ≈0.82 GB for the current dual/3-level config.

That is **≈1.93 GB per sample against a MEASURED 2.10 — within 9%**, and the predicted ≈16 GB at
batch 8 against ≈17 GB extrapolated. The model is therefore calibrated, and the redesign's
projections below apply a **×1.09 correction factor** to it. This is the strongest evidence available
that the per-component breakdown is sound; it remains INFERRED for the redesign, which has never been
instantiated.

### Projected peaks for the redesign (INFERRED, calibrated)

Per-sample slope, calibrated; intercept held at the measured ≈0.2 GB (the weakest part of the model —
it is a two-point fit and the optimizer state moves with the parameter count):

| Config | GB/sample | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|---:|
| Current master (dual, 3 levels, pre-W5) | **2.10 (MEASURED)** | **4.42** | **8.64** | **OOM ≈17** |
| New, 4 levels, **pre**-W5 | 1.45 | 3.1 | 6.0 | 11.8 |
| New, 3 levels, post-W5 | 1.04 | 2.3 | 4.4 | 8.5 |
| **New, 4 levels, post-W5 (the target config)** | **1.22** | **≈2.6** | **≈5.1** | ≈10.0 |

**Conclusions, and how they changed.** The first draft's headline — "the design does not fit 6.4 GB at
`batch_size=8`" — is confirmed by measurement, and it is now clear that the *current* master does not
fit either: it OOMs at batch 4 in practice. What changes is the framing. At the resolved
`batch_size: 2` the target config projects **≈2.6 GB against 6.44 GB available**, which is not tight
at all. The redesign roughly halves per-sample cost (2.10 → 1.22 GB), and **W5 alone accounts for
≈0.23 GB/sample** — 19% of the target config's slope, which is why it stays first on the ladder.

### Batch policy and the OOM ladder (Q5 **resolved**)

**`batch_size: 2` locally, `effective_batch: 8` via gradient accumulation (D-H).** The wall-clock
cost the maintainer accepted is, by the saturation measurement, approximately **zero**: 2.25 img/s at
batch 2 versus 2.32 at batch 1, and the card is already saturated, so the batch-4 headroom the table
above suggests would not buy throughput even if it fits. The decision is well supported by its own
measurement, not merely accepted.

The ladder is retained as a **contingency** should V0 contradict the projection, with rungs 1-3
verbatim from D-7 and rungs 4-5 appended:

| # | Rung | Status |
|---|---|---|
| 1 | Remove the duplicate head stem compute (W5) | **Already taken** — it is in scope as a defect fix regardless of memory |
| 2 | P2 head at 128 channels | Contingency; costs H-E attribution (asymmetric with P3-P5) |
| 3 | `batch_size` → 2 | **Already taken** (Q5), with `effective_batch` held at 8 by accumulation |
| 4 | `torch.utils.checkpoint` on `backbone.stages` | Contingency; ≈30-40% slower, **numerics unchanged** |
| 5 | `batch_size` 2 → 1, `grad_accum_steps` 4 → 8 | Contingency; `effective_batch` still 8 |

Q5's original tension — whether rung 4 should precede rung 3 on attribution grounds — is **moot**:
gradient accumulation makes rung 3 attribution-neutral, because `effective_batch` no longer moves
when `batch_size` does. Rungs 1 and 3 together take the projection from ≈17 GB to ≈2.6 GB, which is
why the remaining rungs are contingencies rather than a plan.

**Not on the ladder, and why.** `image_size` — reducing it changes the pixel scale of a 30 px lesion
and invalidates the entire D-3 rationale for P2. `precision` — see D-I and H-BF16: bf16 is a
*hypothesis with a pre-registered test*, not a fallback to reach for mid-experiment, and it is
unavailable on the Turing card anyway. Whatever is taken must be frozen before V1 and held identical
across V1/H-D/H-E (D-7), and `experiment_sha256` + `effective_batch` (D-H) make a violation detectable
after the fact.

### `amp` claim audit

Three facts, all VERIFIED, **none of them a measurement**:

1. `training_mango.yaml:42` — `amp: false  # FP16 causes NaN in YOLOv8Loss — using FP32`. An author's
   comment, not a recorded experiment. `loop.py:364-366` does contain a NaN/Inf guard, consistent
   with NaNs having been observed, but not proof of the fp16 attribution.
2. `backbone.py:160-168` — forces `autocast(enabled=False)`, attributing it to ConvNeXt stage-4
   fragility **in the dual-stream setup**, not to the loss. See D-I for what survives W1.
3. `config.py:116` defaults `amp: True`, so **any** config omitting the key silently enables fp16.

The proposal's phrasing ("`YOLOv8Loss` NaNs under fp16 **and** the backbone forces autocast off for
exactly that reason") conflates (1) and (2), which have different stated causes. D-I resolves all
three: `amp` is replaced by an explicit `precision` enum defaulting to `fp32`, so fact (3) becomes
impossible by construction; and H-BF16 converts fact (1) from folklore into a tested claim on the one
precision where the described overflow mechanism does not apply.

---

## 5. The evaluation-path hazard, and how to stop it recurring

### What is actually silent, and what is not

The brief states both scripts "would silently misread a 4-level checkpoint". Verified, that is only
partly true, and the distinction changes the fix:

| Site | Behaviour with a 4-level model | Loud? |
|---|---|---|
| `visualize_damage_predictions.py:73` → `:347` | `decode_detections` raises `ValueError` on the length guard at `decode.py:77-81` | **Loud** |
| `loop.py:523` (`strides=(8,16,32)` in `_decode_predictions_static`) | same `ValueError` | **Loud** |
| `evaluate_checkpoint.py:149` (`YOLOv8Loss(strides=[8,16,32])`) | `zip` at `loss.py:312` truncates to 3 levels → 8,400 anchors vs 34,000 preds → broadcast `RuntimeError` at `loss.py:568` | Loud, but the message names neither strides nor levels |
| `evaluate_checkpoint.py:110-116` (`MasterModel(...)` without `head_strides`) | builds 3 heads; `load_state_dict(strict=True)` at `:131` rejects `head.heads.3.*` | Loud, unreadable |
| **`assigner_level_ranges` shorter than `len(strides)-1`** | the coarsest level receives **zero** positives | **SILENT** |
| **`ProjectionLayers.forward` zip at `distill_projections.py:85`** | 3 projections zip against a 4-level teacher pyramid → P2/P3/P4 distilled into student P3/P4/P5 | **SILENT** |

The two silent rows are the real hazard, and **neither is in the proposal's scope list.** The
`ProjectionLayers` one contradicts the proposal's "No projection shapes change" (W7): that holds for
`backbone_projections` (`[384, 768]`, VERIFIED `distill_projections.py:106`) but **not** for
`fpn_projections` and `head_projections`, which are 3-entry presets (VERIFIED
`distill_projections.py:94, 118`) that will silently truncate against a 4-level teacher.

### Fix — one resolver, no literals, two guards

1. **New `src/training/strides.py`**: `STRIDE_TO_LEVEL`, `resolve_head_strides(config)`,
   `resolve_from_checkpoint(ckpt)` (reads `ckpt["config"]["head_strides"]`, which
   `_save_checkpoint` already persists via `config.__dict__` — VERIFIED `loop.py:563`, so strides
   travel with every checkpoint at zero extra cost), and `validate_strides(strides, level_ranges)`.
2. **Delete every literal** at the eight sites listed in §D-D. Each call site takes its strides from
   the config or from the checkpoint.
3. **Remove the defaults** so the parameter becomes required: `YOLOv8Loss(strides=...)`
   (`loss.py:352, 362`) and `decode_detections(strides=...)` (`decode.py:40`). A forgotten argument
   then becomes a `TypeError` at the call site instead of a wrong-but-plausible number. This is the
   mechanism that makes the class of defect non-recurrable, rather than fixing two instances of it.
4. **Repo guard test**: a test that scans `src/` and `scripts/` for the literals `[8, 16, 32]` and
   `(8, 16, 32)` and fails. Cheap, and it catches the next reintroduction rather than the current one.
5. **Config invariant** in `TrainingConfig.__post_init__` (`config.py:138`):
   `len(assigner_level_ranges) == len(head_strides) - 1`, and `head_strides` a strictly increasing
   subsequence of `[4,8,16,32]`.
6. **`ProjectionLayers.forward`** (`distill_projections.py:71-86`): replace the bare `zip` with an
   explicit length assertion, and have `kd_trainer.py` slice the teacher pyramid/head features to the
   student's own strides (`[8,16,32]`) by index rather than by truncation.
7. **`--override` list coercion** in `train.py:88-99` — **blocking for H-E**. VERIFIED: it coerces
   `bool`/`int`/`float` but **not `list`**, so `--override head_strides=[4,8,16,32]` would assign the
   *string* `"[4,8,16,32]"`, and `len()` of it is 12 — a 12-level pyramid, which would fail deep
   inside the neck rather than at the flag. `evaluate_checkpoint.py:103-106` already implements the
   list branch (`json.loads` when the value starts with `[`, else comma-split floats) — port that
   exact code so the two entrypoints cannot drift. Without it H-E's own ablation flag does not work,
   and neither does `--override assigner_level_ranges=[64,128]`, which H-E must change in lockstep
   (§D-D's silent-failure invariant).

---

## 6. Validation mechanics

Entrypoint for every training rung (VERIFIED `train.py:33-75, 78-129`), with the D-H config split:

```
./.venv/bin/python -m src.training.train \
    --config  configs/experiment/fusion.yaml \
    --machine configs/machines/gtx1660s.yaml \
    --model master --override seed=<S> <rung-specific overrides>
```

`configs/experiment/fusion.yaml` is a **new** file (the fusion architecture's schedule is not the
two-phase schedule `training_mango.yaml` encodes at lines 18-23, and overwriting that file would break
the damage-map-audit baselines it reproduces). It carries: `head_strides: [4, 8, 16, 32]`,
`assigner_level_ranges: [32, 64, 128]`, `schedule: end_to_end`, `epochs: 100`, `lr: 0.001`,
`backbone_lr_mult: 0.1`, **`effective_batch: 8`**, `patience: 20`,
`split_manifest: data/annotations/yolo/splits.json`, and — carried unchanged from
`training_mango.yaml` so the guardrail comparison stays valid — `class_weights: [0.5, 1.5]`,
`conf_threshold: 0.25`, `nms_iou_threshold: 0.5`, `nms_enabled: true`, `decode_per_class: true`,
`assigner_center_radius: 0.0`. It contains **no** `batch_size`, `device`, `num_workers`,
`pin_memory` or `precision` — those are machine-profile keys, and the whitelist in D-H rejects the
experiment file if it sets them.

`configs/machines/gtx1660s.yaml`: `device: cuda`, `batch_size: 2`, `num_workers: 2`,
`pin_memory: true`, `precision: fp32`. The 3080 and A5000 profiles differ only in `batch_size`
(4 / 8) and, for H-BF16 only, `precision: bf16`.

**Seeds: `42, 1337, 2024`** for every 3-seed rung, fixed here so no rung can pick a favourable set
after the fact. `train.py:126-129` seeds `torch` and CUDA only; dataloader-worker and cuDNN
nondeterminism remain, and that is intentional — `σ_d` is defined to absorb exactly that.

| Rung | Command / override | Runs | Readout |
|---|---|---|---|
| **D-G** | test suite: an all-OOM epoch raises; an all-NaN epoch raises; `steps_taken == 0` raises | 0 training | **BLOCKING — no rung below may run until green** |
| V0 | `scripts/measure_vram.py --config configs/experiment/fusion.yaml --machine <profile> --variant {master_v1,new_3lvl_preW5,new_3lvl,new_4lvl}` (new script; one fwd+bwd+step, `reset_peak_memory_stats()` then `max_memory_allocated()`, batches 1/2/4) | 4×3 | peak GB per config; reconciles §4's projection against measurement |
| H-A | `scripts/stem_init_audit.py --init {inflation,zero,copy}` — per-stage activation mean/std over the train split vs an RGB-only pretrained reference | 0 training | within 2× reference across-image std at every stage |
| H-B | test: NIR-only input perturbation, count params with non-zero grad | 0 training | ≥ 27M (vs 1,824) |
| H-C | test: no `adaptive_avg_pool2d` in the backbone→head module tree; `min(head_strides) ≤ 8` | 0 training | static assert |
| V1 | `--override seed={42,1337,2024}` | 3 | mean damage AP50, `σ_d`, `σ_L5`, **measured s/epoch** |
| H-D (a) | as V1 (4-channel) | reuse V1's 3 | — |
| H-D (b) | `--override in_channels=3 seed={42,1337,2024}` | 3 | AP50(4ch) − AP50(3ch) ≥ 2·σ_d |
| H-E | `--override head_strides=[8,16,32] assigner_level_ranges=[64,128] seed={42,1337,2024}` | 3 | Δ vs V1 ≥ 2·σ_d |
| H-F | read `oom_skipped`, `nan_skipped`, `steps_taken` and per-module grad norms from every run above | 0 extra | no module bit-exactly at init |
| **H-BF16** | `--machine configs/machines/rtx_a5000.yaml --override precision=bf16 epochs=5 seed=42` | 1, **Ampere only** | see below |

**Total: 9 training runs** for the main ladder (V1's three are reused as H-D arm (a) — H-D's
authorised "6 runs, 3 seeds per arm" is satisfied by V1×3 + control×3, so the maintainer's Q4 spend is
met exactly, not exceeded), plus one 5-epoch H-BF16 probe that runs only if Ampere hardware is
available and gates nothing.

### H-BF16 — test design

**Hypothesis.** The `amp: false` restriction is an fp16 exponent-overflow artefact, so bf16 — which
carries fp32's exponent range — trains without non-finite losses.

**Comparison.** The first 5 epochs of the fp32 V1 run at `seed=42`, versus a bf16 run at `seed=42`,
on the **same machine** (an A5000) with the same `experiment_sha256` and the same `effective_batch: 8`
— only `precision` differs. Running the fp32 reference on the 1660 SUPER and the bf16 arm on the
A5000 would confound precision with hardware, so both arms are Ampere.

**Pre-registered bars (from Round 2, with the one derivation the proposal left open):**

- **CONFIRM:** all 5 epochs complete with `nan_skipped == 0` **and** the epoch-5 mean training loss is
  within `1·σ_L5` of the fp32 run's epoch-5 mean loss.
- **REFUTE:** any non-finite loss at any step, i.e. `nan_skipped > 0`.

`σ_L5` — the between-seed standard deviation of the epoch-5 mean loss — is **not** an extra cost:
V1 already runs three seeds, so its three epoch-5 losses give `σ_L5` for free. Without D-G's
`nan_skipped` counter this bar is untestable, because `loop.py:364-366` would silently drop the
non-finite batches and the epoch mean would be computed over the survivors — turning a REFUTE into a
CONFIRM. That dependency is why D-G is listed as blocking for H-BF16 as well as for the main ladder.

**Gating.** H-BF16 is a prerequisite for nothing. It is an optimisation probe: a CONFIRM buys an
estimated further 1.5-2× on Ampere (Round 2, an estimate from specification ratios, not a
measurement), a REFUTE simply keeps fp32. No result of H-BF16 changes any bar in V1/H-D/H-E, and no
V1/H-D/H-E run may switch precision mid-ladder.

### Wall clock — measured, not assumed

| Quantity | Value | Basis |
|---|---|---|
| Current master, 148 images | **≈65 s/epoch** (2.28 img/s) | MEASURED, Round 2 |
| Current master, 80 epochs | **≈87 min** | MEASURED |
| Redesign, per epoch | **0.55-0.95× the above ⇒ ≈36-62 s** | INFERRED — one backbone stream instead of two, fusion and its four 400×400 attention stages gone, one FPN instead of two, W5 halving head stem compute; against that, P2 adds a 160×160 head |
| Main ladder, 9 runs × ≤100 epochs, 1660 SUPER | **≈10-16 h** | INFERRED at 65 s/epoch (conservative: assumes no speedup); `patience: 20` early stopping should land at the low end |
| Same ladder on an A5000 | ≈3-5 h | ESTIMATED from Round 2's 3-5× band (specification ratios, not measured) |

Gradient accumulation does **not** change epoch wall clock: the same 148 images are processed, only
the number of optimizer steps drops by `grad_accum_steps`. Batch 2 versus batch 1 costs nothing
either — the card is saturated (MEASURED, 2.32 vs 2.25 img/s).

The first V1 seed publishes measured `s/epoch` from `loop.py:219, 227`
(`elapsed = time.time() - t0`, already recorded per epoch), and the schedule is restated from that
number **before** H-D and H-E are scheduled. No cloud figure — neither the 6.6-6.9 s/epoch in
`proposal.md` nor any "4-6 minutes" estimate — carries over to this hardware. If the measured cost
puts 9 runs beyond the available window, the options are (a) move to the A5000 with the same
`experiment_sha256` and `effective_batch`, which D-H makes safe, or (b) a maintainer decision to cut
scope. Quietly dropping seeds is not an option: every bar in this document is denominated in `σ_d`,
and a 1-seed rung cannot produce one.

**Guardrail on every training rung:** mango AP50 ≥ 0.774.

### Testing strategy

| Layer | What | How |
|---|---|---|
| Unit | Stem inflation values: `W[:, :3] == W_imagenet*0.75`, `W[:, 3] == mean·0.75`; bias and LN copied verbatim | `torch.allclose` against a freshly loaded `convnext_tiny` |
| Unit | `FPNNeck` emits exactly the levels named by `strides`, finest-first, at 256 ch | shape assertions for `[4,8,16,32]` and `[8,16,32]` |
| Unit | W5: post-fix head output `torch.allclose` with the pre-fix output; `cls_stem` called exactly once per level | forward-hook call counter |
| Unit | `_generate_anchors` at `[4,8,16,32]`, 640² → 34,000 anchors, per-level counts `[25600,6400,1600,400]` | direct call |
| Unit | `_level_admissibility` with `[32,64,128]` maps a 30 px box to level 0 | direct call |
| Unit | Config invariant: `len(level_ranges) != len(strides)-1` raises | `pytest.raises` |
| Unit | `--override head_strides=[4,8,16,32]` yields a `list[int]` | `train.py` override parser |
| Unit | `precision` accepts only `fp32\|fp16\|bf16`; the default is `fp32`; `amp` is gone | replaces the old `amp`-default regression |
| Unit | `bf16` on a non-bf16 device raises and names the card | monkeypatched `is_bf16_supported` |
| Unit | `ProjectionLayers.forward` raises on a length mismatch | `pytest.raises` |
| Unit (D-H) | machine profile setting a non-whitelisted key raises; `effective_batch % batch_size != 0` raises; `grad_accum_steps` is derived, never read from a file | `pytest.raises` |
| Unit (D-H) | `experiment_sha256` is stable across machine profiles and changes when the experiment file changes by one byte | two loads |
| Unit (D-G) | **blocking**: an epoch where every batch OOMs raises; where every batch is NaN raises; `steps_taken == 0` raises; `n_batches = max(n_batches, 1)` is gone | injected fault in a fake loader |
| Integration (D-H) | `batch_size=2, accum=4` and `batch_size=8, accum=1` produce `torch.allclose` parameter updates after one effective step on a fixed seed | the accumulation correctness proof |
| Integration (D-H) | `grad_clip` is applied once per effective step, not per micro-batch | call counter on `clip_grad_norm_` |
| Repo guard | no `[8, 16, 32]` / `(8, 16, 32)` literal in `src/` or `scripts/` | source scan |
| Integration | `MasterModel(head_strides=[4,8,16,32])` fwd+bwd on CPU at 128²; output dict has exactly the 7 keys; every emitted level receives a non-`None` grad | direct — this is D3's regression test |
| Integration | H-B gradient audit; H-C static assertions | as above |
| Integration | v1 checkpoint load raises the `arch_version` error | synthetic v1 dict |
| E2E | `tests/test_final_training_pipeline.py` and `tests/test_training_loop.py` updated to the 4-level model | existing harness |

Baseline to hold: 134 tests green via `./.venv/bin/python -m pytest` (~64 s). NOT RUN in this session
— no shell access.

---

## 7. File changes

| File | Action | Notes |
|---|---|---|
| `src/models/master/backbone.py` | Modify | `DualConvNeXtBackbone` → `EarlyFusionBackbone`; `_load_pretrained_stem` |
| `src/models/master/fusion.py` | **Delete** | whole file |
| `src/models/master/neck.py` | Modify | delete `DualFPN` (`neck.py:106-173`); add `FPNNeck`; `SingleFPN` untouched; module docstring rewritten (`neck.py:1-22` describes the deleted design) |
| `src/models/master/head.py` | Modify | W5 only |
| `src/models/master/master_model.py` | Modify | rewire; 7-key dict; `in_channels`; freeze helpers; docstring `master_model.py:1-32` |
| `src/models/master/__init__.py` | Modify | VERIFIED `__init__.py:1-25` re-exports all three deleted classes — a stale import here breaks every consumer |
| `src/models/master/distill_projections.py` | Modify | length assertion in `forward` (`:71-86`) |
| `src/models/master/test_arch.py` | Modify | VERIFIED `:10-12, 28-41` import and instantiate all three deleted classes |
| `src/models/master/sanity_check.py`, `README.md` | Modify | reference the deleted architecture |
| `scripts/visualize_attention.py`, `scripts/attention_comparison.py` | **Delete** | there is no attention to visualise after W3 |
| `src/training/strides.py` | **Create** | stride resolver + validation |
| `src/training/precision.py` | **Create** (W9) | `autocast_ctx`, `make_scaler`, bf16 capability check |
| `src/training/machine.py` | **Create** (W9) | machine-profile loader, key whitelist, `experiment_sha256`, derived `grad_accum_steps` |
| `src/training/config.py` | Modify | `head_strides`, `backbone_lr_mult`, `schedule`, `in_channels`; **W9**: `device`, `pin_memory`, `effective_batch`, `precision` replacing `amp` (`:116`); `__post_init__` invariants |
| `src/training/loop.py` | Modify | `end_to_end` schedule, param groups, per-module grad norms, **D-G counters (`:364-366`, `:391-396`, `:404`)**, strides from config (`:68`, `:523`), **W9**: device (`:58`), accumulation (`:344-402`), `autocast`/`GradScaler` via `precision` (`:23`, `:75`, `:353`, `:466`) |
| `src/training/loss.py`, `src/training/decode.py` | Modify | make `strides` required (`loss.py:352,362`; `decode.py:40`) |
| `src/training/train.py` | Modify | list coercion in `--override` (`:88-99`); **W9**: `--machine` flag, `pin_memory` from config (`:176`, `:186`) |
| `src/training/kd_train.py` | Modify (W9) | `pin_memory` (`:147`, `:157`), `precision` — kept consistent so KD does not diverge |
| `src/training/kd_trainer.py` | Modify | `distill_backbone` (`:151`); explicit level slicing; `precision` |
| `src/training/run_artifacts.py` | Modify (W9) | record `experiment_sha256`, `effective_batch`, `precision`, `device` in `stage_summary.json` (`:224`) |
| `configs/experiment/fusion.yaml` | **Create** | the experiment — byte-identical across machines |
| `configs/machines/{gtx1660s,rtx3080,rtx_a5000,cpu}.yaml` | **Create** (W9) | per-machine `device`/`batch_size`/`num_workers`/`pin_memory`/`precision` |
| `configs/training_mango.yaml` | Modify | `amp:` → `precision:` (`:42`), `batch_size: 8` → 2 (it does not fit — MEASURED), comment pointing at the new config; **no schedule change** |
| `configs/training_student.yaml`, other configs setting `amp` | Modify (W9) | `amp:` → `precision:` — mechanical, but `amp` is removed rather than deprecated |
| `scripts/evaluate_checkpoint.py`, `scripts/visualize_damage_predictions.py` | Modify | strides from checkpoint; `arch_version` check |
| `scripts/measure_vram.py`, `scripts/stem_init_audit.py` | **Create** | V0 and H-A |
| `tests/**` | Modify/Create | per §6 |
| `openspec/specs/{testing-teacher-arch,training-loop,kd-training}` | Modify | delta specs; `training-loop` gains the D-G, accumulation, precision and portability requirements |

### Authored-line estimate (additions + deletions)

| Workstream | Est. | Δ vs first draft |
|---|---:|---|
| W1 backbone | 320 | — |
| W2 neck | 155 | — |
| W3 fusion deletion + `__init__`/`test_arch`/`sanity_check`/README/attention scripts | 470 | — |
| W4 master_model | 240 | — |
| W5 head | 15 | — |
| W6 loop (schedule, param groups, grad norms) | 200 | — |
| W7 KD + projections | 45 | — |
| W8a config + strides module + call sites + `train.py` | 160 | — |
| W8b scripts (eval, visualize, measure_vram, stem_init_audit) | 220 | — |
| **D-G blocking counters** (`oom_skipped`, `nan_skipped`, `steps_taken`, remove `max(n,1)`) | **45** | **new** |
| **W9a precision enum** (`precision.py`, config swap, 4 call sites, KD, every config file's `amp:`) | **150** | **new** |
| **W9b machine profiles** (`machine.py`, whitelist, `experiment_sha256`, `run_artifacts`, `--machine`, `pin_memory`, 4 profile files + the experiment file) | **255** | **new** |
| **W9c gradient accumulation + device selection** | **65** | **new** |
| **H-BF16 harness + report section** | **30** | **new** |
| Tests (original scope) | 550 | — |
| **Tests (W9 + D-G)** — accumulation equivalence, clip-once, whitelist, sha256 stability, precision enum, bf16 gate, D-G fault injection | **260** | **new** |
| Specs | 300 | +100 — `training-loop` gains D-G, accumulation, precision and portability requirements |
| **Total** | **≈3,480** | **+905** |

**Stated honestly, as instructed.** This is ≈4.4× the 800-line budget and ≈2.3× the proposal's
≈1,485 estimate. My previous band was **2,500-3,200**; the Round 2 workstream adds ≈905 lines, so the
restated band is **3,200-4,200**.

Where the growth came from, so the reviewer can audit it: the proposal's ≈1,485 omitted
`__init__.py`, `test_arch.py`, `sanity_check.py`, the two attention scripts, the strides module,
`train.py`, `distill_projections.py` and the two measurement scripts (all VERIFIED as touched);
Round 2 then added the hardware-portability workstream and raised D-G into scope. The previous change
ran 2,311 against an estimated 1,165 (2.0×); this estimate is bottom-up from an enumerated file list
rather than top-down, so I expect the multiplier to be smaller, but I am not claiming precision I do
not have.

**This is now large enough that I am obliged to say so plainly**, without reopening a settled
decision: the `size:exception` and the single-PR delivery are the maintainer's Q2 decision and stand.
If the reviewer would prefer to split, the natural seam is **W9 + D-G as a separate preparatory PR
(~1,105 lines)** — it is self-contained, touches only the training loop and configuration, has its own
tests, is independently revertible, and is a **blocking prerequisite** for every training rung anyway,
so it would land first regardless. That would leave the architecture change at ~2,375. Offered as
information for the Q2 holder, not as a re-split.

---

## 8. Migration and coexistence

**Branch point — now READY.** `damage-map-audit` has **merged to main** (PRs #10 and #11) as of
commit **`9e43aa6`**. `scripts/visualize_damage_predictions.py`, `src/training/decode.py`,
`scripts/evaluate_checkpoint.py` and the reconciled splits (148/18/21,
`data/annotations/yolo/splits.json`) are all on `main`. Maintainer Decision Q3's precondition is
satisfied and `sdd-apply` is unblocked. Branch from `9e43aa6`.

This supersedes the first draft's note that `scripts/visualize_damage_predictions.py` was untracked;
it was untracked in the working tree at design time and has since landed.

**File-level overlap with what has already landed:**

| File | Landed there | Overlap here | Conflict risk |
|---|---|---|---|
| `configs/training_mango.yaml` | decode block (`:62-69`), assigner block (`:71-76`), `split_manifest` (`:60`) | `amp:` → `precision:` (`:42`), `batch_size` 8 → 2, pointer comment. The experiment lives in the new `configs/experiment/fusion.yaml` | Low — small, mechanical |
| `src/training/config.py` | `conf_threshold`…`assigner_collect_stats` (`:88-105`), `split_manifest` (`:59`) | 4 architecture fields + 4 W9 fields + `amp` removal + `__post_init__` body | Low-Medium — additive, but `amp` removal touches a line every config reads |
| `src/training/loop.py` | decode wiring (`:517-531`), assigner args (`:69-71`), OOM/NaN guards (`:364-366`, `:391-396`) | schedule, param groups, grad norms, `:68`, `:523`, **plus W9 device/accumulation/precision and D-G rewriting the very guards that landed there** | **High — the largest overlap, and now larger than in the first draft** |
| `src/models/master/master_model.py` | `unfreeze_rgb_stem` E6/Q10 fix (`:200-232`) | that method is rewritten (no `rgb_stem` exists) | **Medium — semantic, not textual.** The E6 fix's *rationale* is superseded by `freeze_stages=0`; the design keeps an `unfreeze_stem` equivalent so the capability is not silently lost |
| `src/training/run_artifacts.py` | atomic run layout, `stage_summary.json` (`:182-224`) | adds `experiment_sha256`, `effective_batch`, `precision`, `device` to the summary payload | Low — additive to a dict |
| `tests/` | `test_assigner.py`, `test_decode.py`, `test_split_integrity.py`, `test_freeze_policy.py`, `test_final_training_pipeline.py` | `test_freeze_policy.py` rewritten; `test_final_training_pipeline.py` and `test_training_loop.py` updated for W9/D-G; the other three untouched | Low-Medium |
| `scripts/visualize_damage_predictions.py`, `scripts/evaluate_checkpoint.py`, `src/training/decode.py` | **now on `main` at `9e43aa6`** | strides from checkpoint, `arch_version` check, required `strides` argument | Low — no longer blocked |

**The `loop.py` overlap deserves a sequencing note.** D-G rewrites the exact OOM/NaN guards that
`damage-map-audit` shipped, and W9 restructures the same `_train_epoch` body. Because D-G is a
blocking prerequisite anyway, the apply phase should land D-G + W9 in `loop.py` **first**, run the
test suite green, and only then apply the architecture rewiring — so that a failure in the second
half is not confounded with a training-loop regression in the first.

**Coexistence.** `training_mango.yaml` keeps working end-to-end (three levels, two-phase) — except
that no v1 checkpoint loads into the v2 `MasterModel`. That is the accepted hard break; the
`damage-map-audit` numbers survive as published results, not as reproducible runs, and `arch_version`
makes the boundary explicit instead of cryptic.

**Rollback.** `head_strides` and `assigner_level_ranges` are config-only, so P2 reverts without code
changes. `schedule` and `backbone_lr_mult` are config-only, so the two-phase schedule is restorable
by config. The only irreversible step is the deletion of `fusion.py` and `DualFPN`, which
`git revert` of the single PR restores; no checkpoint depends on the new files until V1 runs.

---

## Threat Matrix

**N/A** — no routing, shell, subprocess, VCS/PR automation, executable-file classification, or
process-integration boundary. The change touches CLI scripts, but only their argument parsing and
tensor handling; no new subprocess, no new file-execution path, no network or credential surface.
The two new scripts (`measure_vram.py`, `stem_init_audit.py`) are read-only measurement harnesses
that write into `reports/`.

---

## Open Questions

**Resolved since the first draft:**

- [x] **Q1** — Maintainer Decision R1.1: relative bars only. The limitation sentence — the change can
      prove the redesign is *better*, not that it is *sufficient* — is a required section of the
      validation report, not an optional caveat.
- [x] **Q5** — Maintainer Decision R2.5: `batch_size: 2` locally, time cost accepted. The measurement
      shows the cost is ≈zero (the card is saturated at batch 1), and gradient accumulation makes the
      choice attribution-neutral, so the ladder-ordering tension the first draft raised is moot.
- [x] **Q6** — Maintainer Decision R2.6: hardware portability becomes W9, designed in D-H/D-I. The
      first draft's ≈9.5 GB concern is confirmed in direction and resolved in practice: at batch 2 the
      target config projects ≈2.6 GB against 6.44 GB.

**Still open:**

- [ ] **Q7 (new, low stakes).** `effective_batch: 8` is chosen to match the value
      `training_mango.yaml:19` intended, so the redesign is compared against the schedule the previous
      runs meant to use. But no previous run *achieved* batch 8 on this hardware, so there is no
      measured precedent for it either. It is a free choice; 8 is proposed because it divides evenly
      into 2 / 4 / 8 across all three machines (D-H). Confirm 8, or name another value — it must be
      fixed before V1 and cannot change afterwards.
- [ ] **Q8 (new, conditional).** H-BF16 requires Ampere hardware and the same-machine fp32 reference
      (§6). If the A5000/3080 are not reachable during this change's window, H-BF16 is deferred with
      its bars registered — it gates nothing. Confirm deferral is acceptable rather than a scope cut
      needing a decision.
- [ ] **Q9 (informational, for the Q2 holder).** The authored-line estimate is restated at ≈3,480
      (band 3,200-4,200), up from 2,575. The single-PR `size:exception` stands; §7 records the one
      natural seam (W9 + D-G, ≈1,105 lines, a blocking prerequisite that lands first anyway) purely so
      the reviewer knows the option exists.
