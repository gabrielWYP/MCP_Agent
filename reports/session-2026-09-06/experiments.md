# Session log — 2026-09-06

Everything measured in one day, in the order it happened, including the results that
contradicted the hypothesis being tested and the three hypotheses that were withdrawn.

Companion documents:
- [`reports/fusion-redesign/validation-report.md`](../fusion-redesign/validation-report.md) — discharges the pre-registered Phase 13
- [`reports/damage-map-audit/`](../damage-map-audit/) — the prior baseline audit
- Raw metrics: `reports/clean-labels/*/metrics.json`

---

## 1. The dataset was wrong, and the pipeline was writing it wrong

### The defect

`scripts/convert_nir_labels.py` returned early when the Label Studio export held no
damage boxes for an image:

```python
if not damage_bboxes_rgb:
    skipped += 1
    continue          # writes nothing; destination file keeps whatever was there
```

The pipeline passes the same directory as `--splits` and `--output-dir`, and that file
already held class-1 boxes written by `NIRSegmenter` in
`src/annotation/annotation_generator.py`. So every image an annotator had declared
damage-free kept algorithmic damage and stopped being a negative example.

Fixed in `4f360b4`: the mango-line lookup moved above the check, and a damage-free image
is now rewritten with the mango box alone.

### Measured effect

| | before | after |
|---|---|---|
| damage boxes, train | 253 | 190 |
| damage boxes, val | 31 | 27 |
| damage boxes, test | 36 | 26 |
| **damage boxes, total** | **320** | **243** |
| **damage-free images** | **34** | **50** |
| images | 187 | 186 |

243 and 50 match the human Label Studio export exactly (186 tasks, 243 `damage`
rectangles across 136 images).

The phantom boxes were **not** detectable by geometry: same median normalised area as
the human ones (ratio 0.98×), and all 320 fell inside the mango box (minimum containment
0.923). Only the tail differed — p95 normalised area 0.01010 human vs 0.01478 generated.

Also removed: `mango_rgb_1780685735`, which carried a label but whose RGB and NIR images
did not exist (the "unmanifested stem" left open in `damage-map-audit/w0-reconcile.md`).
Backup at `data/annotations/removed_1780685735_20260906_142558`.

**Still open:** `mango_rgb_1782752988` has RGB and NIR images but no label file, so it is
never used. Harmless, but wasted on a dataset this size.

**Consequence:** every damage AP measured before this date — including the 0.0643
baseline — was computed against contaminated ground truth.

---

## 2. Runs

Four training runs, one recovered checkpoint. All on a GTX 1660 SUPER, fp32,
`effective_batch: 8` (batch 2 × grad_accum 4), seed 42.

| run | architecture | levels | labels | epochs | checkpoint |
|---|---|---|---|---|---|
| V1 | early fusion | `[4,8,16,32]` | dirty | 62 (early stop) | `fusion_redesign/v1_seed42` |
| H-E | early fusion | `[8,16,32]` | dirty | 100 | `fusion_redesign/he_seed42` |
| V1-clean | early fusion | `[4,8,16,32]` | **clean** | 70 (early stop) | `fusion_redesign/v1_clean_seed42` |
| two-stream-clean | cross-modal | `[8,16,32]` | **clean** | 65 (early stop) | `twostream_clean_seed42` |
| two-stream-old | cross-modal | `[8,16,32]` | dirty, **leaked** | 68 (Jun 5) | `mastermodel_mango` |

---

## 3. Results — all models, same clean labels

Val: 18 images, 27 damage instances. Test: 20 images, 26 damage instances.
`conf_threshold 0.25`, NMS on, per-class decode.

### Damage (class 1)

| model | val AP50 | val recall | val fp | test AP50 | test recall | test fp |
|---|---|---|---|---|---|---|
| two-stream, old (leaked) | 0.0540 | 0.6296 | 853 | 0.0637 | 0.5000 | 988 |
| **two-stream, clean** | **0.1282** | 0.5185 | **442** | 0.0417 | 0.3846 | **440** |
| early fusion V1, dirty | 0.0038 | 0.5185 | 2,117 | 0.0034 | 0.4615 | 2,639 |
| early fusion V1, clean | 0.0000 | 0.0000 | 266 | 0.0000 | 0.0000 | 328 |
| early fusion H-E | 0.0000 | 0.0000 | 1,133 | 0.0000 | 0.0000 | 1,172 |

### Mango (class 0)

| model | val AP50 | val precision | test AP50 |
|---|---|---|---|
| two-stream, old | 0.8593 | 0.1343 | 0.7509 |
| two-stream, clean | 0.9889 | 0.1118 | 0.8082 |
| early fusion V1, dirty | 1.0000 | 0.8182 | 0.9225 |
| early fusion V1, clean | 1.0000 | 0.0931 | 0.5951 |
| early fusion H-E | 1.0000 | 0.4865 | 0.7607 |

---

## 4. What the numbers support

**Robust — replicated on two independent splits.** Cross-modal fusion detects damage;
early fusion does not. Every early-fusion configuration scores damage AP50 0.0000 and
recall 0.0000 on both val and test; both cross-modal checkpoints detect on both. That is
*something versus nothing*, and it is far larger than the noise floor measured below.

**Robust — consistent in direction on both splits.** Retraining two-stream on clean data
roughly halved damage false positives (853 → 442 val, 988 → 440 test) and raised
precision (0.0195 → 0.0307 val, 0.0130 → 0.0222 test), while lowering recall
(0.6296 → 0.5185 val, 0.5000 → 0.3846 test). Pooled across both splits: the old
checkpoint produced 30 true positives with 1,841 false ones; the retrained model produces
24 with 882.

**Not established.** Whether the retrained two-stream beats the old contaminated
checkpoint on AP. Val says yes (0.1282 vs 0.0540), test says no (0.0417 vs 0.0637).

**The noise floor.** The same checkpoint scores damage AP50 **0.1282 on val and 0.0417 on
test** — a threefold spread. On 18 and 20 images with 27 and 26 instances, a single
image changes AP by several hundredths. No single-split, single-seed damage AP in this
project should be treated as a measurement.

**Leakage caveat on the old checkpoint.** It is dated Jun 5; the split reconciliation
landed Aug 29. Against the current splits, 3 of 18 val images and 5 of 20 test images
were in its training set. Its numbers are inflated by memorisation; the runs from Aug 31
onward are not.

---

## 5. The clean-label retrain made damage detection worse

Stated separately because it contradicts the reason the labels were cleaned.

| | V1 dirty-trained | V1 clean-trained |
|---|---|---|
| epochs with damage TP > 0 | 14 / 62 | **0 / 70** |
| damage TP, maximum | 15 | **0** |
| val recall (clean labels) | 0.5185 | 0.0000 |

Cleaning removed 25% of the training positives (253 → 190) and added 41% more
damage-free images (29 → 41). Both shifts push a classifier toward suppressing the class.
The removed boxes were `NIRSegmenter` output — plausibly the higher-contrast, easier NIR
regions — so removing them may have stripped the learnable signal and left only the
harder human lesions. On 190 positives across 148 images, more-but-noisy beat
fewer-but-clean.

This does **not** make the phantom labels acceptable. A model trained against invented
ground truth cannot be validated. But label quality was not the bottleneck it appeared to
be, and fixing it does not on its own recover damage AP.

---

## 6. Hypotheses withdrawn during the session

Recorded so they are not re-argued.

| hypothesis | why it was withdrawn |
|---|---|
| "One shared head must span a 30× size range" | `head.py:144` builds an `nn.ModuleList` with one independent `DecoupledHead` per level. Each head covers only its own level. Stride normalisation is a constant shift per level; it does not compress the per-level range. |
| "TAL soft targets collapse to ≈0" | The June fix survived. `loss.py:491` builds `target_cls` with `F.one_hot(...)` — hard binary 1.0 targets. `target_scores` is computed by the assigner and never used in the loss, so the `iou⁶` term affects anchor *selection* only. |
| "Generated boxes are 56% larger than human ones" | An artefact of converting Label Studio percentages to pixels assuming uniform scaling on non-square images. In normalised units the median areas match at 0.98×. |
| "Background swamping is the cause" (via H-E) | Cutting background supervision fourfold made damage false positives rise, not fall. Directionally against the hypothesis; formally uninterpretable because H-E is confounded. |
| "Clean labels will recover damage AP" | Predicted recall would hold near 0.52 and false positives would collapse. Recall went to zero across the whole trajectory. |

Also disproved as a measurement concern: the AP computation itself is correct. It sorts
by confidence, keys `matched_gt` by `(image_idx, gt_idx)` so ground truth cannot collide
across images, and the appended `(recall=1.0, precision=0.0)` point contributes exactly
zero after the monotonic pass. The numbers above are trustworthy at the resolution
their split sizes allow.

---

## 7. Three seeds — `σ_d` measured for the first time

Two-stream, `configs/experiment/twostream.yaml`, `schedule: end_to_end` with all
46,642,930 parameters trainable, clean labels, reconciled splits, seeds {42, 1337, 2024}.
Checkpoints `checkpoints/twostream/seed*`, metrics `reports/twostream-3seeds/`.

### Damage, evaluated on clean labels

| seed | val AP50 | val recall | val tp | test AP50 | test recall | test tp |
|---|---|---|---|---|---|---|
| 42 | 0.1817 | 0.6296 | 17 | 0.1161 | 0.5385 | 14 |
| 1337 | 0.1731 | **0.8148** | **22** | **0.1760** | 0.6538 | 17 |
| 2024 | 0.1340 | 0.2222 | 6 | 0.0108 | 0.1154 | 3 |
| **mean** | **0.1629** | 0.5556 | | **0.1010** | 0.4359 | |
| **σ_d** | **0.0254** | 0.3032 | | **0.0836** | 0.2835 | |

### Verdict against the pre-registered bar (`CONFIRM ≥ 2·σ_d` vs early fusion's 0.0000)

- **val: CONFIRM.** 0.1629 against 2σ = 0.0508 — a 6.4 σ separation.
- **test: does NOT confirm.** 0.1010 against 2σ = 0.1672 — 1.2 σ.

Seed 1337's val recall of **0.8148 (22 of 27 lesions)** is the highest damage recall this
project has produced. Precision remains 0.02–0.05 everywhere: the model finds damage and
buries it under hundreds of false positives. No seed changes that.

### A failed hypothesis about the variance

Seed 2024 underperforms badly, and its `best_model.pt` (epoch 31) sits 2 epochs before
its own trajectory peak (epoch 33, 0.2033). That suggested the broken checkpoint-selection
criterion — the unweighted two-class mean with `>=` at `loop.py:420` — was manufacturing
the variance. **It is not.** Selecting instead on val damage AP50 over every saved
checkpoint picks the *same* three checkpoints:

| seed | candidates (val damage AP50) | selected | resulting test |
|---|---|---|---|
| 42 | ep22 **0.1817** · ep30 0.1191 · ep40 0.0930 | ep22 | 0.1161 |
| 1337 | ep30 0.1339 · ep40 **0.1731** | ep40 | 0.1760 |
| 2024 | ep30 0.0209 · ep31 **0.1340** · ep40 0.1268 | ep31 | 0.0108 |

The trajectory peaks the fix was meant to capture are themselves noise: seed 2024 scores
0.0209 at epoch 30, 0.2033 at 33 and 0.1268 at 40. **On an 18-image validation split the
act of selecting a checkpoint is itself noise-limited**, and no selection criterion
repairs that. `loop.py` was therefore left unchanged; the `>=` defect is real but it costs
early-stopping efficiency, not accuracy.

### The improvement is NOT attributable to the label fix

From 0.0637 (old checkpoint, test) to 0.1010 (three-seed mean, test), **three variables
moved at once**: clean labels, removed split leakage, and a fully unfrozen backbone (the
old checkpoint trained frozen — Phase 2 never ran).

The only single-variable test of the labels available is §5, and it points the other way:
holding architecture and schedule fixed, cleaning the labels took early fusion from
14/62 epochs with true positives to **0/70**. Attributing the two-stream gain to the label
fix is unsupported by anything measured here.

Isolating it needs one more run: two-stream, current config, **dirty labels**
(`data/annotations/yolo/labels_backup_20260906_121042`). With σ_d ≈ 0.08 on test, a single
seed will likely be inconclusive; three would be needed to decide.

---

## 8. What to do next

1. **Isolate the label variable** — the run described immediately above.
2. **Keep P2.** `head_strides: [4,8,16,32]`. See the validation report's 13.5 — the
   pre-registered instruction to retire it must not be executed.
3. **Run H-D** (`in_channels=3`), the only remaining test of whether NIR contributes in
   the early-fusion path.
4. **Two one-line fixes before any further run**, neither of which changes the model:
   `logging.basicConfig` is missing from `src/training/train.py`, so the augmentation
   bbox-drop instrumentation logs at INFO and is silently discarded (zero such lines in
   any run log); and best-checkpoint selection uses the unweighted two-class mean with
   `>=` (`loop.py:420`), which on a saturated mango class re-arms `patience` on every tie
   and never fires. Select on `ap50_class_1`.
5. **Fix `experiment_sha256`** to include resolved overrides. See validation report 13.9.
