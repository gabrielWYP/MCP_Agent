# Cross-attention fusion collapse — diagnosis and opt-in fix

Measured on `checkpoints/mastermodel_mango/best_model.pt` (the pre-redesign two-stream checkpoint,
`fusion_mode="cross_attention"`, head_strides `[8, 16, 32]`) with forward hooks on the val split,
inference only, CPU. No model was retrained for this report.

## 1. The attention is near-uniform

`StageAttentionFusion` runs `nn.MultiheadAttention` with RGB tokens as query and NIR tokens as
key/value over a pooled grid of at most 20x20 tokens. Mean attention entropy, normalised by
`log(num_keys)` (1.0 = perfectly uniform), over 8 val images:

| stage | stride | entropy / log(num_keys) |
|---|---|---|
| 0 | 4 | 0.9989 |
| 1 | 8 | 0.9974 |
| 2 | 16 | 0.9840 |
| 3 | 32 | 0.8872 |

Stages 0-2 are indistinguishable from uniform; only stage 3 has moved away from it.

## 2. The attention output barely varies across positions

Spatial variation of the attention output across query tokens,
`||std over queries|| / ||mean over queries||`, over 4 val images:

| stage | spatial std / mean |
|---|---|
| 0 | 0.1412 |
| 1 | 0.2476 |
| 2 | 0.0054 |
| 3 | 0.2371 |

With near-uniform weights every query receives approximately the same NIR average, so NIR enters
the RGB stream as one global vector per image rather than as a per-location signal. Stage 2 is the
clearest case (0.0054).

For scale: an **untrained** two-stream model (random init, same metric, 8 val images) gives
0.30 / 0.30 / 0.12 / 0.15 and entropy 0.99-1.00 at every stage. Near-uniform attention at init is
normal; the defect is that training did not move stages 0-2 away from it, and on stages 0, 1 and 3
the spatial variation is no lower than at init. Only stage 2 shows a clear collapse on the second
metric.

## 3. NIR is used — globally

Zeroing the NIR input changes the classification logits by **22.0%** (relative L2, mean over the
same 8 images). NIR is not ignored; it is consumed as a global conditioning vector.

## 4. Root cause

RGB and NIR are pixel-aligned, so the useful correspondence between a query token and a key token
is positional: the NIR token at the same location. But the attention logits
`(W_q · LN(rgb_i)) · (W_k · LN(nir_j))` depend on content only — no positional encoding is added
anywhere in `StageAttentionFusion`. Nothing tells a query which key is its own location, so the
softmax has no reason to peak and the cheapest stable solution is a near-uniform average.
Average-pooling to 20x20 (one token per 32x32 px, see `reports/damage-map-audit/h7-mechanism.md`)
makes neighbouring tokens more alike and the content-only signal weaker still.

## 5. Separate finding: 22 parameters get no gradient at `[8, 16, 32]`

With `head_strides: [8, 16, 32]` (`configs/experiment/twostream.yaml`), one backward pass leaves
**22 parameters with `grad is None`**: all of fusion stage 0 and both FPN P2 convolutions. The
stride-4 fused feature feeds only the P2 level, which that pyramid discards before the head
(`reports/damage-map-audit/h7-mechanism.md`). With `head_strides: [4, 8, 16, 32]`
(`configs/experiment/twostream_p2.yaml`, already on this branch) **0 parameters** lack gradient.
No code change is needed for this; `twostream_p2.yaml` resolves it.

## 6. What this change does

Adds `fusion_pos_encoding` (default `false`), plumbed `StageAttentionFusion` -> `CrossModalFusion`
-> `MasterModel` -> `TrainingConfig`, and into every place a `MasterModel` is built (`train.py`,
`scripts/evaluate_checkpoint.py`, `scripts/visualize_damage_predictions.py`, the KD teacher).

- When on, a fixed 2D sinusoidal encoding of the pooled PxP token grid (half the channels encode
  the row, half the column; integer positions, temperature 10000) is added to the Pre-LN
  normalised **query and key** only. The value stays pure NIR content: position decides *where*
  to read, not *what* is read.
- It is computed per forward and registers no parameter or buffer, so the state_dict is
  identical with the flag on or off. Consequence: `strict=True` cannot catch a mismatch, so
  loaders read the flag from the checkpoint's recorded `config` (absent -> `false`, which is
  correct for every existing checkpoint) and warn if the caller's config disagrees. The KD
  teacher reads it from its own checkpoint, never from the student's config.
- Default `false` is bit-identical to the previous forward (tested). `true` is rejected on
  `fusion_mode="early"`, which has no cross-attention.
- New experiment file `configs/experiment/twostream_p2_posenc.yaml`: `twostream_p2.yaml` plus
  `fusion_pos_encoding: true` and its own `output_dir`. A separate file because
  `experiment_sha256` hashes file bytes (`validation-report.md` 13.9).

Tests (`tests/models/master/test_fusion_pos_encoding.py`) prove that with spatially constant RGB
and NIR inputs the attention is exactly uniform without the flag and non-uniform with it — i.e.
position now reaches the logits.

## 7. What this does NOT show

- **No retraining was done.** Whether the encoding breaks the collapse on a trained model is
  unmeasured. At random init it barely changes the entropy (0.99-1.00 either way); the
  projections must learn to use it.
- **AP impact is unmeasured.** Nothing here says damage AP50 goes up.
- The measurements come from one checkpoint (one seed) and 8 (entropy) / 4 (spatial std) val
  images.

## 8. Follow-ups

**(a) Double residual on pooled stages — left unchanged deliberately.** When a stage is pooled,
`fused = upsample(rgb_pooled + attn + ffn) + rgb_feat`, so RGB is added twice (once pooled and
upsampled, once at full resolution); on the unpooled stage (stage 3 at 640 px) it is added once.
This scales the RGB path differently per stage and may further drown the NIR delta. It is left as
is so that the positional-encoding result is attributable to one change.

**(b) Aligned local fusion path.** If positional encoding alone does not break the collapse, the
alternative is to stop asking attention to rediscover an alignment that is already known: an
aligned per-pixel fusion (e.g. a gated conv over `concat(rgb, nir)` at full stage resolution),
alone or alongside the attention.

**(c) The experiment to run.** `twostream_p2.yaml` vs `twostream_p2_posenc.yaml`, 3 seeds each
(42, 1337, 2024), damage AP50 on val and test, plus the entropy and spatial-std diagnostic on each
trained checkpoint. The fix is supported only if the posenc arm both lowers the stage 0-2 entropy
and raises damage AP50 beyond the seed spread.
