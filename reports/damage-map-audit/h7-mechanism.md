# H7 mechanism — why the fusion cannot read NIR

Measured on `checkpoints/mastermodel_mango/best_model.pt` (epoch 68) with forward hooks,
inference only.

## 1. Every fusion stage is pooled to a 20x20 grid

`StageAttentionFusion.__init__` takes `max_tokens_side: int = 20` (`src/models/master/fusion.py`),
and the forward applies `F.adaptive_avg_pool2d` to both RGB and NIR before flattening to sequences.

| stage | feature map | channels | pooled grid | one token covers |
|---|---|---|---|---|
| 1 | 160x160 | 96 | 20x20 | 32x32 px |
| 2 | 80x80 | 192 | 20x20 | 32x32 px |
| 3 | 40x40 | 384 | 20x20 | 32x32 px |
| 4 | 20x20 | 768 | 20x20 | 32x32 px |

Damage lesions measure a median short side of **30 px** at 640 px input, with p25 at 22 px.
**One token covers 32x32 px, so a typical lesion fits inside a single token** — and stage 1 discards
a 160x160 map down to 20x20, an 8x reduction, before the attention ever runs.

Damage in NIR is a subtle local intensity difference. Average-pooling a 32x32 neighbourhood mixes
the lesion with the surrounding healthy tissue. The signal is attenuated before the fusion can
attend to it.

## 2. The attention output is loud, not silent

The first hypothesis was that the NIR branch had gone dead and contributed nothing. That is wrong:

| stage | \|\|attn_out\|\| as % of \|\|query\|\| |
|---|---|
| 1 | 34.1% |
| 2 | 40.7% |
| 3 | 167.7% |
| 4 | 577.6% |

At stage 4 the fused output norm is 32x the RGB feature norm. The NIR path dominates numerically.

## 3. But the attention weights have collapsed to uniform

Averaged over 6 validation images, per-query entropy normalised so 1.00 means a perfectly uniform
distribution over the 400 NIR tokens (no discrimination at all):

| stage | normalised entropy | mean max weight | uniform baseline | std across queries |
|---|---|---|---|---|
| 1 | **0.9978** | 0.00407 | 0.00250 | 0.000130 |
| 2 | **0.9948** | 0.00455 | 0.00250 | 0.000304 |
| 3 | 0.9687 | 0.02555 | 0.00250 | 0.000111 |
| 4 | 0.7811 | 0.10999 | 0.00250 | 0.001789 |

Two things at once:

- **Near-uniform within a query.** At stages 1 and 2 the entropy is 0.995+, so each RGB query
  spreads its attention almost evenly across all 400 NIR tokens rather than selecting the region
  that matters.
- **Near-identical across queries.** The standard deviation of the weights between different query
  positions is 1e-4 to 1e-3. A query at one location and a query at another produce essentially the
  same distribution. There is no spatial selectivity whatsoever.

A uniform attention over values computes their global average. So the fusion adds the same
NIR-derived constant vector at every spatial position.

## 4. This explains all three ablation results

| observation | explanation |
|---|---|
| mispaired NIR changes nothing (damage recall 0.6129, tp=19, both identical) | a global average of NIR values is nearly the same for any real NIR image of the same subject matter |
| zeroing NIR collapses all output to zero, mango included | the constant shifts far out of the distribution the downstream neck and head were trained under |
| attn_out is large in magnitude yet carries no localisation | a large constant added uniformly moves features without encoding where anything is |

## 5. Root-cause chain for the original question

1. Damage is annotated on NIR (`labeling_config.xml` binds `RectangleLabels toName="nir"`,
   186/186 export items are `mango_nir_*`), so the class is defined by NIR appearance.
2. The fusion pools NIR to 20x20, making one token roughly one lesion, attenuating the cue.
3. The attention weights are uniform and position-invariant, so the fusion returns a global NIR
   average rather than a localised read.
4. NIR therefore contributes magnitude but no information. Everything the master detects, it
   detects from RGB.
5. Mango is large and visible in RGB — AP50 0.86, recall 1.00. Damage is defined by a signal the
   model effectively never receives — AP50 0.064, precision 0.023.

This explains what the four refuted hypotheses could not, and it explains why the damage figure
stayed pinned at 0.04-0.09 across every model, every run, and every change made to assignment,
decoding, suppression and labels.

## 6. What this does not settle

- Whether the collapse is a training failure (the attention never learned to discriminate) or a
  structural one (the 20x20 pooling leaves nothing to discriminate). These are separable: raising
  `max_tokens_side` and re-measuring entropy would distinguish them.
- Whether damage is learnable from NIR at all by a fusion that does read it.
- Whether the frozen `rgb_stem` and the never-executed Phase 2 unfreeze contributed.

## 7. Cheapest next experiments

1. **Resolution**: raise `max_tokens_side` so a lesion spans several tokens and re-measure entropy
   and the mispaired-NIR ablation. Attention cost grows with the square of the token count, so
   stage 1 at 160x160 is not affordable — but stage 3 and 4 are, and per-stage values are possible.
2. **Is the signal even there**: train a small NIR-only classifier on damage crops. If it cannot
   separate damage from healthy tissue in NIR, no fusion will help and the premise needs revisiting.
3. **Attention supervision**: the annotations give exact damage locations in NIR, so the attention
   map can be supervised directly rather than left to emerge.

---

# Addendum — the causal chain, verified

Independent architectural review plus direct verification by the orchestrator.

## The chain

**1. There is no NIR trunk.** `MasterBackbone` has exactly three children: `rgb_stem` (4,896 params),
`nir_stem` (1,824 params), and `shared_stages` (**27,813,696 params**). RGB and NIR are pushed
through the *same* stages. The only NIR-specific capacity in the entire model is a 1,824-parameter
stem. Everything above it is ImageNet RGB-texture filters.

**2. Those shared stages were frozen for the whole evaluated run.** Phase 1 trains with
`freeze_stages=4` (`src/training/loop.py:125-131`), freezing all 27.8M. Phase 2 would unfreeze
26.58M — it ran for exactly one epoch (74) before the run ended, and the evaluated checkpoint is
epoch 68, inside phase 1. So during training, the only weights that could adapt to 850 nm
reflectance were those 1,824.

**3. The neck discards the finest fusion stage.** `DualFPN` builds **4** fusion stages but only
**3** `fusion_convs`, and `neck.py:165` zips from `[1:]`, dropping P2. Verified: 4 stages, 3 convs.
A gradient test in the review found `fused_features[0].grad is None` — there is no gradient path at
all. Stage-1 fusion runs on every forward pass and its output is thrown away. That was the only
stage at stride 4, the one resolution where a 24-50 px lesion spans more than a couple of cells.

**4. Therefore stage-1's fusion never trained — provably.** Its LayerNorms are **bit-exactly at
initialisation** after 68 epochs:

```
etapa 1 norm_rgb: max|gamma-1| = 0.000e+00   max|beta| = 0.000e+00
etapa 1 norm_nir: max|gamma-1| = 0.000e+00   max|beta| = 0.000e+00
etapa 2 norm_rgb: max|gamma-1| = 6.711e-02   max|beta| = 2.262e-02
etapa 3 norm_rgb: max|gamma-1| = 4.729e-02   max|beta| = 4.483e-02
etapa 4 norm_rgb: max|gamma-1| = 5.601e-02   max|beta| = 2.714e-02
```

Exact zeros, not small numbers. No gradient ever reached it. Stages 2-4 moved by only 4-7%, so
their attention is also close to its random initialisation.

**5. An untrained cross-attention produces uniform softmax.** Xavier-initialised Q/K on LayerNormed
inputs give logits near zero, hence the measured entropy of 0.9978, hence `attn_out ~= mean(V)`.
The review's decomposition confirms it: **99% of `attn_out` is the query-independent DC term**.

**6. That global NIR descriptor is the same for every image.** Pairwise cosine similarity between
*different* images: **1.0000** at stages 1 and 3, 0.9999 at stage 2, 0.9982 at stage 4. Same rig,
same illumination, and a global average discards everything that distinguishes them. Mispairing NIR
changes `nir_feat` by 13-45% but `attn_out` by only 1.3-4.9%, and the resulting change in the damage
logits is unstructured — correlation 0.079 with the real logit map. Noise, not signal.

## Four independent reasons, any one sufficient

1. NIR has 1,824 dedicated parameters against a 27.8M frozen ImageNet trunk it shares with RGB.
2. Phase 2 never effectively ran, so 26.58M parameters stayed frozen.
3. The finest fusion stage is computed and discarded with no gradient path.
4. `max_tokens_side=20` pools every stage to 32x32 px tokens, and a median lesion is 30 px.

Fixing any one alone would not be enough.

## Revised recommendation

The two zero-cost forward-pass ablations should run before any retraining: replace `attn_out` with
its own query-mean, and separately with zero. If neither moves mAP, the fusion contributes nothing
measurable today and the retraining question becomes a redesign question.

The prior recommendation to raise `max_tokens_side` is demoted. Resolution is real but it is the
fourth constraint, not the first — a fusion fed by a frozen ImageNet trunk, with its finest stage
disconnected from the loss, will not learn a 24-50 px NIR lesion cue no matter how fine the grid.

---

# CORRECTION — the fusion does read NIR structure

The earlier claim in this document, that the fusion "does not read NIR content", was drawn from the
mispairing ablation alone and is **too strong**. Two further zero-cost ablations correct it.

## The four ablations together

| condition | mango AP50 | damage AP50 | damage recall | damage fp |
|---|---|---|---|---|
| baseline | 0.8595 | 0.0643 | 0.6129 | 799 |
| NIR from a different image | 0.9015 | 0.0684 | 0.6129 | 785 |
| **NIR flattened** (spatial mean, structure removed, statistics kept) | 0.4630 | **0.0006** | **0.2258** | 4769 |
| NIR zeroed | 0.0000 | 0.0000 | 0.0000 | 0 |
| `attn_out` replaced by its query-mean (DC) | 0.0278 | 0.0123 | 0.1290 | 107 |
| `attn_out` zeroed | 0.0000 | 0.0000 | 0.0000 | 0 |

## What this changes

**Flattening NIR drops damage recall from 0.6129 to 0.2258 and damage AP by two orders of
magnitude.** NIR spatial structure is therefore functionally used. The fusion is not inert.

**The DC-only ablation also collapses the model** (mango AP50 0.8595 to 0.0278), which refutes the
architectural review's prediction that the query-mean carries all the functional content. The
spatially-varying component of `attn_out` is essential even though it is only 0.5-24% of its norm.

## The reading that fits all six results

The fusion requires NIR to *have* structure, but not the structure *belonging to that image*.

That is explicable without contradiction. Every image in this dataset comes from the same rig, the
same framing, one mango against a similar background. A different image's NIR still presents a
mango-shaped bright region in roughly the same place, so mispairing perturbs the coarse structure
very little. The 18-image val split makes this especially weak as a probe.

So: **structure at mango scale survives; structure at lesion scale does not.** That is precisely
what pooling to a 20x20 grid predicts — 32 px tokens against a 30 px median lesion. Mango spans
~8x8 tokens and survives. A lesion occupies one token and is averaged away.

## Consequence for the ranking

The resolution constraint (`max_tokens_side=20`) is **promoted back to first place**. It was demoted
on the reasoning that a frozen trunk would prevent learning regardless; the flattening result shows
the fusion does learn to use NIR structure at the scales that survive pooling, which makes the
pooling the binding constraint on the scale that matters.

The other three defects remain real and verified: the 1,824-parameter NIR-specific capacity, the
never-executed Phase 2, and the discarded stage-1 fusion with its bit-exact-at-init LayerNorms. The
last of these is now more damaging than it first appeared — stage 1 is the only stride-4 stage, the
one place lesion-scale structure could have survived, and it is disconnected from the loss.

## What should be measured next, still without training

Re-instantiate the same checkpoint with a larger `max_tokens_side` (the module has no
resolution-dependent parameters, so the weights load unchanged) and repeat the flattening ablation.
If the damage recall gap between real and flattened NIR widens, finer tokens are already carrying
more lesion information and retraining at that resolution is justified. Attention cost grows with
the square of the token count, so use batch size 1 and start at stage 3 and 4.
