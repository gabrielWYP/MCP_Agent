# Upstream finding — the NIR signal itself is weak and underexposed

Measured directly from the image files and label geometry. No model involved.

## 1. Damage is barely separable from healthy tissue by NIR mean intensity

For each image containing both a mango box and damage boxes, comparing NIR pixels inside damage
regions against the rest of the same mango (152 images):

```
NIR mean, damage region:  28.46 +- 7.04
NIR mean, healthy tissue: 29.76 +- 3.45
intra-image difference:   -1.30 +- 6.48
damage darker in:          90/152 images (59%)
Cohen's d:                -0.200   (weak)
t-test vs 0:              t = -2.46, p = 0.0152
```

The effect is statistically detectable but small. Directional consistency of 59% is barely above
chance. **Regional mean brightness is nearly useless as a damage cue.**

### Important limitation

This measures MEAN INTENSITY only. Damage may be characterised by texture, local contrast, or edge
structure rather than average brightness, and a convolutional model can learn those where a mean
comparison sees nothing. This result does NOT show that damage is undetectable in NIR.

What it does show is that the simplest cue is nearly absent, so the usable signal is most likely
textural — which connects directly to the architecture finding: `adaptive_avg_pool2d` to a 20x20
grid preserves regional mean intensity (the weak cue) and destroys texture (the plausible one).
Average pooling is precisely the operation that discards what is most likely carrying the signal.

## 2. The NIR images are severely underexposed

Sampled over 60 cached NIR images against 60 RGB from the same dataset:

| | p1 | p99 | usable range | distinct levels |
|---|---|---|---|---|
| NIR | 0.0 | 51.0 | **51 of 255** | 77 of 256 (30%) |
| RGB (grayscale) | 11.0 | 255.0 | 244 of 255 | — |

**NIR occupies roughly 20% of the available dynamic range** — about five times less than RGB from
the same rig. Whatever subtle intensity or texture difference distinguishes damaged tissue is being
quantised into a fraction of the levels available to represent it.

Placed beside finding 1: the measured damage-versus-healthy difference is 1.3 levels on images
spanning 51. Correctly exposed, the same physical difference would span roughly 6 levels. Still
subtle, but five times more separable — before any model sees it.

### Caveat to check before acting

This was measured on the cached JPEGs under `data/cache/`. JPEG compression and whatever conversion
produced the cache may themselves have reduced the range. The original captures in
`data/mango_nir.zip`, and the acquisition settings, should be checked before concluding the sensor
data is underexposed rather than the pipeline discarding range. If the source has more range, this
is a preprocessing bug and is cheap to fix. If not, it is an acquisition problem.

## 3. Why this reorders everything

No fusion architecture recovers information that was never captured, or that was quantised away
before the model ran. This sits upstream of all four architectural defects.

Revised order of work:

1. **Verify and fix the NIR dynamic range.** Check the source archives and the caching pipeline.
   Per-image contrast normalisation is nearly free and benefits every downstream stage. This is the
   cheapest intervention with the widest reach.
2. **Establish whether damage is learnable from NIR at all**, once exposure is addressed: train a
   small classifier on NIR crops of damaged versus healthy tissue. If it cannot separate them, the
   project's premise needs revisiting before any further architecture work.
3. **Only then the architecture**: reconnect P2, revisit `max_tokens_side`, ensure Phase 2 actually
   executes, and give NIR real dedicated capacity beyond a 1,824-parameter stem.
4. **Retrain and measure** against the pre-registered bars.

Steps 1 and 2 are cheap and decide whether step 3 is retraining or redesign. Running step 3 first
risks rebuilding a fusion around a signal that was never there.

## 4. Why the resolution sweep could not settle it

`max_tokens_side` was swept at inference over 20 / 40 / 80 with the flattening ablation:

| tokens | token size | damage recall (real NIR) | flattened | gap | mango AP50 |
|---|---|---|---|---|---|
| 20 | 32 px | 0.6129 | 0.2258 | 0.3871 | 0.8595 |
| 40 | 16 px | 0.5161 | 0.0968 | 0.4194 | 0.5118 |
| 80 | 8 px | 0.4839 | 0.0968 | 0.3871 | 0.4815 |

The gap does not widen, but every metric degrades including mango, because the model was trained at
20 and changing it at inference is a distribution shift its LayerNorms and downstream stages never
saw. The flat gap is therefore uninterpretable: it is equally consistent with finer tokens carrying
no extra lesion information and with the model being unable to exploit them. **This experiment was
poorly designed and does not answer the retrain-versus-redesign question.** Steps 1 and 2 above do.
