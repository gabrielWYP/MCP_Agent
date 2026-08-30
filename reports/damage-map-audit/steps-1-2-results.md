# Steps 1 and 2 — is the signal there, and is the premise sound?

## Step 1 — NIR dynamic range: an acquisition issue, not a pipeline bug

| source | p1 | p99 | usable range | distinct levels |
|---|---|---|---|---|
| `data/mango_nir.zip` (original) | 0.0 | 49.4 | 49.4 | 77 of 256 |
| `data/cache/` (processed) | 0.0 | 51.0 | 51.0 | 77 of 256 |
| RGB, same rig | 11.0 | 255.0 | 244.0 | — |

The cache did not destroy anything — the source is already this narrow. **The NIR images were
captured underexposed**, occupying roughly 20% of the available range.

**Correction to the earlier recommendation.** This yields no actionable software fix. The dataset
already normalises with `nir_mean=0.0569, nir_std=0.0546`, which is effectively a contrast stretch,
and any further linear rescaling is an affine transform the network can learn on its own — it adds
no information. The irrecoverable part is the quantisation to 77 levels, lost at capture. Worth
correcting for future acquisitions; nothing to fix in this codebase.

## Step 2 — damage IS separable, and the multimodal premise is validated

A small CNN trained on crops from the reconciled splits: damage boxes as positives, healthy regions
inside the same mango as negatives. 478 train crops, 59 val crops, 5 seeds, AUC averaged over the
final 10 epochs.

### Control applied

An initial run scored AUC 0.86 but sampled healthy crops at random sizes while damage crops used
their annotated sizes. Since all crops are resized to 48x48, the classifier could have been reading
resampling artefacts correlated with original box size rather than tissue texture. Re-running with
**each healthy crop matched to the exact dimensions of the damage crop it accompanies** dropped NIR
to 0.711. Part of the original figure was that confound. All numbers below use the size-matched
control.

### Results

| input | AUC (val) | std over 5 seeds |
|---|---|---|
| NIR only | 0.7154 | 0.0516 |
| RGB only | 0.7774 | 0.0333 |
| **RGB + NIR** | **0.9171** | **0.0044** |

Three things follow:

1. **The signal exists in NIR** — 0.715 is well above chance, though modest, consistent with the
   weak mean-intensity effect (Cohen's d 0.20) implying the cue is textural rather than brightness.
2. **RGB alone carries more damage information than NIR alone** — 0.777 versus 0.715. This was not
   expected and matters for how the two branches should be weighted.
3. **The two are strongly complementary.** Together they reach 0.917, far above either alone, and
   with the lowest seed variance of the three. This is synergy, not addition.

## What this settles

**The project's multimodal premise is correct.** RGB+NIR fusion is the right idea and the data
supports it: 0.917 AUC on localised crops.

**The current implementation is not realising it.** The master reaches damage AP50 0.064 on data
that supports 0.917 AUC. The gap is the architecture, not the premise and not the sensor.

So the answer to "retrain or redesign" is **redesign the fusion**. Retraining the existing design
would rebuild the same four defects: a 1,824-parameter NIR-specific capacity against a shared frozen
trunk, a never-executed Phase 2, a discarded stride-4 fusion stage with bit-exact-at-init
LayerNorms, and 32-px tokens against 30-px lesions.

## Limitations that bound this conclusion

- **AUC on pre-localised crops is not detection AP.** These experiments hand the classifier the
  correct box. A detector must localise as well as classify, which is strictly harder. 0.917 is a
  ceiling on what a perfect architecture could exploit, not a prediction of achievable AP.
- **The val set is 59 crops.** The seed variance is small (0.0044 for the combined model), but the
  sample is not large.
- Healthy negatives are drawn only from inside the mango box. Detection also has to reject
  background, which was not tested here.
- The 77-level quantisation still caps what any model can extract from NIR.

## Recommended next work

1. Redesign the fusion around the measured facts: give NIR real dedicated capacity; reconnect the
   stride-4 stage so lesion-scale structure reaches the loss; replace or reduce the 20x20 average
   pooling, which preserves the weak mean-intensity cue and destroys the textural one that carries
   the signal; ensure Phase 2 actually executes.
2. Weight the branches by what was measured — RGB carries more damage signal alone than NIR does, so
   a design that treats NIR as the primary damage channel is starting from a false assumption.
3. Re-run the ladder's training rungs against the pre-registered bars once the fusion changes land.
