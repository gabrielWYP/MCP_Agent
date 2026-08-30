# H7 — The cross-modal fusion ignores NIR content: CONFIRMED

Raised by the maintainer, who challenged the choice of the RGB-only student as the subject for a
damage-detection question. That challenge was correct and led directly to this finding.

## The premise the maintainer identified

Damage ground truth was annotated on NIR images, not RGB. Verified in
`data/label_studio_nir/labeling_config.xml`:

```xml
<Image name="nir" value="$image" zoom="true" brightnessControl="true" contrastControl="true"/>
<RectangleLabels name="damage" toName="nir">
```

All 186 items in `Datasetv2_export.json` reference `mango_nir_*.jpg`. The human annotator adjusted
brightness and contrast on the NIR image to locate damage. **The damage class is defined by what is
visible in NIR.** An RGB-only model is therefore being asked to detect a class whose ground truth
comes from a signal it never receives — which made the student the wrong subject for the question.

## Master baseline (RGB+NIR), val split, NMS enabled

`checkpoints/mastermodel_mango/best_model.pt`, epoch 68.

| class | AP50 | precision | recall | tp | fp | fn |
|---|---|---|---|---|---|---|
| 0 mango | 0.8595 | 0.1682 | 1.0000 | 18 | 89 | 0 |
| 1 damage | 0.0643 | 0.0232 | **0.6129** | 19 | 799 | 12 |

The master finds 19 of 31 damage instances — recall 0.61 against the RGB-only student's 0.097. Its
failure is precision (799 false positives), not detection.

## Ablation 1 — NIR zeroed

| | mAP50 | mango recall | damage recall | tp | fp |
|---|---|---|---|---|---|
| real NIR | 0.4619 | 1.000 | 0.613 | — | — |
| NIR zeroed | **0.0000** | 0.000 | 0.000 | 0 | 0 |

The model emits nothing at all — not even false positives. Mango, a large object plainly visible in
RGB, also collapses. `StageAttentionFusion` carries a residual (`rgb_seq = rgb_seq + attn_out`), so
RGB features should survive a degraded NIR. They do not. This indicates brittleness to an
out-of-distribution NIR tensor rather than genuine reliance on NIR content, so a second, milder
ablation was required to separate the two.

## Ablation 2 — NIR real but MISPAIRED (decisive)

Each RGB image paired with a different image's NIR (`nir.roll(1, dims=0)`). NIR statistics stay in
distribution; only the RGB-NIR correspondence is destroyed.

| | mAP50 | mango AP50 | damage AP50 | damage recall | damage tp |
|---|---|---|---|---|---|
| correct NIR | 0.4619 | 0.8595 | 0.0643 | 0.6129 | 19 |
| **mispaired NIR** | **0.4850** | **0.9015** | **0.0684** | **0.6129** | **19** |

**The model performs identically — marginally better — with NIR from the wrong image.** Damage
recall and true positives are unchanged to the digit.

## Conclusion

The cross-modal fusion does not use NIR content. The model requires *a* NIR tensor to be present
(ablation 1), but it is indifferent to *which* one (ablation 2). Whatever the master detects, it
detects from RGB.

So the master is in exactly the position the maintainer diagnosed for the student: attempting to
detect NIR-defined damage without reading NIR. Its higher recall over the student is explained by a
larger backbone and 68 epochs against 13, not by the second modality.

This supersedes the earlier ranking. H1 (assigner), H2 (NMS), H3 (argmax decode) and H6
(under-annotation) were each refuted against pre-registered bars. H7 is confirmed by direct
ablation and explains what those could not: the architecture's entire premise — RGB+NIR fusion
teaching an RGB student through distillation — is not operating. The teacher has no NIR knowledge
to distil.

## What this does NOT establish

- It does not identify WHY the fusion is inert (weight collapse in `MultiheadAttention`, the
  residual dominating, a training-time issue, or the frozen `rgb_stem` and never-run Phase 2).
- It does not prove damage is learnable from NIR by a working fusion; it proves the current one is
  not reading it.
- The mispairing used `roll(1)` within batches, so neighbouring images supply the substitute NIR.
  Identical tp and recall make a coincidental match implausible, but a full random shuffle would
  strengthen it.

## Next

Diagnose the fusion itself: inspect attention weight distributions and the magnitude of `attn_out`
relative to the RGB residual it is added to. If `attn_out` is negligible, the residual path
dominates and the NIR branch never learned to contribute. That is inference-only and cheap.
