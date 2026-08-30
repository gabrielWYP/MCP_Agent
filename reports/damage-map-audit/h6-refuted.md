# H6 — REFUTED

**Hypothesis:** damage is systematically under-annotated in the training split, teaching the model
that visible damage regions are background.

**Outcome: REFUTED.** Recorded per the maintainer's requirement that negative results be published.

## Method

`data/label_studio_nir/Datasetv2_export.json` is the human Label Studio annotation export. It is
tracked in git (`.gitignore` explicitly un-ignores `data/label_studio_nir/*.json`), covers 186
images, and is authoritative ground truth for the damage class. Comparing it per-image against the
post-reconciliation label files answers H6 directly, with no visual inspection and no training.

Name mapping: the export keys images as `<hash>-mango_nir_<timestamp>.jpg`; label files use
`mango_rgb_<timestamp>.txt`. Matching on the timestamp yields 186 of 187 stems (the unmatched one
is the stem absent from the manifest, which has no human annotation).

## Result

| Comparison | Images | Damage boxes |
|---|---|---|
| Identical count | 165 | — |
| Disk has FEWER than human (damage lost) | **0** | **0** |
| Disk has MORE than human (extra) | 21 | +73 |

**Zero images lost damage.** There is no image where a human annotated damage and the label file
says none. The mechanism H6 proposed — damage present but unlabelled, taught as background — does
not occur in this dataset.

## Why the pre-reconciliation signal looked alarming

Before reconciliation, eight train copies carried zero damage while their twin carried up to 12.
That was real, and it was a genuine defect. It was caused by the split leakage: two copies of the
same image existed, one of them stale. The Q11 resolution rule (keep the copy with the most damage
boxes) recovered the damage-bearing copy in every one of those cases. The maintainer's choice of
that rule, made without visual inspection, produced the correct outcome — measured, not assumed.

The residual rate asymmetry (train 20% damage-free vs val 6% after reconciliation) is therefore a
property of the real data distribution, not evidence of missing labels.

## Separate finding, opposite direction

21 images carry 73 damage boxes MORE than the human export. Only 12 of those 21 were among the
conflict-resolved stems, so this predates and is independent of the Q11 resolution — the two sets
happen to both number 21 by coincidence, overlapping in 12.

Something in the label-generation path adds damage boxes not present in human annotation. This is
over-annotation, not under-annotation, and it is a NEW open question rather than a revival of H6.
It could inflate apparent damage instance counts and inject boxes the image does not support.

## Consequence for the ladder

H6 is closed. H1 (assigner starves small damage boxes of positive anchors) returns as the leading
hypothesis and remains untested. Its test (E2) requires no training.

## Recovery note

Because the human export is versioned, the label files are regenerable from source. The concern
that the destructive reconciliation left no rollback path is resolved: `Datasetv2_export.json`
plus the conversion script reconstructs the damage annotations independently of git history for
`data/annotations/`, which is gitignored.
