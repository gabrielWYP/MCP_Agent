# Proposal: Pipeline Orchestration Documentation

## Intent

Document the two training pipelines (Annotation-to-Training, Three-Stage Final Training) built incrementally across multiple SDD changes but never tied together at integration level. Close the documentation gap so contributors understand stage ordering, config contracts, CLI entry points, and env prerequisites without reading every script. Surface three config inconsistencies and the class_weights duplication risk for maintainer decision.

## Scope

### In Scope
- New `pipeline-orchestration` spec: stage ordering, config contract, CLI contracts, env prerequisites
- "Training Pipeline" section in `README.md` with flow diagram (Pipeline A & B)
- Docstring enrichment for three orchestrator scripts
- Document three config inconsistencies with recommendation + maintainer-confirmation markers
- Env prerequisites list (GPU for Florence-2, `matriz_homografia_aruco.npy`, Label Studio export)

### Out of Scope
- Fixing config inconsistencies (deferred — recommendation only)
- `class_weights` single-source-of-truth refactor (follow-up)
- Config validation script / CI integration (follow-up)
- Pipeline integration tests (follow-up — depends on broken arch test fix)
- Any code behavior changes (documentation-only)

## Capabilities

### New Capabilities
- `pipeline-orchestration`: Stage ordering, data flow, config contracts, CLI entry points, and env prerequisites for Pipeline A (Annotation-to-Training) and Pipeline B (Three-Stage Final Training).

### Modified Capabilities
- None. Documentation-only; no spec-level behavioral requirements change in `training-loop` or `kd-training`.

## Approach

Approach 1 (explore recommendation): create the `pipeline-orchestration` spec, add a README section with the flow diagram, and enrich orchestrator docstrings. Config inconsistencies are documented (not fixed) with a recommendation:

- **`num_workers`**: align `kd_training.yaml` (4) to master/student (2), OR document KD-specific intent — maintainer confirms.
- **`warmup_epochs`**: keep KD=3 as KD-specific tuning, OR align to 5 — maintainer confirms.
- **`log_interval`**: align KD (10) to master (5) for consistency — maintainer confirms.

Recorded in spec as "Proposal Recommendation — Maintainer Confirmation Required".

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `openspec/specs/pipeline-orchestration/spec.md` | New | New capability spec |
| `openspec/changes/pipeline-orchestration-docs/` | New | Delta + design + tasks |
| `README.md` | Modified | "Training Pipeline" section + flow diagram |
| `scripts/run_training_pipeline.py` | Modified | Docstring enrichment |
| `scripts/run_final_training_pipeline.py` | Modified | Docstring enrichment |
| `src/training/run_artifacts.py` | Modified | Docstring enrichment |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Recommendations stall without maintainer decision | Med | Mark each "Confirmation Required"; fixes deferred |
| Docstring edits alter behavior | Low | Verify `git diff` shows only comments/docstrings |
| Spec drift from future pipeline changes | Med | Archive deltas on next pipeline change |

## Rollback Plan

Documentation-only. Revert the commit (or remove delta + README section + docstrings). No runtime artifacts, migrations, or config edits to undo. New spec folder is deletable with no side effects.

## Dependencies

- None external. See `explore.md`.

## Success Criteria

- [ ] `pipeline-orchestration` spec exists with stage ordering, config contract, CLI contracts, env prerequisites
- [ ] README "Training Pipeline" section added with both flow diagrams
- [ ] Orchestrator scripts have module-level docstrings (inputs/outputs/prerequisites)
- [ ] Three config inconsistencies documented with recommendation + maintainer-confirmation marker
- [ ] `class_weights` duplication listed as deferred follow-up item
- [ ] No behavior changes — `git diff` shows only comments/docstrings/README/spec