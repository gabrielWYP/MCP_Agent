# Tasks: Testing Bootstrap — Phase 1 (Teacher Architecture Validation)

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~700 (≈510 additions + ≈190 deletions) |
| 400-line budget risk | **High** — exceeds budget by ~75% |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (Foundation) → PR 2 (Model Tests) → PR 3 (Graph Nodes + Cleanup) |
| Delivery strategy | ask-on-risk |
| Chain strategy | feature-branch-chain |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | pytest config + shared fixtures | PR 1 | Base = `feat/testing-bootstrap-p1` (tracker branch). pyproject.toml, requirements.txt, conftest.py |
| 2 | Model shape tests (6 files) | PR 2 | Base = PR 1 branch. All `tests/models/master/` test files |
| 3 | Graph node tests + delete legacy scripts | PR 3 | Base = PR 2 branch. test_graph_nodes.py + delete 3 files |

## Phase 1: Foundation / Infrastructure

- [x] 1.1 Create `pyproject.toml` with `[tool.pytest.ini_options]`: testpaths=`["tests"]`, markers=`["slow", "model", "cpu_only"]`, addopts=`"-v --tb=short"`
- [x] 1.2 Modify `requirements.txt`: add `pytest-cov` and `pytest-mock` to the Testing section
- [x] 1.3 Create `tests/conftest.py`: session fixtures for `torch.manual_seed(42)`, `_force_cpu()` guard, `dummy_rgb` (2,3,640,640), `dummy_nir` (2,1,640,640), individual model fixtures (backbone, fusion, neck, head), `master_model_factory(**kwargs)` with `_cache` dict, and `pytest_configure` to register markers

## Phase 2: Model Shape Tests

- [x] 2.1 Create `tests/models/master/__init__.py` (empty package init)
- [x] 2.2 Create `tests/models/master/test_backbone.py`: 3 tests — shape correctness (channels [96,192,384,768] + spatial [160,80,40,20]), NIR 1ch input acceptance, distinct RGB/NIR stream outputs with fixed B=2
- [x] 2.3 Create `tests/models/master/test_fusion.py`: 2 tests — spatial dimension preservation through CrossModalFusion, small feature map skip-pooling (H≤20,W≤20) with fixed B=2
- [x] 2.4 Create `tests/models/master/test_head.py`: 2 tests — YOLO format output shapes (preds 6ch, cls 2ch, reg 4ch at strides [8,16,32]), decoupled branch non-degenerate outputs with fixed B=2
- [x] 2.5 Create `tests/models/master/test_neck.py`: 1 test — DualFPN produces 4-level pyramid with correct channel dims from fusion outputs
- [x] 2.6 Create `tests/models/master/test_distill_projections.py`: 1 test — projection layer output shapes match KD requirements
- [x] 2.7 Create `tests/models/master/test_master_model.py`: 3 tests — full forward returns dict with 8 keys and correct shapes, `freeze_backbone(freeze_stages=2)` disables stems+stages[0..1] gradients, `count_parameters()` returns positive counts per module; all with fixed B=2

## Phase 3: Graph Node Tests + Cleanup

- [x] 3.1 Create `tests/agent/__init__.py` (empty package init)
- [x] 3.2 Create `tests/agent/test_graph_nodes.py`: 7 node tests (one per graph node in `src/agent/graph_nodes.py`) mocking `check_new_objects` via `mocker.patch`, asserting state mutations; 2 conditional-edge path tests for `check_deployment_decision` return values
- [x] 3.3 Delete `src/models/master/sanity_check.py` (replaced by Phase 2 tests)
- [x] 3.4 Delete `src/models/master/test_arch.py` (replaced by Phase 2 tests)
- [x] 3.5 Delete `test_s3_connection.py` at repo root (replaced by mocked storage test in Phase 2)

## Phase 4: Verification

- [x] 4.1 Run `pytest tests/ -m "not slow"` — all fast tests pass on CPU (34 passed)
- [x] 4.2 Run `pytest tests/` — full suite passes (slow tests included, ~14s total, 36 passed)
- [x] 4.3 Verify no network calls: graph nodes use sys.modules stubs for s3fs/ObjectStorage; model tests use CPU-only synthetic tensors with pretrained=False

## Rollback Boundaries

| Stop Point | State | Safe? |
|------------|-------|-------|
| After PR 1 (Phase 1) | Config exists, no tests yet | ✅ Yes — only config changes |
| After PR 2 (Phase 1+2) | Model tests pass, legacy scripts still present | ✅ Yes — tests are additive |
| After PR 3 (Full Phase 1) | All tests pass, legacy scripts deleted | ✅ Yes — full deliverable |

## Implementation Notes

- **Batch size**: Fixed B=2 across ALL tests (user decision — NOT parametrized). Matches original sanity scripts.
- **Feature branch**: Create `feat/testing-bootstrap-p1` from `main` before starting. PR 1 targets this tracker branch.
- **CPU guard**: `_force_cpu()` must patch `torch.cuda.is_available` to return `False` to prevent accidental GPU usage.
- **Eval mode**: All session fixtures call `model.eval()` before returning. Tests wrap forwards in `torch.no_grad()`.
- **Deterministic**: `torch.manual_seed(42)` set once in session fixture; all tensor generation uses `torch.randn`.
