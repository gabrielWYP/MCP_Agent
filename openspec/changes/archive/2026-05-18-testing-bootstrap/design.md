# Design: Testing Bootstrap — Phase 1 (Teacher Architecture Validation)

## Technical Approach

Phase 1 bootstraps pytest for CPU-only model shape validation and graph-node unit tests. Test directory mirrors `src/`: `tests/models/master/` per submodule, `tests/agent/` for graph nodes. Session-scoped `conftest.py` caches `MasterModel(pretrained_backbone=False)` on CPU (~8s build, amortized). Existing `sanity_check.py` and `test_arch.py` are converted one-to-one into parametrized pytest tests, then deleted. Deterministic seeds and CPU-only guards ensure CI safety.

## Architecture Decisions

### Decision: Test directory mirrors src/

**Choice**: `tests/models/master/test_backbone.py` layout
**Alternatives**: Flat (`tests/test_backbone.py`) — name clash risk at scale; Package re-exports — over-engineering
**Rationale**: 26 files across 6 domains will grow. Mirroring avoids collisions and matches the proposal.

### Decision: Session-scoped fixture + factory pattern

**Choice**: Cheap submodules get direct `scope="session"` fixtures. MasterModel gets a factory `_factory(**kwargs)` with `_cache` dict keyed by kwargs.
**Alternatives**: Function scope (120s total rebuild); direct session fixture for MasterModel (can't test freeze variants)
**Rationale**: Backbone/fusion/neck/head are ~0.5s each — session scope fine. MasterModel is ~8s and needs config variants (freeze_stages), so a cached factory avoids rebuilds while supporting parameterized tests.

### Decision: Pytest markers

**Choice**: `@pytest.mark.slow` (>5s), `@pytest.mark.model` (PyTorch), `@pytest.mark.cpu_only` — registered in `conftest.py` `pytest_configure`.
**Rationale**: Phase 2 CI can filter `-m "not slow"`. Machine-free CI skips `-m "not model"`.

### Decision: Delete sanity scripts after conversion

**Choice**: Remove `sanity_check.py` and `test_arch.py`; their assertions become pytest tests.
**Alternatives**: Keep alongside (dual maintenance); import from scripts (runner pattern doesn't exist)
**Rationale**: Scripts are print-based with no CI-usable assertions. Single source of truth.

### Decision: S3 mocking deferred to Phase 2

**Choice**: Graph node tests mock only `check_new_objects` via `mocker.patch`. Full `s3fs.S3FileSystem` mocking is Phase 2 scope.
**Rationale**: Phase 1 tests no external services. Only `graph_nodes.py` imports S3 — mock at import site.

### Decision: Test isolation via eval mode + no_grad

**Choice**: All session fixtures call `model.eval()`. Tests wrap forwards in `torch.no_grad()`. Freeze tests use fresh model copies or factory variants.
**Rationale**: Shared weights are read-only in eval mode. No test mutates session state.

### Decision: pyproject.toml for pytest config

**Choice**: Create `pyproject.toml` with `[tool.pytest.ini_options]` for testpaths, markers, addopts.
**Rationale**: No existing pyproject.toml. pytest.ini is deprecated per pytest docs. Config centralizes markers.

## Data Flow

```
conftest.py (session)
  ├── torch.manual_seed(42)
  ├── _force_cpu() — patches torch.cuda.is_available
  ├── dummy_rgb  → (2, 3, 640, 640)
  ├── dummy_nir  → (2, 1, 640, 640)
  ├── backbone/fusion/neck/head → direct session fixtures
  └── master_model_factory(**kwargs) → cached MasterModel

tests/models/master/          tests/agent/
  test_backbone.py             test_graph_nodes.py
  test_fusion.py
  test_neck.py
  test_head.py
  test_distill_projections.py
  test_master_model.py
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `tests/conftest.py` | Create | Session fixtures: seed, tensors, model factories, markers, CPU guard |
| `tests/models/master/__init__.py` | Create | Package init |
| `tests/models/master/test_backbone.py` | Create | 3 scenarios: shape correctness, NIR 1ch input, distinct streams |
| `tests/models/master/test_fusion.py` | Create | 2 scenarios: spatial preservation, small-map skip pooling |
| `tests/models/master/test_head.py` | Create | 2 scenarios: YOLO format shapes, decoupled non-degenerate outputs |
| `tests/models/master/test_neck.py` | Create | DualFPN pyramid shape tests |
| `tests/models/master/test_distill_projections.py` | Create | Projection layer shape tests |
| `tests/models/master/test_master_model.py` | Create | 3 scenarios: full forward, freeze backbone, param counts |
| `tests/agent/__init__.py` | Create | Package init |
| `tests/agent/test_graph_nodes.py` | Create | 7 node tests + 2 conditional-edge paths |
| `requirements.txt` | Modify | Add `pytest-cov`, `pytest-mock` |
| `pyproject.toml` | Create | pytest ini_options: testpaths, markers, addopts |
| `src/models/master/sanity_check.py` | Delete | Replaced by pytest |
| `src/models/master/test_arch.py` | Delete | Replaced by pytest |

## Testing Strategy

| Layer | What | Approach |
|-------|------|----------|
| Unit | Backbone shapes (3 scenarios) | Parametrized shape assertions on session fixture output |
| Unit | Fusion preservation (2 scenarios) | Forward backbone features through fusion; assert shapes |
| Unit | Head output format (2 scenarios) | FPN-sized tensors; assert channel dims |
| Unit | Neck pyramid | Forward fusion outputs; assert 4-level pyramid |
| Unit | MasterModel forward + freeze + params | Factory fixture; assert 8 keys and gradient control |
| Unit | Graph nodes state transforms | Mock `check_new_objects`; assert state mutations per node |
| Unit | Conditional edge | Assert `check_deployment_decision` returns correct strings |

## Migration / Rollback

No data migration. Rollback: `git revert` restores deleted scripts and removes test files.

## Open Questions

- [ ] Should `pyproject.toml` host ruff/mypy placeholders for Phase 3, or add later?
- [ ] Batch size for tests: fixed B=2 (matching sanity scripts) or parametrized [1, 2]?