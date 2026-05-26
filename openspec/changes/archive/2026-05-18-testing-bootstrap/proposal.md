# Proposal: Testing Bootstrap

## Intent

Zero test coverage on 26 Python files. The orchestrator core (7 pure LangGraph nodes), multimodal model, and ML pipeline are entirely untested. This change bootstraps a testing foundation in stages, starting with the highest-value, lowest-effort targets and progressively adding CI enforcement and quality tooling.

## Scope

### In Scope
- Phase 1: pytest + pytest-mock setup; unit tests for all 7 graph nodes; model shape tests converted from manual sanity scripts; `conftest.py` with shared `MLOpsState` fixture
- Phase 2: GitHub Actions CI workflow; `pytest-cov` integration; dataset tests with synthetic fixtures; coverage gate at 40%
- Phase 3: ruff + mypy configuration; annotation pipeline tests; coverage target 60%+
- Feature branch strategy for apply phase (not main)
- Delete manual sanity check scripts after conversion to pytest

### Out of Scope
- E2E/integration tests requiring real S3 or GPU
- FastAPI serving tests (not yet implemented)
- Performance benchmarks
- Notebooks testing
- Replacing hardcoded simulation values in graph nodes (that's a separate refactor)

## Capabilities

### New Capabilities
- `testing-unit`: Unit test suite — graph nodes, model shapes, dataset transforms, annotation pipeline, logger, config loading
- `testing-ci`: CI pipeline — GitHub Actions workflow, coverage enforcement, quality gates
- `testing-quality`: Code quality tooling — ruff linting/formatting, mypy type checking, configuration

### Modified Capabilities
- None (no existing specs to modify)

## Approach

**Staged bootstrap** per exploration recommendation:

| Phase | Focus | Test Files | CI? | Coverage |
|-------|-------|-----------|-----|----------|
| 1 | Graph nodes + model shapes | `tests/agent/`, `tests/models/` | No | Baseline |
| 2 | CI + dataset tests | `.github/workflows/ci.yml`, `tests/ml_pipeline/` | Yes | ≥40% |
| 3 | Quality tools + annotation | `tests/annotation/`, ruff/mypy config | Yes | ≥60% |

Test structure mirrors `src/`: `tests/agent/`, `tests/models/`, `tests/ml_pipeline/`, `tests/annotation/`, `tests/utils/`. Each module gets its own test file. Shared fixtures in `conftest.py` at each level.

Phase 1 uses feature branch `feat/testing-bootstrap-p1`. Later phases chain from it.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `tests/` | New | Entire test suite created from scratch |
| `src/agent/graph_nodes.py` | Modified | Extract hardcoded values for testability |
| `src/models/master/sanity_check.py` | Removed | Replaced by pytest tests |
| `src/models/master/test_arch.py` | Removed | Replaced by pytest tests |
| `test_s3_connection.py` (root) | Removed | Replaced by mocked storage test |
| `requirements.txt` | Modified | Add pytest-cov, pytest-mock, ruff, mypy |
| `.github/workflows/ci.yml` | New | Phase 2 CI pipeline |
| `openspec/config.yaml` | Modified | Update testing section after each phase |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| pytest uninstalled; venv dependency conflicts | High | Create venv first; pin PyTorch CPU-only for CI |
| GPU deps cause CI failures | Medium | All model tests run CPU with `pretrained=False`; no weight downloads |
| S3 mock incompleteness → real S3 calls | Medium | Mock entire `s3fs.S3FileSystem` including constructor |
| Duplicate `MLOpsState` across modules | Medium | Import from single canonical `src/agent/state.py` in all tests |
| Hardcoded simulation values break later | Low | Document as known debt; separate refactor issue |
| No test data directory | Medium | Generate synthetic images programmatically in fixtures |

## Rollback Plan

Each phase is an independent commit on a feature branch. Revert per phase:
- Phase 1: Delete `tests/agent/` and `tests/models/`; restore sanity scripts from git
- Phase 2: Delete `.github/workflows/ci.yml`; remove `pytest-cov` from requirements
- Phase 3: Remove `ruff`/`mypy` config; delete `tests/annotation/`

## Dependencies

- Python 3.13 venv must be created and activated before pytest install
- pytest, pytest-mock, pytest-cov, pytest-asyncio (all in requirements.txt)

## Success Criteria

- [ ] Phase 1: All 7 graph nodes have passing unit tests; model shape tests assert correct tensor dimensions
- [ ] Phase 2: GitHub Actions CI runs green on push; coverage ≥40%
- [ ] Phase 3: ruff + mypy configured; annotation pipeline tested; coverage ≥60%
- [ ] Manual sanity check scripts deleted; equivalent coverage in pytest
- [ ] `openspec/config.yaml` testing section updated with actual runner, coverage, and quality tool info