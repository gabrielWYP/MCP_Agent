## Exploration: testing-bootstrap

### Current State

The project is a Python 3.13 MLOps pipeline for mango ripeness/damage detection with zero test coverage:
- **No tests exist** — `tests/__init__.py` is empty, no test files anywhere in the codebase
- **pytest declared but not installed** — `requirements.txt` lists `pytest` and `pytest-asyncio` but `pip install -r requirements.txt` has never been run (no venv exists)
- **No CI/CD** — no GitHub Actions, no tox.ini, no Makefile
- **No quality tools** — no linter, formatter, or type checker configured
- **Two manual architecture sanity check scripts** exist in `src/models/master/` (`sanity_check.py`, `test_arch.py`) that run forward passes with shape assertions — these are NOT pytest tests, they are scripts
- **One manual S3 connection test** (`test_s3_connection.py`) at project root — also not a test
- **Strict TDD is disabled** in `openspec/config.yaml`

The codebase has 26 Python files across 6 domains organized in clean packages. Key architectural patterns:

| Pattern | Where | Testability |
|---------|-------|------------|
| LangGraph 7-node StateGraph | `src/agent/graph_nodes.py` | **High** — pure functions `(state) -> state` |
| 2-phase PyTorch training pipeline | `src/ml_pipeline/training/process.py` | Medium — GPU/heavy compute |
| Multimodal model (ConvNeXtV2, cross-attention, YOLO head) | `src/models/master/` | Medium — shape checks + forward pass |
| NIR image segmentation + homography | `src/annotation/` | Medium — image processing |
| S3/OCI object storage | `src/utils/ObjectStorage.py` | Low without mocks |
| Thread-safe singleton logger | `src/utils/logger.py` | High |

### Affected Areas

- `src/agent/graph_nodes.py` — 7 pure node functions, the HIGHEST priority for testing. Every node transforms state identically each run given the same inputs. Conditional edge `check_deployment_decision` needs both path tests. Current code has hardcoded simulated values (e.g., `quality_score = 8.5`) that need to be extracted for testability.
- `src/agent/state.py` — `MLOpsState` TypedDict with 9 fields. Simple but ensures test fixtures are correct. No logic to test here.
- `src/models/master/` — 8 Python files with ~1,200 lines. Two manual sanity check scripts already exist and can be converted to pytest. Every module (backbone, fusion, neck, head) can be tested independently with random tensors.
- `src/ml_pipeline/training/dataset.py` — `MangoRipeness` Dataset and `build_dataloaders`. Needs a tiny synthetic dataset fixture (can be generated on-the-fly). Transforms logic (`get_transforms`, `compute_nir_stats`) is independently testable.
- `src/ml_pipeline/training/process.py` — 2-phase fine-tuning with 352 lines. `_train_epoch`, `_eval_epoch`, `_build_scheduler` are pure-ish helpers. Full `process()` is integration-level.
- `src/ml_pipeline/predict/process.py` — Inference functions. `prepare_image`, `predict_single` testable with synthetic images or fixture checkpoints.
- `src/annotation/nir_segmenter.py` — NIR region detection. Test with synthetic grayscale images (dark patches = damage). Parameter sensitivity should be validated.
- `src/annotation/bbox_projector.py` — Homography projection. Test with identity matrix (no distortion) and known matrices with expected transforms.
- `src/annotation/annotation_generator.py` — Format exporters (COCO, YOLO, Label Studio). Test output JSON structure, coordinate normalization, and confidence filtering.
- `src/utils/logger.py` — Singleton pattern with thread safety. Test that multiple `__new__()` calls return the same instance.
- `src/storage_logic/logic.py` — `check_new_objects()` wraps S3 call. Test with mock for both True/False paths.
- `src/variables/configvariables.py` — Env loading. Test with temp `.env` fixture.
- `requirements.txt` — Needs `pytest-cov` added. `pytest` and `pytest-asyncio` already listed but uninstalled.
- `tests/` — Currently empty. Entire test suite must be bootstrapped from zero.

### Approaches

#### 1. **Minimal Viable Bootstrap** — pytest only, graph nodes first

Install pytest + pytest-mock. Create `conftest.py` with a shared `MLOpsState` fixture. Write unit tests for all 7 graph nodes first (highest value, lowest effort, pure functions). Convert `sanity_check.py` to pytest. Skip CI, linters, and coverage tooling for now.

- **Pros**: Immediate test coverage on the orchestrator core. Low setup cost (~30 min). Teaches the team the test pattern on the simplest module.
- **Cons**: Leaves model tests as manual scripts. No coverage measurement. No CI enforcement. Delays the quality tooling problem.
- **Effort**: Low

#### 2. **Full Bootstrap** — pytest + coverage + CI + quality tools

Install pytest, pytest-cov, pytest-mock. Add `.github/workflows/ci.yml` running tests on push. Add `ruff` for linting + formatting, `mypy` for type checking. Write tests for ALL modules. Create shared fixtures for each domain. Set coverage threshold at 60%.

- **Pros**: Complete testing infrastructure. CI enforces quality. Professional-grade setup.
- **Cons**: High effort for a team without testing experience. Risk of overwhelming the team. Many tests will be low-value (e.g., testing model shapes that rarely change). Quality tool configuration is an orthogonal concern that mixes with testing.
- **Effort**: High

#### 3. **Staged Bootstrap** — graph nodes → models → CI → everything else

Phase 1: Install pytest + pytest-mock. Write graph node tests + models shape tests. No CI yet.
Phase 2: Add `.github/workflows/ci.yml`. Add `pytest-cov`. Add dataset tests.
Phase 3: Add quality tools (ruff, mypy). Add annotation pipeline tests. Reach 60%+ coverage.

- **Pros**: Progressive, not overwhelming. Each phase delivers value independently. CI is added AFTER tests exist (so the first CI run is green, not red). Models the SDD iterative approach perfectly.
- **Cons**: Requires discipline to not skip phases. Takes real time to reach Phase 3.
- **Effort**: Medium

### Recommendation

**Approach 3 — Staged Bootstrap** is the correct strategy for this project.

**Why staged over full bootstrap (option 2):**
- The team has NO existing test patterns or conventions. Jumping to full CI + linters + coverage gates all at once is the classic "process overload" mistake. They need to learn ONE thing at a time.
- Graph nodes are the most valuable AND easiest to test — that's the cleanest entry point.

**Why staged over minimal (option 1):**
- Option 1 leaves too much undone. Model architecture tests are essential because the multimodal pipeline (RGB+NIR, cross-attention, YOLO head) is where the project's technical risk lives.
- CI is not optional — without it, tests rot. You MUST have CI before you think you're "done." Phase 2 CI is the minimum viable enforcement.

**Phase breakdown rationale:**

| Phase | What | Why First |
|-------|------|-----------|
| 1 | Graph nodes + models shape tests | Graph nodes: pure functions, fast, high coverage/low effort. Models: already have manual sanity scripts — convert them. Both teach the team the pytest pattern. |
| 2 | CI + coverage + dataset tests | CI enforces that Phase 1 tests stay green. Dataset tests require synthetic data fixture — slightly more complex but critical for the training pipeline. |
| 3 | Quality tools + annotation tests | Linting/typing after testing is correct order — enforce consistency AFTER you have confidence in the code. Annotation tests involve image processing, more complex setup. |

**Graph-based testing strategy specifics:**

For the 7 LangGraph nodes, the test pattern is:
```python
def test_check_for_new_data_has_data(mock_check_new_objects, initial_state):
    mock_check_new_objects.return_value = True
    result = check_for_new_data(initial_state)
    assert result["new_data_path"] != ""
    assert result["failure_analysis_report"] == ""

def test_check_for_new_data_no_data(mock_check_new_objects, initial_state):
    mock_check_new_objects.return_value = False
    result = check_for_new_data(initial_state)
    assert "No se detectaron" in result["failure_analysis_report"]
```

For the conditional edge `check_deployment_decision`:
```python
def test_check_deployment_approve(state):
    state["deployment_decision"] = "APPROVE"
    assert check_deployment_decision(state) == "deploy_to_production"

def test_check_deployment_reject(state):
    state["deployment_decision"] = "REJECT"
    assert check_deployment_decision(state) == "alert_human"
```

For the multimodal model (`src/models/master/`), tests should cover:
- Each submodule independently: backbone, fusion, neck, head
- Shape assertions match expected channels per stage: `[96, 192, 384, 768]`
- Full MasterModel end-to-end forward pass
- Freeze/unfreeze behavior
- Distillation feature accessibility
- CPU compatibility (the models must work on CPU for CI)

**Mocking strategy:**

| Dependency | Mock approach |
|------------|--------------|
| `check_new_objects()` | `pytest-mock` monkeypatch on function reference |
| S3/OCI storage | Mock `s3fs.S3FileSystem` — avoid real S3 calls entirely |
| Gemini/LLM API calls | Mock with simulated responses; the nodes already have hardcoded values used as placeholders |
| PyTorch | Use real PyTorch (CPU). Random tensors for shape tests are fast and deterministic |
| Homography matrix | Use identity matrix (`np.eye(3)`) for projection tests — predictable, no distortion |

### Ready for Proposal

**Yes.** The exploration is complete and actionable. The next phase (sdd-propose) should define:

1. Exact scope: Phase 1 only (graph nodes + model shape tests), or all 3 phases
2. Test file structure convention: mirror `src/` under `tests/` or flat?
3. Coverage threshold target per phase
4. Whether `sanity_check.py` and `test_arch.py` should be deleted (converted to pytest) or kept as-is
5. Delivery strategy for chained PRs if full bootstrap is chosen (high line count risk)

### Risks

- **pytest not installed**: `pip install` requires venv setup first. If the environment doesn't exist, installation may fail with dependency conflicts (PyTorch, timm, opencv have large dependency chains).
- **GPU dependency**: Model tests on CPU require `pretrained=False` to avoid downloading weights. The sanity checks already handle this, but new test authors may accidentally trigger 200MB+ downloads in CI.
- **S3 mocking fragility**: `s3fs.S3FileSystem` has a complex interface. The mock must cover `ls()`, `exists()`, and the constructor — if any part is unmocked, tests will try to connect to real S3.
- **Duplicate state definitions**: `MLOpsState` is defined in BOTH `src/agent/state.py` and `src/utils/ObjectStorage.py`. Tests importing the wrong one will pass on one definition and fail on the other when fields diverge.
- **Hardcoded simulation values in graph_nodes.py**: Nodes 2-4 use placeholder values (e.g., `quality_score = 8.5`, mock metrics). Tests written against these hardcoded values will break when real API integration replaces them.
- **No test data**: The project has no `data/` directory with sample images. Dataset tests will need to generate synthetic images programmatically or download a small fixture — both strategies need to be decided.
