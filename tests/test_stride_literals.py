"""Repo guard: no hardcoded `[8, 16, 32]` / `(8, 16, 32)` stride literal in
`src/` or `scripts/` (fusion-redesign design.md §5).

Before the stride resolver (`src/training/strides.py`), `[8, 16, 32]` was a
literal duplicated at eight call sites; two of them failed *silently* on a
4-level checkpoint (`assigner_level_ranges` too short; `ProjectionLayers`
zip-truncation). Deleting the eight sites once fixes the current bugs; this
test is what stops the ninth site from being reintroduced by the next
author who reaches for a quick literal instead of `resolve_active_strides`
/ `STUDENT_STRIDES`.

Implementation note: this scans the AST for `List`/`Tuple` nodes whose
elements evaluate to exactly `(8, 16, 32)`, not raw source text — a text
scan would also flag every comment and docstring that *explains* this guard
(including this file and every fusion-redesign docstring citing the old
literal), which would make the guard unmaintainable. An AST list/tuple
literal is the only pattern that can actually reach a `strides=` parameter
or a decode/loss call as a real value.
"""

import ast
from pathlib import Path

# The single legitimate definition of the student's fixed 3-level strides
# (fusion-redesign D-2: the student is out of scope and always uses
# `[8, 16, 32]`). Every other file must import `STUDENT_STRIDES` from here
# rather than re-declaring the literal.
EXEMPT_FILES = {
    Path("src/training/strides.py"),
}

TARGET_STRIDES = (8, 16, 32)
SCANNED_ROOTS = ("src", "scripts")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _find_literal_lines(path: Path) -> list[int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            try:
                values = tuple(ast.literal_eval(node))
            except (ValueError, TypeError):
                continue
            if values == TARGET_STRIDES:
                hits.append(node.lineno)
    return hits


def test_no_hardcoded_8_16_32_stride_literal_outside_strides_module():
    root = _repo_root()
    violations: dict[str, list[int]] = {}

    for scanned_root in SCANNED_ROOTS:
        for path in (root / scanned_root).rglob("*.py"):
            relative = path.relative_to(root)
            if relative in EXEMPT_FILES:
                continue
            hits = _find_literal_lines(path)
            if hits:
                violations[str(relative)] = hits

    assert not violations, (
        "Hardcoded [8, 16, 32] / (8, 16, 32) stride literal(s) found outside "
        f"src/training/strides.py: {violations}. Use "
        "src.training.strides.resolve_active_strides(config) or import "
        "STUDENT_STRIDES instead of re-declaring the literal."
    )
