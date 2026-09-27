"""Mobile export of the RGB-only StudentModel.

Modules:
    layout:         Output tensor layout shared by the export wrapper and
                    the reference decode (anchor order, flatten/unflatten).
    student_export: Export wrapper, checkpoint loading, metadata sidecar,
                    and the LiteRT (ai-edge-torch / litert-torch) converter.
    reference:      Python reference preprocessing + decode + NMS operating
                    on the RAW exported outputs — the spec the app mirrors.
    parity:         PyTorch-vs-exported-model comparison helpers.

Nothing here imports `litert_torch` or `ai_edge_litert` at module import
time, so the main training venv can import and test this package without
the export toolchain installed.
"""
