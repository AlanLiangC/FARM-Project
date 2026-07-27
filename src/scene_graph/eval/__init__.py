"""Evaluation harnesses for the scene-graph pipeline.

Subpackages add benchmarks (e.g. OpenEQA). Each subpackage is self-contained:
the only required interface is a saved ``scene_state.pt`` produced by the
offline runner (``python -m scene_graph.offline.run --save-path …``).
"""
