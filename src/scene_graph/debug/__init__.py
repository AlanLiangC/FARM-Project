"""Per-frame structured tracing for the streaming mapping pipeline.

Activated via the ``debug_trace_path`` ROS parameter (``--debug-trace-path``
on ``scene_graph.offline.run``) or via the ``SCENE_GRAPH_DEBUG_TRACE_PATH``
environment variable. Writes JSONL where each line is a structured event
(scene_start, frame, scene_end). Use ``scripts/inspect_pipeline_trace.py``
to summarize.
"""

from .tracer import DebugTracer, resolve_trace_path

__all__ = ["DebugTracer", "resolve_trace_path"]
