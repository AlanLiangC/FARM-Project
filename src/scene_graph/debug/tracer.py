"""JSONL pipeline-trace writer for offline debugging."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(value: Any) -> Any:
    """Recursively coerce a Python value to something json.dumps can handle.

    Replaces NaN / +-Inf with the strings "nan" / "inf" / "-inf" so the trace
    stays valid JSON (the standard json module emits literal NaN/Infinity
    tokens, which most parsers reject) while keeping the information.
    Tensors/ndarrays should be converted before reaching this function;
    anything we can't serialize falls back to ``repr(value)``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=lambda x: repr(x))]
    if isinstance(value, Path):
        return str(value)
    try:
        # Numpy scalars / arrays show up here.
        import numpy as np  # noqa: WPS433 — local import keeps this module light

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return _json_safe(float(value))
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass
    return repr(value)


def resolve_trace_path(
    explicit_path: Optional[str] = None,
    *,
    save_path: Optional[Path] = None,
    env_var: str = "SCENE_GRAPH_DEBUG_TRACE_PATH",
    env_dir: str = "SCENE_GRAPH_DEBUG_TRACE_DIR",
) -> Optional[Path]:
    """Pick a trace file path from explicit override, env var, or save_path-derived default.

    Resolution order:
      1. ``explicit_path`` if set and non-empty.
      2. ``$SCENE_GRAPH_DEBUG_TRACE_PATH`` if set.
      3. ``$SCENE_GRAPH_DEBUG_TRACE_DIR`` + scene stem (requires ``save_path``).
      4. ``None`` -> tracing disabled.
    """
    if explicit_path:
        s = str(explicit_path).strip()
        if s and s not in ("''", '""'):
            return Path(s).expanduser()
    env_p = os.environ.get(env_var, "").strip()
    if env_p:
        return Path(env_p).expanduser()
    env_d = os.environ.get(env_dir, "").strip()
    if env_d and save_path is not None:
        save_path = Path(save_path)
        return Path(env_d).expanduser() / f"{save_path.stem}.trace.jsonl"
    return None


class DebugTracer:
    """Append-only JSONL trace writer.

    Lifecycle:
        tracer = DebugTracer.create(path)        # opens file, writes scene_start
        tracer.start_scene(scene_id, config)     # one per .pt; usually called once
        tracer.record_frame(frame_event_dict)    # per batch
        tracer.end_scene(...)                    # final summary line
        tracer.close()                           # flush + close

    All events get a monotonically increasing ``seq`` field and a wall-clock
    ``ts`` (epoch seconds, float). Tensors / ndarrays must be converted by
    the caller; ``_json_safe`` handles the leftovers.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self._seq = 0
        self._t_start: Optional[float] = None
        self._scene_id: Optional[str] = None

    @classmethod
    def create_if_path(cls, path: Optional[Path]) -> Optional["DebugTracer"]:
        if path is None:
            return None
        return cls(path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def scene_id(self) -> Optional[str]:
        return self._scene_id

    def _write(self, event: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._seq += 1
        record = {
            "schema": self.SCHEMA_VERSION,
            "seq": self._seq,
            "ts": time.time(),
        }
        record.update(event)
        try:
            self._fh.write(json.dumps(_json_safe(record), separators=(",", ":")) + "\n")
        except Exception as exc:  # noqa: BLE001 — never let tracing kill the run
            # Last-ditch: emit an error event with repr fallback.
            try:
                self._fh.write(
                    json.dumps(
                        {
                            "schema": self.SCHEMA_VERSION,
                            "seq": self._seq,
                            "ts": time.time(),
                            "event": "trace_error",
                            "error": repr(exc),
                            "raw_event_keys": list(event.keys()),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            except Exception:
                pass

    def start_scene(self, scene_id: str, config: Dict[str, Any]) -> None:
        self._scene_id = scene_id
        self._t_start = time.perf_counter()
        self._write(
            {
                "event": "scene_start",
                "scene_id": scene_id,
                "config": config,
                "pid": os.getpid(),
            }
        )

    def record_frame(self, frame_event: Dict[str, Any]) -> None:
        if not isinstance(frame_event, dict):
            return
        out = dict(frame_event)
        out.setdefault("event", "frame")
        if self._scene_id is not None:
            out.setdefault("scene_id", self._scene_id)
        self._write(out)

    def end_scene(self, summary: Dict[str, Any]) -> None:
        out = dict(summary)
        out["event"] = "scene_end"
        if self._scene_id is not None:
            out.setdefault("scene_id", self._scene_id)
        if self._t_start is not None:
            out.setdefault("scene_duration_s", time.perf_counter() - self._t_start)
        self._write(out)

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        self._fh = None

    def __del__(self) -> None:
        with _suppress():
            self.close()


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
