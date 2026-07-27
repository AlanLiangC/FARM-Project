"""Runtime-path defaults for the ``mapping`` ROS2 package.

These helpers are consumed by launch files, parameter declarations, and node
code. They live inside the ament package (rather than the pure-Python
``scene_graph`` library) so that ``ros2 launch`` — which is executed by the
system Python interpreter — can import them via the colcon install overlay
without needing the venv's site-packages on ``PYTHONPATH``.

Environment variables honored:

* ``SCENE_GRAPH_MAPPING_DATA_DIR`` — absolute root for mapping artefacts.
  Falls back to ``$ROS_HOME/scene_graph/mapping`` (and ``~/.ros`` when
  ``ROS_HOME`` is unset).
"""

from __future__ import annotations

import os
from pathlib import Path

_DATA_DIR_ENV = "SCENE_GRAPH_MAPPING_DATA_DIR"


def _data_root() -> Path:
    raw = os.environ.get(_DATA_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(os.environ.get("ROS_HOME", "~/.ros")).expanduser() / "scene_graph" / "mapping"


def default_scene_state_path() -> str:
    return str(_data_root() / "scene_state.pt")


def default_scene_graph_json_save_path() -> str:
    return str(_data_root() / "scene_graph.json")


def default_scene_graph_snapshot_dir() -> str:
    return str(_data_root() / "snapshots")


def default_storage_image_dir() -> str:
    return str(_data_root() / "image_store")


def default_adaptation_results_dir() -> str:
    return str(Path(default_scene_graph_snapshot_dir()) / "yoloe_adaptation")
