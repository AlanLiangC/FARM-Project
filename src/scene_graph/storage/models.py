from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .hdf5 import ArrayLike


@dataclass
class ImageRecord:
    """Bookkeeping entry tracking metadata for each RGB frame.

    Two ways the original RGB can be recovered later (e.g. by the viser eval
    viewer):

    * ``storage_path`` — set when ``image_saving_enabled`` is on; points at a
      JPEG/HDF5 file the ``ImageSaveWorker`` actually wrote.
    * ``source_ref`` — set unconditionally at registration time and points back
      at the *source* data (e.g. ``/data/scans/<scene>/<scene>.sens#frame=42``
      or ``<chunk>.npz#frame=17``). Lets the viewer load images at zero extra
      disk cost when source data is still on disk.

    ``scene_graph.scene_state_io.resolve_image_record`` prefers ``storage_path``
    and falls back to ``source_ref``.
    """

    image_id: int
    pose: torch.Tensor | None
    camera_id: str = ""
    storage_path: str = ""
    source_ref: str = ""


@dataclass
class ImageSaveRequest:
    image: ArrayLike
    path: Path
    fmt: str = "h5"
    jpeg_max_width: int = 640
    jpeg_quality: int = 75
    on_success: Optional[Callable[[Path], None]] = None
