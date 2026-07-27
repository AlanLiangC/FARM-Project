from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for jpeg preview export
    cv2 = None

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_image(image: ArrayLike) -> np.ndarray:
    """Convert RGB tensor/array to a contiguous uint8 HWC numpy array."""

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().contiguous().numpy()
    elif isinstance(image, np.ndarray):
        arr = np.array(image, copy=False)
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        # Assume CHW layout -> convert to HWC
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        # Already HWC
        pass
    elif arr.ndim == 2:
        arr = arr[..., None]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

    return np.ascontiguousarray(arr)


def save_image_to_hdf5(image: ArrayLike, path: Path | str, dataset: str = "color") -> None:
    """Persist ``image`` to ``path`` as an HDF5 dataset with lightweight compression."""

    array = _to_numpy_image(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset(
            name=dataset,
            data=array,
            compression="lzf",
            chunks=True,
            shuffle=True,
        )


def load_image_from_hdf5(path: Path | str, dataset: str = "color") -> np.ndarray:
    """Read an HDF5 image previously written by :func:`save_image_to_hdf5`."""

    with h5py.File(str(path), "r") as handle:
        return np.asarray(handle[dataset][...])


def save_image_to_jpeg(
    image: ArrayLike,
    path: Path | str,
    *,
    max_width: int = 640,
    quality: int = 75,
) -> None:
    """Persist ``image`` to ``path`` as a downscaled JPEG preview."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for JPEG preview saving")

    array = _to_numpy_image(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.shape[2] == 1:
        encoded_input = array[:, :, 0]
    else:
        rgb = array[:, :, :3]
        encoded_input = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    width = int(encoded_input.shape[1]) if encoded_input.ndim >= 2 else 0
    target_width = int(max_width)
    if target_width > 0 and width > target_width:
        ratio = float(target_width) / float(width)
        target_height = max(1, int(round(float(encoded_input.shape[0]) * ratio)))
        encoded_input = cv2.resize(encoded_input, (target_width, target_height), interpolation=cv2.INTER_AREA)

    jpeg_quality = int(quality)
    jpeg_quality = max(1, min(100, jpeg_quality))
    ok, encoded = cv2.imencode(".jpg", encoded_input, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError(f"Failed to encode jpeg preview at {path}")
    path.write_bytes(encoded.tobytes())
