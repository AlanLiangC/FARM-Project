"""Dataset factory — resolves a DatasetConfig (or raw dict) to a concrete loader."""

from __future__ import annotations

import dataclasses
import glob
import os
from typing import Union

from scene_graph.datasets.interfaces import BaseDataset


def get_dataset(config) -> BaseDataset:
    """Instantiate the correct dataset loader from a config.

    Accepts either a :class:`~scene_graph.config.DatasetConfig` instance or a
    plain ``dict`` (legacy usage).  The ``name`` field drives dispatch:

    * ``"Replica"`` — :class:`~scene_graph.datasets.replica.ReplicaDataset`
      (auto-promoted to :class:`~scene_graph.datasets.npz.NPZDataset` when NPZ
      archives are found in the sequence directory)
    * ``"ScanNet"`` — :class:`~scene_graph.datasets.scannet.ScannetDataset`
    * ``"Habitat"`` / ``"HabitatSim"`` — :class:`~scene_graph.datasets.replica.HabitatDataset`
    * ``"Isaac"`` / ``"IsaacSim"`` — :class:`~scene_graph.datasets.replica.IsaacDataset`
    * ``"npz"`` / ``"NPZ"`` — :class:`~scene_graph.datasets.npz.NPZDataset`
    """
    # Normalise to a flat dict so both typed and legacy callers work identically.
    if hasattr(config, "__dataclass_fields__"):
        cfg = dataclasses.asdict(config)
    else:
        cfg = dict(config)

    name: str = cfg.get("name", "Replica")

    if name == "Replica":
        base_dir = cfg.get("base_dir", "")
        sequence = cfg.get("sequence", "")
        seq_dir = os.path.join(base_dir, sequence)
        if glob.glob(os.path.join(seq_dir, "*.npz")):
            from scene_graph.datasets.npz import NPZDataset
            return NPZDataset(**cfg)
        from scene_graph.datasets.replica import ReplicaDataset
        return ReplicaDataset(**cfg)

    if name == "ScanNet":
        from scene_graph.datasets.scannet import ScannetDataset
        return ScannetDataset(**cfg)

    if name in ("Habitat", "HabitatSim"):
        from scene_graph.datasets.replica import HabitatDataset
        return HabitatDataset(**cfg)

    if name in ("Isaac", "IsaacSim"):
        from scene_graph.datasets.replica import IsaacDataset
        return IsaacDataset(**cfg)

    if name.lower() == "npz":
        from scene_graph.datasets.npz import NPZDataset
        return NPZDataset(**cfg)

    raise ValueError(
        f"Unknown dataset name '{name}'. "
        "Expected one of: Replica, ScanNet, Habitat, HabitatSim, Isaac, IsaacSim, npz."
    )
