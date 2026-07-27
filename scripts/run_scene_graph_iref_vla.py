#!/usr/bin/env python3
"""Reconstruct scene graphs for IRef-VLA HM3D scenes from rendered NPZ frames.

Mirrors :mod:`scripts.run_scene_graph_openeqa` but ingests NPZ frames produced
by ``scripts/render_hm3d_trajectory.py`` instead of ScanNet ``.sens`` files.

For each scene we expect ``<rendered_dir>/<scene_id>/frames_*.npz`` and a
sidecar ``render_meta.json``. Output goes to ``<out_dir>/<scene_id>.pt``.

Use ``--list`` to see which scenes are available locally; pass ``--scene
<id>`` to limit the run; ``--dry-run`` prints the commands without executing.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("run_scene_graph_iref_vla")


def _scene_ids_from_iref_vla(iref_vla_root: Path, multi_room_only: bool) -> List[str]:
    """Lazy import so this script still works if the iref_vla module is buggy."""
    from scene_graph.eval.iref_vla.dataset import (
        list_local_scenes,
        multi_room_scene_ids,
    )

    if multi_room_only:
        return multi_room_scene_ids(dataset_root=iref_vla_root, min_regions=2)
    return list_local_scenes(dataset_root=iref_vla_root)


def _has_rendered_chunks(rendered_dir: Path) -> bool:
    if not rendered_dir.exists() or not rendered_dir.is_dir():
        return False
    for p in rendered_dir.iterdir():
        if p.name.startswith("frames_") and p.suffix == ".npz":
            return True
    return False


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rendered-dir",
        type=Path,
        default=Path(os.environ.get("IREF_VLA_RENDERED_DIR", "/data/iref_vla/rendered_trajectory")),
        help="Directory of <scene_id>/frames_*.npz from scripts/render_hm3d_trajectory.py.",
    )
    p.add_argument(
        "--iref-vla-root",
        type=Path,
        default=None,
        help=(
            "IRef-VLA HM3D root (used only to look up the scene id list). "
            "Defaults to scene_graph.eval.iref_vla.dataset.default_iref_vla_root()."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/out/iref_vla"),
        help="Where to write per-scene scene_state.pt files.",
    )
    p.add_argument("--scene", action="append", default=None,
                   help="Restrict to specific scene ids; may be repeated.")
    p.add_argument("--multi-room-only", action="store_true",
                   help="Only scenes with >=2 regions in IRef-VLA region_result.csv.")
    p.add_argument("--stride", type=int, default=1, help="Take every Nth NPZ frame.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--caption", action="store_true",
                   help="Enable vLLM caption manager (requires `./run.sh vllm`).")
    p.add_argument("--caption-batch-size", type=int, default=10,
                   help="Forwarded to scene_graph.offline.run when captioning.")
    p.add_argument("--viser", action="store_true")
    p.add_argument("--viser-port", type=int, default=None, help="Port for the online Viser UI.")
    p.add_argument(
        "--viser-record-root",
        type=Path,
        default=None,
        help="If set, write offline Viser replay snapshots under <root>/<scene_id>.",
    )
    p.add_argument("--regions", action="store_true", help="Forward --regions to scene_graph.offline.run.")
    p.add_argument("--target-fps", type=float, default=None, help="Forwarded to scene_graph.offline.run.")
    p.add_argument("--drop-when-late", action="store_true", help="Forwarded to scene_graph.offline.run.")
    p.add_argument("--list", action="store_true",
                   help="List which scenes have rendered frames available, then exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip scenes whose .pt already exists.")
    p.add_argument("--logger-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARN", "ERROR"))
    p.add_argument(
        "--debug-trace-dir", type=Path, default=None,
        help=(
            "If set, the offline runner writes a per-scene JSONL pipeline trace to "
            "<debug-trace-dir>/<scene_id>.trace.jsonl. See scripts/inspect_pipeline_trace.py."
        ),
    )
    p.add_argument("--extra-arg", action="append", default=[],
                   help="Pass-through to scene_graph.offline.run; may be repeated.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = _parse_args(argv)

    all_scenes = _scene_ids_from_iref_vla(args.iref_vla_root, args.multi_room_only)
    target_scenes = args.scene if args.scene else all_scenes

    available, missing = [], []
    for sid in target_scenes:
        scene_render_dir = args.rendered_dir / sid
        (available if _has_rendered_chunks(scene_render_dir) else missing).append(sid)

    LOGGER.info(
        "IRef-VLA HM3D scenes (multi_room_only=%s): %d candidates, %d rendered locally, %d missing",
        args.multi_room_only, len(target_scenes), len(available), len(missing),
    )
    if args.list:
        print("# rendered (have frames_*.npz)")
        for sid in available:
            print(sid)
        print("# missing (no rendered frames)")
        for sid in missing:
            print(sid)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    t_start = time.time()
    for i, sid in enumerate(available):
        save_path = args.out_dir / f"{sid}.pt"
        if args.skip_existing and save_path.exists():
            LOGGER.info("[%d/%d] skip %s (exists)", i + 1, len(available), sid)
            continue
        npz_dir = args.rendered_dir / sid
        cmd = [
            sys.executable,
            "-m",
            "scene_graph.offline.run",
            "--source", "npz",
            "--npz-dir", str(npz_dir),
            "--camera", sid,
            "--stride", str(args.stride),
            "--start", str(args.start),
            "--end", str(args.end),
            "--save-path", str(save_path),
            "--logger-level", args.logger_level,
        ]
        if args.caption:
            cmd.append("--caption")
            cmd.extend(["--caption-batch-size", str(max(1, int(args.caption_batch_size)))])
        if args.viser:
            cmd.append("--viser")
        if args.viser_port is not None:
            cmd.extend(["--viser-port", str(args.viser_port)])
        if args.viser_record_root is not None:
            record_dir = args.viser_record_root / sid
            cmd.extend(["--viser-record-dir", str(record_dir)])
        if args.regions:
            cmd.append("--regions")
        if args.target_fps is not None:
            cmd.extend(["--target-fps", str(args.target_fps)])
        if args.drop_when_late:
            cmd.append("--drop-when-late")
        if args.debug_trace_dir is not None:
            args.debug_trace_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--debug-trace-path", str(args.debug_trace_dir / f"{sid}.trace.jsonl")])
        cmd.extend(args.extra_arg)
        LOGGER.info(
            "[%d/%d] %s -> %s\n  %s",
            i + 1, len(available), sid, save_path, " ".join(shlex.quote(c) for c in cmd),
        )
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, env=os.environ.copy()).returncode
        if rc != 0:
            LOGGER.error("Scene %s exited with code %d", sid, rc)
            failed.append(sid)

    LOGGER.info("Done. %.1fs total, %d failures", time.time() - t_start, len(failed))
    if failed:
        LOGGER.warning("Failed scenes: %s", ",".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
