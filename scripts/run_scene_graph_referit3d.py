#!/usr/bin/env python3
"""Reconstruct scene graphs for every ReferIt3D val∩local scene.

For each ``scene_id`` in (ScanNet v2 val ∩ local scans), look for
``<scans-dir>/<scene_id>/<scene_id>.sens`` and run the offline mapping pipeline
(``python -m scene_graph.offline.run --source sens``). Output goes to
``<out-dir>/<scene_id>.pt`` so the predict phase picks them up by stem.

Run inside the docker container::

    python scripts/run_scene_graph_referit3d.py \
        --scans-dir /data/scans --out-dir /data/out/scannet --stride 1 --caption --skip-existing

Use ``--list`` to see which scenes are/aren't available; ``--scene <id>`` to
limit the run; ``--dry-run`` to print the commands without executing.
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

LOGGER = logging.getLogger("run_scene_graph_referit3d")


def _scene_ids() -> List[str]:
    # Lazy import so the script still works if the dataset module changes.
    from scene_graph.eval.referit3d.dataset import (
        list_local_scenes,
        load_val_scenes,
        utterances_by_scene,
        val_local_subset,
    )

    val_local = sorted(set(load_val_scenes()) & set(list_local_scenes()))
    # Restrict further to scenes that actually have ≥1 utterance in val∩local.
    by_scene = utterances_by_scene(val_local_subset())
    return [s for s in val_local if s in by_scene]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scans-dir", type=Path, default=Path("/data/scans"),
        help="Directory containing ScanNet scans (each subdir holds <scene>.sens). Default: /data/scans.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help=(
            "Where to write per-scene scene_state.pt files. Convention: "
            "/data/out/scannet/<YYYY-MM-DD>-<short-tag>/. Each batch should "
            "land in its own dated dir so different reconstruction configs "
            "stay separated; pass --out-dir explicitly when starting a new batch."
        ),
    )
    p.add_argument("--scene", action="append", default=None, help="Restrict to specific scene ids; may be repeated.")
    p.add_argument(
        "--partial", action="store_true",
        help=(
            "Restrict to the frozen 75-scene partial subset (see "
            "scene_graph.eval.referit3d.partial_scene_ids) for fast iteration. "
            "Combined freely with --scene; both filters apply."
        ),
    )
    p.add_argument("--stride", type=int, default=1, help="(.sens) take every Nth frame (default 1 — full sequence).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--caption", action="store_true", help="Enable vLLM caption manager (requires the VL server up).")
    p.add_argument("--viser", action="store_true")
    p.add_argument("--viser-port", type=int, default=None, help="Port for the online Viser UI.")
    p.add_argument("--target-fps", type=float, default=None, help="Forwarded to scene_graph.offline.run.")
    p.add_argument("--drop-when-late", action="store_true", help="Forwarded to scene_graph.offline.run.")
    p.add_argument("--list", action="store_true", help="Just list which scenes are available and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    p.add_argument("--skip-existing", action="store_true", help="Skip scenes whose .pt already exists.")
    p.add_argument(
        "--require-captions", action="store_true",
        help="With --skip-existing, ALSO re-reconstruct any existing .pt that has no captions.",
    )
    p.add_argument("--logger-level", default="INFO", choices=("DEBUG", "INFO", "WARN", "ERROR"))
    p.add_argument(
        "--debug-trace-dir", type=Path, default=None,
        help=(
            "If set, the offline runner writes a per-scene JSONL pipeline trace to "
            "<debug-trace-dir>/<scene_id>.trace.jsonl. See scripts/inspect_pipeline_trace.py."
        ),
    )
    p.add_argument("--extra-arg", action="append", default=[],
                   help="Pass-through to the offline runner; may be repeated.")
    return p.parse_args(argv)


def _resolve_sens(scans_dir: Path, scene_id: str) -> Optional[Path]:
    candidate = scans_dir / scene_id / f"{scene_id}.sens"
    return candidate if candidate.exists() else None


def _scene_has_captions(pt_path: Path) -> bool:
    """Return True iff the saved state has at least one non-empty caption."""
    if not pt_path.exists():
        return False
    try:
        import torch  # noqa: WPS433 — lazy
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        state = payload.get("state", payload) if isinstance(payload, dict) else {}
        captions = state.get("object_caption", []) or []
        return any(str(c or "").strip() for c in captions)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not inspect %s: %s", pt_path, exc)
        return False


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = _parse_args(argv)

    all_scenes = _scene_ids()
    target_scenes = args.scene if args.scene else all_scenes
    if args.partial:
        from scene_graph.eval.referit3d.dataset import partial_scene_ids
        partial_set = set(partial_scene_ids())
        target_scenes = [s for s in target_scenes if s in partial_set]
        LOGGER.info("--partial: restricted to %d / %d scenes", len(target_scenes), len(all_scenes))

    available, missing = [], []
    for sid in target_scenes:
        sens = _resolve_sens(args.scans_dir, sid)
        (available if sens is not None else missing).append(sid)

    LOGGER.info(
        "ReferIt3D val∩local scenes (with ≥1 utterance): %d total, %d available locally, %d missing",
        len(all_scenes), len(available), len(missing),
    )

    if args.list:
        print("# available")
        for sid in available:
            tag = ""
            pt = args.out_dir / f"{sid}.pt"
            if pt.exists():
                tag = " (RECONSTRUCTED, captions)" if _scene_has_captions(pt) else " (RECONSTRUCTED, no captions)"
            print(f"{sid}{tag}")
        print("# missing")
        for sid in missing:
            print(sid)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    skipped: List[str] = []
    t_start = time.time()
    for i, sid in enumerate(available, start=1):
        save_path = args.out_dir / f"{sid}.pt"
        if args.skip_existing and save_path.exists():
            if args.require_captions and not _scene_has_captions(save_path):
                LOGGER.info("[%d/%d] %s exists but has no captions — re-reconstructing", i, len(available), sid)
            else:
                LOGGER.info("[%d/%d] skip %s (exists)", i, len(available), sid)
                skipped.append(sid)
                continue
        sens = _resolve_sens(args.scans_dir, sid)
        cmd = [
            sys.executable, "-m", "scene_graph.offline.run",
            "--source", "sens", "--sens-path", str(sens),
            "--camera", sid, "--stride", str(args.stride),
            "--start", str(args.start), "--end", str(args.end),
            "--save-path", str(save_path),
            "--logger-level", args.logger_level,
        ]
        if args.caption:
            cmd.append("--caption")
        if args.viser:
            cmd.append("--viser")
        if args.viser_port is not None:
            cmd.extend(["--viser-port", str(args.viser_port)])
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
            i, len(available), sid, save_path, " ".join(shlex.quote(c) for c in cmd),
        )
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, env=os.environ.copy()).returncode
        if rc != 0:
            LOGGER.error("scene %s exited with code %d", sid, rc)
            failed.append(sid)

    LOGGER.info(
        "Done. %.1fs total. ran=%d, skipped=%d, failed=%d",
        time.time() - t_start, len(available) - len(skipped) - len(failed), len(skipped), len(failed),
    )
    if failed:
        LOGGER.warning("Failed scenes: %s", ",".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
