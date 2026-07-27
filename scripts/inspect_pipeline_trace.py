#!/usr/bin/env python3
"""Summarize one or more pipeline-trace JSONL files produced by --debug-trace-path.

Reads JSONL files written by ``scene_graph.debug.tracer.DebugTracer``
(activated via ``--debug-trace-path`` on ``python -m scene_graph.offline.run``,
or the ``debug_trace_path`` ROS parameter online) and prints a human-readable
summary covering:

  - per-scene counts (frames, total objects in/out, active count)
  - segmentation funnel (raw → after each filter → matched/new/merged)
  - aggregate filter drop counts and per-step totals
  - neighbor distance distribution near the threshold
  - Gaussian numerical-health red flags (NaN / Inf / huge jumps / ill-conditioned)
  - voxel-cloud level histogram + objects pegged at the cap
  - frames with the biggest changes (top-K by new objects, merges, jumps)

Usage:

    # one scene
    python scripts/inspect_pipeline_trace.py /data/out/scene0011_00.trace.jsonl

    # a directory of traces
    python scripts/inspect_pipeline_trace.py /data/out/traces/

    # narrow to a specific event field
    python scripts/inspect_pipeline_trace.py <path> --top-merges 10 --top-jumps 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_trace_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
        return
    if p.is_dir():
        for q in sorted(p.rglob("*.trace.jsonl")):
            yield q
        return
    raise SystemExit(f"Path does not exist: {p}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{ln} bad JSON: {exc}", file=sys.stderr)
    return out


def _maybe(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _fmt(x: Any, w: int = 8) -> str:
    if x is None:
        return f"{'—':>{w}}"
    if isinstance(x, float):
        if math.isnan(x):
            return f"{'nan':>{w}}"
        if math.isinf(x):
            return f"{('inf' if x > 0 else '-inf'):>{w}}"
        return f"{x:>{w}.3f}"
    return f"{x:>{w}}"


def _summarize_scene(path: Path) -> Dict[str, Any]:
    events = _read_jsonl(path)
    if not events:
        return {"path": str(path), "error": "empty"}
    scene_start = next((e for e in events if e.get("event") == "scene_start"), None)
    scene_end = next((e for e in events if e.get("event") == "scene_end"), None)
    frames = [e for e in events if e.get("event") == "frame"]
    n_frames = len(frames)

    # ── Filter funnel totals
    filter_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n_in": 0, "n_out": 0, "n_dropped": 0, "frames_active": 0})
    raw_in_total = 0
    final_out_total = 0
    for f in frames:
        steps = _maybe(f, "filtering.steps", []) or []
        if steps:
            raw_in_total += int(steps[0].get("n_in", 0) or 0)
            final_out_total += int(steps[-1].get("n_out", 0) or 0)
        for st in steps:
            name = st.get("name", "?")
            filter_totals[name]["n_in"] += int(st.get("n_in", 0) or 0)
            filter_totals[name]["n_out"] += int(st.get("n_out", 0) or 0)
            filter_totals[name]["n_dropped"] += int(st.get("n_dropped", 0) or 0)
            filter_totals[name]["frames_active"] += 1

    # ── Correspondence totals
    n_new_dets_total = 0
    n_matched_dets_total = 0
    n_objects_merged_total = 0
    n_new_objects_total = 0
    n_far_merges_blocked_total = 0
    for f in frames:
        c = f.get("correspondence", {}) or {}
        n_new_dets_total += int(c.get("n_new_detections", 0) or 0)
        n_matched_dets_total += int(c.get("n_matched_detections", 0) or 0)
        n_objects_merged_total += int(c.get("n_objects_merged", 0) or 0)
        n_new_objects_total += int(c.get("n_new_objects_appended", 0) or 0)
        n_far_merges_blocked_total += int(c.get("n_far_merges_blocked", 0) or 0)

    # Merge-funnel diagnostic (post-fix get_neighbors). For each frame:
    # detections that had ≥1 candidate after the class+feat-sim gate
    # vs. detections that survived all the way through Hellinger.
    n_with_candidates_total = 0
    n_with_hellinger_match_total = 0
    n_active_dets_total = 0
    for f in frames:
        funnel = (f.get("neighbors", {}) or {}).get("funnel") or {}
        n_with_candidates_total += int(funnel.get("n_with_class_feat_candidates", 0) or 0)
        n_with_hellinger_match_total += int(funnel.get("n_with_hellinger_match", 0) or 0)
        n_active_dets_total += int(funnel.get("n_active_detections", 0) or 0)

    # ── Numerical health red flags
    n_nan_means_total = 0
    n_nan_cov6_total = 0
    n_inf_means_total = 0
    n_inf_cov6_total = 0
    n_indef_active_total = 0
    n_eig_lt_floor_total = 0
    n_pos_jump_gt_0p3m_total = 0
    n_pos_jump_gt_1p0m_total = 0
    max_pos_jump_seen = 0.0
    log_cond_p99_max = -math.inf
    eig_min_p1_min = math.inf
    for f in frames:
        g = f.get("gaussian_update", {}) or {}
        n_nan_means_total += int(g.get("n_nan_means", 0) or 0)
        n_nan_cov6_total += int(g.get("n_nan_cov6", 0) or 0)
        n_inf_means_total += int(g.get("n_inf_means", 0) or 0)
        n_inf_cov6_total += int(g.get("n_inf_cov6", 0) or 0)
        n_indef_active_total += int(g.get("n_cov_indef_active", 0) or 0)
        # Accept either old (1e-6) or new (5e-7) field name for back-compat.
        n_eig_lt_floor_total += int(
            (g.get("n_cov_eig_lt_5e-7_active") if "n_cov_eig_lt_5e-7_active" in g else g.get("n_cov_eig_lt_1e-6_active", 0))
            or 0
        )
        n_pos_jump_gt_0p3m_total += int(g.get("n_pos_jump_gt_0p3m", 0) or 0)
        n_pos_jump_gt_1p0m_total += int(g.get("n_pos_jump_gt_1p0m", 0) or 0)
        if isinstance(g.get("max_pos_jump_m"), (int, float)) and not isinstance(g["max_pos_jump_m"], bool):
            max_pos_jump_seen = max(max_pos_jump_seen, float(g["max_pos_jump_m"]))
        p99 = _maybe(g, "log10_cond_pct_active.p99")
        if isinstance(p99, (int, float)) and math.isfinite(p99):
            log_cond_p99_max = max(log_cond_p99_max, float(p99))
        p1 = _maybe(g, "eig_min_pct_active.p1")
        if isinstance(p1, (int, float)) and math.isfinite(p1):
            eig_min_p1_min = min(eig_min_p1_min, float(p1))

    # ── Hellinger / feat-sim distribution near boundary
    hell_p10s, hell_p50s = [], []
    feat_p50s, feat_p90s = [], []
    for f in frames:
        n = f.get("neighbors", {}) or {}
        h = n.get("min_hellinger_per_det_pct")
        if isinstance(h, dict):
            if isinstance(h.get("p10"), (int, float)):
                hell_p10s.append(float(h["p10"]))
            if isinstance(h.get("p50"), (int, float)):
                hell_p50s.append(float(h["p50"]))
        s = n.get("max_feat_sim_per_det_pct")
        if isinstance(s, dict):
            if isinstance(s.get("p50"), (int, float)):
                feat_p50s.append(float(s["p50"]))
            if isinstance(s.get("p90"), (int, float)):
                feat_p90s.append(float(s["p90"]))

    def _avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    final_state = _maybe(scene_end or {}, "final_state", {}) or {}
    final_voxels = _maybe(scene_end or {}, "final_voxel_cloud", {}) or {}
    return {
        "path": str(path),
        "scene_id": (scene_start or {}).get("scene_id") or path.stem.replace(".trace", ""),
        "n_frames": n_frames,
        "duration_s": (scene_end or {}).get("scene_duration_s"),
        "raw_detections_total": raw_in_total,
        "post_filter_detections_total": final_out_total,
        "n_new_detections_total": n_new_dets_total,
        "n_matched_detections_total": n_matched_dets_total,
        "n_objects_merged_total": n_objects_merged_total,
        "n_new_objects_total": n_new_objects_total,
        "n_far_merges_blocked_total": n_far_merges_blocked_total,
        "n_active_dets_total": n_active_dets_total,
        "n_with_candidates_total": n_with_candidates_total,
        "n_with_hellinger_match_total": n_with_hellinger_match_total,
        "filter_totals": dict(filter_totals),
        "n_nan_means_total": n_nan_means_total,
        "n_nan_cov6_total": n_nan_cov6_total,
        "n_inf_means_total": n_inf_means_total,
        "n_inf_cov6_total": n_inf_cov6_total,
        "n_cov_indef_active_total": n_indef_active_total,
        "n_cov_eig_below_floor_active_total": n_eig_lt_floor_total,
        "n_pos_jump_gt_0p3m_total": n_pos_jump_gt_0p3m_total,
        "n_pos_jump_gt_1p0m_total": n_pos_jump_gt_1p0m_total,
        "max_pos_jump_m": max_pos_jump_seen,
        "log10_cond_p99_max": (log_cond_p99_max if log_cond_p99_max > -math.inf else None),
        "eig_min_p1_min": (eig_min_p1_min if eig_min_p1_min < math.inf else None),
        "min_hellinger_avg_p10": _avg(hell_p10s),
        "min_hellinger_avg_p50": _avg(hell_p50s),
        "max_feat_sim_avg_p50": _avg(feat_p50s),
        "max_feat_sim_avg_p90": _avg(feat_p90s),
        "final_n_total": final_state.get("n_total"),
        "final_n_active": final_state.get("n_active"),
        "final_voxel_levels": final_voxels.get("level_histogram"),
        "final_voxels_total": final_voxels.get("voxels_total"),
    }


def _print_funnel(scene: Dict[str, Any]) -> None:
    raw = scene.get("raw_detections_total", 0)
    post = scene.get("post_filter_detections_total", 0)
    new = scene.get("n_new_detections_total", 0)
    matched = scene.get("n_matched_detections_total", 0)
    new_obj = scene.get("n_new_objects_total", 0)
    merged = scene.get("n_objects_merged_total", 0)
    n_active = scene.get("final_n_active") or 0
    n_total = scene.get("final_n_total") or 0
    print(f"  funnel: raw={raw}  →  post-filter={post}  "
          f"(matched={matched}, new_dets={new})  "
          f"→  new_objects={new_obj}, merges={merged}  "
          f"|  final n_total={n_total} n_active={n_active}")
    ft = scene.get("filter_totals", {})
    if ft:
        order = ["border", "num_pixels", "distance", "uninformative_labels", "duplicates_iou"]
        seen = []
        for nm in order + [k for k in ft.keys() if k not in order]:
            if nm in ft and nm not in seen:
                seen.append(nm)
                t = ft[nm]
                pct = (t["n_dropped"] / t["n_in"]) * 100.0 if t["n_in"] > 0 else 0.0
                print(f"    filter[{nm:<22}] n_in={t['n_in']:>7}  n_out={t['n_out']:>7}  "
                      f"dropped={t['n_dropped']:>7} ({pct:5.1f}%)")


def _print_health(scene: Dict[str, Any]) -> None:
    items = [
        ("n_nan_means", scene.get("n_nan_means_total", 0)),
        ("n_nan_cov6", scene.get("n_nan_cov6_total", 0)),
        ("n_inf_means", scene.get("n_inf_means_total", 0)),
        ("n_inf_cov6", scene.get("n_inf_cov6_total", 0)),
        ("n_cov_indef_active", scene.get("n_cov_indef_active_total", 0)),
        ("n_cov_eig_below_floor", scene.get("n_cov_eig_below_floor_active_total", 0)),
        ("n_pos_jump_>0.3m", scene.get("n_pos_jump_gt_0p3m_total", 0)),
        ("n_pos_jump_>1.0m", scene.get("n_pos_jump_gt_1p0m_total", 0)),
    ]
    flags = [(k, v) for k, v in items if isinstance(v, int) and v > 0]
    print(f"  health: max_pos_jump_m={_fmt(scene.get('max_pos_jump_m'), 5).strip()}  "
          f"log10_cond_p99_max={_fmt(scene.get('log10_cond_p99_max'), 5).strip()}  "
          f"eig_min_p1_min={_fmt(scene.get('eig_min_p1_min'), 8).strip()}")
    if flags:
        print("    RED FLAGS:")
        for k, v in flags:
            print(f"      {k:<28} = {v}")
    else:
        print("    (no NaN / Inf / indefinite-cov / large-jump events)")


def _print_neighbors(scene: Dict[str, Any]) -> None:
    h_p10 = scene.get("min_hellinger_avg_p10")
    h_p50 = scene.get("min_hellinger_avg_p50")
    f_p50 = scene.get("max_feat_sim_avg_p50")
    f_p90 = scene.get("max_feat_sim_avg_p90")
    if any(v is not None for v in (h_p10, h_p50, f_p50, f_p90)):
        print(f"  neighbors: closest-Hellinger avg(p10/p50)={_fmt(h_p10, 5).strip()}/{_fmt(h_p50, 5).strip()}  "
              f"best-feat-sim avg(p50/p90)={_fmt(f_p50, 5).strip()}/{_fmt(f_p90, 5).strip()}")


def _print_voxels(scene: Dict[str, Any]) -> None:
    lv = scene.get("final_voxel_levels") or {}
    vt = scene.get("final_voxels_total")
    if lv:
        items = sorted(((int(k), int(v)) for k, v in lv.items()))
        print(f"  voxels: levels={items}  voxels_total={vt}")


def _top_frames(path: Path, *, top_merges: int, top_jumps: int) -> None:
    events = _read_jsonl(path)
    frames = [e for e in events if e.get("event") == "frame"]
    if top_merges > 0:
        merge_frames = sorted(
            ((int((f.get("correspondence") or {}).get("n_objects_merged", 0) or 0), f) for f in frames),
            key=lambda kv: -kv[0],
        )[:top_merges]
        if merge_frames and merge_frames[0][0] > 0:
            print(f"  top-{top_merges} frames by #objects_merged:")
            for n, f in merge_frames:
                if n == 0:
                    break
                step = f.get("step_idx", "?")
                merges = (f.get("correspondence") or {}).get("merges_top", []) or []
                example = ""
                if merges:
                    m = merges[0]
                    example = (f"  e.g. loser_id={m.get('loser_id')}({m.get('loser_caption','').strip()[:18]!r}) "
                               f"-> winner_id={m.get('winner_id')}({m.get('winner_caption','').strip()[:18]!r}) "
                               f"Δ={_fmt(m.get('delta_m'), 4).strip()}m")
                print(f"    step={step:>5}  n_merged={n:>3}{example}")
    if top_jumps > 0:
        jump_frames = sorted(
            ((float((f.get("gaussian_update") or {}).get("max_pos_jump_m") or 0.0), f) for f in frames),
            key=lambda kv: -kv[0],
        )[:top_jumps]
        if jump_frames and jump_frames[0][0] > 0:
            print(f"  top-{top_jumps} frames by max_pos_jump_m:")
            for v, f in jump_frames:
                if v <= 0:
                    break
                step = f.get("step_idx", "?")
                touched = (f.get("gaussian_update") or {}).get("n_active") or "—"
                print(f"    step={step:>5}  max_pos_jump_m={v:6.3f}  n_active={touched}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Trace .jsonl file or directory containing *.trace.jsonl")
    p.add_argument("--top-merges", type=int, default=3, help="Show top-K frames by #objects merged.")
    p.add_argument("--top-jumps", type=int, default=3, help="Show top-K frames by max position jump.")
    p.add_argument("--json", action="store_true", help="Emit per-scene summaries as JSON instead of text.")
    args = p.parse_args(argv)

    summaries = []
    for f in _iter_trace_files(args.path):
        s = _summarize_scene(f)
        summaries.append(s)
        if args.json:
            continue
        print(f"\n=== {s['scene_id']} ({Path(s['path']).name}) ===")
        print(f"  frames={s['n_frames']}  duration_s={_fmt(s.get('duration_s'), 6).strip()}")
        _print_funnel(s)
        _print_neighbors(s)
        _print_health(s)
        _print_voxels(s)
        _top_frames(f, top_merges=args.top_merges, top_jumps=args.top_jumps)

    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
    elif len(summaries) > 1:
        print("\n=== aggregate ===")
        for k in (
            "n_frames", "raw_detections_total", "post_filter_detections_total",
            "n_new_objects_total", "n_objects_merged_total",
            "n_pos_jump_gt_0p3m_total", "n_pos_jump_gt_1p0m_total",
            "n_nan_means_total", "n_nan_cov6_total",
            "n_cov_indef_active_total", "n_cov_eig_below_floor_active_total",
        ):
            tot = sum(int(s.get(k, 0) or 0) for s in summaries)
            print(f"  {k:<32} = {tot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
