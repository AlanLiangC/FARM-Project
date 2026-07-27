"""Iterate ReferIt3D utterances and write a predictions JSON.

Mirrors :mod:`scene_graph.eval.openeqa.runner`'s pattern: utterances are
grouped by ``scan_id``; for each scene we load the scene_state.pt once,
build a SceneGraphRetriever once via ``from_scene_state(...)``, then run
every utterance for that scene. Predictions are persisted incrementally
to support resume-on-rerun.

Output schema (``predictions.json``):

    [
      {
        "uid": "nr3d_32618",
        "dataset": "nr3d",
        "scan_id": "scene0525_00",
        "target_id": 9,
        "distractor_ids": [10, 11, 12, 62],
        "instance_type": "plant",
        "utterance": "The plant at the far right ...",
        "ranked": [
          {"rank": 1, "object_id": 9, "score": 0.21, "bbox_min": [...], "bbox_max": [...], "label": ..., "caption": ...},
          ...
        ],
        "elapsed_s": 1.85
      },
      ...
    ]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .dataset import (
    Utterance,
    list_local_scenes,
    load_val_scenes,
    utterances_by_scene,
    val_local_subset,
)
from .retrieval_adapter import ScenePredictor, ranked_to_dicts

LOGGER = logging.getLogger("scene_graph.eval.referit3d.runner")


# ---------------------------------------------------------------------
# Scene-state discovery
# ---------------------------------------------------------------------


def discover_scene_states(scenes_dir: Path) -> Dict[str, Path]:
    """Return ``{scene_id: scene_state_path}`` for every .pt under ``scenes_dir``."""
    out: Dict[str, Path] = {}
    if not scenes_dir.exists():
        return out
    for entry in sorted(scenes_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".pt":
            out[entry.stem] = entry
        elif entry.is_dir():
            cand = entry / "scene_state.pt"
            if cand.exists():
                out[entry.name] = cand
    return out


# ---------------------------------------------------------------------
# Predictions helpers
# ---------------------------------------------------------------------


def utterance_to_record(utterance: Utterance) -> Dict[str, Any]:
    """Stable JSON shape for one ReferIt3D utterance (without predictions)."""
    return {
        "uid": utterance.uid,
        "dataset": utterance.dataset,
        "scan_id": utterance.scan_id,
        "target_id": int(utterance.target_id),
        "distractor_ids": list(utterance.distractor_ids),
        "instance_type": utterance.instance_type,
        "utterance": utterance.utterance,
        "mentions_target_class": bool(utterance.mentions_target_class),
        "reference_type": utterance.reference_type,
        "coarse_reference_type": utterance.coarse_reference_type,
        "anchor_ids": list(utterance.anchor_ids) if utterance.anchor_ids is not None else None,
        "anchors_types": list(utterance.anchors_types) if utterance.anchors_types is not None else None,
        "uses_object_lang": utterance.uses_object_lang,
        "uses_spatial_lang": utterance.uses_spatial_lang,
        "uses_color_lang": utterance.uses_color_lang,
        "uses_shape_lang": utterance.uses_shape_lang,
    }


def _load_existing(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not parse existing predictions at %s: %s", output_path, exc)
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in payload or []:
        uid = str(entry.get("uid") or "")
        if uid:
            out[uid] = entry
    return out


def _persist(output_path: Path, ordered_uids: Sequence[str], predictions: Dict[str, Dict[str, Any]]) -> None:
    out: List[Dict[str, Any]] = [predictions[u] for u in ordered_uids if u in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Knobs that affect how each utterance is scored."""

    k_sigma: float = 2.5
    max_predictions: int = 20  # ranked list cap per utterance (covers Recall@K up to 20)
    retriever_verbose: bool = False
    embedder_verbose: bool = False
    log_every: int = 50  # progress logging frequency (per scene)
    prefer_voxel_aabb: bool = True  # use sparse voxel-cloud AABB when available
    voxel_sor_k: int = 8
    voxel_sor_alpha: float = 2.0


def run_predictions(
    *,
    scenes_dir: Path,
    output_path: Path,
    cfg: Optional[RunnerConfig] = None,
    utterances: Optional[Iterable[Utterance]] = None,
    scene_filter: Optional[Sequence[str]] = None,
    max_utterances: Optional[int] = None,
    resume: bool = True,
    retriever_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Iterate utterances grouped by scan_id, write predictions JSON.

    Args:
        scenes_dir: Directory containing ``<scene_id>.pt`` files.
        output_path: Predictions JSON path; written incrementally.
        cfg: Runner knobs (k_sigma, max_predictions, …).
        utterances: Iterable of :class:`Utterance` to score. Defaults to the
            full val∩local subset.
        scene_filter: Optional whitelist of scan ids.
        max_utterances: Cap total work for smoke tests.
        resume: When True, skip utterances already present in the JSON.

    Returns:
        The persisted prediction list (also on disk at ``output_path``).
    """
    cfg = cfg or RunnerConfig()
    retriever_kwargs = dict(retriever_kwargs or {})
    # Lazy imports — these pull torch + scene_graph.scene_graph + the retriever stack.
    from scene_graph.llm_utils import EmbedInterface
    from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever

    if utterances is None:
        utterances = val_local_subset()
    utterances = list(utterances)
    if scene_filter is not None:
        allow = set(scene_filter)
        utterances = [u for u in utterances if u.scan_id in allow]
    if max_utterances is not None:
        utterances = utterances[: int(max_utterances)]

    scene_states = discover_scene_states(scenes_dir)
    LOGGER.info(
        "found %d scene_state.pt under %s; %d utterances queued",
        len(scene_states), scenes_dir, len(utterances),
    )

    grouped: Dict[str, List[Utterance]] = utterances_by_scene(utterances)

    existing: Dict[str, Dict[str, Any]] = _load_existing(output_path) if resume else {}
    if existing:
        LOGGER.info("resuming from %d existing predictions", len(existing))

    predictions: Dict[str, Dict[str, Any]] = dict(existing)
    ordered_uids: List[str] = [u.uid for u in utterances]

    embedder = EmbedInterface(verbose=cfg.embedder_verbose)

    n_scenes_done = 0
    n_uts_done = 0
    n_uts_ok = 0
    n_uts_fail = 0
    t_start = time.time()

    for scan_id in sorted(grouped.keys()):
        utts_for_scene = [u for u in grouped[scan_id] if u.uid not in predictions]
        if not utts_for_scene:
            continue
        pt = scene_states.get(scan_id)
        if pt is None:
            LOGGER.warning("skipping %s — no scene_state.pt found", scan_id)
            continue
        LOGGER.info(
            "loading scene %s (%d new utterances) from %s",
            scan_id, len(utts_for_scene), pt.name,
        )
        try:
            retriever = SceneGraphRetriever.from_scene_state(
                pt, embedder=embedder, verbose=cfg.retriever_verbose,
                **retriever_kwargs,
            )
            predictor = ScenePredictor(
                retriever,
                k_sigma=cfg.k_sigma,
                max_predictions=cfg.max_predictions,
                prefer_voxel_aabb=cfg.prefer_voxel_aabb,
                voxel_sor_k=cfg.voxel_sor_k,
                voxel_sor_alpha=cfg.voxel_sor_alpha,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("failed to build retriever for %s: %s", scan_id, exc)
            continue

        n_scenes_done += 1
        scene_started = time.time()

        for i, utt in enumerate(utts_for_scene, start=1):
            t0 = time.time()
            record = utterance_to_record(utt)
            try:
                ranked = predictor.predict(utt.utterance)
                record["ranked"] = ranked_to_dicts(ranked)
                record["error"] = None
                n_uts_ok += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("retrieve failed on %s: %s", utt.uid, exc)
                record["ranked"] = []
                record["error"] = str(exc)
                n_uts_fail += 1
            record["elapsed_s"] = round(time.time() - t0, 4)
            predictions[utt.uid] = record
            n_uts_done += 1

            if i % cfg.log_every == 0 or i == len(utts_for_scene):
                _persist(output_path, ordered_uids, predictions)
                LOGGER.info(
                    "  %s %d/%d utterances (%.1f utt/s)",
                    scan_id, i, len(utts_for_scene),
                    i / max(1e-3, time.time() - scene_started),
                )
        _persist(output_path, ordered_uids, predictions)

    _persist(output_path, ordered_uids, predictions)
    LOGGER.info(
        "done. %d scenes, %d utterances (%d ok, %d failed) in %.1fs (avg %.2f utt/s)",
        n_scenes_done, n_uts_done, n_uts_ok, n_uts_fail,
        time.time() - t_start,
        n_uts_done / max(1e-3, time.time() - t_start),
    )
    return [predictions[u] for u in ordered_uids if u in predictions]
