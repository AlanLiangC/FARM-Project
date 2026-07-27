from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

_LOGGER = logging.getLogger(__name__)

SCENE_GRAPH_MODEL_DIR_ENV = "SCENE_GRAPH_MODEL_DIR"

# DINOv3 object-merge backbones. ``vits16plus`` (ViT-S+/16) is the paper backbone
# — tighter, more stable merging across scenes and imaging conditions — but is a
# gated Meta HF repo. ``vits16`` (ViT-S/16) is non-gated and checked in by
# bootstrap_models.sh so a fresh clone runs offline. See ``resolve_dino_backbone``.
DINOV3_VITS16_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_VITS16PLUS_MODEL = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
_DINO_FALLBACK_WARNED = False


def _looks_like_repo_root(path: Path) -> bool:
    return (path / ".git").exists() or ((path / "README.md").is_file() and (path / "src" / "scene_graph").is_dir())


def _discover_repo_root(start: Path) -> Optional[Path]:
    candidate = start.expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    with contextlib.suppress(Exception):
        candidate = candidate.resolve()
    for path in (candidate, *candidate.parents):
        if _looks_like_repo_root(path):
            return path
    return None


def mapping_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    discovered = _discover_repo_root(Path(__file__).resolve())
    if discovered is not None:
        return discovered
    return Path(__file__).resolve().parents[2]


def default_repo_models_root() -> Path:
    return repo_root() / "models"


def ros_home() -> Path:
    return Path(os.environ.get("ROS_HOME", "~/.ros")).expanduser()


def _unique_existing_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = path.expanduser()
        if not candidate.exists():
            continue
        with contextlib.suppress(Exception):
            candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return tuple(resolved)


def mapping_package_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    with contextlib.suppress(Exception):
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory("scene_graph")))
    candidates.append(repo_root())
    candidates.append(mapping_source_root())
    return _unique_existing_paths(candidates)


def cwd_repo_models_root() -> Optional[Path]:
    discovered = _discover_repo_root(Path.cwd())
    if discovered is None:
        return None
    candidate = discovered / "models"
    if candidate.exists():
        return candidate
    return None


def find_package_file(*relative_parts: str) -> Optional[Path]:
    if not relative_parts:
        return None
    for root in mapping_package_roots():
        candidate = root.joinpath(*relative_parts)
        if candidate.is_file():
            return candidate
    return None


def candidate_model_roots() -> tuple[Path, ...]:
    env_model_dir = str(os.environ.get(SCENE_GRAPH_MODEL_DIR_ENV, "") or "").strip()
    candidates: list[Path] = []
    if env_model_dir:
        candidates.append(Path(env_model_dir))
    # Prefer a repo-local `./models` directory when launching from inside the checkout.
    cwd_models_root = cwd_repo_models_root()
    if cwd_models_root is not None:
        candidates.append(cwd_models_root)
    # Prefer the repository-local `./models` directory by default for source checkouts.
    candidates.append(default_repo_models_root())
    for root in mapping_package_roots():
        candidates.append(root / "models")
    candidates.append(repo_root().parent / "models")
    return _unique_existing_paths(candidates)


def find_model_file(filename: str, *subdirs: str) -> Optional[Path]:
    relative_candidates = [Path(filename)]
    if subdirs:
        relative_candidates.insert(0, Path(*subdirs) / filename)
    for root in candidate_model_roots():
        for relative_path in relative_candidates:
            candidate = root / relative_path
            if candidate.is_file():
                return candidate
    return None


def find_model_dir(dirname: str, *subdirs: str) -> Optional[Path]:
    relative_candidates = [Path(dirname)]
    if subdirs:
        relative_candidates.insert(0, Path(*subdirs) / dirname)
    for root in candidate_model_roots():
        for relative_path in relative_candidates:
            candidate = root / relative_path
            if candidate.is_dir():
                return candidate
    return None


def resolve_dino_backbone() -> tuple[str, Optional[str]]:
    """Choose the DINOv3 object-merge backbone, auto-preferring ViT-S+/16.

    Prefers ``dinov3-vits16plus`` (the paper backbone — tighter, more stable
    merging across scenes and imaging conditions) whenever a local copy is
    present. Otherwise falls back to the non-gated ``dinov3-vits16`` (checked in
    by ``bootstrap_models.sh``) with a one-time warning, so a fresh clone still
    runs fully offline. To enable vits16plus, accept the gated Meta license and
    place the weights at ``models/dinov3-vits16plus`` (see README / EVALUATION.md).

    Returns ``(model_name, weights_path_or_None)``.
    """
    global _DINO_FALLBACK_WARNED
    plus = find_model_dir("dinov3-vits16plus") or find_model_dir("dinov3-vits16plus", "models")
    if plus is not None:
        return DINOV3_VITS16PLUS_MODEL, str(plus)
    base = find_model_dir("dinov3-vits16") or find_model_dir("dinov3-vits16", "models")
    if base is not None:
        if not _DINO_FALLBACK_WARNED:
            _DINO_FALLBACK_WARNED = True
            _LOGGER.warning(
                "DINO merge backbone: dinov3-vits16plus not found; falling back to the "
                "non-gated dinov3-vits16. Object merging is more fragmented and less stable "
                "across scenes/imaging conditions — see README / EVALUATION.md to enable "
                "vits16plus for paper-grade performance."
            )
        return DINOV3_VITS16_MODEL, str(base)
    return DINOV3_VITS16_MODEL, None
