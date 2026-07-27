from __future__ import annotations

import base64
import contextlib
import io  # NEW
import json
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - optional embedding backend
    requests = None

try:
    import torch
except ImportError:  # pragma: no cover - defensive for environments without torch
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - fallback if pillow is unavailable
    Image = None

# You can keep these if you still want the HF path around, but they won't be used anymore
try:
    from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - allow environments without transformers
    AutoModel = None
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

# NEW: Ollama generate (raw mode)
try:
    from ollama import generate as ollama_generate
except ImportError:
    ollama_generate = None

from scene_graph.map_update.cannot_link import are_cannot_linked_indices
from scene_graph.map_update.get_neighbors import get_neighbors_by_hellinger_distance
from scene_graph.runtime_paths import find_model_dir

from .models import ObjectCaptionResult, ObjectCaptionTask
from .structured import CAPTION_SCHEMA, is_explicit_unclear_caption, parse_structured_caption

LEGACY_CAPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["new_caption"],
    "properties": {
        "new_caption": {
            "type": "string",
            # simple length cap (cheap + robust)
            "maxLength": 100,
        },
    },
}

QWEN3_EMBED_TASK = (
    "Embed for object identity and affordance retrieval. "
    "Use the object's category, supercategory, and visible attributes; ignore viewpoint and wording."
)

_DEFAULT_SIGLIP2_LOCAL_DIRNAME = "siglip2-large-patch16-256"
_DEFAULT_SIGLIP2_HF_CKPT = "google/siglip2-large-patch16-256"
_DEFAULT_QWEN3_VL_EMBED_CKPT = "Qwen/Qwen3-VL-Embedding-2B"

_UNKNOWN_CATEGORIES = {"", "unknown", "object", "item", "thing", "part", "other"}
_CATEGORY_ALIASES = {
    "couch": "sofa",
    "settee": "sofa",
    "trash bin": "trash can",
    "garbage can": "trash can",
    "waste bin": "trash can",
    "rubbish bin": "trash can",
    "television": "tv",
    "fire detector": "fire alarm",
    "smoke alarm": "fire alarm",
    "smoke detector": "fire alarm",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _canonical_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = " ".join(text.replace("_", " ").replace("-", " ").split())
    if text.endswith("s") and len(text) > 3:
        text = text[:-1]
    return _CATEGORY_ALIASES.get(text, text)


def _resolve_default_siglip2_ckpt() -> str:
    """
    Prefer local weights (installed with the mapping package or present in the dev repo)
    to avoid relying on HF downloads/caching at runtime.
    """
    local_path = find_model_dir(_DEFAULT_SIGLIP2_LOCAL_DIRNAME)
    if local_path is not None:
        return str(local_path)

    return _DEFAULT_SIGLIP2_HF_CKPT


class CaptionWorker:
    def __init__(
        self,
        scene_state: dict,
        tasks_queue: queue.Queue,
        results_queue: queue.Queue,
        *,
        caption_batch_size: int = 10,
        device: str = "cuda",
        caption_server: str = "vllm",
        max_retries: int = 2,
        debug: bool = False,
        poll_interval: float = 0.1,
        caption_version: int = 20,
        top_p: float = -1.0,
        temperature: float = -1.0,
        caption_spatial_context: Optional[bool] = None,
        caption_spatial_context_include_position: Optional[bool] = None,
        recaption_time_threshold_sec: float = 0.0,
        caption_merge_hellinger_thresh: Optional[float] = None,
        caption_merge_caption_thresh: Optional[float] = None,
        caption_merge_siglip2_thresh: Optional[float] = None,
        caption_merge_require_visual: Optional[bool] = None,
        caption_merge_require_category_compat: Optional[bool] = None,
        caption_visual_prompt_mode: str = "bbox_crop",
        caption_prompt_variant: str = "default",
    ) -> None:
        self.scene_state = scene_state
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue
        self.caption_batch_size = caption_batch_size
        self.device = device
        self.max_retries = max_retries
        self.debug = debug
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._caption_loop, daemon=True, name="CaptionWorker")
        self._model = None
        self._processor = None
        self._base_messages: List[Dict[str, Any]] = []
        self._hq_base_messages: List[Dict[str, Any]] = []
        self._hq_base_messages: List[Dict[str, Any]] = []
        self._base_context = None
        self._model_load_failed = False
        self._session: Optional["requests.Session"] = None

        # Which caption server to use
        self._caption_server: str = caption_server
        self._caption_version: int = caption_version
        self._caption_expects_json: bool = True
        self._caption_visual_prompt_mode = str(
            caption_visual_prompt_mode or os.getenv("CAPTION_VISUAL_PROMPT_MODE", "bbox_crop")
        ).strip().lower()
        if self._caption_visual_prompt_mode not in {"bbox_crop", "mask_crop", "mask_composite"}:
            self._caption_visual_prompt_mode = "bbox_crop"
        self._caption_prompt_variant = str(
            caption_prompt_variant or os.getenv("CAPTION_PROMPT_VARIANT", "default")
        ).strip().lower()
        self._caption_merge_hellinger_thresh = float(
            caption_merge_hellinger_thresh
            if caption_merge_hellinger_thresh is not None
            else _env_float("CAPTION_MERGE_HELLINGER_THRESH", 0.65)
        )
        self._caption_merge_caption_thresh = float(
            caption_merge_caption_thresh
            if caption_merge_caption_thresh is not None
            else _env_float("CAPTION_MERGE_CAPTION_THRESH", 0.92)
        )
        self._caption_merge_siglip2_thresh = float(
            caption_merge_siglip2_thresh
            if caption_merge_siglip2_thresh is not None
            else _env_float("CAPTION_MERGE_SIGLIP2_THRESH", 0.93)
        )
        self._caption_merge_require_visual = (
            bool(caption_merge_require_visual)
            if caption_merge_require_visual is not None
            else _env_bool("CAPTION_MERGE_REQUIRE_VISUAL", True)
        )
        self._caption_merge_require_category_compat = (
            bool(caption_merge_require_category_compat)
            if caption_merge_require_category_compat is not None
            else _env_bool("CAPTION_MERGE_REQUIRE_CATEGORY_COMPAT", True)
        )

        # Settings for the caption server
        self._ollama_model: str = "qwen3-vl:8b-instruct"

        self._sglang_model: str = "Qwen/Qwen3-VL-8B-Instruct"
        self._sglang_url: str = "http://localhost:30000/v1/chat/completions"

        # NOTE: vLLM's OpenAI-compatible API expects the *served* model id (as returned by GET /v1/models),
        # not necessarily the Hugging Face repo name. Our `run.sh` uses `--served-model-name qwen3.5-9b`.
        self._vllm_model: str = os.getenv("VLLM_CAPTION_MODEL") or os.getenv("VLLM_VL_MODEL") or "qwen3.5-9b"
        self._vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._vllm_timeout_s: float = float(os.getenv("VLLM_TIMEOUT_S", "30"))
        self._vllm_prefix_warmup_enabled: bool = str(
            os.getenv("VLLM_CAPTION_PREFIX_WARMUP", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._vllm_prefix_warmup_timeout_s: float = max(
            1.0,
            float(os.getenv("VLLM_CAPTION_PREFIX_WARMUP_TIMEOUT_S", str(min(self._vllm_timeout_s, 10.0)))),
        )
        self._vllm_warmed_prefixes: set[str] = set()
        self._vllm_warmup_lock = threading.Lock()

        if temperature > 0.0:
            self._vllm_temperature = temperature
        else:
            self._vllm_temperature = float(os.getenv("VLLM_TEMPERATURE", "0.0"))

        if top_p > 0.0:
            self._vllm_top_p = top_p
        else:
            self._vllm_top_p = float(os.getenv("VLLM_TOP_P", "0.9"))

        # Structured captions need room for five fields, but should remain compact.
        _default_max_tokens = "128" if self._caption_version >= 19 else ("48" if self._caption_version >= 2 else "24")
        self._vllm_max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", _default_max_tokens))
        self._vllm_api_key: Optional[str] = os.getenv("VLLM_API_KEY")

        # Qwen3.5-9B is a reasoning-capable model; we always run it in non-thinking
        # instruct mode for captioning (fast, no <think>...</think> preamble) by
        # injecting chat_template_kwargs={enable_thinking: False}. Set
        # VLLM_CAPTION_DISABLE_THINKING=0 to fall back to the model's default
        # (thinking) behavior — useful only when pointing at a non-reasoning model
        # that rejects this kwarg.
        self._vllm_disable_thinking: bool = os.getenv("VLLM_CAPTION_DISABLE_THINKING", "1") == "1"

        # Embedding model settings
        self._retry_pending_object_ids: set[int] = set()
        self._embed_model: str = os.getenv("VLLM_EMBED_MODEL", "qwen3-emb-0.6b")
        self._embed_base_url: str = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8002/v1")
        self._embed_timeout_s: float = float(os.getenv("VLLM_EMBED_TIMEOUT_S", "30"))
        self._embed_api_key: Optional[str] = os.getenv("VLLM_EMBED_API_KEY") or self._vllm_api_key
        self._embed_max_retries: int = 3
        self._embed_dim: Optional[int] = None
        self._embed_warned = False
        self._embed_server_ok: Optional[bool] = None
        # Qwen3 embedding prompt contract:
        # - Captions are embedded as DOCUMENTS (not queries), with a shared task string.
        # This must stay aligned with LAM's query embedding wrapper for robust retrieval.
        self._qwen3_embed_task: str = QWEN3_EMBED_TASK

        # Spatial context: include object size (cov6) in the captioning prompt; position is optional.
        if caption_spatial_context is not None:
            self._caption_spatial_context: bool = bool(caption_spatial_context)
        else:
            self._caption_spatial_context: bool = str(os.getenv("CAPTION_SPATIAL_CONTEXT", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if caption_spatial_context_include_position is not None:
            self._caption_spatial_context_include_position: bool = bool(caption_spatial_context_include_position)
        else:
            self._caption_spatial_context_include_position: bool = str(
                os.getenv("CAPTION_SPATIAL_CONTEXT_POSITION", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}

        # SigLIP2 settings (aligned image/text retrieval embeddings from caption crops)
        self._siglip2_enabled: bool = str(os.getenv("SIGLIP2_ENABLED", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        # Default off: the SigLIP2 debug stream can be quite verbose.
        self._siglip2_debug: bool = str(os.getenv("SIGLIP2_DEBUG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self._siglip2_ckpt: str = str(os.getenv("SIGLIP2_CKPT") or _resolve_default_siglip2_ckpt())
        self._siglip2_model = None
        self._siglip2_processor = None
        self._siglip2_mean = None
        self._siglip2_std = None
        self._siglip2_target_hw: Optional[tuple[int, int]] = None
        self._siglip2_dim: Optional[int] = None
        self._siglip2_load_failed: bool = False
        self._siglip2_failure_logged: bool = False
        self._siglip2_rpc_timeout_s: float = max(0.05, float(os.getenv("SIGLIP2_RPC_TIMEOUT_S", "0.35")))
        self._siglip2_hi_queue_max: int = max(1, int(os.getenv("SIGLIP2_SCHED_HI_QUEUE_MAX", "32")))
        self._siglip2_lo_queue_max: int = max(1, int(os.getenv("SIGLIP2_SCHED_LO_QUEUE_MAX", "128")))
        self._siglip2_hi_batch_window_s: float = max(
            0.0, float(os.getenv("SIGLIP2_SCHED_HI_BATCH_WINDOW_MS", "5.0")) / 1000.0
        )
        self._siglip2_fair_hi_batches: int = max(1, int(os.getenv("SIGLIP2_SCHED_FAIR_HI_BATCHES", "3")))
        self._siglip2_text_max_batch: int = max(1, int(os.getenv("SIGLIP2_SCHED_TEXT_MAX_BATCH", "64")))
        self._siglip2_scheduler_stop = threading.Event()
        self._siglip2_hi_queue: queue.Queue = queue.Queue(maxsize=self._siglip2_hi_queue_max)
        self._siglip2_lo_queue: queue.Queue = queue.Queue(maxsize=self._siglip2_lo_queue_max)
        self._siglip2_scheduler_thread = threading.Thread(
            target=self._siglip2_scheduler_loop,
            daemon=True,
            name="Siglip2Scheduler",
        )

        # Qwen3-VL-Embedding settings (crop-image embeddings aligned with text queries).
        self._qwen3_vl_embed_enabled: bool = str(os.getenv("QWEN3_VL_EMBED_ENABLED", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._qwen3_vl_embed_debug: bool = str(os.getenv("QWEN3_VL_EMBED_DEBUG", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._qwen3_vl_embed_backend: str = str(os.getenv("QWEN3_VL_EMBED_BACKEND", "vllm")).strip().lower()
        if self._qwen3_vl_embed_backend not in {"vllm", "hf", "auto"}:
            self._qwen3_vl_embed_backend = "vllm"
        self._qwen3_vl_embed_model_name: str = os.getenv("VLLM_QWEN3_VL_EMBED_MODEL", "qwen3-vl-emb-2b")
        self._qwen3_vl_embed_base_url: str = os.getenv("VLLM_QWEN3_VL_EMBED_BASE_URL", "http://localhost:8006/v1")
        self._qwen3_vl_embed_timeout_s: float = float(os.getenv("VLLM_QWEN3_VL_EMBED_TIMEOUT_S", "30"))
        self._qwen3_vl_embed_api_key: Optional[str] = os.getenv("VLLM_QWEN3_VL_EMBED_API_KEY") or self._vllm_api_key
        self._qwen3_vl_embed_max_retries: int = max(1, int(os.getenv("VLLM_QWEN3_VL_EMBED_MAX_RETRIES", "3")))
        self._qwen3_vl_embed_workers: int = max(
            1, int(os.getenv("QWEN3_VL_EMBED_WORKERS", str(self.caption_batch_size)))
        )
        self._qwen3_vl_embed_token_stride: int = max(1, int(os.getenv("QWEN3_VL_EMBED_TOKEN_STRIDE", "32")))
        self._qwen3_vl_embed_max_hw: int = max(1, int(os.getenv("QWEN3_VL_EMBED_MAX_HW", "200")))
        self._qwen3_vl_embed_max_image_tokens: int = max(0, int(os.getenv("QWEN3_VL_EMBED_MAX_IMAGE_TOKENS", "64")))
        self._qwen3_vl_embed_jpeg_quality: int = min(95, max(40, int(os.getenv("QWEN3_VL_EMBED_JPEG_QUALITY", "85"))))
        self._qwen3_vl_embed_runtime_log: bool = str(os.getenv("QWEN3_VL_EMBED_RUNTIME_LOG", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._qwen3_vl_embed_server_ok: Optional[bool] = None
        self._qwen3_vl_embed_ckpt: str = str(os.getenv("QWEN3_VL_EMBED_CKPT") or _DEFAULT_QWEN3_VL_EMBED_CKPT)
        self._qwen3_vl_embed_hf_model = None
        self._qwen3_vl_embed_hf_processor = None
        self._qwen3_vl_embed_dim: Optional[int] = None
        self._qwen3_vl_embed_load_failed: bool = False
        self._qwen3_vl_embed_failure_logged: bool = False

        # Recaption scheduling (used to opportunistically fill small batches).
        self._recaption_fill_target: int = max(1, min(5, int(self.caption_batch_size)))
        self._recaption_min_distance_m: float = max(0.0, float(os.getenv("RECAPTION_MIN_DISTANCE_M", "40.0")))
        self._recaption_cooldown_s: float = max(0.0, float(os.getenv("RECAPTION_COOLDOWN_S", "30.0")))
        self._recaption_time_threshold_sec: float = max(0.0, float(recaption_time_threshold_sec))
        self._recaption_last_attempt_s: Dict[int, float] = {}

        # Timing metrics
        self._batch_count = 0
        self._batch_size_sum = 0
        self._gen_batch_count = 0
        self._gen_time_sum = 0.0
        self._timing_sum: Dict[str, float] = defaultdict(float)
        self._timing_count: Dict[str, int] = defaultdict(int)

        print(f"Worker initialized. Caption server {caption_server}")

    # Lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()
        if self._siglip2_enabled:
            with contextlib.suppress(RuntimeError):
                self._siglip2_scheduler_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._siglip2_scheduler_stop.set()
        with contextlib.suppress(Exception):
            self.tasks_queue.put_nowait(None)
        with contextlib.suppress(Exception):
            self._siglip2_hi_queue.put_nowait(None)
        with contextlib.suppress(Exception):
            self._siglip2_lo_queue.put_nowait(None)
        with contextlib.suppress(Exception):
            if self._session is not None:
                self._session.close()
                self._session = None

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)
        with contextlib.suppress(Exception):
            if self._siglip2_scheduler_thread.is_alive():
                self._siglip2_scheduler_thread.join(timeout=timeout)

    def _resolve_object_index(self, task: ObjectCaptionTask) -> Optional[int]:
        """
        Map a task's object_id to the current active object index, following id_redirect.
        This defends against queued tasks becoming stale after merges.
        """
        object_ids = self.scene_state.get("object_id")
        if object_ids is None:
            return None
        canonical_id = self._resolve_canonical_object_id(int(task.object_id))
        active_flags = self.scene_state.get("active")

        idx = None
        if hasattr(object_ids, "nonzero"):
            matches = (object_ids == canonical_id).nonzero(as_tuple=False)
            if matches is not None and matches.numel() > 0:
                idx = int(matches.view(-1)[0].item())
        if idx is None:
            try:
                idx = list(object_ids).index(canonical_id)
            except ValueError:
                return None

        if active_flags is not None and idx < len(active_flags):
            try:
                is_active = (
                    bool(active_flags[idx].item()) if hasattr(active_flags[idx], "item") else bool(active_flags[idx])
                )
            except Exception:
                is_active = False
            if not is_active:
                return None
        return idx

    def _resolve_canonical_object_id(self, object_id: int) -> int:
        """Resolve id_redirect transitively (with cycle guard) without relying on object indices."""
        try:
            current = int(object_id)
        except Exception:
            return object_id

        id_redirect = self.scene_state.get("id_redirect") or {}
        if not isinstance(id_redirect, dict) or not id_redirect:
            return current

        seen: set[int] = set()
        while True:
            if current in seen:
                break
            seen.add(current)
            nxt = id_redirect.get(current)
            if nxt is None:
                break
            try:
                nxt_int = int(nxt)
            except Exception:
                break
            if nxt_int == current:
                break
            current = nxt_int
        return current

    # Worker loop -----------------------------------------------------------
    def _caption_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self.tasks_queue.get(timeout=self.poll_interval)
            except queue.Empty:
                continue

            if task is None:
                # Sentinel for shutdown
                with contextlib.suppress(ValueError):
                    self.tasks_queue.task_done()
                break

            with contextlib.suppress(Exception):
                self._retry_pending_object_ids.discard(int(getattr(task, "object_id", -1)))

            batch: List[ObjectCaptionTask] = [task]
            queue_task_count = 1
            # Drain additional tasks up to batch size
            while len(batch) < self.caption_batch_size:
                try:
                    next_task = self.tasks_queue.get_nowait()
                except queue.Empty:
                    break
                if next_task is None:
                    # Preserve sentinel for graceful shutdown
                    self.tasks_queue.put_nowait(None)
                    break
                with contextlib.suppress(Exception):
                    self._retry_pending_object_ids.discard(int(getattr(next_task, "object_id", -1)))
                batch.append(next_task)
                queue_task_count += 1

            # Opportunistically fill small caption batches with recaption candidates.
            fill_target = min(self.caption_batch_size, self._recaption_fill_target)
            if len(batch) < fill_target:
                queued_ids: set[int] = set()
                for queued_task in batch:
                    if queued_task is None:
                        continue
                    with contextlib.suppress(Exception):
                        queued_ids.add(int(getattr(queued_task, "object_id", -1)))
                fill_tasks = self._collect_recaption_fill_tasks(
                    limit=max(0, fill_target - len(batch)),
                    exclude_object_ids=queued_ids,
                )
                if fill_tasks:
                    # Lightweight counter for observability.
                    prev = int(self.scene_state.get("recaption_enqueued_total", 0) or 0)
                    now_total = prev + len(fill_tasks)
                    self.scene_state["recaption_enqueued_total"] = int(now_total)
                    print(f"[CaptionWorker] recaption_enqueued_total={now_total} batch_add={len(fill_tasks)}")
                    batch.extend(fill_tasks)

            # Process collected batch. Any exception here MUST NOT kill the
            # daemon thread: that silently wedges every subsequent producer
            # `put` once the bounded tasks_queue fills (see services.py:169).
            # On failure we emit empty results so producers can release their
            # in-flight slots via drain_results() and continue.
            try:
                self._process_caption_batch(batch)
            except Exception:
                import traceback
                print(
                    "[CaptionWorker] _process_caption_batch raised; emitting empty results and continuing:\n"
                    + traceback.format_exc()
                )
                for failed_task in batch:
                    if failed_task is None:
                        continue
                    try:
                        empty_result = ObjectCaptionResult(
                            object_id=int(getattr(failed_task, "object_id", -1)),
                            caption="",
                            used_view_indices=[],
                            empty_reason="batch_exception",
                            is_recaption=bool(getattr(failed_task, "is_recaption", False)),
                        )
                        with contextlib.suppress(Exception):
                            self.results_queue.put(empty_result, timeout=5.0)
                    except Exception:
                        # Never let result-emission failure kill the worker either.
                        continue

            for _ in range(queue_task_count):
                with contextlib.suppress(ValueError):
                    self.tasks_queue.task_done()

    def _get_spatial_context_instruction(self) -> str:
        """
        Reusable description of optional spatial inputs for the task prompt.

        When caption spatial context is enabled, returns bullet(s) describing
        OBJECT APPROXIMATE SIZE (always when spatial is on) and optionally
        OBJECT POSITION (when caption_spatial_context_include_position is True).
        Returns empty string when spatial context is disabled.
        """
        if not self._caption_spatial_context:
            return ""
        lines: List[str] = []
        if self._caption_spatial_context_include_position:
            lines.append(
                "- OBJECT POSITION (world frame): [x, y, z] in meters. "
                "Use as supporting context for location if provided."
            )
        lines.append(
            "- OBJECT APPROXIMATE SIZE: three axis extents in meters (largest to smallest). "
            "Use size as supporting context (e.g., to disambiguate small vs large objects, "
            "or to understand relative scale) but always prioritize visual evidence from the image."
        )
        return "\n" + "\n".join(lines) + "\n"

    def _append_spatial_context_if_enabled(self, block: str) -> str:
        """
        Append the spatial context instruction to a task block when enabled.
        Called from _build_task_block so all versions get spatial context when enabled.
        For a custom placement, use _get_spatial_context_instruction() and insert the
        returned string where needed in your version's block.
        """
        suffix = self._get_spatial_context_instruction()
        return block + suffix if suffix else block

    def _build_task_block(self) -> str:
        if self._caption_version == 1:
            base = self._build_task_block_v1()
        elif self._caption_version == 2:
            base = self._build_task_block_v2()
        elif self._caption_version == 3:
            base = self._build_task_block_v3()
        elif self._caption_version == 4:
            base = self._build_task_block_v4()
        elif self._caption_version == 5:
            base = self._build_task_block_v5()
        elif self._caption_version == 6:
            base = self._build_task_block_v6()
        elif self._caption_version == 7:
            base = self._build_task_block_v7()
        elif self._caption_version == 8:
            base = self._build_task_block_v8()
        elif self._caption_version == 9:
            base = self._build_task_block_v9()
        elif self._caption_version == 10:
            base = self._build_task_block_v10()
        elif self._caption_version == 11:
            base = self._build_task_block_v11()
        elif self._caption_version == 12:
            base = self._build_task_block_v12()
        elif self._caption_version == 13:
            base = self._build_task_block_v13()
        elif self._caption_version == 14:
            base = self._build_task_block_v14()
        elif self._caption_version == 15:
            base = self._build_task_block_v15()
        elif self._caption_version == 16:
            base = self._build_task_block_v16()
        elif self._caption_version == 17:
            base = self._build_task_block_v17()
        elif self._caption_version == 18:
            base = self._build_task_block_v18()
        elif self._caption_version == 19:
            base = self._build_task_block_v19()
        elif self._caption_version == 20:
            base = self._build_task_block_v20()
        else:
            raise ValueError(f"Invalid caption version: {self._caption_version}")
        return self._append_spatial_context_if_enabled(base)

    def _build_high_quality_task_block(self) -> str:
        if self._caption_version == 12:
            return self._build_task_block_v12_high_quality()
        if self._caption_version in (19, 20):
            return self._build_task_block_v19_high_quality()
        return self._build_task_block()

    def _build_task_block_v1(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robot object captioner.ss\n\nPrimary goal: write a short, accurate caption for the TARGET OBJECT"
            " in the bounding box.\n\n You are given:\n  - An IMAGE.\n  - TARGET BOUNDING BOX:"
            ' <box>(x1,y1),(x2,y2)</box>.\n\nOutput MUST be strict JSON only (no extra text):\n{"new_caption":'
            ' "str"}\n\nRules for "new_caption":\n - Describe only the TARGET object in the bounding box: category +'
            ' 1-3 key attributes.\n - If uncertain about category, use "cat1 or cat2" (2-3 options max). Example: "red'
            ' box or coffee bag".\n - No opinions. No guessing invisible attributes.\n Priors:\n\n - This environment'
            " contains common household, work, and camping items. Many targets belong to this ontology:"
            " clothing/footwear (jacket, shoes, tux, hat, socks, tie), tools (measuring tools, hammer, wrench,"
            " screwdriver, drill), containers (backpack, wallet, jar, box, safe, igloo lunchbox, cooler), food/drinks"
            " (fruits, vegetables, bread, cheese, drink, bottle, mug, cup), kitchen items (toaster, coffee machine,"
            " teapot, bowl, coffee bag), electronics (phone, camera, charger, headlamp), hygiene/medical (toothbrush,"
            " medicine bottle, pills, first aid kit, sunscreen), furniture/props (step stool, bench, ironing board,"
            " trash can, wood block, fireplace), keys/buttons (key, colored wall button), animals (dog, cat, bird,"
            " duck), buildings (hospital, restaurant). \n - Synthetic low-quality image: details may be missing. If"
            " unsure use 'cat1 or cat2'. "
        )

    def _build_task_block_v2(self) -> str:
        self._caption_expects_json = True
        return (
            "You extract structured object information from an image for the TARGET OBJECT in the bounding box."
            "The objects are all from isaac simulation environment."
            "Describe only the TARGET OBJECT and ignore other objects in the image."
            "Return JSON with exactly these fields:"
            "{"
            '"label": "<noun phrase>",'
            '"short_description": "<concise sentence>",'
            "}"
            'Rules for "label":'
            " - Must be a noun phrase."
            " - Lowercase."
            " - One or two words."
            'Rules for "short_description":'
            " - Single short sentence with additional information not present in the label."
            " - Present tense."
            " - Do not add or infer information."
        )

    def _build_task_block_v3(self) -> str:
        self._caption_expects_json = True
        return (
            "You extract structured object information from an image for the TARGET OBJECT in the bounding box."
            "The objects are all from isaac simulation environment."
            "Describe only the TARGET OBJECT and ignore other objects in the image."
            "Return JSON with exactly these fields:"
            "{"
            '"label": "<noun phrase>",'
            '"key_words": "<word1>, <word2>, ...",'
            "}"
            'Rules for "label":'
            " - Must be a noun phrase."
            " - Lowercase."
            " - One or two words."
            'Rules for "key_words":'
            " - List of nouns separated by commas."
        )

    def _build_task_block_v4(self) -> str:
        self._caption_expects_json = True
        return (
            " You are a robot object captioner. Given an image from a simulation environment, write short, accurate"
            " captions for the TARGET OBJECT in the bounding box. The images are from Isaac Sim simulation"
            " environment. The description of the objects are important for completing finding tasks. Bounding box"
            " format: <box>(x1,y1),(x2,y2)</box>. Output MUST be strict JSON only (no extra text):"
            ' {"new_caption":"str"} Rules: - Describe only the TARGET object in the bounding box - Include 1-3 key'
            " attributes that best describe the object."
        )

    def _build_task_block_v5(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robotic vision system generating semantic metadata for a scene graph. Task: Describe the TARGET"
            " OBJECT within the provided bounding box <box>(x1,y1),(x2,y2)</box>.The image is from an Isaac Sim"
            ' environment. Requirements for the "new_caption":1. Precise Name: Use the specific object name (e.g.,'
            ' "Green Apple", not just "Fruit").2. Visual Attributes: Mention 1-2 distinguishing features (color,'
            ' material, or state like "half-full" or "opened").3. Functional Category: Explicitly state the object\'s'
            ' purpose to support task-based queries (e.g., "edible food", "cooking tool", "liquid container"). Output'
            ' MUST be strict JSON:{"object_name": "str", "attributes": ["str", "str"], "functional_category": "str",'
            ' "new_caption": "str"}'
        )

    def _build_task_block_v6(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robotic perception module. Output strict JSON only."
            "Describe the TARGET OBJECT in <box> using minimal, high-impact words."
            "Rules:"
            "1. Name: Use the most specific noun possible."
            "2. Properties: List only 2-3 visual properties (color, material, or state)."
            '3. Task Hint: Mention if it is "Edible", "Wearable", "Tool", or "Storage".'
            'Output format: {"new_caption": "Half-full glass water bottle; Liquid Container; Graspable"}'
        )

    def _build_task_block_v7(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a specialized vision-to-graph tagger for a mobile robot."
            "Describe the TARGET OBJECT in <box>. Use strict JSON."
            "Mandatory Fields:"
            '1. "name": Specific identity (e.g., "Yellow Key", "Log", "Sleeping Bag").'
            '2. "category": High-level group (e.g., "Tool", "Furniture", "Navigation Landmark", "Food/Drink").'
            '3. "affordance": Primary use (e.g., "Sit-on-able", "Open-with-key", "Storage", "Orientation").'
            '4. "attributes": List color, material, and state (e.g., ["metal", "yellow"], ["wooden", "heavy"]).'
            "Output format:"
            '{"new_caption": "NAME; CATEGORY; AFFORDANCE; ATTRIBUTES"}'
        )

    def _build_task_block_v8(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robot object captioner.\n\nPrimary goal: write a short, accurate caption for the TARGET OBJECT"
            " in the provided bounding box.\n\n You are given:\n - An IMAGE.\n - TARGET BOUNDING BOX:"
            ' <box>(x1,y1),(x2,y2)</box>.\n\n Output MUST be strict JSON only (no extra text): {"new_caption":'
            ' "str"}\n\n Rules for "new_caption":\n  - Describe ONLY the target object in the bounding box: category +'
            ' 1-3 key attributes.\n - If uncertain about category, use "cat1 or cat2" (2-3 options max). Example: "red'
            ' box or coffee bag".\n  - No opinions. No guessing invisible attributes. Focus on the object in the'
            " bounding box and ignore other objects.\n  This environment contains common household, work, and camping"
            " items. Many targets belong to this ontology: - clothing/footwear: jacket, shoes, tux, hat, socks, tie  -"
            " tools: measuring tools, hammer, wrench, screwdriver, drill - containers: backpack, wallet, jar, box,"
            " safe, igloo lunchbox, cooler - food/drinks: fruits, vegetables, bread, cheese, drink, bottle, mug, cup -"
            " kitchen items: toaster, coffee machine, teapot, bowl, coffee bag - electronics: phone, camera, charger,"
            " headlamp - keys/buttons: key, colored wall button - animals: dog, cat, bird, duck - buildings: hospital,"
            " restaurant The images are from Isaac Sim simulation environment and can be low-quality. Details may be"
            " missing or simplified."
        )

    def _build_task_block_v9(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robotic vision module. Identify the object in <box>.\n"
            "The environment is Isaac Sim (low-quality or simplified visuals).\n\n"
            "TASK: Provide the 1-3 most likely object names in order of probability.\n"
            'FORMAT: Output strict JSON: {"options": "label1 | label2 | label3", "attr": "color, state"}\n\n'
            "RULES:\n"
            "- Use '|' as a separator for uncertainty. Most likely first.\n"
            "- If certain, provide only 1 name. If ambiguous, provide 2-3.\n"
            "- Keep labels short (e.g., 'soda can' not 'aluminum can of soda').\n"
            "- Use categories from the ontology: [containers, tools, food, electronics, landmarks].\n"
            '\nExample: {"options": "coffee bag | red box", "attr": "red, plastic"}'
        )

    def _build_task_block_v10(self) -> str:
        self._caption_expects_json = True
        return (
            "### CONTEXT\n"
            "You are a robotic vision module in an Isaac Sim environment.\n"
            "Identify the TARGET OBJECT located in the bounding box: <box>(x1,y1),(x2,y2)</box>.\n\n"
            "### TASK\n"
            "Provide 1-3 likely identities for the object in order of probability.\n"
            "Include key physical attributes (color, state, material).\n\n"
            "### ONTOLOGY HINTS\n"
            "- Containers: box, cooler, lunchbox, jar, bottle\n"
            "- Tools: compass, measuring tape, key, camera\n"
            "- Furniture/Landmarks: log, campfire, hospital, tent\n\n"
            "### OUTPUT RULES\n"
            "- Output MUST be strict JSON.\n"
            "- If uncertain, list multiple labels separated by ' or '.\n"
            "- Focus ONLY on the object inside the box.\n\n"
            'JSON FORMAT: {"labels": "name1 or name2", "attributes": "color, material, state"}'
        )

    def _build_task_block_v11(self) -> str:
        self._caption_expects_json = True
        return (
            "### TASK\n"
            "Identify the TARGET object in <box>(x1,y1),(x2,y2)</box> within this Isaac Sim scene.\n\n"
            "### REQUIREMENTS\n"
            "1. Identify the specific name (e.g., 'Red Apple', 'Metal Key').\n"
            "2. List 2-3 visual attributes (color, material, or state) separated by commas.\n"
            "3. Provide 1-2 functional categories (e.g., 'edible', 'tool', 'container').\n"
            "4. If uncertain, list 2-3 possible names separated by 'or'.\n\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with strict JSON:\n"
            "{\n"
            '  "name": "name1 or name2",\n'
            '  "attr": "attribute, attribute, attribute",\n'
            '  "usage": "functional category"\n'
            "}"
        )

    def _build_task_block_v12(self) -> str:
        self._caption_expects_json = True
        return (
            "### TASK\n"
            "Identify the TARGET object in <box>(x1,y1),(x2,y2)</box> within this Isaac Sim scene.\n\n"
            "### REQUIREMENTS\n"
            "1. Identify the specific name.\n"
            "2. List 2-3 visual attributes (color, material, or state) separated by commas.\n"
            "3. Provide 1-2 functional categories.\n"
            "4. If uncertain, list 2-3 possible names separated by 'or'.\n\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with strict JSON:\n"
            "{\n"
            '  "name": "name1 or name2",\n'
            '  "attr": "attribute, attribute, attribute",\n'
            '  "usage": "functional category"\n'
            "}"
        )

    def _build_task_block_v12_high_quality(self) -> str:
        self._caption_expects_json = True
        return (
            "### TASK\n"
            "You are performing CAPTIONING for the same TARGET object using MULTI-VIEW crops.\n"
            "Identify one robust caption that is consistent across all provided views of the object.\n\n"
            "### REQUIREMENTS\n"
            "1. Use evidence shared by all views; ignore view-specific artifacts and occluders.\n"
            "2. Prefer the most specific object name supported by the views.\n"
            "3. List 2-3 stable visual attributes (color, material, or state).\n"
            "4. Provide 1-2 functional categories.\n"
            "5. If still ambiguous, provide 2-3 names separated by 'or', most likely first.\n\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with strict JSON:\n"
            "{\n"
            '  "name": "name1 or name2",\n'
            '  "attr": "attribute, attribute, attribute",\n'
            '  "usage": "functional category"\n'
            "}"
        )

    def _build_task_block_v13(self) -> str:
        self._caption_expects_json = True
        return (
            "### TASK\n"
            "Identify the TARGET object in <box>(x1,y1),(x2,y2)</box> within this Isaac Sim scene.\n\n"
            "### INPUTS\n"
            "You are given:\n"
            "- An IMAGE with the target object highlighted by a bounding box.\n"
            "- TARGET BOUNDING BOX in <box> format.\n"
            "\n### REQUIREMENTS\n"
            "1. Identify the specific name.\n"
            "2. List 2-3 visual attributes (color, material, or state) separated by commas.\n"
            "3. Provide 1-2 functional categories.\n"
            "4. If uncertain, list 2-3 possible names separated by 'or'.\n\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with strict JSON:\n"
            "{\n"
            '  "name": "name1 or name2",\n'
            '  "attr": "attribute, attribute, attribute",\n'
            '  "usage": "functional category"\n'
            "}"
        )

    def _build_task_block_v14(self) -> str:
        self._caption_expects_json = False
        return (
            "### TASK\n"
            "Identify the TARGET object in <box>(x1,y1),(x2,y2)</box> within this Isaac Sim scene.\n\n"
            "### INPUTS\n"
            "You are given:\n"
            "- An IMAGE with the target object highlighted by a bounding box.\n"
            "- TARGET BOUNDING BOX in <box> format.\n"
            "\n### REQUIREMENTS\n"
            "1. Identify the specific name.\n"
            "2. List 2-3 visual attributes (color, material, or state) separated by commas.\n"
            "3. Provide 1-2 functional categories.\n"
            "4. If uncertain, list 2-3 possible names separated by 'or'.\n\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with a string containing the name, attributes, and usage separated by ':' :\n"
            "name1 or name2 : attribute, attribute, attribute : functional category"
        )

    def _build_task_block_v15(self) -> str:
        self._caption_expects_json = False
        return (
            "You are a robotic vision module.\n"
            "### TASK\n"
            "Describe the only the TARGET object within the bounding box <box>(x1,y1),(x2,y2)</box>."
            "The image is from an Isaac Sim simulation environment.\n"
            "### INPUTS\n"
            "- An IMAGE with the target object.\n"
            "- TARGET BOUNDING BOX in <box> format highlighting the interesting object.\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with a string containing the short description of the object\n"
            "If the object is ambiguous you can use 'or' to list a few (up to 3) multiple possible descriptions."
        )

    def _build_task_block_v16(self) -> str:
        self._caption_expects_json = False
        return (
            "You are a robotic vision module.\n"
            "### TASK\n"
            "Caption the TARGET OBJECT within the bounding box <box>(x1,y1),(x2,y2)</box>."
            "The image is from an Isaac Sim simulation environment.\n"
            "### INPUTS\n"
            "- An IMAGE with the target object.\n"
            "- TARGET BOUNDING BOX in <box> format highlighting the interesting object.\n"
            "### OUTPUT FORMAT\n"
            "Respond ONLY with a string containing the caption of the object\n"
            "If the object is ambiguous you can use 'or' to list a few (up to 3) multiple possible captions."
        )

    def _build_task_block_v17(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robot object captioner.\n\nPrimary goal: write a short, accurate caption for the TARGET OBJECT"
            " in the bounding box.\n\n"
            " You are given:\n  - An IMAGE.\n  - TARGET BOUNDING BOX:"
            ' <box>(x1,y1),(x2,y2)</box>.\n\nOutput MUST be strict JSON only (no extra text):\n{"new_caption":'
            ' "str"}\n\nRules for "new_caption":\n - Describe only the TARGET object in the bounding box: category'
            " + 1-3 key attributes.\n  * Attributes: color, material, shape, pattern, parts, state"
            " (open/closed/empty/full), text/label if readable.\n"
            " - No opinions. No guessing invisible attributes. No"
            " description of objects outside the bounding box.\n"
        )

    def _build_task_block_v18(self) -> str:
        self._caption_expects_json = True
        return (
            "You are a robot object captioner.\n\nPrimary goal: write a short, accurate caption for the TARGET OBJECT"
            " in the bounding box.\n\n You are given:\n  - An IMAGE.\n  - TARGET BOUNDING BOX:"
            ' <box>(x1,y1),(x2,y2)</box>.\n\nOutput MUST be strict JSON only (no extra text):\n{"new_caption": "str",'
            ' "too_blurry": bool}\n\nRules for "new_caption":\n - Describe only the TARGET object in the bounding box:'
            " category + 1-3 key attributes.\n  * Attributes: color, material, shape, pattern, parts, state"
            " (open/closed/empty/full), text/label if readable.\n - No opinions. No guessing invisible attributes. No"
            " description of objects outside the bounding box.\n - If the image is too blurry, set 'too_blurry' to"
            " True, otherwise set it to False.\n"
        )

    def _build_task_block_v19(self) -> str:
        return self._build_task_block_v20()

    def _build_task_block_v20(self) -> str:
        self._caption_expects_json = True
        visual_instruction = (
            "Visual targeting rule:\n"
            "- The image may be a normal target crop, a mask-highlighted crop, or a composite image.\n"
            "- In a composite image, the left panel is a small scene thumbnail for context and the right panel is "
            "the target crop.\n"
            "- A colored mask or outline is an annotation showing the target object pixels; it is not an attribute "
            "of the object.\n"
            "- Caption only the highlighted or boxed target object. Use unhighlighted objects only as context or "
            "occlusion evidence.\n\n"
        )
        permissive_extra = ""
        if self._caption_prompt_variant in {"more_permissive", "permissive", "hm3d_permissive"}:
            permissive_extra = (
                "Extra permissive keep rule:\n"
                '- Prefer "decision": "keep" for a single visible physical object, even if it is vague, far, '
                "synthetic, low-detail, or partly occluded. Reject only unusable crops, surfaces, subparts without "
                "their own object category, or clear multi-object boxes.\n\n"
            )
        return (
            "You are a robot object captioner for a 3D scene memory system.\n\n"
            "Your task is to inspect the TARGET OBJECT inside the bounding box and decide whether it should be "
            "kept or dropped for object captioning and scene memory.\n\n"
            "You are given:\n"
            "- An IMAGE.\n"
            "- TARGET BOUNDING BOX: <box>(x1,y1),(x2,y2)</box>.\n\n"
            + visual_instruction
            +
            "Output MUST be strict JSON only.\n"
            "Do not output markdown, comments, explanations, or extra keys.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "category": "string",\n'
            '  "supercategory": "string",\n'
            '  "attributes": ["string"],\n'
            '  "description": "string",\n'
            '  "decision": "keep"\n'
            "}\n\n"
            'The "decision" field must be exactly one of: "keep" or "drop".\n\n'
            "Decision rule:\n"
            '- Set "decision": "keep" if the bounding box contains a recognizable standalone object category useful '
            "for mapping or captioning.\n"
            '- Keep the object even when the crop is small, low-resolution, slightly blurry, far from the camera, '
            "partially cut off, touching the image border, synthetic-looking, or attached to a wall, floor, or "
            "furniture surface.\n"
            '- Be permissive for synthetic rendered scenes such as HM3D: if the object shape and context make a '
            'reasonable object category identifiable, set "decision": "keep" and choose the best concrete category.\n'
            "- If the exact fine-grained category is uncertain but the target is clearly one object, keep it and "
            "choose the most reasonable visible category.\n"
            "- If the detector label appears wrong but the boxed object is real and recognizable, still keep it and "
            "output the corrected visible category.\n"
            "- If one dominant object is visible inside the box and other pixels are just background or incidental "
            'context, set "decision": "keep".\n\n'
            + permissive_extra
            +
            '- Set "decision": "drop" only when the crop is genuinely unusable for mapping or captioning: no '
            "discernible standalone object, mostly background or surface, random texture patch, non-distinct "
            "fragment, merged group of multiple comparable objects, or extremely occluded/blurred so no obvious "
            "category can reasonably be assigned.\n"
            "- Drop subparts such as a chair leg, table edge, sofa cushion fragment, shelf corner, wall patch, or "
            "floor region unless the subpart itself has a recognizable object category useful for mapping.\n"
            '- Never output null for "decision"; it must always be "keep" or "drop".\n\n'
            "Rules for kept objects:\n"
            "- Describe only the target object inside the bounding box.\n"
            "- Ignore background and objects outside the bounding box.\n"
            '- "category" should be a short singular noun phrase, such as "chair", "table", "monitor", "cabinet", '
            '"sofa", "trash can", "fire alarm", or "door handle".\n'
            '- "supercategory" should be a broad type, such as "furniture", "appliance", "electronics", '
            '"container", "fixture", "safety device", "textile", "plant", "personal item", "tool", or "other".\n'
            '- "attributes" should contain 1 to 5 short visible attributes: color, material, shape, pattern, parts, '
            "state, readable text, or distinctive visual details.\n"
            '- "description" should be one short phrase combining category and key attributes.\n'
            "- Use only visible evidence. Do not guess hidden attributes, function, brand, material, or text.\n"
            '- Do not use alternatives like "chair or stool"; choose the best category when the target is a clear '
            "single object.\n\n"
            "Rules for dropped objects:\n"
            '- If "decision" is "drop", output exactly:\n'
            '{"category":"unknown","supercategory":"unknown","attributes":[],"description":"","decision":"drop"}'
            "\n\n"
            "Examples:\n\n"
            '{"category":"chair","supercategory":"furniture","attributes":["black","wooden","with armrests"],'
            '"description":"black wooden chair with armrests","decision":"keep"}\n\n'
            '{"category":"monitor","supercategory":"electronics","attributes":["black","rectangular","flat screen"],'
            '"description":"black rectangular flat-screen monitor","decision":"keep"}\n\n'
            '{"category":"fire alarm","supercategory":"safety device","attributes":["red","wall-mounted","with FIRE text"],'
            '"description":"red wall-mounted fire alarm with FIRE text","decision":"keep"}\n\n'
            '{"category":"unknown","supercategory":"unknown","attributes":[],"description":"","decision":"drop"}'
        )

    def _build_task_block_v19_high_quality(self) -> str:
        return self._build_task_block_v19()

    def _build_examples_block(self) -> str:
        """Few-shot examples covering diverse merge behaviors with minimal redundancy."""
        return ""

    def _process_caption_batch(self, batch: List[ObjectCaptionTask]) -> None:
        """
        Prepare inputs, run caption model, and emit results for the batch.
        """
        t_prepare_start = time.perf_counter()
        batch_len = len([b for b in batch if b is not None])
        self._batch_count += 1
        self._batch_size_sum += batch_len

        (
            images,
            meta,
            boxes,
            is_recaption_flags,
            crop_images_uint8,
            crop_encodings,
            prepared_task_indices,
            skipped_task_reasons,
        ) = self._prepare_caption_inputs(batch)
        prepare_dur = time.perf_counter() - t_prepare_start
        self._record_time("prepare_inputs", prepare_dur)
        if self.debug:
            num_imgs = len(images) if images is not None else 0
            print(f"[CaptionWorker] Prepared {num_imgs} images for {len(batch)} tasks")

        # Retry tasks that yielded no usable images
        no_image_task_ids: set[int] = set()
        for task in batch:
            if task is None:
                continue
            if task.object_index not in prepared_task_indices:
                no_image_task_ids.add(int(task.object_id))
                self._maybe_retry_task(task, reason="no_images")

        caption_payloads: List[Dict[str, Any]] = []
        if images:
            t0 = time.perf_counter()
            try:
                caption_payloads = self._run_caption_model(images, boxes, is_recaption_flags=is_recaption_flags)
            except Exception as exc:
                if self.debug:
                    print(f"[CaptionWorker] Caption model failed: {exc}")
                caption_payloads = []
            else:
                t1 = time.perf_counter()
                self._gen_batch_count += 1
                self._gen_time_sum += t1 - t0
                self._record_time("caption_total", t1 - t0)
                if self.debug:
                    print(f"#images: {len(images)}")
                    print(
                        "[CaptionWorker] batch timings prepare={:.2f}ms caption_total={:.2f}ms".format(
                            prepare_dur * 1000.0, (t1 - t0) * 1000.0
                        )
                    )

        siglip2_cls_by_meta: dict[tuple[int, int], List[float]] = {}
        qwen3_vl_cls_by_meta: dict[tuple[int, int], List[float]] = {}
        per_object_prepared_views: dict[int, List[int]] = {}
        for obj_idx, view_idx in meta:
            per_object_prepared_views.setdefault(int(obj_idx), []).append(int(view_idx))

        print(
            f"[CaptionWorker] SigLIP2 enabled: {self._siglip2_enabled}, crop_images_uint8:"
            f" {crop_images_uint8 is not None}"
        )

        if self._siglip2_enabled and crop_images_uint8:
            t_siglip2_start = time.perf_counter()
            try:
                print(f"[CaptionWorker] Running SigLIP2 on {len(crop_images_uint8)} crops")
                cls_vecs = self._run_siglip2_cls_embeddings(crop_images_uint8, crop_encodings)
                nonempty = sum(1 for v in cls_vecs if v)
                print(f"[CaptionWorker] SigLIP2 returned {len(cls_vecs)} vecs (nonempty={nonempty})")
            except Exception as exc:
                print(f"[CaptionWorker] SigLIP2 embedding failed: {exc}")
                cls_vecs = []
            siglip2_dur = time.perf_counter() - t_siglip2_start
            self._record_time("siglip2_cls", siglip2_dur)
            if cls_vecs:
                print("[CaptionWorker] siglip2_total={:.2f}ms".format(siglip2_dur * 1000.0))
                for (obj_idx, view_idx), vec in zip(meta, cls_vecs):
                    if vec:
                        siglip2_cls_by_meta[(int(obj_idx), int(view_idx))] = list(vec)

        if self._qwen3_vl_embed_enabled and crop_images_uint8:
            t_qwen3_vl_start = time.perf_counter()
            try:
                qwen3_vl_vecs = self._run_qwen3_vl_cls_embeddings(crop_images_uint8, crop_encodings)
            except Exception as exc:
                if self.debug:
                    print(f"[CaptionWorker] Qwen3-VL embedding failed: {exc}")
                qwen3_vl_vecs = []
            qwen3_vl_dur = time.perf_counter() - t_qwen3_vl_start
            self._record_time("qwen3_vl_cls", qwen3_vl_dur)
            nonempty = sum(1 for v in qwen3_vl_vecs if v)
            if self._qwen3_vl_embed_runtime_log:
                print(
                    "[CaptionWorker] qwen3_vl_embedding_total={:.2f}ms backend={} nonempty={}/{}".format(
                        qwen3_vl_dur * 1000.0,
                        self._qwen3_vl_embed_backend,
                        nonempty,
                        len(qwen3_vl_vecs),
                    )
                )
            for (obj_idx, view_idx), vec in zip(meta, qwen3_vl_vecs):
                if vec:
                    qwen3_vl_cls_by_meta[(int(obj_idx), int(view_idx))] = list(vec)

        # Aggregate captions per object (keep most recent)
        per_object_caps: dict[int, List[str]] = {}
        per_object_payloads: dict[int, List[Dict[str, Any]]] = {}
        per_object_views: dict[int, List[int]] = {}
        per_object_error_reason: dict[int, str] = {}
        for idx, (obj_idx, view_idx) in enumerate(meta):
            payload = caption_payloads[idx] if idx < len(caption_payloads) else {}
            if isinstance(payload, dict):
                per_object_payloads.setdefault(obj_idx, []).append(payload)
            cap = str(payload.get("caption", "") or "")
            error_reason = payload.get("error_reason")
            if cap:
                per_object_caps.setdefault(obj_idx, []).append(cap)
                per_object_views.setdefault(obj_idx, []).append(view_idx)
            if error_reason and obj_idx not in per_object_error_reason:
                per_object_error_reason[obj_idx] = str(error_reason)

        results: List[ObjectCaptionResult] = []
        embed_texts: List[str] = []
        embed_indices: List[int] = []
        result_object_indices: List[Optional[int]] = []

        for task in batch:
            if task is None:
                continue
            resolved_idx = self._resolve_object_index(task)
            skip_reason = skipped_task_reasons.get(int(task.object_id))
            if skip_reason:
                result = ObjectCaptionResult(
                    object_id=task.object_id,
                    caption="",
                    used_view_indices=[],
                    merge_object_ids=[],
                    empty_reason=str(skip_reason),
                    is_recaption=bool(getattr(task, "is_recaption", False)),
                )
                results.append(result)
                result_object_indices.append(resolved_idx)
                continue
            caps = per_object_caps.get(resolved_idx, []) if resolved_idx is not None else []
            views = per_object_views.get(resolved_idx, []) if resolved_idx is not None else []
            if caps:
                caption_text = caps[-1]
                used_views = views[-1:] if views else []
            else:
                caption_text = ""
                used_views = []
            last_payload: Dict[str, Any] = {}
            payloads = per_object_payloads.get(resolved_idx, []) if resolved_idx is not None else []
            if payloads:
                maybe = payloads[-1]
                if isinstance(maybe, dict):
                    last_payload = maybe
            result_is_clear: Optional[bool] = None
            result_decision: Optional[str] = None
            if isinstance(last_payload, dict) and "is_clear_object" in last_payload:
                raw_clear = last_payload.get("is_clear_object")
                if isinstance(raw_clear, bool):
                    result_is_clear = raw_clear
            if isinstance(last_payload, dict):
                raw_decision = str(last_payload.get("decision") or "").strip().lower()
                if raw_decision in {"keep", "drop"}:
                    result_decision = raw_decision
            if result_decision is None and result_is_clear is not None:
                result_decision = "keep" if result_is_clear else "drop"

            chosen_view_for_siglip2: Optional[int] = None
            if used_views:
                chosen_view_for_siglip2 = int(used_views[-1])
            elif resolved_idx is not None:
                cand = per_object_prepared_views.get(int(resolved_idx), [])
                if cand:
                    chosen_view_for_siglip2 = int(cand[-1])

            if not caption_text:
                if result_is_clear is False:
                    empty_reason = per_object_error_reason.get(resolved_idx, "unclear_object")
                elif resolved_idx is None:
                    empty_reason = "stale_or_inactive"
                elif task.object_id in no_image_task_ids:
                    empty_reason = "no_image"
                else:
                    empty_reason = per_object_error_reason.get(resolved_idx, "vlm_failure_or_empty")
                if result_is_clear is not False:
                    self._maybe_retry_task(task, reason="empty_caption")
            else:
                empty_reason = None
            result = ObjectCaptionResult(
                object_id=task.object_id,
                caption=caption_text,
                used_view_indices=used_views,
                merge_object_ids=[],
                empty_reason=empty_reason,
                is_clear_object=result_is_clear,
                decision=result_decision,
                is_recaption=bool(getattr(task, "is_recaption", False)),
            )
            if caption_text:
                with contextlib.suppress(Exception):
                    result.category = str(last_payload.get("category", "") or "").strip() or None
                with contextlib.suppress(Exception):
                    result.supercategory = str(last_payload.get("supercategory", "") or "").strip() or None
                with contextlib.suppress(Exception):
                    cands = last_payload.get("category_candidates")
                    if isinstance(cands, list):
                        result.category_candidates = [str(x or "").strip() for x in cands if str(x or "").strip()]
                with contextlib.suppress(Exception):
                    attrs = last_payload.get("key_attributes")
                    if isinstance(attrs, list):
                        result.key_attributes = [str(x or "").strip() for x in attrs if str(x or "").strip()]
            embedding_text = self._caption_embedding_text_for_result(result)
            if embedding_text:
                # Track the exact text that caption embeddings are computed from so we can
                # recompute if merges later modify the final semantic string.
                result.caption_embedding_text = embedding_text
            if resolved_idx is not None and chosen_view_for_siglip2 is not None:
                result.siglip2_cls_embedding = siglip2_cls_by_meta.get(
                    (int(resolved_idx), int(chosen_view_for_siglip2))
                )
                result.qwen3_vl_cls_embedding = qwen3_vl_cls_by_meta.get(
                    (int(resolved_idx), int(chosen_view_for_siglip2))
                )

            results.append(result)
            result_object_indices.append(resolved_idx)
            if embedding_text:
                embed_texts.append(embedding_text)
                embed_indices.append(len(results) - 1)

        if embed_texts:
            t_embed_start = time.perf_counter()
            caption_embeddings = self._run_caption_embeddings(embed_texts)
            embed_dur = time.perf_counter() - t_embed_start
            self._record_time("caption_embed", embed_dur)
            print("[CaptionWorker] embedding_total={:.2f}ms".format(embed_dur * 1000.0))
            for result_idx, embedding in zip(embed_indices, caption_embeddings):
                results[result_idx].caption_embedding = embedding

        self._maybe_apply_caption_merges(results, result_object_indices)

        # If merges changed the caption text, the caption embedding is now stale (it was computed
        # on the pre-merge caption string). Recompute embeddings for the updated captions so
        # downstream retrieval uses an embedding that matches the published caption text.
        changed_indices: List[int] = []
        changed_texts: List[str] = []
        for i, res in enumerate(results):
            desired_embedding_text = self._caption_embedding_text_for_result(res)
            cap_emb_text = str(getattr(res, "caption_embedding_text", "") or "")
            if not desired_embedding_text or not cap_emb_text:
                continue
            if desired_embedding_text == cap_emb_text:
                continue
            if getattr(res, "caption_embedding", None) is None:
                continue
            changed_indices.append(i)
            changed_texts.append(desired_embedding_text)

        if changed_texts:
            try:
                t_embed_start = time.perf_counter()
                new_embeddings = self._run_caption_embeddings(changed_texts)
                embed_dur = time.perf_counter() - t_embed_start
                self._record_time("caption_embed_merge_fixup", embed_dur)
                if self.debug:
                    print(
                        "[CaptionWorker] merge_fixup_embeddings_total={:.2f}ms (count={})".format(
                            embed_dur * 1000.0, len(changed_texts)
                        )
                    )
                for idx, emb in zip(changed_indices, new_embeddings):
                    results[idx].caption_embedding = emb
                    results[idx].caption_embedding_text = self._caption_embedding_text_for_result(results[idx])
            except Exception as exc:
                if self.debug:
                    print(f"[CaptionWorker] Failed to recompute caption embeddings after merges: {exc}")

        for result in results:
            self.results_queue.put(result)

    # ------------------------------------------------------------------
    # Model helpers
    def _record_time(self, key: str, duration_s: float) -> None:
        """Accumulate timing statistics for a named section."""
        if duration_s < 0:
            return
        self._timing_sum[key] += duration_s
        self._timing_count[key] += 1

    def _siglip2_dbg(self, msg: str) -> None:
        if self._siglip2_debug or self.debug:
            print(f"[CaptionWorker][SigLIP2] {msg}")

    def _qwen3_vl_dbg(self, msg: str) -> None:
        if self._qwen3_vl_embed_debug or self.debug:
            print(f"[CaptionWorker][Qwen3-VL-Embedding] {msg}")

    @staticmethod
    def _normalize_embedding(vec: Sequence[float]) -> List[float]:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []
        denom = float(np.linalg.norm(arr) + 1e-12)
        return (arr / denom).tolist()

    def _uses_structured_caption_schema(self) -> bool:
        return int(self._caption_version) >= 19

    def _caption_ollama_format(self) -> Dict[str, Any]:
        return CAPTION_SCHEMA if self._uses_structured_caption_schema() else LEGACY_CAPTION_SCHEMA

    def _caption_response_format(self) -> Optional[Dict[str, Any]]:
        if not getattr(self, "_caption_expects_json", False):
            return None
        if self._uses_structured_caption_schema():
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_object_caption",
                    "schema": CAPTION_SCHEMA,
                },
            }
        return {"type": "json_object"}

    @staticmethod
    def _vllm_prefix_warmup_text(is_recaption: bool) -> str:
        if is_recaption:
            return (
                "NEW INPUT:\n"
                "INPUT VIEWS: 2 cropped views of the same object.\n"
                "TARGET BOUNDING BOX: "
            )
        return (
            "NEW INPUT:\n"
            "INPUT VIEWS: 1 cropped view.\n"
            "TARGET BOUNDING BOX: "
        )

    def _warm_vllm_prefix_cache(self, *, is_recaption: bool) -> None:
        """
        Prime vLLM Automatic Prefix Caching for caption batches.

        The OpenAI-compatible chat endpoint does not provide a multi-conversation
        batch field for independent VL completions. We therefore keep the normal
        per-crop requests, but first issue a tiny text-only request with the same
        system prompt and the same start of the user prompt. When vLLM is served
        with prefix caching enabled, subsequent crop requests can reuse the KV
        blocks for that shared prefix.
        """
        if self._caption_server != "vllm" or not self._vllm_prefix_warmup_enabled:
            return
        if requests is None:
            return

        messages = self._hq_base_messages if is_recaption and self._hq_base_messages else self._base_messages
        if not messages:
            return
        key_payload = {
            "model": self._vllm_model,
            "messages": messages,
            "is_recaption": bool(is_recaption),
            "prompt_prefix": self._vllm_prefix_warmup_text(is_recaption),
        }
        try:
            key = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = str((self._vllm_model, bool(is_recaption), self._vllm_prefix_warmup_text(is_recaption)))

        if key in self._vllm_warmed_prefixes:
            return

        with self._vllm_warmup_lock:
            if key in self._vllm_warmed_prefixes:
                return

            content = [{"type": "text", "text": self._vllm_prefix_warmup_text(is_recaption)}]
            payload: Dict[str, Any] = {
                "model": self._vllm_model,
                "messages": [*messages, {"role": "user", "content": content}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 1,
            }
            if self._vllm_disable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            headers = {"Content-Type": "application/json"}
            if self._vllm_api_key:
                headers["Authorization"] = f"Bearer {self._vllm_api_key}"

            try:
                response = self._get_session().post(
                    self._vllm_chat_url(),
                    json=payload,
                    headers=headers,
                    timeout=self._vllm_prefix_warmup_timeout_s,
                )
                response.raise_for_status()
                self._vllm_warmed_prefixes.add(key)
                if self.debug:
                    print(
                        "[CaptionWorker] warmed vLLM caption prefix "
                        f"recaption={bool(is_recaption)} tokens=max1"
                    )
            except Exception as exc:
                # Prefix warmup is an optimization; never fail captioning if APC
                # is disabled or the server rejects the tiny text-only request.
                self._vllm_warmed_prefixes.add(key)
                if self.debug:
                    print(f"[CaptionWorker] vLLM prefix warmup skipped/failed: {exc!r}")

    @staticmethod
    def _normalize_semantic_part(value: Any) -> str:
        return " ".join(str(value or "").strip().split())

    @classmethod
    def _build_structured_semantic_text(
        cls,
        *,
        category: Optional[str],
        supercategory: Optional[str],
        attributes: Optional[Sequence[str]],
        description: Optional[str],
    ) -> str:
        attr_parts: List[str] = []
        if isinstance(attributes, (list, tuple)):
            for attr in attributes:
                text = cls._normalize_semantic_part(attr)
                if text and text not in attr_parts:
                    attr_parts.append(text)
        parts = [
            cls._normalize_semantic_part(category),
            cls._normalize_semantic_part(supercategory),
            ", ".join(attr_parts),
            cls._normalize_semantic_part(description),
        ]
        return "; ".join(part for part in parts if part)

    def _caption_embedding_text_for_result(self, result: ObjectCaptionResult) -> str:
        caption = self._normalize_semantic_part(getattr(result, "caption", "") or "")
        if not self._uses_structured_caption_schema():
            return caption
        if getattr(result, "is_clear_object", None) is not True:
            return ""
        return self._build_structured_semantic_text(
            category=getattr(result, "category", None),
            supercategory=getattr(result, "supercategory", None),
            attributes=getattr(result, "key_attributes", None),
            description=caption,
        )

    @staticmethod
    def _set_siglip2_request_result(
        request: Dict[str, Any],
        *,
        ok: bool,
        error: str = "",
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        request["ok"] = bool(ok)
        request["error"] = str(error or "")
        request["embeddings"] = embeddings or []
        event = request.get("event")
        if isinstance(event, threading.Event):
            event.set()

    def request_siglip2_text_embeddings(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        timeout_s: Optional[float] = None,
        client_id: int = 0,
    ) -> tuple[bool, str, List[List[float]]]:
        timeout = float(timeout_s) if timeout_s is not None else self._siglip2_rpc_timeout_s
        if timeout <= 0.0:
            timeout = self._siglip2_rpc_timeout_s
        if not self._siglip2_enabled:
            return False, "siglip2_disabled", []

        text_list = [str(x or "").strip() for x in texts]
        if not text_list or not any(text_list):
            return True, "", [[] for _ in range(len(text_list))]
        if not self._siglip2_scheduler_thread.is_alive():
            try:
                vectors = self._run_siglip2_text_embeddings_impl(text_list, normalize=bool(normalize))
                return True, "", vectors
            except Exception as exc:
                return False, f"text_embed_failed: {exc}", []

        request: Dict[str, Any] = {
            "event": threading.Event(),
            "client_id": int(client_id),
            "normalize": bool(normalize),
            "texts": text_list,
        }
        try:
            self._siglip2_hi_queue.put_nowait(request)
        except queue.Full:
            return False, "busy", []

        event = request["event"]
        if not event.wait(timeout=timeout):
            return False, "timeout", []
        return bool(request.get("ok")), str(request.get("error") or ""), list(request.get("embeddings") or [])

    def _enqueue_siglip2_image_embeddings(
        self,
        crop_images_uint8: List[Any],
        crop_encodings: List[str],
        *,
        timeout_s: Optional[float] = None,
    ) -> List[List[float]]:
        timeout = float(timeout_s) if timeout_s is not None else self._siglip2_rpc_timeout_s
        if timeout <= 0.0:
            timeout = self._siglip2_rpc_timeout_s
        if not self._siglip2_scheduler_thread.is_alive():
            return self._run_siglip2_cls_embeddings_impl(crop_images_uint8, crop_encodings)
        request: Dict[str, Any] = {
            "event": threading.Event(),
            "images": crop_images_uint8,
            "encodings": crop_encodings,
        }
        try:
            self._siglip2_lo_queue.put_nowait(request)
        except queue.Full:
            self._siglip2_dbg("low-priority SigLIP2 queue full")
            return [[] for _ in range(len(crop_images_uint8))]

        event = request["event"]
        if not event.wait(timeout=timeout):
            self._siglip2_dbg("low-priority SigLIP2 request timeout")
            return [[] for _ in range(len(crop_images_uint8))]
        embeddings = request.get("embeddings")
        if not isinstance(embeddings, list):
            return [[] for _ in range(len(crop_images_uint8))]
        return embeddings

    def _collect_hi_siglip2_requests(self, *, block: bool) -> List[Dict[str, Any]]:
        requests_batch: List[Dict[str, Any]] = []
        if self._siglip2_scheduler_stop.is_set():
            return requests_batch

        first_req = None
        try:
            if block:
                first_req = self._siglip2_hi_queue.get(timeout=0.01)
            else:
                first_req = self._siglip2_hi_queue.get_nowait()
        except queue.Empty:
            return requests_batch

        if isinstance(first_req, dict):
            requests_batch.append(first_req)

        if self._siglip2_hi_batch_window_s <= 0.0:
            return requests_batch

        deadline = time.perf_counter() + self._siglip2_hi_batch_window_s
        while len(requests_batch) < self._siglip2_text_max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                req = self._siglip2_hi_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if isinstance(req, dict):
                requests_batch.append(req)
        return requests_batch

    def _process_hi_siglip2_requests(self, requests_batch: List[Dict[str, Any]]) -> None:
        if not requests_batch:
            return
        flat_texts: List[str] = []
        spans: List[tuple[int, int, Dict[str, Any]]] = []
        for request in requests_batch:
            texts = [str(x or "").strip() for x in list(request.get("texts") or [])]
            start = len(flat_texts)
            flat_texts.extend(texts)
            spans.append((start, len(flat_texts), request))

        if not flat_texts:
            for _start, _end, request in spans:
                self._set_siglip2_request_result(
                    request,
                    ok=True,
                    error="",
                    embeddings=[[] for _ in range(len(list(request.get("texts") or [])))],
                )
            return

        try:
            all_embeddings = self._run_siglip2_text_embeddings_impl(flat_texts, normalize=False)
            if len(all_embeddings) != len(flat_texts):
                raise RuntimeError(f"unexpected text embedding count {len(all_embeddings)} for {len(flat_texts)} texts")
            for start, end, request in spans:
                sub = all_embeddings[start:end]
                if bool(request.get("normalize", True)):
                    sub = [self._normalize_embedding(v) for v in sub]
                self._set_siglip2_request_result(request, ok=True, error="", embeddings=sub)
        except Exception as exc:
            err = f"text_embed_failed: {exc}"
            for _start, _end, request in spans:
                self._set_siglip2_request_result(request, ok=False, error=err, embeddings=[])

    def _process_lo_siglip2_request(self, *, block: bool) -> bool:
        if self._siglip2_scheduler_stop.is_set():
            return False
        try:
            request = self._siglip2_lo_queue.get(timeout=0.01) if block else self._siglip2_lo_queue.get_nowait()
        except queue.Empty:
            return False
        if not isinstance(request, dict):
            return False

        images = list(request.get("images") or [])
        encodings = list(request.get("encodings") or [])
        try:
            embeddings = self._run_siglip2_cls_embeddings_impl(images, encodings)
            self._set_siglip2_request_result(request, ok=True, error="", embeddings=embeddings)
        except Exception as exc:
            self._set_siglip2_request_result(request, ok=False, error=f"image_embed_failed: {exc}", embeddings=[])
        return True

    def _siglip2_scheduler_loop(self) -> None:
        hi_since_lo = 0
        while not self._siglip2_scheduler_stop.is_set():
            hi_requests = self._collect_hi_siglip2_requests(block=True)
            if hi_requests:
                self._process_hi_siglip2_requests(hi_requests)
                hi_since_lo += 1
                if hi_since_lo >= self._siglip2_fair_hi_batches and self._process_lo_siglip2_request(block=False):
                    hi_since_lo = 0
                continue

            if self._process_lo_siglip2_request(block=True):
                hi_since_lo = 0

    def _get_session(self) -> "requests.Session":
        if requests is None:
            raise RuntimeError("requests not available")
        if self._session is None:
            sess = requests.Session()
            sess.headers.update({"Content-Type": "application/json"})
            self._session = sess
        return self._session

    def _build_sglang_model(self) -> bool:
        if self._model is not None:
            return True

        t_start = time.perf_counter()

        try:
            if requests is None:
                if self.debug:
                    print("[CaptionWorker] requests not available; skipping caption model")
                self._model_load_failed = True
                self._record_time("build_model", time.perf_counter() - t_start)
                return False

            self._model = self._sglang_model
            self._processor = None

            base_prompt = self._build_task_block()
            high_quality_prompt = self._build_high_quality_task_block()
            examples = self._build_examples_block()
            if examples:
                base_prompt = base_prompt + "\n\n" + examples
                high_quality_prompt = high_quality_prompt + "\n\n" + examples
            self._base_messages = [{"role": "system", "content": base_prompt}]
            self._hq_base_messages = [{"role": "system", "content": high_quality_prompt}]

            payload = {
                "model": self._model,
                "messages": [*self._base_messages, {"role": "user", "content": "OK"}],
                "temperature": 0.0,
                "max_tokens": 2,
                "chat_template_kwargs": {"enable_thinking": False},
            }

            sess = self._get_session()
            response = sess.post(self._sglang_url, json=payload, timeout=10)
            response.raise_for_status()
            if self.debug:
                with contextlib.suppress(Exception):
                    result = response.json()
                    choices = result.get("choices", []) if isinstance(result, dict) else []
                    out = (choices[0].get("message", {}) or {}).get("content", "") if choices else ""
                    print(f"[CaptionWorker] SGLang warmup response: {str(out).strip()}")

            self._record_time("build_model", time.perf_counter() - t_start)

            return True

        except Exception as exc:
            if not self._model_load_failed:
                print(f"[CaptionWorker] Failed to initialize SGLang caption backend: {exc}")
                self._model_load_failed = True
            self._model = None
            self._processor = None
            self._base_messages = []
            self._hq_base_messages = []
            self._record_time("build_model", time.perf_counter() - t_start)
            return False

    def _build_ollama_model(self) -> bool:
        """
        'Build' the Ollama-backed caption model.

        There is no local model to load in this process; we just make sure the
        Ollama client is available. Ollama itself lazily loads and caches qwen3-vl.
        """
        if self._model is not None:
            return True

        t_start = time.perf_counter()
        try:
            if ollama_generate is None:
                if self.debug:
                    print(
                        "[CaptionWorker] ollama Python client not available. "
                        "Install with `pip install ollama` and ensure the "
                        "server is running and `qwen3-vl:8b-instruct` is pulled."
                    )
                self._model_load_failed = True
                self._record_time("build_model", time.perf_counter() - t_start)
                return False

            # Mark the backend as initialized. First generate() call will pull/load the model.
            self._model = self._ollama_model
            self._processor = None
            self._record_time("build_model", time.perf_counter() - t_start)

            # Make warmup deterministic so the saved "context" is stable.
            base_prompt = self._build_task_block() + "\n\n" + self._build_examples_block() + "\n\nReply with OK."

            resp = ollama_generate(
                model=self._ollama_model,
                prompt=base_prompt,
                stream=False,
                think=False,
                keep_alive=-1,
                options={
                    "temperature": 0.0,
                    "num_predict": 2,
                    "stop": ["\n"],
                },
            )
            if isinstance(resp, dict):
                self._base_context = resp.get("context")
            else:
                self._base_context = getattr(resp, "context", None)

            return True

        except Exception as exc:
            if not self._model_load_failed:
                print(f"[CaptionWorker] Failed to initialize Ollama caption backend: {exc}")
                self._model_load_failed = True
            self._model = None
            self._processor = None
            self._hq_base_messages = []
            self._record_time("build_model", time.perf_counter() - t_start)
            return False

    def _build_model(self) -> bool:
        """Backend dispatcher used by _run_caption_model()."""
        if self._caption_server == "sglang":
            return self._build_sglang_model()
        if self._caption_server == "ollama":
            return self._build_ollama_model()
        if self._caption_server == "vllm":
            return self._build_vllm_model()
        if self.debug:
            print(f"[CaptionWorker] Invalid caption server: {self._caption_server}")
        return False

    def _build_vllm_model(self) -> bool:
        """
        'Build' the vLLM-backed caption model.

        There is no local model to load in this process; we just make sure the
        vLLM server is reachable and cache the base prompt messages.
        """
        if self._model is not None:
            return True

        t_start = time.perf_counter()
        try:
            if requests is None:
                if self.debug:
                    print(
                        "[CaptionWorker] requests not available. "
                        "Install with `pip install requests` and ensure the "
                        "vLLM server is running."
                    )
                self._model_load_failed = True
                self._record_time("build_model", time.perf_counter() - t_start)
                return False

            base_prompt = self._build_task_block()
            high_quality_prompt = self._build_high_quality_task_block()
            examples = self._build_examples_block()
            if examples:
                base_prompt = base_prompt + "\n\n" + examples
                high_quality_prompt = high_quality_prompt + "\n\n" + examples
            self._base_messages = [{"role": "system", "content": base_prompt}]
            self._hq_base_messages = [{"role": "system", "content": high_quality_prompt}]

            if not self._probe_vllm_server():
                self._model_load_failed = True
                self._record_time("build_model", time.perf_counter() - t_start)
                return False

            self._model = self._vllm_model
            self._processor = None
            self._record_time("build_model", time.perf_counter() - t_start)
            return True

        except Exception as exc:
            if not self._model_load_failed:
                print(f"[CaptionWorker] Failed to initialize vLLM caption backend: {exc}")
            self._model_load_failed = True
            self._model = None
            self._processor = None
            self._base_messages = []
            self._hq_base_messages = []
            self._record_time("build_model", time.perf_counter() - t_start)
            return False

    def _run_caption_model(
        self,
        images: List[Any],
        boxes: List[Optional[dict]],
        *,
        is_recaption_flags: Optional[List[bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run Qwen3-VL on a batch of cropped object images.

        Backend is selected by `caption_server` (vLLM by default), and calls
        are done in parallel while preserving the original order of captions.
        """
        if not images:
            return []

        if not self._build_model():
            # Backend not available -> return empty captions
            return [
                {
                    "caption": "",
                    "merge_object_ids": [],
                    "error_reason": "vlm_unavailable",
                }
                for _ in range(len(images))
            ]

        if Image is None:
            if self.debug:
                print("[CaptionWorker] PIL not available; cannot run visual captions")
            return [{"caption": "", "merge_object_ids": [], "error_reason": "no_pil"} for _ in range(len(images))]

        # --- 1. Build prompts and encode images as raw bytes ---
        t_template_start = time.perf_counter()
        prompts: List[str] = []
        image_bytes_list: List[List[bytes]] = []
        if is_recaption_flags is None:
            is_recaption_flags = [False] * len(images)
        elif len(is_recaption_flags) < len(images):
            is_recaption_flags = list(is_recaption_flags) + [False] * (len(images) - len(is_recaption_flags))
        else:
            is_recaption_flags = list(is_recaption_flags[: len(images)])

        for idx, image_entry in enumerate(images):
            box_info = boxes[idx] if idx < len(boxes) else None
            box_tag = self._format_qwen_box(box_info)

            if box_tag:
                box_line = f"TARGET BOUNDING BOX: {box_tag}"
            else:
                box_line = "TARGET BOUNDING BOX: [none; describe the main visible object]."

            image_list = list(image_entry) if isinstance(image_entry, (list, tuple)) else [image_entry]
            image_count = len(image_list)
            if image_count > 1:
                view_line = f"INPUT VIEWS: {image_count} cropped views of the same object."
            elif self._caption_visual_prompt_mode == "mask_composite":
                view_line = (
                    "INPUT VIEWS: 1 composite image with a scene thumbnail on the left and "
                    "a highlighted target crop on the right."
                )
            elif self._caption_visual_prompt_mode == "mask_crop":
                view_line = "INPUT VIEWS: 1 cropped view with the target object highlighted."
            else:
                view_line = "INPUT VIEWS: 1 cropped view."

            prompt_lines = [
                "NEW INPUT:",
                view_line,
                box_line,
            ]
            spatial_line = self._format_spatial_context(box_info)
            if spatial_line:
                prompt_lines.append(spatial_line)

            prompt_text = "\n".join(prompt_lines) + "\n\nReturn the strict JSON object for the target."
            prompts.append(prompt_text)

            # Ensure each entry is a PIL image and convert to PNG bytes for the backend call.
            encoded_images: List[bytes] = []
            for img in image_list:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if not isinstance(img, Image.Image):
                    continue
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                encoded_images.append(buf.getvalue())
            image_bytes_list.append(encoded_images)

        template_dur = time.perf_counter() - t_template_start
        self._record_time("chat_template", template_dur)

        # --- 2. Call the caption backend for each image IN PARALLEL ---
        t_gen_start = time.perf_counter()

        # Result list in the same order as inputs
        captions: List[str] = ["" for _ in range(len(prompts))]
        errors: List[Optional[str]] = [None for _ in range(len(prompts))]

        # How many parallel caption calls we allow
        max_workers = min(len(prompts), getattr(self, "caption_batch_size", 4))

        high_quality_task_block = self._build_high_quality_task_block()

        def _caption_one_ollama(
            idx: int,
            prompt_text: str,
            img_bytes_list_item: List[bytes],
            is_recaption: bool,
        ) -> tuple[int, str, Optional[str]]:
            """Worker function for a single image/prompt."""
            if not img_bytes_list_item:
                return idx, "", "no_image"

            # IMPORTANT: copy context so each thread gets its own list object.
            ctx = list(self._base_context) if (self._base_context and not is_recaption) else None
            final_prompt = prompt_text
            if is_recaption:
                final_prompt = high_quality_task_block + "\n\n" + prompt_text

            try:
                resp = ollama_generate(
                    model=self._ollama_model,
                    prompt=final_prompt,
                    images=img_bytes_list_item,
                    stream=False,
                    raw=False,  # do NOT use raw=True with VL here
                    think=False,  # disable thinking mode for speed
                    context=ctx,
                    keep_alive=-1,  # keep model loaded in memory
                    format=self._caption_ollama_format(),
                    options={
                        # Tunable for realtime latency:
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "num_predict": self._vllm_max_tokens,
                    },
                )

                if self.debug:
                    print(f"[CaptionWorker] ollama resp (idx={idx}): {resp}")

                # Python client typically returns an object with .response
                if isinstance(resp, dict):
                    text = resp.get("response", "")
                else:
                    text = getattr(resp, "response", "") or ""

                # Fallback: if thinking accidentally appears, grab that
                if not text:
                    thinking = getattr(resp, "thinking", "") or ""
                    if thinking:
                        # Take the first sentence as a crude caption
                        text = thinking.split(".")[0].rstrip(",; ") + "."

                if not text:
                    return idx, "", "vlm_empty_response"
                return idx, text.strip(), None
            except Exception as exc:
                if self.debug:
                    print(f"[CaptionWorker] Ollama generate() failed for idx={idx}: {exc}")
                return idx, "", "vlm_failure"

        def _caption_one_sglang(
            idx: int,
            prompt_text: str,
            img_bytes_list_item: List[bytes],
            is_recaption: bool,
        ) -> tuple[int, str, Optional[str]]:
            """Worker function for a single image/prompt."""
            if not img_bytes_list_item:
                return idx, "", "no_image"
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            for img_bytes in img_bytes_list_item:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            user_message = {"role": "user", "content": content}
            messages = self._hq_base_messages if is_recaption and self._hq_base_messages else self._base_messages

            payload = {
                "model": self._model,
                "messages": [*messages, user_message],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": self._vllm_max_tokens,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            }

            if self.debug:
                print(f"[CaptionWorker] SGLang POST {self._sglang_url}")
            sess = self._get_session()
            response = sess.post(self._sglang_url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            choices = result.get("choices", [])
            if choices:
                out = choices[0].get("message", {}).get("content", "").strip()
            else:
                out = ""

            if self.debug:
                print(f"[CaptionWorker] SGLang output (idx={idx}): {out}")

            return idx, out, None

        def _caption_one_vllm(
            idx: int,
            prompt_text: str,
            img_bytes_list_item: List[bytes],
            is_recaption: bool,
        ) -> tuple[int, str, Optional[str]]:
            """Worker function for a single image/prompt."""
            if not img_bytes_list_item:
                return idx, "", "no_image"

            try:
                content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
                for img_bytes in img_bytes_list_item:
                    image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
                user_message = {"role": "user", "content": content}
                messages = self._hq_base_messages if is_recaption and self._hq_base_messages else self._base_messages

                payload = {
                    "model": self._vllm_model,
                    "messages": [*messages, user_message],
                    "temperature": self._vllm_temperature,
                    "top_p": self._vllm_top_p,
                    "max_tokens": self._vllm_max_tokens,
                }
                response_format = self._caption_response_format()
                if response_format is not None:
                    payload["response_format"] = response_format
                if self._vllm_disable_thinking:
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                headers = {"Content-Type": "application/json"}
                if self._vllm_api_key:
                    headers["Authorization"] = f"Bearer {self._vllm_api_key}"
                url = self._vllm_chat_url()
                sess = self._get_session()
                response = sess.post(url, json=payload, headers=headers, timeout=self._vllm_timeout_s)
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception as json_exc:
                    text_preview = (response.text or "")[:200].strip()
                    if "<" in text_preview and ">" in text_preview:
                        print(
                            f"[CaptionWorker] WARNING: vLLM returned HTML instead of JSON (idx={idx}). "
                            f"Wrong URL or server not running? url={url!r} response_preview={text_preview!r}"
                        )
                    else:
                        print(
                            f"[CaptionWorker] WARNING: vLLM returned invalid JSON (idx={idx}). "
                            f"url={url!r} error={json_exc!r}"
                        )
                    return idx, "", "vlm_failure"

                if self.debug:
                    print(f"[CaptionWorker] vllm resp (idx={idx}): {data}")

                choices = data.get("choices", [])
                if not choices:
                    print(
                        f"[CaptionWorker] WARNING: vLLM returned no choices (idx={idx}); no caption. "
                        "Check that the model is responding correctly."
                    )
                    return idx, "", "vlm_empty_response"
                message = choices[0].get("message", {}) or {}
                text = message.get("content", "") or ""
                if not text:
                    print(f"[CaptionWorker] WARNING: vLLM returned empty content (idx={idx}); no caption.")
                    return idx, "", "vlm_empty_response"
                return idx, text.strip(), None
            except Exception as exc:
                print(f"[CaptionWorker] WARNING: vLLM caption request failed (idx={idx}); no response. error={exc!r}")
                if self.debug:
                    print(f"[CaptionWorker] vLLM request failed for idx={idx}: {exc}")
                return idx, "", "vlm_failure"

        if self._caption_server == "ollama":
            _caption_one = _caption_one_ollama
        elif self._caption_server == "sglang":
            _caption_one = _caption_one_sglang
        elif self._caption_server == "vllm":
            _caption_one = _caption_one_vllm
        else:
            raise ValueError(f"Invalid caption server: {self._caption_server}")

        if self._caption_server == "vllm" and self._vllm_prefix_warmup_enabled:
            t_warm = time.perf_counter()
            for recap in sorted({bool(x) for x in is_recaption_flags}):
                self._warm_vllm_prefix_cache(is_recaption=recap)
            self._record_time("vllm_prefix_warmup", time.perf_counter() - t_warm)

        if max_workers <= 1:
            # Sequential fallback (e.g. if there's only 1 item)
            for i, (p, b, recap) in enumerate(zip(prompts, image_bytes_list, is_recaption_flags)):
                _, text, err = _caption_one(i, p, b, bool(recap))
                captions[i] = text
                errors[i] = err
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(_caption_one, i, p, b, bool(recap))
                    for i, (p, b, recap) in enumerate(zip(prompts, image_bytes_list, is_recaption_flags))
                ]
                for fut in as_completed(futures):
                    idx, text, err = fut.result()
                    captions[idx] = text
                    errors[idx] = err

        generate_dur = time.perf_counter() - t_gen_start
        self._record_time("generate", generate_dur)

        if self.debug:
            print(f"#images: {len(images)}")
            print(
                "[CaptionWorker] timing chat_template={:.2f}ms generate={:.2f}ms (workers={})".format(
                    template_dur * 1000.0, generate_dur * 1000.0, max_workers
                )
            )

        payloads: List[Dict[str, Any]] = []
        for text, err in zip(captions, errors):
            payload = self._parse_caption_response(text)
            if err:
                payload["error_reason"] = err
            payloads.append(payload)
        return payloads

    def _run_caption_embeddings(self, captions: List[str]) -> List[List[float]]:
        """
        Run Qwen3-embedding via vLLM on caption texts.
        Returns embeddings aligned with the caption list.
        """
        if not captions:
            return []

        def _canonicalize_for_embedding(text: str) -> str:
            # Light cleanup only: lowercase (plus trim whitespace).
            return (text or "").strip().lower()

        def _wrap_qwen3_document(text: str) -> str:
            canon = _canonicalize_for_embedding(text)
            if not canon:
                return ""
            # Use Qwen3's recommended retrieval prompt format.
            return f"Instruct: {self._qwen3_embed_task}\nDocument: {canon}"

        if requests is None:
            if self.debug and not self._embed_warned:
                print("[CaptionWorker] requests not available; skipping caption embeddings")
                self._embed_warned = True
            return [[] for _ in range(len(captions))]

        if self._embed_server_ok is None:
            self._embed_server_ok = self._probe_embed_server()
        if not self._embed_server_ok:
            if self.debug and not self._embed_warned:
                print(
                    "[CaptionWorker] vLLM embedding backend unavailable or model not served; "
                    f"model={self._embed_model!r} base_url={self._embed_base_url!r}"
                )
                self._embed_warned = True
            return [[] for _ in range(len(captions))]

        embeddings: List[List[float]] = [[] for _ in range(len(captions))]
        url = self._vllm_embed_url()
        headers: Dict[str, str] = {}
        if self._embed_api_key:
            headers["Authorization"] = f"Bearer {self._embed_api_key}"

        wrapped_texts: List[str] = []
        wrapped_to_out_idx: List[int] = []
        for out_idx, caption in enumerate(captions):
            wrapped = _wrap_qwen3_document(caption)
            if not wrapped:
                continue
            wrapped_to_out_idx.append(out_idx)
            wrapped_texts.append(wrapped)

        if not wrapped_texts:
            return embeddings

        def _parse_embeddings_payload(payload: Any, expected: int) -> List[List[float]]:
            items = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ValueError("Unexpected embeddings response shape")

            by_index: Dict[int, Any] = {}
            for fallback_idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                idx_raw = item.get("index", fallback_idx)
                try:
                    idx = int(idx_raw)
                except Exception:
                    idx = int(fallback_idx)
                by_index[idx] = item.get("embedding")

            out: List[List[float]] = []
            for i in range(expected):
                vec = np.asarray(by_index.get(i) or [], dtype=np.float32).reshape(-1)
                if vec.size > 0:
                    denom = float(np.linalg.norm(vec) + 1e-12)
                    vec = vec / denom
                    self._embed_dim = int(vec.size)
                    out.append(vec.tolist())
                else:
                    out.append([])
            return out

        # vLLM supports OpenAI-style batching: send input=[...] in a single /embeddings call.
        sess = self._get_session()
        payload = {"model": self._embed_model, "input": wrapped_texts}
        batch_vecs: Optional[List[List[float]]] = None
        last_exc: Optional[Exception] = None
        for attempt in range(self._embed_max_retries):
            try:
                response = sess.post(url, json=payload, headers=headers, timeout=self._embed_timeout_s)
                response.raise_for_status()
                batch_vecs = _parse_embeddings_payload(response.json(), expected=len(wrapped_texts))
                break
            except Exception as exc:
                last_exc = exc
                if attempt < self._embed_max_retries - 1:
                    time.sleep(0.1)

        if batch_vecs is None:
            if self.debug:
                print(f"[CaptionWorker] Batch embedding failed after retries: {last_exc}")
            if self._embed_dim:
                zero = np.zeros(self._embed_dim, dtype=np.float32).tolist()
                for out_idx in wrapped_to_out_idx:
                    embeddings[out_idx] = zero
            return embeddings

        for out_idx, vec in zip(wrapped_to_out_idx, batch_vecs):
            embeddings[out_idx] = vec

        return embeddings

    def request_caption_text_embeddings(
        self,
        texts: Sequence[str],
        *,
        normalize: bool = True,
        timeout_s: Optional[float] = None,
        client_id: int = 0,
    ) -> tuple[bool, str, List[List[float]]]:
        """
        Synchronous caption-text embedding RPC used by the Viser editor.
        Returns: (ok, error, embeddings)
        """
        cleaned = [str(t or "").strip() for t in (texts or [])]
        if not cleaned or any(not t for t in cleaned):
            return False, "bad_request_empty_texts", []

        # Probe once; _run_caption_embeddings() currently returns [] vectors when backend is down.
        if self._embed_server_ok is None:
            self._embed_server_ok = self._probe_embed_server()
        if not self._embed_server_ok:
            return (
                False,
                f"embed_backend_unavailable:model={self._embed_model!r} base_url={self._embed_base_url!r}",
                [],
            )

        try:
            vecs = self._run_caption_embeddings(cleaned)
        except Exception as exc:
            return False, f"embed_exception:{exc}", []

        # Enforce shape; if backend returned blanks, treat as failure for interactive edits.
        if not isinstance(vecs, list) or len(vecs) != len(cleaned) or any((not v) for v in vecs):
            return False, "embed_failed_empty_vectors", []

        # _run_caption_embeddings() already normalizes; honor normalize=False if needed.
        if not normalize:
            return True, "", vecs
        return True, "", vecs

    def _build_siglip2_model(self) -> bool:
        if not self._siglip2_enabled:
            self._siglip2_dbg("disabled via SIGLIP2_ENABLED")
            return False
        if self._siglip2_load_failed:
            self._siglip2_dbg("previous init failure; not retrying in this process")
            return False
        if self._siglip2_model is not None:
            return True
        if torch is None or AutoModel is None or AutoProcessor is None:
            self._siglip2_dbg(
                "missing deps:"
                f" torch={torch is not None}"
                f" AutoModel={AutoModel is not None}"
                f" AutoProcessor={AutoProcessor is not None}"
            )
            self._siglip2_load_failed = True
            return False

        t0 = time.perf_counter()
        try:
            torch_device = torch.device(self.device)
            if torch_device.type == "cuda" and not torch.cuda.is_available():
                torch_device = torch.device("cpu")
            is_cuda = torch_device.type == "cuda"
            dtype = torch.float16 if is_cuda else torch.float32

            attn_impl = "sdpa"
            ckpt = self._siglip2_ckpt
            ckpt_is_dir = os.path.isdir(ckpt)
            self._siglip2_dbg(
                "loading"
                f" ckpt={ckpt!r}"
                f" is_dir={ckpt_is_dir}"
                f" device={str(torch_device)!r}"
                f" dtype={str(dtype)!r}"
                f" attn={attn_impl!r}"
            )
            try:
                model = AutoModel.from_pretrained(
                    self._siglip2_ckpt,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
            except TypeError:
                try:
                    model = AutoModel.from_pretrained(
                        self._siglip2_ckpt,
                        torch_dtype=dtype,
                        attn_implementation=attn_impl,
                    )
                except TypeError:
                    try:
                        model = AutoModel.from_pretrained(self._siglip2_ckpt, dtype=dtype)
                    except TypeError:
                        model = AutoModel.from_pretrained(self._siglip2_ckpt, torch_dtype=dtype)

            model = model.to(torch_device).eval()

            processor = AutoProcessor.from_pretrained(self._siglip2_ckpt, use_fast=True)
            img_proc = getattr(processor, "image_processor", None)
            if img_proc is None:
                raise RuntimeError("SigLIP2 processor missing image_processor")

            self._siglip2_model = model
            self._siglip2_processor = processor
            size = getattr(img_proc, "size", {}) or {}
            self._siglip2_dbg(f"loaded ok; processor_size={size}")
            self._record_time("build_siglip2", time.perf_counter() - t0)
            return True
        except Exception as exc:
            import traceback as _tb
            print(
                f"[CaptionWorker] SigLIP2 _build_siglip2_model failed: {exc!r}\n"
                f"  ckpt={self._siglip2_ckpt!r} is_dir={os.path.isdir(self._siglip2_ckpt)}\n"
                + _tb.format_exc()
            )
            self._siglip2_dbg(f"failed to initialize: {exc!r}")
            self._siglip2_load_failed = True
            self._siglip2_model = None
            self._siglip2_processor = None
            self._siglip2_mean = None
            self._siglip2_std = None
            self._siglip2_target_hw = None
            self._record_time("build_siglip2", time.perf_counter() - t0)
            return False

    def _run_siglip2_cls_embeddings(self, crop_images_uint8: List[Any], crop_encodings: List[str]) -> List[List[float]]:
        """
        Returns per-crop SigLIP2 pooled image embeddings, aligned with crop_images_uint8.
        """
        if not crop_images_uint8:
            return []
        return self._enqueue_siglip2_image_embeddings(crop_images_uint8, crop_encodings)

    def _build_qwen3_vl_embed_model(self) -> bool:
        if not self._qwen3_vl_embed_enabled:
            self._qwen3_vl_dbg("disabled via QWEN3_VL_EMBED_ENABLED")
            return False
        if self._qwen3_vl_embed_load_failed:
            self._qwen3_vl_dbg("previous init failure; not retrying in this process")
            return False
        if self._qwen3_vl_embed_hf_model is not None:
            return True
        if torch is None or AutoModel is None or AutoProcessor is None:
            self._qwen3_vl_dbg(
                "missing deps:"
                f" torch={torch is not None}"
                f" AutoModel={AutoModel is not None}"
                f" AutoProcessor={AutoProcessor is not None}"
            )
            self._qwen3_vl_embed_load_failed = True
            return False

        try:
            torch_device = torch.device(self.device)
            if torch_device.type == "cuda" and not torch.cuda.is_available():
                torch_device = torch.device("cpu")
            is_cuda = torch_device.type == "cuda"
            dtype = torch.float16 if is_cuda else torch.float32

            self._qwen3_vl_dbg(
                f"loading ckpt={self._qwen3_vl_embed_ckpt!r} device={str(torch_device)!r} dtype={str(dtype)!r}"
            )
            try:
                model = AutoModel.from_pretrained(
                    self._qwen3_vl_embed_ckpt,
                    dtype=dtype,
                    trust_remote_code=True,
                )
            except TypeError:
                try:
                    model = AutoModel.from_pretrained(
                        self._qwen3_vl_embed_ckpt,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    )
                except TypeError:
                    model = AutoModel.from_pretrained(self._qwen3_vl_embed_ckpt, trust_remote_code=True)

            model = model.to(torch_device).eval()
            processor = AutoProcessor.from_pretrained(self._qwen3_vl_embed_ckpt, trust_remote_code=True)

            self._qwen3_vl_embed_hf_model = model
            self._qwen3_vl_embed_hf_processor = processor
            return True
        except Exception as exc:
            import traceback as _tb
            print(
                f"[CaptionWorker] Qwen3-VL embed _build_qwen3_vl_embed_model failed: {exc!r}\n"
                f"  ckpt={self._qwen3_vl_embed_ckpt!r}\n"
                + _tb.format_exc()
            )
            self._qwen3_vl_dbg(f"failed to initialize: {exc!r}")
            self._qwen3_vl_embed_load_failed = True
            self._qwen3_vl_embed_hf_model = None
            self._qwen3_vl_embed_hf_processor = None
            return False

    def _run_qwen3_vl_cls_embeddings(
        self, crop_images_uint8: List[Any], crop_encodings: List[str]
    ) -> List[List[float]]:
        if not crop_images_uint8:
            return []
        backend = self._qwen3_vl_embed_backend
        if backend == "vllm":
            return self._run_qwen3_vl_cls_embeddings_vllm(crop_images_uint8, crop_encodings)
        if backend == "hf":
            return self._run_qwen3_vl_cls_embeddings_hf_impl(crop_images_uint8, crop_encodings)

        # auto: prefer vLLM, fallback to HF for robustness.
        vectors = self._run_qwen3_vl_cls_embeddings_vllm(crop_images_uint8, crop_encodings)
        if any(vectors):
            return vectors
        return self._run_qwen3_vl_cls_embeddings_hf_impl(crop_images_uint8, crop_encodings)

    def _run_qwen3_vl_cls_embeddings_vllm(
        self, crop_images_uint8: List[Any], crop_encodings: List[str]
    ) -> List[List[float]]:
        if requests is None:
            return [[] for _ in range(len(crop_images_uint8))]

        if self._qwen3_vl_embed_server_ok is None:
            self._qwen3_vl_embed_server_ok = self._probe_qwen3_vl_embed_server()
        if not self._qwen3_vl_embed_server_ok:
            return [[] for _ in range(len(crop_images_uint8))]

        def _to_hwc_uint8_numpy(img: Any, enc: str) -> Optional[np.ndarray]:
            x = img
            if torch is not None and torch.is_tensor(x):
                with contextlib.suppress(Exception):
                    x = x.detach().to("cpu", copy=False).numpy()
            with contextlib.suppress(Exception):
                x = np.asarray(x)
            if not isinstance(x, np.ndarray) or x.ndim != 3:
                return None
            if x.shape[-1] in {3, 4}:
                hwc = x
            elif x.shape[0] in {3, 4}:
                hwc = np.transpose(x, (1, 2, 0))
            else:
                return None
            if hwc.dtype != np.uint8:
                with np.errstate(all="ignore"):
                    vmax = float(np.nanmax(hwc)) if hwc.size else 0.0
                if vmax <= 1.0:
                    hwc = hwc * 255.0
                hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
            else:
                hwc = hwc.copy()
            enc_norm = str(enc or "").strip().lower()
            if hwc.shape[2] == 4:
                hwc = hwc[:, :, :3]
            if enc_norm in {"bgr8", "bgr", "bgra8", "bgra"}:
                hwc = hwc[:, :, [2, 1, 0]]
            return np.ascontiguousarray(hwc)

        def _to_data_uri_jpeg(arr: np.ndarray) -> str:
            if Image is None:
                raise RuntimeError("PIL not available")
            img = Image.fromarray(arr, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._qwen3_vl_embed_jpeg_quality, subsampling=2)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"

        def _ceil_div(a: int, b: int) -> int:
            return (int(a) + int(b) - 1) // int(b)

        def _estimate_vision_tokens(height: int, width: int) -> int:
            stride = max(1, int(self._qwen3_vl_embed_token_stride))
            h = max(1, int(height))
            w = max(1, int(width))
            return _ceil_div(h, stride) * _ceil_div(w, stride)

        def _downscale_to_caps(rgb: np.ndarray) -> np.ndarray:
            if Image is None:
                return rgb
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                return rgb
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            if h <= 0 or w <= 0:
                return rgb

            max_hw = max(1, int(self._qwen3_vl_embed_max_hw))
            max_tokens = max(0, int(self._qwen3_vl_embed_max_image_tokens))
            stride = max(1, int(self._qwen3_vl_embed_token_stride))

            scale_hw = min(1.0, float(max_hw) / float(max(h, w)))
            new_h = max(1, int(round(h * scale_hw)))
            new_w = max(1, int(round(w * scale_hw)))

            if max_tokens > 0:
                cur_tokens = _estimate_vision_tokens(new_h, new_w)
                if cur_tokens > max_tokens:
                    scale_tok = (float(max_tokens) / float(max(cur_tokens, 1))) ** 0.5
                    new_h = max(1, int(round(new_h * scale_tok)))
                    new_w = max(1, int(round(new_w * scale_tok)))

            if new_h >= stride:
                new_h = max(stride, (new_h // stride) * stride)
            if new_w >= stride:
                new_w = max(stride, (new_w // stride) * stride)

            if new_h > max_hw:
                new_h = max_hw
            if new_w > max_hw:
                new_w = max_hw

            if max_tokens > 0:
                for _ in range(16):
                    if _estimate_vision_tokens(new_h, new_w) <= max_tokens:
                        break
                    if new_w >= new_h and new_w > 1:
                        new_w = max(1, new_w - stride)
                    elif new_h > 1:
                        new_h = max(1, new_h - stride)
                    else:
                        break

            if new_h == h and new_w == w:
                return rgb

            pil = Image.fromarray(rgb, mode="RGB")
            resized = pil.resize((int(new_w), int(new_h)), resample=Image.BILINEAR)
            if self._qwen3_vl_embed_debug or self.debug:
                t0 = _estimate_vision_tokens(h, w)
                new_tokens = _estimate_vision_tokens(int(new_h), int(new_w))
                print(
                    "[CaptionWorker][Qwen3-VL-Embedding] downscale {}x{} -> {}x{} tok {} -> {} (max_hw={} max_tok={})"
                    .format(w, h, new_w, new_h, t0, new_tokens, max_hw, max_tokens)
                )
            return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))

        items: List[tuple[int, str]] = []
        for i, img in enumerate(crop_images_uint8):
            enc = crop_encodings[i] if i < len(crop_encodings) else ""
            rgb = _to_hwc_uint8_numpy(img, enc)
            if rgb is None:
                continue
            rgb = _downscale_to_caps(rgb)
            try:
                data_uri = _to_data_uri_jpeg(rgb)
            except Exception:
                continue
            items.append((i, data_uri))

        out_all: List[List[float]] = [[] for _ in range(len(crop_images_uint8))]
        if not items:
            return out_all

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._qwen3_vl_embed_api_key:
            headers["Authorization"] = f"Bearer {self._qwen3_vl_embed_api_key}"
        url = self._vllm_qwen3_vl_embed_url()
        sess = self._get_session()

        def _embed_one_vllm(src_i: int, data_uri: str) -> tuple[int, List[float]]:
            payload = {
                "model": self._qwen3_vl_embed_model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }],
                "encoding_format": "float",
            }

            last_exc: Optional[Exception] = None
            for attempt in range(self._qwen3_vl_embed_max_retries):
                try:
                    resp = sess.post(url, json=payload, headers=headers, timeout=self._qwen3_vl_embed_timeout_s)
                    if not resp.ok:
                        body = ""
                        with contextlib.suppress(Exception):
                            body = (resp.text or "")[:300]
                        raise RuntimeError(f"status={resp.status_code} body={body!r}")
                    response_payload = resp.json()
                    rows = response_payload.get("data") if isinstance(response_payload, dict) else None
                    if not isinstance(rows, list) or not rows:
                        raise ValueError("invalid embeddings payload shape")
                    row0 = rows[0] if isinstance(rows[0], dict) else {}
                    vec = row0.get("embedding") if isinstance(row0, dict) else None
                    if isinstance(vec, list):
                        return src_i, vec
                    raise ValueError("missing embedding vector in response")
                except Exception as exc:
                    last_exc = exc
                    if attempt < self._qwen3_vl_embed_max_retries - 1:
                        time.sleep(0.1)
            raise RuntimeError(f"qwen3-vl per-image embedding failed src_i={src_i}: {last_exc!r}")

        max_workers = min(len(items), self._qwen3_vl_embed_workers)
        if max_workers <= 1:
            for src_i, data_uri in items:
                try:
                    _, vec_raw = _embed_one_vllm(src_i, data_uri)
                except Exception as exc:
                    if self._qwen3_vl_embed_debug or self.debug:
                        print(f"[Qwen3-VL-Embedding] request failed: {exc}")
                    continue
                vec = np.asarray(vec_raw, dtype=np.float32).reshape(-1)
                if vec.size <= 0:
                    continue
                denom = float(np.linalg.norm(vec) + 1e-12)
                vec = vec / denom
                if self._qwen3_vl_embed_dim is None:
                    self._qwen3_vl_embed_dim = int(vec.size)
                out_all[int(src_i)] = vec.tolist()
            return out_all

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_embed_one_vllm, src_i, data_uri) for src_i, data_uri in items]
            for fut in as_completed(futures):
                try:
                    src_i, vec_raw = fut.result()
                except Exception as exc:
                    if self._qwen3_vl_embed_debug or self.debug:
                        print(f"[Qwen3-VL-Embedding] request failed: {exc}")
                    continue
                vec = np.asarray(vec_raw, dtype=np.float32).reshape(-1)
                if vec.size <= 0:
                    continue
                denom = float(np.linalg.norm(vec) + 1e-12)
                vec = vec / denom
                if self._qwen3_vl_embed_dim is None:
                    self._qwen3_vl_embed_dim = int(vec.size)
                out_all[int(src_i)] = vec.tolist()
        return out_all

    def _run_qwen3_vl_cls_embeddings_hf_impl(
        self, crop_images_uint8: List[Any], crop_encodings: List[str]
    ) -> List[List[float]]:
        if torch is None:
            if not self._qwen3_vl_embed_failure_logged:
                print("[CaptionWorker] Qwen3-VL embedding unavailable: torch not importable in this process")
                self._qwen3_vl_embed_failure_logged = True
            return [[] for _ in range(len(crop_images_uint8))]
        if not crop_images_uint8:
            return []
        if not self._build_qwen3_vl_embed_model():
            if not self._qwen3_vl_embed_failure_logged:
                print(
                    "[CaptionWorker] Qwen3-VL embedding unavailable: model init failed or deps missing "
                    f"(ckpt={self._qwen3_vl_embed_ckpt!r} "
                    f"torch={torch is not None} "
                    f"AutoModel={AutoModel is not None} "
                    f"AutoProcessor={AutoProcessor is not None})"
                )
                self._qwen3_vl_embed_failure_logged = True
            return [[] for _ in range(len(crop_images_uint8))]

        model = self._qwen3_vl_embed_hf_model
        processor = self._qwen3_vl_embed_hf_processor
        if model is None or processor is None:
            return [[] for _ in range(len(crop_images_uint8))]

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def _to_hwc_uint8_numpy(img: Any, enc: str) -> Optional[np.ndarray]:
            x = img
            if torch.is_tensor(x):
                with contextlib.suppress(Exception):
                    x = x.detach().to("cpu", copy=False).numpy()
            with contextlib.suppress(Exception):
                x = np.asarray(x)
            if not isinstance(x, np.ndarray):
                return None
            if x.ndim != 3:
                return None

            if x.shape[-1] in {3, 4}:
                hwc = x
            elif x.shape[0] in {3, 4}:
                hwc = np.transpose(x, (1, 2, 0))
            else:
                return None

            if hwc.dtype != np.uint8:
                with np.errstate(all="ignore"):
                    vmax = float(np.nanmax(hwc)) if hwc.size else 0.0
                if vmax <= 1.0:
                    hwc = hwc * 255.0
                hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
            else:
                hwc = hwc.copy()

            enc_norm = str(enc or "").strip().lower()
            if hwc.shape[2] == 4:
                hwc = hwc[:, :, :3]
            if enc_norm in {"bgr8", "bgr", "bgra8", "bgra"}:
                return hwc[:, :, [2, 1, 0]]
            return hwc

        with torch.inference_mode():
            images_rgb: List[np.ndarray] = []
            valid_indices: List[int] = []
            for i, img in enumerate(crop_images_uint8):
                enc = crop_encodings[i] if i < len(crop_encodings) else ""
                rgb = _to_hwc_uint8_numpy(img, enc)
                if rgb is None:
                    continue
                images_rgb.append(rgb)
                valid_indices.append(i)

            if not images_rgb:
                return [[] for _ in range(len(crop_images_uint8))]

            try:
                image_inputs = processor(images=images_rgb, return_tensors="pt")
            except Exception:
                # Some processors expect a text field even for image-only features.
                image_inputs = processor(
                    text=[""] * len(images_rgb),
                    images=images_rgb,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

            inputs: Dict[str, Any] = {}
            for key, value in dict(image_inputs).items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device=device, non_blocking=True)

            image_features = None
            if hasattr(model, "get_image_features"):
                try:
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            image_features = model.get_image_features(**inputs)
                    else:
                        image_features = model.get_image_features(**inputs)
                except TypeError:
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            image_features = model.get_image_features(inputs)
                    else:
                        image_features = model.get_image_features(inputs)
            elif hasattr(model, "encode_image"):
                try:
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            image_features = model.encode_image(**inputs)
                    else:
                        image_features = model.encode_image(**inputs)
                except TypeError:
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            image_features = model.encode_image(inputs)
                    else:
                        image_features = model.encode_image(inputs)
            else:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = model(**inputs, return_dict=True)
                else:
                    out = model(**inputs, return_dict=True)
                if hasattr(out, "image_embeds") and out.image_embeds is not None:
                    image_features = out.image_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    image_features = out.pooler_output
                elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    image_features = out.last_hidden_state[:, 0, :]

            if isinstance(image_features, (list, tuple)) and image_features:
                image_features = image_features[0]
            if not torch.is_tensor(image_features):
                raise RuntimeError("Qwen3-VL embedding image features unavailable")
            if image_features.ndim == 3:
                image_features = image_features[:, 0, :]

            image_features = image_features.to(dtype=torch.float32)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1, eps=1e-12)
            if self._qwen3_vl_embed_dim is None:
                with contextlib.suppress(Exception):
                    self._qwen3_vl_embed_dim = int(image_features.shape[-1])

            feats_cpu = image_features.detach().to("cpu", copy=False).numpy()
            vectors = [row.tolist() for row in feats_cpu]

        out_all: List[List[float]] = [[] for _ in range(len(crop_images_uint8))]
        for src_i, vec in zip(valid_indices, vectors):
            out_all[int(src_i)] = vec
        return out_all

    def _run_siglip2_text_embeddings_impl(self, texts: Sequence[str], *, normalize: bool = True) -> List[List[float]]:
        if torch is None:
            if not self._siglip2_failure_logged:
                print("[CaptionWorker] SigLIP2 unavailable: torch not importable in this process")
                self._siglip2_failure_logged = True
            return [[] for _ in range(len(texts))]
        text_list = [str(x or "").strip() for x in texts]
        if not text_list:
            return []
        if not self._build_siglip2_model():
            if not self._siglip2_failure_logged:
                ckpt = self._siglip2_ckpt
                print(
                    "[CaptionWorker] SigLIP2 unavailable: model init failed or deps missing "
                    f"(ckpt={ckpt!r} is_dir={os.path.isdir(ckpt)} "
                    f"torch={torch is not None} "
                    f"AutoModel={AutoModel is not None} "
                    f"AutoProcessor={AutoProcessor is not None})"
                )
                self._siglip2_failure_logged = True
            return [[] for _ in range(len(text_list))]

        model = self._siglip2_model
        processor = self._siglip2_processor
        if model is None or processor is None:
            return [[] for _ in range(len(text_list))]

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            text_inputs = processor(
                text=text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {}
            for key, value in dict(text_inputs).items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device=device, non_blocking=True)

            if hasattr(model, "get_text_features"):
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        text_features = model.get_text_features(**inputs)
                else:
                    text_features = model.get_text_features(**inputs)
            else:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = model(**inputs, return_dict=True)
                else:
                    out = model(**inputs, return_dict=True)
                if hasattr(out, "text_embeds") and out.text_embeds is not None:
                    text_features = out.text_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    text_features = out.pooler_output
                elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    text_features = out.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("SigLIP2 text features unavailable")

            # Some HF SigLIP2 versions return a BaseModelOutputWithPooling wrapper
            # from get_text_features() instead of a tensor; unwrap to a tensor so
            # downstream .to() / .normalize() work.
            if not torch.is_tensor(text_features):
                if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                    text_features = text_features.text_embeds
                elif hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                    text_features = text_features.pooler_output
                elif hasattr(text_features, "last_hidden_state") and text_features.last_hidden_state is not None:
                    text_features = text_features.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError(
                        f"SigLIP2 get_text_features() returned unexpected type: {type(text_features).__name__}"
                    )

            text_features = text_features.to(dtype=torch.float32)
            if normalize:
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1, eps=1e-12)
            if self._siglip2_dim is None:
                with contextlib.suppress(Exception):
                    self._siglip2_dim = int(text_features.shape[-1])
            vectors = text_features.detach().to("cpu", copy=False).numpy()
            return vectors.tolist()

    def _run_siglip2_cls_embeddings_impl(
        self, crop_images_uint8: List[Any], crop_encodings: List[str]
    ) -> List[List[float]]:
        if torch is None:
            if not self._siglip2_failure_logged:
                print("[CaptionWorker] SigLIP2 unavailable: torch not importable in this process")
                self._siglip2_failure_logged = True
            self._siglip2_dbg("torch not available")
            return []
        if not crop_images_uint8:
            self._siglip2_dbg("no crops")
            return []
        if not self._build_siglip2_model():
            if not self._siglip2_failure_logged:
                ckpt = self._siglip2_ckpt
                print(
                    "[CaptionWorker] SigLIP2 unavailable: model init failed or deps missing "
                    f"(ckpt={ckpt!r} is_dir={os.path.isdir(ckpt)} "
                    f"torch={torch is not None} "
                    f"AutoModel={AutoModel is not None} "
                    f"AutoProcessor={AutoProcessor is not None})"
                )
                self._siglip2_failure_logged = True
            self._siglip2_dbg("model build returned False")
            return []

        model = self._siglip2_model
        processor = self._siglip2_processor
        if model is None or processor is None:
            if not self._siglip2_failure_logged:
                print("[CaptionWorker] SigLIP2 unavailable: model components missing after init")
                self._siglip2_failure_logged = True
            self._siglip2_dbg("model components missing after build")
            return []

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def _to_hwc_uint8_numpy(img: Any, enc: str) -> Optional[np.ndarray]:
            x = img
            if torch.is_tensor(x):
                with contextlib.suppress(Exception):
                    x = x.detach().to("cpu", copy=False).numpy()
            with contextlib.suppress(Exception):
                x = np.asarray(x)
            if not isinstance(x, np.ndarray):
                return None
            if x.ndim != 3:
                return None

            if x.shape[-1] in {3, 4}:
                hwc = x
            elif x.shape[0] in {3, 4}:
                hwc = np.transpose(x, (1, 2, 0))
            else:
                return None

            if hwc.dtype != np.uint8:
                with np.errstate(all="ignore"):
                    vmax = float(np.nanmax(hwc)) if hwc.size else 0.0
                if vmax <= 1.0:
                    hwc = hwc * 255.0
                hwc = np.clip(hwc, 0.0, 255.0).astype(np.uint8)
            else:
                hwc = hwc.copy()

            enc_norm = str(enc or "").strip().lower()
            if hwc.shape[2] == 4:
                hwc = hwc[:, :, :3]
            if enc_norm in {"bgr8", "bgr", "bgra8", "bgra"}:
                return hwc[:, :, [2, 1, 0]]
            return hwc

        with torch.inference_mode():
            images_rgb: List[np.ndarray] = []
            valid_indices: List[int] = []
            skip_counts: Dict[str, int] = {}
            sample_skips: List[str] = []
            for i, img in enumerate(crop_images_uint8):
                if img is None:
                    skip_counts["none"] = skip_counts.get("none", 0) + 1
                    continue
                enc = crop_encodings[i] if i < len(crop_encodings) else ""
                x = _to_hwc_uint8_numpy(img, enc)
                if x is None:
                    skip_counts["bad_shape_or_type"] = skip_counts.get("bad_shape_or_type", 0) + 1
                    if (self._siglip2_debug or self.debug) and len(sample_skips) < 3:
                        with contextlib.suppress(Exception):
                            sample_skips.append(f"i={i} type={type(img)}")
                    continue
                images_rgb.append(x)
                valid_indices.append(i)

            self._siglip2_dbg(
                "preprocess"
                f" total={len(crop_images_uint8)}"
                f" valid={len(images_rgb)}"
                f" skips={skip_counts}"
                f" device={str(device)!r}"
                f" dtype={str(dtype)!r}"
            )
            for s in sample_skips:
                self._siglip2_dbg(f"sample_skip {s}")

            if not images_rgb:
                return [[] for _ in range(len(crop_images_uint8))]

            image_inputs = processor(images=images_rgb, return_tensors="pt")
            inputs: Dict[str, Any] = {}
            for key, value in dict(image_inputs).items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device=device, non_blocking=True)

            self._siglip2_dbg(f"forward input_keys={list(inputs.keys())}")

            if hasattr(model, "get_image_features"):
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        image_features = model.get_image_features(**inputs)
                else:
                    image_features = model.get_image_features(**inputs)
            else:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        out = model(**inputs, return_dict=True)
                else:
                    out = model(**inputs, return_dict=True)
                if hasattr(out, "image_embeds") and out.image_embeds is not None:
                    image_features = out.image_embeds
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    image_features = out.pooler_output
                elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    image_features = out.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("SigLIP2 image features unavailable")

            # Some HF SigLIP2 versions return a BaseModelOutputWithPooling wrapper
            # from get_image_features() instead of a tensor; unwrap to a tensor so
            # downstream .to() / .normalize() work.
            if not torch.is_tensor(image_features):
                if hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                    image_features = image_features.pooler_output
                elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
                    image_features = image_features.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError(
                        f"SigLIP2 get_image_features() returned unexpected type: {type(image_features).__name__}"
                    )

            image_features = image_features.to(dtype=torch.float32)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1, eps=1e-12)
            self._siglip2_dbg(f"forward image_feature_shape={tuple(image_features.shape)}")
            if self._siglip2_dim is None:
                with contextlib.suppress(Exception):
                    self._siglip2_dim = int(image_features.shape[-1])

            feats_cpu = image_features.detach().to("cpu", copy=False).numpy()
            vecs = [row.tolist() for row in feats_cpu]

        out_all: List[List[float]] = [[] for _ in range(len(crop_images_uint8))]
        for src_i, vec in zip(valid_indices, vecs):
            out_all[int(src_i)] = vec
        return out_all

    def _caption_merge_category_compatible(self, result: ObjectCaptionResult, neighbor_idx: int) -> bool:
        if not self._caption_merge_require_category_compat:
            return True

        categories_state: List[str] = self.scene_state.get("object_category", []) or []
        category_candidates_state: List[List[str]] = self.scene_state.get("object_category_candidates", []) or []
        supercategories_state: List[str] = self.scene_state.get("object_supercategory", []) or []

        target_categories = {
            _canonical_category(getattr(result, "category", None)),
            *[_canonical_category(x) for x in (getattr(result, "category_candidates", None) or [])],
        }
        target_categories = {x for x in target_categories if x not in _UNKNOWN_CATEGORIES}

        neighbor_categories = set()
        if 0 <= neighbor_idx < len(categories_state):
            neighbor_categories.add(_canonical_category(categories_state[neighbor_idx]))
        if 0 <= neighbor_idx < len(category_candidates_state):
            rows = category_candidates_state[neighbor_idx]
            if isinstance(rows, (list, tuple)):
                neighbor_categories.update(_canonical_category(x) for x in rows)
        neighbor_categories = {x for x in neighbor_categories if x not in _UNKNOWN_CATEGORIES}

        if target_categories and neighbor_categories:
            return bool(target_categories & neighbor_categories)

        target_super = _canonical_category(getattr(result, "supercategory", None))
        neighbor_super = ""
        if 0 <= neighbor_idx < len(supercategories_state):
            neighbor_super = _canonical_category(supercategories_state[neighbor_idx])
        if (
            target_super
            and neighbor_super
            and target_super not in _UNKNOWN_CATEGORIES
            and neighbor_super not in _UNKNOWN_CATEGORIES
        ):
            return target_super == neighbor_super
        return True

    def _maybe_apply_caption_merges(
        self,
        results: List[ObjectCaptionResult],
        result_object_indices: List[Optional[int]],
    ) -> None:
        """
        Suggest merges for captioned objects using spatial Hellinger proximity and
        conservative semantic+visual similarity. Caption merge is an identity
        operation, so use stricter gates than retrieval.
        """
        if torch is None or not results:
            return

        means = self.scene_state.get("means")
        cov6 = self.scene_state.get("cov6")
        object_ids = self.scene_state.get("object_id")
        if (
            not isinstance(means, torch.Tensor)
            or not isinstance(cov6, torch.Tensor)
            or not isinstance(object_ids, torch.Tensor)
        ):
            return
        N = means.shape[0]
        if cov6.shape[0] != N or object_ids.shape[0] != N:
            return

        active_flags = self.scene_state.get("active")
        captions_state: List[str] = self.scene_state.get("object_caption", []) or []
        caption_embeddings_state: List[List[float]] = self.scene_state.get("object_caption_embedding", []) or []
        caption_embeddings_history_state: List[List[List[float]]] = (
            self.scene_state.get("object_caption_embedding_history", []) or []
        )
        siglip2_embeddings_state: List[List[float]] = self.scene_state.get("object_siglip2_embedding", []) or []
        siglip2_embeddings_history_state: List[List[List[float]]] = (
            self.scene_state.get("object_siglip2_embedding_history", []) or []
        )

        caption_merge_thresh = float(self._caption_merge_caption_thresh)
        siglip2_merge_thresh = float(self._caption_merge_siglip2_thresh)

        candidates: List[tuple[int, int, Optional[List[float]], Optional[List[float]]]] = []
        caption_dim = self._embed_dim or 0
        siglip2_dim = self._siglip2_dim or 0
        for res_idx, obj_idx in enumerate(result_object_indices):
            if obj_idx is None or obj_idx < 0 or obj_idx >= N:
                continue
            if bool(getattr(results[res_idx], "is_recaption", False)):
                continue
            if isinstance(active_flags, torch.Tensor) and obj_idx < active_flags.shape[0]:
                with contextlib.suppress(Exception):
                    if not bool(active_flags[obj_idx].item()):
                        continue

            caption_vec: Optional[List[float]] = None
            caption_emb = getattr(results[res_idx], "caption_embedding", None)
            if isinstance(caption_emb, (list, tuple)) and len(caption_emb) > 0:
                if caption_dim <= 0:
                    caption_dim = len(caption_emb)
                if caption_dim > 0 and len(caption_emb) == caption_dim:
                    caption_vec = list(caption_emb)

            siglip2_vec: Optional[List[float]] = None
            siglip2_emb = getattr(results[res_idx], "siglip2_cls_embedding", None)
            if isinstance(siglip2_emb, (list, tuple)) and len(siglip2_emb) > 0:
                if siglip2_dim <= 0:
                    siglip2_dim = len(siglip2_emb)
                if siglip2_dim > 0 and len(siglip2_emb) == siglip2_dim:
                    siglip2_vec = list(siglip2_emb)

            if caption_vec is None and siglip2_vec is None:
                continue
            candidates.append((res_idx, int(obj_idx), caption_vec, siglip2_vec))

        if not candidates:
            return

        device = means.device
        det_indices = torch.as_tensor([entry[1] for entry in candidates], device=device, dtype=torch.long)
        det_means = means[det_indices]
        det_cov6 = cov6[det_indices]

        det_active_mask = None
        if isinstance(active_flags, torch.Tensor) and active_flags.numel() >= N:
            det_active_mask = active_flags.to(device=device, dtype=torch.bool)[det_indices]

        neighbors, _ = get_neighbors_by_hellinger_distance(
            {"means": det_means, "cov6": det_cov6},
            {"means": means, "cov6": cov6, "active": active_flags},
            hellinger_thresh=float(self._caption_merge_hellinger_thresh),
            active_mask=det_active_mask,
        )

        caption_db_rows_by_idx: Dict[int, torch.Tensor] = {}
        if caption_dim > 0:
            for idx in range(N):
                rows: List[List[float]] = []
                hist_rows = caption_embeddings_history_state[idx] if idx < len(caption_embeddings_history_state) else []
                if isinstance(hist_rows, (list, tuple)):
                    for vec in hist_rows:
                        if isinstance(vec, (list, tuple)) and len(vec) == caption_dim:
                            rows.append(list(vec))
                if not rows:
                    emb_vec = caption_embeddings_state[idx] if idx < len(caption_embeddings_state) else []
                    if isinstance(emb_vec, (list, tuple)) and len(emb_vec) == caption_dim:
                        rows.append(list(emb_vec))
                if not rows:
                    continue
                rows_tensor = torch.as_tensor(rows, device=device, dtype=torch.float32)
                rows_tensor = torch.nn.functional.normalize(rows_tensor, p=2, dim=1, eps=1e-12)
                caption_db_rows_by_idx[idx] = rows_tensor

        siglip2_db_rows_by_idx: Dict[int, torch.Tensor] = {}
        if siglip2_dim > 0:
            for idx in range(N):
                rows: List[List[float]] = []
                hist_rows = siglip2_embeddings_history_state[idx] if idx < len(siglip2_embeddings_history_state) else []
                if isinstance(hist_rows, (list, tuple)):
                    for vec in hist_rows:
                        if isinstance(vec, (list, tuple)) and len(vec) == siglip2_dim:
                            rows.append(list(vec))
                if not rows:
                    emb_vec = siglip2_embeddings_state[idx] if idx < len(siglip2_embeddings_state) else []
                    if isinstance(emb_vec, (list, tuple)) and len(emb_vec) == siglip2_dim:
                        rows.append(list(emb_vec))
                if not rows:
                    continue
                rows_tensor = torch.as_tensor(rows, device=device, dtype=torch.float32)
                rows_tensor = torch.nn.functional.normalize(rows_tensor, p=2, dim=1, eps=1e-12)
                siglip2_db_rows_by_idx[idx] = rows_tensor

        if not caption_db_rows_by_idx and not siglip2_db_rows_by_idx:
            return

        caption_det_by_local: Dict[int, torch.Tensor] = {}
        if caption_dim > 0:
            caption_rows: List[List[float]] = []
            caption_locals: List[int] = []
            for det_local_idx, (_, _, caption_vec, _) in enumerate(candidates):
                if caption_vec is None:
                    continue
                caption_rows.append(caption_vec)
                caption_locals.append(det_local_idx)
            if caption_rows:
                caption_det_tensor = torch.as_tensor(caption_rows, device=device, dtype=torch.float32)
                caption_det_tensor = torch.nn.functional.normalize(caption_det_tensor, p=2, dim=1, eps=1e-12)
                for row_idx, det_local_idx in enumerate(caption_locals):
                    caption_det_by_local[det_local_idx] = caption_det_tensor[row_idx]

        siglip2_det_by_local: Dict[int, torch.Tensor] = {}
        if siglip2_dim > 0:
            siglip2_rows: List[List[float]] = []
            siglip2_locals: List[int] = []
            for det_local_idx, (_, _, _, siglip2_vec) in enumerate(candidates):
                if siglip2_vec is None:
                    continue
                siglip2_rows.append(siglip2_vec)
                siglip2_locals.append(det_local_idx)
            if siglip2_rows:
                siglip2_det_tensor = torch.as_tensor(siglip2_rows, device=device, dtype=torch.float32)
                siglip2_det_tensor = torch.nn.functional.normalize(siglip2_det_tensor, p=2, dim=1, eps=1e-12)
                for row_idx, det_local_idx in enumerate(siglip2_locals):
                    siglip2_det_by_local[det_local_idx] = siglip2_det_tensor[row_idx]

        try:
            object_id_list = [int(x) for x in object_ids.cpu().tolist()]
        except Exception:
            return

        for det_local_idx, (res_idx, obj_idx, _, _) in enumerate(candidates):
            neigh = neighbors[det_local_idx] if det_local_idx < len(neighbors) else None
            if neigh is None or neigh.numel() == 0:
                continue
            neigh_idx = neigh.to(device=device, dtype=torch.long)
            neigh_idx = neigh_idx[(neigh_idx != obj_idx) & (neigh_idx >= 0) & (neigh_idx < N)]
            if neigh_idx.numel() == 0:
                continue

            det_caption_vec = caption_det_by_local.get(det_local_idx)
            det_siglip2_vec = siglip2_det_by_local.get(det_local_idx)
            merge_indices_list: List[int] = []
            for neighbor_idx in neigh_idx.tolist():
                if are_cannot_linked_indices(self.scene_state, int(obj_idx), int(neighbor_idx)):
                    continue
                if not self._caption_merge_category_compatible(results[res_idx], int(neighbor_idx)):
                    continue
                lang_merge_pass = False
                siglip2_merge_pass = False
                has_caption_channel = False
                has_siglip2_channel = False

                if det_caption_vec is not None:
                    db_rows_caption = caption_db_rows_by_idx.get(int(neighbor_idx))
                    if db_rows_caption is not None and db_rows_caption.numel() > 0:
                        has_caption_channel = True
                        cos_sim_caption = torch.sum(db_rows_caption * det_caption_vec.unsqueeze(0), dim=1)
                        if cos_sim_caption.numel() > 0 and bool((cos_sim_caption >= caption_merge_thresh).any().item()):
                            lang_merge_pass = True

                if det_siglip2_vec is not None:
                    db_rows_siglip2 = siglip2_db_rows_by_idx.get(int(neighbor_idx))
                    if db_rows_siglip2 is not None and db_rows_siglip2.numel() > 0:
                        has_siglip2_channel = True
                        cos_sim_siglip2 = torch.sum(db_rows_siglip2 * det_siglip2_vec.unsqueeze(0), dim=1)
                        if cos_sim_siglip2.numel() > 0 and bool((cos_sim_siglip2 >= siglip2_merge_thresh).any().item()):
                            siglip2_merge_pass = True

                if self._caption_merge_require_visual:
                    merge_pass = lang_merge_pass and siglip2_merge_pass
                elif has_caption_channel and has_siglip2_channel:
                    merge_pass = lang_merge_pass and siglip2_merge_pass
                else:
                    merge_pass = lang_merge_pass or siglip2_merge_pass

                if merge_pass:
                    merge_indices_list.append(int(neighbor_idx))
            if not merge_indices_list:
                continue
            merge_indices = torch.as_tensor(merge_indices_list, device=device, dtype=torch.long)

            target_canonical = self._resolve_canonical_object_id(int(results[res_idx].object_id))
            merge_ids: List[int] = []
            for idx in merge_indices.tolist():
                if 0 <= idx < len(object_id_list):
                    cand_id = self._resolve_canonical_object_id(int(object_id_list[idx]))
                    if cand_id != target_canonical:
                        merge_ids.append(cand_id)

            if not merge_ids:
                continue

            existing = getattr(results[res_idx], "merge_object_ids", None) or []
            try:
                existing_set = {int(x) for x in existing}
            except Exception:
                existing_set = set()
            merged_ids = sorted(set(merge_ids).union(existing_set))
            results[res_idx].merge_object_ids = merged_ids

            candidate_indices = [int(obj_idx)] + [int(x) for x in merge_indices.tolist()]
            best_obj_idx: Optional[int] = None
            best_area = -1.0
            for cand_idx in candidate_indices:
                area = self._max_caption_bbox_area_for_object(int(cand_idx))
                if area > best_area:
                    best_area = area
                    best_obj_idx = int(cand_idx)

            chosen_caption = ""
            if best_obj_idx is not None and best_area > 0.0:
                if best_obj_idx == int(obj_idx):
                    chosen_caption = str(getattr(results[res_idx], "caption", "") or "").strip()
                elif best_obj_idx < len(captions_state):
                    chosen_caption = str(captions_state[best_obj_idx] or "").strip()

            if not chosen_caption:
                caption_parts = [results[res_idx].caption]
                for idx in merge_indices.tolist():
                    cap = captions_state[idx] if idx < len(captions_state) else ""
                    if cap:
                        caption_parts.append(str(cap))
                chosen_caption = max(
                    (p for p in (str(part).strip() for part in caption_parts) if p),
                    key=len,
                    default="",
                )
            if chosen_caption:
                results[res_idx].caption = chosen_caption

    def _parse_caption_response(self, text: str) -> Dict[str, Any]:
        """
        Parse model output into a structured caption + merge payload. Falls back
        to treating the raw text as the caption if JSON decoding fails.
        """
        default_caption = text.strip() if text else ""
        default = {
            "caption": default_caption,
            "category": "",
            "supercategory": "",
            "category_candidates": [],
            "key_attributes": [],
            "merge_object_ids": [],
            "is_clear_object": None,
            "decision": None,
        }
        if not text:
            return default

        if self._caption_expects_json:
            candidate = text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]

            try:
                data = json.loads(candidate)
            except Exception:
                if self._uses_structured_caption_schema():
                    default["caption"] = ""
                    default["error_reason"] = "invalid_structured_caption"
                return default
        else:
            data = text

        if self._uses_structured_caption_schema():
            structured = parse_structured_caption(text)
            if structured.is_clear_object:
                category = structured.category or ""
                return {
                    "caption": structured.description,
                    "category": category,
                    "supercategory": structured.supercategory or "",
                    "category_candidates": [category] if category else [],
                    "key_attributes": list(structured.attributes),
                    "merge_object_ids": [],
                    "is_clear_object": True,
                    "decision": "keep",
                    "raw_response": structured.raw_response,
                    "valid_json": structured.valid_json,
                }

            explicit_unclear = is_explicit_unclear_caption(structured)
            return {
                "caption": "",
                "category": "",
                "supercategory": "",
                "category_candidates": [],
                "key_attributes": [],
                "merge_object_ids": [],
                "is_clear_object": False if explicit_unclear else None,
                "decision": "drop" if explicit_unclear else structured.decision,
                "raw_response": structured.raw_response,
                "valid_json": structured.valid_json,
                "error_reason": "unclear_object" if explicit_unclear else "invalid_structured_caption",
            }

        if self._caption_version == 1:
            raw_caption = str(data.get("caption", "") or data.get("new_caption", "") or "").strip()
        elif self._caption_version == 2:
            short_description = str(data.get("short_description", "") or "").strip()
            label = str(data.get("label", "") or "").strip()
            raw_caption = f"{label}: {short_description}"
        elif self._caption_version == 3:
            key_words = str(data.get("key_words", "") or "").strip()
            label = str(data.get("label", "") or "").strip()
            raw_caption = f"{label}: {key_words}"
        elif self._caption_version == 4:
            new_caption = str(data.get("new_caption", "") or "").strip()
            raw_caption = new_caption
        elif self._caption_version == 5:
            object_name = str(data.get("object_name", "") or "").strip()
            attributes = str(data.get("attributes", "") or "").strip()
            functional_category = str(data.get("functional_category", "") or "").strip()
            raw_caption = f"{object_name}: {attributes}: {functional_category}"
        elif (self._caption_version == 6) or (self._caption_version == 7) or (self._caption_version == 8):
            new_caption = str(data.get("new_caption", "") or "").strip()
            raw_caption = new_caption
        elif self._caption_version == 9:
            options = str(data.get("options", "") or "").strip()
            attr = str(data.get("attr", "") or "").strip()
            raw_caption = f"{options} : {attr}"
        elif self._caption_version == 10:
            labels = str(data.get("labels", "") or "").strip()
            attributes = str(data.get("attributes", "") or "").strip()
            raw_caption = f"{labels} : {attributes}"
        elif (self._caption_version == 11) or (self._caption_version == 12) or (self._caption_version == 13):
            name = str(data.get("name", "") or "").strip()
            attr = str(data.get("attr", "") or "").strip()
            usage = str(data.get("usage", "") or "").strip()
            raw_caption = f"{name} : {attr} : {usage}"
        elif self._caption_version == 14 or self._caption_version == 15 or self._caption_version == 16:
            raw_caption = str(data).strip()
        elif self._caption_version == 17:
            raw_caption = str(data.get("caption", "") or data.get("new_caption", "") or "").strip()
        elif self._caption_version == 18:
            caption = str(data.get("caption", "") or data.get("new_caption", "") or "").strip()
            too_blurry = bool(data.get("too_blurry", False))
            raw_caption = f"{caption} (too_blurry: {too_blurry})"
        elif self._caption_version == 19:
            raw_caption = str(data.get("caption", "") or data.get("new_caption", "") or "").strip()
        else:
            raise ValueError(f"Invalid caption version: {self._caption_version}")

        if not isinstance(data, dict):
            return default

        # Backwards compatibility: allow both structured and plain caption payloads.

        raw_candidates = data.get("category_candidates")
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        candidates: List[str] = []
        for item in raw_candidates:
            s = str(item or "").strip()
            if not s:
                continue
            if s in candidates:
                continue
            candidates.append(s)
            if len(candidates) >= 3:
                break

        raw_attrs = data.get("key_attributes")
        if not isinstance(raw_attrs, list):
            raw_attrs = []
        attrs: List[str] = []
        for item in raw_attrs:
            s = str(item or "").strip()
            if not s:
                continue
            if s in attrs:
                continue
            attrs.append(s)
            if len(attrs) >= 3:
                break

        caption = raw_caption

        # If the model returned only {"new_caption": "..."} (expected), infer candidates from " or ".
        if caption and not candidates:
            for part in (p.strip() for p in caption.split(":")[0].split(" or ")):
                if not part:
                    continue
                if part in candidates:
                    continue
                candidates.append(part)
                if len(candidates) >= 3:
                    break

        category = candidates[0] if candidates else ""

        return {
            "caption": caption or default_caption,
            "category": category,
            "category_candidates": candidates,
            "key_attributes": attrs,
            "merge_object_ids": [],
        }

    def _format_qwen_box(self, box_info: Optional[dict]) -> Optional[str]:
        """Convert a pixel-space bbox and size dict into Qwen <box> format (0-1000 range)."""
        if not box_info or not isinstance(box_info, dict):
            return None
        bbox = box_info.get("bbox")
        size = box_info.get("size")
        if bbox is None or size is None or len(bbox) != 4 or len(size) != 2:
            return None
        w, h = size
        if w is None or h is None:
            return None
        try:
            w_f = float(w)
            h_f = float(h)
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if w_f <= 0.0 or h_f <= 0.0:
            return None

        def _norm(v: float, denom: float) -> int:
            scaled = (v / denom) * 1000.0
            return int(round(max(0.0, min(999.0, scaled))))

        nx1 = _norm(x1, w_f)
        ny1 = _norm(y1, h_f)
        nx2 = _norm(x2, w_f)
        ny2 = _norm(y2, h_f)
        return f"<box>({nx1},{ny1}),({nx2},{ny2})</box>"

    def _format_spatial_context(self, box_info: Optional[dict]) -> Optional[str]:
        """
        Format object approximate size (and optionally 3D position) from ``box_info``
        into a human-readable string for the captioning prompt.

        Size is taken from cov6; position is included only when
        _caption_spatial_context_include_position is True.
        Returns ``None`` when spatial data is absent (toggle off or cov6 missing).
        """
        if not box_info or not isinstance(box_info, dict):
            return None
        obj_mean = box_info.get("object_mean")
        obj_cov6 = box_info.get("object_cov6")
        if obj_cov6 is None:
            return None

        parts: List[str] = []

        # Position (optional) -------------------------------------------------
        if self._caption_spatial_context_include_position and obj_mean is not None:
            try:
                x, y, z = [float(v) for v in obj_mean[:3]]
                parts.append(f"OBJECT POSITION (world frame): [{x:.2f}, {y:.2f}, {z:.2f}]")
            except Exception:
                pass

        # Approximate size from covariance eigenvalues (default) --------------
        try:
            cov6_vals = [float(v) for v in obj_cov6[:6]]
            # Unpack symmetric 3x3: [xx, xy, xz, yy, yz, zz]
            xx, xy, xz, yy, yz, zz = cov6_vals
            cov_mat = np.array(
                [
                    [xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz],
                ],
                dtype=np.float64,
            )
            eigvals = np.linalg.eigvalsh(cov_mat)
            # Clamp tiny/negative eigenvalues (numerical noise).
            eigvals = np.maximum(eigvals, 0.0)
            # 2*sqrt(eigenvalue) gives approximate axis-aligned extent.
            extents = 2.0 * np.sqrt(eigvals)
            # Sort descending so the output is (largest, ..., smallest).
            extents = np.sort(extents)[::-1]
            parts.append(f"OBJECT APPROXIMATE SIZE (meters): {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}")
        except Exception:
            pass

        return "\n".join(parts) if parts else None

    def warm_up_model(self) -> bool:
        """Eagerly download/initialize the caption model before processing batches."""
        if self._caption_server == "sglang":
            ok = self._build_sglang_model()
        elif self._caption_server == "ollama":
            ok = self._build_ollama_model()
        elif self._caption_server == "vllm":
            ok = self._build_vllm_model()
        else:
            print(f"[CaptionWorker] Invalid caption server: {self._caption_server}")
            ok = False

        # Pre-load auxiliary HF models so the first per-frame batch doesn't
        # time out at the RPC layer while the model is still loading. SigLIP2
        # in particular takes ~1.7s to load and the default RPC timeout is
        # 350ms, which causes the first 1-2 batches to drop their crops.
        if self._siglip2_enabled:
            t0 = time.perf_counter()
            if self._build_siglip2_model():
                print(f"[CaptionWorker] SigLIP2 warm-up successful ({time.perf_counter() - t0:.2f}s)")
            else:
                print("[CaptionWorker] SigLIP2 warm-up failed (see prior log)")
        if self._qwen3_vl_embed_enabled and self._qwen3_vl_embed_backend in {"hf", "auto"}:
            t0 = time.perf_counter()
            if self._build_qwen3_vl_embed_model():
                print(
                    f"[CaptionWorker] Qwen3-VL embed (HF) warm-up successful "
                    f"({time.perf_counter() - t0:.2f}s)"
                )
            else:
                print("[CaptionWorker] Qwen3-VL embed (HF) warm-up failed (see prior log)")

        return ok

    def _vllm_chat_url(self) -> str:
        base = self._vllm_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _vllm_embed_url(self) -> str:
        base = self._embed_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def _vllm_qwen3_vl_embed_url(self) -> str:
        base = self._qwen3_vl_embed_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def _probe_vllm_server(self) -> bool:
        url = self._vllm_base_url.rstrip("/")
        headers = {}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        try:
            resp = requests.get(f"{url}/models", headers=headers, timeout=min(self._vllm_timeout_s, 10))
            if resp.status_code == 200:
                # Validate the configured model id exists on the server to avoid noisy runtime 404s.
                try:
                    payload = resp.json()
                except Exception as json_exc:
                    text_preview = (resp.text or "")[:200].strip()
                    if "<" in text_preview and ">" in text_preview:
                        print(
                            "[CaptionWorker] WARNING: vLLM endpoint returned HTML instead of JSON. "
                            f"Wrong URL or server not running? url={url!r} response_preview={text_preview!r}"
                        )
                    else:
                        print(
                            "[CaptionWorker] WARNING: vLLM endpoint returned invalid JSON. "
                            f"url={url!r} error={json_exc}"
                        )
                    return False
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, list):
                    ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                    if ids and self._vllm_model not in ids:
                        print(
                            "[CaptionWorker] WARNING: vLLM server reachable but model id not served: "
                            f"requested={self._vllm_model!r} available={ids!r}"
                        )
                        return False
                return True
            resp_health = requests.get(f"{url}/health", headers=headers, timeout=min(self._vllm_timeout_s, 10))
            if resp_health.status_code == 200:
                return True
            print(
                "[CaptionWorker] WARNING: vLLM server not reachable or returned error: "
                f"url={url!r} /models status={resp.status_code} /health status={resp_health.status_code} "
                "(response may be HTML if URL is wrong)"
            )
            return False
        except Exception as exc:
            print(
                "[CaptionWorker] WARNING: Cannot reach vLLM caption server. "
                f"url={url!r} error={exc!r} (check VLLM_BASE_URL and that the server is running)"
            )
            return False

    def _probe_embed_server(self) -> bool:
        url = self._embed_base_url.rstrip("/")
        headers = {}
        if self._embed_api_key:
            headers["Authorization"] = f"Bearer {self._embed_api_key}"
        try:
            resp = requests.get(f"{url}/models", headers=headers, timeout=min(self._embed_timeout_s, 10))
            if resp.status_code == 200:
                with contextlib.suppress(Exception):
                    payload = resp.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(data, list):
                        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                        if ids and self._embed_model not in ids:
                            if self.debug:
                                print(
                                    "[CaptionWorker] vLLM embed server reachable but model id not served: "
                                    f"requested={self._embed_model!r} available={ids!r}"
                                )
                            return False
                return True
            resp = requests.get(f"{url}/health", headers=headers, timeout=min(self._embed_timeout_s, 10))
            return resp.status_code == 200
        except Exception:
            return False

    def _probe_qwen3_vl_embed_server(self) -> bool:
        url = self._qwen3_vl_embed_base_url.rstrip("/")
        headers = {}
        if self._qwen3_vl_embed_api_key:
            headers["Authorization"] = f"Bearer {self._qwen3_vl_embed_api_key}"
        try:
            resp = requests.get(f"{url}/models", headers=headers, timeout=min(self._qwen3_vl_embed_timeout_s, 10))
            if resp.status_code == 200:
                with contextlib.suppress(Exception):
                    payload = resp.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(data, list):
                        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                        if ids and self._qwen3_vl_embed_model_name not in ids:
                            if self.debug or self._qwen3_vl_embed_debug:
                                print(
                                    "[CaptionWorker] vLLM qwen3-vl embed server reachable but model id not served: "
                                    f"requested={self._qwen3_vl_embed_model_name!r} available={ids!r}"
                                )
                            return False
                return True
            resp = requests.get(f"{url}/health", headers=headers, timeout=min(self._qwen3_vl_embed_timeout_s, 10))
            return resp.status_code == 200
        except Exception:
            return False

    # Timing summary ------------------------------------------------------
    def timing_summary(self) -> dict:
        avg_batch_size = (self._batch_size_sum / self._batch_count) if self._batch_count else 0.0
        avg_gen_time_ms = (self._gen_time_sum / self._gen_batch_count) * 1000.0 if self._gen_batch_count else 0.0
        return {
            "batches": self._batch_count,
            "avg_batch_size": avg_batch_size,
            "gen_batches": self._gen_batch_count,
            "avg_gen_time_ms": avg_gen_time_ms,
        }

    def log_timing_summary(self) -> None:
        summary = self.timing_summary()
        timing_parts = {}
        for key, total in self._timing_sum.items():
            count = self._timing_count.get(key, 0)
            avg_ms = (total / count) * 1000.0 if count else 0.0
            timing_parts[key] = round(avg_ms, 2)
        print(
            "[CaptionWorker] batches={batches} avg_batch_size={avg_batch_size:.2f} "
            "gen_batches={gen_batches} avg_gen_time_ms={avg_gen_time_ms:.1f} "
            "timings_ms={timing_parts}".format(timing_parts=timing_parts, **summary)
        )

    # ------------------------------------------------------------------
    # View selection helpers
    def _ensure_torch(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for caption worker view selection")

    def _unpack_cov6(self, cov6: torch.Tensor) -> torch.Tensor:
        """
        cov6: (..., 6) -> (..., 3, 3) symmetric covariance matrix
        Order: [xx, xy, xz, yy, yz, zz]
        """
        self._ensure_torch()
        xx, xy, xz, yy, yz, zz = cov6.unbind(-1)
        row0 = torch.stack([xx, xy, xz], dim=-1)
        row1 = torch.stack([xy, yy, yz], dim=-1)
        row2 = torch.stack([xz, yz, zz], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    def _view_variance(self, cov6_vec) -> float:
        """Return scalar variance (trace of covariance) for a single view."""
        self._ensure_torch()
        cov6_tensor = cov6_vec
        if not isinstance(cov6_tensor, torch.Tensor):
            cov6_tensor = torch.as_tensor(cov6_vec, dtype=torch.float32)
        cov = self._unpack_cov6(cov6_tensor.view(1, 6))[0]
        return float(torch.trace(cov).item())

    def _cluster_view_means(self, mu: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Simple k-means on view centers. Returns cluster ids of shape (V,).
        """
        self._ensure_torch()
        assert mu.ndim == 2 and mu.shape[1] == 3
        V = mu.shape[0]
        if V == 0:
            return torch.empty(0, dtype=torch.long, device=mu.device)
        if V <= k:
            # Each view gets its own cluster up to V
            ids = torch.arange(V, device=mu.device)
            if V < k:
                ids = torch.nn.functional.pad(ids, (0, k - V), value=V - 1)
            return ids[:V]

        # Initialize centers using first k views
        centers = mu[:k].clone()
        for _ in range(5):
            # Compute squared distances
            dists = ((mu[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (V, k)
            cluster_ids = torch.argmin(dists, dim=1)  # (V,)

            # Recompute centers
            new_centers = centers.clone()
            for c in range(k):
                mask = cluster_ids == c
                if mask.any():
                    new_centers[c] = mu[mask].mean(dim=0)
            centers = new_centers
        return cluster_ids

    @staticmethod
    def _obs_has_image(obs: object) -> bool:
        if obs is None:
            return False
        if isinstance(obs, dict):
            return obs.get("image_caption") is not None or obs.get("image") is not None
        return True

    def _candidate_view_indices(self, object_index: int) -> List[int]:
        rgb_obs = self.scene_state.get("rgb_observations", [])
        if object_index < 0 or object_index >= len(rgb_obs):
            return []
        obs_list = rgb_obs[object_index]
        if not isinstance(obs_list, list):
            return []
        out: List[int] = []
        for idx, obs in enumerate(obs_list):
            if self._obs_has_image(obs):
                out.append(int(idx))
        return out

    def _select_recaption_views_for_object(self, object_index: int) -> List[int]:
        """
        Select two recaption views:
          1) closest viewpoint to the object center,
          2) viewpoint with the largest angular separation from (1).
        """
        if torch is None:
            return []
        high_quality_views = self.scene_state.get("high_quality_views", []) or []
        means = self.scene_state.get("means")
        if (
            object_index < 0
            or object_index >= len(high_quality_views)
            or not isinstance(means, torch.Tensor)
            or object_index >= int(means.shape[0])
        ):
            return []

        candidate_indices = self._candidate_view_indices(object_index)
        if len(candidate_indices) < 2:
            return []

        obj_center = means[object_index].detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
        if obj_center.numel() != 3 or not torch.isfinite(obj_center).all():
            return []

        row = high_quality_views[object_index]
        if not isinstance(row, list):
            return []

        valid_idx: List[int] = []
        dirs: List[torch.Tensor] = []
        dists: List[float] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(row):
                continue
            vp = row[idx]
            if not isinstance(vp, torch.Tensor):
                with contextlib.suppress(Exception):
                    vp = torch.as_tensor(vp, dtype=torch.float32)
            if not isinstance(vp, torch.Tensor) or vp.numel() < 3:
                continue
            vp = vp.detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
            if not torch.isfinite(vp).all():
                continue
            vec = vp - obj_center
            dist = float(torch.linalg.norm(vec).item())
            if not np.isfinite(dist) or dist <= 1e-6:
                continue
            valid_idx.append(int(idx))
            dirs.append(vec / dist)
            dists.append(dist)

        if len(valid_idx) < 2:
            return []

        first_rel = int(np.argmin(np.asarray(dists, dtype=np.float32)))
        first_idx = int(valid_idx[first_rel])
        first_dir = dirs[first_rel]

        best_second_rel: Optional[int] = None
        best_dot: Optional[float] = None
        for rel_i, dir_i in enumerate(dirs):
            if rel_i == first_rel:
                continue
            dot = float(torch.dot(first_dir, dir_i).item())
            if best_dot is None or dot < best_dot:
                best_dot = dot
                best_second_rel = rel_i

        if best_second_rel is None:
            return []
        second_idx = int(valid_idx[best_second_rel])
        if second_idx == first_idx:
            return []
        return [first_idx, second_idx]

    def _collect_recaption_fill_tasks(
        self,
        *,
        limit: int,
        exclude_object_ids: set[int],
    ) -> List[ObjectCaptionTask]:
        if limit <= 0:
            return []
        if torch is None:
            return []
        object_ids = self.scene_state.get("object_id")
        active_flags = self.scene_state.get("active")
        means = self.scene_state.get("means")
        captions = self.scene_state.get("object_caption", []) or []
        hq_flags = self.scene_state.get("high_quality_captioning", []) or []
        robot_pos = self.scene_state.get("current_robot_position")

        if object_ids is None or not isinstance(means, torch.Tensor):
            return []
        if not isinstance(robot_pos, torch.Tensor):
            with contextlib.suppress(Exception):
                robot_pos = torch.as_tensor(robot_pos, dtype=torch.float32)
        if not isinstance(robot_pos, torch.Tensor) or robot_pos.numel() < 3:
            return []
        robot_pos = robot_pos.detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
        if not torch.isfinite(robot_pos).all():
            return []

        now = time.monotonic()
        candidates: List[tuple[float, int, int]] = []
        num_objects = int(min(len(object_ids), means.shape[0], len(hq_flags)))
        for obj_idx in range(num_objects):
            if not bool(hq_flags[obj_idx]):
                continue
            is_active = True
            if active_flags is not None and obj_idx < len(active_flags):
                with contextlib.suppress(Exception):
                    is_active = (
                        bool(active_flags[obj_idx].item())
                        if hasattr(active_flags[obj_idx], "item")
                        else bool(active_flags[obj_idx])
                    )
            if not is_active:
                continue
            caption_text = ""
            if obj_idx < len(captions):
                with contextlib.suppress(Exception):
                    caption_text = str(captions[obj_idx] or "").strip()
            if not caption_text:
                continue
            obj_mean = means[obj_idx].detach().to("cpu", dtype=torch.float32, copy=False).view(-1)[:3]
            if obj_mean.numel() != 3 or not torch.isfinite(obj_mean).all():
                continue

            # Check if we're in "final pass" mode (time threshold triggered)
            sim_time_remaining = self.scene_state.get("sim_time_remaining_sec")
            in_final_pass = (
                self._recaption_time_threshold_sec > 0.0
                and sim_time_remaining is not None
                and float(sim_time_remaining) < self._recaption_time_threshold_sec
            )

            if not in_final_pass:
                # Distance-based check
                distance = float(torch.linalg.norm(obj_mean - robot_pos).item())
                if not np.isfinite(distance) or distance < self._recaption_min_distance_m:
                    continue
            else:
                # In final pass mode, still compute distance for sorting, but don't filter by it
                distance = float(torch.linalg.norm(obj_mean - robot_pos).item())
                if not np.isfinite(distance):
                    distance = 0.0

            raw_id = object_ids[obj_idx]
            try:
                object_id = int(raw_id.item()) if hasattr(raw_id, "item") else int(raw_id)
            except Exception:
                continue
            if object_id in exclude_object_ids:
                continue
            last_attempt = float(self._recaption_last_attempt_s.get(object_id, -1e9))
            if (now - last_attempt) < self._recaption_cooldown_s:
                continue
            recaption_views = self._select_recaption_views_for_object(obj_idx)
            if len(recaption_views) < 2:
                continue
            candidates.append((distance, int(obj_idx), int(object_id)))

        if not candidates:
            return []
        candidates.sort(key=lambda x: x[0], reverse=True)

        out: List[ObjectCaptionTask] = []
        for _, obj_idx, object_id in candidates[:limit]:
            self._recaption_last_attempt_s[object_id] = now
            out.append(
                ObjectCaptionTask(
                    object_index=int(obj_idx),
                    object_id=int(object_id),
                    is_recaption=True,
                )
            )
        return out

    def _select_views_for_object(self, object_index: int) -> List[int]:
        """Select a single view index for captioning.

        Robust to historical mismatches between `view_cov6/view_means` and `rgb_observations`
        by only selecting indices that exist in `rgb_observations` (and falling back to an
        existing observation when needed).
        """
        view_means = self.scene_state.get("view_means", [])
        view_cov6 = self.scene_state.get("view_cov6", [])
        rgb_obs = self.scene_state.get("rgb_observations", [])
        if object_index >= len(view_means) or object_index >= len(view_cov6):
            return []

        means_list = view_means[object_index]
        cov6_list = view_cov6[object_index]
        V = len(means_list)
        if V == 0:
            return []

        obs_list = rgb_obs[object_index] if object_index < len(rgb_obs) else []
        if not obs_list:
            return []

        # If view lists and observations are out-of-sync, restrict selection to indices that
        # exist in both (we cannot caption without an observation image).
        V_obs = len(obs_list)
        V_cov = len(cov6_list)
        V_common = min(V, V_cov, V_obs)
        if V_common <= 0:
            return [0] if V_obs > 0 else []

        # Prefer indices that actually have a stored image (not just a placeholder).
        candidate_indices: List[int] = [i for i in range(V_common) if self._obs_has_image(obs_list[i])]

        if not candidate_indices:
            # Fall back to any existing observation index.
            fallback_idx = next((i for i, obs in enumerate(obs_list) if obs is not None), 0)
            return [int(min(max(fallback_idx, 0), V_obs - 1))]

        best_idx_by_bbox: Optional[int] = None
        best_area = -1.0
        for i in candidate_indices:
            area = self._caption_crop_area(obs_list[i])
            if area > best_area:
                best_area = area
                best_idx_by_bbox = int(i)
        if best_idx_by_bbox is not None and best_area > 0.0:
            return [int(min(max(best_idx_by_bbox, 0), V_obs - 1))]

        if torch is None:
            # Fallback: choose the first view when torch is unavailable
            return [0]

        # Use variance = trace(cov) = xx + yy + zz (from packed cov6) for speed.
        cov6 = torch.stack(cov6_list[:V_common], dim=0)  # (V_common, 6)
        var_all = cov6[:, 0] + cov6[:, 3] + cov6[:, 5]  # (V_common,)

        cand_idx = torch.as_tensor(candidate_indices, dtype=torch.long)
        cand_vars = var_all[cand_idx]
        if cand_vars.numel() == 0:
            return [int(min(max(candidate_indices[0], 0), V_obs - 1))]

        best_pos = int(torch.argmax(cand_vars).item())
        best_idx_by_var = int(cand_idx[best_pos].item())

        chosen = best_idx_by_var
        return [int(min(max(chosen, 0), V_obs - 1))]

    @staticmethod
    def _bbox_area_from_size(bbox: object, size: object) -> float:
        if not (
            isinstance(bbox, (list, tuple)) and isinstance(size, (list, tuple)) and len(bbox) >= 4 and len(size) >= 2
        ):
            return 0.0
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            w_img = float(size[0])
            h_img = float(size[1])
        except Exception:
            return 0.0
        if w_img <= 0.0 or h_img <= 0.0:
            return 0.0
        w = max(0.0, min(w_img, x1) - max(0.0, x0))
        h = max(0.0, min(h_img, y1) - max(0.0, y0))
        area = w * h
        return float(area) if np.isfinite(area) and area > 0.0 else 0.0

    def _caption_crop_area(self, obs: object) -> float:
        if isinstance(obs, dict):
            area = self._bbox_area_from_size(obs.get("bbox_caption"), obs.get("size_caption"))
            if area > 0.0:
                return area
            area = self._bbox_area_from_size(obs.get("bbox"), obs.get("size"))
            if area > 0.0:
                return area
            area = self._bbox_area_from_size(obs.get("bbox_source"), obs.get("size_source"))
            if area > 0.0:
                return area
            with contextlib.suppress(Exception):
                area_src = float(obs.get("bbox_area_source", 0.0) or 0.0)
                if area_src > 0.0:
                    return area_src
            for image_key in ("image_caption", "image"):
                img = obs.get(image_key)
                shape = getattr(img, "shape", None)
                if shape is None:
                    continue
                with contextlib.suppress(Exception):
                    dims = [int(x) for x in tuple(shape)]
                    if len(dims) >= 2:
                        if len(dims) == 3 and dims[0] in (1, 3, 4):
                            h, w = dims[1], dims[2]
                        else:
                            h, w = dims[0], dims[1]
                        area = float(max(h, 0) * max(w, 0))
                        if area > 0.0:
                            return area
        shape = getattr(obs, "shape", None)
        if shape is not None:
            with contextlib.suppress(Exception):
                dims = [int(x) for x in tuple(shape)]
                if len(dims) >= 2:
                    if len(dims) == 3 and dims[0] in (1, 3, 4):
                        h, w = dims[1], dims[2]
                    else:
                        h, w = dims[0], dims[1]
                    area = float(max(h, 0) * max(w, 0))
                    if area > 0.0:
                        return area
        return 0.0

    def _max_caption_bbox_area_for_object(self, object_index: int) -> float:
        rgb_obs = self.scene_state.get("rgb_observations", []) or []
        if object_index < 0 or object_index >= len(rgb_obs):
            return 0.0
        obs_list = rgb_obs[object_index]
        obs_candidates = list(obs_list) if isinstance(obs_list, (list, tuple)) else [obs_list]
        best = 0.0
        for obs in obs_candidates:
            area = self._caption_crop_area(obs)
            if area > best:
                best = area
        return best

    def _prune_views_for_object(self, object_index: int, keep_indices: List[int]) -> None:
        """
        Keep only the specified view indices for rgb_observations/view_means/view_cov6.
        Used to discard unused views when not debugging.
        """
        if self.debug:
            return
        keep_set = {idx for idx in keep_indices if idx is not None and idx >= 0}
        if not keep_set:
            return
        rgb_obs = self.scene_state.get("rgb_observations", [])
        view_means = self.scene_state.get("view_means", [])
        view_cov6 = self.scene_state.get("view_cov6", [])
        if object_index >= len(rgb_obs) or object_index >= len(view_means) or object_index >= len(view_cov6):
            return

        ordered_keep = sorted(keep_set)
        rgb_list = rgb_obs[object_index]
        mean_list = view_means[object_index]
        cov_list = view_cov6[object_index]

        rgb_obs[object_index] = [rgb_list[i] for i in ordered_keep if i < len(rgb_list)]
        view_means[object_index] = [mean_list[i] for i in ordered_keep if i < len(mean_list)]
        view_cov6[object_index] = [cov_list[i] for i in ordered_keep if i < len(cov_list)]

    def _prepare_caption_inputs(self, batch: List[ObjectCaptionTask]):
        """
        Prepare caption inputs. Each `images` entry corresponds to one prompt and can be:
        - a single PIL image (initial caption), or
        - a list of PIL images (recaption with multiple views).
        `meta` stores (object_index, primary_view_index) for each prompt.
        """
        images: List = []
        meta: List = []
        boxes: List[Optional[dict]] = []
        is_recaption_flags: List[bool] = []
        crop_images_uint8: List[Any] = []
        crop_encodings: List[str] = []
        prepared_task_indices: set[int] = set()
        skipped_task_reasons: Dict[int, str] = {}
        if Image is None:
            return (
                images,
                meta,
                boxes,
                is_recaption_flags,
                crop_images_uint8,
                crop_encodings,
                prepared_task_indices,
                skipped_task_reasons,
            )

        rgb_obs = self.scene_state.get("rgb_observations", [])
        captions = self.scene_state.get("object_caption", []) or []

        for task in batch:
            if task is None:
                continue
            is_recaption = bool(getattr(task, "is_recaption", False))
            resolved_idx = self._resolve_object_index(task)
            if resolved_idx is None:
                continue
            # Initial captioning tasks skip already-captioned objects; recaption tasks do not.
            existing_caption = ""
            if resolved_idx < len(captions):
                try:
                    existing_caption = str(captions[resolved_idx] or "").strip()
                except Exception:
                    existing_caption = ""
            if existing_caption and not is_recaption:
                prepared_task_indices.add(int(task.object_index))
                skipped_task_reasons[int(task.object_id)] = "already_captioned"
                continue
            if is_recaption:
                view_indices = self._select_recaption_views_for_object(resolved_idx)
            else:
                view_indices = self._select_views_for_object(resolved_idx)
            if resolved_idx >= len(rgb_obs):
                continue

            obs_list = rgb_obs[resolved_idx]
            if is_recaption:
                if len(view_indices) < 2:
                    continue
                selected_views = [int(view_indices[0]), int(view_indices[1])]
                prompt_images: List[Image.Image] = []
                primary_view_idx: Optional[int] = None
                first_bbox = None
                first_size = None
                primary_crop_image_uint8 = None
                primary_crop_encoding = ""
                for view_idx in selected_views:
                    if view_idx < 0 or view_idx >= len(obs_list):
                        continue
                    obs = obs_list[view_idx]
                    img, bbox, size, crop_image_uint8, crop_encoding = self._extract_image_and_box(obs)
                    if img is None:
                        continue
                    prompt_images.append(img)
                    if primary_view_idx is None:
                        primary_view_idx = int(view_idx)
                        first_bbox = bbox
                        first_size = size
                        primary_crop_image_uint8 = crop_image_uint8
                        primary_crop_encoding = crop_encoding
                if len(prompt_images) < 2 or primary_view_idx is None:
                    continue
                images.append(prompt_images)
                meta.append((resolved_idx, int(primary_view_idx)))
                boxes.append({"bbox": first_bbox, "size": first_size})
                is_recaption_flags.append(True)
                crop_images_uint8.append(primary_crop_image_uint8)
                crop_encodings.append(primary_crop_encoding)
                prepared_task_indices.add(task.object_index)
            else:
                for view_idx in view_indices:
                    if view_idx < 0 or view_idx >= len(obs_list):
                        continue
                    obs = obs_list[view_idx]
                    img, bbox, size, crop_image_uint8, crop_encoding = self._extract_image_and_box(obs)
                    if img is None:
                        continue
                    images.append(img)
                    meta.append((resolved_idx, view_idx))
                    boxes.append({"bbox": bbox, "size": size})
                    is_recaption_flags.append(False)
                    crop_images_uint8.append(crop_image_uint8)
                    crop_encodings.append(crop_encoding)
                    prepared_task_indices.add(task.object_index)
        return (
            images,
            meta,
            boxes,
            is_recaption_flags,
            crop_images_uint8,
            crop_encodings,
            prepared_task_indices,
            skipped_task_reasons,
        )

    def _extract_image_and_box(self, obs) -> tuple[Optional[Image.Image], Optional[tuple], Optional[tuple], Any, str]:
        """Extract PIL image + bbox metadata + raw uint8 crop tensor from an observation entry."""
        bbox = None
        size = None
        encoding = ""
        img_data = obs
        embed_img_data = obs
        embed_encoding = ""
        if isinstance(obs, dict):
            encoding = str(obs.get("encoding", "") or obs.get("color_encoding", "") or "").strip().lower()
            if obs.get("image_caption") is not None:
                img_data = obs.get("image_caption")
                bbox = obs.get("bbox_caption", obs.get("bbox"))
                size = obs.get("size_caption", obs.get("size"))
            else:
                img_data = obs.get("image")
                bbox = obs.get("bbox")
                size = obs.get("size")
            embed_img_data = obs.get("image_embedding")
            if embed_img_data is None:
                embed_img_data = obs.get("image")
            if embed_img_data is None:
                embed_img_data = img_data
            embed_encoding = str(obs.get("embedding_encoding", "") or encoding).strip().lower()
        if img_data is None:
            return None, bbox, size, None, encoding
        crop_image_uint8 = embed_img_data
        try:
            arr = img_data.detach().cpu().numpy() if hasattr(img_data, "detach") else np.asarray(img_data)
        except Exception:
            return None, bbox, size, crop_image_uint8, embed_encoding or encoding
        if arr is None:
            return None, bbox, size, crop_image_uint8, embed_encoding or encoding
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] == 3 and encoding in {"bgr8", "bgr", "bgra8", "bgra"}:
            arr = arr[..., ::-1]
        try:
            img = Image.fromarray(arr)
        except Exception:
            return None, bbox, size, crop_image_uint8, embed_encoding or encoding
        if size is None and hasattr(arr, "shape") and len(arr.shape) >= 2:
            size = (arr.shape[1], arr.shape[0])
        return img, bbox, size, crop_image_uint8, embed_encoding or encoding

    def _maybe_retry_task(self, task: ObjectCaptionTask, reason: str) -> None:
        """Re-enqueue a task if under retry limit."""
        if bool(getattr(task, "is_recaption", False)):
            return
        if self.max_retries <= 0:
            return
        with contextlib.suppress(Exception):
            if int(task.object_id) in self._retry_pending_object_ids:
                return
        attempts = getattr(task, "attempts", 0) or 0
        if attempts >= self.max_retries:
            if self.debug:
                print(
                    f"[CaptionWorker] Dropping caption task for object_id={task.object_id} after {attempts} attempts"
                    f" ({reason})"
                )
            return
        new_task = ObjectCaptionTask(
            object_index=task.object_index,
            object_id=task.object_id,
            version=task.version,
            attempts=attempts + 1,
            is_recaption=bool(getattr(task, "is_recaption", False)),
        )
        try:
            with contextlib.suppress(Exception):
                self._retry_pending_object_ids.add(int(task.object_id))
            self.tasks_queue.put_nowait(new_task)
            if self.debug:
                print(f"[CaptionWorker] Retrying object_id={task.object_id} attempt={attempts + 1} ({reason})")
        except queue.Full:
            with contextlib.suppress(Exception):
                self._retry_pending_object_ids.discard(int(task.object_id))
            if self.debug:
                print(f"[CaptionWorker] Retry queue full; dropping object_id={task.object_id} ({reason})")


if __name__ == "__main__":
    worker = CaptionWorker(
        scene_state={},
        tasks_queue=queue.Queue(),
        results_queue=queue.Queue(),
        caption_server="sglang",
    )
    worker.warm_up_model()
    print(worker._base_context)
