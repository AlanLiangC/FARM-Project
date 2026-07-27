"""Scene graph retrieval utilities."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scene_graph.llm_utils.embed_interface import EmbedInterface
from scene_graph.runtime_paths import find_model_dir

try:  # Optional dependency for decoding caption crops from snapshot state.h5.
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:  # Optional dependency for writing decoded crops as png files.
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:  # Optional dependency for vLLM-backed Qwen3-VL query embeddings.
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

try:  # Optional dependency for Qwen3-VL query embeddings.
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:  # Optional dependency for Qwen3-VL query embeddings.
    from transformers import AutoModel, AutoProcessor
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoProcessor = None


_DEFAULT_QWEN3_EMBED_TASK = (
    "Embed for object identity and affordance retrieval. "
    "Use the object's category, supercategory, and visible attributes; ignore viewpoint and wording."
)
_DEFAULT_QWEN3_VL_EMBED_CKPT = "Qwen/Qwen3-VL-Embedding-2B"
_DEFAULT_QWEN3_VL_EMBED_MODEL = "qwen3-vl-emb-2b"
_DEFAULT_QWEN3_VL_EMBED_BASE_URL = "http://localhost:8006/v1"
_DEFAULT_QWEN3_VL_EMBED_QUERY_PROMPT = (
    "Retrieve images of an object that matches this description. "
    "Focus on the object's category and visible attributes."
)
_DEFAULT_QWEN3_VL_RERANK_CKPT = "Qwen/Qwen3-VL-Reranker-2B"
_DEFAULT_QWEN3_VL_RERANK_INSTRUCTION = "Retrieve images or text relevant to the user's query."
_DEFAULT_SIGLIP2_LOCAL_DIRNAME = "siglip2-large-patch16-256"
_DEFAULT_SIGLIP2_HF_CKPT = "google/siglip2-large-patch16-256"
_DEFAULT_TASK_IMAGE_QUERY_PROMPT = (
    "Retrieve images containing physical objects that a person would need to complete this task.\n\nTask: {query}"
)
_DEFAULT_TASK_CAPTION_QUERY_PROMPT = (
    "Retrieve object caption related to physical objects that a person would need to complete this task.\n\n"
    "Task: {query}"
)
_YOLOE_EMBED_PREFIX = (
    "Instruct: Represent this for matching by category and affordance (typical use). "
    "Ignore color, size, brand, and viewpoint."
)


def _resolve_default_siglip2_ckpt() -> str:
    local_path = find_model_dir(_DEFAULT_SIGLIP2_LOCAL_DIRNAME)
    if local_path is not None:
        return str(local_path)
    return _DEFAULT_SIGLIP2_HF_CKPT


class _Qwen3QueryWrapper:
    def __init__(self, embedder: Any, *, task: Optional[str] = None) -> None:
        self._embedder = embedder
        self._task = str(task) if task is not None else (os.getenv("QWEN3_EMBED_TASK") or _DEFAULT_QWEN3_EMBED_TASK)

    @staticmethod
    def _canonicalize(text: str) -> str:
        return (text or "").strip().lower()

    def encode(self, text: str) -> np.ndarray:
        fn = getattr(self._embedder, "encode_query", None)
        if callable(fn):
            return np.asarray(fn(text, task=self._task), dtype=np.float32)

        canon = self._canonicalize(str(text or ""))
        wrapped = f"Instruct: {self._task}\nQuery: {canon}" if canon else ""
        return np.asarray(self._embedder.encode(wrapped), dtype=np.float32)


class _Qwen3VLEmbedQueryWrapper:
    def __init__(
        self,
        *,
        ckpt: Optional[str] = None,
        device: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self._ckpt = str(ckpt or os.getenv("QWEN3_VL_EMBED_CKPT") or _DEFAULT_QWEN3_VL_EMBED_CKPT)
        self._device_hint = str(device or os.getenv("QWEN3_VL_EMBED_DEVICE") or "cuda")
        self._backend = str(os.getenv("QWEN3_VL_EMBED_BACKEND", "vllm")).strip().lower()
        if self._backend not in {"vllm", "hf", "auto"}:
            self._backend = "vllm"
        self._vllm_model_name = str(os.getenv("VLLM_QWEN3_VL_EMBED_MODEL", _DEFAULT_QWEN3_VL_EMBED_MODEL))
        self._vllm_base_url = str(os.getenv("VLLM_QWEN3_VL_EMBED_BASE_URL", _DEFAULT_QWEN3_VL_EMBED_BASE_URL))
        self._vllm_timeout_s = max(1.0, float(os.getenv("VLLM_QWEN3_VL_EMBED_TIMEOUT_S", "30")))
        self._vllm_api_key = os.getenv("VLLM_QWEN3_VL_EMBED_API_KEY") or os.getenv("VLLM_API_KEY")
        self._vllm_max_retries = max(1, int(os.getenv("VLLM_QWEN3_VL_EMBED_MAX_RETRIES", "3")))
        self._query_prompt = str(os.getenv("QWEN3_VL_EMBED_QUERY_PROMPT", _DEFAULT_QWEN3_VL_EMBED_QUERY_PROMPT))
        self._vllm_server_ok: Optional[bool] = None
        self._debug = bool(debug)
        self._model = None
        self._processor = None
        self._init_failed = False
        self._lock = threading.Lock()

    @staticmethod
    def _canonicalize(text: str) -> str:
        return (text or "").strip().lower()

    def _build_hf(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._init_failed:
            raise RuntimeError("Qwen3-VL embedding model unavailable (previous init failed)")
        if torch is None or AutoModel is None or AutoProcessor is None:
            self._init_failed = True
            raise RuntimeError("Qwen3-VL embedding requires torch+transformers")

        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            if self._init_failed:
                raise RuntimeError("Qwen3-VL embedding model unavailable (previous init failed)")
            try:
                device = self._device_hint
                if device.startswith("cuda") and not torch.cuda.is_available():
                    device = "cpu"
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                try:
                    model = AutoModel.from_pretrained(
                        self._ckpt,
                        dtype=dtype,
                        trust_remote_code=True,
                    )
                except TypeError:
                    try:
                        model = AutoModel.from_pretrained(
                            self._ckpt,
                            torch_dtype=dtype,
                            trust_remote_code=True,
                        )
                    except TypeError:
                        model = AutoModel.from_pretrained(self._ckpt, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(self._ckpt, trust_remote_code=True)
                model = model.to(device).eval()
                self._model = model
                self._processor = processor
                if self._debug:
                    print(
                        "[SceneGraphRetriever] Qwen3-VL query embedder ready"
                        f" backend='hf' ckpt={self._ckpt!r} device={device!r}"
                    )
            except Exception as exc:
                self._init_failed = True
                raise RuntimeError(f"Qwen3-VL embedding init failed: {exc}") from exc

    def _vllm_qwen3_vl_embed_url(self) -> str:
        base = self._vllm_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def _probe_qwen3_vl_embed_server(self) -> bool:
        if requests is None:
            return False
        url = self._vllm_base_url.rstrip("/")
        headers = {}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        try:
            resp = requests.get(f"{url}/models", headers=headers, timeout=min(self._vllm_timeout_s, 10))
            if resp.status_code == 200:
                with contextlib.suppress(Exception):
                    payload = resp.json()
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(data, list):
                        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                        if ids and self._vllm_model_name not in ids:
                            if self._debug:
                                print(
                                    "[SceneGraphRetriever] vLLM qwen3-vl embed server reachable but model id not"
                                    f" served: requested={self._vllm_model_name!r} available={ids!r}"
                                )
                            return False
                return True
            resp = requests.get(f"{url}/health", headers=headers, timeout=min(self._vllm_timeout_s, 10))
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _parse_vllm_embed_response(payload: Any) -> np.ndarray:
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("invalid embeddings payload shape")
        row0 = rows[0] if isinstance(rows[0], dict) else {}
        vec = row0.get("embedding") if isinstance(row0, dict) else None
        arr = np.asarray(vec if isinstance(vec, list) else [], dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            raise ValueError("missing embedding vector in response")
        denom = float(np.linalg.norm(arr) + 1e-12)
        return (arr / denom).astype(np.float32, copy=False)

    def _encode_vllm(self, query: str) -> np.ndarray:
        if requests is None:
            raise RuntimeError("Qwen3-VL embedding vLLM backend requires requests")

        if self._vllm_server_ok is None:
            self._vllm_server_ok = self._probe_qwen3_vl_embed_server()
        if not self._vllm_server_ok:
            raise RuntimeError(
                "Qwen3-VL embedding vLLM server unavailable or model not served "
                f"(model={self._vllm_model_name!r} base_url={self._vllm_base_url!r})"
            )

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        url = self._vllm_qwen3_vl_embed_url()

        prompt = f"{self._query_prompt}\nQuery: {query}"
        messages_payload = {
            "model": self._vllm_model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "encoding_format": "float",
        }
        input_payload = {
            "model": self._vllm_model_name,
            "input": [prompt],
            "encoding_format": "float",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self._vllm_max_retries):
            for payload in (messages_payload, input_payload):
                try:
                    resp = requests.post(url, json=payload, headers=headers, timeout=self._vllm_timeout_s)
                    if not resp.ok:
                        body = ""
                        with contextlib.suppress(Exception):
                            body = (resp.text or "")[:300]
                        raise RuntimeError(f"status={resp.status_code} body={body!r}")
                    return self._parse_vllm_embed_response(resp.json())
                except Exception as exc:
                    last_exc = exc
            if attempt < self._vllm_max_retries - 1:
                with contextlib.suppress(Exception):
                    time.sleep(0.1)
        raise RuntimeError(f"Qwen3-VL embedding vLLM request failed: {last_exc!r}")

    def _encode_hf(self, query: str) -> np.ndarray:
        self._build_hf()
        model = self._model
        processor = self._processor
        if model is None or processor is None or torch is None:
            raise RuntimeError("Qwen3-VL query embedder unavailable")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            prompt = f"{self._query_prompt}\nQuery: {query}"
            text_inputs = processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {}
            for key, value in dict(text_inputs).items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device=device, non_blocking=True)

            def _extract_text_features(obj: Any) -> Any:
                if torch.is_tensor(obj):
                    return obj
                if isinstance(obj, (list, tuple)) and obj:
                    return _extract_text_features(obj[0])
                for attr in ("text_embeds", "pooler_output"):
                    candidate = getattr(obj, attr, None)
                    if torch.is_tensor(candidate):
                        return candidate
                lhs = getattr(obj, "last_hidden_state", None)
                if torch.is_tensor(lhs) and lhs.ndim >= 2:
                    return lhs[:, 0, :]
                return None

            def _run_text_only(callable_obj: Any) -> Any:
                with contextlib.suppress(TypeError):
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            return callable_obj(**inputs)
                    return callable_obj(**inputs)
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        return callable_obj(inputs)
                return callable_obj(inputs)

            text_features = None
            for entry in ("get_text_features", "encode_text", "text_model"):
                if not hasattr(model, entry):
                    continue
                with contextlib.suppress(Exception):
                    text_features = _extract_text_features(_run_text_only(getattr(model, entry)))
                if text_features is not None:
                    break
            if text_features is None:
                out = _run_text_only(lambda **kw: model(**kw, return_dict=True))
                text_features = _extract_text_features(out)

            if not torch.is_tensor(text_features):
                raise RuntimeError("Qwen3-VL query text features unavailable")
            if text_features.ndim == 3:
                text_features = text_features[:, 0, :]

            text_features = text_features.to(dtype=torch.float32)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1, eps=1e-12)
            return text_features[0].detach().to("cpu", copy=False).numpy().astype(np.float32, copy=False)

    def encode(self, text: str) -> np.ndarray:
        query = self._canonicalize(str(text or ""))
        if not query:
            raise ValueError("query text cannot be empty")

        if self._backend == "vllm":
            return self._encode_vllm(query)
        if self._backend == "hf":
            return self._encode_hf(query)

        # auto: prefer vLLM, fallback to local HF.
        try:
            return self._encode_vllm(query)
        except Exception as exc:
            if self._debug:
                print(f"[SceneGraphRetriever] Qwen3-VL vLLM query embedding failed, fallback to HF: {exc!r}")
            return self._encode_hf(query)


class _Siglip2TextEmbedWrapper:
    def __init__(
        self,
        *,
        ckpt: Optional[str] = None,
        device: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        if torch is None or AutoModel is None or AutoProcessor is None:
            raise RuntimeError("SigLIP2 local embedding requires torch+transformers")
        self._ckpt = str(
            ckpt or os.getenv("LAM_SIGLIP2_CKPT") or os.getenv("SIGLIP2_CKPT") or _resolve_default_siglip2_ckpt()
        )
        self._device_hint = str(device or os.getenv("LAM_SIGLIP2_DEVICE") or os.getenv("SIGLIP2_DEVICE") or "cuda")
        self._debug = bool(debug)
        self._model = None
        self._processor = None
        self._init_failed = False
        self._lock = threading.Lock()

    def _build(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if self._init_failed:
            raise RuntimeError("SigLIP2 local embedding unavailable (previous init failed)")

        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            if self._init_failed:
                raise RuntimeError("SigLIP2 local embedding unavailable (previous init failed)")
            try:
                device = self._device_hint
                if device.startswith("cuda") and not torch.cuda.is_available():
                    device = "cpu"
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                attn_impl = "sdpa"
                try:
                    model = AutoModel.from_pretrained(
                        self._ckpt,
                        dtype=dtype,
                        attn_implementation=attn_impl,
                    )
                except TypeError:
                    try:
                        model = AutoModel.from_pretrained(
                            self._ckpt,
                            torch_dtype=dtype,
                            attn_implementation=attn_impl,
                        )
                    except TypeError:
                        try:
                            model = AutoModel.from_pretrained(self._ckpt, dtype=dtype)
                        except TypeError:
                            model = AutoModel.from_pretrained(self._ckpt, torch_dtype=dtype)

                model = model.to(device).eval()
                processor = AutoProcessor.from_pretrained(self._ckpt, use_fast=True)
                self._model = model
                self._processor = processor
                if self._debug:
                    print(f"[SceneGraphRetriever] SigLIP2 query embedder ready ckpt={self._ckpt!r} device={device!r}")
            except Exception as exc:
                self._init_failed = True
                raise RuntimeError(f"SigLIP2 local embedding init failed: {exc}") from exc

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        text_list = [str(x or "").strip() for x in texts]
        if not text_list:
            return np.empty((0, 0), dtype=np.float32)
        if not all(text_list):
            raise ValueError("siglip2 query text cannot be empty")

        self._build()
        model = self._model
        processor = self._processor
        if model is None or processor is None:
            raise RuntimeError("SigLIP2 local embedding model unavailable")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with torch.inference_mode():
            text_inputs = processor(
                text=text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs: Dict[str, Any] = {}
            for key, value in dict(text_inputs).items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device=device, non_blocking=True)

            def _extract_text_features(obj: Any) -> Any:
                if torch.is_tensor(obj):
                    return obj
                if isinstance(obj, (list, tuple)) and obj:
                    return _extract_text_features(obj[0])
                # transformers BaseModelOutputWithPooling and friends:
                # try the most-aligned attributes first.
                for attr in ("text_embeds", "pooler_output"):
                    candidate = getattr(obj, attr, None)
                    if torch.is_tensor(candidate):
                        return candidate
                lhs = getattr(obj, "last_hidden_state", None)
                if torch.is_tensor(lhs) and lhs.ndim >= 2:
                    return lhs[:, 0, :]
                return None

            def _run_text_only(callable_obj: Any) -> Any:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        return callable_obj(**inputs)
                return callable_obj(**inputs)

            text_features: Any = None
            # 1) Preferred: model.get_text_features (modern transformers may return
            #    a BaseModelOutputWithPooling rather than a bare tensor).
            if hasattr(model, "get_text_features"):
                with contextlib.suppress(Exception):
                    text_features = _extract_text_features(_run_text_only(model.get_text_features))
            # 2) Fallback: text submodule directly (canonical path, returns a
            #    BaseModelOutputWithPooling whose pooler_output is the text vec).
            if text_features is None and hasattr(model, "text_model"):
                with contextlib.suppress(Exception):
                    text_features = _extract_text_features(_run_text_only(model.text_model))
            # 3) Last resort: full forward, then extract.
            if text_features is None:
                out = _run_text_only(lambda **kw: model(**kw, return_dict=True))
                text_features = _extract_text_features(out)

            if not torch.is_tensor(text_features):
                raise RuntimeError("SigLIP2 text features unavailable")
            if text_features.ndim == 3:
                text_features = text_features[:, 0, :]
            text_features = text_features.to(dtype=torch.float32)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1, eps=1e-12)
            return text_features.detach().to("cpu", copy=False).numpy().astype(np.float32, copy=False)

    def encode(self, text: str) -> np.ndarray:
        vectors = self.encode_batch([text])
        if vectors.ndim != 2 or vectors.shape[0] <= 0:
            raise RuntimeError("SigLIP2 local embedding returned empty vector")
        return vectors[0]


class SceneGraphRetriever:
    """
    Retrieval utility backed by mapping.scene_graph_processing.SceneGraphProcessing.

    Pipeline:
    1) Ensure latest scene graph is loaded
    2) Rebuild covisibility graph only when a new scene graph is loaded
    3) retrieve_by_caption(query, retrieval_k=50, top_k=10)
    4) retrieve_by_siglip2(query, top_k=10)
    5) retrieve_by_qwen3_vl(query, top_k=10)
    6) Union + cluster candidates by covisibility/distance
    7) Return clusters with candidates, neighbors, and minimum covering images
    """

    def __init__(
        self,
        scene_graph_json_path: str,
        *,
        embedder: EmbedInterface,
        snapshot_root: Optional[str] = None,
        image_store_root: Optional[str] = None,
        siglip2_embedder: Optional[Any] = None,
        siglip2_ckpt: Optional[str] = None,
        siglip2_device: Optional[str] = None,
        siglip2_local_enabled: Optional[bool] = None,
        siglip2_local_fallback_to_service: bool = False,
        qwen3_vl_embedder: Optional[Any] = None,
        use_siglip2_service: bool = False,
        qwen3_vl_k: int = 10,
        candidate_cap: int = 20,
        cluster_distance_threshold_m: float = 5.0,
        cluster_vl_rerank_enabled: Optional[bool] = None,
        cluster_vl_rerank_ckpt: Optional[str] = None,
        cluster_vl_rerank_instruction: Optional[str] = None,
        cluster_vl_rerank_batch_size: Optional[int] = None,
        cluster_vl_rerank_max_length: Optional[int] = None,
        cluster_vl_doc_max_images: Optional[int] = None,
        cluster_vl_doc_max_captions: Optional[int] = None,
        cluster_vl_doc_max_chars: Optional[int] = None,
        adaptation_results_dir: Optional[str] = None,
        adaptation_similarity_threshold: float = 0.7,
        adaptation_top_k_per_class: int = 10,
        verbose: bool = False,
        covisibility_kwargs: Optional[Dict[str, Any]] = None,
        debug_dir: str | Path | None = None,
        processor: Optional[Any] = None,
        region_boost_enabled: bool = False,
        region_boost_factor: float = 1.3,
    ) -> None:
        raw_scene_graph_json_path = str(scene_graph_json_path or "").strip()
        if not raw_scene_graph_json_path and processor is None:
            raise ValueError("scene_graph_json_path must be non-empty (or pass processor=…)")
        self.verbose = bool(verbose)
        # When ``processor`` is supplied (in-memory entry point), the JSON path is
        # optional — it's only used as a sentinel for cache-invalidation logging.
        self.scene_graph_json_path: Optional[Path] = (
            self._resolve_scene_graph_json_path(raw_scene_graph_json_path)
            if raw_scene_graph_json_path
            else None
        )
        self.snapshot_root = Path(snapshot_root).expanduser() if snapshot_root else None
        self.image_store_root = Path(image_store_root).expanduser() if image_store_root else None
        self.embedder = embedder
        self.siglip2_embedder = siglip2_embedder
        if siglip2_local_enabled is None:
            siglip2_local_enabled = str(os.getenv("LAM_SIGLIP2_LOCAL_ENABLED", "1")).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        self._siglip2_local_enabled = bool(siglip2_local_enabled)
        self._siglip2_ckpt = str(
            siglip2_ckpt
            or os.getenv("LAM_SIGLIP2_CKPT")
            or os.getenv("SIGLIP2_CKPT")
            or _resolve_default_siglip2_ckpt()
        )
        self._siglip2_device = str(
            siglip2_device or os.getenv("LAM_SIGLIP2_DEVICE") or os.getenv("SIGLIP2_DEVICE") or "cuda"
        )
        self._siglip2_local_failed = False
        # SigLIP2 runs strictly local in this retriever path; ROS2 service fallback is disabled.
        self._siglip2_local_fallback_to_service = False
        self.qwen3_vl_embedder = qwen3_vl_embedder
        self.use_siglip2_service = False
        if self.verbose and bool(use_siglip2_service):
            print("[SceneGraphRetriever] Ignoring use_siglip2_service=True (SigLIP2 is local-only)")
        if self.verbose and bool(siglip2_local_fallback_to_service):
            print("[SceneGraphRetriever] Ignoring siglip2_local_fallback_to_service=True (SigLIP2 is local-only)")
        self.qwen3_vl_k = max(1, int(qwen3_vl_k))
        self.candidate_cap = int(max(1, candidate_cap)) if candidate_cap is not None else None
        self.cluster_distance_threshold_m = float(cluster_distance_threshold_m)
        self.covisibility_kwargs = dict(covisibility_kwargs or {})
        self.debug_dir = Path(debug_dir).expanduser() if debug_dir else None
        # NOTE: The Qwen3-VL reranker step is intentionally disabled.
        # It added latency/complexity without improving downstream behavior in practice.
        #
        # Keep the constructor args for API compatibility, but fully skip building/using
        # the VL reranker regardless of env vars or passed flags.
        self._cluster_vl_rerank_enabled = False
        self._cluster_vl_rerank_ckpt = str(
            cluster_vl_rerank_ckpt or os.getenv("QWEN3_VL_RERANK_CKPT") or _DEFAULT_QWEN3_VL_RERANK_CKPT
        )
        self._cluster_vl_rerank_instruction = str(
            cluster_vl_rerank_instruction
            or os.getenv("QWEN3_VL_RERANK_INSTRUCTION")
            or _DEFAULT_QWEN3_VL_RERANK_INSTRUCTION
        )
        self._cluster_vl_rerank_batch_size = max(
            1, int(cluster_vl_rerank_batch_size or os.getenv("QWEN3_VL_CLUSTER_RERANK_BATCH_SIZE", "4"))
        )
        self._cluster_vl_rerank_max_length = max(
            128, int(cluster_vl_rerank_max_length or os.getenv("QWEN3_VL_CLUSTER_RERANK_MAX_LENGTH", "8192"))
        )
        self._cluster_vl_doc_max_images = max(
            1, int(cluster_vl_doc_max_images or os.getenv("QWEN3_VL_CLUSTER_DOC_MAX_IMAGES", "8"))
        )
        self._cluster_vl_doc_max_captions = max(
            1, int(cluster_vl_doc_max_captions or os.getenv("QWEN3_VL_CLUSTER_DOC_MAX_CAPTIONS", "8"))
        )
        self._cluster_vl_doc_max_chars = max(
            64, int(cluster_vl_doc_max_chars or os.getenv("QWEN3_VL_CLUSTER_DOC_MAX_CHARS", "1200"))
        )
        self._cluster_vl_reranker: Optional[Any] = None
        adaptation_dir_raw = (
            str(adaptation_results_dir).strip()
            if adaptation_results_dir is not None
            else str(os.getenv("YOLOE_ADAPTATION_RESULTS_DIR", "log/snapshots/yoloe_adaptation")).strip()
        )
        self._adaptation_results_dir = Path(adaptation_dir_raw).expanduser() if adaptation_dir_raw else None
        self._adaptation_similarity_threshold = float(max(0.0, adaptation_similarity_threshold))
        self._adaptation_top_k_per_class = int(max(1, adaptation_top_k_per_class))
        self._adaptation_manifest_sig: Optional[Tuple[str, int, int]] = None
        self._adaptation_records_cache: List[Dict[str, Any]] = []
        self._adaptation_class_embed_cache: Dict[str, np.ndarray] = {}
        self._adaptation_class_list_cache: List[str] = []
        self._adaptation_class_embed_sig: Optional[Tuple[str, int, int]] = None
        self._rerank_query_prefix = (
            os.getenv("QWEN3_RERANK_QUERY_PREFIX")
            or "Match by category + key attributes + typical use. Ignore viewpoint/wording."
        )
        try:
            self._rerank_doc_max_chars = max(64, int(os.getenv("QWEN3_RERANK_DOC_MAX_CHARS", "400")))
        except Exception:
            self._rerank_doc_max_chars = 400

        # ``_processor_externally_provided`` short-circuits _ensure_latest_scene_graph's
        # JSON-mtime / state.h5 reload detection when the caller used the in-memory
        # entry point (``from_scene_state`` / ``processor=…``). The first call still
        # needs to build the covisibility graph; subsequent calls become a no-op.
        self._processor_externally_provided: bool = processor is not None
        self._processor: Optional[Any] = processor
        self._lock = threading.Lock()
        self._loaded_scene_graph_sig: Optional[Tuple[int, int]] = None
        self._loaded_state_sig: Optional[Tuple[str, int, int]] = None
        self._crop_cache: Dict[str, Dict[str, Any]] = {}
        self._qwen3_vl_embed_enabled: bool = str(os.getenv("QWEN3_VL_EMBED_ENABLED", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        self._region_boost_enabled: bool = bool(region_boost_enabled)
        self._region_boost_factor: float = float(region_boost_factor)

        # Best-effort warmup: precompute YOLOE class embeddings so per-query we only
        # embed the query once and do fast dot-products over cached class vectors.
        # Never crash init for this; it is an optimization.
        with contextlib.suppress(Exception):
            self._prime_yoloe_class_embedding_cache(_Qwen3QueryWrapper(self.embedder))

    @classmethod
    def from_scene_state(
        cls,
        state: "Dict[str, Any] | str | Path",
        *,
        embedder: EmbedInterface,
        feature_dim: Optional[int] = None,
        snapshot_root: Optional[str] = None,
        image_store_root: Optional[str] = None,
        verbose: bool = False,
        min_obs_for_retrieval: int = 1,
        **retriever_kwargs: Any,
    ) -> "SceneGraphRetriever":
        """Build a retriever directly from an offline-saved SceneState.

        Args:
            state: An in-memory SceneState dict or a path to a saved ``.pt`` file
                produced by :func:`scene_graph.scene_state_io.save_scene_state`.
            embedder: vLLM-backed text embedder (``scene_graph.llm_utils.EmbedInterface``).
            feature_dim: Required when ``state`` is a path and the .pt header
                does not record ``feature_dim`` (rare).
            snapshot_root, image_store_root: Optional roots used by the
                retriever for resolving image storage paths.
            verbose: Forwarded to both processor and retriever.
            **retriever_kwargs: Forwarded to :class:`SceneGraphRetriever`'s
                constructor (e.g. ``cluster_distance_threshold_m``,
                ``candidate_cap``, ``covisibility_kwargs``).

        Bypasses the JSON-manifest / state.h5 disk round-trip the live ROS path
        uses, so the offline pipeline's saved scene_state.pt feeds the
        retriever directly.
        """
        # Lazy import to keep scene_graph.scene_graph optional at module load time.
        from scene_graph.scene_graph import SceneGraphProcessing

        processor = SceneGraphProcessing.from_scene_state(
            state,
            snapshot_root=snapshot_root,
            feature_dim=feature_dim,
            min_obs_for_retrieval=min_obs_for_retrieval,
        )
        return cls(
            scene_graph_json_path="",  # ignored when processor= is given
            embedder=embedder,
            snapshot_root=snapshot_root,
            image_store_root=image_store_root,
            verbose=verbose,
            processor=processor,
            **retriever_kwargs,
        )

    @staticmethod
    def _object_key(object_id: Any) -> str:
        return str(object_id)

    @staticmethod
    def _resolve_scene_graph_json_path(raw_path: str) -> Path:
        path = Path(str(raw_path).strip()).expanduser()
        if path.is_dir():
            direct_candidates = [
                path / "scene_graph.json",
                path / "scene_graph_latest.json",
                path / "mapping" / "scene_graph.json",
            ]
            for candidate in direct_candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate

            discovered = [p for p in path.rglob("scene_graph*.json") if p.is_file()]
            if discovered:
                discovered.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
                return discovered[0]

            return path / "scene_graph.json"

        return path

    @staticmethod
    def _file_signature(path: Path) -> Optional[Tuple[int, int]]:
        if not path.exists():
            return None
        stat = path.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    @staticmethod
    def _normalize_reranker_score(score: Any) -> float:
        try:
            value = float(score)
        except Exception:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if 0.0 <= value <= 1.0:
            return value
        if -1.0 <= value <= 1.0:
            return 0.5 * (value + 1.0)
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _clip_siglip_score(score: Any) -> float:
        try:
            value = float(score)
        except Exception:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _reciprocal_rank_score(rank: Optional[int], *, offset: int = 5) -> float:
        try:
            rank_i = int(rank) if rank is not None else 0
        except Exception:
            return 0.0
        if rank_i <= 0:
            return 0.0
        return 1.0 / float(int(offset) + rank_i)

    _REGION_PREPOSITIONS = (
        "in the ", "from the ", "inside the ", "within the ",
        "of the ", "in a ", "from a ", "inside a ",
    )

    @classmethod
    def _detect_query_region(cls, query: str, region_labels: List[str]) -> Optional[int]:
        """Return the region index only if the query explicitly references a room.

        Requires a spatial preposition immediately before the room name to avoid
        false positives like 'office' in 'office chair' or 'bed' in 'bedroom'.
        Matches patterns like "in the kitchen", "from the bathroom".
        """
        query_lower = query.lower()
        best_idx: Optional[int] = None
        best_len = 0
        for idx, label in enumerate(region_labels):
            label_lower = label.lower()
            for prep in cls._REGION_PREPOSITIONS:
                phrase = prep + label_lower
                if phrase in query_lower and len(label_lower) > best_len:
                    best_len = len(label_lower)
                    best_idx = idx
                    break
        return best_idx

    @staticmethod
    def _import_scene_graph_processing() -> Any:
        module = importlib.import_module("mapping.scene_graph_processing")
        loader = getattr(module, "load_scene_graph_processing", None)
        if loader is None:
            raise RuntimeError("mapping.scene_graph_processing.load_scene_graph_processing not found")
        return loader

    def _get_qwen3_vl_query_embedder(self) -> Optional[Any]:
        if self.qwen3_vl_embedder is not None:
            return self.qwen3_vl_embedder
        if not self._qwen3_vl_embed_enabled:
            return None
        self.qwen3_vl_embedder = _Qwen3VLEmbedQueryWrapper(debug=self.verbose)
        return self.qwen3_vl_embedder

    def _get_siglip2_query_embedder(self) -> Optional[Any]:
        if self.siglip2_embedder is not None:
            return self.siglip2_embedder
        if not self._siglip2_local_enabled or self._siglip2_local_failed:
            return None
        try:
            self.siglip2_embedder = _Siglip2TextEmbedWrapper(
                ckpt=self._siglip2_ckpt,
                device=self._siglip2_device,
                debug=self.verbose,
            )
        except Exception as exc:
            self._siglip2_local_failed = True
            if self.verbose:
                print(f"[SceneGraphRetriever] Local SigLIP2 embedder unavailable: {exc}")
            return None
        return self.siglip2_embedder

    def _get_cluster_vl_reranker(self) -> Optional[Any]:
        # Fully disabled: keep the implementation around, but do not build the model.
        return None

    def _state_signature(self, processor: Any) -> Optional[Tuple[str, int, int]]:
        state_path = getattr(processor, "snapshot_state_path", None)
        if state_path is None:
            return None
        path = Path(state_path)
        if not path.exists():
            return None
        stat = path.stat()
        return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)

    def _snapshot_hint_path(self) -> Optional[str]:
        """
        Best-effort state path hint from the scene-graph JSON payload.
        Used to detect when the latest snapshot changed even before reloading.
        """
        if self.scene_graph_json_path is None:
            return None
        try:
            payload = json.loads(self.scene_graph_json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return None
        raw = snapshot.get("state_path")
        if not raw:
            return None
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = self.scene_graph_json_path.parent / candidate
        if self.snapshot_root is not None and not candidate.exists():
            candidate_alt = self.snapshot_root / Path(str(raw)).expanduser()
            if candidate_alt.exists():
                candidate = candidate_alt
        return str(candidate.resolve()) if candidate.exists() else str(candidate)

    def _crop_output_dir(self) -> Path:
        sig_raw = str(self._loaded_state_sig or self.scene_graph_json_path)
        digest = hashlib.sha1(sig_raw.encode("utf-8")).hexdigest()[:12]
        out_dir = Path("/tmp") / "scene_graph_retrieval_crops" / digest
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _safe_token(value: Any) -> str:
        token = str(value or "").strip()
        token = re.sub(r"\s+", "_", token)
        token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
        return token[:120] or "query"

    def _adaptation_manifest_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self._adaptation_results_dir is not None:
            candidates.append(self._adaptation_results_dir)
        if self.snapshot_root is not None:
            candidates.append(self.snapshot_root / "yoloe_adaptation")
            with contextlib.suppress(Exception):
                candidates.append(self.snapshot_root.parent / "snapshots" / "yoloe_adaptation")

        # Common local workspace layouts.
        cwd = Path.cwd()
        candidates.extend([
            cwd / "models" / "snapshots" / "yoloe_adaptation",
            cwd / "log" / "snapshots" / "yoloe_adaptation",
        ])

        # Also try each known media-resolution base to make path handling robust when cwd differs.
        for base in self._media_resolution_bases():
            candidates.extend([
                base / "models" / "snapshots" / "yoloe_adaptation",
                base / "log" / "snapshots" / "yoloe_adaptation",
                base / "snapshots" / "yoloe_adaptation",
                base / "yoloe_adaptation",
            ])

        uniq_roots: List[Path] = []
        seen: set[str] = set()
        for root in candidates:
            if root is None:
                continue
            with contextlib.suppress(Exception):
                root = root.expanduser()
            key = str(root)
            with contextlib.suppress(Exception):
                key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            uniq_roots.append(root)

        for root in uniq_roots:
            manifest = root / "objects.json"
            if manifest.exists() and manifest.is_file():
                return manifest

        if not uniq_roots:
            return None
        return uniq_roots[0] / "objects.json"

    @staticmethod
    def _normalize_unit_vector(vec: Any) -> Optional[np.ndarray]:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 0.0 or not np.isfinite(norm):
            return None
        return (arr / norm).astype(np.float32, copy=False)

    @staticmethod
    def _dot_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        if int(a.size) == 0 or int(b.size) == 0 or int(a.size) != int(b.size):
            return 0.0
        return float(np.dot(a, b))

    @staticmethod
    def _yoloe_embed_text(text: str) -> str:
        """Wrap text with shared embedding template for YOLOE class matching.

        This template instructs the embedder to focus on category + affordance
        and ignore visual properties like color, size, brand, viewpoint.
        Use consistently for both class labels and queries.
        """
        t = str(text or "").strip()
        return f"{_YOLOE_EMBED_PREFIX}\nTEXT: {t}"

    def _prime_yoloe_class_embedding_cache(self, embedder: Any) -> None:
        """
        Precompute (in-memory) embeddings for every unique YOLOE class label
        in the adaptation manifest.
        """
        records = self._load_adaptation_records()
        if not records:
            return

        sig = self._adaptation_manifest_sig
        if sig is None:
            return

        # Refresh cache only if the manifest changed.
        if self._adaptation_class_embed_sig != sig:
            self._adaptation_class_embed_cache = {}
            self._adaptation_class_list_cache = []
            self._adaptation_class_embed_sig = sig

        classes = sorted(
            {str(r.get("class_name", "") or "").strip() for r in records if str(r.get("class_name", "") or "").strip()}
        )
        if not classes:
            return

        for cls in classes:
            if cls in self._adaptation_class_embed_cache:
                continue
            vec = np.asarray(embedder.encode(self._yoloe_embed_text(cls)), dtype=np.float32).reshape(-1)
            unit = self._normalize_unit_vector(vec)
            if unit is not None:
                self._adaptation_class_embed_cache[cls] = unit

        self._adaptation_class_list_cache = [c for c in classes if c in self._adaptation_class_embed_cache]

    def _yoloe_matching_classes(self, query_text: str, *, embedder: Any) -> List[Tuple[str, float]]:
        """
        Returns [(class_name, cosine_similarity)] for all YOLOE classes whose
        cosine similarity to the query >= threshold, sorted by similarity desc.
        """
        q = str(query_text or "").strip()
        if not q:
            return []

        if not self._adaptation_class_list_cache:
            self._prime_yoloe_class_embedding_cache(embedder)
        if not self._adaptation_class_list_cache:
            return []

        q_vec = np.asarray(embedder.encode(self._yoloe_embed_text(q)), dtype=np.float32).reshape(-1)
        q_unit = self._normalize_unit_vector(q_vec)
        if q_unit is None:
            return []

        thr = float(self._adaptation_similarity_threshold)
        out: List[Tuple[str, float]] = []
        for cls in self._adaptation_class_list_cache:
            sim = self._dot_similarity(q_unit, self._adaptation_class_embed_cache.get(cls))
            if sim >= thr:
                out.append((cls, float(sim)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _load_adaptation_records(self) -> List[Dict[str, Any]]:
        manifest_path = self._adaptation_manifest_path()
        if manifest_path is None:
            return []
        file_sig = self._file_signature(manifest_path)
        if file_sig is None:
            self._adaptation_manifest_sig = None
            self._adaptation_records_cache = []
            return []
        with contextlib.suppress(Exception):
            manifest_path = manifest_path.resolve()
        sig = (str(manifest_path), int(file_sig[0]), int(file_sig[1]))
        if self._adaptation_manifest_sig == sig:
            return list(self._adaptation_records_cache)

        records: List[Dict[str, Any]] = []
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            self._adaptation_manifest_sig = sig
            self._adaptation_records_cache = []
            return []

        rows: List[Any] = []
        if isinstance(payload, dict):
            rows = list(payload.get("objects", []) or [])
        elif isinstance(payload, list):
            rows = list(payload)

        for row in rows:
            if not isinstance(row, dict):
                continue
            detection_id = row.get("detection_id")
            with contextlib.suppress(Exception):
                detection_id = int(detection_id)
            class_name = str(row.get("class_name", row.get("label", "")) or "").strip()
            if detection_id is None or not class_name:
                continue
            object_id = str(row.get("object_id", f"yoloe_{int(detection_id)}") or f"yoloe_{int(detection_id)}")
            conf = row.get("confidence_mean", 0.0)
            with contextlib.suppress(Exception):
                conf = float(conf)
            if not np.isfinite(float(conf)):
                conf = 0.0
            pos_raw = row.get("position_3d", row.get("position_mean", []))
            pos = [0.0, 0.0, 0.0]
            if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 3:
                with contextlib.suppress(Exception):
                    pos = [float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2])]
            full_image = str(row.get("full_image", "") or "").strip()
            full_image_camera_id = str(row.get("full_image_camera_id", "") or "").strip()
            top_crops = [
                str(x or "").strip() for x in list(row.get("top_cropped_images", []) or []) if str(x or "").strip()
            ]
            if not top_crops:
                with contextlib.suppress(Exception):
                    top1 = str(row.get("top_crop", "") or "").strip()
                    if top1:
                        top_crops = [top1]
            viewpoints_raw = list(row.get("viewpoints", []) or [])
            viewpoints: List[List[float]] = []
            for vp in viewpoints_raw:
                if not isinstance(vp, (list, tuple)) or len(vp) < 3:
                    continue
                with contextlib.suppress(Exception):
                    xyz = [float(vp[0]), float(vp[1]), float(vp[2])]
                    if len(vp) >= 7:
                        quat = [float(vp[3]), float(vp[4]), float(vp[5]), float(vp[6])]
                    else:
                        quat = [0.0, 0.0, 0.0, 1.0]
                    entry = xyz + quat
                    if np.isfinite(np.asarray(entry, dtype=np.float32)).all():
                        viewpoints.append(entry)
            viewpoint_camera_ids = [str(x or "").strip() for x in list(row.get("viewpoint_camera_ids", []) or [])]
            if not viewpoints and len(pos) >= 3:
                viewpoints = [[float(pos[0]), float(pos[1]), float(pos[2]), 0.0, 0.0, 0.0, 1.0]]
            if viewpoints and len(viewpoint_camera_ids) < len(viewpoints):
                fill = full_image_camera_id or "frontleft"
                viewpoint_camera_ids.extend([fill] * (len(viewpoints) - len(viewpoint_camera_ids)))
            if not full_image_camera_id and viewpoint_camera_ids:
                full_image_camera_id = str(viewpoint_camera_ids[0] or "")
            records.append({
                "object_id": object_id,
                "detection_id": int(detection_id),
                "class_name": class_name,
                "confidence_mean": float(conf),
                "position_3d": pos,
                "viewpoints": viewpoints,
                "viewpoint_camera_ids": viewpoint_camera_ids[: len(viewpoints)],
                "full_image": full_image,
                "full_image_camera_id": full_image_camera_id,
                "top_cropped_images": top_crops[:1],
            })

        self._adaptation_manifest_sig = sig
        self._adaptation_records_cache = records
        return list(records)

    def _build_adaptation_candidate_object(
        self,
        row: Dict[str, Any],
        *,
        class_similarity: float,
        final_score: float,
    ) -> Dict[str, Any]:
        object_id = str(row.get("object_id", ""))
        full_image = str(row.get("full_image", "") or "").strip()
        full_image_camera_id = str(row.get("full_image_camera_id", "") or "").strip()
        cropped_images = [
            str(x or "").strip() for x in list(row.get("top_cropped_images", []) or []) if str(x or "").strip()
        ]
        cropped_image = cropped_images[0] if cropped_images else ""
        viewpoints = list(row.get("viewpoints", []) or [])
        viewpoint_camera_ids = list(row.get("viewpoint_camera_ids", []) or [])
        if viewpoints and len(viewpoint_camera_ids) < len(viewpoints):
            fill = full_image_camera_id or "frontleft"
            viewpoint_camera_ids = list(viewpoint_camera_ids) + [fill] * (len(viewpoints) - len(viewpoint_camera_ids))
        if not full_image_camera_id and viewpoint_camera_ids:
            full_image_camera_id = str(viewpoint_camera_ids[0] or "")
        crop_meta_list = [
            {
                "path": path,
                "image_id": None,
                "source_image_path": full_image,
                "camera_id": full_image_camera_id or None,
                "bbox_xyxy": [],
            }
            for path in cropped_images
        ]
        crop_meta = (
            crop_meta_list[0]
            if crop_meta_list
            else {
                "path": cropped_image,
                "image_id": None,
                "source_image_path": full_image,
                "camera_id": full_image_camera_id or None,
                "bbox_xyxy": [],
            }
        )
        return {
            "object_id": object_id,
            "caption": str(row.get("class_name", "") or ""),
            "cropped_image": cropped_image,
            "cropped_images": cropped_images,
            "crop_metadata": crop_meta,
            "crop_metadata_list": crop_meta_list,
            "full_images": [full_image] if full_image else [],
            "full_image": full_image,
            "full_image_camera_id": full_image_camera_id,
            "final_retrieval_score": float(final_score),
            "rerank_score": None,
            "rerank_score_normalized": float(class_similarity),
            "siglip2_similarity": 0.0,
            "qwen3_vl_similarity": 0.0,
            "retrieved_by": ["yoloe"],
            "position": list(row.get("position_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]),
            "viewpoints": viewpoints,
            "viewpoint_camera_ids": viewpoint_camera_ids[: len(viewpoints)],
            "yoloe_class_similarity": float(class_similarity),
            "yoloe_confidence_mean": float(row.get("confidence_mean", 0.0)),
            "yoloe_detection_id": int(row.get("detection_id", -1)),
        }

    @staticmethod
    def _adaptation_confidence(row: Dict[str, Any]) -> float:
        try:
            value = float(row.get("confidence_mean", 0.0))
        except Exception:
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        return float(value)

    def _build_forced_priority_adaptation_clusters(
        self,
        records: List[Dict[str, Any]],
        *,
        query: str,
        task_query: str,
        caption_embedder: Any,
    ) -> List[Dict[str, Any]]:
        """
        Force-add top-N YOLOE detections by confidence as highest-scored candidates.

        User-requested behavior:
          1) precompute embeddings for all YOLOE classes at startup
          2) at query time, compare query embedding vs each YOLOE class embedding
          3) if multiple classes have cosine sim >= threshold, include ALL of them
          4) inject top-K detections per selected class (ranked by confidence)
        """
        if not records:
            return []

        query_text = str(query or "").strip()

        # Select matching classes by embedding similarity (cosine).
        matched_classes = self._yoloe_matching_classes(query_text, embedder=caption_embedder)

        # If nothing passes threshold, keep old behavior as a conservative fallback:
        # exact/substring match only (do NOT fall back to task_query).
        class_to_sim: Dict[str, float] = {}
        if matched_classes:
            class_to_sim = {cls: sim for cls, sim in matched_classes}
        else:
            for row in records:
                cls = str(row.get("class_name", "") or "").strip()
                if cls and self._adaptation_class_matches_query(cls, query_text):
                    class_to_sim[cls] = 1.0

        if not class_to_sim:
            return []

        top_k_per_class = int(max(1, self._adaptation_top_k_per_class))

        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for row in records:
            cls = str(row.get("class_name", "") or "").strip()
            if not cls or cls not in class_to_sim:
                continue
            by_class.setdefault(cls, []).append(row)

        sorted_classes = sorted(class_to_sim.items(), key=lambda x: x[1], reverse=True)

        base_priority_score = 2.0  # higher than all regular retrieval scores in [0, 1]
        within_class_step = 1e-3
        between_class_step = 1e-2
        out: List[Dict[str, Any]] = []
        global_rank = 0
        for class_rank, (cls, sim) in enumerate(sorted_classes, start=1):
            rows = by_class.get(cls, [])
            if not rows:
                continue
            top_rows = sorted(rows, key=self._adaptation_confidence, reverse=True)[:top_k_per_class]
            for idx, row in enumerate(top_rows):
                global_rank += 1
                priority_score = float(
                    base_priority_score
                    + float(sim) * 0.1
                    - (class_rank * between_class_step)
                    - (idx * within_class_step)
                )
                confidence = self._adaptation_confidence(row)
                candidate = self._build_adaptation_candidate_object(
                    row,
                    class_similarity=float(sim),
                    final_score=priority_score,
                )
                candidate["yoloe_priority_injected"] = True
                candidate["yoloe_priority_rank"] = int(global_rank)
                candidate["yoloe_priority_class_rank"] = int(class_rank)
                candidate["yoloe_priority_within_class_rank"] = int(idx + 1)
                candidate["yoloe_priority_class_name"] = str(cls)

                full_path = str(candidate.get("full_image", "") or "").strip()
                covering_images: List[Dict[str, Any]] = []
                if full_path:
                    camera_ids = list(candidate.get("viewpoint_camera_ids", []) or [])
                    covering_images.append({
                        "image_id": None,
                        "camera_id": camera_ids[0] if camera_ids else None,
                        "storage_path": full_path,
                        "covered_object_ids": [candidate.get("object_id")],
                    })

                out.append({
                    "cluster_score": priority_score,
                    "max_final_retrieval_score": priority_score,
                    "max_text_reciprocal_rank_score": 1.0,
                    "max_vision_reciprocal_rank_score": 1.0,
                    "max_rerank_score_normalized": 1.0,
                    "max_siglip2_similarity": 0.0,
                    "max_qwen3_vl_similarity": 0.0,
                    "candidate_objects": [candidate],
                    "neighbor_objects": [],
                    "covering_images": covering_images,
                    "uncovered_object_ids": [],
                    "cluster_score_source": "yoloe_forced_priority",
                    "retrieval_cluster_score": priority_score,
                    "adaptation_class_name": str(cls),
                    "adaptation_class_similarity": float(sim),
                    "adaptation_literal_query_match": not bool(matched_classes),
                    "adaptation_forced_priority": True,
                    "adaptation_priority_rank": int(global_rank),
                    "adaptation_priority_confidence_mean": float(confidence),
                })

        return out

    def _build_adaptation_clusters(
        self,
        *,
        query: str,
        task_query: str,
        caption_embedder: Any,
    ) -> List[Dict[str, Any]]:
        records = self._load_adaptation_records()
        if not records:
            return []

        # Only return the top-N YOLOE detections (forced priority). We no longer append the
        # full YOLOE adaptation clusters, which can drown out scene-graph candidates and
        # create duplicate IDs across clusters.
        return self._build_forced_priority_adaptation_clusters(
            records,
            query=query,
            task_query=task_query,
            caption_embedder=caption_embedder,
        )

    @staticmethod
    def _normalize_adaptation_text(text: Any) -> str:
        value = str(text or "").strip().lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    def _adaptation_class_matches_query(self, class_name: str, query: str) -> bool:
        cls = self._normalize_adaptation_text(class_name)
        q = self._normalize_adaptation_text(query)
        if not cls or not q:
            return False
        if cls == q:
            return True
        return cls in q or q in cls

    @staticmethod
    def _save_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _cap_text(text: Any, max_chars: int) -> str:
        value = " ".join(str(text or "").strip().split())
        if max_chars <= 0:
            return value
        return value if len(value) <= int(max_chars) else value[: int(max_chars)]

    def _build_rerank_query(self, query: str) -> str:
        query_text = " ".join(str(query or "").strip().split())
        prefix = " ".join(str(self._rerank_query_prefix or "").strip().split())
        if not prefix:
            return f"Target: {query_text}"
        return f"{prefix}\nTarget: {query_text}"

    @staticmethod
    def _format_task_query_prompt(template: str, query: str) -> str:
        query_text = " ".join(str(query or "").strip().split())
        template_text = str(template or "").strip()
        if not query_text:
            return template_text
        if "{query}" in template_text:
            return template_text.replace("{query}", query_text)
        if "Task:" in template_text:
            return f"{template_text}\n{query_text}"
        return f"{template_text}\n\nTask: {query_text}"

    def _build_task_caption_query(self, query: str) -> str:
        template = str(os.getenv("QWEN3_TASK_CAPTION_QUERY_PROMPT") or _DEFAULT_TASK_CAPTION_QUERY_PROMPT)
        return self._format_task_query_prompt(template, query)

    def _build_task_image_query(self, query: str) -> str:
        template = str(os.getenv("QWEN3_TASK_IMAGE_QUERY_PROMPT") or _DEFAULT_TASK_IMAGE_QUERY_PROMPT)
        return self._format_task_query_prompt(template, query)

    def _fallback_caption_ranking(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in candidates[: int(top_k)]:
            obj = dict(item)
            raw = obj.get("retrieval_score", 0.0)
            try:
                score = float(raw)
            except Exception:
                score = 0.0
            obj["rerank_score"] = score
            out.append(obj)
        return out

    def _write_caption_debug(
        self,
        query: str,
        reranked: List[Dict[str, Any]],
        *,
        rerank_query: str,
        debug_log_path: Path | None,
        debug_top_k: int = 10,
    ) -> None:
        if debug_log_path is None:
            return
        top_n = max(1, int(debug_top_k))
        payload = {
            "method": "caption",
            "query": str(query),
            "rerank_query": str(rerank_query),
            "rerank_doc_max_chars": int(self._rerank_doc_max_chars),
            "results": [
                {
                    "rank": rank + 1,
                    "object_id": item.get("object_id"),
                    "caption": str(item.get("object_caption", "") or ""),
                    "retrieval_score": float(item.get("retrieval_score", 0.0)),
                    "rerank_score": float(item.get("rerank_score", 0.0)),
                }
                for rank, item in enumerate(reranked[:top_n])
            ],
        }
        self._save_json(debug_log_path, payload)

    def _rerank_caption_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
        debug_log_path: Path | None = None,
        debug_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        n = max(1, min(int(top_k), len(candidates)))
        rerank_query = self._build_rerank_query(query)
        reranked = self._fallback_caption_ranking(candidates, n)
        self._write_caption_debug(
            query,
            reranked,
            rerank_query=rerank_query,
            debug_log_path=debug_log_path,
            debug_top_k=debug_top_k,
        )
        return reranked

    def _media_resolution_bases(self) -> List[Path]:
        bases: List[Path] = [Path.cwd()]
        if self.scene_graph_json_path is not None:
            bases.append(self.scene_graph_json_path.parent)
            with contextlib.suppress(Exception):
                bases.append(self.scene_graph_json_path.parent.parent)
        # In-memory entry point: use the .pt's parent dir as a fallback base
        # so adaptation manifests / snapshot-root probes can still resolve.
        scene_state_path = getattr(getattr(self, "_processor", None), "_scene_state_path", None)
        if scene_state_path is not None:
            with contextlib.suppress(Exception):
                bases.append(Path(scene_state_path).expanduser().parent)
        if self.snapshot_root is not None:
            bases.append(self.snapshot_root)
            with contextlib.suppress(Exception):
                bases.append(self.snapshot_root.parent)
        if self.image_store_root is not None:
            bases.append(self.image_store_root)
            with contextlib.suppress(Exception):
                bases.append(self.image_store_root.parent)
        uniq: List[Path] = []
        seen: set[str] = set()
        for base in bases:
            if base is None:
                continue
            try:
                key = str(base.resolve())
            except Exception:
                key = str(base)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(base)
        return uniq

    def _resolve_existing_media_path(self, raw_path: Any) -> Optional[Path]:
        raw = str(raw_path or "").strip()
        if not raw:
            return None
        raw_candidate = Path(raw).expanduser()

        def _resolve_relative_variant(candidate: Path) -> Optional[Path]:
            if candidate.exists():
                return candidate
            trimmed_snapshots_raw: Optional[Path] = None
            if len(candidate.parts) >= 3 and candidate.parts[0] == "log" and candidate.parts[1] == "snapshots":
                trimmed_snapshots_raw = Path(*candidate.parts[2:])
            for base in self._media_resolution_bases():
                tests: List[Path] = [base / candidate]
                if len(candidate.parts) >= 2 and candidate.parts[0] == "log":
                    tests.append(base / Path(*candidate.parts[1:]))
                if len(candidate.parts) >= 2 and candidate.parts[0] == "models":
                    tests.append(base / Path(*candidate.parts[1:]))
                if len(candidate.parts) >= 1 and candidate.parts[0] == "snapshots":
                    tests.append(base / "models" / candidate)
                if trimmed_snapshots_raw is not None:
                    tests.append(base / "models" / "snapshots" / trimmed_snapshots_raw)
                    if base.name == "snapshots":
                        tests.append(base / trimmed_snapshots_raw)
                for test in tests:
                    if test.exists():
                        return test
            return None

        if raw_candidate.is_absolute():
            if raw_candidate.exists():
                return raw_candidate
            # Rebase absolute paths produced in another runtime/root (e.g. docker path) to local workspace roots.
            # Example:
            #   /.../ros/ros2_ws/log/snapshots/yoloe_adaptation/... -> models/snapshots/yoloe_adaptation/...
            parts = list(raw_candidate.parts)
            rebased_variants: List[Path] = []
            for anchor in ("log", "models", "snapshots"):
                if anchor in parts:
                    idx = parts.index(anchor)
                    if idx < len(parts):
                        rebased_variants.append(Path(*parts[idx:]))
            if "log" in parts:
                idx = parts.index("log")
                if idx + 1 < len(parts):
                    rebased_variants.append(Path(*parts[idx + 1 :]))
            # If path contains snapshots/yoloe_adaptation, include the direct relative tail.
            if "snapshots" in parts:
                idx = parts.index("snapshots")
                if idx < len(parts):
                    rebased_variants.append(Path(*parts[idx:]))

            seen: set[str] = set()
            for rel_candidate in rebased_variants:
                key = str(rel_candidate)
                if key in seen:
                    continue
                seen.add(key)
                resolved = _resolve_relative_variant(rel_candidate)
                if resolved is not None:
                    return resolved
            return None

        resolved_rel = _resolve_relative_variant(raw_candidate)
        if resolved_rel is not None:
            return resolved_rel
        return None

    def _ensure_latest_scene_graph(self, *, debug_root: Optional[Path] = None) -> bool:
        # In-memory entry point (``from_scene_state`` / ``processor=…``): the JSON
        # path is absent and the processor is already populated. We just need to
        # build the covisibility graph the first time, then no-op forever.
        if self._processor_externally_provided:
            assert self._processor is not None
            covis_shape = getattr(self._processor, "covisibility_matrix", np.zeros((0, 0))).shape[0]
            n_obj_ids = len(getattr(self._processor, "object_ids", []))
            first_call = self._loaded_state_sig is None
            covis_built = covis_shape == n_obj_ids and bool(getattr(self._processor, "covisibility_matrix", np.zeros((0, 0))).any())
            if first_call or not covis_built:
                covis_args = dict(self.covisibility_kwargs)
                if debug_root is not None:
                    covis_args["debug_plot_path"] = debug_root / "covisibility_topdown.png"
                    covis_args["debug_title"] = "Covisibility Graph (top-down x-y)"
                self._processor.build_covisibility_graph(**covis_args)
                # Sentinel signature so subsequent calls treat the processor as fresh.
                self._loaded_state_sig = ("scene-state", 0, n_obj_ids)
                self._crop_cache = {}
                if self.verbose:
                    state_src = getattr(self._processor, "_scene_state_path", None) or "<in-memory>"
                    print(f"[SceneGraphRetriever] Loaded from scene_state: {state_src} ({n_obj_ids} objects)")
                return True
            if debug_root is not None:
                covis_args = dict(self.covisibility_kwargs)
                covis_args["debug_plot_path"] = debug_root / "covisibility_topdown.png"
                covis_args["debug_title"] = "Covisibility Graph (top-down x-y)"
                self._processor.build_covisibility_graph(**covis_args)
            return False

        current_scene_sig = self._file_signature(self.scene_graph_json_path)
        if current_scene_sig is None:
            raise FileNotFoundError(f"Scene graph JSON not found: {self.scene_graph_json_path}")

        if self._processor is None:
            if self.verbose:
                print(f"[SceneGraphRetriever] Loading scene graph: {self.scene_graph_json_path}")
                print(f"[SceneGraphRetriever]   snapshot_root: {self.snapshot_root}")
                print(f"[SceneGraphRetriever]   image_store_root: {self.image_store_root}")
            loader = self._import_scene_graph_processing()
            self._processor = loader(
                self.scene_graph_json_path,
                snapshot_root=self.snapshot_root,
            )
            if self.verbose:
                state_path = getattr(self._processor, "snapshot_state_path", None)
                n_objects = len(getattr(self._processor, "object_ids", []))
                print(f"[SceneGraphRetriever]   state.h5: {state_path}")
                print(f"[SceneGraphRetriever]   objects loaded: {n_objects}")
            # Full rebuild after load. Incremental update for only new objects can be added later.
            covis_args = dict(self.covisibility_kwargs)
            if debug_root is not None:
                covis_args["debug_plot_path"] = debug_root / "covisibility_topdown.png"
                covis_args["debug_title"] = "Covisibility Graph (top-down x-y)"
            self._processor.build_covisibility_graph(**covis_args)
            self._loaded_scene_graph_sig = current_scene_sig
            self._loaded_state_sig = self._state_signature(self._processor)
            self._crop_cache = {}
            return True

        state_sig = self._state_signature(self._processor)
        state_hint = self._snapshot_hint_path()
        loaded_state_path = self._loaded_state_sig[0] if self._loaded_state_sig else None
        covis_shape = getattr(self._processor, "covisibility_matrix", np.zeros((0, 0))).shape[0]
        n_obj_ids = len(getattr(self._processor, "object_ids", []))
        sg_changed = self._loaded_scene_graph_sig != current_scene_sig
        state_changed = self._loaded_state_sig != state_sig
        hint_changed = state_hint is not None and loaded_state_path is not None and state_hint != loaded_state_path
        covis_mismatch = covis_shape != n_obj_ids
        needs_reload = sg_changed or state_changed or hint_changed or covis_mismatch
        if needs_reload:
            if self.verbose:
                reasons = []
                if sg_changed:
                    reasons.append(
                        f"scene_graph_json changed (loaded={self._loaded_scene_graph_sig}, current={current_scene_sig})"
                    )
                if state_changed:
                    reasons.append(f"state.h5 changed (loaded={self._loaded_state_sig}, current={state_sig})")
                if hint_changed:
                    reasons.append(f"snapshot hint changed (loaded={loaded_state_path}, hint={state_hint})")
                if covis_mismatch:
                    reasons.append(f"covisibility size mismatch (matrix={covis_shape}, objects={n_obj_ids})")
                print(f"[SceneGraphRetriever] Reloading scene graph: {self.scene_graph_json_path}")
                print(f"[SceneGraphRetriever]   reason: {'; '.join(reasons)}")
            self._processor.load_latest()
            if self.verbose:
                state_path = getattr(self._processor, "snapshot_state_path", None)
                n_objects = len(getattr(self._processor, "object_ids", []))
                print(f"[SceneGraphRetriever]   state.h5: {state_path}")
                print(f"[SceneGraphRetriever]   objects loaded: {n_objects}")
            # Full rebuild after load. Incremental update for only new objects can be added later.
            covis_args = dict(self.covisibility_kwargs)
            if debug_root is not None:
                covis_args["debug_plot_path"] = debug_root / "covisibility_topdown.png"
                covis_args["debug_title"] = "Covisibility Graph (top-down x-y)"
            self._processor.build_covisibility_graph(**covis_args)
            self._loaded_scene_graph_sig = current_scene_sig
            self._loaded_state_sig = self._state_signature(self._processor)
            self._crop_cache = {}
            return True

        if debug_root is not None:
            # Keep per-query debug map generation deterministic even without reload.
            covis_args = dict(self.covisibility_kwargs)
            covis_args["debug_plot_path"] = debug_root / "covisibility_topdown.png"
            covis_args["debug_title"] = "Covisibility Graph (top-down x-y)"
            self._processor.build_covisibility_graph(**covis_args)
        return False

    def _decode_crop_record(
        self,
        *,
        object_key: str,
        crop_index: int,
        flat: np.ndarray,
        hw: Sequence[int],
        channels: int,
        image_id: Optional[int],
        bbox_xyxy: Sequence[float],
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "path": "",
            "image_id": image_id,
            "source_image_path": "",
            "bbox_xyxy": [float(x) for x in (bbox_xyxy or [])[:4]],
        }
        if image_id is not None:
            processor = self._processor
            source_map = getattr(processor, "_image_id_to_storage_path", {}) if processor is not None else {}
            info["source_image_path"] = str(source_map.get(int(image_id), ""))
        if Image is None:
            return info
        if len(hw) < 2:
            return info
        h, w = int(hw[0]), int(hw[1])
        c = int(channels)
        if h <= 0 or w <= 0 or c <= 0:
            return info
        if int(flat.size) != int(h * w * c):
            return info

        if c == 1:
            arr = flat.reshape((h, w))
            image = Image.fromarray(arr, mode="L")
        else:
            arr = flat.reshape((h, w, c))
            if c == 3:
                image = Image.fromarray(arr, mode="RGB")
            elif c == 4:
                image = Image.fromarray(arr, mode="RGBA")
            else:
                return info

        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", object_key)
        out_path = self._crop_output_dir() / f"object_{safe_key}_{int(crop_index):03d}.png"
        with contextlib.suppress(Exception):
            image.save(out_path, format="PNG")
            info["path"] = str(out_path)
        with contextlib.suppress(Exception):
            mem = io.BytesIO()
            image.save(mem, format="PNG")
            info["png_base64"] = base64.b64encode(mem.getvalue()).decode("ascii")
        return info

    def _resolve_crops(self, object_key: str) -> List[Dict[str, Any]]:
        cached = self._crop_cache.get(object_key)
        if isinstance(cached, dict) and isinstance(cached.get("all"), list):
            return list(cached.get("all") or [])

        empty_info: Dict[str, Any] = {
            "path": "",
            "image_id": None,
            "source_image_path": "",
            "bbox_xyxy": [],
        }
        crops: List[Dict[str, Any]] = []
        processor = self._processor
        if processor is None or h5py is None:
            self._crop_cache[object_key] = {"primary": empty_info, "all": []}
            return []

        state_path = getattr(processor, "snapshot_state_path", None)
        if state_path is None:
            self._crop_cache[object_key] = {"primary": empty_info, "all": []}
            return []
        state_path = Path(state_path)
        if not state_path.exists():
            self._crop_cache[object_key] = {"primary": empty_info, "all": []}
            return []

        row_map = getattr(processor, "_snapshot_object_id_to_row", {})
        row = row_map.get(object_key)
        if row is None:
            self._crop_cache[object_key] = {"primary": empty_info, "all": []}
            return []

        with contextlib.suppress(Exception), h5py.File(state_path, "r") as handle:
            # New multi-crop format.
            multi_needed = {
                "caption_crop_item_row_ptr",
                "caption_crop_item_flat_row_ptr",
                "caption_crop_item_data",
                "caption_crop_item_hw",
                "caption_crop_item_channels",
                "caption_crop_item_image_id",
                "caption_crop_item_bbox_xyxy",
            }
            if multi_needed.issubset(set(handle.keys())):
                item_row_ptr = np.asarray(handle["caption_crop_item_row_ptr"][:], dtype=np.int64)
                if 0 <= int(row) + 1 < int(item_row_ptr.shape[0]):
                    item_start = int(item_row_ptr[int(row)])
                    item_end = int(item_row_ptr[int(row) + 1])
                    flat_row_ptr = np.asarray(handle["caption_crop_item_flat_row_ptr"][:], dtype=np.int64)
                    item_hw = np.asarray(handle["caption_crop_item_hw"][:], dtype=np.int32)
                    item_channels = np.asarray(handle["caption_crop_item_channels"][:], dtype=np.int32)
                    item_image_id = np.asarray(handle["caption_crop_item_image_id"][:], dtype=np.int64)
                    item_bbox = np.asarray(handle["caption_crop_item_bbox_xyxy"][:], dtype=np.float32)
                    item_data = np.asarray(handle["caption_crop_item_data"][:], dtype=np.uint8)
                    item_encoding = (
                        [str(x) for x in handle["caption_crop_item_encoding"][:]]
                        if "caption_crop_item_encoding" in handle
                        else []
                    )
                    for item_idx in range(max(item_start, 0), max(item_end, 0)):
                        if item_idx < 0 or item_idx + 1 >= int(flat_row_ptr.shape[0]):
                            continue
                        if item_idx >= int(item_hw.shape[0]) or item_idx >= int(item_channels.shape[0]):
                            continue
                        start = int(flat_row_ptr[item_idx])
                        end = int(flat_row_ptr[item_idx + 1])
                        if not (0 <= start <= end <= int(item_data.shape[0])):
                            continue
                        flat = np.asarray(item_data[start:end], dtype=np.uint8)
                        hw = np.asarray(item_hw[item_idx], dtype=np.int32).tolist()
                        channels = int(item_channels[item_idx])
                        image_id = int(item_image_id[item_idx]) if item_idx < int(item_image_id.shape[0]) else None
                        bbox = (
                            np.asarray(item_bbox[item_idx], dtype=np.float32).tolist()
                            if item_idx < int(item_bbox.shape[0])
                            else []
                        )
                        rec = self._decode_crop_record(
                            object_key=object_key,
                            crop_index=item_idx - item_start,
                            flat=flat,
                            hw=hw,
                            channels=channels,
                            image_id=image_id,
                            bbox_xyxy=bbox,
                        )
                        if item_idx < len(item_encoding):
                            enc_raw = item_encoding[item_idx]
                            if isinstance(enc_raw, bytes):
                                rec["encoding"] = enc_raw.decode("utf-8", errors="replace")
                            else:
                                rec["encoding"] = str(enc_raw or "")
                        crops.append(rec)

            # Legacy single-crop format fallback.
            if not crops and "caption_crop_row_ptr" in handle:
                row_ptr = handle["caption_crop_row_ptr"]
                if int(row) >= 0 and int(row) + 1 < int(row_ptr.shape[0]):
                    start = int(row_ptr[int(row)])
                    end = int(row_ptr[int(row) + 1])
                    if start < end and {"caption_crop_data", "caption_crop_hw", "caption_crop_channels"}.issubset(
                        set(handle.keys())
                    ):
                        image_id = None
                        if "caption_crop_image_id" in handle:
                            with contextlib.suppress(Exception):
                                image_id = int(handle["caption_crop_image_id"][int(row)])
                        bbox = []
                        if "caption_crop_bbox_xyxy" in handle:
                            with contextlib.suppress(Exception):
                                bbox = np.asarray(handle["caption_crop_bbox_xyxy"][int(row)], dtype=np.float32).tolist()
                        flat = np.asarray(handle["caption_crop_data"][start:end], dtype=np.uint8)
                        hw = np.asarray(handle["caption_crop_hw"][int(row)], dtype=np.int32).tolist()
                        channels = int(handle["caption_crop_channels"][int(row)])
                        rec = self._decode_crop_record(
                            object_key=object_key,
                            crop_index=0,
                            flat=flat,
                            hw=hw,
                            channels=channels,
                            image_id=image_id,
                            bbox_xyxy=bbox,
                        )
                        if "caption_crop_encoding" in handle:
                            with contextlib.suppress(Exception):
                                rec["encoding"] = str(handle["caption_crop_encoding"][int(row)] or "")
                        crops.append(rec)

        primary = crops[0] if crops else empty_info
        self._crop_cache[object_key] = {"primary": primary, "all": crops}
        return crops

    def _resolve_crop(self, object_key: str) -> Dict[str, Any]:
        crops = self._resolve_crops(object_key)
        if crops:
            return dict(crops[0])
        return {
            "path": "",
            "image_id": None,
            "source_image_path": "",
            "bbox_xyxy": [],
        }

    def _build_candidate_object(
        self,
        object_key: str,
        *,
        final_retrieval_score: float,
        rerank_score: Optional[float],
        rerank_score_normalized: float,
        siglip2_similarity: float,
        qwen3_vl_similarity: float,
        retrieved_by_l: bool = False,
        retrieved_by_lt: bool = False,
        retrieved_by_sig: bool = False,
        retrieved_by_vl: bool = False,
        retrieved_by_vlt: bool = False,
    ) -> Dict[str, Any]:
        retrieved_by: List[str] = []
        # Prefer explicit membership flags (more reliable than score thresholds).
        if bool(retrieved_by_l):
            retrieved_by.append("l")
        if bool(retrieved_by_lt):
            retrieved_by.append("lt")
        if bool(retrieved_by_vl):
            retrieved_by.append("vl")
        if bool(retrieved_by_vlt):
            retrieved_by.append("vlt")
        if bool(retrieved_by_sig):
            retrieved_by.append("sig")
        if not retrieved_by:
            # Backward-compatible inference.
            if float(rerank_score_normalized or 0.0) != 0.0:
                retrieved_by.append("l")
            if float(qwen3_vl_similarity or 0.0) != 0.0:
                retrieved_by.append("vl")
            if float(siglip2_similarity or 0.0) != 0.0:
                retrieved_by.append("sig")
        if not retrieved_by:
            retrieved_by = ["unk"]

        processor = self._processor
        if processor is None:
            return {
                "object_id": object_key,
                "caption": "",
                "position": [0.0, 0.0, 0.0],
                "viewpoints": [],
                "viewpoint_camera_ids": [],
                "cropped_image": "",
                "cropped_images": [],
                "crop_metadata_list": [],
                "full_images": [],
                "full_image": "",
                "final_retrieval_score": float(final_retrieval_score),
                "rerank_score": rerank_score,
                "rerank_score_normalized": rerank_score_normalized,
                "siglip2_similarity": siglip2_similarity,
                "qwen3_vl_similarity": qwen3_vl_similarity,
                "retrieved_by": retrieved_by,
            }

        obj = getattr(processor, "objects_by_id", {}).get(object_key, {})
        object_id_raw = obj.get("object_id", obj.get("id", object_key))
        position = list(obj.get("mean", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        viewpoints = list(obj.get("viewpoints", []) or [])
        viewpoint_camera_ids = list(obj.get("viewpoint_camera_ids", []) or [])
        crop_infos = self._resolve_crops(object_key)
        crop_info = crop_infos[0] if crop_infos else self._resolve_crop(object_key)
        cropped_images = [
            str(info.get("path", "") or "").strip() for info in crop_infos if str(info.get("path", "")).strip()
        ]
        cropped_image = cropped_images[0] if cropped_images else str(crop_info.get("path", "") or "")
        full_images: List[str] = []
        for info in crop_infos:
            raw = str(info.get("source_image_path", "") or "").strip()
            if raw and raw not in full_images:
                full_images.append(raw)

        obj_index = getattr(processor, "_object_key_to_index", {}).get(object_key)
        region_label = processor.get_region_label_for_object(obj_index) if obj_index is not None else None

        return {
            "object_id": object_id_raw,
            "caption": str(obj.get("object_caption", "") or ""),
            "cropped_image": cropped_image,
            "cropped_images": cropped_images,
            "crop_metadata": crop_info,
            "crop_metadata_list": crop_infos,
            "full_images": full_images,
            "full_image": full_images[0] if full_images else "",
            "final_retrieval_score": float(final_retrieval_score),
            "rerank_score": rerank_score,
            "rerank_score_normalized": rerank_score_normalized,
            "siglip2_similarity": siglip2_similarity,
            "qwen3_vl_similarity": qwen3_vl_similarity,
            "retrieved_by": retrieved_by,
            "position": position,
            "viewpoints": viewpoints,
            "viewpoint_camera_ids": viewpoint_camera_ids,
            "region_label": region_label,
        }

    def _cluster_folder_id(self, cluster: Dict[str, Any], *, rank_override: Optional[int] = None) -> str:
        rank = int(rank_override if rank_override is not None else cluster.get("rank", 0))
        if str(cluster.get("cluster_score_source", "") or "").strip().lower().startswith("yoloe"):
            return f"cluster_{rank:02d}_yoloe"
        candidates = list(cluster.get("candidate_objects", []) or [])
        suffix_tags: List[str] = []
        seen: set[str] = set()
        for cand in candidates[:2]:
            if not isinstance(cand, dict):
                continue
            raw = cand.get("retrieved_by", [])
            tags = [str(x) for x in (raw if isinstance(raw, list) else []) if str(x)]
            if not tags:
                tags = ["unk"]
            for tag in tags:
                if tag in seen:
                    continue
                seen.add(tag)
                suffix_tags.append(tag)
        if len(suffix_tags) > 1 and "unk" in suffix_tags:
            suffix_tags = [t for t in suffix_tags if t != "unk"]
        suffix = "_".join(suffix_tags) if suffix_tags else ""
        return f"cluster_{rank:02d}_{suffix}" if suffix else f"cluster_{rank:02d}"

    def _write_cluster_media(self, query_debug_dir: Path, ranked_clusters: List[Dict[str, Any]]) -> None:
        for cluster in ranked_clusters:
            rank = int(cluster.get("rank", 0))
            cluster_id = self._cluster_folder_id(cluster, rank_override=rank)
            cluster["cluster_id"] = cluster_id
            cluster_dir = query_debug_dir / cluster_id
            crop_dir = cluster_dir / "cropped_images"
            full_dir = cluster_dir / "full_images"
            cover_dir = cluster_dir / "minimum_covering_images"
            crop_dir.mkdir(parents=True, exist_ok=True)
            full_dir.mkdir(parents=True, exist_ok=True)
            cover_dir.mkdir(parents=True, exist_ok=True)

            crop_manifest: List[Dict[str, Any]] = []
            for idx, candidate in enumerate(cluster.get("candidate_objects", [])):
                obj_id = self._safe_token(candidate.get("object_id", idx))
                crop_sources: List[str] = []
                if isinstance(candidate.get("cropped_images", []), list):
                    crop_sources.extend(str(x or "").strip() for x in candidate.get("cropped_images", []))
                crop_single = str(candidate.get("cropped_image", "") or "").strip()
                if crop_single:
                    crop_sources.append(crop_single)
                seen_crop_sources: set[str] = set()
                added_for_candidate = False
                for crop_rank, src in enumerate(crop_sources, start=1):
                    if not src or src in seen_crop_sources:
                        continue
                    seen_crop_sources.add(src)
                    copied_path = ""
                    resolved_src = self._resolve_existing_media_path(src)
                    if resolved_src is not None and resolved_src.exists() and resolved_src.is_file():
                        suffix = resolved_src.suffix or ".png"
                        dst = crop_dir / f"{idx + 1:02d}_{obj_id}_{crop_rank:02d}{suffix}"
                        shutil.copy2(resolved_src, dst)
                        copied_path = str(dst)
                    crop_manifest.append({
                        "rank": idx + 1,
                        "crop_rank": int(crop_rank),
                        "object_id": candidate.get("object_id"),
                        "caption": candidate.get("caption"),
                        "source": src,
                        "resolved_source": str(resolved_src) if resolved_src is not None else "",
                        "saved_path": copied_path,
                    })
                    added_for_candidate = True
                if not added_for_candidate:
                    crop_meta = candidate.get("crop_metadata", {}) or {}
                    encoded = str(crop_meta.get("png_base64", "") or "").strip()
                    if encoded:
                        with contextlib.suppress(Exception):
                            raw = base64.b64decode(encoded)
                            dst = crop_dir / f"{idx + 1:02d}_{obj_id}.png"
                            dst.write_bytes(raw)
                            crop_manifest.append({
                                "rank": idx + 1,
                                "crop_rank": 1,
                                "object_id": candidate.get("object_id"),
                                "caption": candidate.get("caption"),
                                "source": crop_single,
                                "resolved_source": "",
                                "saved_path": str(dst),
                            })

            full_manifest: List[Dict[str, Any]] = []
            for idx, candidate in enumerate(cluster.get("candidate_objects", [])):
                obj_id = self._safe_token(candidate.get("object_id", idx))
                full_sources = candidate.get("full_images", []) or []
                if not isinstance(full_sources, list):
                    full_sources = []
                full_single = str(candidate.get("full_image", "") or "").strip()
                if full_single:
                    full_sources.append(full_single)
                seen_full_sources: set[str] = set()
                for full_rank, src in enumerate(full_sources, start=1):
                    src_norm = str(src or "").strip()
                    if not src_norm or src_norm in seen_full_sources:
                        continue
                    seen_full_sources.add(src_norm)
                    copied_path = ""
                    resolved_src = self._resolve_existing_media_path(src_norm)
                    if resolved_src is not None and resolved_src.exists() and resolved_src.is_file():
                        suffix = resolved_src.suffix or ".png"
                        dst = full_dir / f"{idx + 1:02d}_{obj_id}_{full_rank:02d}{suffix}"
                        shutil.copy2(resolved_src, dst)
                        copied_path = str(dst)
                    full_manifest.append({
                        "rank": idx + 1,
                        "full_rank": int(full_rank),
                        "object_id": candidate.get("object_id"),
                        "source": src_norm,
                        "resolved_source": str(resolved_src) if resolved_src is not None else "",
                        "saved_path": copied_path,
                    })

            covering_manifest: List[Dict[str, Any]] = []
            for idx, image_item in enumerate(cluster.get("covering_images", [])):
                src = str(image_item.get("storage_path", "") or "").strip()
                copied_path = ""
                resolved_src = self._resolve_existing_media_path(src)
                if src:
                    src_path = resolved_src
                    if src_path is not None and src_path.exists() and src_path.is_file():
                        suffix = src_path.suffix or ".png"
                        image_id = image_item.get("image_id")
                        if image_id is None or str(image_id).strip() == "":
                            image_token = f"idx_{idx + 1:02d}"
                        else:
                            image_token = ""
                            with contextlib.suppress(Exception):
                                image_token = str(int(image_id))
                            if not image_token:
                                image_token = self._safe_token(image_id)
                        dst = cover_dir / f"{idx + 1:02d}_image_{image_token}{suffix}"
                        shutil.copy2(src_path, dst)
                        copied_path = str(dst)
                # Expose the debug copy location to callers so the returned result matches the saved artifacts.
                image_item["saved_path"] = copied_path
                image_item["resolved_storage_path"] = str(resolved_src) if resolved_src is not None else ""
                covering_manifest.append({
                    "rank": idx + 1,
                    "image_id": image_item.get("image_id"),
                    "camera_id": image_item.get("camera_id"),
                    "covered_object_ids": image_item.get("covered_object_ids", []),
                    "source": src,
                    "resolved_source": str(resolved_src) if resolved_src is not None else "",
                    "saved_path": copied_path,
                })

            # Keep a stable name for consumers that expect "minimum_covering_images" rather than "covering_images".
            cluster["minimum_covering_images"] = list(cluster.get("covering_images", []) or [])

            self._save_json(
                cluster_dir / "cluster_media_manifest.json",
                {
                    "cluster_id": cluster_id,
                    "cluster_rank": rank,
                    "cluster_score": cluster.get("cluster_score"),
                    "cropped_images": crop_manifest,
                    "full_images": full_manifest,
                    "minimum_covering_images": covering_manifest,
                },
            )

    @staticmethod
    def _candidate_primary_score(candidate: Dict[str, Any], score_key: str) -> float:
        raw = candidate.get(score_key, candidate.get("final_retrieval_score", 0.0))
        try:
            score = float(raw)
        except Exception:
            score = 0.0
        if not np.isfinite(score):
            score = 0.0
        return float(score)

    def _select_corresponding_covering_image(
        self,
        cluster: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        covering_all = list(cluster.get("covering_images", []) or [])
        crop_meta = candidate.get("crop_metadata", {}) or {}
        crop_meta_list = candidate.get("crop_metadata_list", []) or []
        candidate_meta_rows: List[Dict[str, Any]] = []
        if isinstance(crop_meta, dict):
            candidate_meta_rows.append(crop_meta)
        if isinstance(crop_meta_list, list):
            for entry in crop_meta_list:
                if isinstance(entry, dict):
                    candidate_meta_rows.append(entry)
        candidate_obj_id = candidate.get("object_id")
        source_images: List[str] = []
        source_image_ids: List[int] = []
        for meta_row in candidate_meta_rows:
            source_image = str(meta_row.get("source_image_path", "") or "").strip()
            if source_image and source_image not in source_images:
                source_images.append(source_image)
            raw_image_id = meta_row.get("image_id")
            with contextlib.suppress(Exception):
                image_id_int = int(raw_image_id)
                if image_id_int not in source_image_ids:
                    source_image_ids.append(image_id_int)

        source_resolved_paths: List[Path] = []
        for source in source_images:
            resolved = self._resolve_existing_media_path(source)
            if resolved is not None:
                source_resolved_paths.append(resolved)

        selected: Optional[Dict[str, Any]] = None
        for source_image_id in source_image_ids:
            for item in covering_all:
                try:
                    item_id = int(item.get("image_id"))
                except Exception:
                    continue
                if item_id == source_image_id:
                    selected = dict(item)
                    break
            if selected is not None:
                break

        if selected is None and source_resolved_paths:
            source_keys = {str(p.resolve()) for p in source_resolved_paths}
            for item in covering_all:
                raw = str(item.get("storage_path", "") or "").strip()
                if not raw:
                    continue
                resolved = self._resolve_existing_media_path(raw)
                if resolved is None:
                    continue
                if str(resolved.resolve()) in source_keys:
                    selected = dict(item)
                    break

        if selected is None and source_images:
            selected = {
                "image_id": source_image_ids[0] if source_image_ids else None,
                "camera_id": None,
                "storage_path": source_images[0],
                "covered_object_ids": [candidate_obj_id] if candidate_obj_id is not None else [],
            }
        elif selected is None and covering_all:
            selected = dict(covering_all[0])
        elif selected is not None and candidate_obj_id is not None:
            selected["covered_object_ids"] = [candidate_obj_id]

        return [selected] if selected is not None else []

    def _prune_cluster_to_best_candidate(self, cluster: Dict[str, Any], *, score_key: str) -> None:
        candidates = [cand for cand in list(cluster.get("candidate_objects", []) or []) if isinstance(cand, dict)]
        if not candidates:
            cluster["candidate_objects"] = []
            cluster["covering_images"] = []
            cluster["best_candidate_object_id"] = None
            cluster["best_cropped_image"] = ""
            cluster["best_full_image"] = ""
            return

        best = max(
            candidates,
            key=lambda cand: (
                self._candidate_primary_score(cand, score_key),
                self._candidate_primary_score(cand, "final_retrieval_score"),
            ),
        )
        selected_covering = self._select_corresponding_covering_image(cluster, best)
        cluster["candidate_objects"] = [best]
        cluster["covering_images"] = selected_covering
        cluster["best_candidate_object_id"] = best.get("object_id")
        cropped_images = best.get("cropped_images", []) or []
        if isinstance(cropped_images, list) and cropped_images:
            cluster["best_cropped_image"] = str(cropped_images[0] or "")
        else:
            cluster["best_cropped_image"] = str(best.get("cropped_image", "") or "")
        cluster["best_cropped_images"] = [
            str(x or "") for x in (cropped_images if isinstance(cropped_images, list) else [])
        ]
        full_images = best.get("full_images", []) or []
        cluster["best_full_images"] = [str(x or "") for x in (full_images if isinstance(full_images, list) else [])]
        if selected_covering:
            cluster["best_full_image"] = str(selected_covering[0].get("storage_path", "") or "")
        else:
            cluster["best_full_image"] = str(best.get("full_image", "") or "")

    def _rerank_clusters_with_qwen3_vl(
        self,
        *,
        query: str,
        ranked_clusters: List[Dict[str, Any]],
        query_debug_dir: Optional[Path],
        prune_to_best: bool = True,
    ) -> List[Dict[str, Any]]:
        del query_debug_dir
        if not ranked_clusters:
            return ranked_clusters
        for cluster in ranked_clusters:
            cluster["retrieval_cluster_score"] = float(cluster.get("cluster_score", 0.0))
            cluster.setdefault("cluster_score_source", "reciprocal_rank_fusion")
            if prune_to_best:
                self._prune_cluster_to_best_candidate(cluster, score_key="final_retrieval_score")
            else:
                # Keep all candidates but sort within the cluster by final
                # retrieval score so the predictor flatten yields a sensible
                # ordering (best-in-cluster first, then deeper).
                cands = [
                    c for c in (cluster.get("candidate_objects") or [])
                    if isinstance(c, dict)
                ]
                cands.sort(
                    key=lambda c: (
                        self._candidate_primary_score(c, "final_retrieval_score"),
                        self._candidate_primary_score(c, "cluster_score"),
                    ),
                    reverse=True,
                )
                cluster["candidate_objects"] = cands
        return ranked_clusters

    @staticmethod
    def _compact_result_for_json(result: Dict[str, Any]) -> Dict[str, Any]:
        compact = json.loads(json.dumps(result))
        for cluster in compact.get("clusters", []):
            for candidate in cluster.get("candidate_objects", []):
                crop_meta = candidate.get("crop_metadata")
                if isinstance(crop_meta, dict) and "png_base64" in crop_meta:
                    crop_meta.pop("png_base64", None)
                crop_meta_list = candidate.get("crop_metadata_list", [])
                if isinstance(crop_meta_list, list):
                    for row in crop_meta_list:
                        if isinstance(row, dict) and "png_base64" in row:
                            row.pop("png_base64", None)
        return compact

    def _persist_debug_outputs(
        self,
        *,
        query_debug_dir: Optional[Path],
        ranked_clusters: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> None:
        if query_debug_dir is None:
            return
        result["debug_dir"] = str(query_debug_dir)
        with contextlib.suppress(Exception):
            self._write_cluster_media(query_debug_dir, ranked_clusters)
        with contextlib.suppress(Exception):
            self._save_json(query_debug_dir / "retrieval_result.json", self._compact_result_for_json(result))

    def retrieve(
        self,
        query: str,
        *,
        task_query: str | None = None,
        caption_retrieval_k: int = 10,
        caption_rerank_k: int = 10,
        siglip2_k: int = 10,
        candidate_cap: int | None = None,
        cluster_distance_threshold_m: Optional[float] = None,
        require_saved_path_for_covering_images: bool = True,
        debug_output_dir: str | Path | None = None,
        debug_query_id: Any | None = None,
        prune_clusters_to_best: bool = True,
        qwen3_vl_k: int | None = None,
        enable_caption_base: bool = True,
        enable_caption_task: bool = True,
        enable_siglip2: bool = True,
        enable_qwen3_vl: bool = True,
        enable_qwen3_vl_task: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve and structure scene-graph candidates for a text query.

        ``prune_clusters_to_best``: when True (default — the
        Referit3D/grounding-style behavior) each cluster is reduced to its
        single best candidate. Set to False to keep every retrieved
        candidate in its cluster (per-GT similarity eval needs this so
        bumping per-pipeline-k actually grows the predictor's output).

        Returns:
            {
              "query": str,
              "scene_graph_reloaded": bool,
              "clusters": [ ... ranked cluster payloads ... ],
            }
        """
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must be non-empty")
        task_query_text = str(task_query or "").strip() or query

        debug_root = Path(debug_output_dir).expanduser() if debug_output_dir else self.debug_dir
        query_debug_dir: Optional[Path] = None
        if debug_root is not None:
            token = self._safe_token(query)
            if debug_query_id is not None:
                token = f"{self._safe_token(debug_query_id)}_{token}"
            query_debug_dir = debug_root / token
            query_debug_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            loaded_new = self._ensure_latest_scene_graph(debug_root=query_debug_dir)
            processor = self._processor
            if processor is None:
                print(f"[SceneGraphRetriever] processor is None for query '{query}' — returning empty")
                return {
                    "query": query,
                    "scene_graph_reloaded": loaded_new,
                    "caption_candidates": 0,
                    "caption_task_candidates": 0,
                    "siglip2_candidates": 0,
                    "qwen3_vl_candidates": 0,
                    "qwen3_vl_task_candidates": 0,
                    "clusters": [],
                }
            n_objects = len(getattr(processor, "objects_by_id", {}) or {})
            print(
                f"[SceneGraphRetriever] retrieve('{query}'): processor has {n_objects} objects, reloaded={loaded_new}"
            )

            caption_embedder = _Qwen3QueryWrapper(self.embedder)
            caption_k = int(max(1, min(caption_retrieval_k, caption_rerank_k)))
            caption_debug_path = (
                (query_debug_dir / "qwen3_embedding" / "caption_top10.json") if query_debug_dir is not None else None
            )
            caption_task_debug_path = (
                (query_debug_dir / "embedding_task" / "caption_top10.json") if query_debug_dir is not None else None
            )
            try:
                if not enable_caption_base:
                    caption_hits_base = []
                else:
                    caption_candidates = processor.retrieve_by_caption(
                        query,
                        embedder=caption_embedder,
                        reranker=None,
                        retrieval_k=caption_k,
                        top_k=caption_k,
                        debug_log_path=caption_debug_path,
                        debug_top_k=10,
                    )
                    caption_hits_base = self._rerank_caption_candidates(
                        query,
                        caption_candidates,
                        top_k=int(max(1, caption_rerank_k)),
                        debug_log_path=caption_debug_path,
                        debug_top_k=10,
                    )
            except Exception:
                caption_hits_base = []

            try:
                if not enable_caption_task:
                    caption_hits_task = []
                else:
                    task_caption_query = self._build_task_caption_query(task_query_text)
                    caption_task_candidates = processor.retrieve_by_caption(
                        task_caption_query,
                        embedder=caption_embedder,
                        reranker=None,
                        retrieval_k=caption_k,
                        top_k=caption_k,
                        debug_log_path=caption_task_debug_path,
                        debug_top_k=10,
                    )
                    caption_hits_task = self._rerank_caption_candidates(
                        task_caption_query,
                        caption_task_candidates,
                        top_k=int(max(1, caption_rerank_k)),
                        debug_log_path=caption_task_debug_path,
                        debug_top_k=10,
                    )
            except Exception as exc:
                caption_hits_task = []
                if caption_task_debug_path is not None:
                    with contextlib.suppress(Exception):
                        self._save_json(
                            caption_task_debug_path,
                            {
                                "method": "embedding_task",
                                "query": str(task_query_text),
                                "base_query": str(query),
                                "error": repr(exc),
                            },
                        )

            try:
                siglip_debug_path = (
                    (query_debug_dir / "siglip_embedding" / "siglip2_top10.json")
                    if query_debug_dir is not None
                    else None
                )
                if not enable_siglip2:
                    siglip_hits = []
                else:
                    siglip_query_embedder = self._get_siglip2_query_embedder()
                    if siglip_query_embedder is None:
                        raise RuntimeError("local SigLIP2 query embedder unavailable")
                    siglip_hits = processor.retrieve_by_siglip2(
                        query,
                        embedder=siglip_query_embedder,
                        use_ros_service=False,
                        top_k=int(max(1, siglip2_k)),
                        debug_log_path=siglip_debug_path,
                        debug_top_k=10,
                    )
            except Exception as exc:
                if isinstance(siglip_query_embedder, _Siglip2TextEmbedWrapper):
                    self._siglip2_local_failed = True
                    self.siglip2_embedder = None
                siglip_hits = []
                if siglip_debug_path is not None:
                    with contextlib.suppress(Exception):
                        self._save_json(
                            siglip_debug_path,
                            {
                                "method": "siglip2",
                                "query": str(query),
                                "error": repr(exc),
                            },
                        )
            qwen3_vl_debug_path = (
                (query_debug_dir / "qwen3_vl_embedding" / "qwen3_vl_top10.json")
                if query_debug_dir is not None
                else None
            )
            qwen3_vl_task_debug_path = (
                (query_debug_dir / "vl_embedding_task" / "qwen3_vl_top10.json") if query_debug_dir is not None else None
            )
            qwen3_vl_query_embedder = None
            try:
                if not enable_qwen3_vl:
                    qwen3_vl_hits_base = []
                else:
                    qwen3_vl_query_embedder = self._get_qwen3_vl_query_embedder()
                    effective_qwen3_vl_k = int(max(1, qwen3_vl_k if qwen3_vl_k is not None else self.qwen3_vl_k))
                    if qwen3_vl_query_embedder is None:
                        qwen3_vl_hits_base = []
                    else:
                        qwen3_vl_hits_base = processor.retrieve_by_qwen3_vl(
                            query,
                            embedder=qwen3_vl_query_embedder,
                            top_k=effective_qwen3_vl_k,
                            debug_log_path=qwen3_vl_debug_path,
                            debug_top_k=10,
                        )
            except Exception as exc:
                qwen3_vl_hits_base = []
                if qwen3_vl_debug_path is not None:
                    with contextlib.suppress(Exception):
                        self._save_json(
                            qwen3_vl_debug_path,
                            {
                                "method": "qwen3_vl",
                                "query": str(query),
                                "error": repr(exc),
                            },
                        )
            try:
                if not enable_qwen3_vl_task or qwen3_vl_query_embedder is None:
                    qwen3_vl_hits_task = []
                else:
                    task_image_query = self._build_task_image_query(task_query_text)
                    qwen3_vl_hits_task = processor.retrieve_by_qwen3_vl(
                        task_image_query,
                        embedder=qwen3_vl_query_embedder,
                        top_k=effective_qwen3_vl_k,
                        debug_log_path=qwen3_vl_task_debug_path,
                        debug_top_k=10,
                    )
            except Exception as exc:
                qwen3_vl_hits_task = []
                if qwen3_vl_task_debug_path is not None:
                    with contextlib.suppress(Exception):
                        self._save_json(
                            qwen3_vl_task_debug_path,
                            {
                                "method": "vl_embedding_task",
                                "query": str(task_query_text),
                                "base_query": str(query),
                                "error": repr(exc),
                            },
                        )

            reciprocal_rank_offset = 5
            rerank_raw_by_id: Dict[str, float] = {}
            rerank_norm_by_id: Dict[str, float] = {}
            caption_base_by_id: set[str] = set()
            caption_task_by_id: set[str] = set()
            caption_base_rank_by_id: Dict[str, int] = {}
            caption_task_rank_by_id: Dict[str, int] = {}
            for rank, obj in enumerate(caption_hits_base, start=1):
                key = self._object_key(obj.get("object_id", obj.get("id")))
                caption_base_rank_by_id.setdefault(key, int(rank))
                caption_base_by_id.add(key)
                raw = obj.get("rerank_score")
                if raw is None:
                    continue
                try:
                    raw_f = float(raw)
                except Exception:
                    continue
                norm_f = self._normalize_reranker_score(raw_f)
                if key not in rerank_raw_by_id or raw_f > rerank_raw_by_id[key]:
                    rerank_raw_by_id[key] = raw_f
                    rerank_norm_by_id[key] = norm_f
            for rank, obj in enumerate(caption_hits_task, start=1):
                key = self._object_key(obj.get("object_id", obj.get("id")))
                caption_task_rank_by_id.setdefault(key, int(rank))
                caption_task_by_id.add(key)
                raw = obj.get("rerank_score")
                if raw is None:
                    continue
                try:
                    raw_f = float(raw)
                except Exception:
                    continue
                norm_f = self._normalize_reranker_score(raw_f)
                if key not in rerank_raw_by_id or raw_f > rerank_raw_by_id[key]:
                    rerank_raw_by_id[key] = raw_f
                    rerank_norm_by_id[key] = norm_f

            siglip_by_id: Dict[str, float] = {}
            siglip_rank_by_id: Dict[str, int] = {}
            for rank, obj in enumerate(siglip_hits, start=1):
                key = self._object_key(obj.get("object_id", obj.get("id")))
                siglip_rank_by_id.setdefault(key, int(rank))
                raw = obj.get("retrieval_score", 0.0)
                score = self._clip_siglip_score(raw)
                if key not in siglip_by_id or score > siglip_by_id[key]:
                    siglip_by_id[key] = score

            qwen3_vl_by_id: Dict[str, float] = {}
            qwen3_vl_base_by_id: set[str] = set()
            qwen3_vl_task_by_id: set[str] = set()
            qwen3_vl_base_rank_by_id: Dict[str, int] = {}
            qwen3_vl_task_rank_by_id: Dict[str, int] = {}
            for rank, obj in enumerate(qwen3_vl_hits_base, start=1):
                key = self._object_key(obj.get("object_id", obj.get("id")))
                qwen3_vl_base_rank_by_id.setdefault(key, int(rank))
                raw = obj.get("retrieval_score", 0.0)
                score = self._normalize_reranker_score(raw)
                qwen3_vl_base_by_id.add(key)
                if key not in qwen3_vl_by_id or score > qwen3_vl_by_id[key]:
                    qwen3_vl_by_id[key] = score
            for rank, obj in enumerate(qwen3_vl_hits_task, start=1):
                key = self._object_key(obj.get("object_id", obj.get("id")))
                qwen3_vl_task_rank_by_id.setdefault(key, int(rank))
                raw = obj.get("retrieval_score", 0.0)
                score = self._normalize_reranker_score(raw)
                qwen3_vl_task_by_id.add(key)
                if key not in qwen3_vl_by_id or score > qwen3_vl_by_id[key]:
                    qwen3_vl_by_id[key] = score

            adaptation_clusters = self._build_adaptation_clusters(
                query=query,
                task_query=task_query_text,
                caption_embedder=caption_embedder,
            )
            adaptation_manifest_path = self._adaptation_manifest_path()
            adaptation_manifest_exists = bool(
                adaptation_manifest_path is not None and adaptation_manifest_path.exists()
            )

            n_cap_base = len(caption_hits_base)
            n_cap_task = len(caption_hits_task)
            n_siglip = len(siglip_hits)
            n_vl_base = len(qwen3_vl_hits_base)
            n_vl_task = len(qwen3_vl_hits_task)
            n_adapt = len(adaptation_clusters)
            print(
                f"[SceneGraphRetriever] Pipeline hits for '{query}': "
                f"caption_base={n_cap_base}, caption_task={n_cap_task}, "
                f"siglip2={n_siglip}, qwen3_vl={n_vl_base}, qwen3_vl_task={n_vl_task}, "
                f"adaptation={n_adapt}"
            )

            candidate_keys = sorted(
                set(siglip_rank_by_id.keys())
                | set(caption_base_rank_by_id.keys())
                | set(caption_task_rank_by_id.keys())
                | set(qwen3_vl_base_rank_by_id.keys())
                | set(qwen3_vl_task_rank_by_id.keys())
            )
            print(f"[SceneGraphRetriever] Unique candidate keys (union): {len(candidate_keys)}")
            if not candidate_keys:
                ranked_clusters = list(adaptation_clusters)
                ranked_clusters.sort(key=lambda c: c.get("cluster_score", 0.0), reverse=True)
                for idx, cluster in enumerate(ranked_clusters):
                    cluster["rank"] = idx + 1
                result = {
                    "query": query,
                    "scene_graph_reloaded": loaded_new,
                    "clusters": ranked_clusters,
                    "caption_candidates": 0,
                    "caption_task_candidates": 0,
                    "siglip2_candidates": 0,
                    "qwen3_vl_candidates": 0,
                    "qwen3_vl_task_candidates": 0,
                    "adaptation_candidates": sum(
                        len(list(cluster.get("candidate_objects", []) or [])) for cluster in ranked_clusters
                    ),
                    "adaptation_cluster_count": len(ranked_clusters),
                    "adaptation_manifest_path": str(adaptation_manifest_path) if adaptation_manifest_path else "",
                    "adaptation_manifest_exists": bool(adaptation_manifest_exists),
                }
                self._persist_debug_outputs(
                    query_debug_dir=query_debug_dir,
                    ranked_clusters=ranked_clusters,
                    result=result,
                )
                return result

            final_retrieval_score_by_id: Dict[str, float] = {key: 0.0 for key in candidate_keys}
            text_reciprocal_score_by_id: Dict[str, float] = {key: 0.0 for key in candidate_keys}
            vision_reciprocal_score_by_id: Dict[str, float] = {key: 0.0 for key in candidate_keys}
            for key in candidate_keys:
                text_score = float(
                    (
                        self._reciprocal_rank_score(
                            caption_base_rank_by_id.get(key),
                            offset=reciprocal_rank_offset,
                        )
                        + self._reciprocal_rank_score(
                            caption_task_rank_by_id.get(key),
                            offset=reciprocal_rank_offset,
                        )
                        / 5
                    )
                    / 1.2
                )
                vision_score = float(
                    (
                        self._reciprocal_rank_score(
                            qwen3_vl_base_rank_by_id.get(key),
                            offset=reciprocal_rank_offset,
                        )
                        + self._reciprocal_rank_score(
                            qwen3_vl_task_rank_by_id.get(key),
                            offset=reciprocal_rank_offset,
                        )
                        / 5
                        + self._reciprocal_rank_score(
                            siglip_rank_by_id.get(key),
                            offset=reciprocal_rank_offset,
                        )
                    )
                    / 2.2
                )
                text_reciprocal_score_by_id[key] = float(text_score)
                vision_reciprocal_score_by_id[key] = float(vision_score)
                final_retrieval_score_by_id[key] = float(text_score + vision_score)

            # Region-aware scoring boost: multiply scores for candidates in a
            # region that the query mentions (e.g. "the chair in the kitchen").
            if self._region_boost_enabled and hasattr(processor, "region_labels") and processor.region_labels:
                matched_region_idx = self._detect_query_region(query, processor.region_labels)
                if matched_region_idx is not None:
                    for key in candidate_keys:
                        obj_idx = getattr(processor, "_object_key_to_index", {}).get(key)
                        if obj_idx is not None and obj_idx < len(processor.region_ids):
                            if processor.region_ids[obj_idx] == matched_region_idx:
                                final_retrieval_score_by_id[key] *= self._region_boost_factor

            candidate_union_count = int(len(candidate_keys))
            effective_candidate_cap = candidate_cap if candidate_cap is not None else self.candidate_cap
            if effective_candidate_cap is not None:
                cap_n = int(max(1, effective_candidate_cap))
                if len(candidate_keys) > cap_n:
                    candidate_keys = sorted(
                        candidate_keys,
                        key=lambda key: (
                            float(final_retrieval_score_by_id.get(key, 0.0)),
                            float(text_reciprocal_score_by_id.get(key, 0.0)),
                            float(vision_reciprocal_score_by_id.get(key, 0.0)),
                            float(rerank_norm_by_id.get(key, 0.0)),
                            float(siglip_by_id.get(key, 0.0)),
                            float(qwen3_vl_by_id.get(key, 0.0)),
                        ),
                        reverse=True,
                    )[:cap_n]

            cluster_distance = (
                float(self.cluster_distance_threshold_m)
                if cluster_distance_threshold_m is None
                else float(cluster_distance_threshold_m)
            )
            raw_candidate_ids: List[Any] = []
            for key in candidate_keys:
                obj = getattr(processor, "objects_by_id", {}).get(key, {})
                raw_candidate_ids.append(obj.get("object_id", obj.get("id", key)))

            clustered = processor.cluster_objects(
                raw_candidate_ids,
                distance_threshold_m=cluster_distance,
                debug_plot_path=(query_debug_dir / "clusters_topdown.png") if query_debug_dir is not None else None,
                retrieved_object_ids=raw_candidate_ids,
                debug_title=f"Clustered retrieved objects for query: {query}",
            )

            ranked_clusters: List[Dict[str, Any]] = []
            for raw_cluster in clustered:
                cluster_keys = [self._object_key(x) for x in raw_cluster]
                cluster_keys = [key for key in cluster_keys if key in getattr(processor, "objects_by_id", {})]
                if not cluster_keys:
                    continue

                max_rerank_norm = max((rerank_norm_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)
                max_siglip = max((siglip_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)
                max_qwen3_vl = max((qwen3_vl_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)
                max_text_rr = max((text_reciprocal_score_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)
                max_vision_rr = max((vision_reciprocal_score_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)
                cluster_score = max((final_retrieval_score_by_id.get(key, 0.0) for key in cluster_keys), default=0.0)

                sorted_cluster_keys = sorted(
                    cluster_keys,
                    key=lambda key: float(final_retrieval_score_by_id.get(key, 0.0)),
                    reverse=True,
                )

                candidate_objects = [
                    self._build_candidate_object(
                        key,
                        final_retrieval_score=float(final_retrieval_score_by_id.get(key, 0.0)),
                        rerank_score=rerank_raw_by_id.get(key),
                        rerank_score_normalized=rerank_norm_by_id.get(key, 0.0),
                        siglip2_similarity=siglip_by_id.get(key, 0.0),
                        qwen3_vl_similarity=qwen3_vl_by_id.get(key, 0.0),
                        retrieved_by_l=(key in caption_base_by_id),
                        retrieved_by_lt=(key in caption_task_by_id),
                        retrieved_by_sig=(key in siglip_by_id),
                        retrieved_by_vl=(key in qwen3_vl_base_by_id),
                        retrieved_by_vlt=(key in qwen3_vl_task_by_id),
                    )
                    for key in sorted_cluster_keys
                ]

                cluster_id_set = set(sorted_cluster_keys)
                neighbor_map: Dict[str, Dict[str, Any]] = {}
                for key in sorted_cluster_keys:
                    obj = getattr(processor, "objects_by_id", {}).get(key, {})
                    raw_id = obj.get("object_id", obj.get("id", key))
                    for neighbor in processor.get_covisible_neighbors(raw_id):
                        nkey = self._object_key(neighbor.get("object_id", neighbor.get("id")))
                        if nkey in cluster_id_set or nkey in neighbor_map:
                            continue
                        neighbor_map[nkey] = {
                            "object_id": neighbor.get("object_id", neighbor.get("id")),
                            "caption": str(neighbor.get("object_caption", "") or ""),
                            "position": list(neighbor.get("mean", []) or []),
                        }

                cluster_raw_ids: List[Any] = []
                for key in sorted_cluster_keys:
                    obj = getattr(processor, "objects_by_id", {}).get(key, {})
                    cluster_raw_ids.append(obj.get("object_id", obj.get("id", key)))

                covering = processor.minimum_covering_images(
                    cluster_raw_ids,
                    require_saved_path=bool(require_saved_path_for_covering_images),
                )

                ranked_clusters.append({
                    "cluster_score": float(cluster_score),
                    "max_final_retrieval_score": float(cluster_score),
                    "max_text_reciprocal_rank_score": float(max_text_rr),
                    "max_vision_reciprocal_rank_score": float(max_vision_rr),
                    "max_rerank_score_normalized": float(max_rerank_norm),
                    "max_siglip2_similarity": float(max_siglip),
                    "max_qwen3_vl_similarity": float(max_qwen3_vl),
                    "candidate_objects": candidate_objects,
                    "neighbor_objects": list(neighbor_map.values()),
                    "covering_images": list(covering.get("selected_images", [])),
                    "uncovered_object_ids": list(covering.get("uncovered_object_ids", [])),
                })

            ranked_clusters.sort(key=lambda c: c.get("cluster_score", 0.0), reverse=True)
            ranked_clusters = self._rerank_clusters_with_qwen3_vl(
                query=query,
                ranked_clusters=ranked_clusters,
                query_debug_dir=query_debug_dir,
                prune_to_best=prune_clusters_to_best,
            )
            if adaptation_clusters:
                ranked_clusters.extend(adaptation_clusters)
                ranked_clusters.sort(key=lambda c: c.get("cluster_score", 0.0), reverse=True)
            for idx, cluster in enumerate(ranked_clusters):
                cluster["rank"] = idx + 1

            if self.verbose:
                print(
                    "[SceneGraphRetriever] query='{}' reloaded={} clusters={} caption_hits={} siglip_hits={}"
                    " qwen3_vl_hits={} caption_task_hits={} qwen3_vl_task_hits={} adaptation_clusters={}".format(
                        query,
                        loaded_new,
                        len(ranked_clusters),
                        len(caption_hits_base),
                        len(siglip_hits),
                        len(qwen3_vl_hits_base),
                        len(caption_hits_task),
                        len(qwen3_vl_hits_task),
                        len(adaptation_clusters),
                    )
                )

            result = {
                "query": query,
                "task_query": task_query_text,
                "scene_graph_reloaded": loaded_new,
                "candidate_union_count": int(candidate_union_count),
                "candidate_union_count_capped": int(len(candidate_keys)),
                "candidate_cap": int(effective_candidate_cap) if effective_candidate_cap is not None else None,
                "caption_candidates": len(caption_hits_base),
                "caption_task_candidates": len(caption_hits_task),
                "siglip2_candidates": len(siglip_hits),
                "qwen3_vl_candidates": len(qwen3_vl_hits_base),
                "qwen3_vl_task_candidates": len(qwen3_vl_hits_task),
                "adaptation_candidates": sum(
                    len(list(cluster.get("candidate_objects", []) or [])) for cluster in adaptation_clusters
                ),
                "adaptation_cluster_count": len(adaptation_clusters),
                "adaptation_manifest_path": str(adaptation_manifest_path) if adaptation_manifest_path else "",
                "adaptation_manifest_exists": bool(adaptation_manifest_exists),
                "cluster_rerank_enabled": bool(self._cluster_vl_rerank_enabled),
                "cluster_rerank_model": self._cluster_vl_rerank_ckpt,
                "clusters": ranked_clusters,
            }
            if query_debug_dir is not None:
                self._persist_debug_outputs(
                    query_debug_dir=query_debug_dir,
                    ranked_clusters=ranked_clusters,
                    result=result,
                )
            return result
