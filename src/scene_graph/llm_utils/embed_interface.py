"""
Unified Embedding Interface (vLLM)

Routes embedding calls through vLLM's OpenAI-compatible endpoint.

Usage::

    from scene_graph.llm_utils import EmbedInterface

    embedder = EmbedInterface(verbose=True)
    emb = embedder.encode("Hello world")
    embs = embedder.encode_batch(["text1", "text2"])

Environment Variables (see :mod:`scene_graph.llm_utils.llm_config`):
    VLLM_EMBED_BASE_URL, VLLM_EMBED_API_KEY, VLLM_EMBED_TIMEOUT_S, VLLM_EMBED_MODEL
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import requests

from scene_graph.llm_utils.llm_config import LLMConfig, get_config

QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT = os.getenv("QWEN3_EMBED_TASK") or (
    "Embed for object identity and affordance retrieval. "
    "Use the object's category, supercategory, and visible attributes; ignore viewpoint and wording."
)


def _canonicalize_qwen3_retrieval_text(text: str) -> str:
    return (text or "").strip().lower()


def wrap_qwen3_retrieval_query(text: str, *, task: str = QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT) -> str:
    canon = _canonicalize_qwen3_retrieval_text(text)
    if not canon:
        return ""
    return f"Instruct: {task}\nQuery: {canon}"


def wrap_qwen3_retrieval_document(text: str, *, task: str = QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT) -> str:
    canon = _canonicalize_qwen3_retrieval_text(text)
    if not canon:
        return ""
    return f"Instruct: {task}\nDocument: {canon}"


class BaseEmbedBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, text: str, timeout: int = 30) -> np.ndarray:
        """Generate embedding for a single text."""


class VLLMEmbedBackend(BaseEmbedBackend):
    """vLLM embedding backend using OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        model_name: Optional[str] = None,
        verbose: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config
        self.model_name = config.resolve_model(model_name) if model_name else config.embed_model
        self.verbose = verbose
        self.base_url = base_url or config.embed_url
        self.api_key = api_key if api_key is not None else config.vllm_embed_api_key

    def _vllm_embed_url(self) -> str:
        base = str(self.base_url).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def encode(self, text: str, timeout: int = 30) -> np.ndarray:
        payload = {"model": self.model_name, "input": text}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self._vllm_embed_url(), json=payload, headers=headers, timeout=float(timeout))
        response.raise_for_status()
        result = response.json()
        data = result.get("data") or []
        if data:
            return np.array(data[0].get("embedding", []), dtype=np.float32)
        return np.zeros(1, dtype=np.float32)

    def encode_batch(self, texts: List[str], timeout: int = 30) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        payload = {"model": self.model_name, "input": texts}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self._vllm_embed_url(), json=payload, headers=headers, timeout=float(timeout))
        response.raise_for_status()
        result = response.json()
        data = result.get("data") or []
        if not data:
            return np.zeros((len(texts), 1), dtype=np.float32)

        # OpenAI-compatible: data is list of {index, embedding}. Keep stable order by index.
        out: List[Optional[List[float]]] = [None for _ in range(len(texts))]
        for row in data:
            try:
                idx = int(row.get("index", 0))
            except Exception:
                idx = 0
            if 0 <= idx < len(out):
                out[idx] = row.get("embedding") or []

        first = next((v for v in out if v), None)
        dim = len(first) if first else 1
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for i, vec in enumerate(out):
            if not vec:
                continue
            mat[i, : len(vec)] = np.asarray(vec, dtype=np.float32)
        return mat


class EmbedInterface:
    """Unified embedding interface using vLLM's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str = None,
        verbose: bool = True,
        config: LLMConfig = None,
    ):
        self.config = config or get_config()
        self.verbose = verbose
        self.model_name = self.config.resolve_model(model_name) if model_name else self.config.embed_model
        self._backend = VLLMEmbedBackend(self.config, self.model_name, verbose=verbose)
        if self.verbose:
            print(f"[EMBED] Using vLLM embeddings at {self.config.embed_url} (model={self.model_name})")

    def encode(self, text: str, max_retries: int = 3) -> np.ndarray:
        """Generate embedding for a single text."""
        for attempt in range(max_retries):
            try:
                return self._backend.encode(text, timeout=float(self.config.vllm_embed_timeout_s))
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[EMBED ERROR] Failed to encode after {max_retries} attempts: {e}")
                    return np.zeros(1, dtype=np.float32)
                time.sleep(0.1)
        return np.zeros(1, dtype=np.float32)

    def encode_query(
        self,
        text: str,
        *,
        task: Optional[str] = None,
        max_retries: int = 3,
    ) -> np.ndarray:
        """Generate a Qwen3 retrieval query embedding (Instruct/Query wrapper)."""
        wrapped = wrap_qwen3_retrieval_query(
            text,
            task=str(task) if task is not None else QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT,
        )
        return self.encode(wrapped, max_retries=max_retries)

    def encode_document(
        self,
        text: str,
        *,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """Generate a Qwen3 retrieval document embedding (Instruct/Document wrapper)."""
        wrapped = wrap_qwen3_retrieval_document(
            text, task=str(task) if task is not None else QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT
        )
        return self.encode(wrapped)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if self.verbose:
            print(f"[EMBED BATCH] Encoding batch of size {len(texts)}")
        try:
            return self._backend.encode_batch(texts, timeout=float(self.config.vllm_embed_timeout_s))
        except Exception:
            embeddings = [self.encode(text) for text in texts]
            return np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])

    def encode_query_batch(
        self,
        texts: List[str],
        *,
        task: Optional[str] = None,
    ) -> np.ndarray:
        wrapped = [
            wrap_qwen3_retrieval_query(
                text,
                task=str(task) if task is not None else QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT,
            )
            for text in (texts or [])
        ]
        return self.encode_batch(wrapped)

    def encode_document_batch(
        self,
        texts: List[str],
        *,
        task: Optional[str] = None,
    ) -> np.ndarray:
        wrapped = [
            wrap_qwen3_retrieval_document(
                text, task=str(task) if task is not None else QWEN3_RETRIEVAL_EMBED_TASK_DEFAULT
            )
            for text in (texts or [])
        ]
        return self.encode_batch(wrapped)


def search_vectors(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Cosine similarity search."""
    norm_query = np.linalg.norm(query_embedding)
    if norm_query > 0:
        query_embedding = query_embedding / norm_query

    norm_docs = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    norm_docs[norm_docs == 0] = 1
    document_embeddings = document_embeddings / norm_docs

    similarities = np.dot(document_embeddings, query_embedding)
    top_indices = np.argsort(-similarities)[:top_k]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
