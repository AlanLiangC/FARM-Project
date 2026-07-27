"""
LLM / Embedding / Reranker Configuration

Routes LLM/VLM and embeddings through vLLM endpoints.

Expected servers (start with this repo's ``./run.sh vllm``):
  - VLM/LLM server: default http://localhost:8000/v1 (or 8001/v1 for the smaller model)
  - Embedding server: default http://localhost:8002/v1

Environment Variables:
  VLLM_BASE_URL: base URL for chat completions (default: "http://localhost:8000/v1")
  VLLM_API_KEY: optional auth token (default: unset)
  VLLM_TIMEOUT_S: request timeout seconds (default: 60)

  VLLM_VL_MODEL / VLLM_LLM_MODEL / VLLM_MODEL: served model id (default: "qwen3.5-9b")
  VLLM_DISABLE_THINKING: when 1 (default), inject chat_template_kwargs={enable_thinking: False}
                         into chat completions so reasoning-capable models (Qwen3.5-9B) run in
                         non-thinking instruct mode. Set 0 for non-reasoning models.

  VLLM_EMBED_BASE_URL: base URL for embeddings (default: "http://localhost:8002/v1")
  VLLM_EMBED_API_KEY: optional auth token (default: VLLM_API_KEY)
  VLLM_EMBED_TIMEOUT_S: request timeout seconds (default: 30)
  VLLM_EMBED_MODEL: served embedding model id (default: "qwen3-emb-0.6b")
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

VLLM_HOST = "localhost"

@dataclass
class LLMConfig:
    """Configuration for vLLM (OpenAI-compatible) endpoints and model names."""

    # Chat / VLM endpoint (OpenAI-compatible /v1/chat/completions)
    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", f"http://{VLLM_HOST}:8000/v1"))
    vllm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("VLLM_API_KEY") or None)
    vllm_timeout_s: float = field(default_factory=lambda: float(os.getenv("VLLM_TIMEOUT_S", "60")))

    # Served model id (GET /v1/models). Prefer VLLM_VL_MODEL for multimodal, but allow VLLM_LLM_MODEL / VLLM_MODEL.
    vllm_model: str = field(
        default_factory=lambda: os.getenv("VLLM_VL_MODEL")
        or os.getenv("VLLM_LLM_MODEL")
        or os.getenv("VLLM_MODEL")
        or "qwen3.5-9b"
    )

    # Inject chat_template_kwargs={enable_thinking: False} into chat-completion
    # payloads so reasoning-capable models (default served slot is Qwen3.5-9B)
    # run in non-thinking instruct mode. Override with VLLM_DISABLE_THINKING=0
    # when pointing at a non-reasoning model that rejects the kwarg.
    disable_thinking: bool = field(
        default_factory=lambda: os.getenv("VLLM_DISABLE_THINKING", "1") == "1"
    )

    # Embeddings endpoint (OpenAI-compatible /v1/embeddings)
    vllm_embed_base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_EMBED_BASE_URL", f"http://{VLLM_HOST}:8002/v1")
    )
    vllm_embed_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("VLLM_EMBED_API_KEY") or os.getenv("VLLM_API_KEY") or None
    )
    vllm_embed_timeout_s: float = field(default_factory=lambda: float(os.getenv("VLLM_EMBED_TIMEOUT_S", "30")))
    vllm_embed_model: str = field(default_factory=lambda: os.getenv("VLLM_EMBED_MODEL", "qwen3-emb-0.6b"))

    # Generation parameters
    temperature: float = field(
        default_factory=lambda: float(os.getenv("VLLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.01")))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("VLLM_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "3072")))
    )
    top_p: float = field(default_factory=lambda: float(os.getenv("VLLM_TOP_P", "0.9")))

    # Request parameters
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")))

    def __post_init__(self):
        # Translate legacy Ollama/SGLang/HF model names to current vLLM served-model ids.
        # vLLM in this stack uses --served-model-name {qwen3-vl-8b, qwen3-vl-4b, qwen3-emb-0.6b}.
        self._model_aliases: Dict[str, str] = {
            "qwen3-vl:8b-instruct": "qwen3-vl-8b",
            "qwen3-vl:4b-instruct": "qwen3-vl-4b",
            "qwen3-embedding:0.6b": "qwen3-emb-0.6b",
            "Qwen/Qwen3-VL-8B-Instruct": "qwen3-vl-8b",
            "Qwen/Qwen3-VL-4B-Instruct": "qwen3-vl-4b",
            "Qwen/Qwen3-Embedding-0.6B": "qwen3-emb-0.6b",
            "Qwen/Qwen3.5-9B": "qwen3.5-9b",
        }

    @property
    def llm_url(self) -> str:
        return self.vllm_base_url

    @property
    def embed_url(self) -> str:
        return self.vllm_embed_base_url

    @property
    def llm_model(self) -> str:
        return self.vllm_model

    @property
    def embed_model(self) -> str:
        return self.vllm_embed_model

    def resolve_model(self, model_name: Optional[str]) -> Optional[str]:
        """Resolve a (possibly legacy) model name to the vLLM served-model id."""
        if model_name is None:
            return None
        return self._model_aliases.get(model_name, model_name)


# Global configuration instance
_config: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    global _config
    if _config is None:
        _config = LLMConfig()
    return _config  # noqa: R504


def set_config(config: LLMConfig) -> None:
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to re-read from environment."""
    global _config
    _config = None
