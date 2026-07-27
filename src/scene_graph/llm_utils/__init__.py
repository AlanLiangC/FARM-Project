"""LLM and embedding client utilities used by scene-graph retrieval.

This is a minimal, self-contained closure so the retrieval subpackage has no
external agent-framework dependency at runtime:

- :class:`EmbedInterface` — vLLM-backed text embedder
- :class:`LLMInterface`   — vLLM-backed text/VLM chat client
"""

from .embed_interface import EmbedInterface
from .llm_config import LLMConfig, get_config, reset_config, set_config
from .llm_interface import LLMInterface

__all__ = [
    "EmbedInterface",
    "LLMConfig",
    "LLMInterface",
    "get_config",
    "reset_config",
    "set_config",
]
