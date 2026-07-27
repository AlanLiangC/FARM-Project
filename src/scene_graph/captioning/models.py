from dataclasses import asdict, dataclass
from typing import List, Optional

from .structured import StructuredCaption

__all__ = [
    "ObjectCaptionTask",
    "ObjectCaptionResult",
    "StructuredCaption",
    "asdict",
]


@dataclass
class ObjectCaptionTask:
    """Work item requesting a caption for a single object in the scene graph."""

    object_index: int
    object_id: int
    version: Optional[int] = None  # Optional state versioning for downstream use
    attempts: int = 0  # Retry counter for robustness
    is_recaption: bool = False  # True when this task refreshes an existing caption


@dataclass
class ObjectCaptionResult:
    """Caption output corresponding to an object id."""

    object_id: int
    caption: str
    used_view_indices: List[int]
    # Optional structured fields for downstream scene-graph logic.
    category: Optional[str] = None
    supercategory: Optional[str] = None
    category_candidates: Optional[List[str]] = None
    key_attributes: Optional[List[str]] = None
    is_clear_object: Optional[bool] = None
    decision: Optional[str] = None
    caption_embedding: Optional[List[float]] = None
    # Caption text used to generate `caption_embedding` (for consistency checks / recompute-after-merge).
    caption_embedding_text: Optional[str] = None
    siglip2_cls_embedding: Optional[List[float]] = None
    qwen3_vl_cls_embedding: Optional[List[float]] = None
    merge_object_ids: Optional[List[int]] = None
    empty_reason: Optional[str] = None
    is_recaption: bool = False
