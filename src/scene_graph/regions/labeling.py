"""LLM-based region/room labeling for object clusters.

Uses a text-only vLLM chat completion call (no images needed) to classify
a cluster of objects into a room type.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None

REGION_LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "confidence"],
    "properties": {
        "label": {"type": "string", "maxLength": 40},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

_SYSTEM_PROMPT = (
    "You are a room/region classifier for indoor environments. "
    "Given a list of objects found in a spatial cluster, determine what type "
    "of room or functional area they most likely belong to. "
    "Respond ONLY with JSON: {\"label\": \"<room_type>\", \"confidence\": <0.0-1.0>}"
)

_COMMON_ROOM_TYPES = (
    "Common room types: kitchen, bathroom, bedroom, living room, dining room, "
    "hallway, office, laundry room, garage, closet, storage room, entryway, "
    "conference room, lobby, utility room, pantry, studio, nursery"
)


def _build_user_prompt(
    object_captions: List[str],
    object_categories: Optional[List[str]] = None,
    max_objects_in_prompt: int = 25,
) -> str:
    lines = ["Objects in this spatial cluster:"]
    n = min(len(object_captions), max_objects_in_prompt)
    for i in range(n):
        cap = object_captions[i]
        cat = object_categories[i] if object_categories and i < len(object_categories) else None
        if cat and cat != cap:
            lines.append(f"- {cap} ({cat})")
        else:
            lines.append(f"- {cap}")
    if len(object_captions) > max_objects_in_prompt:
        lines.append(f"... and {len(object_captions) - max_objects_in_prompt} more objects")
    lines.append("")
    lines.append("What type of room or functional area do these objects belong to?")
    lines.append(_COMMON_ROOM_TYPES)
    return "\n".join(lines)


class RegionLabeler:
    """Label a cluster of objects with a room/region type via vLLM."""

    def __init__(
        self,
        *,
        vllm_base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        max_objects_in_prompt: int = 25,
    ) -> None:
        from scene_graph.llm_utils.llm_config import get_config

        cfg = get_config()
        self._base_url = (vllm_base_url or cfg.vllm_base_url).rstrip("/")
        self._model = model or cfg.vllm_model
        self._api_key = api_key or cfg.vllm_api_key or ""
        self._timeout_s = timeout_s
        self._max_objects = max_objects_in_prompt
        self._session: Optional[Any] = None

    def _get_session(self) -> Any:
        if self._session is None:
            if _requests is None:
                raise RuntimeError("requests library is required for RegionLabeler")
            self._session = _requests.Session()
        return self._session

    def _chat_url(self) -> str:
        base = self._base_url
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def label_region(
        self,
        object_captions: List[str],
        object_categories: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """Classify a cluster of objects into a room type.

        Returns (label, confidence) where label is a room type string and
        confidence is 0.0-1.0.
        """
        if not object_captions:
            return "unknown", 0.0

        user_prompt = _build_user_prompt(
            object_captions,
            object_categories,
            max_objects_in_prompt=self._max_objects,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            sess = self._get_session()
            response = sess.post(
                self._chat_url(),
                json=payload,
                headers=headers,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content") or ""
            ).strip()
            parsed = json.loads(content)
            label = str(parsed.get("label", "unknown")).strip().lower()
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            return label, confidence
        except Exception as exc:
            print(f"[RegionLabeler] WARNING: labeling failed: {exc}")
            return self._fallback_label(object_captions, object_categories), 0.1

    @staticmethod
    def _fallback_label(
        captions: List[str],
        categories: Optional[List[str]] = None,
    ) -> str:
        """Heuristic fallback when the LLM is unavailable."""
        text = " ".join(captions + (categories or [])).lower()
        room_keywords = {
            "kitchen": ["oven", "stove", "microwave", "refrigerator", "fridge", "sink", "toaster", "dishwasher"],
            "bathroom": ["toilet", "shower", "bathtub", "towel rack", "soap"],
            "bedroom": ["bed", "pillow", "nightstand", "dresser", "wardrobe", "mattress"],
            "living room": ["sofa", "couch", "tv", "television", "coffee table", "armchair"],
            "office": ["desk", "monitor", "keyboard", "computer", "office chair", "printer"],
            "dining room": ["dining table", "dining chair", "plates", "cutlery"],
            "laundry room": ["washer", "dryer", "washing machine", "laundry"],
        }
        best_label = "room"
        best_count = 0
        for label, keywords in room_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count = count
                best_label = label
        return best_label
