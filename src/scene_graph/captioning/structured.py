from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

REQUIRED_KEYS = {
    "category",
    "supercategory",
    "attributes",
    "description",
    "decision",
}

LEGACY_REQUIRED_KEYS = {
    "category",
    "supercategory",
    "attributes",
    "description",
    "IsClearObject",
}

FALSE_CAPTION = {
    "category": "unknown",
    "supercategory": "unknown",
    "attributes": [],
    "description": "",
    "decision": "drop",
}

CAPTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category",
        "supercategory",
        "attributes",
        "description",
        "decision",
    ],
    "properties": {
        "category": {
            "type": "string",
            "description": "Short singular object category, or 'unknown' if unclear.",
        },
        "supercategory": {
            "type": "string",
            "description": (
                "Broad category such as furniture, electronics, appliance, container, fixture, textile, "
                "plant, personal item, tool, other, or unknown."
            ),
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
            "description": "Visible attributes only: color, material, shape, pattern, parts, state, readable text.",
        },
        "description": {
            "type": "string",
            "description": "One short phrase describing the target object. Empty string if unclear.",
        },
        "decision": {
            "type": "string",
            "enum": ["keep", "drop"],
            "description": "Whether this detection is usable for object captioning and scene memory.",
        },
    },
}


@dataclass
class StructuredCaption:
    category: Optional[str]
    supercategory: Optional[str]
    attributes: List[str]
    description: str
    is_clear_object: bool
    decision: Optional[str]
    raw_response: Dict[str, Any]
    valid_json: bool


def _safe_json_loads(raw_text: str) -> tuple[Any, bool]:
    text = str(raw_text or "").strip()
    if not text:
        return None, False
    try:
        return json.loads(text), True
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, False
    try:
        return json.loads(text[start : end + 1]), True
    except Exception:
        return None, False


def _coerce_bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return False


def _normalize_text(value: Any, *, lower: bool = True) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'")
    return text.lower() if lower else text


def _normalize_attributes(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_text(item)
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        if len(out) >= 5:
            break
    return out


def _normalize_description(value: Any) -> str:
    return _normalize_text(value, lower=False)


def _invalid_caption(raw_text: str, *, raw_obj: Optional[Dict[str, Any]] = None, valid_json: bool = False) -> StructuredCaption:
    del raw_text
    return StructuredCaption(
        category=None,
        supercategory=None,
        attributes=[],
        description="",
        is_clear_object=False,
        decision=None,
        raw_response=raw_obj if isinstance(raw_obj, dict) else {},
        valid_json=bool(valid_json),
    )


def _normalize_decision(obj: Dict[str, Any]) -> Optional[str]:
    value = obj.get("decision")
    if isinstance(value, str):
        decision = value.strip().lower()
        if decision in {"keep", "drop"}:
            return decision

    legacy_value = obj.get("IsClearObject")
    if isinstance(legacy_value, bool):
        return "keep" if legacy_value else "drop"
    if isinstance(legacy_value, str):
        normalized = legacy_value.strip().lower()
        if normalized == "true":
            return "keep"
        if normalized == "false":
            return "drop"
    return None


def parse_structured_caption(raw_text: str) -> StructuredCaption:
    obj, valid_json = _safe_json_loads(raw_text)

    if not isinstance(obj, dict):
        return _invalid_caption(raw_text, valid_json=False)

    has_new_schema = REQUIRED_KEYS.issubset(obj.keys())
    has_legacy_schema = LEGACY_REQUIRED_KEYS.issubset(obj.keys())
    if not has_new_schema and not has_legacy_schema:
        return _invalid_caption(raw_text, raw_obj=obj, valid_json=valid_json)

    decision = _normalize_decision(obj)
    if decision not in {"keep", "drop"}:
        return _invalid_caption(raw_text, raw_obj=obj, valid_json=valid_json)

    if decision == "drop":
        return StructuredCaption(
            category=None,
            supercategory=None,
            attributes=[],
            description="",
            is_clear_object=False,
            decision="drop",
            raw_response=obj,
            valid_json=valid_json,
        )

    category = _normalize_text(obj.get("category", ""))
    supercategory = _normalize_text(obj.get("supercategory", ""))
    attributes = _normalize_attributes(obj.get("attributes", []))
    description = _normalize_description(obj.get("description", ""))

    if category in {"", "unknown", "object", "item", "thing", "part"}:
        return _invalid_caption(raw_text, raw_obj=obj, valid_json=valid_json)

    if description.lower() in {"", "unknown", "unclear object", "object"}:
        return _invalid_caption(raw_text, raw_obj=obj, valid_json=valid_json)

    return StructuredCaption(
        category=category,
        supercategory=supercategory or "other",
        attributes=attributes,
        description=description,
        is_clear_object=True,
        decision="keep",
        raw_response=obj,
        valid_json=valid_json,
    )


def is_explicit_unclear_caption(caption: StructuredCaption) -> bool:
    raw = caption.raw_response if isinstance(caption.raw_response, dict) else {}
    if REQUIRED_KEYS.issubset(raw.keys()):
        value = raw.get("decision")
        return isinstance(value, str) and value.strip().lower() == "drop"

    if not LEGACY_REQUIRED_KEYS.issubset(raw.keys()):
        return False
    value = raw.get("IsClearObject")
    if isinstance(value, bool):
        return value is False
    if isinstance(value, str):
        return value.strip().lower() == "false"
    return False
