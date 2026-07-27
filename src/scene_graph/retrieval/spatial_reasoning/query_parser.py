"""LLM-based query decomposition into a QueryGraph."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from scene_graph.llm_utils import LLMInterface

from .models import Predicate, QueryGraph
from .prompts import QUERY_PARSER_PROMPT

logger = logging.getLogger(__name__)

VALID_PREDICATE_NAMES = frozenset([
    "Near", "On", "Above", "Below", "NextTo", "Between",
    "Inside", "InRegion", "HasAttribute", "IsCategory",
    "Closest", "Farthest", "LeftOf", "RightOf", "InFrontOf", "Behind",
])


def parse_query(query: str, llm: LLMInterface) -> Optional[QueryGraph]:
    """Parse a natural language query into a QueryGraph.

    Returns None if parsing fails or produces no predicates (simple lookup).
    """
    prompt = QUERY_PARSER_PROMPT.replace("{query}", query)
    try:
        # Parser responses are short JSON (~150 tokens).
        original_max_tokens = llm.config.max_tokens
        llm.config.max_tokens = min(512, original_max_tokens)
        try:
            response = llm.query(prompt)
        finally:
            llm.config.max_tokens = original_max_tokens
    except Exception as e:
        logger.warning("Query parser LLM call failed: %s", e)
        return None

    parsed = _extract_json(response)
    if parsed is None:
        logger.warning("Query parser failed to produce valid JSON. Response: %.200s", response)
        return None

    graph = _validate_parsed(parsed)
    if graph is not None:
        graph = fixup_superlative_predicates(query, graph)
    return graph


def fixup_superlative_predicates(query: str, graph: QueryGraph) -> QueryGraph:
    """Post-parser fixup: convert Near -> Closest/Farthest based on query wording.

    If the raw query contains superlative language (closest, nearest, farthest,
    furthest), convert matching Near predicates to their superlative equivalents.
    """
    query_lower = query.lower()
    has_closest = any(w in query_lower for w in ("closest", "nearest"))
    has_farthest = any(w in query_lower for w in ("farthest", "furthest"))

    if not has_closest and not has_farthest:
        return graph

    new_predicates: List[Predicate] = []
    for pred in graph.predicates:
        if pred.name == "Near" and has_closest:
            new_predicates.append(Predicate(name="Closest", args=pred.args, kwargs=pred.kwargs))
        elif pred.name == "Near" and has_farthest:
            new_predicates.append(Predicate(name="Farthest", args=pred.args, kwargs=pred.kwargs))
        else:
            new_predicates.append(pred)

    return QueryGraph(
        target_description=graph.target_description,
        predicates=new_predicates,
        reasoning=graph.reasoning,
    )


def _extract_json(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _validate_parsed(data: Dict[str, Any]) -> Optional[QueryGraph]:
    """Validate parsed JSON and convert to QueryGraph."""
    target = data.get("target_description", "").strip()
    if not target:
        return None

    raw_predicates = data.get("predicates", [])
    if not isinstance(raw_predicates, list):
        return None

    if not raw_predicates:
        return None

    predicates: List[Predicate] = []
    for raw in raw_predicates:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name", "")
        if name not in VALID_PREDICATE_NAMES:
            continue
        args = raw.get("args", [])
        if not isinstance(args, list) or len(args) < 2:
            continue
        args = [str(a) for a in args]
        kwargs = raw.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        predicates.append(Predicate(name=name, args=args, kwargs=kwargs))

    if not predicates:
        return None

    reasoning = str(data.get("reasoning", ""))
    # Track B (2026-05-17): structured target_class extracted
    # by the LLM alongside the noun-phrase target_description. None when the
    # LLM is unsure (e.g., paraphrased queries with no obvious class) — the
    # executor's soft-class path falls back to regex tokenization of
    # target_description.
    raw_target_class = data.get("target_class")
    target_class: Optional[str] = None
    if isinstance(raw_target_class, str) and raw_target_class.strip():
        cleaned = raw_target_class.strip()
        if cleaned.lower() not in {"null", "none", "unknown", "n/a"}:
            target_class = cleaned
    return QueryGraph(
        target_description=target,
        predicates=predicates,
        reasoning=reasoning,
        target_class=target_class,
    )
