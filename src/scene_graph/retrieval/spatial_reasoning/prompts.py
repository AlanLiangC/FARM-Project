"""VLM prompt templates for predicate evaluation."""

from __future__ import annotations

PREDICATE_PROMPTS: dict[str, str] = {
    "Near": (
        "You are evaluating whether two objects are spatially near each other in an indoor scene.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Euclidean distance: {distance:.2f} meters\n\n"
        "In a typical indoor environment, would you consider these objects to be "
        '"near" each other? Consider both the metric distance and the spatial context.\n\n'
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "On": (
        "You are evaluating whether one object is resting on top of another.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Height difference (A.z - B.z): {height_diff:.2f} meters\n"
        "Horizontal distance: {horiz_dist:.2f} meters\n\n"
        "Is Object A physically resting on the surface of Object B?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "Above": (
        "You are evaluating whether one object is above another.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Height difference (A.z - B.z): {height_diff:.2f} meters\n"
        "Horizontal distance: {horiz_dist:.2f} meters\n\n"
        "Is Object A above Object B?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "Below": (
        "You are evaluating whether one object is below another.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Height difference (A.z - B.z): {height_diff:.2f} meters\n"
        "Horizontal distance: {horiz_dist:.2f} meters\n\n"
        "Is Object A below Object B?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "NextTo": (
        "You are evaluating whether two objects are adjacent to each other on the same horizontal plane.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Horizontal distance: {horiz_dist:.2f} meters\n"
        "Height difference: {height_diff:.2f} meters\n\n"
        "Are these objects next to each other (adjacent, side by side)?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "Between": (
        "You are evaluating whether an object is positioned between two other objects.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{ref_a_description}" at position [{ref_a_pos}] (shown in Image 2)\n'
        'Object C: "{ref_b_description}" at position [{ref_b_pos}] (shown in Image 3)\n\n'
        "Distance A-B: {dist_ab:.2f}m, Distance A-C: {dist_ac:.2f}m, Distance B-C: {dist_bc:.2f}m\n\n"
        "Is Object A positioned between Objects B and C?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "Inside": (
        "You are evaluating whether one object is inside or contained by another object.\n\n"
        'Object A: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Object B: "{anchor_description}" at position [{anchor_pos}] (shown in Image 2)\n'
        "Center-to-container distance: {distance:.2f} meters\n\n"
        "Is Object A inside or contained by Object B?\n\n"
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "HasAttribute": (
        "You are evaluating whether an object has a specific visual attribute.\n\n"
        'Object: "{subject_description}" (shown in Image 1)\n'
        'Attribute to verify: "{attribute}"\n\n'
        'Does this object clearly exhibit the attribute "{attribute}"?\n\n'
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "IsCategory": (
        "You are evaluating whether an object belongs to a specific category.\n\n"
        'Object: "{subject_description}" (shown in Image 1)\n'
        'Category to verify: "{category}"\n\n'
        'Is this object a "{category}"?\n\n'
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
    "InRegion": (
        "You are evaluating whether an object is located in a specific room or area.\n\n"
        'Object: "{subject_description}" at position [{subject_pos}] (shown in Image 1)\n'
        'Target region: "{region}"\n'
        'Assigned region label: "{assigned_region}" (confidence: {region_confidence:.0%})\n'
        "Other objects in this region: {region_objects}\n\n"
        'Is this object located in the "{region}"?\n\n'
        'Respond with a JSON object: {{"score": <0-100>, "reasoning": "<brief explanation>"}}'
    ),
}

QUERY_PARSER_PROMPT = """\
You are a spatial query parser. Given a natural language query describing an object \
in a 3D scene, decompose it into a target object and a set of typed predicates.

## Predicate types

- Near(subject, anchor) — spatially close (use for general proximity)
- On(subject, surface) — subject rests on top of surface
- Above(subject, reference) — subject is vertically above reference
- Below(subject, reference) — subject is vertically below reference
- NextTo(subject, reference) — adjacent on same horizontal plane
- Between(subject, ref_a, ref_b) — subject is between two references
- Inside(subject, container) — subject is inside/within/contained by container
- InRegion(subject, region) — subject is inside a named room/area
- HasAttribute(subject, attr) — subject has a visual/semantic attribute
- IsCategory(subject, category) — subject matches a semantic category
- Closest(subject, anchor) — subject is the nearest object to anchor (superlative)
- Farthest(subject, anchor) — subject is the farthest object from anchor (superlative)
- LeftOf(subject, anchor) — subject is to the left of anchor (view-dependent)
- RightOf(subject, anchor) — subject is to the right of anchor (view-dependent)
- InFrontOf(subject, anchor) — subject is in front of anchor (view-dependent)
- Behind(subject, anchor) — subject is behind anchor (view-dependent)

## Conventions

- "$target" refers to the object being searched for.
- Anchor arguments are short descriptions (e.g., "white table", "window").
- A query may have zero or more predicates.
- If the query is just a simple object name with no spatial/attribute constraints, \
return an empty predicate list.
- Modifiers (color, size, material) describing the target object are kept in \
``target_description`` AND extracted as ``HasAttribute`` predicates. The \
``target_description`` is used for semantic-embedding retrieval (richer is \
better); the ``HasAttribute`` predicates feed explicit attribute scoring. \
``target_class`` is the bare class noun, with no modifiers — used for category \
matching against object labels. Modifiers describing anchors stay inside the \
anchor description string.
- Extract ALL constraints from the query — be thorough. Multiple predicates are common.
- Use Near for general "near/by/close to"; use Closest/Farthest ONLY for explicit \
superlatives ("closest", "nearest", "farthest", "furthest").
- Use LeftOf/RightOf/InFrontOf/Behind for explicit directional language.
- Use Above for both "above" and "over"; use Inside for object-in-object language.

## Output fields

Every parsed query must include:

- `target_description`: a clean, short noun phrase the user is looking for \
(e.g., "mug", "chair", "trash can"). Used for semantic retrieval embedding. \
This is what was already extracted in previous versions.
- `target_class`: the **canonical object category** name in singular form, \
suitable for matching against a closed-vocab object label (e.g., "chair", \
"lamp", "trash_can", "monitor"). This is your best guess at the semantic \
category, even if the query is paraphrased or functional. Use snake_case \
for multi-word categories. Examples:
  * "the red chair" → target_class: "chair"
  * "something to sit on" → target_class: "chair"
  * "the broken one" (context unclear) → target_class: null
  * "trash can" → target_class: "trash_can"
- `predicates`: list of typed predicates (see Predicate types above).
- `reasoning`: 1-sentence rationale.

## Examples

Query: "red mug on the white table near the window in the kitchen"
```json
{
  "target_description": "red mug",
  "target_class": "mug",
  "predicates": [
    {"name": "HasAttribute", "args": ["$target", "red"]},
    {"name": "On", "args": ["$target", "white table"]},
    {"name": "Near", "args": ["white table", "window"]},
    {"name": "InRegion", "args": ["$target", "kitchen"]}
  ],
  "reasoning": "Target is a red mug, on a white table, that table is near a window, all in the kitchen."
}
```

Query: "chair between the desk and the bookshelf"
```json
{
  "target_description": "chair",
  "target_class": "chair",
  "predicates": [
    {"name": "Between", "args": ["$target", "desk", "bookshelf"]}
  ],
  "reasoning": "Target is a chair positioned between a desk and a bookshelf."
}
```

Query: "the large wooden bookshelf next to the window"
```json
{
  "target_description": "large wooden bookshelf",
  "target_class": "bookshelf",
  "predicates": [
    {"name": "HasAttribute", "args": ["$target", "large"]},
    {"name": "HasAttribute", "args": ["$target", "wooden"]},
    {"name": "NextTo", "args": ["$target", "window"]}
  ],
  "reasoning": "Target is a large wooden bookshelf next to the window. Size and material modifiers stay in target_description; target_class is the bare noun."
}
```

Query: "the lamp above the couch in the living room"
```json
{
  "target_description": "lamp",
  "target_class": "lamp",
  "predicates": [
    {"name": "Above", "args": ["$target", "couch"]},
    {"name": "InRegion", "args": ["$target", "living room"]}
  ],
  "reasoning": "Target is a lamp that is above a couch, located in the living room."
}
```

Query: "the chair closest to the door"
```json
{
  "target_description": "chair",
  "target_class": "chair",
  "predicates": [
    {"name": "Closest", "args": ["$target", "door"]}
  ],
  "reasoning": "Target is the chair nearest to the door (superlative)."
}
```

Query: "the plant farthest from the window"
```json
{
  "target_description": "plant",
  "target_class": "plant",
  "predicates": [
    {"name": "Farthest", "args": ["$target", "window"]}
  ],
  "reasoning": "Target is the plant farthest from the window (superlative)."
}
```

Query: "the trash can to the left of the printer"
```json
{
  "target_description": "trash can",
  "target_class": "trash_can",
  "predicates": [
    {"name": "LeftOf", "args": ["$target", "printer"]}
  ],
  "reasoning": "Target is a trash can to the left of the printer."
}
```

Query: "the shelf behind the sofa"
```json
{
  "target_description": "shelf",
  "target_class": "shelf",
  "predicates": [
    {"name": "Behind", "args": ["$target", "sofa"]}
  ],
  "reasoning": "Target is a shelf behind the sofa."
}
```

Query: "backpack"
```json
{
  "target_description": "backpack",
  "target_class": "backpack",
  "predicates": [],
  "reasoning": "Simple object lookup with no spatial or attribute constraints."
}
```

## Rules

- Keep "reasoning" to ONE short sentence (max 15 words).
- Output ONLY the JSON object, no extra text.

## Your task

Parse the following query.

Query: "{query}"
"""
