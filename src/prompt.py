import json
from typing import Any

from pydantic import BaseModel, Field


EXTRACTION_AGENT_PROMPT = """You are a precise knowledge graph extraction engine.
Your task is to jointly extract entities, attributes, and relation triples from the target text in one pass.

You must STRICTLY follow the schema below and must not output any type, relation, or attribute outside the schema.

## Schema
{schema}

## Task
From the target text, extract:
1. entities
2. attributes of entities
3. relation triples between entities

## Strict Rules
1. Each entity must be an object in the form:
   {{"name": "...", "type": "..."}}
   - "name" must be normalized, concise, and consistent.
   - "type" must be one of the allowed node types in the schema.

2. Each triple must be in the form:
   {{"Head": "subject", "Relation": "relation", "Tail": "object"}}
   - "Relation" must be one of the allowed relations in the schema.
   - "Head" and "Tail" must both appear in the entity list.
   - Do not create triples with values that should instead be attributes.

3. Attributes must be extracted only from the allowed attribute list in the schema.
   Format:
   {{
     "entity_name": ["attribute_type: value", ...]
   }}

4. Attributes and triples must be complementary and not redundant:
   - Use attributes for descriptive properties such as age, gender, birthday, nationality, profession, title, color, release date, address, status, number, size, etc.
   - Use triples only for relations between two entities.
   - Example: profession should be an attribute, not a triple.
   - Example: birthday should be an attribute, not a triple.

5. Do not fabricate information.
   - Only extract information explicitly stated or strongly and unambiguously expressed in the target text.
   - If uncertain, omit it.

6. Normalize repeated mentions of the same entity.
   - Merge aliases, abbreviations, pronouns, and shortened mentions into one canonical entity name when unambiguous.

7. Only use the exact relation labels provided in the schema.
   - Do not invent relation names.
   - Do not convert attributes into relations.
   - Do not output type information as triples.

8. Time expressions must NOT be created as standalone entities unless the text clearly treats them as entities under one of the allowed node types.
   - Normally, time/date/year should be represented using the "time" attribute or "release date" attribute when appropriate.

9. For family relations:
   - Only use "father of/ mother of / spouse of/ grandparent of" when the text explicitly states the relationship.
   - Do not infer unstated family relations.

10. Keep the output minimal but complete:
   - Include all important entities, valid attributes, and supported triples.
   - Avoid duplicates.

11. You may use the reference context only to resolve aliases, pronouns, or abbreviated mentions.
   - Do not output any entity, attribute, or triple that is supported only by the reference context.
   - Every extracted fact must be grounded in the target text span.

## Target Text
{text}

## Reference Context
{context_text}

## Output Format
Return JSON only. Do not output markdown. Do not output explanations.

{{
  "entities": [
    {{"name": "...", "type": "..."}}
  ],
  "attributes": {{
    "entity_name": ["attribute_type: value"]
  }},
  "triples": [
    {{"Head": "subject", "Relation": "relation", "Tail": "object"}}
  ]
}}
"""


REFLECTION_AGENT_PROMPT = """You are a knowledge graph extraction reviewer.

Your job is to review the extraction result produced by the extraction agent, score its quality, identify problems, and provide a corrected version if needed.

You must STRICTLY use the schema below as the only standard.

## Schema
{schema}

## Input Text
{text}

## Extraction Result To Review
{extraction_result}

## Review Goals
Check the extraction result carefully from the following aspects:

1. Schema Compliance
   - Are all entity types within the allowed node types?
   - Are all relations within the allowed relation types?
   - Are all attributes within the allowed attribute types?
   - Are there any unsupported labels, fields, or invented categories?

2. Entity Quality
   - Are important entities missing?
   - Are there duplicate entities referring to the same real-world entity?
   - Are entity names normalized consistently and concisely?
   - Are entity types correct?

3. Triple Quality
   - Do all triples use valid relation names from the schema?
   - Do Head and Tail both exist in the entity list?
   - Are the triples faithful to the text?
   - Are any triples actually attributes and therefore incorrectly represented?
   - Are any important supported triples missing?

4. Attribute Quality
   - Are attributes attached to the correct entity?
   - Are attribute names valid according to the schema?
   - Are attribute values faithful to the text?
   - Are any attributes missing?
   - Are any attributes redundant with triples?

5. Faithfulness
   - Does the result contain hallucinated information not supported by the text?
   - Is there any over-inference beyond the text?

6. Completeness
   - Considering only the provided schema, does the result capture the main extractable information from the text?
   - Are there obvious omissions of major entities, attributes, or triples?

## Scoring
Give an overall quality score from 0 to 100.

Scoring guideline:
- 90-100: Highly accurate, schema-compliant, complete, almost no issues
- 75-89: Good overall, minor issues or a few omissions
- 50-74: Noticeable problems in correctness, compliance, or completeness
- 0-49: Serious problems, many errors, unsupported schema usage, or major omissions

## Output Requirements
Return JSON only. No markdown. No explanations outside JSON.

Output format:
{{
  "score": 0,
  "problems": [
    "problem 1",
    "problem 2"
  ],
  "suggestions": [
    "suggestion 1",
    "suggestion 2"
  ],
  "revised_json": {{
    "entities": [
      {{"name": "...", "type": "..."}}
    ],
    "attributes": {{
      "entity_name": ["attribute_type: value"]
    }},
    "triples": [
      {{"Head": "subject", "Relation": "relation", "Tail": "object"}}
    ]
  }}
}}

## Important Constraints
1. If the extraction result is already good, keep revised_json identical or nearly identical to the input extraction result.
2. Do not introduce information not supported by the text.
3. revised_json must strictly comply with the schema.
4. Do not output any unsupported entity type, relation, or attribute.
5. Prefer precise corrections over excessive rewriting.
"""


DEFAULT_SCHEMA = {
    "Nodes": [
        "person",
        "location",
        "organization",
        "event",
        "item",
        "brand",
        "award",
        "game",
        "work",
        "film",
    ],
    "Relations": [
        "belong to",
        "located in",
        "used for",
        "take place in",
        "based on",
        "participate in",
        "establish",
        "organized by",
        "owner of",
        "is member of",
        "similar to",
        "better than",
        "from",
        "older than",
        "beat",
        "play",
        "create",
        "become",
        "follow",
        "represent",
        "near",
        "close to",
        "father of",
        "mother of",
        "spouse of",
        "grandparent of",
    ],
    "Attributes": [
        "time",
        "name",
        "career",
        "number",
        "gender",
        "size",
        "title",
        "address",
        "age",
        "nationality",
        "color",
        "status",
        "height",
        "birthday",
        "birthplace",
        "deathplace",
        "deathday",
        "live place",
        "work place",
        "educated place",
        "profession",
        "partner",
        "child",
        "known for",
        "release date",
    ],
}


class EntityItem(BaseModel):
    name: str = Field(default="")
    type: str = Field(default="")


class TripleItem(BaseModel):
    Head: str = Field(default="")
    Relation: str = Field(default="")
    Tail: str = Field(default="")


class ExtractionResult(BaseModel):
    entities: list[EntityItem] = Field(default_factory=list)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    triples: list[TripleItem] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    score: int = Field(default=0)
    problems: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    revised_json: ExtractionResult = Field(default_factory=ExtractionResult)


def _normalize_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_schema(raw_schema: Any) -> dict[str, list[str]]:
    if raw_schema in ("", None, []):
        candidate = DEFAULT_SCHEMA
    elif isinstance(raw_schema, str):
        candidate = json.loads(raw_schema)
    elif isinstance(raw_schema, dict):
        candidate = raw_schema
    else:
        raise ValueError("Schema must be a dict, JSON string, or empty.")

    schema = {
        "Nodes": _normalize_string_list(candidate.get("Nodes") or candidate.get("nodes")),
        "Relations": _normalize_string_list(candidate.get("Relations") or candidate.get("relations")),
        "Attributes": _normalize_string_list(candidate.get("Attributes") or candidate.get("attributes")),
    }
    if not schema["Nodes"] or not schema["Relations"] or not schema["Attributes"]:
        raise ValueError("Schema must contain non-empty Nodes, Relations, and Attributes.")
    return schema


def render_schema(schema: dict[str, list[str]]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_extraction_prompt(text: str, schema: dict[str, list[str]], context_text: str = "") -> str:
    return EXTRACTION_AGENT_PROMPT.format(
        text=text,
        context_text=context_text or text,
        schema=render_schema(schema),
    )


def build_reflection_prompt(
    text: str,
    schema: dict[str, list[str]],
    extraction_result: dict[str, Any],
) -> str:
    return REFLECTION_AGENT_PROMPT.format(
        text=text,
        schema=render_schema(schema),
        extraction_result=json.dumps(extraction_result, ensure_ascii=False, indent=2),
    )


def coerce_extraction_result(raw_result: Any) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raw_result = {}

    entities: list[dict[str, str]] = []
    entity_seen: set[tuple[str, str]] = set()
    for item in raw_result.get("entities", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "")).strip()
        if not name or not entity_type:
            continue
        key = (name, entity_type)
        if key in entity_seen:
            continue
        entity_seen.add(key)
        entities.append({"name": name, "type": entity_type})

    attributes: dict[str, list[str]] = {}
    raw_attributes = raw_result.get("attributes", {})
    if isinstance(raw_attributes, dict):
        for entity_name, values in raw_attributes.items():
            normalized_entity = str(entity_name).strip()
            normalized_values = _normalize_string_list(values)
            if normalized_entity and normalized_values:
                attributes[normalized_entity] = normalized_values

    triples: list[dict[str, str]] = []
    triple_seen: set[tuple[str, str, str]] = set()
    for item in raw_result.get("triples", []):
        if not isinstance(item, dict):
            continue
        head = str(item.get("Head", "")).strip()
        relation = str(item.get("Relation", "")).strip()
        tail = str(item.get("Tail", "")).strip()
        if not head or not relation or not tail:
            continue
        key = (head, relation, tail)
        if key in triple_seen:
            continue
        triple_seen.add(key)
        triples.append({"Head": head, "Relation": relation, "Tail": tail})

    return ExtractionResult.model_validate(
        {
            "entities": entities,
            "attributes": attributes,
            "triples": triples,
        }
    ).model_dump()


def merge_extraction_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged_entities: list[dict[str, str]] = []
    entity_types: dict[str, str] = {}
    merged_attributes: dict[str, list[str]] = {}
    merged_triples: list[dict[str, str]] = []
    triple_seen: set[tuple[str, str, str]] = set()

    for result in results:
        normalized = coerce_extraction_result(result)
        for entity in normalized["entities"]:
            entity_name = entity["name"]
            if entity_name in entity_types:
                continue
            entity_types[entity_name] = entity["type"]
            merged_entities.append(entity)

        for entity_name, values in normalized["attributes"].items():
            bucket = merged_attributes.setdefault(entity_name, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)

        for triple in normalized["triples"]:
            key = (triple["Head"], triple["Relation"], triple["Tail"])
            if key in triple_seen:
                continue
            triple_seen.add(key)
            merged_triples.append(triple)

    return coerce_extraction_result(
        {
            "entities": merged_entities,
            "attributes": merged_attributes,
            "triples": merged_triples,
        }
    )


def enforce_schema_compliance(raw_result: Any, schema: dict[str, list[str]]) -> dict[str, Any]:
    result = coerce_extraction_result(raw_result)
    allowed_nodes = set(schema["Nodes"])
    allowed_relations = set(schema["Relations"])
    allowed_attributes = set(schema["Attributes"])

    entities: list[dict[str, str]] = []
    entity_names: set[str] = set()
    for entity in result["entities"]:
        if entity["type"] not in allowed_nodes or entity["name"] in entity_names:
            continue
        entity_names.add(entity["name"])
        entities.append(entity)

    attributes: dict[str, list[str]] = {}
    for entity_name, values in result["attributes"].items():
        if entity_name not in entity_names:
            continue
        filtered_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            attribute_type, separator, attribute_value = value.partition(":")
            normalized_type = attribute_type.strip()
            normalized_value = attribute_value.strip()
            normalized_item = f"{normalized_type}: {normalized_value}" if separator else value.strip()
            if (
                not separator
                or not normalized_type
                or not normalized_value
                or normalized_type not in allowed_attributes
                or normalized_item in seen
            ):
                continue
            seen.add(normalized_item)
            filtered_values.append(normalized_item)
        if filtered_values:
            attributes[entity_name] = filtered_values

    triples: list[dict[str, str]] = []
    triple_seen: set[tuple[str, str, str]] = set()
    for triple in result["triples"]:
        key = (triple["Head"], triple["Relation"], triple["Tail"])
        if (
            triple["Relation"] not in allowed_relations
            or triple["Head"] not in entity_names
            or triple["Tail"] not in entity_names
            or key in triple_seen
        ):
            continue
        triple_seen.add(key)
        triples.append(triple)

    return ExtractionResult.model_validate(
        {
            "entities": entities,
            "attributes": attributes,
            "triples": triples,
        }
    ).model_dump()


def coerce_reflection_result(raw_result: Any, extraction_result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raw_result = {}

    score = raw_result.get("score", 0)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    reflection = ReflectionResult.model_validate(
        {
            "score": score,
            "problems": _normalize_string_list(raw_result.get("problems", [])),
            "suggestions": _normalize_string_list(raw_result.get("suggestions", [])),
            "revised_json": coerce_extraction_result(raw_result.get("revised_json", extraction_result)),
        }
    )
    return reflection.model_dump()
