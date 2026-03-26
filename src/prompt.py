import json
from typing import Any

from pydantic import BaseModel, Field

EXTRACTION_AGENT_PROMPT = """You are an information extraction engine.

Extract entities, entity attributes, and relation triples from the target text using only the allowed schema.

## Schema
{schema}

## Rules
1. Use only entity types, relation labels, and attribute types defined in the schema.
2. Extract:
   - entities
   - attributes of entities
   - relation triples between entities
3. Normalize repeated mentions of the same entity into one canonical name when unambiguous.
4. Use attributes for literal/descriptive values (e.g. age, gender, birthday, nationality, profession, title, color, release date, address, status, number, size, time).
5. Use triples only for relations between two extracted entities.
6. Every triple must use:
   - Head and Tail that both appear in the entity list
   - a Relation label allowed by the schema
7. Do not fabricate facts. Extract only information explicitly stated in the target text.
8. Reference context may be used only to resolve aliases, pronouns, or abbreviated mentions. Do not extract facts supported only by the reference context.
9. Return minimal but complete results with no duplicates.

## Target Text
{text}

## Reference Context
{context_text}

## Output Format
Return JSON only:

{{
  "entities": [
    {{"name": "...", "type": "..."}}
  ],
  "attributes": {{
    "entity_name": {{
      "attribute_type": "value"
    }}
  }},
  "triples": [
    {{"Head": "subject", "Relation": "relation", "Tail": "object"}}
  ]
}}"""


REFLECTION_AGENT_PROMPT = """You are a knowledge graph extraction reviewer.

Review the extraction result against the schema and the source text.

## Input Text
{text}

## Extraction Result
{extraction_result}

## Review Criteria
Evaluate only these aspects:
1. schema compliance
2. faithfulness to the text
3. completeness under the schema
4. entity normalization and deduplication

## Rules
1. Use the schema as the only standard.
2. Do not use any information not supported by the input text.
3. Focus on major errors and important omissions.
4. Be concise and objective.

## Scoring
Give scores from 0 to 100 for:
- schema_compliance
- faithfulness
- completeness
- normalization

Then give an overall_score.

## Output
Return JSON only:

{{
  "schema_compliance": 0,
  "faithfulness": 0,
  "completeness": 0,
  "normalization": 0,
  "overall_score": 0,
  "major_problems": [
    "..."
  ],
  "minor_problems": [
    "..."
  ]
}}"""


REVERSE_GENERATION_AGENT_PROMPT = """You are a precise reverse knowledge graph verbalization engine.

Your job is to express the full information from the given triplets in high-quality, fluent, and natural text.

## Goals
1. Cover all facts in the triplets.
2. Do not add any information that is not present in the triplets.
3. Combine related facts naturally when possible.
4. Keep the text concise, faithful, and readable.
5. Resolve repeated subjects or objects into smooth wording only when the meaning stays exactly the same.

## Triplets To Express
{triplets}

## Output Requirements
Return text only. Do not output JSON. Do not output markdown. Do not explain your reasoning.
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
    schema_compliance: int = Field(default=0)
    faithfulness: int = Field(default=0)
    completeness: int = Field(default=0)
    normalization: int = Field(default=0)
    overall_score: int = Field(default=0)
    major_problems: list[str] = Field(default_factory=list)
    minor_problems: list[str] = Field(default_factory=list)


class GenerationTripleItem(BaseModel):
    subject: str = Field(default="")
    relation: str = Field(default="")
    object: str = Field(default="")


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


def normalize_generation_triplets(raw_triplets: Any) -> list[dict[str, str]]:
    if raw_triplets in ("", None, []):
        return []
    candidate = raw_triplets
    if isinstance(raw_triplets, str):
        candidate = json.loads(raw_triplets)
    if isinstance(candidate, dict):
        candidate = [candidate]
    if not isinstance(candidate, list):
        raise ValueError("Triplets must be a list, dict, or JSON string.")

    normalized_triplets: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in candidate:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", item.get("Head", ""))).strip()
        relation = str(item.get("relation", item.get("Relation", ""))).strip()
        obj = str(item.get("object", item.get("Tail", ""))).strip()
        if not subject or not relation or not obj:
            continue
        key = (subject, relation, obj)
        if key in seen:
            continue
        seen.add(key)
        normalized_triplets.append(
            GenerationTripleItem(subject=subject, relation=relation, object=obj).model_dump()
        )
    return normalized_triplets


def render_generation_triplets(triplets: list[dict[str, str]]) -> str:
    return "\n".join(json.dumps(triplet, ensure_ascii=False) for triplet in triplets)


def build_generation_prompt(triplets: list[dict[str, str]]) -> str:
    return REVERSE_GENERATION_AGENT_PROMPT.format(triplets=render_generation_triplets(triplets))


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


def _coerce_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def coerce_reflection_result(raw_result: Any) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raw_result = {}

    reflection = ReflectionResult.model_validate(
        {
            "schema_compliance": _coerce_score(raw_result.get("schema_compliance", raw_result.get("overall_score", 0))),
            "faithfulness": _coerce_score(raw_result.get("faithfulness", raw_result.get("overall_score", 0))),
            "completeness": _coerce_score(raw_result.get("completeness", raw_result.get("overall_score", 0))),
            "normalization": _coerce_score(raw_result.get("normalization", raw_result.get("overall_score", 0))),
            "overall_score": _coerce_score(raw_result.get("overall_score", raw_result.get("score", 0))),
            "major_problems": _normalize_string_list(raw_result.get("major_problems", raw_result.get("problems", []))),
            "minor_problems": _normalize_string_list(raw_result.get("minor_problems", raw_result.get("suggestions", []))),
        }
    )
    return reflection.model_dump()
