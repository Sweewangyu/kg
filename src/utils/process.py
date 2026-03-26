import json
import os
import re

import yaml

from .config_manager import ConfigManager
from .logger import logger


def _resolve_config_relative_path(base_path: str, target_path: str) -> str:
    if os.path.isabs(target_path):
        return target_path
    return os.path.join(os.path.dirname(os.path.abspath(base_path)), target_path)


def _load_generation_triplets(generation_config: dict, yaml_path: str):
    triplets_path = generation_config.get("triplets_json_path", "")
    raw_triplets = generation_config.get("triplets_json", generation_config.get("triplets", []))

    if isinstance(triplets_path, str) and triplets_path.strip():
        resolved_path = _resolve_config_relative_path(yaml_path, triplets_path.strip())
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Triplets JSON file '{resolved_path}' does not exist.")
        with open(resolved_path, "r", encoding="utf-8") as file:
            return json.load(file)

    if raw_triplets in ("", None, []):
        return []
    if isinstance(raw_triplets, (list, dict)):
        return raw_triplets
    if isinstance(raw_triplets, str):
        stripped_triplets = raw_triplets.strip()
        resolved_path = _resolve_config_relative_path(yaml_path, stripped_triplets)
        if os.path.exists(resolved_path) and resolved_path.endswith(".json"):
            with open(resolved_path, "r", encoding="utf-8") as file:
                return json.load(file)
        try:
            return json.loads(stripped_triplets)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "generation.triplets_json must be JSON data, a JSON string, or a path to a JSON file."
            ) from exc
    raise ValueError(
        "generation.triplets_json must be JSON data, a JSON string, or a path to a JSON file."
    )


def load_extraction_config(yaml_path: str) -> dict:
    logger.debug(f"Loading extraction config from {yaml_path}")
    if not os.path.exists(yaml_path):
        logger.error(f"Config file '{yaml_path}' does not exist.")
        raise FileNotFoundError(f"Config file '{yaml_path}' does not exist.")

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    model_config = config.get("model", {})
    raw_task_config = config.get("task", "extraction")
    extraction_config = config.get("extraction", {})
    generation_config = config.get("generation", {})
    agent_config = config.get("agent", {})
    if isinstance(raw_task_config, dict):
        task_type = raw_task_config.get("type", "extraction")
        use_reflection = raw_task_config.get("use_reflection", True)
    else:
        task_type = raw_task_config
        use_reflection = config.get("use_reflection", True)

    raw_triplets = _load_generation_triplets(generation_config, yaml_path)

    return {
        "task": {
            "type": task_type,
            "use_reflection": bool(use_reflection),
        },
        "model": {
            "model_name_or_path": model_config.get("model_name_or_path", ""),
            "api_key": model_config.get("api_key", ""),
            "base_url": model_config.get("base_url", ""),
        },
        "extraction": {
            "text": extraction_config.get("text", ""),
            "use_file": extraction_config.get("use_file", False),
            "file_path": extraction_config.get("file_path", ""),
            "language": extraction_config.get("language", "auto"),
            "show_trajectory": extraction_config.get("show_trajectory", False),
        },
        "generation": {
            "triplets_json": raw_triplets,
            "show_trajectory": generation_config.get("show_trajectory", False),
        },
        "agent": {
            "chunk_char_limit": agent_config.get("chunk_char_limit", 1024),
            "chunk_overlap_sentences": agent_config.get("chunk_overlap_sentences", 2),
        },
    }


def _split_sentences(text: str) -> list[str]:
    normalized_text = re.sub(r"\r\n?", "\n", text)
    segments = re.split(r"(?<=[.!?。！？])(?:\s+|\n+)|\n{2,}", normalized_text)
    sentences = [item.strip() for item in segments if item and item.strip()]
    return sentences or [normalized_text.strip()]


def _get_chunk_settings() -> tuple[int, int]:
    agent_config = ConfigManager.get_config().get("agent", {})
    chunk_char_limit = int(agent_config.get("chunk_char_limit", 1024) or 1024)
    chunk_overlap_sentences = int(agent_config.get("chunk_overlap_sentences", 2) or 0)
    return max(1, chunk_char_limit), max(0, chunk_overlap_sentences)


def chunk_str(text: str) -> list[dict[str, str]]:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return []

    sentences = _split_sentences(cleaned_text)
    if not sentences:
        return [{"text": cleaned_text, "context_text": cleaned_text}]

    limit, overlap_sentences = _get_chunk_settings()
    chunk_ranges: list[tuple[int, int]] = []
    current_start = 0
    current_length = 0

    for index, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        if current_length and current_length + sentence_length > limit:
            chunk_ranges.append((current_start, index))
            current_start = index
            current_length = 0
        current_length += sentence_length

    if current_start < len(sentences):
        chunk_ranges.append((current_start, len(sentences)))

    chunks: list[dict[str, str]] = []
    for start, end in chunk_ranges:
        context_start = max(0, start - overlap_sentences)
        context_end = min(len(sentences), end + overlap_sentences)
        chunk_text = " ".join(sentences[start:end]).strip()
        context_text = " ".join(sentences[context_start:context_end]).strip()
        if not chunk_text:
            continue
        chunks.append(
            {
                "text": chunk_text,
                "context_text": context_text or chunk_text,
            }
        )
    return chunks or [{"text": cleaned_text, "context_text": cleaned_text}]


def load_text_from_file(file_path: str) -> str:
    logger.debug(f"Loading text from file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Input file '{file_path}' does not exist.")
        raise FileNotFoundError(f"Input file '{file_path}' does not exist.")

    if file_path.endswith(".pdf"):
        logger.info(f"Parsing PDF file: {file_path}")
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        content = "".join(page.extract_text() or "" for page in reader.pages).strip()
    elif file_path.endswith(".txt"):
        logger.info(f"Reading TXT file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
    else:
        logger.error(f"Unsupported file format: {file_path}")
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.debug(f"Successfully loaded {len(content)} characters from {file_path}")
    return content


def chunk_file(file_path: str) -> list[dict[str, str]]:
    content = load_text_from_file(file_path)
    return chunk_str(content)


def process_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\w)'|'(?!\w)", '"', text)


def remove_empty_values(data):
    def is_empty(value) -> bool:
        return value is None or value == [] or value == "" or value == {}

    if isinstance(data, dict):
        return {key: remove_empty_values(value) for key, value in data.items() if not is_empty(value)}
    if isinstance(data, list):
        return [remove_empty_values(item) for item in data if not is_empty(item)]
    return data


def extract_json_dict(text):
    if isinstance(text, dict):
        return remove_empty_values(text)
    if not isinstance(text, str):
        return text

    logger.debug("Attempting to extract JSON dict from text...")
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    matches = re.findall(pattern, text)
    if not matches:
        logger.warning("No JSON structure found in text.")
        return text

    json_string = matches[-1]
    try:
        parsed_json = json.loads(json_string)
        logger.debug("Successfully parsed JSON dict.")
        return remove_empty_values(parsed_json)
    except json.JSONDecodeError as e:
        normalized_json_string = process_single_quotes(json_string)
        if normalized_json_string != json_string:
            try:
                parsed_json = json.loads(normalized_json_string)
                logger.debug("Successfully parsed JSON dict after normalizing single quotes.")
                return remove_empty_values(parsed_json)
            except json.JSONDecodeError:
                pass
        logger.error(f"Failed to parse JSON: {e}")
        return json_string
