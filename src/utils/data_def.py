from dataclasses import dataclass, field
from typing import Any

from .process import extract_json_dict

@dataclass
class ExtractionRequest:
    text: str = ""
    use_file: bool = False
    file_path: str = ""
    show_trajectory: bool = False
    use_reflection: bool = True
    triplets: Any = field(default_factory=list)


@dataclass
class DataPoint:
    request: ExtractionRequest
    source_text: str = ""
    chunks: list[dict[str, str]] = field(default_factory=list)
    chunk_results: list[dict[str, Any]] = field(default_factory=list)
    generation_triplets: list[dict[str, str]] = field(default_factory=list)
    generated_text: str = ""
    schema: dict[str, list[str]] = field(default_factory=dict)
    extraction_result: dict[str, Any] = field(default_factory=dict)
    review_result: dict[str, Any] = field(default_factory=dict)
    result_trajectory: dict[str, Any] = field(default_factory=dict)
    pred: Any = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.request.text

    @property
    def use_file(self) -> bool:
        return self.request.use_file

    @property
    def file_path(self) -> str:
        return self.request.file_path

    @property
    def triplets(self) -> Any:
        return self.request.triplets

    def set_source_text(self, text: str) -> None:
        self.source_text = text

    def set_chunks(self, chunks: list[dict[str, str]]) -> None:
        self.chunks = chunks

    def set_chunk_results(self, results: list[dict[str, Any]]) -> None:
        self.chunk_results = [extract_json_dict(result) for result in results]

    def set_generation_triplets(self, triplets: list[dict[str, str]]) -> None:
        self.generation_triplets = [extract_json_dict(triplet) for triplet in triplets]

    def set_generated_text(self, text: str) -> None:
        self.generated_text = str(text).strip()

    def set_schema(self, schema: dict[str, list[str]]) -> None:
        self.schema = schema

    def set_extraction_result(self, result: Any) -> None:
        self.extraction_result = extract_json_dict(result)

    def set_review_result(self, result: Any) -> None:
        self.review_result = extract_json_dict(result)

    def set_pred(self, pred: Any) -> None:
        self.pred = extract_json_dict(pred)

    def update_trajectory(self, step_name: str, result: Any) -> None:
        self.result_trajectory[step_name] = result

    def get_result_trajectory(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_text": self.source_text,
            "chunks": self.chunks,
            "chunk_results": self.chunk_results,
            "triplets": self.generation_triplets,
            "generated_text": self.generated_text,
            "schema": self.schema,
            "trajectory": self.result_trajectory,
            "pred": self.pred,
        }
