import json
from typing import Any

from models import BaseEngine
from modules.extraction_agent import ExtractionAgent
from modules.reflection_agent import ReflectionAgent
from modules.reverse_generation_agent import ReverseGenerationAgent
from prompt import normalize_generation_triplets, normalize_schema
from utils.data_def import DataPoint, ExtractionRequest
from utils.process import chunk_str, load_text_from_file
from utils.logger import logger


class Pipeline:
    def __init__(self, llm: BaseEngine):
        self.extraction_agent = ExtractionAgent(llm=llm)
        self.reflection_agent = ReflectionAgent(llm=llm)
        self.reverse_generation_agent = ReverseGenerationAgent(llm=llm)

    def _build_extract_request(
        self,
        text: str,
        use_file: bool,
        file_path: str,
        show_trajectory: bool,
        use_reflection: bool,
    ) -> ExtractionRequest:
        if use_file and not file_path:
            raise ValueError("file_path is required when use_file is True.")
        if not use_file and not text.strip():
            raise ValueError("text is required when use_file is False.")
        return ExtractionRequest(
            text=text,
            use_file=use_file,
            file_path=file_path,
            show_trajectory=show_trajectory,
            use_reflection=use_reflection,
        )

    def _build_generation_request(self, triplets: Any, show_trajectory: bool) -> ExtractionRequest:
        normalized_triplets = normalize_generation_triplets(triplets)
        if not normalized_triplets:
            raise ValueError("triplets are required for reverse generation.")
        return ExtractionRequest(
            show_trajectory=show_trajectory,
            triplets=normalized_triplets,
        )

    def _prepare(self, data: DataPoint) -> DataPoint:
        logger.info(f"Preparing data for extraction. use_file={data.request.use_file}")
        source_text = load_text_from_file(data.request.file_path) if data.request.use_file else data.request.text
        if not source_text.strip():
            logger.error("No text content available for extraction.")
            raise ValueError("No text content available for extraction.")
        
        logger.debug(f"Source text loaded, length: {len(source_text)}")
        data.set_source_text(source_text)
        data.set_chunks(chunk_str(source_text))
        data.set_schema(normalize_schema(None))
        data.update_trajectory(
            "prepare",
            {
                "schema": data.schema,
                "chunks": data.chunks,
            },
        )
        return data

    def _show_result(self, data: DataPoint) -> None:
        if data.request.show_trajectory:
            print("Pipeline Trajectory:\n", json.dumps(data.get_result_trajectory(), indent=2, ensure_ascii=False))
        print("Pipeline Result:\n", json.dumps(data.pred, indent=2, ensure_ascii=False))

    def get_extract_result(
        self,
        text: str = "",
        use_file: bool = False,
        file_path: str = "",
        show_trajectory: bool = False,
        use_reflection: bool = True,
    ):
        logger.info("Starting extraction pipeline.")
        request = self._build_extract_request(text, use_file, file_path, show_trajectory, use_reflection)
        data = DataPoint(request=request)
        data = self._prepare(data)
        
        logger.info("Running extraction agent...")
        data = self.extraction_agent.extract(data)

        if use_reflection:
            logger.info("Running reflection agent...")
            data = self.reflection_agent.reflect(data)
        else:
            logger.info("Skipping reflection agent.")
            review_result = {
                "enabled": False,
                "skipped": True,
            }
            final_result = {
                "chunks": data.chunk_results,
                "document": data.extraction_result,
                "review": review_result,
            }
            data.set_review_result(review_result)
            data.set_pred(final_result)
            data.update_trajectory(
                "reflection_agent",
                {
                    "skipped": True,
                    "final_result": data.pred,
                },
            )
        
        logger.info("Extraction pipeline finished.")
        self._show_result(data)
        return data.pred, data.get_result_trajectory()

    def get_generation_result(
        self,
        triplets: Any,
        show_trajectory: bool = False,
    ):
        logger.info("Starting reverse generation pipeline.")
        request = self._build_generation_request(triplets, show_trajectory)
        data = DataPoint(request=request)

        logger.info("Running reverse generation agent...")
        data = self.reverse_generation_agent.generate(data)

        logger.info("Reverse generation pipeline finished.")
        self._show_result(data)
        return data.pred, data.get_result_trajectory()
