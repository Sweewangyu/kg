import json

from models import BaseEngine
from modules.extraction_agent import ExtractionAgent
from modules.reflection_agent import ReflectionAgent
from prompt import normalize_schema
from utils.data_def import DataPoint, ExtractionRequest
from utils.process import chunk_str, load_text_from_file
from utils.logger import logger


class Pipeline:
    def __init__(self, llm: BaseEngine):
        self.extraction_agent = ExtractionAgent(llm=llm)
        self.reflection_agent = ReflectionAgent(llm=llm)

    def _build_request(
        self,
        text: str,
        use_file: bool,
        file_path: str,
        show_trajectory: bool,
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
            print("Extraction Trajectory:\n", json.dumps(data.get_result_trajectory(), indent=2, ensure_ascii=False))
        print("Extraction Result:\n", json.dumps(data.pred, indent=2, ensure_ascii=False))

    def get_extract_result(
        self,
        text: str = "",
        use_file: bool = False,
        file_path: str = "",
        show_trajectory: bool = False,
    ):
        logger.info("Starting extraction pipeline.")
        request = self._build_request(text, use_file, file_path, show_trajectory)
        data = DataPoint(request=request)
        data = self._prepare(data)
        
        logger.info("Running extraction agent...")
        data = self.extraction_agent.extract(data)
        
        logger.info("Running reflection agent...")
        data = self.reflection_agent.reflect(data)
        
        logger.info("Extraction pipeline finished.")
        self._show_result(data)
        return data.pred, data.get_result_trajectory()
