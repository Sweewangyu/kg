from models.llm_def import BaseEngine
from prompt import build_extraction_prompt, coerce_extraction_result, merge_extraction_results
from utils.data_def import DataPoint
from utils.logger import logger

from .base_agent import BaseAgent


class ExtractionAgent(BaseAgent):
    def __init__(self, llm: BaseEngine):
        super().__init__(llm)

    def extract(self, data: DataPoint) -> DataPoint:
        chunks = data.chunks or [{"text": data.source_text, "context_text": data.source_text}]
        chunk_results: list[dict] = []

        for index, chunk in enumerate(chunks, start=1):
            logger.debug(f"Building extraction prompt for chunk {index}/{len(chunks)}...")
            prompt = build_extraction_prompt(
                text=chunk["text"],
                context_text=chunk.get("context_text", chunk["text"]),
                schema=data.schema,
            )

            logger.info(f"Invoking LLM for extraction on chunk {index}/{len(chunks)}...")
            result = self.invoke_llm(prompt)
            normalized_result = coerce_extraction_result(result)
            chunk_results.append(
                {
                    "text": chunk["text"],
                    "entities": normalized_result["entities"],
                    "attributes": normalized_result["attributes"],
                    "triples": normalized_result["triples"],
                }
            )

        logger.debug("Merging chunk extraction results...")
        data.set_chunk_results(chunk_results)
        data.set_extraction_result(merge_extraction_results(chunk_results))
        data.update_trajectory(
            "extraction_agent",
            {
                "chunks": data.chunk_results,
                "merged": data.extraction_result,
            },
        )
        
        logger.info("Extraction completed.")
        return data
