from models.llm_def import BaseEngine
from prompt import build_reflection_prompt, coerce_reflection_result
from utils.data_def import DataPoint
from utils.logger import logger

from .base_agent import BaseAgent


class ReflectionAgent(BaseAgent):
    def __init__(self, llm: BaseEngine):
        super().__init__(llm)

    def reflect(self, data: DataPoint) -> DataPoint:
        logger.debug("Building reflection prompt...")
        prompt = build_reflection_prompt(
            text=data.source_text,
            schema=data.schema,
            extraction_result=data.extraction_result,
        )
        
        logger.info("Invoking LLM for reflection...")
        result = self.invoke_llm(prompt, agent_name="reflection_agent")
        
        logger.debug("Coercing reflection result...")
        review_result = coerce_reflection_result(result)

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
                "review": data.review_result,
                "final_result": data.pred,
            },
        )

        logger.info("Reflection completed.")
        return data
