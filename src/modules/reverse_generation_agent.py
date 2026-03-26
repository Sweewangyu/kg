from models.llm_def import BaseEngine
from prompt import build_generation_prompt, normalize_generation_triplets
from utils.data_def import DataPoint
from utils.logger import logger

from .base_agent import BaseAgent


class ReverseGenerationAgent(BaseAgent):
    def __init__(self, llm: BaseEngine):
        super().__init__(llm)

    def generate(self, data: DataPoint) -> DataPoint:
        triplets = normalize_generation_triplets(data.request.triplets)
        if not triplets:
            raise ValueError("triplets are required for reverse generation.")

        logger.debug("Building reverse generation prompt...")
        prompt = build_generation_prompt(triplets)

        logger.info("Invoking LLM for reverse generation...")
        generated_text = str(
            self.invoke_llm(
                prompt,
                extract_json=False,
                agent_name="reverse_generation_agent",
            )
        ).strip()

        final_result = {
            "triples": triplets,
            "text": generated_text,
        }
        data.set_generation_triplets(triplets)
        data.set_generated_text(generated_text)
        data.set_pred(final_result)
        data.update_trajectory(
            "reverse_generation_agent",
            {
                "triples": triplets,
                "text": generated_text,
            },
        )

        logger.info("Reverse generation completed.")
        return data
