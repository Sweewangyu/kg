from models.llm_def import BaseEngine
from utils.process import extract_json_dict
from utils.logger import logger


class BaseAgent:
    def __init__(self, llm: BaseEngine):
        self.llm = llm

    def invoke_llm(self, prompt: str, extract_json: bool = True):
        logger.debug(f"Invoking LLM with prompt length: {len(prompt)}")
        response = self.llm.get_chat_response(prompt)
        logger.debug(f"Received LLM response length: {len(response)}")
        
        if extract_json:
            logger.debug("Extracting JSON from LLM response...")
            return extract_json_dict(response)
        return response
