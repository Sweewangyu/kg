import time

from models.llm_def import BaseEngine
from utils.process import extract_json_dict
from utils.logger import logger


class BaseAgent:
    def __init__(self, llm: BaseEngine):
        self.llm = llm

    def invoke_llm(self, prompt: str, extract_json: bool = True, agent_name: str = "agent"):
        logger.debug(f"Invoking LLM with prompt length: {len(prompt)}")
        max_retries = 3
        response = ""
        
        for attempt in range(max_retries):
            try:
                response = self.llm.get_chat_response(prompt)
                logger.debug(f"Received LLM response length: {len(response)} (Attempt {attempt + 1}/{max_retries})")
                logger.info(f"{agent_name} raw output:\n{response}")
                
                if response.strip():
                    break
                    
                logger.warning(f"Received empty response from LLM (Attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"LLM call failed (Attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(2)
                
        if not response.strip():
            logger.error(f"Failed to get a valid response from LLM after {max_retries} attempts.")
        
        if extract_json:
            logger.debug("Extracting JSON from LLM response...")
            return extract_json_dict(response)
        return response
