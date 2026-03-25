from models.llm_def import BaseEngine
from models.prompt_template import get_prompt
from utils.process import extract_json_dict

class BaseAgent:
    def __init__(self, llm: BaseEngine):
        self.llm = llm

    def invoke_llm(self, mode: str, extract_json=True, **kwargs):
        prompt = get_prompt(mode, **kwargs)
        response = self.llm.get_chat_response(prompt)
        
        if extract_json:
            return extract_json_dict(response)
        return response
