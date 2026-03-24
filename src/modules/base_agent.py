from models.llm_def import BaseEngine
from models.prompt_template import PROMPT_REGISTRY
from utils.process import extract_json_dict

class BaseAgent:
    def __init__(self, llm: BaseEngine):
        self.llm = llm

    def invoke_llm(self, mode: str, extract_json=True, **kwargs):
        prompt_template = PROMPT_REGISTRY.get(mode)
        if not prompt_template:
            raise ValueError(f"Prompt template for mode '{mode}' not found.")
            
        prompt = prompt_template.format(**kwargs)
        response = self.llm.get_chat_response(prompt)
        
        if extract_json:
            return extract_json_dict(response)
        return response
