"""
Surpported Models.
Supports:
- Closed Source: OpenAI-compatible APIs (OpenAIModel)
"""

import os
from abc import ABC, abstractmethod
from openai import OpenAI

class BaseEngine(ABC):
    def __init__(self, model_name_or_path: str):
        self.name = None
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = 1024

    @abstractmethod
    def get_chat_response(self, prompt: str) -> str:
        pass

    def set_hyperparameter(self, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

class OpenAIModel(BaseEngine):
    def __init__(self, model_name_or_path: str, api_key: str = "", base_url="https://api.openai.com/v1"):
        super().__init__(model_name_or_path)
        self.name = "OpenAIModel"
        self.model = model_name_or_path
        self.base_url = base_url
        self.max_tokens = 4096 # Close source model
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": input_text},
            ],
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=None
        )
        return response.choices[0].message.content

class LLMFactory:
    @staticmethod
    def create_llm(model_name_or_path: str, api_key: str = "", base_url: str = "https://api.openai.com/v1") -> BaseEngine:
        return OpenAIModel(model_name_or_path, api_key, base_url)
