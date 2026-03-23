"""
Surpported Models.
Supports:
- Closed Source: OpenAI-compatible APIs (OpenAIModel)
"""

import os
from openai import OpenAI

class BaseEngine:
    def __init__(self, model_name_or_path: str):
        self.name = None
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = 1024

    def get_chat_response(self, prompt):
        raise NotImplementedError

    def set_hyperparameter(self, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

class OpenAIModel(BaseEngine):
    def __init__(self, model_name_or_path: str, api_key: str = "", base_url="https://api.openai.com/v1"):
        self.name = "OpenAIModel"
        self.model = model_name_or_path
        self.base_url = base_url
        self.temperature = 0.2
        self.top_p = 0.9
        self.max_tokens = 4096 # Close source model
        if api_key != "":
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_chat_response(self, input):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": input},
            ],
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=None
        )
        return response.choices[0].message.content
