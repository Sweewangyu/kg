import os
from abc import ABC, abstractmethod

from utils.logger import logger


class BaseEngine(ABC):

    def __init__(self, model_name_or_path: str):
        self.name = None
        self.temperature = 0.2
        self.top_p = 0.9

    @abstractmethod
    def get_chat_response(self, prompt: str) -> str:
        raise NotImplementedError

    def set_hyperparameter(self, temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1024):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens


class OpenAIModel(BaseEngine):
    def __init__(self, model_name_or_path: str, api_key: str = "", base_url="https://api.openai.com/v1"):
        super().__init__(model_name_or_path)
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required.")
        self.name = "OpenAIModel"
        self.model = model_name_or_path
        self.base_url = base_url
        self.max_tokens = 4096
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_request_kwargs(self) -> dict:
        request_kwargs: dict = {}
        if "siliconflow" in self.base_url and "Qwen/Qwen3" in self.model:
            request_kwargs["extra_body"] = {"enable_thinking": False}
        return request_kwargs

    def get_chat_response(self, input_text: str) -> str:
        logger.debug(f"Sending request to OpenAI API with model {self.model} (max_tokens: {self.max_tokens}, temperature: {self.temperature})")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": input_text},
                ],
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=None,
                **self._build_request_kwargs(),
            )
            message = response.choices[0].message
            content = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None) or ""
            logger.debug(
                "Received response from OpenAI API. finish_reason=%s, content_length=%s, reasoning_length=%s",
                response.choices[0].finish_reason,
                len(content),
                len(reasoning_content),
            )
            if not content.strip() and reasoning_content.strip():
                logger.warning(
                    "Model returned empty content but non-empty reasoning_content. Consider disabling thinking mode or increasing max_tokens."
                )
            logger.debug("Received successful response from OpenAI API.")
            return content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise


class LLMFactory:
    @staticmethod
    def create_llm(model_name_or_path: str, api_key: str = "", base_url: str = "https://api.openai.com/v1") -> BaseEngine:
        logger.info(f"Creating LLM instance for {model_name_or_path} at {base_url}")
        return OpenAIModel(model_name_or_path, api_key, base_url)
