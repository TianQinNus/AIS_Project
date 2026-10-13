from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.services.assistant.llm.base import LLMProvider


class QwenLLMProvider(LLMProvider):

    def __init__(self):

        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            timeout=60.0,
            max_retries=2,
        )

        self.model = settings.llm_model

    def create_response(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "extra_body": {
                "enable_thinking": settings.llm_enable_thinking
            },
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return self.client.chat.completions.create(
            **kwargs
        )