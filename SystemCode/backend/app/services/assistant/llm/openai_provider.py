from openai import OpenAI

from app.core.config import settings
from app.services.assistant.llm.base import (
    LLMProvider,
)


class OpenAILLMProvider(LLMProvider):

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def create_response(
        self,
        *,
        instructions: str,
        input,
        tools: list[dict],
        previous_response_id: str | None = None,
    ):
        kwargs = {
            "model": settings.llm_model,
            "instructions": instructions,
            "input": input,
            "tools": tools,
        }

        if previous_response_id:
            kwargs["previous_response_id"] = (
                previous_response_id
            )

        return self.client.responses.create(
            **kwargs
        )