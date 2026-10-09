from functools import lru_cache

from app.services.assistant.llm.qwen_provider import (
    QwenLLMProvider,
)
from app.services.assistant.orchestrator import (
    AssistantOrchestrator,
)
from app.services.assistant.tools.registry import (
    ToolRegistry,
)


@lru_cache
def get_assistant() -> AssistantOrchestrator:

    llm = QwenLLMProvider()

    tools = ToolRegistry()

    return AssistantOrchestrator(
        llm=llm,
        tool_registry=tools,
    )