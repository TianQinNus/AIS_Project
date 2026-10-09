from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):

    @abstractmethod
    def create_response(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate an LLM response."""
        raise NotImplementedError