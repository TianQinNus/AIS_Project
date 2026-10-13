import json

from sqlalchemy.orm import Session

from app.services.assistant.llm.base import LLMProvider
from app.services.assistant.prompts import SYSTEM_PROMPT
from app.services.assistant.tools.registry import ToolRegistry


class AssistantOrchestrator:

    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        *,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm
        self.tool_registry = tool_registry

    def answer(
        self,
        *,
        db: Session,
        message: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ]

        for _ in range(self.MAX_TOOL_ROUNDS):

            response = self.llm.create_response(
                messages=messages,
                tools=self.tool_registry.schemas,
            )

            assistant_message = response.choices[0].message

            # Save assistant message, including tool calls
            messages.append(
                assistant_message.model_dump(
                    exclude_none=True
                )
            )

            tool_calls = assistant_message.tool_calls or []

            # No tool calls: return final answer
            if not tool_calls:
                return assistant_message.content or ""

            # Execute requested tools
            for tool_call in tool_calls:

                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be an object."
                        )

                    result = self.tool_registry.execute(
                        name=tool_name,
                        db=db,
                        arguments=arguments,
                    )

                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as exc:

                    result = {
                        "success": False,
                        "error": f"Invalid tool arguments: {exc}",
                    }

                # Send tool result back to model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                })

        return (
            "I could not complete the request "
            "because the maximum number of tool "
            "rounds was reached."
        )