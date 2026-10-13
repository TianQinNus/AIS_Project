from sqlalchemy.orm import Session

from app.services.assistant.tools.vehicle_analysis import (
    VehicleAnalysisTool,
)

from app.services.assistant.tools.knowledge_search import (
    KnowledgeSearchTool,
)

class ToolRegistry:

    def __init__(self):

        self._tools = {
            VehicleAnalysisTool.name: VehicleAnalysisTool,
            KnowledgeSearchTool.name: KnowledgeSearchTool,
        }

    @property
    def schemas(self) -> list[dict]:

        result = []

        for tool in self._tools.values():

            schema = tool.schema

            result.append({
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            })

        return result

    def execute(
        self,
        *,
        name: str,
        db: Session,
        arguments: dict,
    ) -> dict:

        tool = self._tools.get(name)

        if tool is None:
            return {
                "success": False,
                "error": f"Unknown tool: {name}",
            }

        return tool.execute(
            db=db,
            **arguments,
        )