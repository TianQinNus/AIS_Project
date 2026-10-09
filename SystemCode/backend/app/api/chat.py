from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from app.services.assistant.dependencies import (
    get_assistant,
)
from app.services.assistant.orchestrator import (
    AssistantOrchestrator,
)


router = APIRouter(
    prefix="/chat",
    tags=["AI Assistant"],
)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with FleetCare AI",
)
def chat(
    request: ChatRequest,

    db: Session = Depends(get_db),

    assistant: AssistantOrchestrator = Depends(
        get_assistant
    ),
) -> ChatResponse:

    answer = assistant.answer(
        db=db,
        message=request.message,
    )

    return ChatResponse(
        answer=answer
    )