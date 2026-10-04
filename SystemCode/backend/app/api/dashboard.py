from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.dependencies import get_db
from app.schemas.dashboard import DashboardSummaryResponse
from app.services.dashboard_service import DashboardService


router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)


@router.get(
    "",
    response_model=DashboardSummaryResponse,
    summary="Get dashboard overview",
)
def get_dashboard(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:

    return DashboardService.get_dashboard(db)