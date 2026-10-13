from pydantic import BaseModel


class FleetOverview(BaseModel):
    total: int
    analyzed: int


class RiskOverview(BaseModel):
    at_risk: int
    high: int
    critical: int
    average_score: float
    previous_average_score: float | None
    average_score_change: float | None
    average_score_trend: str


class RiskSnapshotItem(BaseModel):
    action: str
    count: int
    percentage: float


class RiskAlert(BaseModel):
    analysis_id: int
    vehicle_id: int
    state: str | None
    risk_score: float
    priority: str
    action: str
    reason: str | None

class MaintenancePlanSummary(BaseModel):
    immediate: int
    short_term: int
    monitor: int
    manual_review: int


class DashboardSummaryResponse(BaseModel):
    fleet: FleetOverview
    risk: RiskOverview
    risk_snapshot: list[RiskSnapshotItem]
    maintenance_plan: MaintenancePlanSummary
    top_risk_alerts: list[RiskAlert]

