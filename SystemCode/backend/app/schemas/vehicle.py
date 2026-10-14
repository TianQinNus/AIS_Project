from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class VehicleListItem(CamelModel):
    vehicle_id: str
    state: str
    predicted_class: str
    risk_score: float
    in_maintenance: bool = False
    is_active: bool = True


class VehicleMaintenanceUpdate(CamelModel):
    in_maintenance: bool


class VehicleMaintenanceResponse(CamelModel):
    vehicle_id: str
    in_maintenance: bool


class VehicleListResponse(CamelModel):
    total: int
    page: int
    page_size: int
    items: list[VehicleListItem]


class VehicleActionOption(CamelModel):
    action: str
    label: str
    count: int


class VehicleActionListResponse(CamelModel):
    items: list[VehicleActionOption]


class VehicleRiskReading(CamelModel):
    label: str
    value: str


class VehicleTrendPoint(CamelModel):
    label: str = Field(
        description="Calendar day of this prediction, formatted YYYY-MM-DD HH:MM.",
    )
    risk_class: str = Field(
        description="Single predicted window: Risk6, Risk12, Risk24, Risk48, or Beyond 48.",
    )
    time_step: float | None = None
    trend_direction: str | None = None
    risk48: float | None = None
    risk24: float | None = None
    risk12: float | None = None
    risk6: float | None = None


class VehicleFactor(CamelModel):
    name: str
    impact: float = Field(
        ge=0,
        le=1,
        description="Absolute SHAP magnitude scaled to the strongest returned factor.",
    )


class VehicleDetailResponse(CamelModel):
    vehicle_id: str
    badge: str
    tag_color: str
    predicted_state: str
    action: str
    history: str
    risks: list[VehicleRiskReading]
    trend: list[VehicleTrendPoint]
    rule: str
    reasons: list[str]
    factors: list[VehicleFactor]
    reason: str | None = None
    priority: str | None = None
    confidence: float | None = None
    risk48_percentile: float | None = None
    risk24_percentile: float | None = None
    risk12_percentile: float | None = None
    risk48_score: float | None = None
    risk24_score: float | None = None
    risk12_score: float | None = None
    risk6_score: float | None = None
    risk6_percentile: float | None = None
    time_step: float | None = None
    trend_direction: str | None = None
