from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class VehicleListItem(CamelModel):
    vehicle_id: str
    predicted_class: str
    risk_score: int


class VehicleListResponse(CamelModel):
    total: int
    page: int
    page_size: int
    items: list[VehicleListItem]


class VehicleRiskReading(CamelModel):
    label: str
    value: str


class VehicleTrendPoint(CamelModel):
    label: str
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
    risk12_percentile: float | None = None
    risk6_score: float | None = None
    risk6_percentile: float | None = None
