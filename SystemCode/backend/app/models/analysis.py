from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.prediction_shap import PredictionShap
    from app.models.truck import Truck


class Analysis(Base):
    __tablename__ = "analysis"

    analysis_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    vehicle_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("truck.vehicle_id"),
        nullable=False,
        index=True,
    )

    prediction_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    state: Mapped[str | None] = mapped_column(String(10))

    confidence_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6)
    )

    risk48_percentile: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )

    risk12_percentile: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )

    risk6_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6)
    )

    risk12_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6)
    )

    risk24_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6)
    )

    risk48_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6)
    )

    risk6_percentile: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )

    history_quality: Mapped[str | None] = mapped_column(
        String(20)
    )

    risk6_persistence: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4)
    )

    risk6_deterioration_percentile: Mapped[Decimal | None] = (
        mapped_column(Numeric(5, 2))
    )

    class_worsening: Mapped[bool | None] = mapped_column(Boolean)

    rule_id: Mapped[str | None] = mapped_column(String(20))

    priority: Mapped[str | None] = mapped_column(String(20))

    action: Mapped[str | None] = mapped_column(String(50))

    evidence: Mapped[str | None] = mapped_column(Text)

    reason: Mapped[str | None] = mapped_column(Text)

    risk48_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    risk24_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    risk24_percentile: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    risk12_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    risk48_persistence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    risk12_persistence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    risk48_deterioration_percentile: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )
    risk12_deterioration_percentile: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2)
    )
    time_step: Mapped[float | None] = mapped_column(Double)
    ranking: Mapped[int | None] = mapped_column(Integer)
    history_count: Mapped[int | None] = mapped_column(Integer)
    trend_direction: Mapped[str | None] = mapped_column(Text)
    rule_trace: Mapped[str | None] = mapped_column(Text)
    hybrid_ranking_score: Mapped[float | None] = mapped_column(Double)
    priority_score: Mapped[float | None] = mapped_column(Double)
    risk24_persistence: Mapped[float | None] = mapped_column(Double)
    risk48_trend: Mapped[float | None] = mapped_column(Double)
    risk24_trend: Mapped[float | None] = mapped_column(Double)
    risk12_trend: Mapped[float | None] = mapped_column(Double)
    risk6_trend: Mapped[float | None] = mapped_column(Double)
    risk24_deterioration_percentile: Mapped[float | None] = mapped_column(Double)
    is_risk48_persistent: Mapped[bool | None] = mapped_column(Boolean)
    is_risk24_persistent: Mapped[bool | None] = mapped_column(Boolean)
    is_risk12_persistent: Mapped[bool | None] = mapped_column(Boolean)
    is_risk6_persistent: Mapped[bool | None] = mapped_column(Boolean)
    fast_risk48_deterioration: Mapped[bool | None] = mapped_column(Boolean)
    fast_risk24_deterioration: Mapped[bool | None] = mapped_column(Boolean)
    fast_risk12_deterioration: Mapped[bool | None] = mapped_column(Boolean)
    fast_risk6_deterioration: Mapped[bool | None] = mapped_column(Boolean)

    truck: Mapped["Truck"] = relationship(
        back_populates="analyses"
    )

    shap_values: Mapped[list["PredictionShap"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )