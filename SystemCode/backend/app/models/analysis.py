from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
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

    truck: Mapped["Truck"] = relationship(
        back_populates="analyses"
    )

    shap_values: Mapped[list["PredictionShap"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )