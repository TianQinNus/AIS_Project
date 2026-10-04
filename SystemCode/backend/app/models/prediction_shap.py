from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    SmallInteger,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class PredictionShap(Base):
    __tablename__ = "prediction_shap"

    shap_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    analysis_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "analysis.analysis_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    feature_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    feature_value: Mapped[float | None] = mapped_column(Float)

    shap_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    contribution_rank: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
    )

    analysis: Mapped["Analysis"] = relationship(
        back_populates="shap_values"
    )