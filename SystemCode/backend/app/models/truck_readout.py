from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TruckReadout(Base):
    __tablename__ = "truck_readout"

    readout_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    vehicle_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "truck.vehicle_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    time_step: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
    )

    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )