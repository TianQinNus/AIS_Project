from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String, false, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class Truck(Base):
    __tablename__ = "truck"

    vehicle_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )

    spec_0: Mapped[str | None] = mapped_column(String(20))
    spec_1: Mapped[str | None] = mapped_column(String(20))
    spec_2: Mapped[str | None] = mapped_column(String(20))
    spec_3: Mapped[str | None] = mapped_column(String(20))
    spec_4: Mapped[str | None] = mapped_column(String(20))
    spec_5: Mapped[str | None] = mapped_column(String(20))
    spec_6: Mapped[str | None] = mapped_column(String(20))
    spec_7: Mapped[str | None] = mapped_column(String(20))

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    in_maintenance: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="truck"
    )