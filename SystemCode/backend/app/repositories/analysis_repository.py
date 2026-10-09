from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis


class AnalysisRepository:

    @staticmethod
    def get_by_id(
        db: Session,
        analysis_id: int,
    ) -> Analysis | None:
        return db.get(
            Analysis,
            analysis_id,
        )

    @staticmethod
    def get_latest_by_vehicle(
        db: Session,
        vehicle_id: int,
    ) -> Analysis | None:

        stmt = (
            select(Analysis)
            .where(
                Analysis.vehicle_id == vehicle_id
            )
            .order_by(
                Analysis.prediction_at.desc(),
                Analysis.analysis_id.desc(),
            )
            .limit(1)
        )

        return db.scalar(stmt)

    @staticmethod
    def list_by_vehicle(
        db: Session,
        vehicle_id: int,
        *,
        limit: int = 5,
    ) -> list[Analysis]:

        stmt = (
            select(Analysis)
            .where(
                Analysis.vehicle_id == vehicle_id
            )
            .order_by(
                Analysis.prediction_at.desc(),
                Analysis.analysis_id.desc(),
            )
            .limit(limit)
        )

        return list(
            db.scalars(stmt).all()
        )