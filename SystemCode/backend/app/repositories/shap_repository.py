from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.prediction_shap import PredictionShap


class ShapRepository:

    @staticmethod
    def get_by_analysis(
        db: Session,
        analysis_id: int,
        *,
        limit: int = 5,
    ) -> list[PredictionShap]:

        stmt = (
            select(PredictionShap)
            .where(
                PredictionShap.analysis_id
                == analysis_id
            )
            .order_by(
                PredictionShap.contribution_rank
            )
            .limit(limit)
        )

        return list(
            db.scalars(stmt).all()
        )