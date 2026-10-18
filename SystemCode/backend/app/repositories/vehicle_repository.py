from sqlalchemy import String, cast, desc, func, select
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.prediction_shap import PredictionShap
from app.models.truck import Truck


class VehicleRepository:

    @staticmethod
    def _latest_analysis_subquery():
        return (
            select(
                Analysis.vehicle_id,
                Analysis.state,
                Analysis.action,
                Analysis.risk6_percentile,
                func.row_number()
                .over(
                    partition_by=Analysis.vehicle_id,
                    order_by=(
                        Analysis.prediction_at.desc(),
                        Analysis.analysis_id.desc(),
                    ),
                )
                .label("row_num"),
            )
            .subquery()
        )

    @classmethod
    def list_latest(
        cls,
        db: Session,
        *,
        action: str | None,
        vehicle_digits: str | None,
        page: int,
        page_size: int,
    ) -> tuple[int, list[dict]]:
        latest = cls._latest_analysis_subquery()
        filters = [latest.c.row_num == 1]

        if action is not None:
            filters.append(latest.c.action == action)

        if vehicle_digits is not None:
            filters.append(
                cast(latest.c.vehicle_id, String).contains(vehicle_digits)
            )

        total = (
            db.scalar(
                select(func.count())
                .select_from(latest)
                .where(*filters)
            )
            or 0
        )

        stmt = (
            select(
                latest.c.vehicle_id,
                latest.c.state,
                latest.c.action,
                latest.c.risk6_percentile,
                Truck.in_maintenance,
                Truck.is_active,
            )
            .join(Truck, Truck.vehicle_id == latest.c.vehicle_id)
            .where(*filters)
            .order_by(
                desc(latest.c.risk6_percentile).nulls_last(),
                latest.c.vehicle_id,
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        rows = db.execute(stmt).mappings().all()
        return int(total), [dict(row) for row in rows]

    @classmethod
    def list_actions(cls, db: Session) -> list[dict]:
        """Actions on the latest analysis, most urgent fleet risk first."""
        latest = cls._latest_analysis_subquery()
        urgency = func.avg(latest.c.risk6_percentile)
        stmt = (
            select(
                latest.c.action,
                func.count().label("count"),
            )
            .where(
                latest.c.row_num == 1,
                latest.c.action.is_not(None),
                latest.c.action != "",
            )
            .group_by(latest.c.action)
            .order_by(urgency.desc().nulls_last(), latest.c.action)
        )
        return [dict(row) for row in db.execute(stmt).mappings().all()]

    @staticmethod
    def set_maintenance(
        db: Session,
        vehicle_id: int,
        in_maintenance: bool,
    ) -> bool | None:
        truck = db.get(Truck, vehicle_id)
        if truck is None:
            return None
        truck.in_maintenance = in_maintenance
        db.commit()
        return truck.in_maintenance

    @staticmethod
    def get_history(
        db: Session,
        vehicle_id: int,
    ) -> list[dict]:
        stmt = (
            select(
                Analysis.analysis_id,
                Analysis.vehicle_id,
                Analysis.prediction_at,
                Analysis.state,
                Analysis.confidence_score,
                Analysis.risk48_percentile,
                Analysis.risk12_percentile,
                Analysis.risk6_score,
                Analysis.risk6_percentile,
                Analysis.history_quality,
                Analysis.rule_id,
                Analysis.priority,
                Analysis.action,
                Analysis.evidence,
                Analysis.reason,
                Analysis.risk48_score,
                Analysis.risk24_score,
                Analysis.risk24_percentile,
                Analysis.risk12_score,
                Analysis.time_step,
                Analysis.trend_direction,
                Analysis.risk48_trend,
                Analysis.risk24_trend,
                Analysis.risk12_trend,
                Analysis.risk6_trend,
            )
            .where(Analysis.vehicle_id == vehicle_id)
            .order_by(
                Analysis.prediction_at.desc(),
                Analysis.analysis_id.desc(),
            )
        )

        rows = [dict(row) for row in db.execute(stmt).mappings().all()]
        rows.reverse()
        return rows

    @staticmethod
    def get_factors(
        db: Session,
        analysis_id: int,
        *,
        limit: int = 4,
    ) -> list[dict]:
        stmt = (
            select(
                PredictionShap.feature_name,
                PredictionShap.shap_value,
                PredictionShap.contribution_rank,
            )
            .where(PredictionShap.analysis_id == analysis_id)
            .order_by(PredictionShap.contribution_rank.asc())
            .limit(limit)
        )

        return [dict(row) for row in db.execute(stmt).mappings().all()]
