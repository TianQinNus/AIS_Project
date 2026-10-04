from sqlalchemy import (
    case,
    desc,
    func,
    select,
)
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.truck import Truck


class DashboardRepository:

    @staticmethod
    def _latest_analysis_subquery():
        return (
            select(
                Analysis.analysis_id,
                Analysis.vehicle_id,
                Analysis.prediction_at,
                Analysis.state,
                Analysis.confidence_score,
                Analysis.risk6_score,
                Analysis.risk6_percentile,
                Analysis.priority,
                Analysis.action,
                Analysis.evidence,
                Analysis.reason,
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
    def get_summary(cls, db: Session) -> dict:
        latest = cls._latest_analysis_subquery()

        total_fleet = (
            db.scalar(
                select(func.count())
                .select_from(Truck)
            )
            or 0
        )

        stmt = select(
            func.count().label("analyzed"),

            func.sum(
                case(
                    (
                        latest.c.priority.in_(
                            ["HIGH", "CRITICAL"]
                        ),
                        1,
                    ),
                    else_=0,
                )
            ).label("at_risk"),

            func.sum(
                case(
                    (
                        latest.c.priority == "HIGH",
                        1,
                    ),
                    else_=0,
                )
            ).label("high"),

            func.sum(
                case(
                    (
                        latest.c.priority == "CRITICAL",
                        1,
                    ),
                    else_=0,
                )
            ).label("critical"),

            func.avg(
                latest.c.risk6_percentile
            ).label("average_risk_score"),

        ).where(
            latest.c.row_num == 1
        )

        row = db.execute(stmt).mappings().one()

        return {
            "total_fleet": total_fleet,
            "analyzed": int(row["analyzed"] or 0),
            "at_risk": int(row["at_risk"] or 0),
            "high": int(row["high"] or 0),
            "critical": int(row["critical"] or 0),
            "average_risk_score": (
                float(row["average_risk_score"])
                if row["average_risk_score"] is not None
                else 0.0
            ),
        }

    @classmethod
    def get_risk_snapshot(cls, db: Session) -> dict[str, int]:
        latest = cls._latest_analysis_subquery()

        stmt = (
            select(
                latest.c.action,
                func.count().label("count"),
            )
            .where(latest.c.row_num == 1)
            .group_by(latest.c.action)
        )

        rows = db.execute(stmt).all()

        result = {
            "CONTINUE": 0,
            "MONITOR": 0,
            "ENHANCED_MONITOR": 0,
            "INSPECT": 0,
            "PRIORITY_INSPECT": 0,
        }

        for action, count in rows:
            if action in result:
                result[action] = count

        return result

    @classmethod
    def get_top_risk_alerts(
        cls,
        db: Session,
        *,
        limit: int = 3,
    ) -> list[dict]:
        latest = cls._latest_analysis_subquery()

        stmt = (
            select(
                latest.c.analysis_id,
                latest.c.vehicle_id,
                latest.c.state,
                latest.c.risk6_percentile,
                latest.c.priority,
                latest.c.action,
                latest.c.reason,
            )
            .where(
                latest.c.row_num == 1,
                latest.c.priority.in_(
                    ["HIGH", "CRITICAL"]
                ),
            )
            .order_by(
                desc(latest.c.risk6_percentile)
            )
            .limit(limit)
        )

        rows = db.execute(stmt).mappings().all()

        return [dict(row) for row in rows]