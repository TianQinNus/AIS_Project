from sqlalchemy import (
    Date,
    case,
    cast,
    desc,
    func,
    select,
    text,
)
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.truck import Truck


class DashboardRepository:

    @staticmethod
    def _latest_daily_risk_subquery():
        daily_latest = (
            select(
                Analysis.vehicle_id,
                cast(
                    Analysis.prediction_at,
                    Date,
                ).label("prediction_date"),
                Analysis.risk6_score,
                Analysis.risk12_score,
                Analysis.risk24_score,
                Analysis.risk48_score,
                func.row_number()
                .over(
                    partition_by=(
                        Analysis.vehicle_id,
                        cast(
                            Analysis.prediction_at,
                            Date,
                        ),
                    ),
                    order_by=(
                        Analysis.prediction_at.desc(),
                        Analysis.analysis_id.desc(),
                    ),
                )
                .label("row_num"),
            )
            .where(
                Analysis.prediction_at
                >= func.current_date()
                - text("INTERVAL '4 days'"),
                Analysis.prediction_at
                < func.current_date()
                + text("INTERVAL '1 day'"),
            )
            .subquery()
        )

        return (
            select(
                daily_latest.c.prediction_date,
                (
                    0.40 * daily_latest.c.risk6_score
                    + 0.30 * daily_latest.c.risk12_score
                    + 0.20 * daily_latest.c.risk24_score
                    + 0.10 * daily_latest.c.risk48_score
                ).label("composite_risk"),
            )
            .join(
                Truck,
                Truck.vehicle_id == daily_latest.c.vehicle_id,
            )
            .where(
                daily_latest.c.row_num == 1,
                Truck.is_active.is_(True),
                Truck.in_maintenance.is_(False),
            )
            .subquery()
        )

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
            .join(
                Truck,
                Truck.vehicle_id == Analysis.vehicle_id,
            )
            .where(
                func.coalesce(
                    Truck.in_maintenance,
                    False,
                ).is_(False),
                Truck.is_active.is_(True),
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
                .where(
                    func.coalesce(
                        Truck.in_maintenance,
                        False,
                    ).is_(False),
                    Truck.is_active.is_(True),
                )
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

        ).where(
            latest.c.row_num == 1
        )

        row = db.execute(stmt).mappings().one()

        daily_scores = cls._latest_daily_risk_subquery()
        daily_averages = (
            select(
                daily_scores.c.prediction_date,
                func.avg(
                    daily_scores.c.composite_risk
                ).label("fleet_risk_index")
            )
            .group_by(daily_scores.c.prediction_date)
            .order_by(daily_scores.c.prediction_date.desc())
            .limit(2)
        )

        average_rows = db.execute(
            daily_averages
        ).mappings().all()

        average_risk_score = (
            float(average_rows[0]["fleet_risk_index"])
            * 100
            if average_rows
            and average_rows[0]["fleet_risk_index"]
            is not None
            else 0.0
        )

        previous_average_risk_score = (
            float(average_rows[1]["fleet_risk_index"])
            * 100
            if len(average_rows) > 1
            and average_rows[1]["fleet_risk_index"]
            is not None
            else None
        )

        average_score_change = (
            average_risk_score
            - previous_average_risk_score
            if previous_average_risk_score is not None
            else None
        )

        if average_score_change is None:
            average_score_trend = "NO_DATA"
        elif average_score_change > 0:
            average_score_trend = "UP"
        elif average_score_change < 0:
            average_score_trend = "DOWN"
        else:
            average_score_trend = "STABLE"

        return {
            "total_fleet": total_fleet,
            "analyzed": int(row["analyzed"] or 0),
            "at_risk": int(row["at_risk"] or 0),
            "high": int(row["high"] or 0),
            "critical": int(row["critical"] or 0),
            "average_risk_score": (
                average_risk_score
            ),
            "previous_average_risk_score": (
                previous_average_risk_score
            ),
            "average_score_change": average_score_change,
            "average_score_trend": average_score_trend,
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
            .order_by(
                case(
                    {
                        "PRIORITY_INSPECT": 1,
                        "INSPECT": 2,
                        "MANUAL_REVIEW": 3,
                        "ENHANCED_MONITOR": 4,
                        "MONITOR": 5,
                        "CONTINUE": 6,
                        "COLLECT_MORE_DATA": 7,
                    },
                    value=latest.c.action,
                    else_=8,
                )
            )
        )

        rows = db.execute(stmt).all()

        result = {
            "PRIORITY_INSPECT": 0,
            "INSPECT": 0,
            "MANUAL_REVIEW": 0,
            "ENHANCED_MONITOR": 0,
            "MONITOR": 0,
            "CONTINUE": 0,
            "COLLECT_MORE_DATA": 0,
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