
from sqlalchemy import (
    Date,
    case,
    cast,
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
        """
        Latest analysis for each active vehicle
        that is not currently under maintenance.
        """
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
                Truck.is_active.is_(True),
                Truck.in_maintenance.is_(False),
            )
            .subquery()
        )

    @staticmethod
    def _latest_daily_risk_subquery():
        """
        Select the latest prediction per vehicle per day.

        No fixed calendar window is applied.
        Only complete risk-score records are eligible.
        """

        prediction_date = cast(
            Analysis.prediction_at,
            Date,
        )

        daily_latest = (
            select(
                Analysis.vehicle_id,
                prediction_date.label("prediction_date"),
                Analysis.risk6_score,
                Analysis.risk12_score,
                Analysis.risk24_score,
                Analysis.risk48_score,
                func.row_number()
                .over(
                    partition_by=(
                        Analysis.vehicle_id,
                        prediction_date,
                    ),
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
                Truck.is_active.is_(True),
                Truck.in_maintenance.is_(False),
            )
            .subquery()
        )

        composite_risk = (
            0.40 * daily_latest.c.risk6_score
            + 0.30 * daily_latest.c.risk12_score
            + 0.20 * daily_latest.c.risk24_score
            + 0.10 * daily_latest.c.risk48_score
        )

        return (
            select(
                daily_latest.c.prediction_date,
                daily_latest.c.vehicle_id,
                composite_risk.label("composite_risk"),
            )
            .where(
                daily_latest.c.row_num == 1,
                daily_latest.c.risk6_score.is_not(None),
                daily_latest.c.risk12_score.is_not(None),
                daily_latest.c.risk24_score.is_not(None),
                daily_latest.c.risk48_score.is_not(None),
            )
            .subquery()
        )

    @classmethod
    def get_average_risk(cls, db: Session) -> dict:
        """
        Calculate fleet average risk for the two most
        recent dates containing valid prediction data.
        """

        daily_scores = cls._latest_daily_risk_subquery()

        stmt = (
            select(
                daily_scores.c.prediction_date,
                func.avg(
                    daily_scores.c.composite_risk
                ).label("fleet_risk_index"),
                func.count().label("vehicle_count"),
            )
            .group_by(
                daily_scores.c.prediction_date
            )
            .order_by(
                daily_scores.c.prediction_date.desc()
            )
            .limit(2)
        )

        rows = db.execute(stmt).mappings().all()

        if not rows:
            return {
                "average_risk_score": None,
                "previous_average_risk_score": None,
                "average_score_change": None,
                "average_score_trend": "NO_DATA",
                "average_score_date": None,
                "previous_average_score_date": None,
            }

        current = rows[0]

        current_score = (
            float(current["fleet_risk_index"]) * 100
        )

        previous_score = None
        previous_date = None

        if len(rows) > 1:
            previous = rows[1]

            previous_score = (
                float(previous["fleet_risk_index"]) * 100
            )

            previous_date = previous["prediction_date"]

        change = (
            current_score - previous_score
            if previous_score is not None
            else None
        )

        if change is None:
            trend = "NO_DATA"
        elif abs(change) < 1e-9:
            trend = "STABLE"
        elif change > 0:
            trend = "UP"
        else:
            trend = "DOWN"

        return {
            "average_risk_score": current_score,
            "previous_average_risk_score": previous_score,
            "average_score_change": change,
            "average_score_trend": trend,
            "average_score_date": current["prediction_date"],
            "previous_average_score_date": previous_date,
        }

    @classmethod
    def get_summary(cls, db: Session) -> dict:

        latest = cls._latest_analysis_subquery()

        total_fleet = (
            db.scalar(
                select(func.count())
                .select_from(Truck)
                .where(
                    Truck.is_active.is_(True),
                    Truck.in_maintenance.is_(False),
                )
            )
            or 0
        )

        stmt = (
            select(
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
            )
            .where(latest.c.row_num == 1)
        )

        row = db.execute(stmt).mappings().one()

        average = cls.get_average_risk(db)

        return {
            "total_fleet": total_fleet,
            "analyzed": int(row["analyzed"] or 0),
            "at_risk": int(row["at_risk"] or 0),
            "high": int(row["high"] or 0),
            "critical": int(row["critical"] or 0),
            **average,
        }

    @classmethod
    def get_risk_snapshot(
        cls,
        db: Session,
    ) -> dict[str, int]:

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

        rows = db.execute(stmt).all()
        return {
            action: int(count)
            for action, count in rows
            if action
        }

    @classmethod
    def get_top_risk_alerts(
        cls,
        db: Session,
        *,
        limit: int = 5,
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
                desc(latest.c.risk6_percentile),
                latest.c.vehicle_id,
            )
            .limit(limit)
        )

        rows = db.execute(stmt).mappings().all()
        

        return [dict(row) for row in rows]
    @staticmethod
    def count_in_maintenance(db: Session) -> int:
        return (
            db.scalar(
                select(func.count())
                .select_from(Truck)
                .where(
                    Truck.is_active.is_(True),
                    Truck.in_maintenance.is_(True),
                )
            )
            or 0
        )
