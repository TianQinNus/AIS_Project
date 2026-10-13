from sqlalchemy.orm import Session

from app.repositories.dashboard_repository import (
    DashboardRepository,
)
from app.schemas.dashboard import (
    DashboardSummaryResponse,
    FleetOverview,
    MaintenancePlanSummary,
    RiskAlert,
    RiskOverview,
    RiskSnapshotItem,
)


class DashboardService:

    @staticmethod
    def get_dashboard(
        db: Session,
    ) -> DashboardSummaryResponse:

        summary = DashboardRepository.get_summary(db)

        snapshot = DashboardRepository.get_risk_snapshot(db)

        alerts = DashboardRepository.get_top_risk_alerts(
            db,
            limit=5,
        )

        analyzed = summary["analyzed"]

        # Risk Snapshot
        snapshot_items = []

        for action, count in snapshot.items():
            percentage = (
                round(count / analyzed * 100, 2)
                if analyzed
                else 0.0
            )

            snapshot_items.append(
                RiskSnapshotItem(
                    action=action,
                    count=count,
                    percentage=percentage,
                )
            )

        # Maintenance Plan Summary
        maintenance_plan = MaintenancePlanSummary(
            immediate=snapshot.get("PRIORITY_INSPECT", 0),
            short_term=(
                snapshot.get("INSPECT", 0)
                + snapshot.get("ENHANCED_MONITOR", 0)
            ),
            monitor=(
                snapshot.get("MONITOR", 0)
                + snapshot.get("CONTINUE", 0)
            ),
            manual_review=snapshot.get("MANUAL_REVIEW", 0),
        )

        # Top Risk Alerts
        risk_alerts = [
            RiskAlert(
                analysis_id=row["analysis_id"],
                vehicle_id=row["vehicle_id"],
                state=row["state"],
                risk_score=float(
                    row["risk6_percentile"] or 0
                ),
                priority=row["priority"],
                action=row["action"],
                reason=row["reason"],
            )
            for row in alerts
        ]

        return DashboardSummaryResponse(
            fleet=FleetOverview(
                total=summary["total_fleet"],
                analyzed=analyzed,
            ),
            risk=RiskOverview(
                at_risk=summary["at_risk"],
                high=summary["high"],
                critical=summary["critical"],
                average_score=round(
                    summary["average_risk_score"],
                    4,
                ),
                previous_average_score=(
                    round(
                        summary[
                            "previous_average_risk_score"
                        ],
                        4,
                    )
                    if summary[
                        "previous_average_risk_score"
                    ] is not None
                    else None
                ),
                average_score_change=(
                    round(
                        summary["average_score_change"],
                        4,
                    )
                    if summary["average_score_change"]
                    is not None
                    else None
                ),
                average_score_trend=(
                    summary["average_score_trend"]
                ),
            ),
            risk_snapshot=snapshot_items,
            maintenance_plan=maintenance_plan,
            top_risk_alerts=risk_alerts,
        )