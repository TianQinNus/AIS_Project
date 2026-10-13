from sqlalchemy.orm import Session

from app.repositories.analysis_repository import (
    AnalysisRepository,
)
from app.repositories.shap_repository import (
    ShapRepository,
)


class VehicleAnalysisTool:

    name = "get_vehicle_analysis"

    description = (
        "Retrieve FleetCare prediction information for a "
        "specific vehicle. Use this tool when the user asks "
        "about a vehicle's current state, risk, prediction "
        "history, deterioration, SHAP explanation, "
        "maintenance priority, or recommended action."
    )

    schema = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "vehicle_id": {
                    "type": "integer",
                    "description": (
                        "The vehicle ID to retrieve."
                    ),
                }
            },
            "required": [
                "vehicle_id"
            ],
            "additionalProperties": False,
        },
    }

    @staticmethod
    def execute(
        db: Session,
        *,
        vehicle_id: int,
        history_limit: int = 5,
    ) -> dict:

        vehicle_id = int(vehicle_id)

        analyses = (
            AnalysisRepository.list_by_vehicle(
                db,
                vehicle_id,
                limit=history_limit,
            )
        )

        if not analyses:
            return {
                "success": False,
                "vehicle_id": vehicle_id,
                "error": (
                    f"No analysis was found for "
                    f"vehicle {vehicle_id}."
                ),
            }

        current = analyses[0]

        shap_rows = (
            ShapRepository.get_by_analysis(
                db,
                current.analysis_id,
                limit=5,
            )
        )

        # Oldest -> newest
        history = [
            {
                "prediction_at": str(
                    row.prediction_at
                ),
                "state": row.state,
                "risk6_score": (
                    float(row.risk6_score)
                    if row.risk6_score is not None
                    else None
                ),
                "risk6_percentile": (
                    float(row.risk6_percentile)
                    if row.risk6_percentile
                    is not None
                    else None
                ),
                "priority": row.priority,
                "action": row.action,
            }
            for row in reversed(analyses)
        ]

        shap = [
            {
                "rank": int(
                    row.contribution_rank
                ),
                "feature": row.feature_name,
                "feature_value": (
                    float(row.feature_value)
                    if row.feature_value
                    is not None
                    else None
                ),
                "shap_value": float(
                    row.shap_value
                ),
            }
            for row in shap_rows
        ]

        return {
            "success": True,

            "vehicle_id": vehicle_id,

            "current": {
                "analysis_id": (
                    current.analysis_id
                ),

                "prediction_at": str(
                    current.prediction_at
                ),

                "state": current.state,

                "confidence_score": (
                    float(
                        current.confidence_score
                    )
                    if current.confidence_score
                    is not None
                    else None
                ),

                "risk48_percentile": (
                    float(
                        current.risk48_percentile
                    )
                    if current.risk48_percentile
                    is not None
                    else None
                ),

                "risk12_percentile": (
                    float(
                        current.risk12_percentile
                    )
                    if current.risk12_percentile
                    is not None
                    else None
                ),

                "risk6_score": (
                    float(current.risk6_score)
                    if current.risk6_score
                    is not None
                    else None
                ),

                "risk6_percentile": (
                    float(
                        current.risk6_percentile
                    )
                    if current.risk6_percentile
                    is not None
                    else None
                ),

                "history_quality": (
                    current.history_quality
                ),

                "risk6_persistence": (
                    float(
                        current.risk6_persistence
                    )
                    if current.risk6_persistence
                    is not None
                    else None
                ),

                "risk6_deterioration_percentile": (
                    float(
                        current
                        .risk6_deterioration_percentile
                    )
                    if current
                    .risk6_deterioration_percentile
                    is not None
                    else None
                ),

                "class_worsening": (
                    current.class_worsening
                ),

                "rule_id": current.rule_id,

                "priority": current.priority,

                "action": current.action,

                "evidence": current.evidence,

                "reason": current.reason,
            },

            "history": history,

            "shap": shap,
        }