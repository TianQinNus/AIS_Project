from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# DATABASE
# ============================================================

from database import create_postgres_engine


# ============================================================
# VEHICLE ANALYSIS TOOL
# ============================================================

def get_vehicle_analysis(
    vehicle_id: int,
    engine,
    history_limit: int = 5,
):
    """
    Retrieve the latest FleetCare analysis,
    recent prediction history and SHAP explanation
    for one vehicle.
    """

    if isinstance(vehicle_id, bool) or not isinstance(vehicle_id, int):
        raise ValueError("vehicle_id must be an integer")
    if not 1 <= history_limit <= 20:
        raise ValueError("history_limit must be between 1 and 20")

    # ========================================================
    # 1. GET RECENT ANALYSIS HISTORY
    # ========================================================

    history_query = text(
        """
        SELECT
            analysis_id,
            vehicle_id,
            prediction_at,
            time_step, history_count, risk48_score, risk24_score, risk12_score,
            risk24_percentile, rule_trace, hybrid_ranking_score, priority_score,

            state,
            confidence_score,

            risk48_percentile,
            risk12_percentile,

            risk6_score,
            risk6_percentile,

            history_quality,

            risk6_persistence,
            risk6_deterioration_percentile,

            class_worsening,

            rule_id,
            priority,
            action,

            evidence,
            reason

        FROM public.analysis

        WHERE vehicle_id = :vehicle_id

        ORDER BY prediction_at DESC, analysis_id DESC

        LIMIT :history_limit
        """
    )

    with engine.connect() as conn:

        history_rows = (
            conn.execute(
                history_query,
                {
                    "vehicle_id": vehicle_id,
                    "history_limit": history_limit,
                },
            )
            .mappings()
            .all()
        )

    if not history_rows:

        return {
            "success": False,
            "vehicle_id": vehicle_id,
            "error": (
                f"No analysis was found for vehicle "
                f"{vehicle_id}."
            ),
        }

    # Latest prediction
    current = dict(history_rows[0])

    # Put history oldest -> newest
    history_rows = list(
        reversed(history_rows)
    )

    # ========================================================
    # 2. GET SHAP FOR CURRENT ANALYSIS
    # ========================================================

    shap_query = text(
        """
        SELECT
            feature_name,
            feature_value,
            shap_value,
            contribution_rank

        FROM public.prediction_shap

        WHERE analysis_id = :analysis_id

        ORDER BY contribution_rank ASC
        """
    )

    with engine.connect() as conn:

        shap_rows = (
            conn.execute(
                shap_query,
                {
                    "analysis_id":
                        current["analysis_id"]
                },
            )
            .mappings()
            .all()
        )

    # ========================================================
    # 3. BUILD HISTORY
    # ========================================================

    history = []

    for row in history_rows:

        history.append(
            {
                "analysis_id": int(row["analysis_id"]),
                "time_step": float(row["time_step"]) if row["time_step"] is not None else None,
                "prediction_at":
                    str(row["prediction_at"]),

                "state":
                    row["state"],

                "risk6_score":
                    (
                        float(row["risk6_score"])
                        if row["risk6_score"] is not None
                        else None
                    ),

                "risk6_percentile":
                    (
                        float(row["risk6_percentile"])
                        if row["risk6_percentile"] is not None
                        else None
                    ),

                "action":
                    row["action"],
            }
        )

    # ========================================================
    # 4. BUILD SHAP
    # ========================================================

    shap = []

    for row in shap_rows:

        shap.append(
            {
                "rank":
                    int(row["contribution_rank"]),

                "feature":
                    row["feature_name"],

                "feature_value":
                    (
                        float(row["feature_value"])
                        if row["feature_value"] is not None
                        else None
                    ),

                "shap_value":
                    (float(row["shap_value"]) if row["shap_value"] is not None else None),
            }
        )

    # ========================================================
    # 5. STRUCTURED TOOL RESULT
    # ========================================================

    return {

        "success": True,
        "history_semantics": "Recent stored analysis runs ordered by prediction_at; not necessarily distinct operational readouts.",
        "units": {"risk6_score": "0-1 model score, not calibrated repair probability",
                  "percentile": "0-100 fleet-relative rank",
                  "repair_horizon": "operational units, not calendar days",
                  "prediction_at": "analysis run timestamp, not operational observation time"},

        "vehicle_id":
            vehicle_id,

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        "current": {
            "time_step": float(current["time_step"]) if current["time_step"] is not None else None,
            "history_count": int(current["history_count"]) if current["history_count"] is not None else None,
            "risk48_score": float(current["risk48_score"]) if current["risk48_score"] is not None else None,
            "risk24_score": float(current["risk24_score"]) if current["risk24_score"] is not None else None,
            "risk12_score": float(current["risk12_score"]) if current["risk12_score"] is not None else None,
            "risk24_percentile": float(current["risk24_percentile"]) if current["risk24_percentile"] is not None else None,
            "rule_trace": current["rule_trace"],
            "hybrid_ranking_score": float(current["hybrid_ranking_score"]) if current["hybrid_ranking_score"] is not None else None,
            "priority_score": float(current["priority_score"]) if current["priority_score"] is not None else None,


            "analysis_id":
                int(current["analysis_id"]),

            "prediction_at":
                str(current["prediction_at"]),

            "state":
                current["state"],

            "confidence_score":
                (
                    float(current["confidence_score"])
                    if current["confidence_score"] is not None
                    else None
                ),

            "risk48_percentile":
                (
                    float(current["risk48_percentile"])
                    if current["risk48_percentile"] is not None
                    else None
                ),

            "risk12_percentile":
                (
                    float(current["risk12_percentile"])
                    if current["risk12_percentile"] is not None
                    else None
                ),

            "risk6_score":
                (
                    float(current["risk6_score"])
                    if current["risk6_score"] is not None
                    else None
                ),

            "risk6_percentile":
                (
                    float(current["risk6_percentile"])
                    if current["risk6_percentile"] is not None
                    else None
                ),

            "history_quality":
                current["history_quality"],

            "risk6_persistence":
                (
                    float(current["risk6_persistence"])
                    if current["risk6_persistence"] is not None
                    else None
                ),

            "risk6_deterioration_percentile":
                (
                    float(
                        current[
                            "risk6_deterioration_percentile"
                        ]
                    )
                    if current[
                        "risk6_deterioration_percentile"
                    ] is not None
                    else None
                ),

            "class_worsening":
                (bool(current["class_worsening"])
                 if current["class_worsening"] is not None else None),

            "rule_id":
                current["rule_id"],

            "priority":
                current["priority"],

            "action":
                current["action"],

            "evidence":
                current["evidence"],

            "reason":
                current["reason"],
        },

        # ----------------------------------------------------
        # Historical trend
        # ----------------------------------------------------

        "history":
            history,

        # ----------------------------------------------------
        # Model explanation
        # ----------------------------------------------------

        "shap":
            shap,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import json

    engine = create_postgres_engine()

    result = get_vehicle_analysis(
        vehicle_id=1492,
        engine=engine,
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )