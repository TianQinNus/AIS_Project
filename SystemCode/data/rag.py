from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# DATABASE
# ============================================================

def create_mysql_engine(
    username="root",
    password="",
    host="127.0.0.1",
    port=3306,
    database="fleetcareai",
):

    url = URL.create(
        drivername="mysql+pymysql",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )

    return create_engine(
        url,
        pool_pre_ping=True,
    )


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

    vehicle_id = int(vehicle_id)

    # ========================================================
    # 1. GET RECENT ANALYSIS HISTORY
    # ========================================================

    history_query = text(
        """
        SELECT
            analysis_id,
            vehicle_id,
            prediction_at,

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

        FROM analysis

        WHERE vehicle_id = :vehicle_id

        ORDER BY prediction_at DESC

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

        FROM prediction_shap

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
                    float(row["shap_value"]),
            }
        )

    # ========================================================
    # 5. STRUCTURED TOOL RESULT
    # ========================================================

    return {

        "success": True,

        "vehicle_id":
            vehicle_id,

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        "current": {

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
                bool(current["class_worsening"]),

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

    engine = create_mysql_engine()

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