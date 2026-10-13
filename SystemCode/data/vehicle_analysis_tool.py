from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


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


engine = create_mysql_engine()


def get_vehicle_analysis(
    vehicle_id: int,
    history_limit: int = 5,
):

    vehicle_id = int(vehicle_id)

    # ========================================================
    # ANALYSIS HISTORY
    # ========================================================

    query = text("""
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
    """)

    with engine.connect() as conn:

        rows = (
            conn.execute(
                query,
                {
                    "vehicle_id": vehicle_id,
                    "history_limit": history_limit,
                },
            )
            .mappings()
            .all()
        )

    if not rows:

        return {
            "success": False,
            "vehicle_id": vehicle_id,
            "error": "No prediction found for this vehicle.",
        }

    current = dict(rows[0])

    # ========================================================
    # SHAP
    # ========================================================

    shap_query = text("""
        SELECT
            feature_name,
            feature_value,
            shap_value,
            contribution_rank
        FROM prediction_shap
        WHERE analysis_id = :analysis_id
        ORDER BY contribution_rank
    """)

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
    # HISTORY
    # ========================================================

    history = []

    for row in reversed(rows):

        history.append({

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
        })

    # ========================================================
    # SHAP OUTPUT
    # ========================================================

    shap_output = []

    for row in shap_rows:

        shap_output.append({

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
        })

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "success": True,

        "vehicle_id":
            vehicle_id,

        "current": {

            "state":
                current["state"],

            "confidence_score":
                float(
                    current["confidence_score"]
                ),

            "risk48_percentile":
                float(
                    current["risk48_percentile"]
                ),

            "risk12_percentile":
                float(
                    current["risk12_percentile"]
                ),

            "risk6_score":
                float(
                    current["risk6_score"]
                ),

            "risk6_percentile":
                float(
                    current["risk6_percentile"]
                ),

            "history_quality":
                current["history_quality"],

            "risk6_persistence":
                float(
                    current["risk6_persistence"]
                ),

            "risk6_deterioration_percentile":
                float(
                    current[
                        "risk6_deterioration_percentile"
                    ]
                ),

            "class_worsening":
                bool(
                    current["class_worsening"]
                ),

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

        "history":
            history,

        "shap":
            shap_output,
    }