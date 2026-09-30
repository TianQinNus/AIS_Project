from pathlib import Path

from fleetcare_inference import run_fleetcare_prediction


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "processed" / "final_model"


if __name__ == "__main__":
    result = run_fleetcare_prediction(
        model_dir=MODEL_DIR,
        username="root",
        password="",
        host="127.0.0.1",
        port=3306,
        database="fleetcareai",
        analysis_table="analysis",
        shap_table="prediction_shap",
        top_shap=5,
    )

    print("\nPrediction complete")
    print(
        result[
            [
                "vehicle_id",
                "state",
                "risk6_score",
                "risk6_percentile",
                "rule_id",
                "priority",
                "action",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
