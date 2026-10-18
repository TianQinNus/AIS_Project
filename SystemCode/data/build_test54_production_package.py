"""Build the deployable Test54 + five-class FleetCare model package."""
from __future__ import annotations

import json
from pathlib import Path

import joblib


def main():
    root = Path(__file__).resolve().parent
    test54_dir = root / "processed" / "test_54_endpoint_feature_parameter"
    test43_dir = root / "processed" / "test_43_refit_train_validation"
    output = root / "processed" / "test_54_production_model"
    output.mkdir(parents=True, exist_ok=True)

    test54 = joblib.load(test54_dir / "fleetcare_c4_endpoint_tuned_candidate.joblib")
    test43 = joblib.load(test43_dir / "fleetcare_c4_risk_ensemble.joblib")
    meta54 = json.loads((test54_dir / "metadata.json").read_text(encoding="utf-8"))
    meta43 = json.loads((test43_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta54["feature_names"] != meta43["feature_names"]:
        raise ValueError("Test54 and five-class models use different feature schemas")
    if meta54["raw_columns"] != meta43["raw_columns"]:
        raise ValueError("Test54 and five-class models use different raw columns")
    if meta54["spec_columns"] != meta43["spec_columns"]:
        raise ValueError("Test54 and five-class models use different specification schemas")

    package = {
        "c4_xgboost": test54["models"]["xgb"],
        "c4_lightgbm": test54["models"]["lgb"],
        "horizon_classifier": test43["horizon_classifier"],
    }
    model_file = "fleetcare_c4_risk_ensemble.joblib"
    joblib.dump(package, output / model_file)
    metadata = {
        "model_name": "FleetCare Test54 Production Ensemble",
        "model_version": "TEST54_MODEL_FIRST_RULES",
        "model_file": model_file,
        "model_components": list(package),
        "c4_ranking_formula": "percentile_rank(mean(rank(Test54 XGBoost), rank(Test54 LightGBM)))",
        "probability_source": "Train+Validation five-class XGBoost",
        "raw_columns": meta54["raw_columns"],
        "spec_columns": meta54["spec_columns"],
        "feature_names": meta54["feature_names"],
        "lags": [1, 3, 5],
        "n_features": len(meta54["feature_names"]),
        "rule_profile": {
            "p6": 99.0,
            "p12": 97.0,
            "inspect6": 95.0,
            "p24": 90.0,
            "p48": 80.0,
            "persistence": 0.8,
            "fast_percentile": 85.0,
            "min_rising": 2,
            "review_near": 97.0,
            "volatility_percentile": 95.0,
            "uncertainty_percentile": 90.0,
        },
        "evaluation": json.loads((test54_dir / "results.json").read_text(encoding="utf-8")),
        "test_note": (
            "Official Test was inspected during iterative project development; "
            "reported Test metrics are comparative, not a pristine holdout estimate."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(output)
    print(metadata["model_name"], metadata["model_components"], metadata["n_features"])


if __name__ == "__main__":
    main()
