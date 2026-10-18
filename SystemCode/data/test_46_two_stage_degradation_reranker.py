"""Test 46: conservative two-stage degradation reranking.

Validation selected a fixed geometric blend containing 95% of the existing C4
rank and 5% of the explicit degradation-stage rank. The stage model is refit on
Train plus Validation, then the frozen blend is evaluated on Test once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    MODEL_FILE,
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    ranking_metrics,
    select_training,
    training_pool,
)
from test_45_degradation_stage_features import (
    fit_ranker,
    read_operational,
    score_ranker,
    stage_matrix,
)


BASE_WEIGHT = 0.95


def blend(base: np.ndarray, stage: np.ndarray) -> np.ndarray:
    return np.power(np.clip(base, 1e-6, 1.0), BASE_WEIGHT) * np.power(
        np.clip(stage, 1e-6, 1.0), 1.0 - BASE_WEIGHT
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--base-model-dir", type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    parser.add_argument(
        "--current-model-dir", type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_46_two_stage_degradation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_op = read_operational(args.data_dir, "train")
    validation_op = read_operational(args.data_dir, "validation")
    train_spec = load_csv(args.data_dir, "train_specifications.csv")
    validation_spec = load_csv(args.data_dir, "validation_specifications.csv")
    tte = load_csv(args.data_dir, "train_tte.csv")
    validation_labels = load_csv(args.data_dir, "validation_labels.csv")
    raw_columns = [c for c in train_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(train_spec)
    train_selected = select_training(training_pool(train_op, tte), seed=42)
    validation_selected = last_rows(validation_op, validation_labels)
    validation_selected["sw"] = 1.0
    print("Building Train+Validation degradation-stage representation...", flush=True)
    x_train, feature_names = stage_matrix(
        train_op, train_selected, train_spec, schema, raw_columns
    )
    x_validation, validation_names = stage_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Validation feature schema differs")
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_class = np.concatenate([
        train_selected["class_label"].to_numpy(int),
        validation_selected["class_label"].to_numpy(int),
    ])
    fit_weight = np.concatenate([
        train_selected["sw"].to_numpy(float),
        np.ones(len(validation_selected)),
    ])
    stage_models = fit_ranker(x_fit, fit_class, fit_weight)
    print("Frozen two-stage fit complete; loading Test once...", flush=True)

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = stage_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if feature_names != test_names:
        raise ValueError("Test feature schema differs")
    base_models = joblib.load(args.base_model_dir / MODEL_FILE)
    base_score, _, _ = ensemble_c4_risk_score(x_test[:, :534], base_models)
    stage_score, _, _ = score_ranker(x_test, stage_models)
    final_score = blend(base_score, stage_score)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y_test, final_score)

    current = json.loads(
        (args.current_model_dir / "evaluation_results.json").read_text()
    )["test"]
    test43 = json.loads(
        (args.base_model_dir / "comparison.json").read_text()
    )["candidate_train_plus_validation"]
    result = {
        "selection_protocol": (
            "Validation chose geometric blend: 95% existing rank and 5% "
            "degradation-stage rank; Test loaded after refit."
        ),
        "current_train_only": current,
        "test43_base_train_plus_validation": test43,
        "test46_two_stage": report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was not used to select the blend weight in this experiment, "
            "but had been inspected during earlier project development."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "base_score": base_score,
        "degradation_stage_score": stage_score,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {"base_models": base_models, "stage_models": stage_models,
         "base_weight": BASE_WEIGHT},
        args.output_dir / "fleetcare_c4_two_stage_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare C4 Two-Stage Degradation Reranker",
            "model_file": "fleetcare_c4_two_stage_candidate.joblib",
            "base_weight": BASE_WEIGHT,
            "stage_weight": 1.0 - BASE_WEIGHT,
            "n_base_features": 534,
            "n_stage_features": len(feature_names) - 534,
            "n_total_features": len(feature_names),
            "feature_names": feature_names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
