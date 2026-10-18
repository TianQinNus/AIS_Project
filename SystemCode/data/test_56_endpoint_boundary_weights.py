"""Test 56: sharpen the endpoint C4 boundary with near-stage weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    build_matrix, class_weights, fit_spec_schema, last_rows, load_csv,
    percentile_rank, ranking_metrics,
)
from test_52_endpoint_aligned_model import read_operational, train_endpoints
from test_54_endpoint_feature_parameter_search import make_pair


PROFILES = {
    "BASE": {1: 1.0, 2: 1.0, 3: 1.0},
    "C3_X2": {1: 1.0, 2: 1.0, 3: 2.0},
    "C3_X4": {1: 1.0, 2: 1.0, 3: 4.0},
    "C3_X8": {1: 1.0, 2: 1.0, 3: 8.0},
    "NEAR_X2": {1: 2.0, 2: 2.0, 3: 2.0},
    "NEAR_X4": {1: 4.0, 2: 4.0, 3: 4.0},
    "NEAR_LOW": {1: 0.25, 2: 0.25, 3: 0.25},
}


def fit_weighted_pair(x, labels, profile):
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, np.ones(len(y)), 0.75)
    for stage, multiplier in PROFILES[profile].items():
        weight[labels == stage] *= multiplier
    models = make_pair("FLEXIBLE")
    models["xgb"].fit(x, y, sample_weight=weight)
    models["lgb"].fit(x, y, sample_weight=weight)
    return models


def score_pair(x, models):
    xgb = percentile_rank(models["xgb"].predict_proba(x)[:, 1])
    lgb = percentile_rank(models["lgb"].predict_proba(x)[:, 1])
    return percentile_rank((xgb + lgb) / 2)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_56_endpoint_boundary_weights",
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
    endpoints = train_endpoints(train_op, tte)
    validation_selected = last_rows(validation_op, validation_labels)
    labels = endpoints["class_label"].to_numpy(int)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)
    x_train, names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if names != validation_names:
        raise ValueError("Validation schema differs")
    candidates = []
    packages = {}
    for profile in PROFILES:
        print(f"Training boundary profile {profile}...", flush=True)
        model = fit_weighted_pair(x_train, labels, profile)
        packages[profile] = model
        candidates.append({
            "profile": profile,
            **ranking_metrics(y_validation, score_pair(x_validation, model)),
        })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print(table.to_string(index=False), flush=True)

    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    print(f"Refitting profile {winner['profile']} before Test...", flush=True)
    final_model = fit_weighted_pair(x_fit, fit_labels, winner["profile"])
    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if names != test_names:
        raise ValueError("Test schema differs")
    final_score = score_pair(x_test, final_model)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y_test, final_score)
    reference54 = json.loads(
        (here / "processed" / "test_54_endpoint_feature_parameter" / "results.json").read_text()
    )["test54_feature_parameter"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "validation_winner": winner,
        "test54_reference": reference54,
        "test56_boundary_weighted": report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {"models": final_model, "profile": winner["profile"]},
        args.output_dir / "fleetcare_c4_boundary_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Endpoint Boundary-Weighted Candidate",
            "model_file": "fleetcare_c4_boundary_candidate.joblib",
            "profile": winner["profile"],
            "feature_names": names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
