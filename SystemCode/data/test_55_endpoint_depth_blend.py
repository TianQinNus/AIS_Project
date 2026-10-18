"""Test 55: Validation-selected blend of Test 52 and Test 54 endpoint depths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    build_matrix, fit_spec_schema, last_rows, load_csv, ranking_metrics,
)
from test_52_endpoint_aligned_model import read_operational, train_endpoints
from test_54_endpoint_feature_parameter_search import fit_pair, score_pair


def blend(stable, flexible, flexible_weight):
    return np.power(np.clip(stable, 1e-6, 1), 1 - flexible_weight) * np.power(
        np.clip(flexible, 1e-6, 1), flexible_weight
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_55_endpoint_depth_blend",
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
    x_train, names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if names != validation_names:
        raise ValueError("Validation schema differs")
    labels = endpoints["class_label"].to_numpy(int)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)
    print("Training stable and flexible endpoint pairs...", flush=True)
    stable_model = fit_pair(x_train, labels, "CURRENT")
    flexible_model = fit_pair(x_train, labels, "FLEXIBLE")
    stable = score_pair(x_validation, stable_model)
    flexible = score_pair(x_validation, flexible_model)
    candidates = []
    for weight in np.linspace(0, 1, 11):
        candidates.append({
            "flexible_weight": float(weight),
            **ranking_metrics(y_validation, blend(stable, flexible, weight)),
        })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print(table.to_string(index=False), flush=True)

    test52 = pd.read_csv(
        here / "processed" / "test_52_endpoint_aligned" / "test_predictions.csv"
    )[["vehicle_id", "actual_class", "c4_risk_score"]].rename(
        columns={"c4_risk_score": "stable_score"}
    )
    test54 = pd.read_csv(
        here / "processed" / "test_54_endpoint_feature_parameter" / "test_predictions.csv"
    )[["vehicle_id", "c4_risk_score"]].rename(
        columns={"c4_risk_score": "flexible_score"}
    )
    predictions = test52.merge(test54, on="vehicle_id", validate="one_to_one")
    final_score = blend(
        predictions["stable_score"].to_numpy(float),
        predictions["flexible_score"].to_numpy(float),
        float(winner["flexible_weight"]),
    )
    y_test = (predictions["actual_class"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, final_score)
    predictions["c4_risk_score"] = final_score
    predictions["ranking"] = pd.Series(final_score).rank(
        ascending=False, method="first"
    ).astype(int)
    predictions.sort_values("ranking").to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    reference52 = json.loads(
        (here / "processed" / "test_52_endpoint_aligned" / "results.json").read_text()
    )["test52_endpoint_aligned"]
    reference54 = json.loads(
        (here / "processed" / "test_54_endpoint_feature_parameter" / "results.json").read_text()
    )["test54_feature_parameter"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "validation_winner": winner,
        "test52_reference": reference52,
        "test54_reference": reference54,
        "test55_depth_blend": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Blend weight was selected on Validation; Test predictions were "
            "combined only after selection. Official Test had been inspected "
            "during earlier project work."
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Endpoint Stable/Flexible Blend",
            "stable_model": "Test 52 CURRENT endpoint pair",
            "flexible_model": "Test 54 FLEXIBLE endpoint pair",
            "flexible_weight": winner["flexible_weight"],
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
