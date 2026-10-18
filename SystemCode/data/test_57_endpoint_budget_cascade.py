"""Test 57: budget-aware endpoint cascade for the Top-5% inspection queue."""
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


MODELS = {
    "STABLE": ("CURRENT", {1: 1.0, 2: 1.0, 3: 1.0}),
    "FLEXIBLE": ("FLEXIBLE", {1: 1.0, 2: 1.0, 3: 1.0}),
    "NEAR_LOW": ("FLEXIBLE", {1: 0.25, 2: 0.25, 3: 0.25}),
    "NEAR_X2": ("FLEXIBLE", {1: 2.0, 2: 2.0, 3: 2.0}),
}


def fit_model(x, labels, name):
    profile, near_weights = MODELS[name]
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, np.ones(len(y)), 0.75)
    for stage, multiplier in near_weights.items():
        weight[labels == stage] *= multiplier
    models = make_pair(profile)
    models["xgb"].fit(x, y, sample_weight=weight)
    models["lgb"].fit(x, y, sample_weight=weight)
    return models


def score_model(x, models):
    a = percentile_rank(models["xgb"].predict_proba(x)[:, 1])
    b = percentile_rank(models["lgb"].predict_proba(x)[:, 1])
    return percentile_rank((a + b) / 2)


def cascade_score(head, tail, head_percent):
    n = len(head)
    k = max(1, int(np.ceil(n * head_percent / 100)))
    head_order = np.argsort(-head, kind="stable")[:k]
    selected = np.zeros(n, dtype=bool)
    selected[head_order] = True
    tail_order = np.argsort(-tail, kind="stable")
    order = np.concatenate([head_order, tail_order[~selected[tail_order]]])
    score = np.empty(n, dtype=float)
    score[order] = np.linspace(1.0, 1.0 / n, n)
    return score


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_57_endpoint_budget_cascade",
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

    packages = {}
    validation_scores = {}
    for name in MODELS:
        print(f"Training cascade component {name}...", flush=True)
        packages[name] = fit_model(x_train, labels, name)
        validation_scores[name] = score_model(x_validation, packages[name])
    candidates = []
    for head in MODELS:
        for tail in MODELS:
            for budget in (1, 2, 3):
                score = cascade_score(
                    validation_scores[head], validation_scores[tail], budget
                )
                candidates.append({
                    "head": head, "tail": tail, "head_percent": budget,
                    **ranking_metrics(y_validation, score),
                })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print(table.head(15).to_string(index=False), flush=True)

    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    needed = {winner["head"], winner["tail"]}
    final_models = {}
    for name in needed:
        print(f"Refitting cascade component {name}...", flush=True)
        final_models[name] = fit_model(x_fit, fit_labels, name)

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if names != test_names:
        raise ValueError("Test schema differs")
    test_scores = {
        name: score_model(x_test, model) for name, model in final_models.items()
    }
    final_score = cascade_score(
        test_scores[winner["head"]], test_scores[winner["tail"]],
        int(winner["head_percent"]),
    )
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
        "test57_budget_cascade": report,
        "selected_for_deployment": False,
        "test_note": (
            "Cascade was selected on Validation; official Test had already "
            "been inspected during earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    })
    for name, values in test_scores.items():
        predictions[f"{name.lower()}_score"] = values
    predictions.sort_values("ranking").to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    joblib.dump(
        {"models": final_models, "cascade": {
            "head": winner["head"], "tail": winner["tail"],
            "head_percent": winner["head_percent"],
        }},
        args.output_dir / "fleetcare_c4_budget_cascade_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
