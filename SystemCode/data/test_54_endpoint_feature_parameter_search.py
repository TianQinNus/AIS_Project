"""Test 54: endpoint feature-family and tree-complexity search.

Starting from Test 52, compare five predeclared feature collections and three
tree-complexity profiles. Class weighting and endpoint sampling stay fixed.
Validation selects one pair; only that pair is refit before Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from fleetcare_c4_risk_inference import (
    build_matrix,
    class_weights,
    fit_spec_schema,
    last_rows,
    load_csv,
    percentile_rank,
    ranking_metrics,
)
from test_52_endpoint_aligned_model import read_operational, train_endpoints


FEATURE_SETS = (
    "ALL",
    "NO_SPEC",
    "NO_MIX",
    "RATE_RATIO_SPEC",
    "TEMPORAL_SPEC",
)
PARAMETERS = ("CURRENT", "SHALLOW", "FLEXIBLE")


def feature_indices(names, feature_set):
    prefixes = np.asarray([name.split(":", 1)[0] for name in names])
    if feature_set == "ALL":
        keep = np.ones(len(names), dtype=bool)
    elif feature_set == "NO_SPEC":
        keep = prefixes != "spec"
    elif feature_set == "NO_MIX":
        keep = ~np.char.startswith(prefixes.astype(str), "mix")
    elif feature_set == "RATE_RATIO_SPEC":
        keep = np.isin(prefixes, ("rate", "ratio1", "ratio3", "ratio5", "spec"))
    elif feature_set == "TEMPORAL_SPEC":
        keep = prefixes != "prop"
    else:
        raise ValueError(feature_set)
    return np.flatnonzero(keep)


def make_pair(profile, seed=42):
    if profile == "CURRENT":
        xgb = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=300, learning_rate=0.04, max_depth=5,
            min_child_weight=5, gamma=0.05, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=3.0,
            random_state=seed, n_jobs=4,
        )
        lgb = LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.035,
            num_leaves=15, min_child_samples=35, colsample_bytree=0.8,
            subsample=0.8, subsample_freq=1, reg_lambda=2.0,
            verbosity=-1, random_state=seed, n_jobs=4,
        )
    elif profile == "SHALLOW":
        xgb = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=500, learning_rate=0.025, max_depth=3,
            min_child_weight=10, gamma=0.10, subsample=0.85,
            colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=5.0,
            random_state=seed, n_jobs=4,
        )
        lgb = LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.025,
            num_leaves=7, min_child_samples=50, colsample_bytree=0.8,
            subsample=0.85, subsample_freq=1, reg_alpha=0.1, reg_lambda=4.0,
            verbosity=-1, random_state=seed, n_jobs=4,
        )
    elif profile == "FLEXIBLE":
        xgb = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=350, learning_rate=0.035, max_depth=6,
            min_child_weight=3, gamma=0.02, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=2.0,
            random_state=seed, n_jobs=4,
        )
        lgb = LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.03,
            num_leaves=31, min_child_samples=25, colsample_bytree=0.85,
            subsample=0.8, subsample_freq=1, reg_lambda=2.0,
            verbosity=-1, random_state=seed, n_jobs=4,
        )
    else:
        raise ValueError(profile)
    return {"xgb": xgb, "lgb": lgb}


def fit_pair(x, labels, profile):
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, np.ones(len(y)), 0.75)
    models = make_pair(profile)
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
        default=here / "processed" / "test_54_endpoint_feature_parameter",
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
    print("Building endpoint matrices...", flush=True)
    x_train, feature_names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Validation feature schema differs")

    candidates = []
    packages = {}
    for feature_set in FEATURE_SETS:
        indices = feature_indices(feature_names, feature_set)
        for profile in PARAMETERS:
            key = f"{feature_set}__{profile}"
            print(f"Training {key} ({len(indices)} features)...", flush=True)
            models = fit_pair(x_train[:, indices], labels, profile)
            packages[key] = models
            report = ranking_metrics(
                y_validation, score_pair(x_validation[:, indices], models)
            )
            candidates.append({
                "feature_set": feature_set, "parameter_set": profile,
                "n_features": len(indices), **report,
            })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation feature/parameter leaders:", flush=True)
    print(table.head(15).to_string(index=False), flush=True)

    winner_indices = feature_indices(feature_names, winner["feature_set"])
    validation_selected["sw"] = 1.0
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    print("Refitting Validation winner before Test...", flush=True)
    final_models = fit_pair(
        x_fit[:, winner_indices], fit_labels, winner["parameter_set"]
    )

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if feature_names != test_names:
        raise ValueError("Test feature schema differs")
    final_score = score_pair(x_test[:, winner_indices], final_models)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, final_score)
    test52 = json.loads(
        (here / "processed" / "test_52_endpoint_aligned" / "results.json").read_text()
    )["test52_endpoint_aligned"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "validation_winner": winner,
        "test52_reference": test52,
        "test54_feature_parameter": test_report,
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
        {
            "models": final_models,
            "feature_indices": winner_indices,
            "feature_set": winner["feature_set"],
            "parameter_set": winner["parameter_set"],
        },
        args.output_dir / "fleetcare_c4_endpoint_tuned_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Tuned Endpoint Candidate",
            "model_file": "fleetcare_c4_endpoint_tuned_candidate.joblib",
            "feature_names": feature_names,
            "selected_feature_names": [feature_names[i] for i in winner_indices],
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "winner": winner,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
