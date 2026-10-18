"""Test 59: seed bagging and component-weight tuning for endpoint C4 trees.

The tree structures come from Test 58.  Repeated stochastic fits are averaged
to reduce rare-class ranking variance.  Official Validation selects the number
of seeds, LightGBM structure, and XGBoost/LightGBM rank weight.  Test is loaded
only after that choice is frozen.
"""
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
from test_58_endpoint_tree_finetune import make_lgb, make_xgb


SEEDS = (42, 17, 29, 43, 71)
SEED_COUNTS = (1, 3, 5)
LGB_CONFIGS = ("L_31_FLEX", "L_47")
XGB_WEIGHTS = (0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0)


def fit_seed_models(x, y, weight, lgb_config):
    xgb_models, lgb_models = [], []
    for seed in SEEDS:
        print(f"Fit seed {seed}: X_D6_FLEX + {lgb_config}", flush=True)
        xgb = make_xgb("X_D6_FLEX", seed)
        lgb = make_lgb(lgb_config, seed)
        xgb.fit(x, y, sample_weight=weight)
        lgb.fit(x, y, sample_weight=weight)
        xgb_models.append(xgb)
        lgb_models.append(lgb)
    return xgb_models, lgb_models


def seed_rank_matrix(models, x):
    return np.column_stack([
        percentile_rank(model.predict_proba(x)[:, 1]) for model in models
    ])


def bagged_rank(rank_matrix, n_seeds):
    return percentile_rank(rank_matrix[:, :n_seeds].mean(axis=1))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_59_endpoint_seed_bagging",
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
    x_train, names = build_matrix(train_op, endpoints, train_spec, schema, raw_columns)
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if names != validation_names:
        raise ValueError("Validation schema differs")
    train_labels = endpoints["class_label"].to_numpy(int)
    y_train = (train_labels == 4).astype(np.int8)
    train_weight = class_weights(y_train, np.ones(len(y_train)), 0.75)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)

    trained = {}
    validation_ranks = {}
    for lgb_config in LGB_CONFIGS:
        xgb_models, lgb_models = fit_seed_models(
            x_train, y_train, train_weight, lgb_config
        )
        trained[lgb_config] = (xgb_models, lgb_models)
        validation_ranks[lgb_config] = (
            seed_rank_matrix(xgb_models, x_validation),
            seed_rank_matrix(lgb_models, x_validation),
        )

    rows = []
    for lgb_config, (xr, lr) in validation_ranks.items():
        for n_seeds in SEED_COUNTS:
            xscore = bagged_rank(xr, n_seeds)
            lscore = bagged_rank(lr, n_seeds)
            for xgb_weight in XGB_WEIGHTS:
                score = percentile_rank(
                    xgb_weight * xscore + (1.0 - xgb_weight) * lscore
                )
                rows.append({
                    "lgb_config": lgb_config,
                    "n_seeds": n_seeds,
                    "xgb_weight": xgb_weight,
                    **ranking_metrics(y_validation, score),
                })
    table = pd.DataFrame(rows).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation leaders:\n", table.head(15).to_string(index=False), flush=True)

    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        train_labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_y = (fit_labels == 4).astype(np.int8)
    fit_weight = class_weights(fit_y, np.ones(len(fit_y)), 0.75)
    selected_n = int(winner["n_seeds"])
    selected_lgb = str(winner["lgb_config"])
    selected_xgb_weight = float(winner["xgb_weight"])
    final_xgb, final_lgb = [], []
    for seed in SEEDS[:selected_n]:
        print(f"Refit selected seed {seed}", flush=True)
        xmodel = make_xgb("X_D6_FLEX", seed)
        lmodel = make_lgb(selected_lgb, seed)
        xmodel.fit(x_fit, fit_y, sample_weight=fit_weight)
        lmodel.fit(x_fit, fit_y, sample_weight=fit_weight)
        final_xgb.append(xmodel)
        final_lgb.append(lmodel)

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if names != test_names:
        raise ValueError("Test schema differs")
    xrank = bagged_rank(seed_rank_matrix(final_xgb, x_test), selected_n)
    lrank = bagged_rank(seed_rank_matrix(final_lgb, x_test), selected_n)
    final_score = percentile_rank(
        selected_xgb_weight * xrank + (1.0 - selected_xgb_weight) * lrank
    )
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y_test, final_score)
    reference54 = json.loads(
        (here / "processed" / "test_54_endpoint_feature_parameter" / "results.json").read_text()
    )["test54_feature_parameter"]
    result = {
        "selection_rule": (
            "Validation selects seed count, LightGBM structure, and component "
            "weight by Top-5 C4, Top-2 C4, AP, then AUC."
        ),
        "validation_winner": winner,
        "test54_reference": reference54,
        "test59_seed_bagging": report,
        "selected_for_deployment": bool(
            report["top5_c4_found"] > reference54["top5_c4_found"]
        ),
        "test_note": (
            "Test was loaded only after Validation selection, but it is not a "
            "pristine holdout because prior project experiments inspected it."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "xgb_bag_rank": xrank,
        "lgb_bag_rank": lrank,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump({
        "xgb_models": final_xgb,
        "lgb_models": final_lgb,
        "xgb_config": "X_D6_FLEX",
        "lgb_config": selected_lgb,
        "seeds": list(SEEDS[:selected_n]),
        "xgb_weight": selected_xgb_weight,
    }, args.output_dir / "fleetcare_c4_seed_bagged_candidate.joblib")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model_name": "FleetCare Endpoint Seed-Bagged Tree Ensemble",
        "model_file": "fleetcare_c4_seed_bagged_candidate.joblib",
        "feature_names": names,
        "raw_columns": raw_columns,
        "spec_columns": schema,
        "winner": winner,
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
