"""Test 58: cross-validated tree fine-tuning for the endpoint C4 model.

Eight XGBoost and eight LightGBM profiles are screened with three-fold
stratified CV inside Train endpoints. Only the leading profiles are fitted on
all Train endpoints and paired on official Validation. The frozen winning pair
is then refit on Train plus Validation and evaluated on Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from fleetcare_c4_risk_inference import (
    build_matrix, class_weights, fit_spec_schema, last_rows, load_csv,
    percentile_rank, ranking_metrics,
)
from test_52_endpoint_aligned_model import read_operational, train_endpoints


XGB_CONFIGS = {
    "X_D4_REG": dict(n_estimators=450, learning_rate=0.03, max_depth=4,
                     min_child_weight=6, gamma=0.05, subsample=0.85,
                     colsample_bytree=0.80, reg_alpha=0.10, reg_lambda=4.0),
    "X_D5_BAL": dict(n_estimators=400, learning_rate=0.035, max_depth=5,
                     min_child_weight=4, gamma=0.03, subsample=0.85,
                     colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=3.0),
    "X_D6_FLEX": dict(n_estimators=350, learning_rate=0.035, max_depth=6,
                      min_child_weight=3, gamma=0.02, subsample=0.80,
                      colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=2.0),
    "X_D7": dict(n_estimators=325, learning_rate=0.035, max_depth=7,
                 min_child_weight=4, gamma=0.05, subsample=0.80,
                 colsample_bytree=0.75, reg_alpha=0.10, reg_lambda=3.0),
    "X_D6_LOWCHILD": dict(n_estimators=450, learning_rate=0.025, max_depth=6,
                          min_child_weight=1, gamma=0.00, subsample=0.85,
                          colsample_bytree=0.70, reg_alpha=0.05, reg_lambda=3.0),
    "X_D5_HIGHREG": dict(n_estimators=500, learning_rate=0.025, max_depth=5,
                         min_child_weight=8, gamma=0.10, subsample=0.85,
                         colsample_bytree=0.75, reg_alpha=0.25, reg_lambda=6.0),
    "X_D4_SLOW": dict(n_estimators=650, learning_rate=0.02, max_depth=4,
                      min_child_weight=4, gamma=0.03, subsample=0.90,
                      colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=3.0),
    "X_D8_REG": dict(n_estimators=300, learning_rate=0.04, max_depth=8,
                     min_child_weight=6, gamma=0.10, subsample=0.80,
                     colsample_bytree=0.70, reg_alpha=0.20, reg_lambda=5.0),
}

LGB_CONFIGS = {
    "L_15_REG": dict(n_estimators=500, learning_rate=0.025, num_leaves=15,
                     min_child_samples=45, colsample_bytree=0.80,
                     subsample=0.85, reg_alpha=0.10, reg_lambda=4.0),
    "L_23": dict(n_estimators=450, learning_rate=0.03, num_leaves=23,
                 min_child_samples=35, colsample_bytree=0.85,
                 subsample=0.85, reg_alpha=0.05, reg_lambda=3.0),
    "L_31_FLEX": dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                      min_child_samples=25, colsample_bytree=0.85,
                      subsample=0.80, reg_alpha=0.00, reg_lambda=2.0),
    "L_47": dict(n_estimators=400, learning_rate=0.03, num_leaves=47,
                 min_child_samples=30, colsample_bytree=0.80,
                 subsample=0.80, reg_alpha=0.10, reg_lambda=3.0),
    "L_63_REG": dict(n_estimators=350, learning_rate=0.035, num_leaves=63,
                     min_child_samples=40, colsample_bytree=0.75,
                     subsample=0.85, reg_alpha=0.20, reg_lambda=5.0),
    "L_31_LOWCHILD": dict(n_estimators=450, learning_rate=0.025, num_leaves=31,
                          min_child_samples=15, colsample_bytree=0.80,
                          subsample=0.85, reg_alpha=0.05, reg_lambda=3.0),
    "L_15_SLOW": dict(n_estimators=650, learning_rate=0.02, num_leaves=15,
                      min_child_samples=30, colsample_bytree=0.90,
                      subsample=0.90, reg_alpha=0.05, reg_lambda=3.0),
    "L_31_HIGHREG": dict(n_estimators=500, learning_rate=0.025, num_leaves=31,
                         min_child_samples=60, colsample_bytree=0.75,
                         subsample=0.85, reg_alpha=0.30, reg_lambda=7.0),
}


def make_xgb(config, seed=42):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=4, **XGB_CONFIGS[config]
    )


def make_lgb(config, seed=42):
    return LGBMClassifier(
        objective="binary", subsample_freq=1, verbosity=-1,
        random_state=seed, n_jobs=4, **LGB_CONFIGS[config]
    )


def cv_family(x, y, weight, configs, maker, family):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261018)
    rows = []
    for name in configs:
        print(f"CV {family} {name}...", flush=True)
        fold_metrics = []
        for fold, (fit_idx, hold_idx) in enumerate(folds.split(x, y), 1):
            model = maker(name, 42 + fold)
            model.fit(x[fit_idx], y[fit_idx], sample_weight=weight[fit_idx])
            score = model.predict_proba(x[hold_idx])[:, 1]
            fold_metrics.append(ranking_metrics(y[hold_idx], score))
        rows.append({
            "family": family, "config": name,
            "recall5_mean": float(np.mean([m["recall_at_5pct"] for m in fold_metrics])),
            "recall5_std": float(np.std([m["recall_at_5pct"] for m in fold_metrics])),
            "recall2_mean": float(np.mean([m["recall_at_2pct"] for m in fold_metrics])),
            "ap_mean": float(np.mean([m["c4_average_precision"] for m in fold_metrics])),
            "auc_mean": float(np.mean([m["c4_roc_auc"] for m in fold_metrics])),
        })
    return pd.DataFrame(rows).sort_values(
        ["recall5_mean", "ap_mean", "auc_mean"], ascending=False
    ).reset_index(drop=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_58_endpoint_tree_finetune",
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
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, np.ones(len(y)), 0.75)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)

    xgb_cv = cv_family(x_train, y, weight, XGB_CONFIGS, make_xgb, "XGB")
    lgb_cv = cv_family(x_train, y, weight, LGB_CONFIGS, make_lgb, "LGB")
    cv_table = pd.concat([xgb_cv, lgb_cv], ignore_index=True)
    cv_table.to_csv(args.output_dir / "train_cv_results.csv", index=False)
    print("XGB CV leaders:\n", xgb_cv.head(5).to_string(index=False), flush=True)
    print("LGB CV leaders:\n", lgb_cv.head(5).to_string(index=False), flush=True)

    xgb_names = list(dict.fromkeys(xgb_cv.head(3).config.tolist() + ["X_D6_FLEX"]))
    lgb_names = list(dict.fromkeys(lgb_cv.head(3).config.tolist() + ["L_31_FLEX"]))
    xgb_models = {}
    xgb_scores = {}
    for name in xgb_names:
        model = make_xgb(name)
        model.fit(x_train, y, sample_weight=weight)
        xgb_models[name] = model
        xgb_scores[name] = percentile_rank(model.predict_proba(x_validation)[:, 1])
    lgb_models = {}
    lgb_scores = {}
    for name in lgb_names:
        model = make_lgb(name)
        model.fit(x_train, y, sample_weight=weight)
        lgb_models[name] = model
        lgb_scores[name] = percentile_rank(model.predict_proba(x_validation)[:, 1])
    candidates = []
    for xname, xs in xgb_scores.items():
        for lname, ls in lgb_scores.items():
            score = percentile_rank((xs + ls) / 2)
            candidates.append({
                "xgb_config": xname, "lgb_config": lname,
                **ranking_metrics(y_validation, score),
            })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_pairs.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation pair leaders:\n", table.head(12).to_string(index=False), flush=True)

    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_y = (fit_labels == 4).astype(np.int8)
    fit_weight = class_weights(fit_y, np.ones(len(fit_y)), 0.75)
    final_xgb = make_xgb(winner["xgb_config"])
    final_lgb = make_lgb(winner["lgb_config"])
    print("Refitting CV-selected tree pair before Test...", flush=True)
    final_xgb.fit(x_fit, fit_y, sample_weight=fit_weight)
    final_lgb.fit(x_fit, fit_y, sample_weight=fit_weight)

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if names != test_names:
        raise ValueError("Test schema differs")
    xs = percentile_rank(final_xgb.predict_proba(x_test)[:, 1])
    ls = percentile_rank(final_lgb.predict_proba(x_test)[:, 1])
    final_score = percentile_rank((xs + ls) / 2)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y_test, final_score)
    reference54 = json.loads(
        (here / "processed" / "test_54_endpoint_feature_parameter" / "results.json").read_text()
    )["test54_feature_parameter"]
    result = {
        "selection_rule": (
            "Train 3-fold CV screens each family; Validation selects the final "
            "XGB/LGB pair by Top-5 C4, Top-2 C4, AP, then AUC."
        ),
        "validation_winner": winner,
        "test54_reference": reference54,
        "test58_tree_finetuned": report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after CV and Validation selection; official "
            "Test had been inspected during earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "xgb_rank": xs, "lgb_rank": ls, "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {"xgb": final_xgb, "lgb": final_lgb,
         "xgb_config": winner["xgb_config"],
         "lgb_config": winner["lgb_config"]},
        args.output_dir / "fleetcare_c4_tree_finetuned_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare CV-Tuned Endpoint Tree Pair",
            "model_file": "fleetcare_c4_tree_finetuned_candidate.joblib",
            "feature_names": names, "raw_columns": raw_columns,
            "spec_columns": schema, "winner": winner,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
