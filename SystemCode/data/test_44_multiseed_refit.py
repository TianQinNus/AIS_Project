"""Evaluate a predeclared multi-seed Train+Validation C4 ranking ensemble.

This experiment keeps Test 43's data, features, weights, hyperparameters and
veto rule. It averages component ranks over seeds 42, 77 and 2026 before the
existing conservative minimum with the C0-vs-C4 veto. Test is scored only
after all component models have been fitted.
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
    build_matrix,
    class_weights,
    fit_spec_schema,
    last_rows,
    load_csv,
    make_lgb_c4,
    make_veto,
    make_xgb_c4,
    percentile_rank,
    ranking_metrics,
    select_training,
    training_pool,
)


SEEDS = (42, 77, 2026)


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def multi_seed_score(x: np.ndarray, package: dict) -> tuple[np.ndarray, ...]:
    xgb_ranks = np.vstack([
        percentile_rank(model.predict_proba(x)[:, 1])
        for model in package["c4_xgboost_seeds"]
    ])
    lgb_ranks = np.vstack([
        percentile_rank(model.predict_proba(x)[:, 1])
        for model in package["c4_lightgbm_seeds"]
    ])
    veto_ranks = np.vstack([
        percentile_rank(model.predict_proba(x)[:, 1])
        for model in package["c4_veto_seeds"]
    ])
    base_rank = percentile_rank(np.vstack([xgb_ranks, lgb_ranks]).mean(axis=0))
    veto_rank = percentile_rank(veto_ranks.mean(axis=0))
    return np.minimum(base_rank, veto_rank), base_rank, veto_rank


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--seed42-dir",
        type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    parser.add_argument(
        "--current-model-dir",
        type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "processed" / "test_44_multiseed_refit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building the frozen Train+Validation matrix...", flush=True)
    train_op = read_operational(args.data_dir, "train")
    validation_op = read_operational(args.data_dir, "validation")
    train_spec = load_csv(args.data_dir, "train_specifications.csv")
    validation_spec = load_csv(args.data_dir, "validation_specifications.csv")
    tte = load_csv(args.data_dir, "train_tte.csv")
    validation_labels = load_csv(args.data_dir, "validation_labels.csv")
    raw_columns = [
        column for column in train_op if column not in ("vehicle_id", "time_step")
    ]
    schema = fit_spec_schema(train_spec)
    train_selected = select_training(training_pool(train_op, tte), seed=42)
    validation_selected = last_rows(validation_op, validation_labels)
    validation_selected["sw"] = 1.0
    x_train, feature_names = build_matrix(
        train_op, train_selected, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Train and Validation feature schemas differ")
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_class = np.concatenate([
        train_selected["class_label"].to_numpy(int),
        validation_selected["class_label"].to_numpy(int),
    ])
    base_weight = np.concatenate([
        train_selected["sw"].to_numpy(float),
        np.ones(len(validation_selected)),
    ])
    y_c4 = (fit_class == 4).astype(np.int8)
    c4_weight = class_weights(y_c4, base_weight, 0.5)
    veto_mask = np.isin(fit_class, (0, 4))
    veto_weight = class_weights(
        y_c4[veto_mask], base_weight[veto_mask], 0.25
    )

    seed42 = joblib.load(args.seed42_dir / MODEL_FILE)
    package = {
        "seeds": list(SEEDS),
        "c4_xgboost_seeds": [seed42["c4_xgboost"]],
        "c4_lightgbm_seeds": [seed42["c4_lightgbm"]],
        "c4_veto_seeds": [seed42["c4_veto"]],
        "horizon_classifier": seed42["horizon_classifier"],
    }
    for seed in SEEDS[1:]:
        print(f"Training seed {seed}...", flush=True)
        xgb = make_xgb_c4(seed)
        lgb = make_lgb_c4(seed)
        veto = make_veto(seed + 19)
        xgb.fit(x_fit, y_c4, sample_weight=c4_weight)
        lgb.fit(x_fit, y_c4, sample_weight=c4_weight)
        veto.fit(
            x_fit[veto_mask], y_c4[veto_mask], sample_weight=veto_weight
        )
        package["c4_xgboost_seeds"].append(xgb)
        package["c4_lightgbm_seeds"].append(lgb)
        package["c4_veto_seeds"].append(veto)

    joblib.dump(package, args.output_dir / "fleetcare_c4_multiseed_candidate.joblib")
    print("Frozen multi-seed fit complete; loading Test once...", flush=True)
    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    selected = last_rows(test_op, test_labels)
    x_test, test_names = build_matrix(
        test_op, selected, test_spec, schema, raw_columns
    )
    if feature_names != test_names:
        raise ValueError("Test feature schema differs")
    score, base_rank, veto_rank = multi_seed_score(x_test, package)
    y_test = (selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y_test, score)
    predictions = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"].to_numpy(),
        "actual_class": selected["class_label"].to_numpy(int),
        "c4_risk_score": score,
        "base_ensemble_rank": base_rank,
        "veto_rank": veto_rank,
        "ranking": pd.Series(score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    current = json.loads(
        (args.current_model_dir / "evaluation_results.json").read_text()
    )["test"]
    seed42_report = json.loads(
        (args.seed42_dir / "comparison.json").read_text()
    )["candidate_train_plus_validation"]
    comparison = {
        "protocol": (
            "Predeclared three-seed rank average using the frozen Test 43 "
            "recipe; Test scored after fitting."
        ),
        "current_train_only": current,
        "single_seed_train_plus_validation": seed42_report,
        "multiseed_train_plus_validation": report,
        "selected_for_deployment": False,
        "test_note": (
            "Official Test had already been inspected in earlier project "
            "experiments; results are comparative rather than pristine."
        ),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": "FleetCare C4 Three-Seed Refit Candidate",
        "model_file": "fleetcare_c4_multiseed_candidate.joblib",
        "seeds": list(SEEDS),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "raw_columns": raw_columns,
        "spec_columns": schema,
        "score_formula": (
            "min(rank(mean of six XGB/LGB component ranks), "
            "rank(mean of three veto ranks))"
        ),
        "test_evaluation": report,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
