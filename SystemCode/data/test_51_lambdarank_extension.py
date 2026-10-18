"""Test 51: LambdaRank extension of the Test 43 C4 ensemble.

Test 43's features and data recipe are unchanged. A LightGBM LambdaRank model
directly optimises C4 ordering instead of binary log-loss. Validation selects
the ranker configuration and conservative blend with Test 43; Test is loaded
only after the selected ranker is refit on Train plus Validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from fleetcare_c4_risk_inference import (
    MODEL_FILE,
    build_matrix,
    class_weights,
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    percentile_rank,
    ranking_metrics,
    select_training,
    training_pool,
)


CONFIGS = {
    "R15_W00": (15, 35, 0.00),
    "R15_W25": (15, 35, 0.25),
    "R31_W00": (31, 45, 0.00),
    "R31_W25": (31, 45, 0.25),
}


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def make_ranker(leaves: int, min_child: int, seed: int = 42):
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=leaves,
        min_child_samples=min_child,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.10,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def fit_ranker(x, labels, base_weight, query_time, config, seed=42):
    leaves, min_child, exponent = CONFIGS[config]
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, base_weight, exponent)
    ranker = make_ranker(leaves, min_child, seed)
    # LightGBM caps a query at 10,000 rows. Operational-time deciles form
    # cross-sectional fleet queues of comparable vehicle exposure and prevent
    # the ranker from winning merely by learning vehicle age.
    query = np.asarray(pd.qcut(
        np.asarray(query_time, dtype=float), q=10, labels=False,
        duplicates="drop",
    ))
    order = np.argsort(query, kind="stable")
    groups = np.bincount(query[order].astype(int))
    groups = groups[groups > 0].tolist()
    ranker.fit(
        x[order], y[order], group=groups, sample_weight=weight[order]
    )
    return ranker


def blend(base, ranker, ranker_weight):
    return np.power(np.clip(base, 1e-6, 1), 1 - ranker_weight) * np.power(
        np.clip(ranker, 1e-6, 1), ranker_weight
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--current-model-dir", type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--refit-model-dir", type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_51_lambdarank_extension",
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
    print("Building unchanged Test 43 feature matrices...", flush=True)
    x_train, feature_names = build_matrix(
        train_op, train_selected, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Validation feature schema differs")
    labels = train_selected["class_label"].to_numpy(int)
    weights = train_selected["sw"].to_numpy(float)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)
    current_models = joblib.load(args.current_model_dir / MODEL_FILE)
    base_validation, _, _ = ensemble_c4_risk_score(x_validation, current_models)
    candidates = [{
        "config": "TEST43_BASE", "ranker_weight": 0.0,
        **ranking_metrics(y_validation, base_validation),
    }]
    packages = {}
    for config in CONFIGS:
        print(f"Training {config}...", flush=True)
        ranker = fit_ranker(
            x_train, labels, weights,
            train_selected["time_step"].to_numpy(float), config,
        )
        packages[config] = ranker
        rank = percentile_rank(ranker.predict(x_validation))
        for ranker_weight in (0.15, 0.25, 0.50, 0.75, 1.0):
            score = blend(base_validation, rank, ranker_weight)
            candidates.append({
                "config": config, "ranker_weight": ranker_weight,
                **ranking_metrics(y_validation, score),
            })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation LambdaRank leaders:", flush=True)
    print(table.head(12).to_string(index=False), flush=True)
    if winner["config"] == "TEST43_BASE":
        result = {"winner": winner, "test_loaded": False}
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("LambdaRank rejected; Test was not loaded.", flush=True)
        return

    validation_selected["sw"] = 1.0
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_weights = np.concatenate([weights, np.ones(len(validation_selected))])
    print(f"Refitting winner {winner['config']} before Test...", flush=True)
    fit_time = np.concatenate([
        train_selected["time_step"].to_numpy(float),
        validation_selected["time_step"].to_numpy(float),
    ])
    final_ranker = fit_ranker(
        x_fit, fit_labels, fit_weights, fit_time, winner["config"]
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
    refit_models = joblib.load(args.refit_model_dir / MODEL_FILE)
    base_test, _, _ = ensemble_c4_risk_score(x_test, refit_models)
    ranker_test = percentile_rank(final_ranker.predict(x_test))
    final_score = blend(
        base_test, ranker_test, float(winner["ranker_weight"])
    )
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, final_score)
    test43 = json.loads(
        (args.refit_model_dir / "comparison.json").read_text()
    )["candidate_train_plus_validation"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "validation_winner": winner,
        "test43_reference": test43,
        "test51_lambdarank": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "test43_score": base_test,
        "lambdarank_score": ranker_test,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "base_models": refit_models,
            "lambdarank": final_ranker,
            "config": winner["config"],
            "ranker_weight": winner["ranker_weight"],
        },
        args.output_dir / "fleetcare_c4_lambdarank_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Test 43 + LambdaRank Candidate",
            "model_file": "fleetcare_c4_lambdarank_candidate.joblib",
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "winner": winner,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
