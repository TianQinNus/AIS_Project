"""Test 49: C4 hard-negative reranking on explicit interval sequences.

The first-stage ensemble identifies high-scoring C0 vehicles. A second-stage
XGB/LGB pair is trained on all C4 rows, all C1-C3 rows and only the hardest C0
rows. Validation selects the mining fraction, class weight and conservative
blend with the first-stage score. Test is loaded only after selection/refit.
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
    class_weights,
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    make_lgb_c4,
    make_xgb_c4,
    percentile_rank,
    ranking_metrics,
    select_training,
    training_pool,
)
from test_48_interval_sequence_search import read_operational, sequence_matrix


CONFIGS = {
    "HN70_W25": (0.70, 0.25),
    "HN80_W25": (0.80, 0.25),
    "HN80_W35": (0.80, 0.35),
    "HN90_W35": (0.90, 0.35),
}


def fit_reranker(x, labels, base_weight, first_score, c0_quantile, exponent, seed=42):
    y = (labels == 4).astype(np.int8)
    c0 = labels == 0
    cutoff = float(np.quantile(first_score[c0], c0_quantile))
    keep = (~c0) | (first_score >= cutoff)
    weight = class_weights(y[keep], base_weight[keep], exponent)
    xgb = make_xgb_c4(seed)
    lgb = make_lgb_c4(seed)
    xgb.fit(x[keep], y[keep], sample_weight=weight)
    lgb.fit(x[keep], y[keep], sample_weight=weight)
    return {"xgb": xgb, "lgb": lgb}, {
        "training_rows": int(keep.sum()),
        "hard_c0_rows": int((keep & c0).sum()),
        "c0_score_cutoff": cutoff,
    }


def reranker_score(x, models):
    xgb = percentile_rank(models["xgb"].predict_proba(x)[:, 1])
    lgb = percentile_rank(models["lgb"].predict_proba(x)[:, 1])
    return percentile_rank((xgb + lgb) / 2)


def blend(first, second, second_weight):
    return np.power(np.clip(first, 1e-6, 1), 1 - second_weight) * np.power(
        np.clip(second, 1e-6, 1), second_weight
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
        default=here / "processed" / "test_49_hard_negative_reranker",
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
    print("Building hard-negative sequence matrices...", flush=True)
    x_train, feature_names = sequence_matrix(
        train_op, train_selected, train_spec, schema, raw_columns
    )
    x_validation, validation_names = sequence_matrix(
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
    train_first, _, _ = ensemble_c4_risk_score(x_train[:, :534], current_models)
    validation_first, _, _ = ensemble_c4_risk_score(
        x_validation[:, :534], current_models
    )
    current_report = ranking_metrics(y_validation, validation_first)
    candidates = [{
        "config": "CURRENT", "second_weight": 0.0,
        "training_rows": len(x_train), "hard_c0_rows": 0,
        **current_report,
    }]
    packages = {}
    mining = {}
    for config, (quantile, exponent) in CONFIGS.items():
        print(f"Training {config}...", flush=True)
        models, info = fit_reranker(
            x_train, labels, weights, train_first, quantile, exponent
        )
        packages[config] = models
        mining[config] = info
        second = reranker_score(x_validation, models)
        for second_weight in (0.25, 0.50, 0.75, 1.0):
            score = blend(validation_first, second, second_weight)
            candidates.append({
                "config": config, "second_weight": second_weight,
                **info, **ranking_metrics(y_validation, score),
            })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation leaders:", flush=True)
    print(table.head(12).to_string(index=False), flush=True)
    if winner["config"] == "CURRENT":
        result = {"winner": winner, "test_loaded": False}
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("Hard-negative rerankers rejected; Test was not loaded.", flush=True)
        return

    validation_selected["sw"] = 1.0
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_weights = np.concatenate([weights, np.ones(len(validation_selected))])
    refit_models = joblib.load(args.refit_model_dir / MODEL_FILE)
    fit_first, _, _ = ensemble_c4_risk_score(x_fit[:, :534], refit_models)
    quantile, exponent = CONFIGS[winner["config"]]
    print(f"Refitting winner {winner['config']} before Test...", flush=True)
    final_reranker, final_mining = fit_reranker(
        x_fit, fit_labels, fit_weights, fit_first, quantile, exponent
    )

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = sequence_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if feature_names != test_names:
        raise ValueError("Test feature schema differs")
    test_first, _, _ = ensemble_c4_risk_score(x_test[:, :534], refit_models)
    test_second = reranker_score(x_test, final_reranker)
    final_score = blend(
        test_first, test_second, float(winner["second_weight"])
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
        "final_mining": final_mining,
        "test43_reference": test43,
        "test49_hard_negative": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "first_stage_score": test_first,
        "hard_negative_score": test_second,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "base_models": refit_models,
            "hard_negative_models": final_reranker,
            "config": winner["config"],
            "second_weight": winner["second_weight"],
        },
        args.output_dir / "fleetcare_c4_hard_negative_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare C4 Hard-Negative Reranker Candidate",
            "model_file": "fleetcare_c4_hard_negative_candidate.joblib",
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
