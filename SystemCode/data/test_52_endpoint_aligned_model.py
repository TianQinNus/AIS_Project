"""Test 52: endpoint-aligned extension of Test 43.

Official evaluation scores one last readout per vehicle, while Test 43 learns
from multiple historical stages. This experiment trains an additional binary
C4 model on exactly one last eligible training readout per vehicle. Validation
selects class weighting and blending with Test 43. Test is loaded only after
the selected endpoint model is refit with Validation endpoints.
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
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    make_lgb_c4,
    make_xgb_c4,
    percentile_rank,
    ranking_metrics,
    training_pool,
)


EXPONENTS = (0.0, 0.25, 0.50, 0.75)


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def train_endpoints(op, tte):
    rows = training_pool(op, tte).sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).groupby("vehicle_id", sort=False).tail(1).copy()
    rows["sw"] = 1.0
    return rows.reset_index(drop=True)


def fit_pair(x, labels, exponent, seed=42):
    y = (labels == 4).astype(np.int8)
    base = np.ones(len(y), dtype=float)
    weight = class_weights(y, base, exponent)
    xgb = make_xgb_c4(seed)
    lgb = make_lgb_c4(seed)
    xgb.fit(x, y, sample_weight=weight)
    lgb.fit(x, y, sample_weight=weight)
    return {"xgb": xgb, "lgb": lgb}


def pair_rank(x, models):
    xgb = percentile_rank(models["xgb"].predict_proba(x)[:, 1])
    lgb = percentile_rank(models["lgb"].predict_proba(x)[:, 1])
    return percentile_rank((xgb + lgb) / 2)


def blend(base, endpoint, endpoint_weight):
    return np.power(np.clip(base, 1e-6, 1), 1 - endpoint_weight) * np.power(
        np.clip(endpoint, 1e-6, 1), endpoint_weight
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
        default=here / "processed" / "test_52_endpoint_aligned",
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
    print(
        f"Endpoint rows={len(endpoints):,}; "
        f"C4={int((endpoints.class_label == 4).sum()):,}", flush=True,
    )
    x_train, feature_names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    x_validation, validation_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Validation feature schema differs")
    labels = endpoints["class_label"].to_numpy(int)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)
    current_models = joblib.load(args.current_model_dir / MODEL_FILE)
    base_validation, _, _ = ensemble_c4_risk_score(x_validation, current_models)
    candidates = [{
        "exponent": -1.0, "endpoint_weight": 0.0,
        **ranking_metrics(y_validation, base_validation),
    }]
    packages = {}
    for exponent in EXPONENTS:
        print(f"Training endpoint exponent={exponent:.2f}...", flush=True)
        models = fit_pair(x_train, labels, exponent)
        packages[exponent] = models
        endpoint_rank = pair_rank(x_validation, models)
        for endpoint_weight in (0.15, 0.25, 0.50, 0.75, 1.0):
            candidates.append({
                "exponent": exponent, "endpoint_weight": endpoint_weight,
                **ranking_metrics(
                    y_validation,
                    blend(base_validation, endpoint_rank, endpoint_weight),
                ),
            })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation endpoint leaders:", flush=True)
    print(table.head(12).to_string(index=False), flush=True)
    if float(winner["endpoint_weight"]) == 0:
        result = {"winner": winner, "test_loaded": False}
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("Endpoint models rejected; Test was not loaded.", flush=True)
        return

    # Refit endpoints with one final row from every Validation vehicle.
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    print("Refitting selected endpoint model before Test...", flush=True)
    final_endpoint = fit_pair(x_fit, fit_labels, float(winner["exponent"]))

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
    endpoint_test = pair_rank(x_test, final_endpoint)
    final_score = blend(
        base_test, endpoint_test, float(winner["endpoint_weight"])
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
        "test52_endpoint_aligned": test_report,
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
        "endpoint_score": endpoint_test,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "base_models": refit_models,
            "endpoint_models": final_endpoint,
            "exponent": winner["exponent"],
            "endpoint_weight": winner["endpoint_weight"],
        },
        args.output_dir / "fleetcare_c4_endpoint_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Test 43 + Endpoint-Aligned Candidate",
            "model_file": "fleetcare_c4_endpoint_candidate.joblib",
            "n_features": len(feature_names),
            "endpoint_training_rows": len(endpoints),
            "endpoint_c4_rows": int((labels == 4).sum()),
            "feature_names": feature_names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "winner": winner,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
