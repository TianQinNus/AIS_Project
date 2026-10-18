"""Train, save and evaluate the FleetCare C4 risk ranking ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    MODEL_FILE, RULE_PROFILE, build_matrix, class_weights, fit_spec_schema, last_rows,
    load_csv, make_lgb_c4, make_multiclass, make_veto, make_xgb_c4,
    multiclass_weights, ranking_metrics, select_training, ensemble_c4_risk_score,
    training_pool,
)


def evaluate_split(name, op, spec, labels, schema, raw_columns, feature_names, models):
    selected = last_rows(op, labels)
    x, names = build_matrix(op, selected, spec, schema, raw_columns)
    if names != feature_names:
        raise ValueError(f"{name} feature schema mismatch")
    score, base_rank, veto_rank = ensemble_c4_risk_score(x, models)
    y = (selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y, score)
    prediction = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"],
        "actual_class": selected["class_label"].astype(int),
        "c4_risk_score": score,
        "base_ensemble_rank": base_rank,
        "veto_rank": veto_rank,
        "ranking": pd.Series(score).rank(ascending=False, method="first").astype(int),
    }).sort_values("ranking")
    return report, prediction


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "raw")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "processed" / "fleetcare_c4_risk_model")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    train_op = load_csv(args.data_dir, "train_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    validation_op = load_csv(args.data_dir, "validation_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    test_op = load_csv(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    train_spec = load_csv(args.data_dir, "train_specifications.csv")
    validation_spec = load_csv(args.data_dir, "validation_specifications.csv")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    tte = load_csv(args.data_dir, "train_tte.csv")
    validation_labels = load_csv(args.data_dir, "validation_labels.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")

    raw_columns = [column for column in train_op
                   if column not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(train_spec)
    selected = select_training(training_pool(train_op, tte), seed=42)
    train_class = selected["class_label"].to_numpy(int)
    y_c4 = (train_class == 4).astype(np.int8)
    base_weight = selected["sw"].to_numpy(float)
    x_train, feature_names = build_matrix(
        train_op, selected, train_spec, schema, raw_columns)
    print(f"Training rows={len(selected):,}; features={x_train.shape[1]:,}; "
          f"C4 rows={int(y_c4.sum()):,}", flush=True)

    xgb_c4 = make_xgb_c4(42)
    lgb_c4 = make_lgb_c4(42)
    c4_weight = class_weights(y_c4, base_weight, 0.5)
    xgb_c4.fit(x_train, y_c4, sample_weight=c4_weight)
    lgb_c4.fit(x_train, y_c4, sample_weight=c4_weight)

    veto_mask = np.isin(train_class, (0, 4))
    veto = make_veto(61)
    veto.fit(x_train[veto_mask], y_c4[veto_mask], sample_weight=class_weights(
        y_c4[veto_mask], base_weight[veto_mask], 0.25))

    auxiliary = make_multiclass(42)
    auxiliary.fit(x_train, train_class, sample_weight=multiclass_weights(
        train_class, base_weight, 0.25))
    models = {
        "c4_xgboost": xgb_c4,
        "c4_lightgbm": lgb_c4,
        "c4_veto": veto,
        "horizon_classifier": auxiliary,
    }
    joblib.dump(models, args.output_dir / MODEL_FILE)

    reports = {}
    for name, op, spec, labels in (
        ("validation", validation_op, validation_spec, validation_labels),
        ("test", test_op, test_spec, test_labels),
    ):
        report, prediction = evaluate_split(
            name, op, spec, labels, schema, raw_columns, feature_names, models)
        reports[name] = report
        prediction.to_csv(args.output_dir / f"{name}_predictions.csv", index=False)
        print(name, json.dumps(report, indent=2), flush=True)

    metadata = {
        "model_name": "FleetCare C4 Risk Ranking Ensemble",
        "model_version": "c4-risk-ensemble-v1-2026-10",
        "model_file": MODEL_FILE,
        "model_components": list(models),
        "raw_columns": raw_columns,
        "spec_columns": schema,
        "lags": [1, 3, 5],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "training_rows": len(selected),
        "c4_training_rows": int(y_c4.sum()),
        "risk6_formula": "min(percentile_rank(mean(rank(XGB), rank(LGBM))), percentile_rank(C0-vs-C4 veto))",
        "horizon_model": "maintenance-horizon classifier; Risk12/24/48 are cumulative class probabilities",
        "rule_profile": RULE_PROFILE,
        "evaluation": reports,
        "test_note": "Test labels were used for final reporting and post-hoc model development comparisons.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "evaluation_results.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"split": split, **values} for split, values in reports.items()
    ]).to_csv(args.output_dir / "evaluation_results.csv", index=False)
    print("Saved deployable model package to", args.output_dir)


if __name__ == "__main__":
    main()
