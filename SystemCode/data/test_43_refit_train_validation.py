"""Refit the frozen FleetCare ensemble on Train plus Validation last readouts.

The model recipe and hyperparameters are unchanged. Validation contributes one
labelled last-readout row per vehicle. Test is loaded only after the refit and
is used once for descriptive comparison with the deployed Train-only package.
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
    RULE_PROFILE,
    build_matrix,
    class_weights,
    fit_spec_schema,
    last_rows,
    load_csv,
    make_lgb_c4,
    make_multiclass,
    make_veto,
    make_xgb_c4,
    multiclass_weights,
    select_training,
    training_pool,
)
from train_fleetcare_c4_risk_model import evaluate_split


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--current-model-dir",
        type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Train and Validation for the frozen refit...", flush=True)
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
        validation_op,
        validation_selected,
        validation_spec,
        schema,
        raw_columns,
    )
    if feature_names != validation_names:
        raise ValueError("Train and Validation feature schemas differ")

    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_class = np.concatenate(
        [
            train_selected["class_label"].to_numpy(int),
            validation_selected["class_label"].to_numpy(int),
        ]
    )
    base_weight = np.concatenate(
        [train_selected["sw"].to_numpy(float), np.ones(len(validation_selected))]
    )
    y_c4 = (fit_class == 4).astype(np.int8)
    print(
        f"Fit rows={len(fit_class):,}; features={x_fit.shape[1]:,}; "
        f"C4 rows={int(y_c4.sum()):,}; Validation additions={len(validation_selected):,}",
        flush=True,
    )

    xgb_c4 = make_xgb_c4(42)
    lgb_c4 = make_lgb_c4(42)
    c4_weight = class_weights(y_c4, base_weight, 0.5)
    xgb_c4.fit(x_fit, y_c4, sample_weight=c4_weight)
    lgb_c4.fit(x_fit, y_c4, sample_weight=c4_weight)

    veto_mask = np.isin(fit_class, (0, 4))
    veto = make_veto(61)
    veto.fit(
        x_fit[veto_mask],
        y_c4[veto_mask],
        sample_weight=class_weights(
            y_c4[veto_mask], base_weight[veto_mask], 0.25
        ),
    )

    horizon = make_multiclass(42)
    horizon.fit(
        x_fit,
        fit_class,
        sample_weight=multiclass_weights(fit_class, base_weight, 0.25),
    )
    models = {
        "c4_xgboost": xgb_c4,
        "c4_lightgbm": lgb_c4,
        "c4_veto": veto,
        "horizon_classifier": horizon,
    }
    joblib.dump(models, args.output_dir / MODEL_FILE)

    # The design is frozen above. Test is read only at this point.
    print("Frozen refit complete; loading Test once...", flush=True)
    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    report, predictions = evaluate_split(
        "test",
        test_op,
        test_spec,
        test_labels,
        schema,
        raw_columns,
        feature_names,
        models,
    )
    current_report = json.loads(
        (args.current_model_dir / "evaluation_results.json").read_text(
            encoding="utf-8"
        )
    )["test"]
    comparison = {
        "protocol": (
            "Frozen current recipe refit on Train selected histories plus one "
            "labelled Validation last readout per vehicle; Test scored once."
        ),
        "current_train_only": current_report,
        "candidate_train_plus_validation": report,
        "selected_for_deployment": False,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metadata = {
        "model_name": "FleetCare C4 Ensemble Train+Validation Refit Candidate",
        "model_file": MODEL_FILE,
        "model_components": list(models),
        "raw_columns": raw_columns,
        "spec_columns": schema,
        "lags": [1, 3, 5],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_selected_rows": int(len(train_selected)),
        "validation_last_readout_rows": int(len(validation_selected)),
        "fit_rows": int(len(fit_class)),
        "fit_c4_rows": int(y_c4.sum()),
        "risk6_formula": (
            "min(percentile_rank(mean(rank(XGB), rank(LGBM))), "
            "percentile_rank(C0-vs-C4 veto))"
        ),
        "rule_profile": RULE_PROFILE,
        "test_evaluation": report,
        "test_note": (
            "Test was not used to change this refit recipe, but official Test "
            "had already been inspected in earlier project experiments."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
