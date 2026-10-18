"""Test 48: broad C4 search with explicit interval-sequence features.

The current representation aggregates changes over 1, 3 and 5 readouts. This
experiment exposes each of the last five intervals separately so tree models
can distinguish a new onset, persistent drift and acceleration. Four class-
weight regimes and three veto strategies are compared on Validation, including
conservative blends with the current rank. Only the Validation winner is refit
with Validation labels and allowed to score Test.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    MODEL_FILE,
    build_matrix,
    class_weights,
    detect_groups,
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    make_lgb_c4,
    make_veto,
    make_xgb_c4,
    percentile_rank,
    ranking_metrics,
    safe_divide,
    select_training,
    training_pool,
    vehicle_start,
)


CONFIGS = {
    "W35_V15": (0.35, 0.15),
    "W50_V25": (0.50, 0.25),
    "W65_V25": (0.65, 0.25),
    "W80_V35": (0.80, 0.35),
}


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def interval_sequence_features(
    op: pd.DataFrame, selected: pd.DataFrame, raw_columns: list[str], steps: int = 5
) -> tuple[np.ndarray, list[str]]:
    """Return last-five individual interval features using past data only."""
    histograms, singles = detect_groups(raw_columns)
    values = op[raw_columns].to_numpy(dtype=np.float32)
    vehicles = op["vehicle_id"].to_numpy()
    times = op["time_step"].to_numpy(dtype=np.float64)
    starts = vehicle_start(vehicles)
    positions = selected["row_id"].to_numpy(dtype=int)
    current = values[positions].astype(np.float64)
    age = times[positions] + 1.0
    single_lifetime = safe_divide(current[:, singles], age[:, None])
    group_lifetime = {
        key: safe_divide(current[:, indices].sum(axis=1), age)
        for key, indices in histograms
    }
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def add(array, labels):
        array = np.asarray(array, dtype=np.float32)
        arrays.append(array[:, None] if array.ndim == 1 else array)
        names.extend(labels)

    for step in range(1, steps + 1):
        end = np.maximum(positions - (step - 1), starts[positions])
        begin = np.maximum(positions - step, starts[positions])
        elapsed = times[end] - times[begin]
        difference = values[end].astype(np.float64) - values[begin].astype(np.float64)
        difference = np.where(difference < 0, np.nan, difference)
        interval_single_rate = safe_divide(difference[:, singles], elapsed[:, None])
        add(
            safe_divide(interval_single_rate, single_lifetime),
            [f"step_ratio{step}:{raw_columns[i]}" for i in singles],
        )
        for key, indices in histograms:
            total = difference[:, indices].sum(axis=1)
            add(
                safe_divide(difference[:, indices], total[:, None]),
                [f"step_mix{step}:{raw_columns[i]}" for i in indices],
            )
            add(
                safe_divide(safe_divide(total, elapsed), group_lifetime[key]),
                [f"step_ratio{step}:{key}_total"],
            )
    del values
    gc.collect()
    return np.hstack(arrays).astype(np.float32), names


def sequence_matrix(op, selected, spec, schema, raw_columns):
    base, base_names = build_matrix(op, selected, spec, schema, raw_columns)
    sequence, sequence_names = interval_sequence_features(op, selected, raw_columns)
    return np.hstack([base, sequence]).astype(np.float32), base_names + sequence_names


def fit_components(x, labels, base_weight, c4_exponent, veto_exponent, seed=42):
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, base_weight, c4_exponent)
    xgb = make_xgb_c4(seed)
    lgb = make_lgb_c4(seed)
    xgb.fit(x, y, sample_weight=weight)
    lgb.fit(x, y, sample_weight=weight)
    mask = np.isin(labels, (0, 4))
    veto = make_veto(seed + 19)
    veto.fit(
        x[mask], y[mask],
        sample_weight=class_weights(
            y[mask], base_weight[mask], veto_exponent
        ),
    )
    return {"c4_xgboost": xgb, "c4_lightgbm": lgb, "c4_veto": veto}


def component_ranks(x, models):
    xgb = percentile_rank(models["c4_xgboost"].predict_proba(x)[:, 1])
    lgb = percentile_rank(models["c4_lightgbm"].predict_proba(x)[:, 1])
    base = percentile_rank((xgb + lgb) / 2)
    veto = percentile_rank(models["c4_veto"].predict_proba(x)[:, 1])
    return base, veto


def veto_score(base, veto, strategy):
    if strategy == "hard_min":
        return np.minimum(base, veto)
    if strategy == "soft15":
        return np.power(np.clip(base, 1e-6, 1), 0.85) * np.power(
            np.clip(veto, 1e-6, 1), 0.15
        )
    if strategy == "soft30":
        return np.power(np.clip(base, 1e-6, 1), 0.70) * np.power(
            np.clip(veto, 1e-6, 1), 0.30
        )
    raise ValueError(strategy)


def blend(current, sequence, sequence_weight):
    if sequence_weight == 1.0:
        return sequence
    return np.power(np.clip(current, 1e-6, 1), 1 - sequence_weight) * np.power(
        np.clip(sequence, 1e-6, 1), sequence_weight
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
        default=here / "processed" / "test_48_interval_sequence_search",
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
    print("Building explicit five-interval sequence matrices...", flush=True)
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
    current_validation, _, _ = ensemble_c4_risk_score(
        x_validation[:, :534], current_models
    )
    current_report = ranking_metrics(y_validation, current_validation)
    candidates = [{
        "config": "CURRENT", "strategy": "current", "sequence_weight": 0.0,
        **current_report,
    }]
    packages = {}
    for config, (c4_exp, veto_exp) in CONFIGS.items():
        print(f"Training {config}...", flush=True)
        models = fit_components(x_train, labels, weights, c4_exp, veto_exp)
        packages[config] = models
        base, veto = component_ranks(x_validation, models)
        for strategy in ("hard_min", "soft15", "soft30"):
            sequence_score = veto_score(base, veto, strategy)
            for sequence_weight in (0.50, 0.75, 1.0):
                score = blend(current_validation, sequence_score, sequence_weight)
                report = ranking_metrics(y_validation, score)
                candidates.append({
                    "config": config,
                    "strategy": strategy,
                    "sequence_weight": sequence_weight,
                    **report,
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
        print("Sequence candidates rejected; Test was not loaded.", flush=True)
        return

    # Refit only the frozen Validation winner.
    validation_selected["sw"] = 1.0
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_weights = np.concatenate([weights, np.ones(len(validation_selected))])
    c4_exp, veto_exp = CONFIGS[winner["config"]]
    print(f"Refitting winner {winner['config']} before Test...", flush=True)
    final_sequence_models = fit_components(
        x_fit, fit_labels, fit_weights, c4_exp, veto_exp
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
    refit_models = joblib.load(args.refit_model_dir / MODEL_FILE)
    current_test, _, _ = ensemble_c4_risk_score(x_test[:, :534], refit_models)
    sequence_base, sequence_veto = component_ranks(x_test, final_sequence_models)
    sequence_test = veto_score(
        sequence_base, sequence_veto, winner["strategy"]
    )
    final_score = blend(
        current_test, sequence_test, float(winner["sequence_weight"])
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
        "test48_interval_sequence": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "current_score": current_test,
        "sequence_score": sequence_test,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "base_models": refit_models,
            "sequence_models": final_sequence_models,
            "config": winner["config"],
            "strategy": winner["strategy"],
            "sequence_weight": winner["sequence_weight"],
        },
        args.output_dir / "fleetcare_c4_interval_sequence_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare C4 Explicit Interval Sequence Candidate",
            "model_file": "fleetcare_c4_interval_sequence_candidate.joblib",
            "n_base_features": 534,
            "n_sequence_features": len(feature_names) - 534,
            "n_total_features": len(feature_names),
            "feature_names": feature_names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "winner": winner,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
