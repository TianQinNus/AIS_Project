"""Test 45: explicit degradation-stage summaries for C4 ranking.

The existing 534-feature representation already contains per-signal recent to
lifetime ratios. This experiment adds past-only row summaries that describe a
vehicle-level regime: breadth, magnitude, persistence and acceleration of
recent deviations. Selection is performed on Validation. Test is loaded only
if the augmented representation wins the predeclared Validation comparison.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
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


def read_operational(folder: Path, split: str) -> pd.DataFrame:
    return load_csv(folder, f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)


def _summary(values: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    finite = np.isfinite(values)
    safe = np.where(finite, values, np.nan)
    coverage = finite.mean(axis=1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q10, q25, q50, q75, q90 = (
            np.nanquantile(safe, q, axis=1) for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        )
        mean = np.nanmean(safe, axis=1)
        mad = np.nanmedian(np.abs(safe - q50[:, None]), axis=1)
    arrays = [coverage, mean, q10, q25, q50, q75, q90, mad]
    names = [
        f"{prefix}:coverage", f"{prefix}:mean", f"{prefix}:q10",
        f"{prefix}:q25", f"{prefix}:median", f"{prefix}:q75",
        f"{prefix}:q90", f"{prefix}:mad",
    ]
    return arrays, names


def add_degradation_stage_features(
    matrix: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Append aggregate regime features derived only from existing past features."""
    index = {name: i for i, name in enumerate(names)}
    ratio_keys = sorted({name.split(":", 1)[1] for name in names if name.startswith("ratio1:")})
    ratio = {}
    for lag in (1, 3, 5):
        columns = [index[f"ratio{lag}:{key}"] for key in ratio_keys]
        raw = matrix[:, columns].astype(np.float64)
        valid = np.isfinite(raw) & (raw >= 0)
        # A symmetric representation: 0 means recent equals lifetime; positive
        # means faster recent accumulation and negative means slower.
        ratio[lag] = np.where(valid, np.log(np.clip(raw, 1e-3, 1e3)), np.nan)

    arrays: list[np.ndarray] = []
    extra_names: list[str] = []
    for lag in (1, 3, 5):
        values = ratio[lag]
        part, part_names = _summary(values, f"regime_logratio{lag}")
        arrays.extend(part)
        extra_names.extend(part_names)
        finite = np.isfinite(values)
        denom = np.maximum(finite.sum(axis=1), 1)
        for label, condition in (
            ("up25", values > np.log(1.25)),
            ("up50", values > np.log(1.50)),
            ("up100", values > np.log(2.00)),
            ("down20", values < np.log(0.80)),
            ("down50", values < np.log(0.50)),
        ):
            arrays.append((condition & finite).sum(axis=1) / denom)
            extra_names.append(f"regime_logratio{lag}:fraction_{label}")

    valid_all = np.isfinite(ratio[1]) & np.isfinite(ratio[3]) & np.isfinite(ratio[5])
    denom_all = np.maximum(valid_all.sum(axis=1), 1)
    for label, threshold in (("up25", 1.25), ("up50", 1.50), ("up100", 2.0)):
        condition = (
            (ratio[1] > np.log(threshold))
            & (ratio[3] > np.log(threshold))
            & (ratio[5] > np.log(threshold))
            & valid_all
        )
        arrays.append(condition.sum(axis=1) / denom_all)
        extra_names.append(f"regime:persistent_{label}")
    persistent_down = (
        (ratio[1] < np.log(0.8))
        & (ratio[3] < np.log(0.8))
        & (ratio[5] < np.log(0.8))
        & valid_all
    )
    arrays.append(persistent_down.sum(axis=1) / denom_all)
    extra_names.append("regime:persistent_down20")

    acceleration = ratio[1] - ratio[5]
    part, part_names = _summary(acceleration, "regime_acceleration_1_vs_5")
    arrays.extend(part)
    extra_names.extend(part_names)
    arrays.append(((acceleration > np.log(1.25)) & valid_all).sum(axis=1) / denom_all)
    extra_names.append("regime_acceleration_1_vs_5:fraction_up25")

    direction = (
        (np.sign(ratio[1]) == np.sign(ratio[3]))
        & (np.sign(ratio[3]) == np.sign(ratio[5]))
        & valid_all
    )
    arrays.append(direction.sum(axis=1) / denom_all)
    extra_names.append("regime:directional_agreement")

    # Composition movement: recent activity distribution versus the wider
    # five-readout window for histogram-bin features.
    mix_keys = sorted({name.split(":", 1)[1] for name in names if name.startswith("mix1:")})
    if mix_keys:
        mix1 = matrix[:, [index[f"mix1:{key}"] for key in mix_keys]].astype(np.float64)
        mix5 = matrix[:, [index[f"mix5:{key}"] for key in mix_keys]].astype(np.float64)
        mix_shift = np.abs(mix1 - mix5)
        mix_shift[~(np.isfinite(mix1) & np.isfinite(mix5))] = np.nan
        part, part_names = _summary(mix_shift, "regime_mix_shift_1_vs_5")
        arrays.extend(part)
        extra_names.extend(part_names)

    extra = np.column_stack(arrays).astype(np.float32)
    return np.hstack([matrix, extra]).astype(np.float32), names + extra_names


def stage_matrix(op, selected, spec, schema, raw_columns):
    matrix, names = build_matrix(op, selected, spec, schema, raw_columns)
    return add_degradation_stage_features(matrix, names)


def fit_ranker(x, labels, base_weight, seed=42):
    y = (labels == 4).astype(np.int8)
    weight = class_weights(y, base_weight, 0.5)
    xgb = make_xgb_c4(seed)
    lgb = make_lgb_c4(seed)
    xgb.fit(x, y, sample_weight=weight)
    lgb.fit(x, y, sample_weight=weight)
    mask = np.isin(labels, (0, 4))
    veto = make_veto(seed + 19)
    veto.fit(
        x[mask], y[mask],
        sample_weight=class_weights(y[mask], base_weight[mask], 0.25),
    )
    return {"c4_xgboost": xgb, "c4_lightgbm": lgb, "c4_veto": veto}


def score_ranker(x, models):
    xgb = percentile_rank(models["c4_xgboost"].predict_proba(x)[:, 1])
    lgb = percentile_rank(models["c4_lightgbm"].predict_proba(x)[:, 1])
    base = percentile_rank((xgb + lgb) / 2)
    veto = percentile_rank(models["c4_veto"].predict_proba(x)[:, 1])
    return np.minimum(base, veto), base, veto


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--current-model-dir", type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_45_degradation_stage",
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
    print("Creating explicit degradation-stage features...", flush=True)
    x_train, feature_names = stage_matrix(
        train_op, train_selected, train_spec, schema, raw_columns
    )
    x_validation, validation_names = stage_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if feature_names != validation_names:
        raise ValueError("Validation feature schema differs")
    train_class = train_selected["class_label"].to_numpy(int)
    base_weight = train_selected["sw"].to_numpy(float)
    models = fit_ranker(x_train, train_class, base_weight)
    validation_score, _, _ = score_ranker(x_validation, models)
    validation_y = (validation_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    validation_report = ranking_metrics(validation_y, validation_score)
    validation_predictions = pd.DataFrame({
        "vehicle_id": validation_selected["vehicle_id"].to_numpy(),
        "actual_class": validation_selected["class_label"].to_numpy(int),
        "stage_score": validation_score,
    })
    validation_predictions.to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )
    joblib.dump(models, args.output_dir / "validation_stage_models.joblib")
    current_reports = json.loads(
        (args.current_model_dir / "evaluation_results.json").read_text()
    )
    current_validation = current_reports["validation"]
    selected = (
        validation_report["top5_c4_found"], validation_report["c4_average_precision"]
    ) > (
        current_validation["top5_c4_found"], current_validation["c4_average_precision"]
    )
    selection = {
        "selection_rule": "higher Validation Top-5 C4 found; AP breaks ties",
        "current_validation": current_validation,
        "stage_candidate_validation": validation_report,
        "candidate_selected": bool(selected),
    }
    print(json.dumps(selection, indent=2), flush=True)
    if not selected:
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        print("Candidate rejected on Validation; Test was not loaded.", flush=True)
        return

    # Refit the frozen selected representation on Train plus one last readout
    # from every Validation vehicle.
    validation_selected["sw"] = 1.0
    x_fit = np.vstack([x_train, x_validation]).astype(np.float32)
    fit_class = np.concatenate([
        train_class, validation_selected["class_label"].to_numpy(int)
    ])
    fit_weight = np.concatenate([base_weight, np.ones(len(validation_selected))])
    print("Validation won; refitting before loading Test...", flush=True)
    final_models = fit_ranker(x_fit, fit_class, fit_weight)

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    x_test, test_names = stage_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if feature_names != test_names:
        raise ValueError("Test feature schema differs")
    score, base_rank, veto_rank = score_ranker(x_test, final_models)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, score)
    joblib.dump(final_models, args.output_dir / "fleetcare_c4_stage_candidate.joblib")
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "c4_risk_score": score,
        "base_ensemble_rank": base_rank,
        "veto_rank": veto_rank,
        "ranking": pd.Series(score).rank(ascending=False, method="first").astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    result = {
        **selection,
        "current_test": current_reports["test"],
        "stage_candidate_test": test_report,
        "n_base_features": 534,
        "n_stage_features": len(feature_names) - 534,
        "n_total_features": len(feature_names),
        "test_note": (
            "Test was loaded after the Validation selection, but it had been "
            "inspected during earlier project development."
        ),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare C4 Degradation-Stage Candidate",
            "feature_names": feature_names,
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "model_file": "fleetcare_c4_stage_candidate.joblib",
            "score_formula": "existing XGB/LGB rank ensemble with C0-vs-C4 veto",
            "stage_feature_definition": (
                "Past-only aggregate breadth, magnitude, persistence, acceleration "
                "and histogram-composition change from 1/3/5-readout ratios."
            ),
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
