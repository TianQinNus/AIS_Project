"""Multi-seed heterogeneous ensemble for low-capacity C4 precision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import load
from test_31_time_robust_model import fit_spec_schema, last_rows
from test_33_c4_recall_search import (
    build_matrix,
    class_weights,
    metrics,
    model,
    select_training,
    training_pool,
)


SEEDS = (42, 77, 2026)


def rank_score(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def precision_objective(report: dict) -> float:
    return float(0.50 * report["precision_top1pct"] +
                 0.30 * report["precision_top2pct"] +
                 0.20 * report["precision_top5pct"])


def combine_predictions(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    xr = [rank_score(predictions[f"xgb_{s}"]) for s in SEEDS]
    lr = [rank_score(predictions[f"lgb_{s}"]) for s in SEEDS]
    return {
        "xgb_seed42": predictions["xgb_42"],
        "lgb_seed42": predictions["lgb_42"],
        "pair_seed42_rank": (xr[0] + lr[0]) / 2,
        "xgb_3seed_rank": np.mean(xr, axis=0),
        "lgb_3seed_rank": np.mean(lr, axis=0),
        "heterogeneous_6model_rank": np.mean(xr + lr, axis=0),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--output-dir", type=Path, default=root / "processed" / "test_34_c4_precision")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train and validation...", flush=True)
    tr_op = load(args.data_dir, "train_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    va_op = load(args.data_dir, "validation_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    tr_spec = load(args.data_dir, "train_specifications.csv")
    va_spec = load(args.data_dir, "validation_specifications.csv")
    tte = load(args.data_dir, "train_tte.csv")
    va_labels = load(args.data_dir, "validation_labels.csv")
    raw_cols = [c for c in tr_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(tr_spec)
    selected = select_training(training_pool(tr_op, tte), "dense10", 42)
    va_sel = last_rows(va_op, va_labels)
    y_train = (selected["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    y_val = (va_sel["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    x_train, names = build_matrix(
        tr_op, selected, tr_spec, schema, raw_cols, (1, 3, 5), "temporal")
    x_val, val_names = build_matrix(
        va_op, va_sel, va_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if names != val_names:
        raise ValueError("Feature schema mismatch")
    weights = class_weights(y_train, selected["sw"].to_numpy(float), 0.5)

    fitted_models = {}
    val_base_predictions = {}
    for seed in SEEDS:
        for family, prefix in (("xgboost", "xgb"), ("lightgbm", "lgb")):
            key = f"{prefix}_{seed}"
            fitted = model(family, seed)
            fitted.fit(x_train, y_train, sample_weight=weights)
            fitted_models[key] = fitted
            val_base_predictions[key] = fitted.predict_proba(x_val)[:, 1]
            joblib.dump(fitted, args.output_dir / f"{key}.joblib")
            print("Trained", key, flush=True)

    val_candidates = combine_predictions(val_base_predictions)
    rows = []
    val_out = pd.DataFrame({"vehicle_id": va_sel["vehicle_id"], "actual_c4": y_val})
    for candidate, score in val_candidates.items():
        report = metrics(y_val, score)
        report["precision_objective"] = precision_objective(report)
        rows.append({"candidate": candidate, **report})
        val_out[candidate] = score
        print(f"  {candidate:28s} objective={report['precision_objective']:.4f} P@1={report['precision_top1pct']:.4f} P@2={report['precision_top2pct']:.4f} P@5={report['precision_top5pct']:.4f} R@5={report['recall_top5pct']:.4f}", flush=True)
    result = pd.DataFrame(rows).sort_values(
        ["precision_objective", "c4_ap"], ascending=False).reset_index(drop=True)
    winner = str(result.iloc[0]["candidate"])
    result.to_csv(args.output_dir / "validation_results.csv", index=False)
    val_out.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    selection = {
        "winner": winner,
        "winner_metrics": result.iloc[0].to_dict(),
        "selection_rule": "0.50*Precision@Top1% + 0.30*Precision@Top2% + 0.20*Precision@Top5%; C4 AP tie-break",
        "seeds": SEEDS,
        "feature_design": "dense10 C4 snapshots; temporal lags 1,3,5",
        "weight_exponent": 0.5,
        "raw_cols": raw_cols,
        "spec_columns": schema,
        "feature_names": names,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # The ensemble design is frozen before test files are opened.
    print("Frozen winner:", winner, "Loading post-hoc test...", flush=True)
    te_op = load(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    te_spec = load(args.data_dir, "test_specifications.csv")
    te_labels = load(args.data_dir, "test_labels.csv")
    te_sel = last_rows(te_op, te_labels)
    y_test = (te_sel["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    x_test, test_names = build_matrix(
        te_op, te_sel, te_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if names != test_names:
        raise ValueError("Test feature schema mismatch")
    test_base = {key: fitted.predict_proba(x_test)[:, 1]
                 for key, fitted in fitted_models.items()}
    test_candidates = combine_predictions(test_base)
    test_score = test_candidates[winner]
    test_report = metrics(y_test, test_score)
    test_report.update({
        "winner": winner,
        "vehicles": int(len(y_test)),
        "c4_total": int(y_test.sum()),
        "warning": "Official Test was inspected in earlier project experiments; this frozen ensemble was selected without using its Test result.",
    })
    pd.DataFrame({
        "vehicle_id": te_sel["vehicle_id"], "actual_c4": y_test,
        "c4_score": test_score,
        "c4_rank": pd.Series(test_score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n=== TEST ===")
    print(json.dumps(test_report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
