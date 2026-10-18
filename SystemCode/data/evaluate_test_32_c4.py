"""Evaluate the frozen Test 32 C4 specialist once on official test vehicles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from test21 import load
from test_31_time_robust_model import last_rows, prepare_dynamic, spec_matrix
from test_32_c4_f1_search import feature_variant, transform_if_needed


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--model-dir", type=Path, default=root / "processed" / "test_32_c4_f1")
    args = ap.parse_args()

    meta = json.loads((args.model_dir / "selection.json").read_text(encoding="utf-8"))
    model = joblib.load(args.model_dir / "c4_specialist_model.joblib")
    imputer_path = args.model_dir / "c4_specialist_imputer.joblib"
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None
    variant = str(meta["winner"]["feature_variant"])
    threshold = float(meta["c4_probability_threshold"])

    op = load(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(args.data_dir, "test_specifications.csv")
    labels = load(args.data_dir, "test_labels.csv")
    selected = last_rows(op, labels)
    y = (selected["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    dynamic, names, blocks = prepare_dynamic(op, selected, meta["raw_cols"], meta["lags"])
    spec_x = spec_matrix(spec, selected["vehicle_id"].to_numpy(), meta["spec_columns"])
    x, _ = feature_variant(dynamic, names, blocks, spec_x, variant)
    score = model.predict_proba(transform_if_needed(x, imputer))[:, 1]
    pred = (score >= threshold).astype(np.int8)

    result = {
        "dataset": "official_test_posthoc_reuse",
        "warning": "The project had already inspected official test outcomes in earlier experiments; treat this as a post-hoc comparison, not a pristine independent estimate.",
        "vehicles": int(len(y)),
        "c4_total": int(y.sum()),
        "feature_variant": variant,
        "model": meta["winner"]["model"],
        "sampling": meta["winner"]["sampling"],
        "threshold_frozen_from_validation": threshold,
        "predicted_c4": int(pred.sum()),
        "c4_found": int((pred & y).sum()),
        "c4_f1": float(f1_score(y, pred, zero_division=0)),
        "c4_precision": float(precision_score(y, pred, zero_division=0)),
        "c4_recall": float(recall_score(y, pred, zero_division=0)),
        "c4_average_precision": float(average_precision_score(y, score)),
        "c4_roc_auc": float(roc_auc_score(y, score)),
    }
    out = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"],
        "time_step": selected["time_step"],
        "actual_c4": y,
        "c4_score": score,
        "predicted_c4": pred,
    })
    out.to_csv(args.model_dir / "test_c4_predictions.csv", index=False)
    (args.model_dir / "test_c4_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
