"""Evaluate the validation-selected XGB/LGBM top-2% consensus policy."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import load
from test_31_time_robust_model import last_rows
from test_33_c4_recall_search import build_matrix


def main() -> None:
    root = Path(__file__).resolve().parent
    model_dir = root / "processed" / "test_34_c4_precision"
    meta = json.loads((model_dir / "selection.json").read_text(encoding="utf-8"))
    op = load(root / "raw", "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(root / "raw", "test_specifications.csv")
    labels = load(root / "raw", "test_labels.csv")
    selected = last_rows(op, labels)
    y = (selected["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    x, names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        (1, 3, 5), "temporal")
    if names != meta["feature_names"]:
        raise ValueError("Feature schema mismatch")
    xgb_score = joblib.load(model_dir / "xgb_42.joblib").predict_proba(x)[:, 1]
    lgb_score = joblib.load(model_dir / "lgb_42.joblib").predict_proba(x)[:, 1]
    n = len(y)
    k = int(np.ceil(0.02 * n))
    xgb_top = set(np.argsort(-xgb_score, kind="stable")[:k])
    lgb_top = set(np.argsort(-lgb_score, kind="stable")[:k])
    consensus = np.array(sorted(xgb_top & lgb_top), dtype=int)
    found = int(y[consensus].sum())
    result = {
        "policy": "intersection of XGBoost Top 2% and LightGBM Top 2%",
        "vehicles": n,
        "c4_total": int(y.sum()),
        "candidate_count": int(len(consensus)),
        "c4_found": found,
        "precision": float(found / max(len(consensus), 1)),
        "recall": float(found / max(int(y.sum()), 1)),
        "warning": "The consensus rule was chosen on validation before this Test calculation; official Test had been inspected in earlier project experiments.",
    }
    out = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"], "actual_c4": y,
        "xgb_c4_score": xgb_score, "lgb_c4_score": lgb_score,
        "xgb_top2pct": [i in xgb_top for i in range(n)],
        "lgb_top2pct": [i in lgb_top for i in range(n)],
        "c4_consensus": [i in set(consensus) for i in range(n)],
    })
    out.to_csv(model_dir / "test_consensus_predictions.csv", index=False)
    (model_dir / "test_consensus_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
