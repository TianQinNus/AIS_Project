"""Post-hoc test evaluation for the frozen Test 33 C4 recall model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import load
from test_31_time_robust_model import last_rows
from test_33_c4_recall_search import build_matrix, metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--model-dir", type=Path, default=root / "processed" / "test_33_c4_recall")
    args = ap.parse_args()

    meta = json.loads((args.model_dir / "selection.json").read_text(encoding="utf-8"))
    winner = meta["winner"]
    model = joblib.load(args.model_dir / "model.joblib")
    imputer_path = args.model_dir / "imputer.joblib"
    imputer = joblib.load(imputer_path) if imputer_path.exists() else None
    op = load(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(args.data_dir, "test_specifications.csv")
    labels = load(args.data_dir, "test_labels.csv")
    selected = last_rows(op, labels)
    y = (selected["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    lags = tuple(int(v) for v in str(winner["lags"]).split(","))
    x, names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        lags, str(winner["feature_mode"]))
    if names != meta["feature_names"]:
        raise ValueError("Test feature schema differs from frozen training schema")
    if imputer is not None:
        x = imputer.transform(x).astype(np.float32)
    score = model.predict_proba(x)[:, 1]
    report = metrics(y, score)
    report.update({
        "dataset": "official_test_posthoc_reuse",
        "warning": "Official test outcomes were inspected in earlier project experiments; this is a post-hoc comparison and was not used to alter the frozen Test 33 model.",
        "vehicles": int(len(y)),
        "c4_total": int(y.sum()),
        "winner": winner,
    })
    out = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"],
        "time_step": selected["time_step"],
        "actual_c4": y,
        "c4_score": score,
        "c4_rank": pd.Series(score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank")
    out.to_csv(args.model_dir / "test_predictions.csv", index=False)
    (args.model_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
