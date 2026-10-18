"""Sensitivity analysis for the post-hoc C0/C4 veto blend."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import load
from test_31_time_robust_model import last_rows
from test_33_c4_recall_search import build_matrix, metrics
from test_34_c4_precision_ensemble import combine_predictions
from test_38_c4_cascade_easyensemble import rank_score


def split_scores(root: Path, split: str, meta: dict, artifacts: dict, old_models: dict):
    op = load(root / "raw", f"{split}_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(root / "raw", f"{split}_specifications.csv")
    labels = load(root / "raw", f"{split}_labels.csv")
    selected = last_rows(op, labels)
    actual = selected["class_label"].to_numpy(int)
    x, names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        (1, 3, 5), "temporal")
    if names != meta["feature_names"]:
        raise ValueError("Feature schema mismatch")
    old = {key: model.predict_proba(x)[:, 1] for key, model in old_models.items()}
    baseline = combine_predictions(old)["pair_seed42_rank"]
    veto = artifacts["veto_c0c4"].predict_proba(x)[:, 1]
    return selected, actual, rank_score(baseline), rank_score(veto)


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "processed" / "test_39_c4_veto_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "processed" / "test_38_c4_cascade_easyensemble"
    meta = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    artifacts = joblib.load(source / "models.joblib")
    old_dir = root / "processed" / "test_34_c4_precision"
    old_models = {key: joblib.load(old_dir / f"{key}.joblib")
                  for key in ("xgb_42", "lgb_42", "xgb_77", "lgb_77", "xgb_2026", "lgb_2026")}

    val_sel, val_actual, val_base, val_veto = split_scores(
        root, "validation", meta, artifacts, old_models)
    test_sel, test_actual, test_base, test_veto = split_scores(
        root, "test", meta, artifacts, old_models)
    rows = []
    score_cache = {}
    for method in ("linear", "geometric", "minimum"):
        for weight in np.round(np.arange(0, 1.0001, 0.025), 3):
            if method == "linear":
                val_score = (1 - weight) * val_base + weight * val_veto
                test_score = (1 - weight) * test_base + weight * test_veto
            elif method == "geometric":
                val_score = np.power(val_base, 1 - weight) * np.power(val_veto, weight)
                test_score = np.power(test_base, 1 - weight) * np.power(test_veto, weight)
            else:
                val_score = (1 - weight) * val_base + weight * np.minimum(val_base, val_veto)
                test_score = (1 - weight) * test_base + weight * np.minimum(test_base, test_veto)
            vm = metrics((val_actual == 4).astype(np.int8), val_score)
            tm = metrics((test_actual == 4).astype(np.int8), test_score)
            name = f"{method}_w{weight:.3f}"
            score_cache[name] = test_score
            rows.append({
                "candidate": name, "method": method, "weight": weight,
                **{f"val_{k}": v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            })
    result = pd.DataFrame(rows)
    result["robust_top2"] = np.minimum(
        result["val_precision_top2pct"], result["test_precision_top2pct"])
    result = result.sort_values(
        ["test_precision_top2pct", "test_precision_top5pct", "test_c4_ap"],
        ascending=False).reset_index(drop=True)
    result.to_csv(out / "weight_sensitivity.csv", index=False)
    best = result.iloc[0]
    name = str(best["candidate"])
    pd.DataFrame({
        "vehicle_id": test_sel["vehicle_id"], "actual_class": test_actual,
        "c4_score": score_cache[name],
        "c4_rank": pd.Series(score_cache[name]).rank(
            ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(out / "posthoc_best_predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "posthoc_winner": name,
        "metrics": best.to_dict(),
        "warning": "Weight and combination method compared on Test; post-hoc only.",
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    cols = ["candidate", "val_found_top2pct", "val_precision_top2pct",
            "test_found_top2pct", "test_precision_top2pct",
            "test_found_top5pct", "test_precision_top5pct", "test_c4_ap"]
    print(result[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
