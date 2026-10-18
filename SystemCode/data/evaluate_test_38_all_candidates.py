"""Post-hoc Test comparison for every Test 38 candidate."""
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
from test_38_c4_cascade_easyensemble import objective, rank_score


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "processed" / "test_38_c4_cascade_easyensemble"
    meta = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    artifacts = joblib.load(out / "models.joblib")
    old_dir = root / "processed" / "test_34_c4_precision"
    old_models = {key: joblib.load(old_dir / f"{key}.joblib")
                  for key in ("xgb_42", "lgb_42", "xgb_77", "lgb_77", "xgb_2026", "lgb_2026")}

    op = load(root / "raw", "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(root / "raw", "test_specifications.csv")
    labels = load(root / "raw", "test_labels.csv")
    selected = last_rows(op, labels)
    actual = selected["class_label"].to_numpy(int)
    y = (actual == 4).astype(np.int8)
    x, names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        (1, 3, 5), "temporal")
    if names != meta["feature_names"]:
        raise ValueError("Feature schema mismatch")

    old_base = {key: model.predict_proba(x)[:, 1] for key, model in old_models.items()}
    baseline = combine_predictions(old_base)["pair_seed42_rank"]
    val_result = pd.read_csv(out / "validation_results.csv")
    rows = []
    score_columns = {}
    for _, row in val_result.iterrows():
        name = str(row["candidate"])
        recipe = json.loads(row["recipe"])
        if recipe["type"] == "old":
            score = baseline
        elif recipe["type"] == "cascade":
            family = recipe["family"]
            ps = artifacts["severe_models"][family].predict_proba(x)[:, 1]
            pc = artifacts["conditional_models"][family].predict_proba(x)[:, 1]
            method = recipe["method"]
            if method == "product":
                score = ps * pc
            elif method == "rank_equal":
                score = (rank_score(ps) + rank_score(pc)) / 2
            elif method == "rank_cond70":
                score = 0.3 * rank_score(ps) + 0.7 * rank_score(pc)
            else:
                score = 0.5 * rank_score(baseline) + 0.25 * rank_score(ps) + 0.25 * rank_score(pc)
        elif recipe["type"] in ("easybag", "baseline_easybag"):
            bag = np.mean([rank_score(m.predict_proba(x)[:, 1])
                           for m in artifacts["bag_models"][recipe["key"]]], axis=0)
            score = bag if recipe["type"] == "easybag" else (
                0.5 * rank_score(baseline) + 0.5 * rank_score(bag))
        elif recipe["type"] == "veto":
            vs = artifacts[f"veto_{recipe['label']}"] .predict_proba(x)[:, 1]
            weight = float(recipe["weight"])
            score = (1 - weight) * rank_score(baseline) + weight * rank_score(vs)
        else:
            raise ValueError(recipe)
        report = metrics(y, score)
        report["precision_objective"] = objective(report)
        rows.append({"candidate": name, **report})
        score_columns[name] = score

    result = pd.DataFrame(rows).sort_values(
        ["precision_objective", "c4_ap"], ascending=False).reset_index(drop=True)
    result.to_csv(out / "posthoc_test_all_candidates.csv", index=False)
    winner = str(result.iloc[0]["candidate"])
    score = score_columns[winner]
    pd.DataFrame({
        "vehicle_id": selected["vehicle_id"], "actual_class": actual,
        "c4_score": score,
        "c4_rank": pd.Series(score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(out / "posthoc_test_best_predictions.csv", index=False)
    summary = {
        "posthoc_winner": winner,
        "metrics": result.iloc[0].to_dict(),
        "warning": "Selected after comparing Test labels; descriptive/post-hoc only.",
    }
    (out / "posthoc_test_best.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(result.head(15).to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
