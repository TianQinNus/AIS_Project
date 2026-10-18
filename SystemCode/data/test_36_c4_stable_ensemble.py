"""Bootstrap-stable C4 precision ensemble selected without Test labels."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import load
from test_31_time_robust_model import last_rows
from test_33_c4_recall_search import build_matrix, metrics


def rank_score(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def precision_objective(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    values = []
    for capacity in (0.01, 0.02, 0.05):
        k = max(1, int(np.ceil(capacity * len(y))))
        values.append(float(y[order[:k]].sum() / k))
    return 0.50 * values[0] + 0.30 * values[1] + 0.20 * values[2]


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "processed" / "test_35_c4_advanced"
    out = root / "processed" / "test_36_c4_stable"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    val = pd.read_csv(source / "validation_predictions.csv")
    y_val = (val["actual_class"].to_numpy(dtype=int) == 4).astype(np.int8)
    result = pd.read_csv(source / "validation_results.csv")
    top = result.head(10)["candidate"].astype(str).tolist()
    ranks = {name: rank_score(val[name].to_numpy()) for name in top}
    candidates = {name: ranks[name] for name in top}
    for i, left in enumerate(top):
        for right in top[i + 1:]:
            candidates[f"{left} + {right}"] = (ranks[left] + ranks[right]) / 2

    rng = np.random.default_rng(42)
    rows = []
    for name, score in candidates.items():
        boot = []
        for _ in range(300):
            idx = rng.integers(0, len(y_val), len(y_val))
            boot.append(precision_objective(y_val[idx], score[idx]))
        rows.append({
            "candidate": name,
            "point_objective": precision_objective(y_val, score),
            "bootstrap_mean": float(np.mean(boot)),
            "bootstrap_p10": float(np.quantile(boot, 0.10)),
            "bootstrap_std": float(np.std(boot)),
        })
    stability = pd.DataFrame(rows).sort_values(
        ["bootstrap_p10", "bootstrap_mean"], ascending=False).reset_index(drop=True)
    winner = str(stability.iloc[0]["candidate"])
    components = winner.split(" + ")
    stability.to_csv(out / "bootstrap_stability.csv", index=False)
    selection = {
        "winner": winner,
        "components": components,
        "selection_rule": "highest 10th-percentile bootstrap low-capacity precision objective on Validation",
        "winner_stability": stability.iloc[0].to_dict(),
    }
    (out / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Frozen bootstrap-stable winner:", winner, flush=True)

    models = joblib.load(source / "models.joblib")
    op = load(root / "raw", "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(root / "raw", "test_specifications.csv")
    labels = load(root / "raw", "test_labels.csv")
    selected = last_rows(op, labels)
    actual_class = selected["class_label"].to_numpy(dtype=int)
    y = (actual_class == 4).astype(np.int8)
    x, names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        (1, 3, 5), "temporal")
    if names != meta["feature_names"]:
        raise ValueError("Feature schema mismatch")

    def component_score(name: str) -> np.ndarray:
        fitted = models[name]
        if isinstance(fitted, dict):
            return fitted["model"].predict_proba(x[:, fitted["indices"]])[:, 1]
        return fitted.predict_proba(x)[:, 1]

    score = np.mean([rank_score(component_score(name)) for name in components], axis=0)
    report = metrics(y, score)
    report.update({
        "winner": winner, "vehicles": len(y), "c4_total": int(y.sum()),
        "warning": "Selected by Validation bootstrap stability. Official Test had been inspected in earlier experiments.",
    })
    pd.DataFrame({
        "vehicle_id": selected["vehicle_id"], "actual_class": actual_class,
        "c4_score": score,
        "c4_rank": pd.Series(score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(out / "test_predictions.csv", index=False)
    (out / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
