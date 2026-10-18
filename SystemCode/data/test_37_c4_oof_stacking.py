"""OOF-stacked C4 specialist selected on Validation, then frozen for Test."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from test21 import load
from test_31_time_robust_model import last_rows
from test_33_c4_recall_search import build_matrix, metrics


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def precision_objective(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    precision = []
    for capacity in (0.01, 0.02, 0.05):
        k = max(1, int(np.ceil(capacity * len(y))))
        precision.append(float(y[order[:k]].sum() / k))
    return 0.50 * precision[0] + 0.30 * precision[1] + 0.20 * precision[2]


def meta_features(score_frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    raw = np.column_stack([score_frame[name].to_numpy(float) for name in names])
    ranks = np.column_stack([percentile_rank(raw[:, i]) for i in range(raw.shape[1])])
    summary = np.column_stack([
        raw.mean(axis=1), raw.std(axis=1), raw.min(axis=1), raw.max(axis=1),
        ranks.mean(axis=1), ranks.std(axis=1), ranks.min(axis=1), ranks.max(axis=1),
    ])
    return np.column_stack([raw, ranks, summary])


def component_score(fitted: object, x: np.ndarray) -> np.ndarray:
    if isinstance(fitted, dict):
        return fitted["model"].predict_proba(x[:, fitted["indices"]])[:, 1]
    return fitted.predict_proba(x)[:, 1]


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "processed" / "test_35_c4_advanced"
    out = root / "processed" / "test_37_c4_oof_stacking"
    out.mkdir(parents=True, exist_ok=True)

    val_predictions = pd.read_csv(source / "validation_predictions.csv")
    val_results = pd.read_csv(source / "validation_results.csv")
    base_names = val_results.head(10)["candidate"].astype(str).tolist()
    y = (val_predictions["actual_class"].to_numpy(int) == 4).astype(np.int8)
    z = meta_features(val_predictions, base_names)

    configs = []
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        for positive_weight in (1.0, 2.0, 4.0, 8.0):
            configs.append((c, positive_weight))

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for c, positive_weight in configs:
        total = np.zeros(len(y), dtype=float)
        count = np.zeros(len(y), dtype=float)
        for train_idx, valid_idx in cv.split(z, y):
            weights = np.where(y[train_idx] == 1, positive_weight, 1.0)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=3000, solver="lbfgs"),
            )
            model.fit(z[train_idx], y[train_idx], logisticregression__sample_weight=weights)
            total[valid_idx] += model.predict_proba(z[valid_idx])[:, 1]
            count[valid_idx] += 1
        oof = total / count
        name = f"logit_C{c:g}_pw{positive_weight:g}"
        predictions[name] = oof
        rows.append({
            "candidate": name,
            "C": c,
            "positive_weight": positive_weight,
            "precision_objective": precision_objective(y, oof),
            "c4_ap": average_precision_score(y, oof),
            "c4_auc": roc_auc_score(y, oof),
        })

    result = pd.DataFrame(rows).sort_values(
        ["precision_objective", "c4_ap"], ascending=False).reset_index(drop=True)
    result.to_csv(out / "oof_results.csv", index=False)
    winner = result.iloc[0]
    winner_name = str(winner["candidate"])
    oof_score = predictions[winner_name]
    pd.DataFrame({
        "vehicle_id": val_predictions["vehicle_id"],
        "actual_class": val_predictions["actual_class"],
        "oof_c4_score": oof_score,
    }).to_csv(out / "validation_oof_predictions.csv", index=False)

    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(winner["C"]), max_iter=3000, solver="lbfgs"),
    )
    full_weights = np.where(y == 1, float(winner["positive_weight"]), 1.0)
    final.fit(z, y, logisticregression__sample_weight=full_weights)

    meta = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    models = joblib.load(source / "models.joblib")
    op = load(root / "raw", "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    spec = load(root / "raw", "test_specifications.csv")
    labels = load(root / "raw", "test_labels.csv")
    selected = last_rows(op, labels)
    actual_class = selected["class_label"].to_numpy(int)
    y_test = (actual_class == 4).astype(np.int8)
    x_test, feature_names = build_matrix(
        op, selected, spec, meta["spec_columns"], meta["raw_cols"],
        (1, 3, 5), "temporal")
    if feature_names != meta["feature_names"]:
        raise ValueError("Feature schema mismatch")
    base_test = pd.DataFrame({
        name: component_score(models[name], x_test) for name in base_names
    })
    z_test = meta_features(base_test, base_names)
    test_score = final.predict_proba(z_test)[:, 1]
    report = metrics(y_test, test_score)
    report.update({
        "winner": winner_name,
        "validation_oof_precision_objective": float(winner["precision_objective"]),
        "validation_oof_ap": float(winner["c4_ap"]),
        "base_models": base_names,
        "vehicles": len(y_test),
        "c4_total": int(y_test.sum()),
        "warning": "Meta-model selected by repeated OOF on Validation; Test remained frozen for this experiment.",
    })
    pd.DataFrame({
        "vehicle_id": selected["vehicle_id"],
        "actual_class": actual_class,
        "c4_score": test_score,
        "c4_rank": pd.Series(test_score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(out / "test_predictions.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "winner": winner_name,
        "selection_rule": "highest repeated-OOF Validation low-capacity precision objective",
        "base_models": base_names,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    joblib.dump({"model": final, "base_models": base_names}, out / "stacker.joblib")
    print("Frozen OOF stacker:", winner_name, flush=True)
    print(result.head(10).to_string(index=False), flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
