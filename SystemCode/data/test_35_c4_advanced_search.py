"""Advanced C4 precision search: tuning, hierarchy, hard negatives and rankers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from xgboost import XGBClassifier, XGBRanker

from test21 import load
from test_31_time_robust_model import fit_spec_schema, last_rows
from test_33_c4_recall_search import (
    build_matrix,
    class_weights,
    metrics,
    select_training,
    training_pool,
)


SEED = 42
XGB_CONFIGS = {
    "base": dict(n_estimators=300, learning_rate=0.04, max_depth=5,
                 min_child_weight=5, gamma=0.05, subsample=0.8,
                 colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=3.0),
    "shallow": dict(n_estimators=450, learning_rate=0.025, max_depth=3,
                    min_child_weight=10, gamma=0.10, subsample=0.85,
                    colsample_bytree=0.70, reg_alpha=0.2, reg_lambda=6.0),
    "depth4": dict(n_estimators=350, learning_rate=0.035, max_depth=4,
                   min_child_weight=8, gamma=0.10, subsample=0.8,
                   colsample_bytree=0.70, reg_alpha=0.2, reg_lambda=5.0),
    "conservative": dict(n_estimators=400, learning_rate=0.03, max_depth=5,
                         min_child_weight=15, gamma=0.20, subsample=0.85,
                         colsample_bytree=0.60, reg_alpha=0.5, reg_lambda=10.0),
    "deep": dict(n_estimators=250, learning_rate=0.04, max_depth=6,
                 min_child_weight=10, gamma=0.15, subsample=0.8,
                 colsample_bytree=0.70, reg_alpha=0.2, reg_lambda=8.0),
}


def classifier(params: dict, seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=4, **params)


def rank_percentile(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def precision_objective(report: dict) -> float:
    return float(0.50 * report["precision_top1pct"] +
                 0.30 * report["precision_top2pct"] +
                 0.20 * report["precision_top5pct"])


def add_candidate(name: str, score: np.ndarray, y: np.ndarray,
                  rows: list[dict], predictions: dict[str, np.ndarray], recipe: dict) -> None:
    report = metrics(y, score)
    report["precision_objective"] = precision_objective(report)
    rows.append({"candidate": name, **report, "recipe": json.dumps(recipe)})
    predictions[name] = score
    print(f"  {name:32s} obj={report['precision_objective']:.4f} P1={report['precision_top1pct']:.4f} P2={report['precision_top2pct']:.4f} P5={report['precision_top5pct']:.4f} AP={report['c4_ap']:.4f}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--output-dir", type=Path, default=root / "processed" / "test_35_c4_advanced")
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
    selected = select_training(training_pool(tr_op, tte), "dense10", SEED)
    va_sel = last_rows(va_op, va_labels)
    train_class = selected["class_label"].to_numpy(dtype=int)
    val_class = va_sel["class_label"].to_numpy(dtype=int)
    y_train = (train_class == 4).astype(np.int8)
    y_val = (val_class == 4).astype(np.int8)
    base_weight = selected["sw"].to_numpy(float)
    x_train, feature_names = build_matrix(
        tr_op, selected, tr_spec, schema, raw_cols, (1, 3, 5), "temporal")
    x_val, val_names = build_matrix(
        va_op, va_sel, va_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if feature_names != val_names:
        raise ValueError("Feature schema mismatch")

    models = {}
    recipes = {}
    rows = []
    val_predictions = {}

    print("Stage A: focused XGBoost tuning...", flush=True)
    for config, params in XGB_CONFIGS.items():
        exponents = (0.0, 0.25, 0.5) if config in ("base", "shallow") else (0.0, 0.25)
        for exponent in exponents:
            key = f"xgb_{config}_w{exponent:g}"
            fitted = classifier(params)
            fitted.fit(x_train, y_train,
                       sample_weight=class_weights(y_train, base_weight, exponent))
            models[key] = fitted
            recipes[key] = {"type": "model", "model": key, "features": "all"}
            add_candidate(key, fitted.predict_proba(x_val)[:, 1], y_val,
                          rows, val_predictions, recipes[key])

    # Training-only feature importance selection.
    base_key = "xgb_base_w0"
    importance = models[base_key].feature_importances_
    print("Stage B: importance-based feature selection...", flush=True)
    for top_n in (75, 150, 250, 350):
        idx = np.argsort(-importance)[:top_n]
        key = f"top{top_n}_features"
        fitted = classifier(XGB_CONFIGS["base"])
        fitted.fit(x_train[:, idx], y_train,
                   sample_weight=class_weights(y_train, base_weight, 0.0))
        models[key] = {"model": fitted, "indices": idx}
        recipes[key] = {"type": "model", "model": key, "features": "selected"}
        add_candidate(key, fitted.predict_proba(x_val[:, idx])[:, 1], y_val,
                      rows, val_predictions, recipes[key])

    print("Stage C: hierarchical at-risk then C4...", flush=True)
    risk_y = (train_class > 0).astype(np.int8)
    risk_model = classifier(XGB_CONFIGS["shallow"])
    risk_model.fit(x_train, risk_y, sample_weight=class_weights(risk_y, base_weight, 0.25))
    at_risk = train_class > 0
    cond_model = classifier(XGB_CONFIGS["base"])
    cond_y = (train_class[at_risk] == 4).astype(np.int8)
    cond_model.fit(x_train[at_risk], cond_y,
                   sample_weight=class_weights(cond_y, base_weight[at_risk], 0.0))
    models["risk48"] = risk_model
    models["conditional_c4"] = cond_model
    p_risk = risk_model.predict_proba(x_val)[:, 1]
    p_cond = cond_model.predict_proba(x_val)[:, 1]
    direct = val_predictions[base_key]
    for name, score, recipe in [
        ("hierarchy_product", p_risk * p_cond,
         {"type": "hierarchy", "method": "product"}),
        ("hierarchy_rank", (rank_percentile(p_risk) + rank_percentile(p_cond)) / 2,
         {"type": "hierarchy", "method": "rank"}),
        ("direct_plus_hierarchy", (rank_percentile(direct) + rank_percentile(p_risk * p_cond)) / 2,
         {"type": "hierarchy", "method": "direct_product_rank"}),
    ]:
        recipes[name] = recipe
        add_candidate(name, score, y_val, rows, val_predictions, recipe)

    print("Stage D: hard-negative mining...", flush=True)
    train_base_score = models[base_key].predict_proba(x_train)[:, 1]
    negatives = np.flatnonzero(y_train == 0)
    positives = np.flatnonzero(y_train == 1)
    for fraction in (0.05, 0.10, 0.20, 0.35):
        k = max(1, int(round(fraction * len(negatives))))
        hard = negatives[np.argsort(-train_base_score[negatives])[:k]]
        idx = np.r_[positives, hard]
        key = f"hardneg_{int(fraction*100)}pct"
        fitted = classifier(XGB_CONFIGS["shallow"])
        fitted.fit(x_train[idx], y_train[idx], sample_weight=base_weight[idx])
        models[key] = fitted
        rerank = fitted.predict_proba(x_val)[:, 1]
        recipes[key] = {"type": "hard_negative", "model": key, "fraction": fraction,
                        "combine": "base_reranker_rank"}
        score = (rank_percentile(direct) + rank_percentile(rerank)) / 2
        add_candidate(key, score, y_val, rows, val_predictions, recipes[key])

    print("Stage E: pairwise/listwise ranking...", flush=True)
    for objective in ("rank:pairwise", "rank:ndcg"):
        key = objective.replace(":", "_")
        fitted = XGBRanker(
            objective=objective, n_estimators=350, learning_rate=0.035,
            max_depth=4, min_child_weight=8, gamma=0.1, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=5.0,
            tree_method="hist", random_state=SEED, n_jobs=4,
        )
        fitted.fit(x_train, y_train, group=np.array([len(y_train)]))
        models[key] = fitted
        recipes[key] = {"type": "ranker", "model": key}
        add_candidate(key, fitted.predict(x_val), y_val, rows, val_predictions, recipes[key])
    lgb_rank = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=400,
        learning_rate=0.035, num_leaves=15, min_child_samples=35,
        colsample_bytree=0.8, reg_lambda=2.0, verbosity=-1,
        random_state=SEED, n_jobs=4,
    )
    query_size = 5000
    lgb_groups = [query_size] * (len(y_train) // query_size)
    if len(y_train) % query_size:
        lgb_groups.append(len(y_train) % query_size)
    lgb_rank.fit(x_train, y_train, group=lgb_groups)
    models["lgb_lambdarank"] = lgb_rank
    recipes["lgb_lambdarank"] = {"type": "ranker", "model": "lgb_lambdarank"}
    add_candidate("lgb_lambdarank", lgb_rank.predict(x_val), y_val,
                  rows, val_predictions, recipes["lgb_lambdarank"])

    result = pd.DataFrame(rows).sort_values(
        ["precision_objective", "c4_ap"], ascending=False).reset_index(drop=True)
    winner = str(result.iloc[0]["candidate"])
    result.to_csv(args.output_dir / "validation_results.csv", index=False)
    pd.DataFrame({"vehicle_id": va_sel["vehicle_id"], "actual_class": val_class,
                  **val_predictions}).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    joblib.dump(models, args.output_dir / "models.joblib")
    selection = {
        "winner": winner,
        "winner_metrics": result.iloc[0].to_dict(),
        "winner_recipe": recipes[winner],
        "selection_rule": "0.50*Precision@1% + 0.30*Precision@2% + 0.20*Precision@5%; AP tie-break",
        "raw_cols": raw_cols,
        "spec_columns": schema,
        "feature_names": feature_names,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Freeze first, then open post-hoc test.
    print("Frozen winner:", winner, "Loading Test...", flush=True)
    te_op = load(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    te_spec = load(args.data_dir, "test_specifications.csv")
    te_labels = load(args.data_dir, "test_labels.csv")
    te_sel = last_rows(te_op, te_labels)
    test_class = te_sel["class_label"].to_numpy(dtype=int)
    y_test = (test_class == 4).astype(np.int8)
    x_test, test_names = build_matrix(
        te_op, te_sel, te_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if test_names != feature_names:
        raise ValueError("Test feature schema mismatch")

    def model_score(key: str) -> np.ndarray:
        obj = models[key]
        if isinstance(obj, dict):
            return obj["model"].predict_proba(x_test[:, obj["indices"]])[:, 1]
        if isinstance(obj, (XGBRanker, LGBMRanker)):
            return obj.predict(x_test)
        return obj.predict_proba(x_test)[:, 1]

    recipe = recipes[winner]
    if recipe["type"] in ("model", "ranker"):
        test_score = model_score(recipe["model"])
    elif recipe["type"] == "hard_negative":
        test_score = (rank_percentile(model_score(base_key)) +
                      rank_percentile(model_score(recipe["model"]))) / 2
    elif recipe["type"] == "hierarchy":
        a = model_score("risk48")
        b = model_score("conditional_c4")
        if recipe["method"] == "product":
            test_score = a * b
        elif recipe["method"] == "rank":
            test_score = (rank_percentile(a) + rank_percentile(b)) / 2
        else:
            test_score = (rank_percentile(model_score(base_key)) +
                          rank_percentile(a * b)) / 2
    else:
        raise ValueError(recipe)
    test_report = metrics(y_test, test_score)
    test_report.update({"winner": winner, "vehicles": len(y_test),
                        "c4_total": int(y_test.sum()),
                        "warning": "Frozen on Validation. Official Test had been inspected in earlier experiments."})
    pd.DataFrame({"vehicle_id": te_sel["vehicle_id"], "actual_class": test_class,
                  "c4_score": test_score,
                  "c4_rank": pd.Series(test_score).rank(ascending=False, method="first").astype(int)}
                 ).sort_values("c4_rank").to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n=== TEST ===")
    print(json.dumps(test_report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
