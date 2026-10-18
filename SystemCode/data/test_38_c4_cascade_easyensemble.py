"""C4 precision search with C3/C4 cascade, C0 bagging and false-positive veto."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from test21 import load
from test_31_time_robust_model import fit_spec_schema, last_rows
from test_33_c4_recall_search import build_matrix, class_weights, metrics, select_training, training_pool
from test_34_c4_precision_ensemble import combine_predictions


SEED = 42


def rank_score(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(float)


def xgb(seed: int, shallow: bool = False) -> XGBClassifier:
    if shallow:
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=350, learning_rate=0.03, max_depth=3, min_child_weight=10,
            gamma=0.1, subsample=0.85, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=6.0, random_state=seed, n_jobs=4)
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=300, learning_rate=0.04, max_depth=5, min_child_weight=5,
        gamma=0.05, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=3.0, random_state=seed, n_jobs=4)


def lgb(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=400, learning_rate=0.035,
        num_leaves=15, min_child_samples=35, colsample_bytree=0.8,
        subsample=0.8, subsample_freq=1, reg_lambda=2.0,
        verbosity=-1, random_state=seed, n_jobs=4)


def objective(report: dict) -> float:
    return float(0.25 * report["precision_top1pct"] +
                 0.60 * report["precision_top2pct"] +
                 0.15 * report["precision_top5pct"])


def add(name: str, score: np.ndarray, y: np.ndarray,
        rows: list[dict], predictions: dict[str, np.ndarray], recipe: dict) -> None:
    report = metrics(y, score)
    report["precision_objective"] = objective(report)
    rows.append({"candidate": name, **report, "recipe": json.dumps(recipe)})
    predictions[name] = score
    print(f"  {name:34s} obj={report['precision_objective']:.4f} "
          f"P1={report['precision_top1pct']:.4f} P2={report['precision_top2pct']:.4f} "
          f"P5={report['precision_top5pct']:.4f} AP={report['c4_ap']:.4f}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "raw"
    out = root / "processed" / "test_38_c4_cascade_easyensemble"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading train and Validation...", flush=True)
    tr_op = load(data, "train_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    va_op = load(data, "validation_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    tr_spec = load(data, "train_specifications.csv")
    va_spec = load(data, "validation_specifications.csv")
    tte = load(data, "train_tte.csv")
    va_labels = load(data, "validation_labels.csv")
    raw_cols = [c for c in tr_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(tr_spec)
    selected = select_training(training_pool(tr_op, tte), "dense10", SEED)
    va_sel = last_rows(va_op, va_labels)
    train_class = selected["class_label"].to_numpy(int)
    val_class = va_sel["class_label"].to_numpy(int)
    y_train = (train_class == 4).astype(np.int8)
    y_val = (val_class == 4).astype(np.int8)
    base_weight = selected["sw"].to_numpy(float)
    x_train, feature_names = build_matrix(
        tr_op, selected, tr_spec, schema, raw_cols, (1, 3, 5), "temporal")
    x_val, val_names = build_matrix(
        va_op, va_sel, va_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if feature_names != val_names:
        raise ValueError("Feature schema mismatch")

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    artifacts: dict[str, object] = {}
    recipes: dict[str, dict] = {}

    print("Stage A: existing precision ensemble reference...", flush=True)
    old_dir = root / "processed" / "test_34_c4_precision"
    old_models = {key: joblib.load(old_dir / f"{key}.joblib")
                  for key in ("xgb_42", "lgb_42", "xgb_77", "lgb_77", "xgb_2026", "lgb_2026")}
    old_val_base = {key: model.predict_proba(x_val)[:, 1] for key, model in old_models.items()}
    old_candidates = combine_predictions(old_val_base)
    baseline_score = old_candidates["pair_seed42_rank"]
    add("baseline_test34", baseline_score, y_val, rows, predictions,
        {"type": "old", "candidate": "pair_seed42_rank"})

    print("Stage B: C3/C4 severity cascade...", flush=True)
    severe_y = (train_class >= 3).astype(np.int8)
    severe_models = {}
    conditional_models = {}
    severe_mask = train_class >= 3
    conditional_y = (train_class[severe_mask] == 4).astype(np.int8)
    for family in ("xgb", "lgb"):
        sm = xgb(SEED, shallow=True) if family == "xgb" else lgb(SEED)
        cm = xgb(SEED + 1, shallow=False) if family == "xgb" else lgb(SEED + 1)
        sm.fit(x_train, severe_y,
               sample_weight=class_weights(severe_y, base_weight, 0.25))
        cm.fit(x_train[severe_mask], conditional_y,
               sample_weight=class_weights(conditional_y, base_weight[severe_mask], 0.0))
        severe_models[family] = sm
        conditional_models[family] = cm
        ps = sm.predict_proba(x_val)[:, 1]
        pc = cm.predict_proba(x_val)[:, 1]
        for method, score in {
            "product": ps * pc,
            "rank_equal": (rank_score(ps) + rank_score(pc)) / 2,
            "rank_cond70": 0.3 * rank_score(ps) + 0.7 * rank_score(pc),
            "baseline_cascade": 0.5 * rank_score(baseline_score) + 0.25 * rank_score(ps) + 0.25 * rank_score(pc),
        }.items():
            name = f"cascade_{family}_{method}"
            recipe = {"type": "cascade", "family": family, "method": method}
            recipes[name] = recipe
            add(name, score, y_val, rows, predictions, recipe)
    artifacts["severe_models"] = severe_models
    artifacts["conditional_models"] = conditional_models

    print("Stage C: repeated C0 down-sampling ensemble...", flush=True)
    rng = np.random.default_rng(SEED)
    positives = np.flatnonzero(y_train == 1)
    negatives = np.flatnonzero(y_train == 0)
    bag_models: dict[str, list[XGBClassifier]] = {}
    for ratio in (4, 8, 12):
        scores = []
        fitted = []
        take = min(len(negatives), ratio * len(positives))
        for seed in (42, 77, 311, 997, 2026):
            local = np.random.default_rng(seed + ratio)
            idx = np.r_[positives, local.choice(negatives, size=take, replace=False)]
            local.shuffle(idx)
            model = xgb(seed, shallow=True)
            model.fit(x_train[idx], y_train[idx], sample_weight=base_weight[idx])
            fitted.append(model)
            scores.append(rank_score(model.predict_proba(x_val)[:, 1]))
        key = f"easybag_ratio{ratio}"
        bag_models[key] = fitted
        score = np.mean(scores, axis=0)
        recipes[key] = {"type": "easybag", "key": key}
        add(key, score, y_val, rows, predictions, recipes[key])
        combo_name = f"baseline_plus_{key}"
        recipes[combo_name] = {"type": "baseline_easybag", "key": key}
        add(combo_name, 0.5 * rank_score(baseline_score) + 0.5 * rank_score(score),
            y_val, rows, predictions, recipes[combo_name])
    artifacts["bag_models"] = bag_models

    print("Stage D: false-positive veto models...", flush=True)
    for keep_classes, label in [((0, 3, 4), "c0c3c4"), ((0, 4), "c0c4")]:
        mask = np.isin(train_class, keep_classes)
        veto = xgb(SEED + 19, shallow=True)
        veto.fit(x_train[mask], y_train[mask],
                 sample_weight=class_weights(y_train[mask], base_weight[mask], 0.25))
        artifacts[f"veto_{label}"] = veto
        veto_score = veto.predict_proba(x_val)[:, 1]
        for weight in (0.35, 0.50, 0.65):
            name = f"veto_{label}_w{weight:g}"
            score = ((1 - weight) * rank_score(baseline_score) +
                     weight * rank_score(veto_score))
            recipe = {"type": "veto", "label": label, "weight": weight}
            recipes[name] = recipe
            add(name, score, y_val, rows, predictions, recipe)

    result = pd.DataFrame(rows).sort_values(
        ["precision_objective", "c4_ap"], ascending=False).reset_index(drop=True)
    winner = str(result.iloc[0]["candidate"])
    winner_recipe = recipes.get(winner, {"type": "old", "candidate": "pair_seed42_rank"})
    result.to_csv(out / "validation_results.csv", index=False)
    pd.DataFrame({"vehicle_id": va_sel["vehicle_id"], "actual_class": val_class,
                  **predictions}).to_csv(out / "validation_predictions.csv", index=False)
    joblib.dump(artifacts, out / "models.joblib")
    selection = {
        "winner": winner, "winner_recipe": winner_recipe,
        "winner_metrics": result.iloc[0].to_dict(),
        "selection_rule": "0.25*P@1% + 0.60*P@2% + 0.15*P@5%; AP tie-break",
        "raw_cols": raw_cols, "spec_columns": schema, "feature_names": feature_names,
    }
    (out / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("Frozen Validation winner:", winner, flush=True)

    print("Loading Test after selection...", flush=True)
    te_op = load(data, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    te_spec = load(data, "test_specifications.csv")
    te_labels = load(data, "test_labels.csv")
    te_sel = last_rows(te_op, te_labels)
    test_class = te_sel["class_label"].to_numpy(int)
    y_test = (test_class == 4).astype(np.int8)
    x_test, test_names = build_matrix(
        te_op, te_sel, te_spec, schema, raw_cols, (1, 3, 5), "temporal")
    if feature_names != test_names:
        raise ValueError("Test feature schema mismatch")
    old_test_base = {key: model.predict_proba(x_test)[:, 1] for key, model in old_models.items()}
    test_baseline = combine_predictions(old_test_base)["pair_seed42_rank"]

    recipe = winner_recipe
    if recipe["type"] == "old":
        test_score = test_baseline
    elif recipe["type"] == "cascade":
        family = recipe["family"]
        ps = severe_models[family].predict_proba(x_test)[:, 1]
        pc = conditional_models[family].predict_proba(x_test)[:, 1]
        if recipe["method"] == "product":
            test_score = ps * pc
        elif recipe["method"] == "rank_equal":
            test_score = (rank_score(ps) + rank_score(pc)) / 2
        elif recipe["method"] == "rank_cond70":
            test_score = 0.3 * rank_score(ps) + 0.7 * rank_score(pc)
        else:
            test_score = 0.5 * rank_score(test_baseline) + 0.25 * rank_score(ps) + 0.25 * rank_score(pc)
    elif recipe["type"] in ("easybag", "baseline_easybag"):
        bag = np.mean([rank_score(m.predict_proba(x_test)[:, 1])
                       for m in bag_models[recipe["key"]]], axis=0)
        test_score = bag if recipe["type"] == "easybag" else (
            0.5 * rank_score(test_baseline) + 0.5 * rank_score(bag))
    elif recipe["type"] == "veto":
        veto = artifacts[f"veto_{recipe['label']}"]
        vs = veto.predict_proba(x_test)[:, 1]
        w = recipe["weight"]
        test_score = (1 - w) * rank_score(test_baseline) + w * rank_score(vs)
    else:
        raise ValueError(recipe)

    report = metrics(y_test, test_score)
    report.update({"winner": winner, "vehicles": len(y_test), "c4_total": int(y_test.sum()),
                   "warning": "Winner frozen on Validation before Test scoring."})
    pd.DataFrame({
        "vehicle_id": te_sel["vehicle_id"], "actual_class": test_class,
        "c4_score": test_score,
        "c4_rank": pd.Series(test_score).rank(ascending=False, method="first").astype(int),
    }).sort_values("c4_rank").to_csv(out / "test_predictions.csv", index=False)
    (out / "test_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(result.head(12).to_string(index=False), flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
