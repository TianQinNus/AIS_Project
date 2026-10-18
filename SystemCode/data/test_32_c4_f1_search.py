"""C4-focused imbalance, feature and model search.

The official test set is intentionally not loaded. Sampling choices and the C4
decision rate are developed with a vehicle-grouped train split, then evaluated
on official validation vehicles. This prevents repeated optimisation on test.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from test21 import build_features, choose_rows_multi, detect_groups, load
from test_31_time_robust_model import (
    fit_spec_schema,
    percentile_columns,
    prepare_dynamic,
    spec_matrix,
)


SEED = 42

SAMPLING_CONFIGS = {
    "baseline": ("baseline", None),
    "weight_exp_025": ("weight", 0.25),
    "weight_exp_050": ("weight", 0.50),
    "weight_exp_075": ("weight", 0.75),
    "weight_exp_100": ("weight", 1.00),
    "downsample_neg_10to1": ("downsample", 10.0),
    "downsample_neg_5to1": ("downsample", 5.0),
    "random_over_20pct": ("random_over", 0.20),
    "random_over_50pct": ("random_over", 0.50),
    "smote_20pct": ("smote", 0.20),
    "smote_50pct": ("smote", 0.50),
}

FEATURE_VARIANTS = {
    "raw_all": {"drop_blocks": (), "rank_blocks": (), "short_only": False},
    "rank_rate_ratio": {"drop_blocks": (), "rank_blocks": ("rate", "ratio"), "short_only": False},
    "no_lifetime_rate": {"drop_blocks": ("rate",), "rank_blocks": (), "short_only": False},
    "recent_short": {"drop_blocks": (), "rank_blocks": ("rate", "ratio"), "short_only": True},
}


def make_model(kind: str, seed: int):
    if kind == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=220, learning_rate=0.05, max_depth=5,
            min_child_weight=5, gamma=0.05, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=3.0,
            tree_method="hist", random_state=seed, n_jobs=4,
        )
    if kind == "lightgbm":
        return LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.04,
            num_leaves=15, max_depth=-1, min_child_samples=45,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            reg_lambda=2.0, verbosity=-1, random_state=seed, n_jobs=4,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def feature_variant(dynamic: np.ndarray, names: list[str], blocks: np.ndarray,
                    spec: np.ndarray, variant: str) -> tuple[np.ndarray, list[str]]:
    cfg = FEATURE_VARIANTS[variant]
    keep = ~np.isin(blocks, cfg["drop_blocks"])
    if cfg["short_only"]:
        keep &= np.array([
            not (name.startswith("mix20:") or name.startswith("ratio20:"))
            for name in names
        ])
    x = dynamic[:, keep].astype(np.float32, copy=True)
    kept_blocks = blocks[keep]
    x = percentile_columns(x, np.isin(kept_blocks, cfg["rank_blocks"]))
    kept_names = [name for name, use in zip(names, keep) if use]
    return np.hstack([x, spec]).astype(np.float32), kept_names


def vehicle_split(selected: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    group = pd.DataFrame({"vehicle_id": selected["vehicle_id"], "c4": y == 1})
    summary = group.groupby("vehicle_id", sort=False)["c4"].max().reset_index()
    fit_ids, tune_ids = train_test_split(
        summary["vehicle_id"].to_numpy(), test_size=0.20, random_state=seed,
        stratify=summary["c4"].to_numpy(),
    )
    fit_mask = selected["vehicle_id"].isin(fit_ids).to_numpy()
    tune_mask = selected["vehicle_id"].isin(tune_ids).to_numpy()
    if np.any(fit_mask & tune_mask):
        raise RuntimeError("Vehicle leakage in internal split")
    return fit_mask, tune_mask


def resample(x: np.ndarray, y: np.ndarray, base_weight: np.ndarray,
             config: str, seed: int):
    method, value = SAMPLING_CONFIGS[config]
    imputer = None
    if method == "baseline":
        return x, y, base_weight.copy(), imputer
    if method == "weight":
        ratio = max(int((y == 0).sum()), 1) / max(int((y == 1).sum()), 1)
        weight = base_weight * np.where(y == 1, ratio ** float(value), 1.0)
        return x, y, weight, imputer
    if method == "downsample":
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        keep_neg = rng.choice(neg, size=min(len(neg), int(round(float(value) * len(pos)))), replace=False)
        idx = np.r_[pos, keep_neg]
        rng.shuffle(idx)
        return x[idx], y[idx], base_weight[idx], imputer
    if method == "random_over":
        sampler = RandomOverSampler(sampling_strategy=float(value), random_state=seed)
        xr, yr = sampler.fit_resample(x, y)
        wr = base_weight[sampler.sample_indices_]
        return xr.astype(np.float32), yr.astype(np.int8), wr, imputer
    if method == "smote":
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        xi = imputer.fit_transform(x).astype(np.float32)
        sampler = SMOTE(sampling_strategy=float(value), random_state=seed, k_neighbors=5)
        xr, yr = sampler.fit_resample(xi, y)
        return xr.astype(np.float32), yr.astype(np.int8), np.ones(len(yr)), imputer
    raise ValueError(method)


def transform_if_needed(x: np.ndarray, imputer: SimpleImputer | None) -> np.ndarray:
    return imputer.transform(x).astype(np.float32) if imputer is not None else x


def best_alert_fraction(y: np.ndarray, score: np.ndarray, max_fraction: float = 0.20) -> tuple[float, float]:
    order = np.argsort(-score, kind="stable")
    max_k = max(1, int(np.ceil(max_fraction * len(y))))
    tp = np.cumsum(y[order][:max_k])
    k = np.arange(1, max_k + 1)
    f1 = 2 * tp / np.maximum(k + int(y.sum()), 1)
    best = int(np.argmax(f1))
    return float((best + 1) / len(y)), float(f1[best])


def evaluate(y: np.ndarray, score: np.ndarray, fraction: float) -> dict:
    k = max(1, min(len(y), int(round(fraction * len(y)))))
    pred = np.zeros(len(y), dtype=np.int8)
    pred[np.argsort(-score, kind="stable")[:k]] = 1
    return {
        "c4_f1": float(f1_score(y, pred, zero_division=0)),
        "c4_precision": float(precision_score(y, pred, zero_division=0)),
        "c4_recall": float(recall_score(y, pred, zero_division=0)),
        "c4_ap": float(average_precision_score(y, score)),
        "c4_roc_auc": float(roc_auc_score(y, score)),
        "alert_fraction": float(fraction),
        "alert_count": int(k),
        "c4_found": int((pred & y).sum()),
    }


def best_score_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    if not len(thresholds):
        return 0.5, 0.0
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])


def crossfit_threshold_metrics(y: np.ndarray, score: np.ndarray, seed: int) -> dict:
    """Every validation vehicle is classified by a threshold learned without it."""
    pred = np.zeros(len(y), dtype=np.int8)
    thresholds = []
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fit, held in cv.split(score, y):
        threshold, _ = best_score_threshold(y[fit], score[fit])
        thresholds.append(threshold)
        pred[held] = score[held] >= threshold
    deployment_threshold, apparent_f1 = best_score_threshold(y, score)
    return {
        "c4_f1_cv_threshold": float(f1_score(y, pred, zero_division=0)),
        "c4_precision_cv_threshold": float(precision_score(y, pred, zero_division=0)),
        "c4_recall_cv_threshold": float(recall_score(y, pred, zero_division=0)),
        "cv_alert_count": int(pred.sum()),
        "deployment_threshold": deployment_threshold,
        "apparent_full_validation_f1": apparent_f1,
        "median_fold_threshold": float(np.median(thresholds)),
    }


def fit_candidate(x_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
                  x_val: np.ndarray, y_val: np.ndarray, fit_mask: np.ndarray,
                  tune_mask: np.ndarray, sampling: str, model_kind: str,
                  seed: int) -> tuple[dict, np.ndarray, object, float]:
    xf, yf, wf, imp = resample(x_train[fit_mask], y_train[fit_mask],
                               w_train[fit_mask], sampling, seed)
    model = make_model(model_kind, seed)
    model.fit(xf, yf, sample_weight=wf)
    tune_score = model.predict_proba(transform_if_needed(x_train[tune_mask], imp))[:, 1]
    fraction, internal_f1 = best_alert_fraction(y_train[tune_mask], tune_score)
    del model, xf, yf, wf, tune_score
    gc.collect()

    xa, ya, wa, final_imp = resample(x_train, y_train, w_train, sampling, seed)
    final_model = make_model(model_kind, seed)
    final_model.fit(xa, ya, sample_weight=wa)
    val_score = final_model.predict_proba(transform_if_needed(x_val, final_imp))[:, 1]
    metrics = evaluate(y_val, val_score, fraction)
    metrics.update(crossfit_threshold_metrics(y_val, val_score, seed))
    metrics["internal_tune_f1"] = internal_f1
    metrics["fit_rows_after_sampling"] = int(len(ya))
    metrics["positive_rows_after_sampling"] = int(ya.sum())
    return metrics, val_score, (final_model, final_imp), metrics["deployment_threshold"]


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--output-dir", type=Path, default=root / "processed" / "test_32_c4_f1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lags", default="5,20")
    ap.add_argument("--k-repair", type=int, default=3)
    ap.add_argument("--k-nonrepair", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lags = [int(v) for v in args.lags.split(",") if v]

    print("Loading train and official validation only...", flush=True)
    tr_op = load(args.data_dir, "train_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    va_op = load(args.data_dir, "validation_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    tr_spec = load(args.data_dir, "train_specifications.csv")
    va_spec = load(args.data_dir, "validation_specifications.csv")
    tte = load(args.data_dir, "train_tte.csv")
    va_labels = load(args.data_dir, "validation_labels.csv")
    raw_cols = [c for c in tr_op if c not in ("vehicle_id", "time_step")]
    tr_sel, va_sel = choose_rows_multi(tr_op, tte, va_op, va_labels,
                                       args.k_repair, args.k_nonrepair)
    y_train = (tr_sel["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    y_val = (va_sel["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    w_train = tr_sel["sw"].to_numpy(dtype=float)
    fit_mask, tune_mask = vehicle_split(tr_sel, y_train, args.seed)
    print("C4 counts:", {"train": int(y_train.sum()), "internal_fit": int(y_train[fit_mask].sum()),
                         "internal_tune": int(y_train[tune_mask].sum()), "validation": int(y_val.sum())}, flush=True)

    d_train, dynamic_names, blocks = prepare_dynamic(tr_op, tr_sel, raw_cols, lags)
    d_val, val_names, val_blocks = prepare_dynamic(va_op, va_sel, raw_cols, lags)
    if dynamic_names != val_names or not np.array_equal(blocks, val_blocks):
        raise ValueError("Dynamic schemas differ")
    del tr_op, va_op
    gc.collect()
    schema = fit_spec_schema(tr_spec)
    s_train = spec_matrix(tr_spec, tr_sel["vehicle_id"].to_numpy(), schema)
    s_val = spec_matrix(va_spec, va_sel["vehicle_id"].to_numpy(), schema)

    matrices = {}
    variant_names = {}
    for variant in FEATURE_VARIANTS:
        matrices[variant] = (
            feature_variant(d_train, dynamic_names, blocks, s_train, variant)[0],
            feature_variant(d_val, dynamic_names, blocks, s_val, variant)[0],
        )
        variant_names[variant] = feature_variant(
            d_train[:1], dynamic_names, blocks, s_train[:1], variant)[1]
        print(variant, "features:", matrices[variant][0].shape[1], flush=True)

    rows = []
    predictions = []
    fitted = {}
    print("\nStage A: imbalance strategy screening on raw_all + XGBoost", flush=True)
    for sampling in SAMPLING_CONFIGS:
        x_train, x_val = matrices["raw_all"]
        metrics, score, artifact, threshold = fit_candidate(
            x_train=x_train, y_train=y_train, w_train=w_train,
            x_val=x_val, y_val=y_val,
            fit_mask=fit_mask, tune_mask=tune_mask, sampling=sampling,
            model_kind="xgboost", seed=args.seed,
        )
        row = {"stage": "sampling", "feature_variant": "raw_all", "model": "xgboost",
               "sampling": sampling, **metrics}
        rows.append(row)
        predictions.append(pd.DataFrame({"candidate": f"raw_all|xgboost|{sampling}",
                                         "vehicle_id": va_sel["vehicle_id"],
                                         "actual_c4": y_val, "c4_score": score}))
        fitted[f"raw_all|xgboost|{sampling}"] = (artifact, threshold)
        print(f"  {sampling:24s} cvF1={metrics['c4_f1_cv_threshold']:.4f} P={metrics['c4_precision_cv_threshold']:.4f} R={metrics['c4_recall_cv_threshold']:.4f} AP={metrics['c4_ap']:.4f}", flush=True)

    stage_a = pd.DataFrame(rows).sort_values(["c4_f1_cv_threshold", "c4_ap"], ascending=False)
    best_sampling = str(stage_a.iloc[0]["sampling"])
    print("Best imbalance strategy:", best_sampling, flush=True)

    print("\nStage B: feature and model screening with frozen imbalance strategy", flush=True)
    for variant in FEATURE_VARIANTS:
        for model_kind in ("xgboost", "lightgbm"):
            key = f"{variant}|{model_kind}|{best_sampling}"
            if key in fitted:
                continue
            x_train, x_val = matrices[variant]
            metrics, score, artifact, threshold = fit_candidate(
                x_train=x_train, y_train=y_train, w_train=w_train,
                x_val=x_val, y_val=y_val,
                fit_mask=fit_mask, tune_mask=tune_mask, sampling=best_sampling,
                model_kind=model_kind, seed=args.seed,
            )
            row = {"stage": "feature_model", "feature_variant": variant,
                   "model": model_kind, "sampling": best_sampling, **metrics}
            rows.append(row)
            predictions.append(pd.DataFrame({"candidate": key,
                                             "vehicle_id": va_sel["vehicle_id"],
                                             "actual_c4": y_val, "c4_score": score}))
            fitted[key] = (artifact, threshold)
            print(f"  {variant:20s} {model_kind:8s} cvF1={metrics['c4_f1_cv_threshold']:.4f} P={metrics['c4_precision_cv_threshold']:.4f} R={metrics['c4_recall_cv_threshold']:.4f} AP={metrics['c4_ap']:.4f}", flush=True)

    print("\nStage C: interactions between the top sampling choices and strongest temporal features", flush=True)
    top_sampling = stage_a.head(3)["sampling"].astype(str).tolist()
    for sampling in top_sampling:
        for variant in ("rank_rate_ratio", "recent_short"):
            model_kind = "xgboost"
            key = f"{variant}|{model_kind}|{sampling}"
            if key in fitted:
                continue
            x_train, x_val = matrices[variant]
            metrics, score, artifact, threshold = fit_candidate(
                x_train=x_train, y_train=y_train, w_train=w_train,
                x_val=x_val, y_val=y_val,
                fit_mask=fit_mask, tune_mask=tune_mask, sampling=sampling,
                model_kind=model_kind, seed=args.seed,
            )
            row = {"stage": "interaction", "feature_variant": variant,
                   "model": model_kind, "sampling": sampling, **metrics}
            rows.append(row)
            predictions.append(pd.DataFrame({"candidate": key,
                                             "vehicle_id": va_sel["vehicle_id"],
                                             "actual_c4": y_val, "c4_score": score}))
            fitted[key] = (artifact, threshold)
            print(f"  {variant:20s} {sampling:24s} cvF1={metrics['c4_f1_cv_threshold']:.4f} P={metrics['c4_precision_cv_threshold']:.4f} R={metrics['c4_recall_cv_threshold']:.4f} AP={metrics['c4_ap']:.4f}", flush=True)

    result = pd.DataFrame(rows).sort_values(
        ["c4_f1_cv_threshold", "c4_ap"], ascending=False).reset_index(drop=True)
    winner = result.iloc[0].to_dict()
    winner_key = f"{winner['feature_variant']}|{winner['model']}|{winner['sampling']}"
    (winner_model, winner_imputer), winner_threshold = fitted[winner_key]
    joblib.dump(winner_model, args.output_dir / "c4_specialist_model.joblib")
    if winner_imputer is not None:
        joblib.dump(winner_imputer, args.output_dir / "c4_specialist_imputer.joblib")
    result.to_csv(args.output_dir / "validation_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "validation_predictions.csv", index=False)
    metadata = {
        "objective": "maximise C4 F1",
        "test_set_used": False,
        "selection_data": "official validation with five-fold cross-fitted C4 thresholds; each vehicle is scored using a threshold learned without that vehicle",
        "winner": winner,
        "winner_key": winner_key,
        "c4_probability_threshold": winner_threshold,
        "feature_definition": FEATURE_VARIANTS[str(winner["feature_variant"])],
        "sampling_definition": SAMPLING_CONFIGS[str(winner["sampling"])],
        "raw_cols": raw_cols,
        "lags": lags,
        "dynamic_names": variant_names[str(winner["feature_variant"])],
        "spec_columns": schema,
        "validation_c4_count": int(y_val.sum()),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== SELECTED C4 SPECIALIST (VALIDATION) ===")
    print(json.dumps(winner, indent=2, ensure_ascii=False, default=float))
    print("Test set was not loaded. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
