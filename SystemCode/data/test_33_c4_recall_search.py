"""Recall-oriented C4 search with dense failure-window sampling and short deltas."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from test21 import build_features, detect_groups, load, vehicle_start
from test_31_time_robust_model import fit_spec_schema, last_rows, spec_matrix


SEED = 42
CAPACITIES = (0.01, 0.02, 0.05, 0.10, 0.20)

CANDIDATES = [
    ("standard_lag5_20", "standard", (5, 20), "temporal"),
    ("dense10_lag5_20", "dense10", (5, 20), "temporal"),
    ("denseall_lag5_20", "denseall", (5, 20), "temporal"),
    ("dense10_lag1_3_5", "dense10", (1, 3, 5), "temporal"),
    ("dense10_short_current", "dense10", (1, 3, 5), "current"),
    ("dense10_short_delta", "dense10", (1, 3, 5), "current_delta"),
    ("denseall_short_delta", "denseall", (1, 3, 5), "current_delta"),
]


def tte_label(values: np.ndarray) -> np.ndarray:
    return np.select([values <= 6, values <= 12, values <= 24, values <= 48],
                     [4, 3, 2, 1], default=0).astype(np.int8)


def training_pool(op: pd.DataFrame, tte: pd.DataFrame) -> pd.DataFrame:
    merged = op[["vehicle_id", "time_step"]].merge(
        tte, on="vehicle_id", how="left", validate="many_to_one")
    remaining = merged["length_of_study_time_step"].to_numpy() - merged["time_step"].to_numpy()
    repaired = merged["in_study_repair"].to_numpy() == 1
    eligible = (repaired & (remaining >= 0)) | (~repaired & (remaining > 48))
    return pd.DataFrame({
        "row_id": np.flatnonzero(eligible),
        "vehicle_id": merged["vehicle_id"].to_numpy()[eligible],
        "time_step": merged["time_step"].to_numpy()[eligible],
        "is_repair": repaired[eligible],
        "class_label": np.where(repaired[eligible], tte_label(remaining[eligible]), 0),
    })


def spaced(group: pd.DataFrame, k: int | None) -> pd.DataFrame:
    ordered = group.sort_values("time_step", kind="stable")
    if k is None or len(ordered) <= k:
        return ordered.copy()
    idx = np.unique(np.rint(np.linspace(0, len(ordered) - 1, k)).astype(int))
    return ordered.iloc[idx].copy()


def select_training(pool: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    for (vehicle, cls), group in pool[pool["is_repair"]].groupby(
            ["vehicle_id", "class_label"], sort=False):
        if mode == "standard":
            k = min(3, len(group))
            chosen = group.iloc[rng.choice(len(group), size=k, replace=False)].copy()
        else:
            k = None if mode == "denseall" and int(cls) == 4 else (10 if int(cls) == 4 else 3)
            chosen = spaced(group, k)
        chosen["sw"] = 1.0 / len(chosen)
        chunks.append(chosen)
    for _, group in pool[~pool["is_repair"]].groupby("vehicle_id", sort=False):
        k = min(2, len(group))
        chosen = group.iloc[rng.choice(len(group), size=k, replace=False)].copy()
        chosen["sw"] = 1.0 / len(chosen)
        chunks.append(chosen)
    selected = pd.concat(chunks, ignore_index=True)
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)


def percentile(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(method="average", pct=True, na_option="keep").to_numpy(np.float32)


def current_delta_features(op: pd.DataFrame, selected: pd.DataFrame,
                           raw_cols: list[str], include_delta: bool) -> tuple[np.ndarray, list[str]]:
    raw = op[raw_cols].to_numpy(dtype=np.float32)
    pos = selected["row_id"].to_numpy(dtype=int)
    current = percentile(raw[pos])
    arrays = [current]
    names = [f"current_rank:{c}" for c in raw_cols]
    if include_delta:
        starts = vehicle_start(op["vehicle_id"].to_numpy())
        prior = np.maximum(pos - 1, starts[pos])
        time = op["time_step"].to_numpy(dtype=float)
        dt = time[pos] - time[prior]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (raw[pos].astype(float) - raw[prior].astype(float)) / dt[:, None]
        delta[(dt <= 0), :] = np.nan
        arrays.append(percentile(delta))
        names.extend([f"delta1_rate_rank:{c}" for c in raw_cols])
    return np.hstack(arrays).astype(np.float32), names


def build_matrix(op: pd.DataFrame, selected: pd.DataFrame, spec: pd.DataFrame,
                 spec_schema: list[str], raw_cols: list[str], lags: tuple[int, ...],
                 feature_mode: str) -> tuple[np.ndarray, list[str]]:
    hist, singles = detect_groups(raw_cols)
    temporal, temporal_names, _, _ = build_features(
        op, selected, raw_cols, hist, singles, list(lags))
    parts = [temporal.astype(np.float32)]
    names = list(temporal_names)
    if feature_mode in ("current", "current_delta"):
        extra, extra_names = current_delta_features(
            op, selected, raw_cols, include_delta=feature_mode == "current_delta")
        parts.append(extra)
        names.extend(extra_names)
    sx = spec_matrix(spec, selected["vehicle_id"].to_numpy(), spec_schema)
    parts.append(sx)
    names.extend([f"spec:{c}" for c in spec_schema])
    return np.hstack(parts).astype(np.float32), names


def class_weights(y: np.ndarray, base: np.ndarray, exponent: float) -> np.ndarray:
    neg = float(base[y == 0].sum())
    pos = float(base[y == 1].sum())
    multiplier = (neg / max(pos, 1e-12)) ** exponent
    return base * np.where(y == 1, multiplier, 1.0)


def model(kind: str, seed: int):
    if kind == "xgboost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=300, learning_rate=0.04, max_depth=5,
            min_child_weight=5, gamma=0.05, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=3.0,
            tree_method="hist", random_state=seed, n_jobs=4,
        )
    if kind == "lightgbm":
        return LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=0.035,
            num_leaves=15, min_child_samples=35, colsample_bytree=0.8,
            subsample=0.8, subsample_freq=1, reg_lambda=2.0,
            verbosity=-1, random_state=seed, n_jobs=4,
        )
    if kind == "balanced_rf":
        return BalancedRandomForestClassifier(
            n_estimators=350, max_depth=14, min_samples_leaf=3,
            max_features="sqrt", sampling_strategy="all", replacement=True,
            random_state=seed, n_jobs=4,
        )
    if kind == "easy_ensemble":
        return EasyEnsembleClassifier(n_estimators=12, random_state=seed, n_jobs=4)
    raise ValueError(kind)


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    result = {
        "c4_ap": float(average_precision_score(y, score)),
        "c4_auc": float(roc_auc_score(y, score)),
    }
    recalls = {}
    for capacity in CAPACITIES:
        k = max(1, int(np.ceil(capacity * len(y))))
        found = int(y[order[:k]].sum())
        key = int(capacity * 100)
        recalls[key] = found / max(int(y.sum()), 1)
        result[f"recall_top{key}pct"] = float(recalls[key])
        result[f"found_top{key}pct"] = found
        result[f"precision_top{key}pct"] = float(found / k)
    result["low_capacity_score"] = float(
        0.40 * recalls[1] + 0.30 * recalls[2] + 0.20 * recalls[5] + 0.10 * recalls[10])
    return result


def fit_score(x_train: np.ndarray, y_train: np.ndarray, base_weight: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray, kind: str,
              exponent: float, seed: int):
    imputer = None
    if kind in ("balanced_rf", "easy_ensemble"):
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        x_train = imputer.fit_transform(x_train).astype(np.float32)
        x_val = imputer.transform(x_val).astype(np.float32)
    weights = class_weights(y_train, base_weight, exponent)
    fitted = model(kind, seed)
    if kind == "easy_ensemble":
        fitted.fit(x_train, y_train)
    else:
        fitted.fit(x_train, y_train, sample_weight=weights)
    score = fitted.predict_proba(x_val)[:, 1]
    return metrics(y_val, score), score, fitted, imputer


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--output-dir", type=Path, default=root / "processed" / "test_33_c4_recall")
    ap.add_argument("--seed", type=int, default=SEED)
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
    pool = training_pool(tr_op, tte)
    va_sel = last_rows(va_op, va_labels)
    y_val = (va_sel["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
    schema = fit_spec_schema(tr_spec)
    results = []
    predictions = []
    stage_artifacts = {}
    best_matrix = None
    best_selection = None
    best_names = None

    print("Stage A: sampling, lag and new feature search...", flush=True)
    for candidate, sampling, lags, feature_mode in CANDIDATES:
        selected = select_training(pool, sampling, args.seed)
        y_train = (selected["class_label"].to_numpy(dtype=int) == 4).astype(np.int8)
        x_train, names = build_matrix(
            tr_op, selected, tr_spec, schema, raw_cols, lags, feature_mode)
        x_val, val_names = build_matrix(
            va_op, va_sel, va_spec, schema, raw_cols, lags, feature_mode)
        if names != val_names:
            raise ValueError("Feature schema mismatch")
        report, score, fitted, imputer = fit_score(
            x_train, y_train, selected["sw"].to_numpy(float), x_val, y_val,
            "xgboost", 0.5, args.seed)
        row = {"stage": "feature_sampling", "candidate": candidate,
               "sampling": sampling, "lags": ",".join(map(str, lags)),
               "feature_mode": feature_mode, "model": "xgboost",
               "weight_exponent": 0.5, "train_rows": len(selected),
               "c4_rows": int(y_train.sum()), "n_features": x_train.shape[1], **report}
        results.append(row)
        predictions.append(pd.DataFrame({"candidate": candidate,
                                         "vehicle_id": va_sel["vehicle_id"],
                                         "actual_c4": y_val, "c4_score": score}))
        stage_artifacts[candidate] = (fitted, imputer)
        print(f"  {candidate:26s} score={report['low_capacity_score']:.4f} R@1={report['recall_top1pct']:.3f} R@2={report['recall_top2pct']:.3f} R@5={report['recall_top5pct']:.3f} AP={report['c4_ap']:.4f}", flush=True)
        current_best = max(results, key=lambda r: (r["low_capacity_score"], r["c4_ap"]))
        if current_best["candidate"] == candidate:
            best_matrix = (x_train, x_val, y_train, selected["sw"].to_numpy(float))
            best_selection = (selected, sampling, lags, feature_mode)
            best_names = names
        else:
            del x_train, x_val
        gc.collect()

    stage_a = pd.DataFrame(results).sort_values(
        ["low_capacity_score", "c4_ap"], ascending=False)
    best_candidate = str(stage_a.iloc[0]["candidate"])
    if best_selection is None or best_candidate != max(results, key=lambda r: (r["low_capacity_score"], r["c4_ap"]))["candidate"]:
        raise RuntimeError("Best feature matrix tracking failed")
    x_train, x_val, y_train, base_weight = best_matrix

    print("Best Stage A candidate:", best_candidate)
    print("Stage B: model family and recall-weight search...", flush=True)
    for kind, exponent in [
        ("xgboost", 0.0), ("xgboost", 1.0), ("xgboost", 1.5),
        ("lightgbm", 0.5), ("lightgbm", 1.0),
        ("balanced_rf", 0.0), ("easy_ensemble", 0.0),
    ]:
        name = f"{best_candidate}|{kind}|w{exponent:g}"
        report, score, fitted, imputer = fit_score(
            x_train, y_train, base_weight, x_val, y_val, kind, exponent, args.seed)
        row = {"stage": "model_weight", "candidate": name,
               "sampling": best_selection[1], "lags": ",".join(map(str, best_selection[2])),
               "feature_mode": best_selection[3], "model": kind,
               "weight_exponent": exponent, "train_rows": len(y_train),
               "c4_rows": int(y_train.sum()), "n_features": x_train.shape[1], **report}
        results.append(row)
        predictions.append(pd.DataFrame({"candidate": name,
                                         "vehicle_id": va_sel["vehicle_id"],
                                         "actual_c4": y_val, "c4_score": score}))
        stage_artifacts[name] = (fitted, imputer)
        print(f"  {kind:13s} w={exponent:<3g} score={report['low_capacity_score']:.4f} R@1={report['recall_top1pct']:.3f} R@2={report['recall_top2pct']:.3f} R@5={report['recall_top5pct']:.3f} AP={report['c4_ap']:.4f}", flush=True)

    result = pd.DataFrame(results).sort_values(
        ["low_capacity_score", "c4_ap"], ascending=False).reset_index(drop=True)
    winner = result.iloc[0].to_dict()
    winner_model, winner_imputer = stage_artifacts[str(winner["candidate"])]
    joblib.dump(winner_model, args.output_dir / "model.joblib")
    if winner_imputer is not None:
        joblib.dump(winner_imputer, args.output_dir / "imputer.joblib")
    result.to_csv(args.output_dir / "validation_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "validation_predictions.csv", index=False)
    metadata = {
        "objective": "weighted C4 recall at top 1/2/5/10 percent fleet capacity",
        "test_used_for_selection": False,
        "winner": winner,
        "raw_cols": raw_cols,
        "spec_columns": schema,
        "feature_names": best_names,
        "capacities": CAPACITIES,
        "selection_rule": "0.40*R@1% + 0.30*R@2% + 0.20*R@5% + 0.10*R@10%; AP tie-break",
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== WINNER ===")
    print(json.dumps(winner, indent=2, ensure_ascii=False, default=float))
    print("Saved to", args.output_dir)


if __name__ == "__main__":
    main()
