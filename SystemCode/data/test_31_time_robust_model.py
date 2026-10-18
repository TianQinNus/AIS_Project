"""Time-robust FleetCare model search and one-shot official test evaluation.

The script compares feature variants on the official validation vehicles, freezes
the winner, refits it on train + validation, and evaluates it once on test.
Absolute time_step is never used as a predictor. Rank variants convert selected
dynamic blocks to within-assessment fleet percentiles to reduce calendar/time and
measurement-scale shift.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from test21 import build_features, choose_rows_multi, detect_groups, load


LABELS = np.arange(5)
HORIZONS = {48: 1, 24: 2, 12: 3, 6: 4}

PARAMETER_SETS = {
    "P1_regularized": dict(
        n_estimators=300, learning_rate=0.04, max_depth=4,
        min_child_weight=8, gamma=0.10, subsample=0.80,
        colsample_bytree=0.65, reg_alpha=0.20, reg_lambda=5.0,
    ),
    "P3_medium": dict(
        n_estimators=250, learning_rate=0.05, max_depth=5,
        min_child_weight=5, gamma=0.05, subsample=0.80,
        colsample_bytree=0.75, reg_alpha=0.10, reg_lambda=3.0,
    ),
}

FEATURE_VARIANTS = {
    "V1_raw_all": {"drop": (), "rank": ()},
    "V2_no_lifetime_rate": {"drop": ("rate",), "rank": ()},
    "V3_rank_lifetime_rate": {"drop": (), "rank": ("rate",)},
    "V4_rank_rate_and_ratio": {"drop": (), "rank": ("rate", "ratio")},
    "V5_rank_all_dynamic": {"drop": (), "rank": ("rate", "ratio", "prop", "mix")},
}


def make_model(params: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        tree_method="hist", n_jobs=4, random_state=seed, **params,
    )


def last_rows(op: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = op[["vehicle_id", "time_step"]].copy()
    rows["row_id"] = np.arange(len(rows))
    rows = rows.groupby("vehicle_id", sort=False).tail(1)
    rows = rows.merge(labels[["vehicle_id", "class_label"]], on="vehicle_id",
                      how="left", validate="one_to_one")
    if rows["class_label"].isna().any():
        raise ValueError("Some last-readout vehicles have no class label")
    return rows.reset_index(drop=True)


def fit_spec_schema(spec: pd.DataFrame) -> list[str]:
    feat = spec.drop(columns="vehicle_id")
    cat = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    return list(pd.get_dummies(feat, columns=cat, dtype=np.float32).columns)


def spec_matrix(spec: pd.DataFrame, ids: np.ndarray, schema: list[str]) -> np.ndarray:
    if spec["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate vehicle_id in specifications")
    feat = spec.set_index("vehicle_id")
    cat = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat, dtype=np.float32)
    feat = feat.reindex(columns=schema, fill_value=0.0)
    aligned = feat.reindex(ids)
    if aligned.isna().all(axis=1).any():
        raise ValueError("Some vehicles have no specification row")
    return aligned.to_numpy(dtype=np.float32)


def percentile_columns(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy()
    if not mask.any():
        return out
    ranked = pd.DataFrame(out[:, mask]).rank(axis=0, method="average", pct=True,
                                             na_option="keep")
    out[:, mask] = ranked.to_numpy(dtype=np.float32)
    return out


def transform_variant(dynamic: np.ndarray, blocks: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = FEATURE_VARIANTS[variant]
    keep = ~np.isin(blocks, cfg["drop"])
    x = dynamic[:, keep].astype(np.float32, copy=True)
    kept_blocks = blocks[keep]
    rank_mask = np.isin(kept_blocks, cfg["rank"])
    x = percentile_columns(x, rank_mask)
    return x, keep


def probabilities_to_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(axis=1)
    result = {
        "n": int(len(y)),
        "class_counts": {f"C{i}": int((y == i).sum()) for i in LABELS},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
        "horizons": {},
    }
    aps = []
    for horizon, start in HORIZONS.items():
        truth = (y >= start).astype(int)
        score = p[:, start:].sum(axis=1)
        k = max(1, int(np.ceil(0.05 * len(y))))
        top = np.argsort(-score, kind="stable")[:k]
        positives = int(truth.sum())
        ap = float(average_precision_score(truth, score))
        aps.append(ap)
        result["horizons"][f"Risk{horizon}"] = {
            "positives": positives,
            "prevalence": float(truth.mean()),
            "average_precision": ap,
            "roc_auc": float(roc_auc_score(truth, score)),
            "recall_at_top5pct": float(truth[top].sum() / positives) if positives else None,
            "precision_at_top5pct": float(truth[top].mean()),
        }
    result["mean_horizon_ap"] = float(np.mean(aps))
    return result


def prepare_dynamic(op: pd.DataFrame, selected: pd.DataFrame, raw_cols: list[str],
                    lags: list[int]) -> tuple[np.ndarray, list[str], np.ndarray]:
    hist, singles = detect_groups(raw_cols)
    x, names, blocks, _ = build_features(op, selected, raw_cols, hist, singles, lags)
    return x.astype(np.float32), list(names), np.asarray(blocks)


def save_predictions(path: Path, selected: pd.DataFrame, y: np.ndarray,
                     p: np.ndarray) -> None:
    frame = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"].to_numpy(),
        "time_step": selected["time_step"].to_numpy(),
        "actual_class": y,
        "predicted_class": p.argmax(axis=1),
        **{f"p_c{i}": p[:, i] for i in LABELS},
        **{f"Risk{h}": p[:, start:].sum(axis=1) for h, start in HORIZONS.items()},
    })
    frame.to_csv(path, index=False)


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "raw")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "processed" / "test_31_time_robust")
    ap.add_argument("--lags", default="5,20")
    ap.add_argument("--k-repair", type=int, default=3)
    ap.add_argument("--k-nonrepair", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lags = [int(v) for v in args.lags.split(",") if v]

    print("Loading train / validation files...", flush=True)
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
    y_tr = tr_sel["class_label"].to_numpy(dtype=int)
    y_va = va_sel["class_label"].to_numpy(dtype=int)
    w_tr = tr_sel["sw"].to_numpy(dtype=float)
    print("Rows:", {"train": len(y_tr), "validation": len(y_va)})
    print("Class counts:", {
        "train": np.bincount(y_tr, minlength=5).tolist(),
        "validation": np.bincount(y_va, minlength=5).tolist(),
    }, flush=True)

    print("Building time-normalised dynamic features...", flush=True)
    d_tr, dynamic_names, blocks = prepare_dynamic(tr_op, tr_sel, raw_cols, lags)
    d_va, va_names, va_blocks = prepare_dynamic(va_op, va_sel, raw_cols, lags)
    if dynamic_names != va_names or not np.array_equal(blocks, va_blocks):
        raise ValueError("Dynamic feature schema differs between train and validation")
    del tr_op, va_op
    gc.collect()

    spec_schema = fit_spec_schema(tr_spec)
    s_tr = spec_matrix(tr_spec, tr_sel["vehicle_id"].to_numpy(), spec_schema)
    s_va = spec_matrix(va_spec, va_sel["vehicle_id"].to_numpy(), spec_schema)

    print("\nValidation screening (test labels are not used here):", flush=True)
    screening = []
    cached = {}
    for variant in FEATURE_VARIANTS:
        vt, keep = transform_variant(d_tr, blocks, variant)
        vv, _ = transform_variant(d_va, blocks, variant)
        xt = np.hstack([vt, s_tr]).astype(np.float32)
        xv = np.hstack([vv, s_va]).astype(np.float32)
        cached[variant] = (xt, xv, keep)
        for param_name, params in PARAMETER_SETS.items():
            model = make_model(params, args.seed)
            model.fit(xt, y_tr, sample_weight=w_tr)
            p = model.predict_proba(xv)
            metrics = probabilities_to_metrics(y_va, p)
            row = {
                "feature_variant": variant,
                "parameter_set": param_name,
                "n_features": int(xt.shape[1]),
                "mean_horizon_ap": metrics["mean_horizon_ap"],
                "macro_f1": metrics["macro_f1"],
                **{f"risk{h}_ap": metrics["horizons"][f"Risk{h}"]["average_precision"] for h in HORIZONS},
            }
            screening.append(row)
            print(f"  {variant:25s} {param_name:15s} meanAP={row['mean_horizon_ap']:.5f} macroF1={row['macro_f1']:.5f}", flush=True)
            del model, p
            gc.collect()

    screen_df = pd.DataFrame(screening).sort_values(
        ["mean_horizon_ap", "macro_f1"], ascending=False).reset_index(drop=True)
    screen_df.to_csv(args.output_dir / "validation_screening.csv", index=False)
    winner = screen_df.iloc[0].to_dict()
    variant = str(winner["feature_variant"])
    param_name = str(winner["parameter_set"])
    print("\nFrozen winner:", variant, "+", param_name, flush=True)

    # Test data and labels are deliberately opened only after automatic selection.
    print("Loading untouched official test after freezing the design...", flush=True)
    te_op = load(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    te_spec = load(args.data_dir, "test_specifications.csv")
    te_labels = load(args.data_dir, "test_labels.csv")
    te_sel = last_rows(te_op, te_labels)
    y_te = te_sel["class_label"].to_numpy(dtype=int)
    print("Test rows:", len(y_te), "class counts:",
          np.bincount(y_te, minlength=5).tolist(), flush=True)
    d_te, te_names, te_blocks = prepare_dynamic(te_op, te_sel, raw_cols, lags)
    if dynamic_names != te_names or not np.array_equal(blocks, te_blocks):
        raise ValueError("Dynamic feature schema differs on test")
    s_te = spec_matrix(te_spec, te_sel["vehicle_id"].to_numpy(), spec_schema)
    del te_op
    gc.collect()

    # Refit the frozen design on train + validation. Each source is transformed
    # independently for rank variants; no test labels enter feature construction.
    x_tr, x_va, keep = cached[variant]
    d_te_v, _ = transform_variant(d_te, blocks, variant)
    x_te = np.hstack([d_te_v, s_te]).astype(np.float32)
    x_final = np.vstack([x_tr, x_va])
    y_final = np.r_[y_tr, y_va]
    # Validation vehicles enter once, matching one vehicle-level assessment.
    w_final = np.r_[w_tr, np.ones(len(y_va), dtype=float)]
    final_model = make_model(PARAMETER_SETS[param_name], args.seed)
    print("Final fit on train + validation; evaluating official test once...", flush=True)
    final_model.fit(x_final, y_final, sample_weight=w_final)
    p_test = final_model.predict_proba(x_te)
    test_metrics = probabilities_to_metrics(y_te, p_test)

    kept_dynamic_names = [n for n, k in zip(dynamic_names, keep) if k]
    feature_names = kept_dynamic_names + [f"spec:{n}" for n in spec_schema]
    metadata = {
        "model_family": "XGBoost",
        "protocol": "select_on_official_validation_then_one_shot_official_test",
        "absolute_time_step_used_as_predictor": False,
        "selected_feature_variant": variant,
        "selected_parameter_set": param_name,
        "feature_variant_definition": FEATURE_VARIANTS[variant],
        "model_params": final_model.get_params(),
        "raw_cols": list(map(str, raw_cols)),
        "lags": lags,
        "dynamic_feature_names": kept_dynamic_names,
        "dynamic_blocks": blocks[keep].tolist(),
        "spec_columns": spec_schema,
        "feature_names": feature_names,
        "n_features": int(x_final.shape[1]),
        "train_rows": int(len(y_tr)),
        "validation_rows": int(len(y_va)),
        "test_rows": int(len(y_te)),
    }
    joblib.dump(final_model, args.output_dir / "model.joblib")
    (args.output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "selection.json").write_text(
        json.dumps({"winner": winner, "selection_rule": "highest official-validation mean AP across Risk48/24/12/6; MacroF1 tie-break only"},
                   indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    save_predictions(args.output_dir / "test_predictions.csv", te_sel, y_te, p_test)
    pd.DataFrame(test_metrics["confusion_matrix"],
                 index=[f"true_C{i}" for i in LABELS],
                 columns=[f"pred_C{i}" for i in LABELS]).to_csv(
                     args.output_dir / "test_confusion_matrix.csv")

    print("\n=== OFFICIAL TEST (frozen model) ===")
    print("Accuracy:", round(test_metrics["accuracy"], 6))
    print("Macro F1:", round(test_metrics["macro_f1"], 6))
    print("Mean horizon AP:", round(test_metrics["mean_horizon_ap"], 6))
    for h in HORIZONS:
        m = test_metrics["horizons"][f"Risk{h}"]
        print(f"Risk{h}: AP={m['average_precision']:.6f} AUC={m['roc_auc']:.6f} Recall@5%={m['recall_at_top5pct']:.4f}")
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
