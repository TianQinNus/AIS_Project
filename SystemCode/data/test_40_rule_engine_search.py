"""Search temporal rule engines on top of the frozen Test 39 C4 model."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from test21 import load
from test_31_time_robust_model import fit_spec_schema
from test_33_c4_recall_search import build_matrix, select_training, training_pool


HORIZONS = (48, 24, 12, 6)
IMMEDIATE = {"PRIORITY_INSPECT", "INSPECT"}
ATTENTION = {"PRIORITY_INSPECT", "INSPECT", "ENHANCED_MONITOR", "MONITOR",
             "MANUAL_REVIEW", "COLLECT_MORE_DATA"}


def grouped_rank(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": group})
    return frame.groupby("group", sort=False)["value"].rank(pct=True).to_numpy(float)


def make_history_rows(op: pd.DataFrame, maximum: int = 5) -> pd.DataFrame:
    rows = op.groupby("vehicle_id", sort=False).tail(maximum)[
        ["vehicle_id", "time_step"]].copy()
    rows["row_id"] = rows.index.to_numpy(int)
    rows = rows.sort_values(["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    rows["history_count"] = rows.groupby("vehicle_id")["vehicle_id"].transform("size")
    rows["chronological_index"] = rows.groupby("vehicle_id").cumcount() + 1
    rows["step_from_latest"] = rows["history_count"] - rows["chronological_index"] + 1
    return rows


def multiclass_weights(labels: np.ndarray, base: np.ndarray, exponent: float = 0.25) -> np.ndarray:
    totals = np.array([base[labels == c].sum() for c in range(5)], dtype=float)
    multiplier = np.power(totals.max() / np.maximum(totals, 1e-12), exponent)
    return base * multiplier[labels]


def temporal_summary(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    rows = []
    for vehicle, group in history.groupby("vehicle_id", sort=False):
        g = group.sort_values("time_step", kind="stable")
        row = {
            "vehicle_id": vehicle,
            "history_count": len(g),
            "class_label": int(g["class_label"].iloc[-1]),
            "risk6_score": float(g["Risk6"].iloc[-1]),
            "base_rank": float(g["base_rank"].iloc[-1]),
            "veto_rank": float(g["veto_rank"].iloc[-1]),
        }
        rising_count = 0
        for horizon in HORIZONS:
            values = g[f"Risk{horizon}"].to_numpy(float)
            threshold = profile[f"p{horizon}"] / 100.0
            persistent = len(values) >= 3 and float(np.mean(values >= threshold)) >= profile["persistence"]
            if len(values) >= 3:
                x = np.arange(len(values), dtype=float)
                slope = float(np.polyfit(x, values, 1)[0])
                rising = slope > 0 and values[-1] > values[0]
                change = float(values[-1] - values[0])
            else:
                rising = False
                change = np.nan
            row[f"Risk{horizon}_percentile"] = 100.0 * values[-1]
            row[f"persist{horizon}"] = persistent
            row[f"rising{horizon}"] = rising
            row[f"change{horizon}"] = change
            rising_count += int(rising)
        row["rising_horizons"] = rising_count
        risk6 = g["Risk6"].to_numpy(float)
        row["volatility"] = float(np.std(risk6)) if len(risk6) >= 3 else np.nan
        row["model_disagreement"] = abs(row["base_rank"] - row["veto_rank"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    for horizon in HORIZONS:
        positive = summary[f"change{horizon}"].clip(lower=0)
        cutoff = float(positive.quantile(profile["fast_percentile"] / 100.0))
        summary[f"fast{horizon}"] = summary[f"rising{horizon}"] & (positive > max(cutoff, 0.0))
    vol_cutoff = float(summary["volatility"].quantile(profile["volatility_percentile"] / 100.0))
    disagreement_cutoff = float(summary["model_disagreement"].quantile(
        profile["uncertainty_percentile"] / 100.0))
    summary["is_volatile"] = (summary.history_count >= 3) & (summary.volatility > vol_cutoff)
    summary["is_uncertain"] = summary.model_disagreement > disagreement_cutoff
    return summary


def decide(row: pd.Series, profile: dict) -> tuple[str, str]:
    n = int(row.history_count)
    high = lambda horizon: float(row[f"Risk{horizon}_percentile"]) >= profile[f"p{horizon}"]
    if n < 2:
        return "TRD", "COLLECT_MORE_DATA"
    if n < 3 and (row.Risk6_percentile >= profile["review_near"] or
                  row.Risk12_percentile >= profile["review_near"]):
        return "TR0", "MANUAL_REVIEW"
    if bool(row.is_volatile) and bool(row.is_uncertain):
        return "TRV", "MANUAL_REVIEW"
    if high(6) and (row.persist6 or row.fast6 or row.fast12):
        return "TR1", "PRIORITY_INSPECT"
    if high(12) and (row.persist12 or row.rising12 or row.fast6):
        return "TR2", "INSPECT"
    if row.Risk6_percentile >= profile["inspect6"] and row.rising_horizons >= profile["min_rising"]:
        return "TR3", "INSPECT"
    if high(24) and (row.persist24 or row.rising24):
        return "TR4", "ENHANCED_MONITOR"
    if high(48) and (row.persist48 or row.rising48):
        return "TR5", "MONITOR"
    return "TR6", "CONTINUE"


def apply_rules(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    summary = temporal_summary(history, profile)
    decisions = summary.apply(lambda row: decide(row, profile), axis=1)
    summary["rule_id"] = [x[0] for x in decisions]
    summary["action"] = [x[1] for x in decisions]
    return summary


def top_n_found(data: pd.DataFrame, n: int) -> int:
    if n <= 0:
        return 0
    order = np.lexsort((data.vehicle_id.to_numpy(), -data.risk6_score.to_numpy()))
    return int((data.class_label.to_numpy()[order[:n]] == 4).sum())


def evaluate(data: pd.DataFrame) -> dict:
    truth = data.class_label.to_numpy() == 4
    immediate = data.action.isin(IMMEDIATE).to_numpy()
    priority = (data.action == "PRIORITY_INSPECT").to_numpy()
    attention = data.action.isin(ATTENTION).to_numpy()
    result = {"vehicles": len(data), "c4_total": int(truth.sum())}
    for label, mask in (("priority", priority), ("immediate", immediate), ("attention", attention)):
        total = int(mask.sum())
        found = int((truth & mask).sum())
        ml_found = top_n_found(data, total)
        result[f"{label}_total"] = total
        result[f"{label}_c4"] = found
        result[f"{label}_precision"] = found / total if total else 0.0
        result[f"{label}_recall"] = found / truth.sum() if truth.any() else 0.0
        result[f"{label}_ml_same_n_c4"] = ml_found
        result[f"{label}_gain_same_n"] = found - ml_found
    # Backward-compatible names used by the search and existing result files.
    result["ml_same_n_c4"] = result["immediate_ml_same_n_c4"]
    result["rule_c4_gain_at_same_n"] = result["immediate_gain_same_n"]
    result["immediate_capacity"] = result["immediate_total"] / len(data)
    return result


def profiles() -> list[dict]:
    threshold_sets = [
        (99.5, 98.5, 98.0, 95.0, 85.0, "strict"),
        (99.0, 97.0, 97.0, 90.0, 80.0, "proposed"),
        (98.5, 96.0, 96.0, 88.0, 75.0, "balanced"),
        (98.0, 95.0, 95.0, 85.0, 70.0, "sensitive"),
    ]
    result = []
    for values, persistence, fast, rising in itertools.product(
            threshold_sets, (0.4, 0.6, 0.8), (85.0, 90.0, 95.0), (2, 3)):
        p6, p12, inspect6, p24, p48, label = values
        result.append({
            "name": f"{label}_pers{persistence:g}_fast{fast:g}_rise{rising}",
            "p6": p6, "p12": p12, "inspect6": inspect6,
            "p24": p24, "p48": p48, "persistence": persistence,
            "fast_percentile": fast, "min_rising": rising,
            "review_near": 97.0, "volatility_percentile": 95.0,
            "uncertainty_percentile": 90.0,
        })
    return result


def score_history(root: Path, split: str, op: pd.DataFrame, spec: pd.DataFrame,
                  labels: pd.DataFrame, schema: list[str], raw_cols: list[str],
                  feature_names: list[str], old_models: dict, veto_model: object,
                  multiclass: XGBClassifier) -> pd.DataFrame:
    selected = make_history_rows(op)
    x, names = build_matrix(op, selected, spec, schema, raw_cols, (1, 3, 5), "temporal")
    if names != feature_names:
        raise ValueError(f"{split} feature schema mismatch")
    xgb_score = old_models["xgb_42"].predict_proba(x)[:, 1]
    lgb_score = old_models["lgb_42"].predict_proba(x)[:, 1]
    veto_score = veto_model.predict_proba(x)[:, 1]
    step = selected.step_from_latest.to_numpy()
    base_pair = (grouped_rank(xgb_score, step) + grouped_rank(lgb_score, step)) / 2
    # Test 39 ranks the pair ensemble once more before applying the veto.
    # Preserve that exact ordering here so any gain comes from temporal rules,
    # not from a subtly different implementation of the frozen model score.
    base_rank = grouped_rank(base_pair, step)
    veto_rank = grouped_rank(veto_score, step)
    risk6 = np.minimum(base_rank, veto_rank)
    probability = multiclass.predict_proba(x)
    raw = {
        12: probability[:, 3] + probability[:, 4],
        24: probability[:, 2] + probability[:, 3] + probability[:, 4],
        48: 1.0 - probability[:, 0],
    }
    risk12 = np.maximum(grouped_rank(raw[12], step), risk6)
    risk24 = np.maximum(grouped_rank(raw[24], step), risk12)
    risk48 = np.maximum(grouped_rank(raw[48], step), risk24)
    latest_class = labels.set_index("vehicle_id")["class_label"]
    selected["class_label"] = selected.vehicle_id.map(latest_class).astype(int)
    selected["base_rank"] = base_rank
    selected["veto_rank"] = veto_rank
    selected["Risk6"] = risk6
    selected["Risk12"] = risk12
    selected["Risk24"] = risk24
    selected["Risk48"] = risk48
    return selected


def main() -> None:
    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out = root / "processed" / "test_40_rule_engine_search"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and training auxiliary ordered-horizon model...", flush=True)
    tr_op = load(raw, "train_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    tr_spec = load(raw, "train_specifications.csv")
    tte = load(raw, "train_tte.csv")
    raw_cols = [c for c in tr_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(tr_spec)
    train_selected = select_training(training_pool(tr_op, tte), "dense10", 42)
    train_class = train_selected.class_label.to_numpy(int)
    x_train, feature_names = build_matrix(
        tr_op, train_selected, tr_spec, schema, raw_cols, (1, 3, 5), "temporal")
    multiclass = XGBClassifier(
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        n_estimators=350, learning_rate=0.035, max_depth=5, min_child_weight=8,
        gamma=0.1, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=5.0, tree_method="hist", random_state=42, n_jobs=4)
    multiclass.fit(x_train, train_class, sample_weight=multiclass_weights(
        train_class, train_selected.sw.to_numpy(float)))
    joblib.dump(multiclass, out / "auxiliary_multiclass.joblib")

    old_dir = root / "processed" / "test_34_c4_precision"
    old_models = {key: joblib.load(old_dir / f"{key}.joblib") for key in ("xgb_42", "lgb_42")}
    test38 = joblib.load(root / "processed" / "test_38_c4_cascade_easyensemble" / "models.joblib")
    veto_model = test38["veto_c0c4"]

    histories = {}
    for split in ("validation", "test"):
        print("Scoring five-observation history:", split, flush=True)
        op = load(raw, f"{split}_operational_readouts.csv").sort_values(
            ["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
        spec = load(raw, f"{split}_specifications.csv")
        labels = load(raw, f"{split}_labels.csv")
        histories[split] = score_history(
            root, split, op, spec, labels, schema, raw_cols, feature_names,
            old_models, veto_model, multiclass)
        histories[split].to_csv(out / f"{split}_risk_history.csv", index=False)

    print("Searching rule variants on Validation...", flush=True)
    validation_rows = []
    validation_decisions = {}
    all_profiles = profiles()
    for profile in all_profiles:
        decisions = apply_rules(histories["validation"], profile)
        report = evaluate(decisions)
        validation_rows.append({"profile": profile["name"], **report})
        validation_decisions[profile["name"]] = decisions
    validation = pd.DataFrame(validation_rows)
    eligible = validation[
        validation.immediate_capacity.between(0.01, 0.05, inclusive="both")].copy()
    if eligible.empty:
        raise RuntimeError("No rule profile produced 1%-5% immediate workload")
    eligible = eligible.sort_values(
        ["rule_c4_gain_at_same_n", "immediate_c4", "immediate_precision"],
        ascending=False)
    winner_name = str(eligible.iloc[0].profile)
    winner_profile = next(p for p in all_profiles if p["name"] == winner_name)
    validation.to_csv(out / "validation_rule_search.csv", index=False)
    validation_decisions[winner_name].to_csv(out / "validation_winner_decisions.csv", index=False)

    print("Frozen winner:", winner_name, "Applying to Test...", flush=True)
    test_decisions = apply_rules(histories["test"], winner_profile)
    test_report = evaluate(test_decisions)
    test_decisions.to_csv(out / "test_decisions.csv", index=False)
    by_rule = test_decisions.groupby(["rule_id", "action"], dropna=False).agg(
        vehicles=("vehicle_id", "size"),
        c4=("class_label", lambda x: int((x == 4).sum()))).reset_index()
    by_rule["c4_rate"] = by_rule.c4 / by_rule.vehicles
    by_rule.to_csv(out / "test_by_rule.csv", index=False)
    selection = {
        "winner": winner_name, "profile": winner_profile,
        "validation_metrics": validation[validation.profile == winner_name].iloc[0].to_dict(),
        "test_metrics": test_report,
        "selection_rule": "Validation only: maximize C4 gain over Test39 Top-N at identical 1%-5% immediate workload; then C4 found and precision",
        "risk_design": "Test39 agreement-veto Risk6 plus auxiliary ordered multiclass Risk12/24/48; last five observations",
    }
    (out / "selection_and_test.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nVALIDATION TOP RULES")
    print(eligible.head(12).to_string(index=False))
    print("\nTEST RESULT")
    print(json.dumps(test_report, indent=2))
    print("\nTEST BY RULE")
    print(by_rule.to_string(index=False))


if __name__ == "__main__":
    main()
