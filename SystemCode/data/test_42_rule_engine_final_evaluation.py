"""Final evaluation of the Validation-selected Test 40 temporal rule engine."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from test_40_rule_engine_search import apply_rules, evaluate


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "processed" / "test_40_rule_engine_search"
    out = root / "processed" / "test_42_rule_engine_final"
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / "selection_and_test.json").read_text(encoding="utf-8"))
    profile = selection["profile"]
    history = pd.read_csv(source / "test_risk_history.csv")
    decisions = apply_rules(history, profile)
    report = evaluate(decisions)
    report["attention_ml_same_n_precision"] = (
        report["attention_ml_same_n_c4"] / report["attention_total"])
    report["attention_ml_same_n_recall"] = (
        report["attention_ml_same_n_c4"] / report["c4_total"])
    report["attention_precision_relative_gain"] = (
        report["attention_precision"] / report["attention_ml_same_n_precision"] - 1)
    by_rule = decisions.groupby(["rule_id", "action"], dropna=False).agg(
        vehicles=("vehicle_id", "size"),
        c4=("class_label", lambda x: int((x == 4).sum()))).reset_index()
    by_rule["c4_rate"] = by_rule.c4 / by_rule.vehicles
    decisions.to_csv(out / "test_decisions.csv", index=False)
    by_rule.to_csv(out / "test_by_rule.csv", index=False)
    final = {
        "rule_profile": profile,
        "test_metrics": report,
        "conclusion": {
            "immediate_tier": "No improvement: temporal rules found fewer C4 than Test39 at equal immediate-inspection workload.",
            "attention_tier": "Improvement: temporal rules recovered additional C4 at equal total attention workload.",
            "recommended_hybrid": "Use Test39 ranking for immediate inspection and temporal rules for monitor/review attention coverage.",
        },
        "selection_status": "Rule profile selected on Validation; this file reports its frozen Test result.",
    }
    (out / "final_evaluation.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False, default=float))
    print("\nBY RULE")
    print(by_rule.to_string(index=False))


if __name__ == "__main__":
    main()
