"""Test 61: model-first rule engine on frozen Test54 risk histories."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from test_40_rule_engine_search import evaluate, temporal_summary
from test_60_test54_rule_engine_validation import FROZEN_PROFILE, report_by_rule


def decide_model_first(row: pd.Series, profile: dict) -> tuple[str, str]:
    n = int(row.history_count)
    if n < 2:
        return "TRD", "COLLECT_MORE_DATA"
    # Test54 ranking is the safety floor: temporal conditions may escalate
    # additional vehicles but may not demote the top five percent.
    if row.Risk6_percentile >= 99.0:
        return "TR1", "PRIORITY_INSPECT"
    if row.Risk6_percentile >= 95.0:
        return "TR2A", "INSPECT"
    if n < 3 and row.Risk12_percentile >= profile["review_near"]:
        return "TR0", "MANUAL_REVIEW"
    if bool(row.is_volatile) and bool(row.is_uncertain):
        return "TRV", "MANUAL_REVIEW"
    if (row.Risk12_percentile >= profile["p12"] and
            (row.persist12 or row.rising12 or row.fast6)):
        return "TR2", "INSPECT"
    if (row.Risk24_percentile >= profile["p24"] and
            (row.persist24 or row.rising24)):
        return "TR4", "ENHANCED_MONITOR"
    if (row.Risk48_percentile >= profile["p48"] and
            (row.persist48 or row.rising48)):
        return "TR5", "MONITOR"
    return "TR6", "CONTINUE"


def apply_model_first(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    summary = temporal_summary(history, profile)
    decisions = summary.apply(lambda row: decide_model_first(row, profile), axis=1)
    summary["rule_id"] = [value[0] for value in decisions]
    summary["action"] = [value[1] for value in decisions]
    return summary


def main():
    root = Path(__file__).resolve().parent
    source = root / "processed" / "test_60_test54_rule_engine"
    out = root / "processed" / "test_61_model_first_rule_engine"
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in ("validation", "test"):
        history = pd.read_csv(source / f"{split}_risk_history.csv")
        decisions = apply_model_first(history, FROZEN_PROFILE)
        metrics = evaluate(decisions)
        by_rule = report_by_rule(decisions)
        decisions.to_csv(out / f"{split}_decisions.csv", index=False)
        by_rule.to_csv(out / f"{split}_by_rule.csv", index=False)
        result[split] = metrics
        result[f"{split}_by_rule"] = by_rule.to_dict(orient="records")
    result["design"] = (
        "Test54 P99 is PRIORITY_INSPECT and P95-P99 is INSPECT without a "
        "persistence gate. Temporal Risk12 can add INSPECT vehicles; later "
        "rules route monitoring, uncertainty, and insufficient history."
    )
    result["comparison_note"] = (
        "At equal queue size, ml_same_n_c4 is the number found by direct "
        "Test54 ranking."
    )
    (out / "results.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
