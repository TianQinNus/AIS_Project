"""Post-hoc Test search across the predeclared Test 40 rule profiles."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from test_40_rule_engine_search import apply_rules, evaluate, profiles


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "processed" / "test_40_rule_engine_search"
    out = root / "processed" / "test_41_rule_engine_posthoc_search"
    out.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(source / "test_risk_history.csv")
    rows = []
    decisions = {}
    all_profiles = profiles()
    for profile in all_profiles:
        result = apply_rules(history, profile)
        rows.append({"profile": profile["name"], **evaluate(result)})
        decisions[profile["name"]] = result
    report = pd.DataFrame(rows)
    eligible = report[report.immediate_capacity.between(0.01, 0.05, inclusive="both")].copy()
    eligible = eligible.sort_values(
        ["rule_c4_gain_at_same_n", "immediate_c4", "immediate_precision"],
        ascending=False).reset_index(drop=True)
    winner = str(eligible.iloc[0].profile)
    profile = next(p for p in all_profiles if p["name"] == winner)
    chosen = decisions[winner]
    report.to_csv(out / "all_test_rule_profiles.csv", index=False)
    chosen.to_csv(out / "best_posthoc_decisions.csv", index=False)
    by_rule = chosen.groupby(["rule_id", "action"], dropna=False).agg(
        vehicles=("vehicle_id", "size"),
        c4=("class_label", lambda x: int((x == 4).sum()))).reset_index()
    by_rule["c4_rate"] = by_rule.c4 / by_rule.vehicles
    by_rule.to_csv(out / "best_posthoc_by_rule.csv", index=False)
    summary = {
        "winner": winner,
        "profile": profile,
        "metrics": eligible.iloc[0].to_dict(),
        "warning": "Rule thresholds selected after comparing official Test labels; post-hoc only.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("TOP POST-HOC TEST RULES")
    print(eligible.head(15).to_string(index=False))
    print("\nBEST BY RULE")
    print(by_rule.to_string(index=False))
    print("\n", json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
