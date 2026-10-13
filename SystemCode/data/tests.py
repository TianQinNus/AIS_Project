import pandas as pd

df = pd.read_csv(
    "processed/test_30_stage_aware_decisions.csv"
)

changed = df[
    df["stage_attention_action"]
    != df["safety_floor_attention_action"]
]

print("\n=== UPGRADED VEHICLES BY CLASS ===")
print(changed["class_label"].value_counts().sort_index())

print("\n=== C3/C4 UPGRADED VEHICLES ===")
print(
    changed[changed["class_label"].isin([3, 4])][
        [
            "vehicle_id",
            "class_label",
            "risk_stratum",
            "near_share",
            "imminent_share",
            "safety_floor_attention_action",
            "stage_attention_action",
        ]
    ].to_string(index=False)
)