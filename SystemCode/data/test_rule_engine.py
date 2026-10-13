import pandas as pd
import numpy as np

h = pd.read_csv("processed/prediction_history.csv")
s = pd.read_csv("processed/prediction_temporal_summary.csv")

# 1. 基础完整性
assert h["vehicle_id"].nunique() == len(s)
assert not h.duplicated(["vehicle_id", "time_step"]).any()
assert not s["vehicle_id"].duplicated().any()

# 2. Horizon 概率一致性
assert np.allclose(
    h["Risk48"],
    h[["p_c1", "p_c2", "p_c3", "p_c4"]].sum(axis=1),
    atol=1e-6
)

assert np.allclose(
    h["Risk24"],
    h[["p_c2", "p_c3", "p_c4"]].sum(axis=1),
    atol=1e-6
)

assert np.allclose(
    h["Risk12"],
    h[["p_c3", "p_c4"]].sum(axis=1),
    atol=1e-6
)

assert np.allclose(
    h["Risk6"],
    h["p_c4"],
    atol=1e-6
)

# 3. Risk Horizon 必须满足单调关系
assert (
    (h["Risk48"] >= h["Risk24"] - 1e-6)
    & (h["Risk24"] >= h["Risk12"] - 1e-6)
    & (h["Risk12"] >= h["Risk6"] - 1e-6)
).all()

# 4. Summary 的当前风险必须等于最后一次读数
latest = h.sort_values(
    ["vehicle_id", "time_step"]
).groupby("vehicle_id").tail(1)

merged = latest.merge(
    s, on="vehicle_id", validate="one_to_one"
)

for horizon in (48, 24, 12, 6):
    assert np.allclose(
        merged[f"Risk{horizon}"],
        merged[f"Risk{horizon}_current"],
        atol=1e-6
    )

print("All historical replay consistency tests passed!")