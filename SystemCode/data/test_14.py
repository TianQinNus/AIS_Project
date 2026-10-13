
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
HORIZON = 6
CAPACITIES = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_EXCLUSIONS = {
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
}

def load_csv(filename):
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_op = load_csv("train_operational_readouts.csv")
train_spec = load_csv("train_specifications.csv")
train_tte = load_csv("train_tte.csv")

val_op = load_csv("validation_operational_readouts.csv")
val_spec = load_csv("validation_specifications.csv")
val_labels = load_csv("validation_labels.csv")

raw_features = [
    c for c in train_op.columns
    if c not in ("vehicle_id", "time_step")
]

print("Training readouts:", train_op.shape)
print("Validation readouts:", val_op.shape)


# ============================================================
# 3. HISTORICAL FEATURE ENGINEERING
# Same feature families as Test 13
# ============================================================

def create_history_features(df, feature_cols):
    df = (
        df.sort_values(["vehicle_id", "time_step"])
          .reset_index(drop=True)
          .copy()
    )

    g = df.groupby("vehicle_id", sort=False)
    pieces = [df]

    gap = g["time_step"].diff().replace(0, np.nan)
    pieces.append(gap.rename("time_gap"))

    previous = g[feature_cols].shift(1)
    previous.columns = [
        f"{c}__prev" for c in feature_cols
    ]
    pieces.append(previous)

    delta = df[feature_cols] - g[feature_cols].shift(1)
    delta.columns = [
        f"{c}__delta" for c in feature_cols
    ]
    pieces.append(delta)

    rate = delta.div(gap, axis=0)
    rate.columns = [
        f"{c}__rate" for c in feature_cols
    ]
    pieces.append(rate)

    for lag in (3, 5):
        change = df[feature_cols] - g[feature_cols].shift(lag)
        change.columns = [
            f"{c}__change{lag}" for c in feature_cols
        ]
        pieces.append(change)

    for window in (3, 5):
        mean = (
            g[feature_cols]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
        )
        mean.columns = [
            f"{c}__mean{window}" for c in feature_cols
        ]
        pieces.append(mean)

        std = (
            g[feature_cols]
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
        )
        std.columns = [
            f"{c}__std{window}" for c in feature_cols
        ]
        pieces.append(std)

    result = pd.concat(pieces, axis=1)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return result


print("\nBuilding historical features...")

train_history = create_history_features(
    train_op, raw_features
)
val_history = create_history_features(
    val_op, raw_features
)


# ============================================================
# 4. TRAINING LABELS AND CENSORING
# ============================================================

train_full = train_history.merge(
    train_tte,
    on="vehicle_id",
    how="left",
    validate="many_to_one",
)

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    - train_full["time_step"]
)


def tte_to_class(tte):
    if pd.isna(tte) or tte < 0:
        return np.nan
    if tte <= 6:
        return 4
    if tte <= 12:
        return 3
    if tte <= 24:
        return 2
    if tte <= 48:
        return 1
    return 0


train_full["class_label"] = np.nan

repair_mask = train_full["in_study_repair"] == 1

train_full.loc[repair_mask, "class_label"] = (
    train_full.loc[repair_mask, "time_to_endpoint"]
    .apply(tte_to_class)
)

# Preserve Test 13's five-class eligible population.
safe_nonrepair = (
    (train_full["in_study_repair"] == 0)
    & (train_full["time_to_endpoint"] > 48)
)

train_full.loc[safe_nonrepair, "class_label"] = 0

eligible = train_full.dropna(
    subset=["class_label"]
).copy()

eligible["class_label"] = (
    eligible["class_label"].astype(int)
)

# Binary target:
# 1 = C4 (repair within 6 operational units)
# 0 = C0-C3
#
# Censoring caveat:
# This preserves the existing five-class eligible population.
# It is NOT a fully horizon-specific censored dataset.
eligible["binary_target"] = (
    eligible["class_label"] == 4
).astype(int)


# ============================================================
# 5. VEHICLE-AWARE SAMPLING
# Identical rows for both models
# ============================================================

repair_samples = (
    eligible[eligible["in_study_repair"] == 1]
    .groupby(["vehicle_id", "class_label"])
    .sample(n=1, random_state=SEED)
)

nonrepair_samples = (
    eligible[eligible["in_study_repair"] == 0]
    .groupby("vehicle_id")
    .sample(n=1, random_state=SEED)
)

train_sampled = pd.concat(
    [repair_samples, nonrepair_samples],
    ignore_index=True
).sample(frac=1, random_state=SEED).reset_index(drop=True)

train_sampled = train_sampled.merge(
    train_spec,
    on="vehicle_id",
    how="left",
    validate="many_to_one",
)

print("\nTraining class distribution:")
print(train_sampled["class_label"].value_counts().sort_index())


# ============================================================
# 6. OFFICIAL VALIDATION
# ============================================================

val_last = (
    val_history
    .groupby("vehicle_id", sort=False)
    .tail(1)
    .copy()
)

val_last = val_last.merge(
    val_spec,
    on="vehicle_id",
    how="left",
    validate="one_to_one",
)

val_last = val_last.merge(
    val_labels,
    on="vehicle_id",
    how="left",
    validate="one_to_one",
)

assert val_last["class_label"].notna().all()
assert val_last["vehicle_id"].is_unique

y_train_multi = train_sampled["class_label"].astype(int)
y_train_binary = train_sampled["binary_target"].astype(int)

y_val_multi = val_last["class_label"].astype(int)
y_val_binary = (y_val_multi == 4).astype(int)

assert set(y_train_multi.unique()) == {0, 1, 2, 3, 4}
assert y_train_binary.sum() > 0
assert y_val_binary.sum() > 0
assert y_val_binary.sum() < len(y_val_binary)


# ============================================================
# 7. FEATURE AUDIT
# ============================================================

excluded = FEATURE_EXCLUSIONS | {"binary_target"}

X_train = train_sampled.drop(
    columns=list(excluded),
    errors="ignore"
)

X_val_raw = val_last.drop(
    columns=list(excluded),
    errors="ignore"
)

missing_in_val = sorted(set(X_train.columns) - set(X_val_raw.columns))
extra_in_val = sorted(set(X_val_raw.columns) - set(X_train.columns))

print("\n=== FEATURE AUDIT ===")
print("Train feature count:", len(X_train.columns))
print("Missing validation features:", missing_in_val)
print("Extra validation features:", extra_in_val)

if missing_in_val:
    raise ValueError(
        "Validation is missing training features. "
        "Fix the feature pipeline before comparing models."
    )

X_val = X_val_raw[X_train.columns].copy()

assert X_train.columns.equals(X_val.columns)

# Ensure no known target/endpoint columns remain.
assert not (set(X_train.columns) & excluded)

# IMPORTANT:
# This audit checks known leakage columns and schema consistency.
# It cannot automatically prove that every other feature
# was available at prediction time.


# ============================================================
# 8. PREPROCESSING
# ============================================================

categorical_cols = [
    c for c in X_train.columns
    if c.startswith("Spec_")
]

numeric_cols = [
    c for c in X_train.columns
    if c not in categorical_cols
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

print("\nTransformed features:", X_train_t.shape)


# ============================================================
# 9. MODEL FACTORY
# ============================================================

common_params = dict(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

multiclass_model = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    **common_params
)

binary_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    **common_params
)


# ============================================================
# 10. TRAIN MODELS
# ============================================================

print("\nTraining multiclass baseline...")
multiclass_model.fit(X_train_t, y_train_multi)

print("Training direct binary model...")
binary_model.fit(X_train_t, y_train_binary)

# Baseline: five-class model's C4 probability
multi_classes = list(multiclass_model.classes_)
c4_index = multi_classes.index(4)

baseline_scores = (
    multiclass_model.predict_proba(X_val_t)[:, c4_index]
)

# Direct binary prediction
binary_scores = binary_model.predict_proba(X_val_t)[:, 1]


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate_ranking(name, scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)

    ranking = np.argsort(-scores, kind="stable")
    total_positives = int(labels.sum())

    rows = []

    for capacity in CAPACITIES:
        k = max(1, int(np.ceil(len(labels) * capacity)))
        selected = ranking[:k]

        found = int(labels[selected].sum())

        precision = found / k
        recall = found / total_positives

        rows.append({
            "Model": name,
            "Capacity": capacity,
            "VehiclesSelected": k,
            "C4Found": found,
            "C4Precision": precision,
            "C4Recall": recall,
            "Risk6_AP": ap,
            "Risk6_AUC": auc,
        })

    return pd.DataFrame(rows)


baseline_result = evaluate_ranking(
    "Multiclass_P_C4",
    baseline_scores,
    y_val_binary
)

binary_result = evaluate_ranking(
    "Direct_Binary_C4",
    binary_scores,
    y_val_binary
)

results = pd.concat(
    [baseline_result, binary_result],
    ignore_index=True
)


# ============================================================
# 12. COMPARE AT EQUAL INSPECTION CAPACITY
# ============================================================

comparison = baseline_result.merge(
    binary_result,
    on="Capacity",
    suffixes=("_Baseline", "_Binary")
)

comparison["Recall_Gain_pp"] = (
    comparison["C4Recall_Binary"]
    - comparison["C4Recall_Baseline"]
) * 100

comparison["Precision_Gain_pp"] = (
    comparison["C4Precision_Binary"]
    - comparison["C4Precision_Baseline"]
) * 100

print("\n=== DIRECT BINARY VS MULTICLASS ===")

print(
    comparison[
        [
            "Capacity",
            "C4Recall_Baseline",
            "C4Recall_Binary",
            "Recall_Gain_pp",
            "C4Precision_Baseline",
            "C4Precision_Binary",
        ]
    ].round(4).to_string(index=False)
)

print("\nAverage Precision:")
print(
    "Multiclass:",
    round(average_precision_score(
        y_val_binary, baseline_scores
    ), 4)
)
print(
    "Binary:",
    round(average_precision_score(
        y_val_binary, binary_scores
    ), 4)
)

print("\nROC-AUC:")
print(
    "Multiclass:",
    round(roc_auc_score(
        y_val_binary, baseline_scores
    ), 4)
)
print(
    "Binary:",
    round(roc_auc_score(
        y_val_binary, binary_scores
    ), 4)
)


# ============================================================
# 13. OPTIONAL BINARY THRESHOLD DIAGNOSTIC
# Not used to select the final model
# ============================================================

binary_pred = (binary_scores >= 0.5).astype(int)

print("\n=== BINARY THRESHOLD 0.5 DIAGNOSTIC ===")
print(classification_report(
    y_val_binary,
    binary_pred,
    zero_division=0
))

print(confusion_matrix(
    y_val_binary,
    binary_pred
))

# Do not tune this threshold on the final test set.


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR / "test_14_ranking_results.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "test_14_binary_comparison.csv",
    index=False
)

predictions = pd.DataFrame({
    "vehicle_id": val_last["vehicle_id"].to_numpy(),
    "actual_class": y_val_multi.to_numpy(),
    "actual_C4": y_val_binary.to_numpy(),
    "multiclass_risk6": baseline_scores,
    "binary_risk6": binary_scores,
})

predictions.to_csv(
    OUTPUT_DIR / "test_14_predictions.csv",
    index=False
)

print("\nTEST 14 COMPLETE")
print("Results saved to:", OUTPUT_DIR)
