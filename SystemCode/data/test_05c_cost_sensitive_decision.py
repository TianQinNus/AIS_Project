# ============================================================
# No Cost concern is currently considered in our project
# FleetCare AI
# Model 5: Cost-Sensitive Decision Layer
# Reference model:
#   Vehicle-Aware Temporal XGBoost + W1 Light Class Weighting
#
# Goal:
#   Keep the Model 4 training pipeline fixed, and compare:
#
#   A) Standard argmax decision
#      predicted_class = argmax P(Ck | x)
#
#   B) Minimum Expected Cost (MEC) decision
#      expected_cost(j) = sum_i P(Ci | x) * Cost(actual=i, predicted=j)
#      predicted_class = argmin_j expected_cost(j)
#
# IMPORTANT:
#   This experiment uses the raw probabilities from the weighted,
#   vehicle-aware XGBoost model. Because class weighting and sampling
#   change the effective class distribution, these probabilities are
#   not guaranteed to be perfectly calibrated real-world probabilities.
#   Therefore Model 5 should be treated as a cost-aware decision
#   experiment. Probability calibration should be tested next before
#   presenting scores as literal failure probabilities in the web app.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
NUM_CLASSES = 5
CLASS_NAMES = ["C0", "C1", "C2", "C3", "C4"]

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 260)


# ============================================================
# 2. PATHS
# ============================================================
#
# This file is expected to sit in the same "data" folder as
# test_00 ... test_04, with raw/ and processed/ as subfolders.
#
# We also provide fallbacks so it can still run when launched
# from a slightly different repository structure.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_dir():
    candidates = [
        SCRIPT_DIR / "raw",
        SCRIPT_DIR.parent / "data" / "raw",
        Path("../data/raw").resolve(),
        Path("data/raw").resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Use the most likely path in the final error message.
    return SCRIPT_DIR / "raw"


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = DATA_DIR.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"


# ============================================================
# 3. CHECK FILES
# ============================================================

required_files = [
    TRAIN_OP_PATH,
    TRAIN_SPEC_PATH,
    TRAIN_TTE_PATH,
    VAL_OP_PATH,
    VAL_SPEC_PATH,
    VAL_LABEL_PATH,
]

print("=" * 90)
print("MODEL 5 — COST-SENSITIVE DECISION")
print("=" * 90)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

print("\nChecking files:")
for path in required_files:
    print(f"  {path.name}: {path.exists()}")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Current DATA_DIR = {DATA_DIR}\n"
            "Please place test_05 in the same data folder used by "
            "test_00-test_04, or update DATA_DIR."
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LOADING DATA")
print("=" * 90)

train_op = pd.read_csv(TRAIN_OP_PATH)
train_spec = pd.read_csv(TRAIN_SPEC_PATH)
train_tte = pd.read_csv(TRAIN_TTE_PATH)

val_op = pd.read_csv(VAL_OP_PATH)
val_spec = pd.read_csv(VAL_SPEC_PATH)
val_labels = pd.read_csv(VAL_LABEL_PATH)

print("Train operational:", train_op.shape)
print("Train specifications:", train_spec.shape)
print("Train TTE:", train_tte.shape)

print("Validation operational:", val_op.shape)
print("Validation specifications:", val_spec.shape)
print("Validation labels:", val_labels.shape)


# ============================================================
# 5. BASIC VEHICLE CHECKS
# ============================================================

train_vehicle_count = train_op["vehicle_id"].nunique()
val_vehicle_count = val_op["vehicle_id"].nunique()

assert train_vehicle_count == train_spec["vehicle_id"].nunique()
assert train_vehicle_count == train_tte["vehicle_id"].nunique()

print("\nTraining vehicles:", train_vehicle_count)
print("Validation vehicles:", val_vehicle_count)


# ============================================================
# 6. IDENTIFY RAW OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in ["vehicle_id", "time_step"]
]

print("\nRaw operational features:", len(RAW_FEATURE_COLS))


# ============================================================
# 7. SORT TIME SERIES
# ============================================================

train_op = (
    train_op
    .sort_values(["vehicle_id", "time_step"])
    .reset_index(drop=True)
)

val_op = (
    val_op
    .sort_values(["vehicle_id", "time_step"])
    .reset_index(drop=True)
)


# ============================================================
# 8. TEMPORAL FEATURE ENGINEERING
# ============================================================
#
# Same logic as Model 4:
#   current value x(t)
#   previous value x(t-1)
#   delta = x(t) - x(t-1)
#   rate  = delta / time_gap
#
# The rate term matters because SCANIA readouts are irregularly
# sampled in time_step.
# ============================================================

def create_temporal_features(df, feature_cols):
    df = (
        df
        .sort_values(["vehicle_id", "time_step"])
        .reset_index(drop=True)
        .copy()
    )

    grouped = df.groupby("vehicle_id", sort=False)

    # Time gap
    df["time_gap"] = grouped["time_step"].diff()
    safe_time_gap = df["time_gap"].replace(0, np.nan)

    # Previous values
    previous = grouped[feature_cols].shift(1)
    previous.columns = [f"{col}__prev" for col in feature_cols]

    # Convert to arrays for efficient arithmetic
    current_array = df[feature_cols].to_numpy(dtype=np.float32)
    previous_array = previous.to_numpy(dtype=np.float32)

    # Delta
    delta_array = current_array - previous_array
    delta = pd.DataFrame(
        delta_array,
        index=df.index,
        columns=[f"{col}__delta" for col in feature_cols],
    )

    # Rate
    time_gap_array = (
        safe_time_gap
        .to_numpy(dtype=np.float32)
        [:, None]
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_array = delta_array / time_gap_array

    rate = pd.DataFrame(
        rate_array,
        index=df.index,
        columns=[f"{col}__rate" for col in feature_cols],
    )

    rate = rate.replace([np.inf, -np.inf], np.nan)

    return pd.concat(
        [df, previous, delta, rate],
        axis=1,
    )


print("\n" + "=" * 90)
print("CREATING TEMPORAL FEATURES")
print("=" * 90)

train_temporal = create_temporal_features(
    train_op,
    RAW_FEATURE_COLS,
)

val_temporal = create_temporal_features(
    val_op,
    RAW_FEATURE_COLS,
)

print("Train temporal shape:", train_temporal.shape)
print("Validation temporal shape:", val_temporal.shape)


# ============================================================
# 9. MERGE TRAINING DATA WITH TTE
# ============================================================

train_full = train_temporal.merge(
    train_tte,
    on="vehicle_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 10. TIME TO ENDPOINT
# ============================================================
#
# Repair vehicle:
#   endpoint = Component X repair
#
# Non-repair / censored vehicle:
#   endpoint = study observation end
# ============================================================

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    - train_full["time_step"]
)


# ============================================================
# 11. OFFICIAL C0-C4 CLASS FUNCTION
# ============================================================

def tte_to_class(tte):
    if pd.isna(tte):
        return np.nan

    if tte < 0:
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


# ============================================================
# 12. CREATE CENSORING-AWARE TRAINING LABELS
# ============================================================

train_full["class_label"] = np.nan

# Repair vehicles: endpoint is the observed repair.
repair_mask = train_full["in_study_repair"] == 1

train_full.loc[
    repair_mask,
    "class_label"
] = (
    train_full.loc[
        repair_mask,
        "time_to_endpoint"
    ]
    .apply(tte_to_class)
)

# Non-repair vehicles:
# only observations with >48 known repair-free time units
# are safely labelled as C0.
safe_non_repair_mask = (
    (train_full["in_study_repair"] == 0)
    &
    (train_full["time_to_endpoint"] > 48)
)

train_full.loc[
    safe_non_repair_mask,
    "class_label"
] = 0


# ============================================================
# 13. REMOVE UNKNOWN / UNCERTAIN LABELS
# ============================================================

candidate_train = (
    train_full
    .dropna(subset=["class_label"])
    .copy()
)

candidate_train["class_label"] = (
    candidate_train["class_label"]
    .astype(int)
)

print("\nEligible observation distribution:")
print(
    candidate_train["class_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 14. VEHICLE-AWARE SAMPLING
# ============================================================
#
# Same strategy as Model 4:
#
# Repair vehicle:
#   max one observation per vehicle per available class.
#
# Non-repair vehicle:
#   one safely-labelled C0 observation per vehicle.
# ============================================================

repair_candidates = (
    candidate_train[
        candidate_train["in_study_repair"] == 1
    ]
    .copy()
)

non_repair_candidates = (
    candidate_train[
        candidate_train["in_study_repair"] == 0
    ]
    .copy()
)

repair_sampled = (
    repair_candidates
    .groupby(
        ["vehicle_id", "class_label"],
        group_keys=False,
    )
    .sample(
        n=1,
        random_state=RANDOM_STATE,
    )
    .reset_index(drop=True)
)

non_repair_sampled = (
    non_repair_candidates
    .groupby(
        "vehicle_id",
        group_keys=False,
    )
    .sample(
        n=1,
        random_state=RANDOM_STATE,
    )
    .reset_index(drop=True)
)

train_sampled = pd.concat(
    [
        repair_sampled,
        non_repair_sampled,
    ],
    ignore_index=True,
)

train_sampled = (
    train_sampled
    .sample(
        frac=1,
        random_state=RANDOM_STATE,
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 90)
print("VEHICLE-AWARE TRAINING DISTRIBUTION")
print("=" * 90)

print(
    train_sampled["class_label"]
    .value_counts()
    .sort_index()
)

print("\nTraining rows:", len(train_sampled))
print(
    "Unique vehicles:",
    train_sampled["vehicle_id"].nunique()
)

assert set(train_sampled["class_label"].unique()) == {
    0, 1, 2, 3, 4
}


# ============================================================
# 15. JOIN VEHICLE SPECIFICATIONS
# ============================================================

train_sampled = train_sampled.merge(
    train_spec,
    on="vehicle_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 16. BUILD VALIDATION TABLE
# ============================================================
#
# Validation histories are already truncated by the organisers.
# Temporal features are created first, then only the last
# available readout of each vehicle is used as the prediction
# cutoff.
# ============================================================

val_last = (
    val_temporal
    .groupby(
        "vehicle_id",
        sort=False,
    )
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

print("\nValidation distribution:")
print(
    val_last["class_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 17. DEFINE MODEL FEATURES
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
]

X_train = (
    train_sampled
    .drop(
        columns=NON_FEATURE_COLS,
        errors="ignore",
    )
)

y_train = (
    train_sampled["class_label"]
    .astype(int)
    .copy()
)

X_val = (
    val_last
    .drop(
        columns=[
            "vehicle_id",
            "class_label",
        ],
        errors="ignore",
    )
)

y_val = (
    val_last["class_label"]
    .astype(int)
    .copy()
)


# ============================================================
# 18. ALIGN TRAIN / VALIDATION FEATURE SCHEMAS
# ============================================================

train_only = set(X_train.columns) - set(X_val.columns)
val_only = set(X_val.columns) - set(X_train.columns)

print("\nTrain-only columns:", train_only)
print("Validation-only columns:", val_only)

assert len(train_only) == 0
assert len(val_only) == 0

X_val = X_val[X_train.columns]

assert list(X_train.columns) == list(X_val.columns)

print("✓ Feature schemas aligned.")


# ============================================================
# 19. FEATURE TYPES
# ============================================================

categorical_cols = [
    col
    for col in X_train.columns
    if col.startswith("Spec_")
]

numeric_cols = [
    col
    for col in X_train.columns
    if col not in categorical_cols
]

print("\nNumeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Total input columns:", len(X_train.columns))


# ============================================================
# 20. PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_cols,
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_cols,
        ),
    ]
)


# ============================================================
# 21. OFFICIAL IDA COST MATRIX
# ============================================================
#
# Row    = Actual class
# Column = Predicted class
#
# Missing an urgent case is much more expensive than giving
# an early / unnecessary warning.
# ============================================================

COST_MATRIX = np.array(
    [
        [0,   7,   8,   9,  10],
        [200, 0,   7,   8,   9],
        [300, 200, 0,   7,   8],
        [400, 300, 200, 0,   7],
        [500, 400, 300, 200, 0],
    ],
    dtype=np.float64,
)


def calculate_total_cost(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    return float(
        COST_MATRIX[
            y_true,
            y_pred,
        ]
        .sum()
    )


# ============================================================
# 22. FIX MODEL 4 REFERENCE CONFIGURATION
# ============================================================
#
# Test 04 found W1 Light to be the preferred current strategy.
# We freeze it here so Test 05 changes ONLY the decision rule.
# ============================================================

LIGHT_WEIGHT_MAP = {
    0: 1.00,
    1: 1.25,
    2: 1.50,
    3: 1.75,
    4: 2.00,
}

sample_weights = (
    y_train
    .map(LIGHT_WEIGHT_MAP)
    .to_numpy(dtype=np.float32)
)

model = XGBClassifier(
    objective="multi:softprob",
    num_class=NUM_CLASSES,

    n_estimators=150,
    learning_rate=0.08,

    max_depth=5,
    min_child_weight=3,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    eval_metric="mlogloss",
    tree_method="hist",

    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            model,
        ),
    ]
)


# ============================================================
# 23. TRAIN REFERENCE MODEL
# ============================================================

print("\n" + "=" * 90)
print("TRAINING REFERENCE MODEL")
print("=" * 90)

print("Reference: Vehicle-aware Temporal XGBoost + W1 Light")
print("Weights:", LIGHT_WEIGHT_MAP)

pipeline.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights,
)


# ============================================================
# 24. PREDICT CLASS PROBABILITIES
# ============================================================

y_proba = (
    pipeline
    .predict_proba(X_val)
)

assert y_proba.shape == (
    len(X_val),
    NUM_CLASSES,
)

assert np.allclose(
    y_proba.sum(axis=1),
    1.0,
    atol=1e-5,
)

print("\nProbability matrix shape:", y_proba.shape)


# ============================================================
# 25. DECISION A — STANDARD ARGMAX
# ============================================================

argmax_pred = (
    np.argmax(
        y_proba,
        axis=1,
    )
    .astype(int)
)


# ============================================================
# 26. DECISION B — MINIMUM EXPECTED COST
# ============================================================
#
# For one vehicle:
#
# expected_cost[predicted_class = j]
#     = sum_i P(actual_class = i | x) * COST_MATRIX[i, j]
#
# Matrix form:
#
#   [P(C0), ..., P(C4)] @ COST_MATRIX
#
# gives five expected costs, one for each possible decision.
# ============================================================

expected_cost_matrix = (
    y_proba
    @
    COST_MATRIX
)

assert expected_cost_matrix.shape == (
    len(X_val),
    NUM_CLASSES,
)

mec_pred = (
    np.argmin(
        expected_cost_matrix,
        axis=1,
    )
    .astype(int)
)

mec_min_expected_cost = (
    expected_cost_matrix
    .min(axis=1)
)


# ============================================================
# 27. EVALUATION FUNCTION
# ============================================================

def evaluate_decision(
    name,
    y_true,
    y_pred,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pred_counts = (
        pd.Series(y_pred)
        .value_counts()
    )

    total_cost = calculate_total_cost(
        y_true,
        y_pred,
    )

    result = {
        "Decision": name,

        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Macro Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "Macro Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "Weighted F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "C0 Recall": report["C0"]["recall"],
        "C1 Recall": report["C1"]["recall"],
        "C2 Recall": report["C2"]["recall"],
        "C3 Recall": report["C3"]["recall"],
        "C4 Recall": report["C4"]["recall"],

        "C4 Precision": report["C4"]["precision"],
        "C4 F1": report["C4"]["f1-score"],

        "Pred C0": int(pred_counts.get(0, 0)),
        "Pred C1": int(pred_counts.get(1, 0)),
        "Pred C2": int(pred_counts.get(2, 0)),
        "Pred C3": int(pred_counts.get(3, 0)),
        "Pred C4": int(pred_counts.get(4, 0)),

        "Total Cost": total_cost,
        "Average Cost": total_cost / len(y_true),
    }

    return result


# ============================================================
# 28. COMPARE ARGMAX VS MEC
# ============================================================

argmax_result = evaluate_decision(
    "Argmax",
    y_val,
    argmax_pred,
)

mec_result = evaluate_decision(
    "MinimumExpectedCost",
    y_val,
    mec_pred,
)

results_df = pd.DataFrame(
    [
        argmax_result,
        mec_result,
    ]
)

display_cols = [
    "Decision",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "C0 Recall",
    "C1 Recall",
    "C2 Recall",
    "C3 Recall",
    "C4 Recall",
    "C4 Precision",
    "C4 F1",
    "Pred C0",
    "Pred C1",
    "Pred C2",
    "Pred C3",
    "Pred C4",
    "Total Cost",
    "Average Cost",
]

print("\n" + "=" * 90)
print("ARGMAX VS MINIMUM EXPECTED COST")
print("=" * 90)

print(
    results_df[
        display_cols
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 29. BUSINESS-COST IMPROVEMENT
# ============================================================

argmax_cost = argmax_result["Total Cost"]
mec_cost = mec_result["Total Cost"]

cost_reduction = (
    argmax_cost
    -
    mec_cost
)

if argmax_cost > 0:
    cost_reduction_pct = (
        cost_reduction
        /
        argmax_cost
        *
        100.0
    )
else:
    cost_reduction_pct = np.nan


changed_decisions = (
    argmax_pred
    !=
    mec_pred
)

num_changed = int(
    changed_decisions.sum()
)

pct_changed = (
    num_changed
    /
    len(y_val)
    *
    100.0
)


print("\n" + "=" * 90)
print("COST-AWARE DECISION SUMMARY")
print("=" * 90)

print("Argmax total cost:", argmax_cost)
print("MEC total cost:", mec_cost)
print("Cost reduction:", cost_reduction)
print(
    "Cost reduction (%):",
    round(cost_reduction_pct, 2)
    if not np.isnan(cost_reduction_pct)
    else np.nan,
)

print(
    "\nVehicles whose predicted decision changed:",
    num_changed,
    f"({pct_changed:.2f}%)",
)


# ============================================================
# 30. CLASSIFICATION REPORTS
# ============================================================

for decision_name, pred in [
    ("ARGMAX", argmax_pred),
    ("MINIMUM EXPECTED COST", mec_pred),
]:
    print("\n" + "=" * 90)
    print(f"CLASSIFICATION REPORT — {decision_name}")
    print("=" * 90)

    print(
        classification_report(
            y_val,
            pred,
            labels=[0, 1, 2, 3, 4],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# 31. CONFUSION MATRICES
# ============================================================

argmax_cm = confusion_matrix(
    y_val,
    argmax_pred,
    labels=[0, 1, 2, 3, 4],
)

mec_cm = confusion_matrix(
    y_val,
    mec_pred,
    labels=[0, 1, 2, 3, 4],
)

print("\n" + "=" * 90)
print("CONFUSION MATRIX — ARGMAX")
print("=" * 90)
print(argmax_cm)

print("\n" + "=" * 90)
print("CONFUSION MATRIX — MINIMUM EXPECTED COST")
print("=" * 90)
print(mec_cm)


# ============================================================
# 32. DECISION-CHANGE ANALYSIS
# ============================================================
#
# This table helps answer:
# "When MEC changes the normal XGBoost decision, where does it
# move the prediction?"
# ============================================================

decision_change_table = pd.crosstab(
    pd.Series(
        argmax_pred,
        name="Argmax",
    ),
    pd.Series(
        mec_pred,
        name="MEC",
    ),
    dropna=False,
)

decision_change_table = (
    decision_change_table
    .reindex(
        index=[0, 1, 2, 3, 4],
        columns=[0, 1, 2, 3, 4],
        fill_value=0,
    )
)

print("\n" + "=" * 90)
print("DECISION CHANGE TABLE")
print("Rows = Argmax prediction; Columns = MEC prediction")
print("=" * 90)
print(decision_change_table)


# ============================================================
# 33. SAVE SUMMARY RESULTS
# ============================================================

RESULTS_PATH = (
    OUTPUT_DIR
    /
    "model_05_cost_sensitive_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


# ============================================================
# 34. SAVE VEHICLE-LEVEL PREDICTIONS
# ============================================================

vehicle_results = pd.DataFrame(
    {
        "vehicle_id":
            val_last["vehicle_id"].to_numpy(),

        "actual_class":
            y_val.to_numpy(),

        "argmax_class":
            argmax_pred,

        "mec_class":
            mec_pred,

        "decision_changed":
            changed_decisions,

        "P_C0": y_proba[:, 0],
        "P_C1": y_proba[:, 1],
        "P_C2": y_proba[:, 2],
        "P_C3": y_proba[:, 3],
        "P_C4": y_proba[:, 4],

        "ExpectedCost_PredictC0":
            expected_cost_matrix[:, 0],

        "ExpectedCost_PredictC1":
            expected_cost_matrix[:, 1],

        "ExpectedCost_PredictC2":
            expected_cost_matrix[:, 2],

        "ExpectedCost_PredictC3":
            expected_cost_matrix[:, 3],

        "ExpectedCost_PredictC4":
            expected_cost_matrix[:, 4],

        "MEC_MinExpectedCost":
            mec_min_expected_cost,
    }
)

VEHICLE_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "model_05_validation_predictions.csv"
)

vehicle_results.to_csv(
    VEHICLE_RESULTS_PATH,
    index=False,
)


# ============================================================
# 35. SAVE CONFUSION MATRICES
# ============================================================

argmax_cm_df = pd.DataFrame(
    argmax_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

mec_cm_df = pd.DataFrame(
    mec_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

ARGMAX_CM_PATH = (
    OUTPUT_DIR
    /
    "model_05_argmax_confusion_matrix.csv"
)

MEC_CM_PATH = (
    OUTPUT_DIR
    /
    "model_05_mec_confusion_matrix.csv"
)

argmax_cm_df.to_csv(ARGMAX_CM_PATH)
mec_cm_df.to_csv(MEC_CM_PATH)


# ============================================================
# 36. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 90)
print("MODEL 5 COMPLETE")
print("=" * 90)

if mec_cost < argmax_cost:
    print(
        "MEC reduced the official validation cost relative to "
        "standard argmax."
    )
elif mec_cost == argmax_cost:
    print(
        "MEC and argmax produced the same official validation cost."
    )
else:
    print(
        "MEC increased the official validation cost. "
        "Do NOT adopt it automatically; inspect probability "
        "calibration and decision changes first."
    )

print(
    "\nImportant caveat: raw probabilities from the weighted, "
    "vehicle-aware XGBoost model are not guaranteed to be "
    "well calibrated."
)

print(
    "Recommended next step: probability calibration on a "
    "vehicle-level held-out development set, then repeat the "
    "cost-sensitive decision comparison."
)

print("\nSaved files:")
print(" ", RESULTS_PATH)
print(" ", VEHICLE_RESULTS_PATH)
print(" ", ARGMAX_CM_PATH)
print(" ", MEC_CM_PATH)
