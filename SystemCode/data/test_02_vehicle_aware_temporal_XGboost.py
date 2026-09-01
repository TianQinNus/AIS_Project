# %%
# ============================================================
# SCANIA COMPONENT X
# VEHICLE-AWARE TEMPORAL XGBOOST
# ============================================================
#
# Main idea:
#
# Previous model:
#   Every eligible historical readout = one training sample
#
# Problem:
#   One vehicle may contribute dozens/hundreds of highly
#   correlated C0 observations.
#
# New strategy:
#
#   Repair vehicle:
#       max 1 observation from each available class window
#       C0 / C1 / C2 / C3 / C4
#
#   Non-repair vehicle:
#       max 1 safely-labelled C0 observation
#
# This makes the training unit more vehicle-aware while
# preserving repair-proximity stages.
#
# ============================================================


# %%
# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


RANDOM_STATE = 42

pd.set_option("display.max_columns", 150)
pd.set_option("display.max_rows", 100)


# %%
# ============================================================
# 2. PATHS
# ============================================================

DATA_DIR = Path("../data/raw")

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"

OUTPUT_DIR = Path("../data/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# %%
# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

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


# %%
# ============================================================
# 4. BASIC CHECKS
# ============================================================

print("\nTrain vehicles:")
print(train_op["vehicle_id"].nunique())

print("\nValidation vehicles:")
print(val_op["vehicle_id"].nunique())

assert (
    train_op["vehicle_id"].nunique()
    ==
    train_tte["vehicle_id"].nunique()
)

assert (
    train_op["vehicle_id"].nunique()
    ==
    train_spec["vehicle_id"].nunique()
)


# %%
# ============================================================
# 5. IDENTIFY OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in [
        "vehicle_id",
        "time_step"
    ]
]

print(
    "\nOperational feature columns:",
    len(RAW_FEATURE_COLS)
)


# %%
# ============================================================
# 6. SORT TIME SERIES
# ============================================================

train_op = (
    train_op
    .sort_values(
        ["vehicle_id", "time_step"]
    )
    .reset_index(drop=True)
)

val_op = (
    val_op
    .sort_values(
        ["vehicle_id", "time_step"]
    )
    .reset_index(drop=True)
)


# %%
# ============================================================
# 7. TEMPORAL FEATURE ENGINEERING
# ============================================================
#
# For each operational feature x:
#
#   current:
#       x_t
#
#   previous:
#       x_(t-1)
#
#   delta:
#       x_t - x_(t-1)
#
#   rate:
#       (x_t - x_(t-1)) /
#       (time_t - time_(t-1))
#
# IMPORTANT:
#
# Temporal features MUST be created BEFORE sampling the cutoff.
# Otherwise the selected row would lose access to its previous
# readout.
# ============================================================

def create_temporal_features(
    df,
    feature_cols
):

    df = (
        df
        .sort_values(
            ["vehicle_id", "time_step"]
        )
        .copy()
    )

    grouped = df.groupby(
        "vehicle_id",
        sort=False
    )

    # Time difference
    df["time_gap"] = (
        grouped["time_step"]
        .diff()
    )

    safe_time_gap = (
        df["time_gap"]
        .replace(0, np.nan)
    )

    # Previous observation
    previous = (
        grouped[feature_cols]
        .shift(1)
    )

    previous.columns = [
        f"{col}__prev"
        for col in feature_cols
    ]

    # Current values
    current_values = (
        df[feature_cols]
        .to_numpy(dtype=float)
    )

    previous_values = (
        previous
        .to_numpy(dtype=float)
    )

    # Delta
    delta_values = (
        current_values
        -
        previous_values
    )

    delta = pd.DataFrame(
        delta_values,
        index=df.index,
        columns=[
            f"{col}__delta"
            for col in feature_cols
        ]
    )

    # Rate of change
    rate_values = (
        delta_values
        /
        safe_time_gap.to_numpy()[:, None]
    )

    rate = pd.DataFrame(
        rate_values,
        index=df.index,
        columns=[
            f"{col}__rate"
            for col in feature_cols
        ]
    )

    rate = rate.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = pd.concat(
        [
            df,
            previous,
            delta,
            rate
        ],
        axis=1
    )

    return result


# %%
# ============================================================
# 8. BUILD TEMPORAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEMPORAL FEATURES")
print("=" * 70)

train_temporal = create_temporal_features(
    train_op,
    RAW_FEATURE_COLS
)

val_temporal = create_temporal_features(
    val_op,
    RAW_FEATURE_COLS
)

print(
    "Train temporal:",
    train_temporal.shape
)

print(
    "Validation temporal:",
    val_temporal.shape
)


# %%
# ============================================================
# 9. MERGE TRAIN WITH TTE
# ============================================================

train_full = train_temporal.merge(
    train_tte,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)


# %%
# ============================================================
# 10. TIME TO ENDPOINT
# ============================================================
#
# For repair vehicles:
#   endpoint = first Component X repair
#
# For non-repair vehicles:
#   endpoint = end of observation study
#
# ============================================================

train_full[
    "time_to_endpoint"
] = (
    train_full[
        "length_of_study_time_step"
    ]
    -
    train_full[
        "time_step"
    ]
)


# %%
# ============================================================
# 11. OFFICIAL CLASS FUNCTION
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


# %%
# ============================================================
# 12. CREATE CANDIDATE LABELS
# ============================================================

train_full["class_label"] = np.nan


# Repair vehicles
repair_mask = (
    train_full["in_study_repair"]
    == 1
)

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
# only observations with >48 units of known repair-free
# follow-up are safe C0 observations.

safe_non_repair_mask = (
    (
        train_full[
            "in_study_repair"
        ] == 0
    )
    &
    (
        train_full[
            "time_to_endpoint"
        ] > 48
    )
)

train_full.loc[
    safe_non_repair_mask,
    "class_label"
] = 0


# %%
# ============================================================
# 13. REMOVE UNKNOWN LABELS
# ============================================================

candidate_train = (
    train_full
    .dropna(
        subset=["class_label"]
    )
    .copy()
)

candidate_train[
    "class_label"
] = (
    candidate_train[
        "class_label"
    ]
    .astype(int)
)


print("\n" + "=" * 70)
print("CANDIDATE OBSERVATION DISTRIBUTION")
print("=" * 70)

print(
    candidate_train[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# %%
# ============================================================
# 14. VEHICLE-AWARE SAMPLING
# ============================================================
#
# Strategy:
#
# Repair vehicles:
#   For every class available for that vehicle,
#   randomly select ONE readout.
#
# Example:
#
# Vehicle A:
#
#   C0 -> one sample
#   C1 -> one sample
#   C2 -> one sample
#   C3 -> one sample
#   C4 -> one sample
#
#
# Non-repair vehicles:
#   Select ONE safe C0 observation.
#
# ============================================================


repair_candidates = (
    candidate_train[
        candidate_train[
            "in_study_repair"
        ] == 1
    ]
    .copy()
)

non_repair_candidates = (
    candidate_train[
        candidate_train[
            "in_study_repair"
        ] == 0
    ]
    .copy()
)


# %%
# ============================================================
# 15. SAMPLE REPAIR VEHICLES
# ============================================================
#
# group = vehicle_id + class_label
#
# max 1 observation from each available class window
# ============================================================

repair_sampled = (
    repair_candidates
    .groupby(
        [
            "vehicle_id",
            "class_label"
        ],
        group_keys=False
    )
    .sample(
        n=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


print(
    "\nRepair samples:",
    len(repair_sampled)
)

print(
    repair_sampled[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# %%
# ============================================================
# 16. SAMPLE NON-REPAIR VEHICLES
# ============================================================
#
# One safe C0 cutoff per non-repair vehicle.
#
# ============================================================

non_repair_sampled = (
    non_repair_candidates
    .groupby(
        "vehicle_id",
        group_keys=False
    )
    .sample(
        n=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


print(
    "\nNon-repair C0 samples:",
    len(non_repair_sampled)
)


# %%
# ============================================================
# 17. COMBINE VEHICLE-AWARE TRAINING SET
# ============================================================

train_sampled = pd.concat(
    [
        repair_sampled,
        non_repair_sampled
    ],
    ignore_index=True
)


# Shuffle
train_sampled = (
    train_sampled
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


# %%
# ============================================================
# 18. CHECK VEHICLE-AWARE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("VEHICLE-AWARE TRAINING DISTRIBUTION")
print("=" * 70)

sample_counts = (
    train_sampled[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)

sample_props = (
    train_sampled[
        "class_label"
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
)

print("\nCounts:")
print(sample_counts)

print("\nProportions:")
print(sample_props)

print(
    "\nTraining rows:",
    len(train_sampled)
)

print(
    "Unique vehicles:",
    train_sampled[
        "vehicle_id"
    ].nunique()
)


assert set(
    train_sampled[
        "class_label"
    ].unique()
) == {0, 1, 2, 3, 4}

print(
    "\n✓ All five classes are present."
)


# %%
# ============================================================
# 19. NUMBER OF SAMPLES PER VEHICLE
# ============================================================

samples_per_vehicle = (
    train_sampled
    .groupby("vehicle_id")
    .size()
)

print("\nSamples per vehicle:")

print(
    samples_per_vehicle
    .describe()
)


# %%
# ============================================================
# 20. JOIN VEHICLE SPECIFICATIONS
# ============================================================

train_sampled = (
    train_sampled
    .merge(
        train_spec,
        on="vehicle_id",
        how="left",
        validate="many_to_one"
    )
)


# %%
# ============================================================
# 21. BUILD VALIDATION SET
# ============================================================
#
# Official validation:
# prediction at LAST AVAILABLE READOUT
#
# ============================================================

val_last = (
    val_temporal
    .groupby(
        "vehicle_id",
        sort=False
    )
    .tail(1)
    .copy()
)


val_last = (
    val_last
    .merge(
        val_spec,
        on="vehicle_id",
        how="left",
        validate="one_to_one"
    )
)


val_last = (
    val_last
    .merge(
        val_labels,
        on="vehicle_id",
        how="left",
        validate="one_to_one"
    )
)


print(
    "\nValidation rows:",
    len(val_last)
)

assert (
    len(val_last)
    ==
    val_last[
        "vehicle_id"
    ].nunique()
)


# %%
# ============================================================
# 22. VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

print(
    val_last[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# %%
# ============================================================
# 23. DEFINE MODEL FEATURES
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",

    # Future/outcome information
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint"
]


X_train = (
    train_sampled
    .drop(
        columns=NON_FEATURE_COLS,
        errors="ignore"
    )
)

y_train = (
    train_sampled[
        "class_label"
    ]
    .astype(int)
    .copy()
)


X_val = (
    val_last
    .drop(
        columns=[
            "vehicle_id",
            "class_label"
        ],
        errors="ignore"
    )
)

y_val = (
    val_last[
        "class_label"
    ]
    .astype(int)
    .copy()
)


# %%
# ============================================================
# 24. ALIGN FEATURES
# ============================================================

train_only = (
    set(X_train.columns)
    -
    set(X_val.columns)
)

val_only = (
    set(X_val.columns)
    -
    set(X_train.columns)
)

print(
    "\nTrain-only:",
    train_only
)

print(
    "Validation-only:",
    val_only
)

assert len(train_only) == 0
assert len(val_only) == 0


X_val = X_val[
    X_train.columns
]


assert (
    list(X_train.columns)
    ==
    list(X_val.columns)
)

print(
    "✓ Feature schemas aligned."
)


# %%
# ============================================================
# 25. CATEGORICAL / NUMERICAL FEATURES
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


print(
    "\nNumerical features:",
    len(numeric_cols)
)

print(
    "Categorical features:",
    len(categorical_cols)
)

print(
    "Total features:",
    len(X_train.columns)
)


# %%
# ============================================================
# 26. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_cols
    ),
    (
        "categorical",
        categorical_transformer,
        categorical_cols
    )
])


# %%
# ============================================================
# 27. CLASS WEIGHTS
# ============================================================
#
# Even after vehicle-aware sampling, the classes are not
# perfectly balanced.
#
# Use class weights instead of further random undersampling.
#
# ============================================================

classes = np.array(
    [0, 1, 2, 3, 4]
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_map = dict(
    zip(
        classes,
        class_weights
    )
)


print("\nClass weights:")

for cls, weight in class_weight_map.items():

    print(
        f"C{cls}: {weight:.4f}"
    )


sample_weights = (
    y_train
    .map(
        class_weight_map
    )
    .to_numpy()
)


# %%
# ============================================================
# 28. XGBOOST
# ============================================================
#
# Start with 150 trees.
#
# This is NOT final hyperparameter tuning.
# First determine whether vehicle-aware training improves
# minority-class prediction.
#
# ============================================================

xgb_model = XGBClassifier(

    objective="multi:softprob",
    num_class=5,

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
    n_jobs=-1
)


xgb_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        xgb_model
    )
])


# %%
# ============================================================
# 29. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING VEHICLE-AWARE TEMPORAL XGBOOST")
print("=" * 70)

print(
    "Training rows:",
    len(X_train)
)

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

print(
    "✓ Training complete"
)


# %%
# ============================================================
# 30. PREDICT
# ============================================================

y_pred = (
    xgb_pipeline
    .predict(X_val)
    .astype(int)
)

y_proba = (
    xgb_pipeline
    .predict_proba(X_val)
)


assert y_proba.shape == (
    len(X_val),
    5
)


# %%
# ============================================================
# 31. PREDICTED DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(
        y_pred,
        name="predicted_class"
    )
    .value_counts()
    .sort_index()
)


# %%
# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_pred,

        labels=[
            0,
            1,
            2,
            3,
            4
        ],

        target_names=[
            "C0",
            "C1",
            "C2",
            "C3",
            "C4"
        ],

        digits=4,
        zero_division=0
    )
)


# %%
# ============================================================
# 33. METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted"
)


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        macro_f1,
        4
    )
)

print(
    "Weighted F1:",
    round(
        weighted_f1,
        4
    )
)


# %%
# ============================================================
# 34. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)


disp.plot(
    values_format="d"
)

plt.title(
    "Vehicle-Aware Temporal XGBoost "
    "— Validation"
)

plt.show()


# %%
# ============================================================
# 35. PER-CLASS PERFORMANCE TABLE
# ============================================================

report = classification_report(
    y_val,
    y_pred,

    labels=[
        0,
        1,
        2,
        3,
        4
    ],

    target_names=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ],

    output_dict=True,
    zero_division=0
)


performance = pd.DataFrame({
    "Class": [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ],

    "Precision": [
        report["C0"]["precision"],
        report["C1"]["precision"],
        report["C2"]["precision"],
        report["C3"]["precision"],
        report["C4"]["precision"]
    ],

    "Recall": [
        report["C0"]["recall"],
        report["C1"]["recall"],
        report["C2"]["recall"],
        report["C3"]["recall"],
        report["C4"]["recall"]
    ],

    "F1": [
        report["C0"]["f1-score"],
        report["C1"]["f1-score"],
        report["C2"]["f1-score"],
        report["C3"]["f1-score"],
        report["C4"]["f1-score"]
    ],

    "Support": [
        report["C0"]["support"],
        report["C1"]["support"],
        report["C2"]["support"],
        report["C3"]["support"],
        report["C4"]["support"]
    ]
})


print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print(
    performance.round(4)
)


# %%
# ============================================================
# 36. PROBABILITY ANALYSIS
# ============================================================

proba_df = pd.DataFrame(
    y_proba,
    columns=[
        "P_C0",
        "P_C1",
        "P_C2",
        "P_C3",
        "P_C4"
    ]
)


proba_df[
    "actual_class"
] = y_val.to_numpy()

proba_df[
    "predicted_class"
] = y_pred


mean_proba = (
    proba_df
    .groupby(
        "actual_class"
    )
    [
        [
            "P_C0",
            "P_C1",
            "P_C2",
            "P_C3",
            "P_C4"
        ]
    ]
    .mean()
)


print("\n" + "=" * 70)
print("MEAN PROBABILITY BY ACTUAL CLASS")
print("=" * 70)

print(
    mean_proba.round(4)
)


# %%
# ============================================================
# 37. C4 SEPARATION
# ============================================================

actual = y_val.to_numpy()

c0_mask = (
    actual == 0
)

c4_mask = (
    actual == 4
)


p_c4_actual_c0 = (
    y_proba[
        c0_mask,
        4
    ]
)

p_c4_actual_c4 = (
    y_proba[
        c4_mask,
        4
    ]
)


mean_c0_c4 = (
    p_c4_actual_c0.mean()
)

mean_c4_c4 = (
    p_c4_actual_c4.mean()
)


print("\n" + "=" * 70)
print("C4 PROBABILITY SEPARATION")
print("=" * 70)


print(
    "Mean P(C4) | Actual C0:",
    round(
        mean_c0_c4,
        4
    )
)

print(
    "Mean P(C4) | Actual C4:",
    round(
        mean_c4_c4,
        4
    )
)

print(
    "Difference:",
    round(
        mean_c4_c4
        -
        mean_c0_c4,
        4
    )
)


# %%
# ============================================================
# 38. C4 DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.hist(
    p_c4_actual_c0,
    bins=30,
    alpha=0.6,
    label="Actual C0"
)


plt.hist(
    p_c4_actual_c4,
    bins=30,
    alpha=0.6,
    label="Actual C4"
)


plt.xlabel(
    "Predicted P(C4)"
)

plt.ylabel(
    "Number of Vehicles"
)

plt.title(
    "Vehicle-Aware XGBoost: "
    "P(C4) Distribution"
)

plt.legend()

plt.show()


# %%
# ============================================================
# 39. COMPARE AGAINST PREVIOUS TEMPORAL XGBOOST
# ============================================================
#
# Previous result supplied:
#
# All-readout Temporal XGBoost
#   Accuracy = 0.9556
#   Macro F1 = 0.2041
#   C4 Recall = 0.0263
#
# ============================================================

previous_results = pd.DataFrame({

    "Model": [
        "Always C0",
        "All-Readout Temporal XGBoost",
        "Vehicle-Aware Temporal XGBoost"
    ],

    "Accuracy": [
        0.973048,
        0.9556,
        accuracy
    ],

    "Macro F1": [
        0.197268,
        0.2041,
        macro_f1
    ],

    "C4 Recall": [
        0.0,
        0.0263,
        report[
            "C4"
        ][
            "recall"
        ]
    ]
})


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    previous_results.round(4)
)


# %%
# ============================================================
# 40. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({

    "vehicle_id":
        val_last[
            "vehicle_id"
        ].to_numpy(),

    "actual_class":
        y_val.to_numpy(),

    "predicted_class":
        y_pred,

    "P_C0":
        y_proba[:, 0],

    "P_C1":
        y_proba[:, 1],

    "P_C2":
        y_proba[:, 2],

    "P_C3":
        y_proba[:, 3],

    "P_C4":
        y_proba[:, 4]
})


validation_predictions.to_csv(

    OUTPUT_DIR
    /
    "vehicle_aware_temporal_xgboost_validation.csv",

    index=False
)


print(
    "\n✓ Validation predictions saved."
)


# %%
# ============================================================
# END
# ============================================================