# %%
# ============================================================
# SCANIA COMPONENT X
# TEMPORAL MULTICLASS XGBOOST PIPELINE
#
# Target:
#   C0: > 48 time units before repair
#   C1: 24-48
#   C2: 12-24
#   C3: 6-12
#   C4: 0-6
#
# Features:
#   1. Current operational values
#   2. Rate of change from previous readout
#   3. Vehicle specifications
#
# Evaluation:
#   Accuracy
#   Macro F1
#   Weighted F1
#   Per-class Precision / Recall / F1
#   Confusion Matrix
# ============================================================


# %%
# ============================================================
# 1. IMPORTS
# ============================================================

import xgboost as xgb

print(xgb.__version__)
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
# 4. BASIC DATA CHECKS
# ============================================================

assert train_op["vehicle_id"].nunique() == train_tte["vehicle_id"].nunique()
assert train_op["vehicle_id"].nunique() == train_spec["vehicle_id"].nunique()

print("\nTrain vehicles:",
      train_op["vehicle_id"].nunique())

print("Validation vehicles:",
      val_op["vehicle_id"].nunique())


# %%
# ============================================================
# 5. IDENTIFY RAW OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in ["vehicle_id", "time_step"]
]

print("\nNumber of operational feature columns:",
      len(RAW_FEATURE_COLS))


# %%
# ============================================================
# 6. SORT TIME SERIES
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


# %%
# ============================================================
# 7. TEMPORAL FEATURE ENGINEERING FUNCTION
# ============================================================
#
# For each operational feature x:
#
#   current value:
#       x_t
#
#   previous value:
#       x_(t-1)
#
#   delta:
#       x_t - x_(t-1)
#
#   rate:
#       (x_t - x_(t-1)) /
#       (time_t - time_(t-1))
#
# Because SCANIA sampling is irregular, rate is more meaningful
# than assuming every readout has the same time interval.
# ============================================================

def create_temporal_features(df, feature_cols):

    df = (
        df
        .sort_values(["vehicle_id", "time_step"])
        .copy()
    )

    grouped = df.groupby(
        "vehicle_id",
        sort=False
    )

    # --------------------------------------------------------
    # Time gap
    # --------------------------------------------------------

    df["time_gap"] = (
        grouped["time_step"]
        .diff()
    )

    # Avoid divide-by-zero
    valid_gap = (
        df["time_gap"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    previous = (
        grouped[feature_cols]
        .shift(1)
    )

    previous.columns = [
        f"{col}__prev"
        for col in feature_cols
    ]

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    current_values = df[
        feature_cols
    ]

    delta_values = (
        current_values.to_numpy()
        -
        previous.to_numpy()
    )

    delta = pd.DataFrame(
        delta_values,
        index=df.index,
        columns=[
            f"{col}__delta"
            for col in feature_cols
        ]
    )

    # --------------------------------------------------------
    # Rate of change
    # --------------------------------------------------------

    rate_values = (
        delta_values
        /
        valid_gap.to_numpy()[:, None]
    )

    rate = pd.DataFrame(
        rate_values,
        index=df.index,
        columns=[
            f"{col}__rate"
            for col in feature_cols
        ]
    )

    # Replace infinite values
    rate = rate.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

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
# 8. CREATE TRAIN TEMPORAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN TEMPORAL FEATURES")
print("=" * 70)

train_temporal = create_temporal_features(
    train_op,
    RAW_FEATURE_COLS
)

print(
    "Train temporal shape:",
    train_temporal.shape
)


# %%
# ============================================================
# 9. CREATE VALIDATION TEMPORAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING VALIDATION TEMPORAL FEATURES")
print("=" * 70)

val_temporal = create_temporal_features(
    val_op,
    RAW_FEATURE_COLS
)

print(
    "Validation temporal shape:",
    val_temporal.shape
)


# %%
# ============================================================
# 10. MERGE TRAIN WITH TTE
# ============================================================

train_full = train_temporal.merge(
    train_tte,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)


# %%
# ============================================================
# 11. CALCULATE DISTANCE TO STUDY END / REPAIR
# ============================================================

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    -
    train_full["time_step"]
)


# %%
# ============================================================
# 12. OFFICIAL C0-C4 LABEL FUNCTION
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
# 13. CREATE TRAINING LABELS
# ============================================================

train_full["class_label"] = np.nan


# ------------------------------------------------------------
# Repair vehicles
# ------------------------------------------------------------

repair_mask = (
    train_full["in_study_repair"] == 1
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


# ------------------------------------------------------------
# Censored / non-repair vehicles
#
# If more than 48 time units remain before study end and
# no repair occurred during that period, the observation is
# safely C0 for the 48-unit prediction horizon.
# ------------------------------------------------------------

safe_c0_mask = (
    (train_full["in_study_repair"] == 0)
    &
    (train_full["time_to_endpoint"] > 48)
)

train_full.loc[
    safe_c0_mask,
    "class_label"
] = 0


# %%
# ============================================================
# 14. REMOVE UNCERTAIN / UNLABELLED TRAINING ROWS
# ============================================================

train_model = (
    train_full
    .dropna(subset=["class_label"])
    .copy()
)

train_model["class_label"] = (
    train_model["class_label"]
    .astype(int)
)


# %%
# ============================================================
# 15. CHECK TRAINING CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

train_counts = (
    train_model["class_label"]
    .value_counts()
    .sort_index()
)

train_props = (
    train_model["class_label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nCounts:")
print(train_counts)

print("\nProportions:")
print(train_props)

assert set(
    train_model["class_label"].unique()
) == {0, 1, 2, 3, 4}

print("\n✓ All five classes are present.")


# %%
# ============================================================
# 16. MERGE TRAIN SPECIFICATIONS
# ============================================================

train_model = train_model.merge(
    train_spec,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)


# %%
# ============================================================
# 17. CREATE VALIDATION TABLE
# ============================================================
#
# Validation/test data are already truncated by the organisers.
# We predict only at each vehicle's LAST AVAILABLE READOUT.
#
# Temporal features have been calculated BEFORE selecting the
# last row, so the final row contains information from its
# previous observation.
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

val_last = val_last.merge(
    val_spec,
    on="vehicle_id",
    how="left",
    validate="one_to_one"
)

val_last = val_last.merge(
    val_labels,
    on="vehicle_id",
    how="left",
    validate="one_to_one"
)

print("\nValidation rows:",
      len(val_last))

print("Validation vehicles:",
      val_last["vehicle_id"].nunique())

assert len(val_last) == val_last["vehicle_id"].nunique()


# %%
# ============================================================
# 18. CHECK VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 70)

print(
    val_last["class_label"]
    .value_counts()
    .sort_index()
)


# %%
# ============================================================
# 19. DEFINE FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",

    # Outcome information - NEVER model inputs
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint"
]


X_train = train_model.drop(
    columns=NON_FEATURE_COLS,
    errors="ignore"
)

y_train = (
    train_model["class_label"]
    .copy()
)


X_val = val_last.drop(
    columns=[
        "vehicle_id",
        "class_label"
    ],
    errors="ignore"
)

y_val = (
    val_last["class_label"]
    .astype(int)
    .copy()
)


# %%
# ============================================================
# 20. ALIGN TRAIN / VALIDATION FEATURES
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

print("\nTrain-only columns:",
      train_only)

print("Validation-only columns:",
      val_only)

assert len(train_only) == 0
assert len(val_only) == 0

X_val = X_val[
    X_train.columns
]

assert list(X_train.columns) == list(X_val.columns)

print("✓ Train/validation schemas aligned.")


# %%
# ============================================================
# 21. IDENTIFY CATEGORICAL / NUMERIC FEATURES
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

print("\nNumeric features:",
      len(numeric_cols))

print("Categorical features:",
      len(categorical_cols))

print("Total features:",
      len(X_train.columns))


# %%
# ============================================================
# 22. DUMMY BASELINE
# ============================================================

dummy_pred = np.zeros(
    len(y_val),
    dtype=int
)

dummy_accuracy = accuracy_score(
    y_val,
    dummy_pred
)

dummy_macro_f1 = f1_score(
    y_val,
    dummy_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("DUMMY BASELINE")
print("=" * 70)

print(
    "Accuracy:",
    round(dummy_accuracy, 4)
)

print(
    "Macro F1:",
    round(dummy_macro_f1, 4)
)


# %%
# ============================================================
# 23. PREPROCESSING
# ============================================================
#
# XGBoost can handle NaN numerical values natively.
#
# We still one-hot encode categorical specifications.
#
# Numerical imputation is kept here for a consistent sklearn
# pipeline and to avoid unexpected issues.
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
# 24. CALCULATE CLASS WEIGHTS
# ============================================================
#
# We do NOT undersample here.
#
# Instead, every class receives a balancing weight.
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
    .map(class_weight_map)
    .to_numpy()
)


# %%
# ============================================================
# 25. DEFINE MULTICLASS XGBOOST
# ============================================================

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=5,

    n_estimators=400,
    learning_rate=0.05,

    max_depth=6,
    min_child_weight=3,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    eval_metric="mlogloss",

    random_state=RANDOM_STATE,
    n_jobs=-1,

    tree_method="hist"
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
# 26. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING TEMPORAL XGBOOST")
print("=" * 70)

xgb_pipeline.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

print("✓ XGBoost training complete")


# %%
# ============================================================
# 27. PREDICT VALIDATION
# ============================================================

y_pred_xgb = (
    xgb_pipeline.predict(
        X_val
    )
    .astype(int)
)

y_proba_xgb = (
    xgb_pipeline.predict_proba(
        X_val
    )
)


print(
    "\nProbability matrix shape:",
    y_proba_xgb.shape
)

assert y_proba_xgb.shape == (
    len(X_val),
    5
)


# %%
# ============================================================
# 28. PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

pred_distribution = (
    pd.Series(
        y_pred_xgb,
        name="predicted_class"
    )
    .value_counts()
    .sort_index()
)

print(pred_distribution)


# %%
# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL XGBOOST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_pred_xgb,

        labels=[
            0, 1, 2, 3, 4
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
# 30. MAIN METRICS
# ============================================================

xgb_accuracy = accuracy_score(
    y_val,
    y_pred_xgb
)

xgb_macro_f1 = f1_score(
    y_val,
    y_pred_xgb,
    average="macro"
)

xgb_weighted_f1 = f1_score(
    y_val,
    y_pred_xgb,
    average="weighted"
)

print(
    "Temporal XGBoost Accuracy:",
    round(xgb_accuracy, 4)
)

print(
    "Temporal XGBoost Macro F1:",
    round(xgb_macro_f1, 4)
)

print(
    "Temporal XGBoost Weighted F1:",
    round(xgb_weighted_f1, 4)
)


# %%
# ============================================================
# 31. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred_xgb,
    labels=[
        0, 1, 2, 3, 4
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
    "Temporal XGBoost "
    "— Validation Confusion Matrix"
)

plt.show()


# %%
# ============================================================
# 32. PROBABILITY ANALYSIS
# ============================================================

proba_df = pd.DataFrame(
    y_proba_xgb,
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
] = y_pred_xgb


mean_probability_by_actual = (
    proba_df
    .groupby("actual_class")
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
print("MEAN PREDICTED PROBABILITY BY ACTUAL CLASS")
print("=" * 70)

print(
    mean_probability_by_actual
)


# %%
# ============================================================
# 33. C4 DISCRIMINATION
# ============================================================

actual_array = (
    y_val.to_numpy()
)

actual_c0_mask = (
    actual_array == 0
)

actual_c4_mask = (
    actual_array == 4
)

p_c4_actual_c0 = (
    y_proba_xgb[
        actual_c0_mask,
        4
    ]
)

p_c4_actual_c4 = (
    y_proba_xgb[
        actual_c4_mask,
        4
    ]
)

print("\n" + "=" * 70)
print("C4 PROBABILITY SEPARATION")
print("=" * 70)

print(
    "Mean P(C4) for actual C0:",
    round(
        p_c4_actual_c0.mean(),
        4
    )
)

print(
    "Mean P(C4) for actual C4:",
    round(
        p_c4_actual_c4.mean(),
        4
    )
)

print(
    "Difference:",
    round(
        p_c4_actual_c4.mean()
        -
        p_c4_actual_c0.mean(),
        4
    )
)


# %%
# ============================================================
# 34. PLOT C4 PROBABILITY DISTRIBUTION
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
    "Predicted Probability of C4"
)

plt.ylabel(
    "Number of Vehicles"
)

plt.title(
    "Temporal XGBoost: "
    "P(C4) for Actual C0 vs Actual C4"
)

plt.legend()

plt.show()


# %%
# ============================================================
# 35. PER-CLASS RECALL SUMMARY
# ============================================================

report = classification_report(
    y_val,
    y_pred_xgb,

    labels=[
        0, 1, 2, 3, 4
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

recall_summary = pd.DataFrame({
    "Class": [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
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
    ]
})

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print(
    recall_summary.round(4)
)


# %%
# ============================================================
# 36. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Model": [
        "Always C0",
        "Temporal XGBoost"
    ],

    "Accuracy": [
        dummy_accuracy,
        xgb_accuracy
    ],

    "Macro F1": [
        dummy_macro_f1,
        xgb_macro_f1
    ]
})

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(
    summary.round(4)
)


# %%
# ============================================================
# 37. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({
    "vehicle_id":
        val_last["vehicle_id"].to_numpy(),

    "actual_class":
        y_val.to_numpy(),

    "predicted_class":
        y_pred_xgb,

    "P_C0":
        y_proba_xgb[:, 0],

    "P_C1":
        y_proba_xgb[:, 1],

    "P_C2":
        y_proba_xgb[:, 2],

    "P_C3":
        y_proba_xgb[:, 3],

    "P_C4":
        y_proba_xgb[:, 4]
})

OUTPUT_DIR = Path("../data/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

validation_predictions.to_csv(
    OUTPUT_DIR /
    "temporal_xgboost_validation_predictions.csv",

    index=False
)

print(
    "\n✓ Validation predictions saved."
)


# %%
# ============================================================
# END
# ============================================================