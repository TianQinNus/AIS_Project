#%%
from pathlib import Path
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 150)
pd.set_option("display.max_rows", 100)

RANDOM_STATE = 42

#LOAD DATA
DATA_DIR = Path("../data/raw")

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"

TEST_OP_PATH = DATA_DIR / "test_operational_readouts.csv"
TEST_SPEC_PATH = DATA_DIR / "test_specifications.csv"

train_op = pd.read_csv(TRAIN_OP_PATH)
train_spec = pd.read_csv(TRAIN_SPEC_PATH)
train_tte = pd.read_csv(TRAIN_TTE_PATH)

val_op = pd.read_csv(VAL_OP_PATH)
val_spec = pd.read_csv(VAL_SPEC_PATH)
val_labels = pd.read_csv(VAL_LABEL_PATH)

print("TRAIN")
print("Operational:", train_op.shape)
print("Specifications:", train_spec.shape)
print("TTE:", train_tte.shape)

print("\nVALIDATION")
print("Operational:", val_op.shape)
print("Specifications:", val_spec.shape)
print("Labels:", val_labels.shape)

display(train_op.head())
display(train_spec.head())
display(train_tte.head())

print("Train operational vehicles:",
      train_op["vehicle_id"].nunique())

print("Train specification vehicles:",
      train_spec["vehicle_id"].nunique())

print("Train TTE vehicles:",
      train_tte["vehicle_id"].nunique())

print("Validation vehicles:",
      val_op["vehicle_id"].nunique())

op_ids = set(train_op["vehicle_id"])
spec_ids = set(train_spec["vehicle_id"])
tte_ids = set(train_tte["vehicle_id"])

print("Operational but not specifications:",
      len(op_ids - spec_ids))

print("Operational but not TTE:",
      len(op_ids - tte_ids))

print("TTE but not operational:",
      len(tte_ids - op_ids))

#%%
# CHECK DISTRIBUTION OF READOUTS PER VEHICLE    
readouts_per_vehicle = (
    train_op
    .groupby("vehicle_id")
    .size()
)

display(readouts_per_vehicle.describe())
# %%

# PLOT DISTRIBUTION OF READOUTS PER VEHICLE
plt.figure(figsize=(8, 5))

plt.hist(
    readouts_per_vehicle,
    bins=50
)

plt.xlabel("Number of readouts")
plt.ylabel("Number of vehicles")
plt.title("Readouts per Vehicle")

plt.show()

# %%

train_op = (
    train_op
    .sort_values(["vehicle_id", "time_step"])
    .reset_index(drop=True)
)
train_op["time_gap"] = (
    train_op
    .groupby("vehicle_id")["time_step"]
    .diff()
)
display(
    train_op["time_gap"].describe()
)
plt.figure(figsize=(8, 5))

train_op["time_gap"].dropna().clip(upper=20).hist(
    bins=50
)

plt.xlabel("Time gap between consecutive readouts")
plt.ylabel("Count")
plt.title("Readout Sampling Gaps")

plt.show()
# %%
ignore_cols = {
    "vehicle_id",
    "time_step",
    "time_gap"
}

feature_cols = [
    c for c in train_op.columns
    if c not in ignore_cols
]
variable_groups = {}

for col in feature_cols:
    variable_id = col.split("_")[0]

    if variable_id not in variable_groups:
        variable_groups[variable_id] = []

    variable_groups[variable_id].append(col)

print("Operational columns:", len(feature_cols))
print("Underlying variables:", len(variable_groups))

for variable_id, columns in variable_groups.items():
    print(
        f"Variable {variable_id}: "
        f"{len(columns)} column(s)"
    )

missing_summary = (
    train_op[feature_cols]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .to_frame("missing_rate")
)

display(missing_summary.head(30))    
plt.figure(figsize=(8, 8))

missing_summary.head(30).sort_values(
    "missing_rate"
).plot(
    kind="barh",
    legend=False
)

plt.xlabel("Missing proportion")
plt.title("Top 30 Features by Missingness")

plt.show()
# %%
display(train_spec.head())
spec_cols = [
    c for c in train_spec.columns
    if c != "vehicle_id"
]
for col in spec_cols:
    print(f"\n{col}")
    print(
        train_spec[col]
        .value_counts(dropna=False)
    )
display(train_tte.head())
repair_counts = (
    train_tte["in_study_repair"]
    .value_counts()
    .sort_index()
)

print(repair_counts)
repair_counts.plot(
    kind="bar",
    figsize=(6, 4)
)

plt.xlabel("In-study repair")
plt.ylabel("Number of vehicles")
plt.title("Component X Repair Distribution")

plt.show()
# %%

repair_vehicle_ids = train_tte.loc[
    train_tte["in_study_repair"] == 1,
    "vehicle_id"
]

example_vehicle_id = repair_vehicle_ids.iloc[0]

print("Example repair vehicle:",
      example_vehicle_id)

example_tte = train_tte.loc[
    train_tte["vehicle_id"] == example_vehicle_id
]

display(example_tte)

example_history = train_op.loc[
    train_op["vehicle_id"] == example_vehicle_id
].copy()

display(
    example_history[
        ["vehicle_id", "time_step"]
    ].tail(20)
)

repair_time = example_tte[
    "length_of_study_time_step"
].iloc[0]

print("Repair time:", repair_time)

example_history["tte"] = (
    repair_time
    -
    example_history["time_step"]
)

display(
    example_history[
        ["time_step", "tte"]
    ].tail(30)
)
# %%
def tte_to_class(tte):
    """
    IDA 2024 Component X repair-proximity classes.

    Class 1: 48 to 24 time units before repair
    Class 2: 24 to 12
    Class 3: 12 to 6
    Class 4: 6 to 0
    Class 0: more than 48 time units before repair
    """

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

test_ttes = [
    100,
    49,
    48,
    30,
    24,
    20,
    12,
    10,
    6,
    3,
    0
]

for tte in test_ttes:
    print(
        tte,
        "→",
        tte_to_class(tte)
    )
# %%
example_history["class_label"] = (
    example_history["tte"]
    .apply(tte_to_class)
)
display(
    example_history[
        [
            "time_step",
            "tte",
            "class_label"
        ]
    ].tail(40)
)

plt.figure(figsize=(10, 4))

plt.step(
    example_history["time_step"],
    example_history["class_label"],
    where="post"
)

plt.axvline(
    repair_time,
    linestyle="--",
    label="Component X repair"
)

plt.yticks(
    [0, 1, 2, 3, 4],
    ["C0", "C1", "C2", "C3", "C4"]
)

plt.xlabel("Time step")
plt.ylabel("Repair-proximity class")

plt.title(
    f"Vehicle {example_vehicle_id}: "
    "Repair-Proximity Progression"
)

plt.legend()

plt.show()
# %%
display(val_labels.head())

val_class_counts = (
    val_labels["class_label"]
    .value_counts()
    .sort_index()
)

print(val_class_counts)

val_class_counts.plot(
    kind="bar",
    figsize=(7, 4)
)

plt.xlabel("Class")
plt.ylabel("Number of vehicles")
plt.title("Validation Class Distribution")

plt.show()

# %%
val_last = (
    val_op
    .sort_values(
        ["vehicle_id", "time_step"]
    )
    .groupby("vehicle_id")
    .tail(1)
    .copy()
)
print(
    "Rows:",
    len(val_last)
)

print(
    "Vehicles:",
    val_last["vehicle_id"].nunique()
)
# %%
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
display(val_last.head())

# %%
# %%
# ============================================================
# BUILD TRAINING DATASET
# ============================================================

# Join each operational readout with the vehicle-level
# repair / study-end information.
train_full = train_op.merge(
    train_tte,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)

print("train_full shape:", train_full.shape)

display(
    train_full[
        [
            "vehicle_id",
            "time_step",
            "length_of_study_time_step",
            "in_study_repair"
        ]
    ].head()
)

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    - train_full["time_step"]
)

display(
    train_full[
        [
            "vehicle_id",
            "time_step",
            "length_of_study_time_step",
            "in_study_repair",
            "time_to_endpoint"
        ]
    ].head()
)# %%
train_full["class_label"] = np.nan
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

# %%
repair_class_counts = (
    train_full.loc[
        repair_mask,
        "class_label"
    ]
    .value_counts()
    .sort_index()
)

print("Repair-vehicle observation labels:")
print(repair_class_counts)
# %%
# %%
non_repair_mask = (
    train_full["in_study_repair"] == 0
)

safe_c0_mask = (
    non_repair_mask
    &
    (train_full["time_to_endpoint"] > 48)
)

train_full.loc[
    safe_c0_mask,
    "class_label"
] = 0
# %%
# %%
uncertain_censored_mask = (
    non_repair_mask
    &
    (train_full["time_to_endpoint"] <= 48)
)

print(
    "Uncertain censored observations:",
    uncertain_censored_mask.sum()
)
# %%
# %%
print("\nAll label counts before removing uncertain rows:")

print(
    train_full["class_label"]
    .value_counts(
        dropna=False
    )
    .sort_index()
)
# %%
# %%
labelled_train = (
    train_full
    .dropna(
        subset=["class_label"]
    )
    .copy()
)

labelled_train["class_label"] = (
    labelled_train["class_label"]
    .astype(int)
)# %%
train_class_counts = (
    labelled_train["class_label"]
    .value_counts()
    .sort_index()
)

train_class_proportions = (
    labelled_train["class_label"]
    .value_counts(
        normalize=True
    )
    .sort_index()
)

print("TRAINING CLASS COUNTS")
print(train_class_counts)

print("\nTRAINING CLASS PROPORTIONS")
print(train_class_proportions)
# %%
# %%
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

train_class_proportions.plot(
    kind="bar",
    ax=axes[0]
)

axes[0].set_title(
    "Training Class Distribution"
)

axes[0].set_xlabel("Class")
axes[0].set_ylabel("Proportion")


val_class_proportions = (
    val_labels["class_label"]
    .value_counts(
        normalize=True
    )
    .sort_index()
)

val_class_proportions.plot(
    kind="bar",
    ax=axes[1]
)

axes[1].set_title(
    "Validation Class Distribution"
)

axes[1].set_xlabel("Class")
axes[1].set_ylabel("Proportion")

plt.tight_layout()
plt.show()
# %%
# %%
train_model = labelled_train.merge(
    train_spec,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)

print(
    "Training model shape:",
    train_model.shape
)

display(train_model.head())
NON_FEATURE_COLS = [
    "vehicle_id",

    # Target
    "class_label",

    # Future/outcome information — MUST NOT enter model
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",

    # EDA-created variable; not used in baseline
    "time_gap"
]

X_train = train_model.drop(
    columns=NON_FEATURE_COLS,
    errors="ignore"
)

y_train = (
    train_model["class_label"]
    .copy()
)

print(y_train.value_counts().sort_index())
# %%
X_val = val_last.drop(
    columns=[
        "vehicle_id",
        "class_label"
    ],
    errors="ignore"
)

y_val = (
    val_last["class_label"]
    .copy()
)

print(y_val.value_counts().sort_index())
# %%
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
    "Train-only columns:",
    train_only
)

print(
    "Validation-only columns:",
    val_only
)
# %%
# %%
assert len(train_only) == 0
assert len(val_only) == 0

# Force same column order
X_val = X_val[X_train.columns]

assert (
    list(X_train.columns)
    ==
    list(X_val.columns)
)

print(
    "✓ Train and validation "
    "features are aligned."
)# %%
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
    "Numerical features:",
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


# %%
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# %%
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

print(
    "Dummy Accuracy:",
    dummy_accuracy
)

print(
    "Dummy Macro F1:",
    dummy_macro_f1
)
# %%
# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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

rf_baseline = RandomForestClassifier(
    n_estimators=200,

    # Handle severe class imbalance
    class_weight="balanced",

    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        rf_baseline
    )
])
print("Training Random Forest...")

rf_pipeline.fit(
    X_train,
    y_train
)

print("✓ Training complete")

y_pred_rf = rf_pipeline.predict(
    X_val
)

print("Predicted class distribution:")

print(
    pd.Series(y_pred_rf)
    .value_counts()
    .sort_index()

)

print(
    classification_report(
        y_val,
        y_pred_rf,

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
y_proba_rf = rf_pipeline.predict_proba(X_val)

proba_df = pd.DataFrame(
    y_proba_rf,
    columns=[
        "P_C0",
        "P_C1",
        "P_C2",
        "P_C3",
        "P_C4"
    ]
)

display(proba_df.describe())

print("y_train classes:")
print(np.unique(y_train))

print("\ny_train class counts:")
print(y_train.value_counts().sort_index())

print("\nRF learned classes:")
print(rf_pipeline.named_steps["classifier"].classes_)

print("\npredict_proba shape:")
print(rf_pipeline.predict_proba(X_val).shape)
# %%
# Analyse predicted probabilities by actual class

proba_df = pd.DataFrame(
    y_proba_rf,
    columns=["P_C0", "P_C1", "P_C2", "P_C3", "P_C4"]
)

proba_df["actual_class"] = y_val.to_numpy()

mean_proba_by_actual = (
    proba_df
    .groupby("actual_class")
    [["P_C0", "P_C1", "P_C2", "P_C3", "P_C4"]]
    .mean()
)

display(mean_proba_by_actual)
# %%
# %%
# ============================================================
# EXPERIMENT 2
# RANDOM FOREST WITH MAJORITY-CLASS UNDERSAMPLING
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. CHECK ORIGINAL TRAINING DISTRIBUTION
# ------------------------------------------------------------

print("=" * 60)
print("ORIGINAL TRAINING CLASS DISTRIBUTION")
print("=" * 60)

original_counts = (
    train_model["class_label"]
    .value_counts()
    .sort_index()
)

original_proportions = (
    train_model["class_label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nCounts:")
print(original_counts)

print("\nProportions:")
print(original_proportions)


# ------------------------------------------------------------
# 2. SPLIT TRAINING DATA BY CLASS
# ------------------------------------------------------------

train_c0 = train_model[
    train_model["class_label"] == 0
].copy()

train_c1 = train_model[
    train_model["class_label"] == 1
].copy()

train_c2 = train_model[
    train_model["class_label"] == 2
].copy()

train_c3 = train_model[
    train_model["class_label"] == 3
].copy()

train_c4 = train_model[
    train_model["class_label"] == 4
].copy()

print("\nIndividual class sizes:")

print("C0:", len(train_c0))
print("C1:", len(train_c1))
print("C2:", len(train_c2))
print("C3:", len(train_c3))
print("C4:", len(train_c4))


# ------------------------------------------------------------
# 3. UNDERSAMPLE C0
# ------------------------------------------------------------
#
# C0 currently contains almost 900k observations.
#
# For this experiment, retain 50,000 randomly selected C0
# observations while keeping ALL minority-class observations.
#
# This is an experimental sampling strategy, not the final
# modelling decision.
# ------------------------------------------------------------

N_C0_SAMPLE = 50_000

if len(train_c0) < N_C0_SAMPLE:
    raise ValueError(
        f"C0 contains only {len(train_c0)} rows, "
        f"which is smaller than requested sample size "
        f"{N_C0_SAMPLE}."
    )

train_c0_sampled = train_c0.sample(
    n=N_C0_SAMPLE,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 4. COMBINE SAMPLED C0 WITH ALL C1-C4
# ------------------------------------------------------------

train_sampled = pd.concat(
    [
        train_c0_sampled,
        train_c1,
        train_c2,
        train_c3,
        train_c4
    ],
    ignore_index=True
)

# Shuffle rows
train_sampled = (
    train_sampled
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 5. CHECK NEW CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("UNDERSAMPLED TRAINING CLASS DISTRIBUTION")
print("=" * 60)

sampled_counts = (
    train_sampled["class_label"]
    .value_counts()
    .sort_index()
)

sampled_proportions = (
    train_sampled["class_label"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nCounts:")
print(sampled_counts)

print("\nProportions:")
print(sampled_proportions)


# ------------------------------------------------------------
# 6. CREATE X AND y
# ------------------------------------------------------------

X_train_sampled = train_sampled.drop(
    columns=NON_FEATURE_COLS,
    errors="ignore"
)

y_train_sampled = (
    train_sampled["class_label"]
    .astype(int)
    .copy()
)


# Force exactly the same feature columns/order
# as the original training dataset.

X_train_sampled = X_train_sampled[
    X_train.columns
]


# ------------------------------------------------------------
# 7. SANITY CHECKS
# ------------------------------------------------------------

assert set(
    y_train_sampled.unique()
) == {0, 1, 2, 3, 4}

assert list(
    X_train_sampled.columns
) == list(
    X_val.columns
)

print("\n✓ All five classes are present.")
print("✓ Training and validation schemas match.")

print(
    "\nSampled training shape:",
    X_train_sampled.shape
)

print(
    "Validation shape:",
    X_val.shape
)


# ------------------------------------------------------------
# 8. TRAIN RANDOM FOREST
# ------------------------------------------------------------
#
# IMPORTANT:
#
# We intentionally DO NOT use:
#
#     class_weight="balanced"
#
# in this experiment.
#
# We want to isolate the effect of C0 undersampling.
# ------------------------------------------------------------

rf_sampled = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_sampled_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        rf_sampled
    )
])


print("\n" + "=" * 60)
print("TRAINING UNDERSAMPLED RANDOM FOREST")
print("=" * 60)

rf_sampled_pipeline.fit(
    X_train_sampled,
    y_train_sampled
)

print("✓ Training complete")


# ------------------------------------------------------------
# 9. CHECK WHICH CLASSES RF LEARNED
# ------------------------------------------------------------

learned_classes = (
    rf_sampled_pipeline
    .named_steps["classifier"]
    .classes_
)

print(
    "\nRF learned classes:",
    learned_classes
)

assert len(learned_classes) == 5


# ------------------------------------------------------------
# 10. PREDICT VALIDATION DATA
# ------------------------------------------------------------

y_pred_sampled = (
    rf_sampled_pipeline.predict(
        X_val
    )
)


# ------------------------------------------------------------
# 11. CHECK PREDICTED CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 60)

predicted_distribution = (
    pd.Series(
        y_pred_sampled,
        name="predicted_class"
    )
    .value_counts()
    .sort_index()
)

print(predicted_distribution)


# ------------------------------------------------------------
# 12. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_val,
        y_pred_sampled,
        labels=[0, 1, 2, 3, 4],
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


# ------------------------------------------------------------
# 13. MAIN METRICS
# ------------------------------------------------------------

# Calculate metrics for the previous balanced Random Forest
rf_accuracy = accuracy_score(
    y_val,
    y_pred_rf
)

rf_macro_f1 = f1_score(
    y_val,
    y_pred_rf,
    average="macro"
)
sampled_accuracy = accuracy_score(
    y_val,
    y_pred_sampled
)

sampled_macro_f1 = f1_score(
    y_val,
    y_pred_sampled,
    average="macro"
)

sampled_weighted_f1 = f1_score(
    y_val,
    y_pred_sampled,
    average="weighted"
)

print(
    "Undersampled RF Accuracy:",
    round(sampled_accuracy, 4)
)

print(
    "Undersampled RF Macro F1:",
    round(sampled_macro_f1, 4)
)

print(
    "Undersampled RF Weighted F1:",
    round(sampled_weighted_f1, 4)
)


# ------------------------------------------------------------
# 14. CONFUSION MATRIX
# ------------------------------------------------------------

cm_sampled = confusion_matrix(
    y_val,
    y_pred_sampled,
    labels=[0, 1, 2, 3, 4]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_sampled,
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
    "Undersampled Random Forest "
    "— Validation Confusion Matrix"
)

plt.show()


# ------------------------------------------------------------
# 15. GET CLASS PROBABILITIES
# ------------------------------------------------------------

y_proba_sampled = (
    rf_sampled_pipeline.predict_proba(
        X_val
    )
)

print(
    "\nProbability matrix shape:",
    y_proba_sampled.shape
)

assert y_proba_sampled.shape == (
    len(X_val),
    5
)


# ------------------------------------------------------------
# 16. CREATE PROBABILITY DATAFRAME
# ------------------------------------------------------------

proba_sampled_df = pd.DataFrame(
    y_proba_sampled,
    columns=[
        "P_C0",
        "P_C1",
        "P_C2",
        "P_C3",
        "P_C4"
    ]
)

proba_sampled_df[
    "actual_class"
] = y_val.to_numpy()

proba_sampled_df[
    "predicted_class"
] = y_pred_sampled


# ------------------------------------------------------------
# 17. MEAN PREDICTED PROBABILITY BY ACTUAL CLASS
# ------------------------------------------------------------

mean_proba_by_actual_sampled = (
    proba_sampled_df
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

print("\n" + "=" * 60)
print("MEAN PREDICTED PROBABILITY BY ACTUAL CLASS")
print("=" * 60)

display(
    mean_proba_by_actual_sampled
)


# ------------------------------------------------------------
# 18. SPECIFICALLY COMPARE P(C4)
#     FOR ACTUAL C0 VS ACTUAL C4
# ------------------------------------------------------------

actual_array = y_val.to_numpy()

actual_c0_mask = (
    actual_array == 0
)

actual_c4_mask = (
    actual_array == 4
)


c0_p_c4 = (
    y_proba_sampled[
        actual_c0_mask,
        4
    ]
)

c4_p_c4 = (
    y_proba_sampled[
        actual_c4_mask,
        4
    ]
)


print("\n" + "=" * 60)
print("P(C4) SEPARATION")
print("=" * 60)

print(
    "Mean P(C4) for actual C0:",
    round(
        c0_p_c4.mean(),
        4
    )
)

print(
    "Mean P(C4) for actual C4:",
    round(
        c4_p_c4.mean(),
        4
    )
)

print(
    "Difference:",
    round(
        c4_p_c4.mean()
        -
        c0_p_c4.mean(),
        4
    )
)


# ------------------------------------------------------------
# 19. PLOT P(C4) DISTRIBUTION
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    c0_p_c4,
    bins=30,
    alpha=0.6,
    label="Actual C0"
)

plt.hist(
    c4_p_c4,
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
    "P(C4) Distribution: "
    "Actual C0 vs Actual C4"
)

plt.legend()

plt.show()


# ------------------------------------------------------------
# 20. COMPARE WITH PREVIOUS BASELINES
# ------------------------------------------------------------

results_comparison = pd.DataFrame({
    "Model": [
        "Always C0",
        "Balanced Random Forest",
        "Undersampled Random Forest"
    ],

    "Accuracy": [
        dummy_accuracy,
        rf_accuracy,
        sampled_accuracy
    ],

    "Macro F1": [
        dummy_macro_f1,
        rf_macro_f1,
        sampled_macro_f1
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

display(
    results_comparison
)


# ------------------------------------------------------------
# 21. SHOW EXAMPLES OF TRUE C4 VEHICLES
# ------------------------------------------------------------

c4_examples = (
    proba_sampled_df[
        proba_sampled_df[
            "actual_class"
        ] == 4
    ]
    .sort_values(
        "P_C4",
        ascending=False
    )
)

print(
    "\nHighest P(C4) "
    "among actual C4 vehicles:"
)

display(
    c4_examples.head(20)
)


# ------------------------------------------------------------
# END OF EXPERIMENT 2
# ------------------------------------------------------------
# %%
