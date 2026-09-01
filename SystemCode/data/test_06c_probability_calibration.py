# ============================================================
# FleetCare AI
# Model 6: Vehicle-Level Internal Dev Split
#          + Probability Calibration
#          + Cost-Sensitive Decision Re-test
# ============================================================
#
# Background from Model 5:
#   Direct MEC (Minimum Expected Cost) on the raw probabilities
#   from the weighted, vehicle-aware XGBoost model was too
#   aggressive and predicted almost every validation vehicle as C4.
#
# Main hypothesis:
#   Vehicle-aware sampling + class weighting distort the raw class
#   probabilities. Before using those probabilities in a business
#   cost decision, we should calibrate them on a held-out,
#   vehicle-level development set that is NOT class-balanced.
#
# Test 06 therefore:
#
#   1) Split OFFICIAL TRAINING VEHICLES into:
#        internal train / internal dev
#      at VEHICLE level, stratified by in_study_repair.
#
#   2) Train the current reference architecture only on
#      internal-train vehicles:
#        Vehicle-aware sampling
#        + Current/Previous/Delta/Rate temporal features
#        + Specifications
#        + XGBoost
#        + W1 Light class weights
#
#   3) Build an INTERNAL DEV calibration set without class-aware
#      down/up-sampling.
#
#      All safely-labelled dev observations are retained, but each
#      vehicle receives total weight = 1. This avoids long vehicle
#      histories dominating calibration while still preserving the
#      natural temporal occupancy of C0-C4 within each vehicle.
#
#   4) Compare three probability versions:
#        A. Raw
#        B. Prior-corrected
#        C. Multinomial logit calibrated
#
#   5) Select calibration method ONLY by INTERNAL DEV log loss.
#
#   6) Evaluate on official validation:
#        - probability calibration quality
#        - Argmax decision
#        - Minimum Expected Cost (MEC) decision
#        - C4 Recall / Precision / F1
#        - Macro F1
#        - Official business cost
#
# IMPORTANT:
#   Official validation has already been examined in Models 0-5,
#   so it is no longer a truly untouched test set. Test 06 improves
#   the protocol going forward by moving calibration/model-selection
#   decisions to an internal vehicle-level dev set.
# ============================================================

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DEV_SIZE = 0.20

NUM_CLASSES = 5
CLASS_IDS = np.arange(NUM_CLASSES)
CLASS_NAMES = ["C0", "C1", "C2", "C3", "C4"]

EPS = 1e-12
ECE_BINS = 15

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 300)


# ============================================================
# 2. PATHS
# ============================================================
#
# Expected structure if this script lives in SystemCode/data/:
#
# data/
#   test_06_probability_calibration.py
#   raw/
#   processed/
#
# Fallbacks are included for slightly different repo layouts.
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

print("=" * 100)
print("MODEL 6 — PROBABILITY CALIBRATION")
print("=" * 100)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

print("\nChecking required files:")
for path in required_files:
    print(f"  {path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Current DATA_DIR = {DATA_DIR}\n"
            "Please keep this script in the same project data folder "
            "as test_00-test_05, or update DATA_DIR."
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

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
# 6. VEHICLE-LEVEL INTERNAL TRAIN / DEV SPLIT
# ============================================================
#
# Split by vehicle_id, not observation row.
# Stratification uses the vehicle-level repair indicator so that
# the proportion of repair / non-repair vehicles is similar in
# internal train and internal dev.
# ============================================================

vehicle_table = (
    train_tte[
        [
            "vehicle_id",
            "in_study_repair",
        ]
    ]
    .drop_duplicates("vehicle_id")
    .copy()
)

internal_train_ids, internal_dev_ids = train_test_split(
    vehicle_table["vehicle_id"].to_numpy(),
    test_size=DEV_SIZE,
    random_state=RANDOM_STATE,
    stratify=vehicle_table["in_study_repair"].to_numpy(),
)

internal_train_ids = set(internal_train_ids)
internal_dev_ids = set(internal_dev_ids)

assert internal_train_ids.isdisjoint(internal_dev_ids)
assert len(internal_train_ids) + len(internal_dev_ids) == train_vehicle_count

print("\n" + "=" * 100)
print("INTERNAL VEHICLE SPLIT")
print("=" * 100)

print("Internal train vehicles:", len(internal_train_ids))
print("Internal dev vehicles:", len(internal_dev_ids))

split_summary = (
    vehicle_table
    .assign(
        split=np.where(
            vehicle_table["vehicle_id"].isin(internal_dev_ids),
            "dev",
            "train",
        )
    )
    .groupby("split")["in_study_repair"]
    .agg(["count", "mean"])
)

print("\nVehicle-level repair proportion:")
print(split_summary)


# ============================================================
# 7. IDENTIFY RAW OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in [
        "vehicle_id",
        "time_step",
    ]
]

print("\nRaw operational features:", len(RAW_FEATURE_COLS))


# ============================================================
# 8. SORT TIME SERIES
# ============================================================

train_op = (
    train_op
    .sort_values(
        [
            "vehicle_id",
            "time_step",
        ]
    )
    .reset_index(drop=True)
)

val_op = (
    val_op
    .sort_values(
        [
            "vehicle_id",
            "time_step",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 9. TEMPORAL FEATURE ENGINEERING
# ============================================================
#
# Same feature logic as Models 1-5:
#
# current:
#     x(t)
#
# previous:
#     x(t-1)
#
# delta:
#     x(t) - x(t-1)
#
# rate:
#     [x(t) - x(t-1)] / [t - t_previous]
#
# Rate is useful because SCANIA readouts are irregularly sampled.
# ============================================================

def create_temporal_features(df, feature_cols):
    df = (
        df
        .sort_values(
            [
                "vehicle_id",
                "time_step",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    grouped = df.groupby(
        "vehicle_id",
        sort=False,
    )

    # Time gap
    df["time_gap"] = (
        grouped["time_step"]
        .diff()
    )

    safe_time_gap = (
        df["time_gap"]
        .replace(0, np.nan)
    )

    # Previous values
    previous = (
        grouped[feature_cols]
        .shift(1)
    )

    previous.columns = [
        f"{col}__prev"
        for col in feature_cols
    ]

    # Arrays for efficient arithmetic
    current_array = (
        df[feature_cols]
        .to_numpy(dtype=np.float32)
    )

    previous_array = (
        previous
        .to_numpy(dtype=np.float32)
    )

    # Delta
    delta_array = (
        current_array
        -
        previous_array
    )

    delta = pd.DataFrame(
        delta_array,
        index=df.index,
        columns=[
            f"{col}__delta"
            for col in feature_cols
        ],
    )

    # Rate
    time_gap_array = (
        safe_time_gap
        .to_numpy(dtype=np.float32)
        [:, None]
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        rate_array = (
            delta_array
            /
            time_gap_array
        )

    rate = pd.DataFrame(
        rate_array,
        index=df.index,
        columns=[
            f"{col}__rate"
            for col in feature_cols
        ],
    )

    rate = rate.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return pd.concat(
        [
            df,
            previous,
            delta,
            rate,
        ],
        axis=1,
    )


print("\n" + "=" * 100)
print("CREATING TEMPORAL FEATURES")
print("=" * 100)

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
# 10. MERGE TRAIN WITH TTE
# ============================================================

train_full = (
    train_temporal
    .merge(
        train_tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# 11. CALCULATE TIME TO ENDPOINT
# ============================================================

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    -
    train_full["time_step"]
)


# ============================================================
# 12. C0-C4 LABEL FUNCTION
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
# 13. CENSORING-AWARE LABEL CONSTRUCTION
# ============================================================

train_full["class_label"] = np.nan

# Observed repair vehicles
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

# Non-repair / censored vehicles:
# only use observations where at least another 48 time units of
# known repair-free follow-up are available.
safe_non_repair_mask = (
    (train_full["in_study_repair"] == 0)
    &
    (train_full["time_to_endpoint"] > 48)
)

train_full.loc[
    safe_non_repair_mask,
    "class_label"
] = 0

candidate_train = (
    train_full
    .dropna(subset=["class_label"])
    .copy()
)

candidate_train["class_label"] = (
    candidate_train["class_label"]
    .astype(int)
)


# ============================================================
# 14. SEPARATE INTERNAL TRAIN / DEV OBSERVATIONS
# ============================================================

internal_train_candidates = (
    candidate_train[
        candidate_train["vehicle_id"]
        .isin(internal_train_ids)
    ]
    .copy()
)

internal_dev_candidates = (
    candidate_train[
        candidate_train["vehicle_id"]
        .isin(internal_dev_ids)
    ]
    .copy()
)

assert set(
    internal_train_candidates["vehicle_id"].unique()
).isdisjoint(
    set(internal_dev_candidates["vehicle_id"].unique())
)

print("\n" + "=" * 100)
print("ELIGIBLE OBSERVATIONS BEFORE SAMPLING")
print("=" * 100)

print("\nInternal train candidate distribution:")
print(
    internal_train_candidates["class_label"]
    .value_counts()
    .sort_index()
)

print("\nInternal dev candidate distribution:")
print(
    internal_dev_candidates["class_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. VEHICLE-AWARE SAMPLING FOR INTERNAL TRAIN ONLY
# ============================================================
#
# Repair vehicle:
#   max one observation per vehicle per available class.
#
# Non-repair vehicle:
#   one safely-labelled C0 observation.
#
# This matches the current reference training strategy.
# ============================================================

repair_candidates = (
    internal_train_candidates[
        internal_train_candidates["in_study_repair"] == 1
    ]
    .copy()
)

non_repair_candidates = (
    internal_train_candidates[
        internal_train_candidates["in_study_repair"] == 0
    ]
    .copy()
)

repair_sampled = (
    repair_candidates
    .groupby(
        [
            "vehicle_id",
            "class_label",
        ],
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

print("\n" + "=" * 100)
print("INTERNAL TRAIN — VEHICLE-AWARE DISTRIBUTION")
print("=" * 100)

print(
    train_sampled["class_label"]
    .value_counts()
    .sort_index()
)

print("\nTraining rows:", len(train_sampled))
print(
    "Training vehicles:",
    train_sampled["vehicle_id"].nunique()
)

assert set(
    train_sampled["class_label"].unique()
) == {
    0, 1, 2, 3, 4
}


# ============================================================
# 16. BUILD INTERNAL DEV CALIBRATION WEIGHTS
# ============================================================
#
# IMPORTANT:
# We do NOT class-balance the dev set.
#
# Instead, all safely-labelled dev observations are kept.
# Each vehicle gets total calibration weight = 1:
#
#   row_weight = 1 / (# eligible rows for that vehicle)
#
# This prevents vehicles with long histories from dominating
# calibration, while approximately representing a random cutoff
# drawn from each vehicle's eligible history.
# ============================================================

dev_rows_per_vehicle = (
    internal_dev_candidates
    .groupby("vehicle_id")["vehicle_id"]
    .transform("size")
)

internal_dev_candidates[
    "_calibration_weight"
] = (
    1.0
    /
    dev_rows_per_vehicle
)

# Normalize so mean weight = 1.
internal_dev_candidates[
    "_calibration_weight"
] = (
    internal_dev_candidates[
        "_calibration_weight"
    ]
    *
    (
        len(internal_dev_candidates)
        /
        internal_dev_candidates[
            "_calibration_weight"
        ].sum()
    )
)

print("\n" + "=" * 100)
print("INTERNAL DEV — VEHICLE-BALANCED NATURAL DISTRIBUTION")
print("=" * 100)

dev_weighted_class_mass = (
    internal_dev_candidates
    .groupby("class_label")[
        "_calibration_weight"
    ]
    .sum()
)

dev_weighted_prior = (
    dev_weighted_class_mass
    /
    dev_weighted_class_mass.sum()
)

print("\nWeighted dev class prior:")
for cls in CLASS_IDS:
    print(
        f"  C{cls}: "
        f"{dev_weighted_prior.get(cls, 0.0):.6f}"
    )

present_dev_classes = set(
    internal_dev_candidates[
        "class_label"
    ].unique()
)

if present_dev_classes != set(CLASS_IDS):
    raise ValueError(
        "Internal dev does not contain all five classes. "
        f"Present classes: {sorted(present_dev_classes)}. "
        "Increase DEV_SIZE or change RANDOM_STATE before fitting "
        "a five-class calibrator."
    )


# ============================================================
# 17. JOIN SPECIFICATIONS
# ============================================================

train_sampled = (
    train_sampled
    .merge(
        train_spec,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
)

internal_dev_candidates = (
    internal_dev_candidates
    .merge(
        train_spec,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# 18. BUILD OFFICIAL VALIDATION TABLE
# ============================================================
#
# Validation histories are already truncated.
# Temporal features are built first, then we use the last
# available readout per vehicle.
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

val_last = (
    val_last
    .merge(
        val_spec,
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )
)

val_last = (
    val_last
    .merge(
        val_labels,
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )
)

print("\nOfficial validation distribution:")
print(
    val_last["class_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 19. FREE LARGE INTERMEDIATE OBJECTS
# ============================================================

del train_temporal
del train_full
del candidate_train
del repair_candidates
del non_repair_candidates
del repair_sampled
del non_repair_sampled
gc.collect()


# ============================================================
# 20. DEFINE FEATURES
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
    "_calibration_weight",
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

X_dev = (
    internal_dev_candidates
    .drop(
        columns=NON_FEATURE_COLS,
        errors="ignore",
    )
)

y_dev = (
    internal_dev_candidates[
        "class_label"
    ]
    .astype(int)
    .copy()
)

dev_weights = (
    internal_dev_candidates[
        "_calibration_weight"
    ]
    .to_numpy(dtype=np.float64)
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
# 21. ALIGN FEATURE SCHEMAS
# ============================================================

train_only_dev = (
    set(X_train.columns)
    -
    set(X_dev.columns)
)

dev_only_train = (
    set(X_dev.columns)
    -
    set(X_train.columns)
)

train_only_val = (
    set(X_train.columns)
    -
    set(X_val.columns)
)

val_only_train = (
    set(X_val.columns)
    -
    set(X_train.columns)
)

print("\nTrain-only vs dev:", train_only_dev)
print("Dev-only vs train:", dev_only_train)

print("Train-only vs validation:", train_only_val)
print("Validation-only vs train:", val_only_train)

assert len(train_only_dev) == 0
assert len(dev_only_train) == 0
assert len(train_only_val) == 0
assert len(val_only_train) == 0

X_dev = X_dev[X_train.columns]
X_val = X_val[X_train.columns]

print("✓ Train / dev / validation feature schemas aligned.")


# ============================================================
# 22. FEATURE TYPES
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
# 23. PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
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
# 24. CURRENT REFERENCE MODEL CONFIGURATION
# ============================================================
#
# Freeze the same current architecture:
# Vehicle-aware Temporal XGBoost + W1 Light class weighting.
# ============================================================

LIGHT_WEIGHT_MAP = {
    0: 1.00,
    1: 1.25,
    2: 1.50,
    3: 1.75,
    4: 2.00,
}

train_sample_weights = (
    y_train
    .map(LIGHT_WEIGHT_MAP)
    .to_numpy(dtype=np.float32)
)

xgb_model = XGBClassifier(
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
            xgb_model,
        ),
    ]
)


# ============================================================
# 25. TRAIN INTERNAL-TRAIN MODEL
# ============================================================

print("\n" + "=" * 100)
print("TRAINING INTERNAL-TRAIN REFERENCE MODEL")
print("=" * 100)

print("Training rows:", len(X_train))
print("Internal dev rows:", len(X_dev))
print("Official validation rows:", len(X_val))

pipeline.fit(
    X_train,
    y_train,
    classifier__sample_weight=train_sample_weights,
)


# ============================================================
# 26. RAW PROBABILITIES
# ============================================================

dev_raw_proba = (
    pipeline
    .predict_proba(X_dev)
)

val_raw_proba = (
    pipeline
    .predict_proba(X_val)
)

assert dev_raw_proba.shape[1] == NUM_CLASSES
assert val_raw_proba.shape[1] == NUM_CLASSES

assert np.allclose(
    dev_raw_proba.sum(axis=1),
    1.0,
    atol=1e-5,
)

assert np.allclose(
    val_raw_proba.sum(axis=1),
    1.0,
    atol=1e-5,
)


# ============================================================
# 27. PROBABILITY UTILITY FUNCTIONS
# ============================================================

def normalize_probabilities(proba):
    proba = np.asarray(
        proba,
        dtype=np.float64,
    )

    proba = np.clip(
        proba,
        EPS,
        None,
    )

    row_sum = proba.sum(
        axis=1,
        keepdims=True,
    )

    return (
        proba
        /
        row_sum
    )


def weighted_class_prior(
    y,
    sample_weight=None,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    if sample_weight is None:
        sample_weight = np.ones(
            len(y),
            dtype=np.float64,
        )
    else:
        sample_weight = np.asarray(
            sample_weight,
            dtype=np.float64,
        )

    counts = np.bincount(
        y,
        weights=sample_weight,
        minlength=NUM_CLASSES,
    )

    return (
        counts
        /
        counts.sum()
    )


def multiclass_brier(
    y_true,
    proba,
    sample_weight=None,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    proba = normalize_probabilities(
        proba
    )

    one_hot = np.eye(
        NUM_CLASSES,
        dtype=np.float64,
    )[y_true]

    row_scores = (
        (proba - one_hot) ** 2
    ).sum(axis=1)

    if sample_weight is None:
        return float(
            row_scores.mean()
        )

    return float(
        np.average(
            row_scores,
            weights=sample_weight,
        )
    )


def multiclass_ece(
    y_true,
    proba,
    sample_weight=None,
    n_bins=15,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    proba = normalize_probabilities(
        proba
    )

    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)

    correct = (
        predicted
        ==
        y_true
    ).astype(float)

    if sample_weight is None:
        sample_weight = np.ones(
            len(y_true),
            dtype=np.float64,
        )
    else:
        sample_weight = np.asarray(
            sample_weight,
            dtype=np.float64,
        )

    total_weight = sample_weight.sum()

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for bin_idx in range(n_bins):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]

        if bin_idx == n_bins - 1:
            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if not np.any(mask):
            continue

        bin_weight = sample_weight[mask].sum()

        bin_accuracy = np.average(
            correct[mask],
            weights=sample_weight[mask],
        )

        bin_confidence = np.average(
            confidence[mask],
            weights=sample_weight[mask],
        )

        ece += (
            bin_weight
            /
            total_weight
        ) * abs(
            bin_accuracy
            -
            bin_confidence
        )

    return float(ece)


def calibration_metrics(
    name,
    y_true,
    proba,
    sample_weight=None,
):
    proba = normalize_probabilities(
        proba
    )

    result = {
        "ProbabilityMethod":
            name,

        "LogLoss":
            log_loss(
                y_true,
                proba,
                labels=CLASS_IDS,
                sample_weight=sample_weight,
            ),

        "MulticlassBrier":
            multiclass_brier(
                y_true,
                proba,
                sample_weight=sample_weight,
            ),

        "ECE":
            multiclass_ece(
                y_true,
                proba,
                sample_weight=sample_weight,
                n_bins=ECE_BINS,
            ),

        "ArgmaxAccuracy":
            accuracy_score(
                y_true,
                np.argmax(
                    proba,
                    axis=1,
                ),
                sample_weight=sample_weight,
            ),
    }

    actual_prior = weighted_class_prior(
        y_true,
        sample_weight=sample_weight,
    )

    mean_predicted = np.average(
        proba,
        axis=0,
        weights=sample_weight,
    )

    for cls in CLASS_IDS:
        result[
            f"ActualPrior_C{cls}"
        ] = actual_prior[cls]

        result[
            f"MeanPred_C{cls}"
        ] = mean_predicted[cls]

    return result


# ============================================================
# 28. CALIBRATION METHOD A — PRIOR CORRECTION
# ============================================================
#
# Approximate effective training prior:
#   class mass after vehicle-aware sampling AND W1 sample weights.
#
# Target prior:
#   vehicle-balanced natural internal-dev prior.
#
# Label-shift style correction:
#
#   p_corrected(y|x)
#       ∝ p_raw(y|x)
#         × target_prior(y)
#         / effective_train_prior(y)
# ============================================================

effective_train_prior = (
    weighted_class_prior(
        y_train,
        sample_weight=train_sample_weights,
    )
)

target_dev_prior = (
    weighted_class_prior(
        y_dev,
        sample_weight=dev_weights,
    )
)

prior_ratio = (
    target_dev_prior
    /
    np.clip(
        effective_train_prior,
        EPS,
        None,
    )
)


def apply_prior_correction(
    raw_proba,
    correction_ratio,
):
    corrected = (
        raw_proba
        *
        correction_ratio[None, :]
    )

    return normalize_probabilities(
        corrected
    )


dev_prior_proba = (
    apply_prior_correction(
        dev_raw_proba,
        prior_ratio,
    )
)

val_prior_proba = (
    apply_prior_correction(
        val_raw_proba,
        prior_ratio,
    )
)

print("\n" + "=" * 100)
print("PRIOR INFORMATION")
print("=" * 100)

print("\nEffective weighted internal-train prior:")
for cls in CLASS_IDS:
    print(
        f"  C{cls}: "
        f"{effective_train_prior[cls]:.6f}"
    )

print("\nVehicle-balanced internal-dev target prior:")
for cls in CLASS_IDS:
    print(
        f"  C{cls}: "
        f"{target_dev_prior[cls]:.6f}"
    )

print("\nPrior correction ratio:")
for cls in CLASS_IDS:
    print(
        f"  C{cls}: "
        f"{prior_ratio[cls]:.6f}"
    )


# ============================================================
# 29. CALIBRATION METHOD B — MULTINOMIAL LOGIT CALIBRATION
# ============================================================
#
# We use log(raw probability) as 5 calibration inputs.
#
# The calibrator is fitted ONLY on internal dev, with each vehicle
# contributing total weight 1.
#
# This is more flexible than a single temperature because it can
# learn class-specific bias corrections caused by sampling/weights.
# ============================================================

def probability_to_log_features(
    proba
):
    proba = normalize_probabilities(
        proba
    )

    return np.log(
        np.clip(
            proba,
            EPS,
            1.0,
        )
    )


calibrator = LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)

calibrator.fit(
    probability_to_log_features(
        dev_raw_proba
    ),
    y_dev,
    sample_weight=dev_weights,
)

if set(calibrator.classes_) != set(CLASS_IDS):
    raise ValueError(
        "Calibration model did not learn all five classes. "
        f"classes_ = {calibrator.classes_}"
    )


def aligned_calibrator_proba(
    fitted_calibrator,
    raw_proba,
):
    partial = (
        fitted_calibrator
        .predict_proba(
            probability_to_log_features(
                raw_proba
            )
        )
    )

    aligned = np.zeros(
        (
            len(raw_proba),
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )

    for column_index, class_id in enumerate(
        fitted_calibrator.classes_
    ):
        aligned[
            :,
            int(class_id)
        ] = partial[
            :,
            column_index
        ]

    return normalize_probabilities(
        aligned
    )


dev_logit_proba = (
    aligned_calibrator_proba(
        calibrator,
        dev_raw_proba,
    )
)

val_logit_proba = (
    aligned_calibrator_proba(
        calibrator,
        val_raw_proba,
    )
)


# ============================================================
# 30. INTERNAL DEV CALIBRATION COMPARISON
# ============================================================

DEV_PROBABILITY_SETS = {
    "Raw":
        dev_raw_proba,

    "PriorCorrected":
        dev_prior_proba,

    "LogitCalibrated":
        dev_logit_proba,
}

dev_calibration_rows = []

for method_name, proba in (
    DEV_PROBABILITY_SETS.items()
):
    dev_calibration_rows.append(
        calibration_metrics(
            method_name,
            y_dev,
            proba,
            sample_weight=dev_weights,
        )
    )

dev_calibration_df = pd.DataFrame(
    dev_calibration_rows
)

print("\n" + "=" * 100)
print("INTERNAL DEV CALIBRATION COMPARISON")
print("=" * 100)

dev_display_cols = [
    "ProbabilityMethod",
    "LogLoss",
    "MulticlassBrier",
    "ECE",
    "ArgmaxAccuracy",
]

print(
    dev_calibration_df[
        dev_display_cols
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 31. SELECT CALIBRATION METHOD USING INTERNAL DEV ONLY
# ============================================================

selected_row = (
    dev_calibration_df
    .sort_values(
        "LogLoss",
        ascending=True,
    )
    .iloc[0]
)

SELECTED_PROBABILITY_METHOD = (
    selected_row[
        "ProbabilityMethod"
    ]
)

print("\n" + "=" * 100)
print("SELECTED PROBABILITY METHOD")
print("=" * 100)

print(
    "Selected ONLY by internal-dev LogLoss:",
    SELECTED_PROBABILITY_METHOD,
)

print(
    "Internal-dev LogLoss:",
    round(
        float(
            selected_row[
                "LogLoss"
            ]
        ),
        6,
    )
)


# ============================================================
# 32. OFFICIAL VALIDATION CALIBRATION METRICS
# ============================================================
#
# These metrics are for evaluation/diagnosis only.
# They do NOT choose the calibrator.
# ============================================================

VAL_PROBABILITY_SETS = {
    "Raw":
        val_raw_proba,

    "PriorCorrected":
        val_prior_proba,

    "LogitCalibrated":
        val_logit_proba,
}

val_calibration_rows = []

for method_name, proba in (
    VAL_PROBABILITY_SETS.items()
):
    val_calibration_rows.append(
        calibration_metrics(
            method_name,
            y_val,
            proba,
            sample_weight=None,
        )
    )

val_calibration_df = pd.DataFrame(
    val_calibration_rows
)

print("\n" + "=" * 100)
print("OFFICIAL VALIDATION — CALIBRATION DIAGNOSTICS")
print("=" * 100)

print(
    val_calibration_df[
        dev_display_cols
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 33. OFFICIAL IDA COST MATRIX
# ============================================================
#
# Row    = Actual
# Column = Predicted
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


def calculate_total_cost(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    return float(
        COST_MATRIX[
            y_true,
            y_pred
        ]
        .sum()
    )


# ============================================================
# 34. ARGMAX / MEC DECISION FUNCTIONS
# ============================================================

def argmax_decision(
    proba
):
    return (
        np.argmax(
            proba,
            axis=1,
        )
        .astype(int)
    )


def mec_decision(
    proba
):
    proba = normalize_probabilities(
        proba
    )

    expected_cost = (
        proba
        @
        COST_MATRIX
    )

    prediction = (
        np.argmin(
            expected_cost,
            axis=1,
        )
        .astype(int)
    )

    return (
        prediction,
        expected_cost,
    )


# ============================================================
# 35. DECISION EVALUATION
# ============================================================

def evaluate_decision(
    probability_method,
    decision_method,
    y_true,
    y_pred,
):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pred_counts = (
        pd.Series(y_pred)
        .value_counts()
    )

    total_cost = (
        calculate_total_cost(
            y_true,
            y_pred,
        )
    )

    return {
        "ProbabilityMethod":
            probability_method,

        "DecisionMethod":
            decision_method,

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Macro Precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Macro Recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Macro F1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "Weighted F1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

        "C0 Recall":
            report["C0"]["recall"],

        "C1 Recall":
            report["C1"]["recall"],

        "C2 Recall":
            report["C2"]["recall"],

        "C3 Recall":
            report["C3"]["recall"],

        "C4 Recall":
            report["C4"]["recall"],

        "C4 Precision":
            report["C4"]["precision"],

        "C4 F1":
            report["C4"]["f1-score"],

        "Pred C0":
            int(
                pred_counts.get(
                    0,
                    0,
                )
            ),

        "Pred C1":
            int(
                pred_counts.get(
                    1,
                    0,
                )
            ),

        "Pred C2":
            int(
                pred_counts.get(
                    2,
                    0,
                )
            ),

        "Pred C3":
            int(
                pred_counts.get(
                    3,
                    0,
                )
            ),

        "Pred C4":
            int(
                pred_counts.get(
                    4,
                    0,
                )
            ),

        "Total Cost":
            total_cost,

        "Average Cost":
            total_cost
            /
            len(y_true),
    }


# ============================================================
# 36. COMPARE ALL PROBABILITY + DECISION COMBINATIONS
# ============================================================

decision_rows = []

prediction_store = {}
expected_cost_store = {}

for probability_method, proba in (
    VAL_PROBABILITY_SETS.items()
):
    # Argmax
    arg_pred = argmax_decision(
        proba
    )

    prediction_store[
        (
            probability_method,
            "Argmax",
        )
    ] = arg_pred

    decision_rows.append(
        evaluate_decision(
            probability_method,
            "Argmax",
            y_val,
            arg_pred,
        )
    )

    # MEC
    mec_pred, expected_cost = (
        mec_decision(
            proba
        )
    )

    prediction_store[
        (
            probability_method,
            "MEC",
        )
    ] = mec_pred

    expected_cost_store[
        probability_method
    ] = expected_cost

    decision_rows.append(
        evaluate_decision(
            probability_method,
            "MEC",
            y_val,
            mec_pred,
        )
    )


decision_results_df = pd.DataFrame(
    decision_rows
)

decision_display_cols = [
    "ProbabilityMethod",
    "DecisionMethod",
    "Accuracy",
    "Macro F1",
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

print("\n" + "=" * 100)
print("OFFICIAL VALIDATION — DECISION COMPARISON")
print("=" * 100)

print(
    decision_results_df[
        decision_display_cols
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 37. REPORT SELECTED CALIBRATION METHOD
# ============================================================
#
# We do NOT select the probability method by validation cost.
# The probability method was already selected on internal dev.
#
# We show both Argmax and MEC for that selected probability method.
# ============================================================

selected_decision_rows = (
    decision_results_df[
        decision_results_df[
            "ProbabilityMethod"
        ]
        ==
        SELECTED_PROBABILITY_METHOD
    ]
    .copy()
)

print("\n" + "=" * 100)
print(
    "OFFICIAL VALIDATION — INTERNALLY SELECTED "
    "PROBABILITY METHOD"
)
print("=" * 100)

print(
    selected_decision_rows[
        decision_display_cols
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 38. PROBABILITY PRIOR SUMMARY
# ============================================================

validation_actual_prior = (
    weighted_class_prior(
        y_val
    )
)

probability_summary_rows = []

for probability_method, proba in (
    VAL_PROBABILITY_SETS.items()
):
    mean_pred = (
        proba.mean(axis=0)
    )

    row = {
        "ProbabilityMethod":
            probability_method,
    }

    for cls in CLASS_IDS:
        row[
            f"ActualPrior_C{cls}"
        ] = (
            validation_actual_prior[cls]
        )

        row[
            f"MeanPred_C{cls}"
        ] = (
            mean_pred[cls]
        )

    probability_summary_rows.append(
        row
    )

probability_summary_df = pd.DataFrame(
    probability_summary_rows
)

print("\n" + "=" * 100)
print("VALIDATION ACTUAL PRIOR VS MEAN PREDICTED PROBABILITY")
print("=" * 100)

print(
    probability_summary_df
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 39. SAVE VEHICLE-LEVEL VALIDATION RESULTS
# ============================================================

validation_predictions_df = pd.DataFrame(
    {
        "vehicle_id":
            val_last[
                "vehicle_id"
            ].to_numpy(),

        "actual_class":
            y_val.to_numpy(),
    }
)

for probability_method, proba in (
    VAL_PROBABILITY_SETS.items()
):
    safe_name = (
        probability_method
        .replace(" ", "")
    )

    for cls in CLASS_IDS:
        validation_predictions_df[
            f"{safe_name}_P_C{cls}"
        ] = proba[
            :,
            cls
        ]

    validation_predictions_df[
        f"{safe_name}_Argmax"
    ] = prediction_store[
        (
            probability_method,
            "Argmax",
        )
    ]

    validation_predictions_df[
        f"{safe_name}_MEC"
    ] = prediction_store[
        (
            probability_method,
            "MEC",
        )
    ]


# ============================================================
# 40. CONFUSION MATRICES FOR SELECTED PROBABILITY METHOD
# ============================================================

selected_argmax_pred = (
    prediction_store[
        (
            SELECTED_PROBABILITY_METHOD,
            "Argmax",
        )
    ]
)

selected_mec_pred = (
    prediction_store[
        (
            SELECTED_PROBABILITY_METHOD,
            "MEC",
        )
    ]
)

selected_argmax_cm = confusion_matrix(
    y_val,
    selected_argmax_pred,
    labels=CLASS_IDS,
)

selected_mec_cm = confusion_matrix(
    y_val,
    selected_mec_pred,
    labels=CLASS_IDS,
)

selected_argmax_cm_df = pd.DataFrame(
    selected_argmax_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

selected_mec_cm_df = pd.DataFrame(
    selected_mec_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)


# ============================================================
# 41. SAVE OUTPUTS
# ============================================================

DEV_CALIBRATION_PATH = (
    OUTPUT_DIR
    /
    "model_06_calibration_metrics_dev.csv"
)

VAL_CALIBRATION_PATH = (
    OUTPUT_DIR
    /
    "model_06_calibration_metrics_validation.csv"
)

DECISION_RESULTS_PATH = (
    OUTPUT_DIR
    /
    "model_06_decision_results.csv"
)

PROBABILITY_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_06_probability_summary.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR
    /
    "model_06_validation_predictions.csv"
)

SELECTED_ARGMAX_CM_PATH = (
    OUTPUT_DIR
    /
    "model_06_selected_argmax_confusion_matrix.csv"
)

SELECTED_MEC_CM_PATH = (
    OUTPUT_DIR
    /
    "model_06_selected_mec_confusion_matrix.csv"
)

METHOD_SELECTION_PATH = (
    OUTPUT_DIR
    /
    "model_06_selected_probability_method.txt"
)

dev_calibration_df.to_csv(
    DEV_CALIBRATION_PATH,
    index=False,
)

val_calibration_df.to_csv(
    VAL_CALIBRATION_PATH,
    index=False,
)

decision_results_df.to_csv(
    DECISION_RESULTS_PATH,
    index=False,
)

probability_summary_df.to_csv(
    PROBABILITY_SUMMARY_PATH,
    index=False,
)

validation_predictions_df.to_csv(
    VALIDATION_PREDICTIONS_PATH,
    index=False,
)

selected_argmax_cm_df.to_csv(
    SELECTED_ARGMAX_CM_PATH
)

selected_mec_cm_df.to_csv(
    SELECTED_MEC_CM_PATH
)

METHOD_SELECTION_PATH.write_text(
    (
        "Selected probability method by internal-dev LogLoss:\n"
        f"{SELECTED_PROBABILITY_METHOD}\n\n"
        "Selection did NOT use official validation cost.\n"
    ),
    encoding="utf-8",
)


# ============================================================
# 42. FINAL INTERPRETATION GUIDE
# ============================================================

print("\n" + "=" * 100)
print("MODEL 6 COMPLETE")
print("=" * 100)

print(
    "\nProbability method selected by internal dev:",
    SELECTED_PROBABILITY_METHOD,
)

selected_argmax_row = (
    selected_decision_rows[
        selected_decision_rows[
            "DecisionMethod"
        ] == "Argmax"
    ]
    .iloc[0]
)

selected_mec_row = (
    selected_decision_rows[
        selected_decision_rows[
            "DecisionMethod"
        ] == "MEC"
    ]
    .iloc[0]
)

print("\nSelected method — Argmax:")
print(
    "  Total Cost:",
    selected_argmax_row[
        "Total Cost"
    ]
)
print(
    "  C4 Recall:",
    round(
        float(
            selected_argmax_row[
                "C4 Recall"
            ]
        ),
        6,
    )
)
print(
    "  C4 Precision:",
    round(
        float(
            selected_argmax_row[
                "C4 Precision"
            ]
        ),
        6,
    )
)

print("\nSelected method — MEC:")
print(
    "  Total Cost:",
    selected_mec_row[
        "Total Cost"
    ]
)
print(
    "  C4 Recall:",
    round(
        float(
            selected_mec_row[
                "C4 Recall"
            ]
        ),
        6,
    )
)
print(
    "  C4 Precision:",
    round(
        float(
            selected_mec_row[
                "C4 Precision"
            ]
        ),
        6,
    )
)

print(
    "\nInterpretation rule:"
)
print(
    "  1) First check whether calibration reduced internal-dev "
    "LogLoss / Brier / ECE."
)
print(
    "  2) Then check whether official-validation mean predicted "
    "class probabilities are closer to actual class prevalence."
)
print(
    "  3) Only then judge whether MEC becomes less extreme and "
    "reduces official business cost without flooding C4 alerts."
)
print(
    "  4) Do NOT automatically adopt MEC just because C4 Recall "
    "increases. Precision, alert volume, and total cost must also "
    "remain acceptable."
)

print(
    "\nNote: Test 06 trains the base model on 80% of official "
    "training vehicles to preserve a clean internal calibration set. "
    "Therefore its raw Argmax result is NOT expected to exactly match "
    "Test 05, which trained on the full official training set."
)

print("\nSaved files:")
print(" ", DEV_CALIBRATION_PATH)
print(" ", VAL_CALIBRATION_PATH)
print(" ", DECISION_RESULTS_PATH)
print(" ", PROBABILITY_SUMMARY_PATH)
print(" ", VALIDATION_PREDICTIONS_PATH)
print(" ", SELECTED_ARGMAX_CM_PATH)
print(" ", SELECTED_MEC_CM_PATH)
print(" ", METHOD_SELECTION_PATH)
