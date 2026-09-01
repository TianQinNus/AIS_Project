# ============================================================
# FleetCare AI
# Model 7: Three-Way Vehicle-Level Calibration Protocol
#          + Multi-Seed Robustness
# ============================================================
#
# Motivation:
#
# Test 06 confirmed that probability calibration is necessary,
# but it used the same Internal Dev set to:
#   1) fit the calibration model
#   2) evaluate calibration quality
#   3) select the calibration method
#
# This creates in-sample optimism.
#
# Test 07 fixes that by using THREE disjoint vehicle-level sets:
#
#   70% Base Train
#   15% Calibration Set
#   15% Selection Dev Set
#
# For each random seed:
#
#   Base Train
#       -> train Vehicle-Aware Temporal XGBoost + W1 Light
#
#   Calibration Set
#       -> fit calibration methods ONLY
#
#   Selection Dev Set
#       -> evaluate Raw / PriorCorrected / LogitCalibrated
#       -> choose best method for that split by LogLoss
#
# In addition, we repeat the entire three-way split over multiple
# random seeds to measure robustness.
#
# IMPORTANT:
#   Official validation is NEVER used to fit or select calibration.
#   It is used only as an external diagnostic after internal
#   selection.
#
# Main outputs:
#   - per-seed Selection Dev calibration metrics
#   - mean/std robustness summary
#   - calibration-method win counts
#   - external official-validation diagnostics
#   - robust selected probability method
#
# This test focuses on CALIBRATION PROTOCOL.
# Capacity-constrained maintenance policy is intentionally left
# for Test 08.
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
    log_loss,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [
    42,
    52,
    62,
    72,
    82,
]

BASE_TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
SELECTION_DEV_FRACTION = 0.15

NUM_CLASSES = 5
CLASS_IDS = np.arange(NUM_CLASSES)
CLASS_NAMES = ["C0", "C1", "C2", "C3", "C4"]

EPS = 1e-12
ECE_BINS = 15

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 320)


# ============================================================
# 2. PATHS
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

print("=" * 110)
print("MODEL 7 — THREE-WAY CALIBRATION + MULTI-SEED ROBUSTNESS")
print("=" * 110)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

print("\nChecking files:")

for path in required_files:
    print(f"  {path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Current DATA_DIR = {DATA_DIR}\n"
            "Please keep this script in the same project data folder "
            "used by test_00-test_06, or update DATA_DIR."
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 110)
print("LOADING DATA")
print("=" * 110)

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
# 6. IDENTIFY RAW FEATURES
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
# 7. SORT TIME SERIES
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
# 8. TEMPORAL FEATURE ENGINEERING
# ============================================================

def create_temporal_features(
    df,
    feature_cols,
):
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
        .replace(
            0,
            np.nan,
        )
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

    current_array = (
        df[feature_cols]
        .to_numpy(
            dtype=np.float32
        )
    )

    previous_array = (
        previous
        .to_numpy(
            dtype=np.float32
        )
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
        .to_numpy(
            dtype=np.float32
        )
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


print("\n" + "=" * 110)
print("CREATING TEMPORAL FEATURES ONCE")
print("=" * 110)

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
# 9. MERGE TRAIN WITH TTE
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
# 10. TIME TO ENDPOINT
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


# ============================================================
# 11. CLASS FUNCTION
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
# 12. CENSORING-AWARE LABELS
# ============================================================

train_full[
    "class_label"
] = np.nan


repair_mask = (
    train_full[
        "in_study_repair"
    ]
    ==
    1
)

train_full.loc[
    repair_mask,
    "class_label"
] = (
    train_full.loc[
        repair_mask,
        "time_to_endpoint"
    ]
    .apply(
        tte_to_class
    )
)


safe_non_repair_mask = (
    (
        train_full[
            "in_study_repair"
        ]
        ==
        0
    )
    &
    (
        train_full[
            "time_to_endpoint"
        ]
        >
        48
    )
)

train_full.loc[
    safe_non_repair_mask,
    "class_label"
] = 0


candidate_all = (
    train_full
    .dropna(
        subset=[
            "class_label"
        ]
    )
    .copy()
)

candidate_all[
    "class_label"
] = (
    candidate_all[
        "class_label"
    ]
    .astype(int)
)


# ============================================================
# 13. BUILD OFFICIAL VALIDATION TABLE ONCE
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
    val_last[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 14. VEHICLE TABLE FOR THREE-WAY SPLITS
# ============================================================

vehicle_table = (
    train_tte[
        [
            "vehicle_id",
            "in_study_repair",
        ]
    ]
    .drop_duplicates(
        "vehicle_id"
    )
    .copy()
)


# ============================================================
# 15. HELPER: THREE-WAY VEHICLE SPLIT
# ============================================================

def three_way_vehicle_split(
    vehicle_df,
    random_state,
):
    """
    70% Base Train
    15% Calibration
    15% Selection Dev

    Stratification uses vehicle-level in_study_repair.
    """

    train_ids, temp_ids = (
        train_test_split(
            vehicle_df[
                "vehicle_id"
            ].to_numpy(),

            test_size=(
                CALIBRATION_FRACTION
                +
                SELECTION_DEV_FRACTION
            ),

            random_state=random_state,

            stratify=vehicle_df[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    temp_table = (
        vehicle_df[
            vehicle_df[
                "vehicle_id"
            ].isin(
                temp_ids
            )
        ]
        .copy()
    )

    calibration_relative_fraction = (
        CALIBRATION_FRACTION
        /
        (
            CALIBRATION_FRACTION
            +
            SELECTION_DEV_FRACTION
        )
    )

    calibration_ids, selection_ids = (
        train_test_split(
            temp_table[
                "vehicle_id"
            ].to_numpy(),

            train_size=calibration_relative_fraction,

            random_state=random_state,

            stratify=temp_table[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    train_ids = set(
        train_ids
    )

    calibration_ids = set(
        calibration_ids
    )

    selection_ids = set(
        selection_ids
    )

    assert train_ids.isdisjoint(
        calibration_ids
    )

    assert train_ids.isdisjoint(
        selection_ids
    )

    assert calibration_ids.isdisjoint(
        selection_ids
    )

    assert (
        len(train_ids)
        +
        len(calibration_ids)
        +
        len(selection_ids)
        ==
        len(vehicle_df)
    )

    return (
        train_ids,
        calibration_ids,
        selection_ids,
    )


# ============================================================
# 16. HELPER: VEHICLE-AWARE BASE-TRAIN SAMPLING
# ============================================================

def vehicle_aware_sample(
    candidate_df,
    random_state,
):
    repair_candidates = (
        candidate_df[
            candidate_df[
                "in_study_repair"
            ]
            ==
            1
        ]
        .copy()
    )

    non_repair_candidates = (
        candidate_df[
            candidate_df[
                "in_study_repair"
            ]
            ==
            0
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
            random_state=random_state,
        )
        .reset_index(
            drop=True
        )
    )

    non_repair_sampled = (
        non_repair_candidates
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
        .reset_index(
            drop=True
        )
    )

    sampled = pd.concat(
        [
            repair_sampled,
            non_repair_sampled,
        ],
        ignore_index=True,
    )

    sampled = (
        sampled
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(
            drop=True
        )
    )

    return sampled


# ============================================================
# 17. HELPER: VEHICLE-BALANCED CALIBRATION WEIGHTS
# ============================================================
#
# Keep ALL safely-labelled rows, but every vehicle contributes
# total weight = 1.
# ============================================================

def add_vehicle_balanced_weights(
    df,
    weight_column,
):
    df = df.copy()

    rows_per_vehicle = (
        df
        .groupby(
            "vehicle_id"
        )[
            "vehicle_id"
        ]
        .transform(
            "size"
        )
    )

    df[
        weight_column
    ] = (
        1.0
        /
        rows_per_vehicle
    )

    # Normalize mean row weight to 1.
    df[
        weight_column
    ] = (
        df[
            weight_column
        ]
        *
        (
            len(df)
            /
            df[
                weight_column
            ].sum()
        )
    )

    return df


# ============================================================
# 18. PROBABILITY UTILITIES
# ============================================================

def normalize_probabilities(
    proba
):
    proba = np.asarray(
        proba,
        dtype=np.float64,
    )

    proba = np.clip(
        proba,
        EPS,
        None,
    )

    return (
        proba
        /
        proba.sum(
            axis=1,
            keepdims=True,
        )
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
        (
            proba
            -
            one_hot
        )
        **
        2
    ).sum(
        axis=1
    )

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

    confidence = (
        proba.max(
            axis=1
        )
    )

    predicted = (
        proba.argmax(
            axis=1
        )
    )

    correct = (
        predicted
        ==
        y_true
    ).astype(
        float
    )

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

    total_weight = (
        sample_weight.sum()
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for bin_idx in range(
        n_bins
    ):
        lower = edges[
            bin_idx
        ]

        upper = edges[
            bin_idx + 1
        ]

        if bin_idx == (
            n_bins - 1
        ):
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

        if not np.any(
            mask
        ):
            continue

        bin_weight = (
            sample_weight[
                mask
            ].sum()
        )

        bin_accuracy = (
            np.average(
                correct[
                    mask
                ],
                weights=sample_weight[
                    mask
                ],
            )
        )

        bin_confidence = (
            np.average(
                confidence[
                    mask
                ],
                weights=sample_weight[
                    mask
                ],
            )
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

    return float(
        ece
    )


def calibration_metrics(
    seed,
    evaluation_split,
    method_name,
    y_true,
    proba,
    sample_weight=None,
):
    proba = normalize_probabilities(
        proba
    )

    actual_prior = (
        weighted_class_prior(
            y_true,
            sample_weight=sample_weight,
        )
    )

    if sample_weight is None:
        mean_pred = (
            proba.mean(
                axis=0
            )
        )

    else:
        mean_pred = (
            np.average(
                proba,
                axis=0,
                weights=sample_weight,
            )
        )

    row = {
        "Seed":
            seed,

        "EvaluationSplit":
            evaluation_split,

        "ProbabilityMethod":
            method_name,

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

    for cls in CLASS_IDS:
        row[
            f"ActualPrior_C{cls}"
        ] = (
            actual_prior[
                cls
            ]
        )

        row[
            f"MeanPred_C{cls}"
        ] = (
            mean_pred[
                cls
            ]
        )

    return row


# ============================================================
# 19. CALIBRATION METHOD: PRIOR CORRECTION
# ============================================================

def apply_prior_correction(
    raw_proba,
    effective_train_prior,
    calibration_target_prior,
):
    ratio = (
        calibration_target_prior
        /
        np.clip(
            effective_train_prior,
            EPS,
            None,
        )
    )

    corrected = (
        raw_proba
        *
        ratio[
            None,
            :
        ]
    )

    return (
        normalize_probabilities(
            corrected
        ),
        ratio,
    )


# ============================================================
# 20. CALIBRATION METHOD: MULTINOMIAL LOGIT
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


def fit_logit_calibrator(
    raw_proba,
    y,
    sample_weight,
    random_state,
):
    calibrator = (
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    )

    calibrator.fit(
        probability_to_log_features(
            raw_proba
        ),
        y,
        sample_weight=sample_weight,
    )

    if set(
        calibrator.classes_
    ) != set(
        CLASS_IDS
    ):
        raise ValueError(
            "Logit calibrator did not learn all five classes. "
            f"classes_ = {calibrator.classes_}"
        )

    return calibrator


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

    for (
        column_index,
        class_id
    ) in enumerate(
        fitted_calibrator.classes_
    ):
        aligned[
            :,
            int(
                class_id
            )
        ] = partial[
            :,
            column_index
        ]

    return normalize_probabilities(
        aligned
    )


# ============================================================
# 21. MODEL CONFIGURATION
# ============================================================

LIGHT_WEIGHT_MAP = {
    0: 1.00,
    1: 1.25,
    2: 1.50,
    3: 1.75,
    4: 2.00,
}


def build_reference_pipeline(
    numeric_cols,
    categorical_cols,
    random_state,
):
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

        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
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
# 22. FEATURE COLUMNS THAT NEVER ENTER MODEL
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
    "_calibration_weight",
    "_selection_weight",
]


# ============================================================
# 23. OFFICIAL VALIDATION FEATURES
# ============================================================

X_val_template = (
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
    val_last[
        "class_label"
    ]
    .astype(int)
    .copy()
)


# ============================================================
# 24. MULTI-SEED EXPERIMENT
# ============================================================

selection_dev_rows = []
validation_rows = []
split_rows = []
prior_rows = []
per_seed_winner_rows = []


print("\n" + "=" * 110)
print("STARTING MULTI-SEED ROBUSTNESS EXPERIMENT")
print("=" * 110)

for seed in SEEDS:

    print("\n" + "#" * 110)
    print(f"SEED = {seed}")
    print("#" * 110)

    # --------------------------------------------------------
    # A. Three-way vehicle split
    # --------------------------------------------------------

    (
        base_train_ids,
        calibration_ids,
        selection_ids,
    ) = three_way_vehicle_split(
        vehicle_table,
        random_state=seed,
    )

    split_rows.append(
        {
            "Seed":
                seed,

            "BaseTrainVehicles":
                len(
                    base_train_ids
                ),

            "CalibrationVehicles":
                len(
                    calibration_ids
                ),

            "SelectionDevVehicles":
                len(
                    selection_ids
                ),
        }
    )

    # --------------------------------------------------------
    # B. Candidate observations per split
    # --------------------------------------------------------

    base_train_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                base_train_ids
            )
        ]
        .copy()
    )

    calibration_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                calibration_ids
            )
        ]
        .copy()
    )

    selection_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                selection_ids
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # C. Vehicle-aware sampling ONLY for base train
    # --------------------------------------------------------

    base_train_sampled = (
        vehicle_aware_sample(
            base_train_candidates,
            random_state=seed,
        )
    )

    # --------------------------------------------------------
    # D. Calibration and selection keep natural eligible rows
    #    but use equal total weight per vehicle
    # --------------------------------------------------------

    calibration_candidates = (
        add_vehicle_balanced_weights(
            calibration_candidates,
            "_calibration_weight",
        )
    )

    selection_candidates = (
        add_vehicle_balanced_weights(
            selection_candidates,
            "_selection_weight",
        )
    )

    # --------------------------------------------------------
    # E. Ensure all five classes exist in calibration/selection
    # --------------------------------------------------------

    calibration_classes = set(
        calibration_candidates[
            "class_label"
        ].unique()
    )

    selection_classes = set(
        selection_candidates[
            "class_label"
        ].unique()
    )

    if calibration_classes != set(
        CLASS_IDS
    ):
        raise ValueError(
            f"Seed {seed}: Calibration split missing classes. "
            f"Present = {sorted(calibration_classes)}"
        )

    if selection_classes != set(
        CLASS_IDS
    ):
        raise ValueError(
            f"Seed {seed}: Selection split missing classes. "
            f"Present = {sorted(selection_classes)}"
        )

    # --------------------------------------------------------
    # F. Join specifications
    # --------------------------------------------------------

    base_train_sampled = (
        base_train_sampled
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    calibration_candidates = (
        calibration_candidates
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    selection_candidates = (
        selection_candidates
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # G. X / y tables
    # --------------------------------------------------------

    X_train = (
        base_train_sampled
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_train = (
        base_train_sampled[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    X_cal = (
        calibration_candidates
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_cal = (
        calibration_candidates[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    cal_weights = (
        calibration_candidates[
            "_calibration_weight"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_sel = (
        selection_candidates
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_sel = (
        selection_candidates[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    selection_weights = (
        selection_candidates[
            "_selection_weight"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # H. Align feature schemas
    # --------------------------------------------------------

    assert set(
        X_train.columns
    ) == set(
        X_cal.columns
    )

    assert set(
        X_train.columns
    ) == set(
        X_sel.columns
    )

    assert set(
        X_train.columns
    ) == set(
        X_val_template.columns
    )

    X_cal = X_cal[
        X_train.columns
    ]

    X_sel = X_sel[
        X_train.columns
    ]

    X_val = X_val_template[
        X_train.columns
    ].copy()

    # --------------------------------------------------------
    # I. Feature types
    # --------------------------------------------------------

    categorical_cols = [
        col
        for col in X_train.columns
        if col.startswith(
            "Spec_"
        )
    ]

    numeric_cols = [
        col
        for col in X_train.columns
        if col not in categorical_cols
    ]

    # --------------------------------------------------------
    # J. Build and train base model
    # --------------------------------------------------------

    train_sample_weights = (
        y_train
        .map(
            LIGHT_WEIGHT_MAP
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    pipeline = (
        build_reference_pipeline(
            numeric_cols,
            categorical_cols,
            random_state=seed,
        )
    )

    print(
        "Base train rows:",
        len(
            X_train
        )
    )

    print(
        "Calibration rows:",
        len(
            X_cal
        )
    )

    print(
        "Selection rows:",
        len(
            X_sel
        )
    )

    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=train_sample_weights,
    )

    # --------------------------------------------------------
    # K. Raw probabilities
    # --------------------------------------------------------

    cal_raw = (
        pipeline
        .predict_proba(
            X_cal
        )
    )

    sel_raw = (
        pipeline
        .predict_proba(
            X_sel
        )
    )

    val_raw = (
        pipeline
        .predict_proba(
            X_val
        )
    )

    # --------------------------------------------------------
    # L. Priors
    # --------------------------------------------------------

    effective_train_prior = (
        weighted_class_prior(
            y_train,
            sample_weight=train_sample_weights,
        )
    )

    calibration_target_prior = (
        weighted_class_prior(
            y_cal,
            sample_weight=cal_weights,
        )
    )

    selection_actual_prior = (
        weighted_class_prior(
            y_sel,
            sample_weight=selection_weights,
        )
    )

    validation_actual_prior = (
        weighted_class_prior(
            y_val,
            sample_weight=None,
        )
    )

    # --------------------------------------------------------
    # M. Prior correction FIT on calibration set only
    # --------------------------------------------------------

    (
        cal_prior,
        prior_ratio,
    ) = (
        apply_prior_correction(
            cal_raw,
            effective_train_prior,
            calibration_target_prior,
        )
    )

    sel_prior = (
        normalize_probabilities(
            sel_raw
            *
            prior_ratio[
                None,
                :
            ]
        )
    )

    val_prior = (
        normalize_probabilities(
            val_raw
            *
            prior_ratio[
                None,
                :
            ]
        )
    )

    # --------------------------------------------------------
    # N. Logit calibrator FIT on calibration set only
    # --------------------------------------------------------

    logit_calibrator = (
        fit_logit_calibrator(
            cal_raw,
            y_cal,
            sample_weight=cal_weights,
            random_state=seed,
        )
    )

    cal_logit = (
        aligned_calibrator_proba(
            logit_calibrator,
            cal_raw,
        )
    )

    sel_logit = (
        aligned_calibrator_proba(
            logit_calibrator,
            sel_raw,
        )
    )

    val_logit = (
        aligned_calibrator_proba(
            logit_calibrator,
            val_raw,
        )
    )

    # --------------------------------------------------------
    # O. Save prior diagnostics
    # --------------------------------------------------------

    prior_row = {
        "Seed":
            seed,
    }

    for cls in CLASS_IDS:
        prior_row[
            f"EffectiveTrain_C{cls}"
        ] = effective_train_prior[
            cls
        ]

        prior_row[
            f"CalibrationTarget_C{cls}"
        ] = calibration_target_prior[
            cls
        ]

        prior_row[
            f"SelectionActual_C{cls}"
        ] = selection_actual_prior[
            cls
        ]

        prior_row[
            f"ValidationActual_C{cls}"
        ] = validation_actual_prior[
            cls
        ]

        prior_row[
            f"PriorRatio_C{cls}"
        ] = prior_ratio[
            cls
        ]

    prior_rows.append(
        prior_row
    )

    # --------------------------------------------------------
    # P. Selection Dev evaluation
    #    THIS is the real calibration model-selection split.
    # --------------------------------------------------------

    selection_probability_sets = {
        "Raw":
            sel_raw,

        "PriorCorrected":
            sel_prior,

        "LogitCalibrated":
            sel_logit,
    }

    current_seed_selection_rows = []

    for (
        method_name,
        proba
    ) in selection_probability_sets.items():

        row = (
            calibration_metrics(
                seed=seed,
                evaluation_split="SelectionDev",
                method_name=method_name,
                y_true=y_sel,
                proba=proba,
                sample_weight=selection_weights,
            )
        )

        selection_dev_rows.append(
            row
        )

        current_seed_selection_rows.append(
            row
        )

    current_seed_selection_df = (
        pd.DataFrame(
            current_seed_selection_rows
        )
    )

    seed_winner_row = (
        current_seed_selection_df
        .sort_values(
            "LogLoss",
            ascending=True,
        )
        .iloc[0]
    )

    per_seed_winner_rows.append(
        {
            "Seed":
                seed,

            "SelectedProbabilityMethod":
                seed_winner_row[
                    "ProbabilityMethod"
                ],

            "SelectionDevLogLoss":
                seed_winner_row[
                    "LogLoss"
                ],

            "SelectionDevBrier":
                seed_winner_row[
                    "MulticlassBrier"
                ],

            "SelectionDevECE":
                seed_winner_row[
                    "ECE"
                ],
        }
    )

    print(
        "Selection-Dev winner:",
        seed_winner_row[
            "ProbabilityMethod"
        ],
        "| LogLoss =",
        round(
            float(
                seed_winner_row[
                    "LogLoss"
                ]
            ),
            6,
        ),
    )

    # --------------------------------------------------------
    # Q. External official-validation diagnostics
    #    NOT used for calibration selection.
    # --------------------------------------------------------

    validation_probability_sets = {
        "Raw":
            val_raw,

        "PriorCorrected":
            val_prior,

        "LogitCalibrated":
            val_logit,
    }

    for (
        method_name,
        proba
    ) in validation_probability_sets.items():

        validation_rows.append(
            calibration_metrics(
                seed=seed,
                evaluation_split="OfficialValidation",
                method_name=method_name,
                y_true=y_val,
                proba=proba,
                sample_weight=None,
            )
        )

    # --------------------------------------------------------
    # R. Cleanup per-seed objects
    # --------------------------------------------------------

    del base_train_candidates
    del calibration_candidates
    del selection_candidates
    del base_train_sampled

    del X_train
    del X_cal
    del X_sel
    del X_val

    del pipeline
    del logit_calibrator

    del cal_raw
    del sel_raw
    del val_raw

    del cal_prior
    del sel_prior
    del val_prior

    del cal_logit
    del sel_logit
    del val_logit

    gc.collect()


# ============================================================
# 25. BUILD RESULT TABLES
# ============================================================

selection_dev_df = pd.DataFrame(
    selection_dev_rows
)

validation_df = pd.DataFrame(
    validation_rows
)

split_df = pd.DataFrame(
    split_rows
)

prior_df = pd.DataFrame(
    prior_rows
)

per_seed_winners_df = pd.DataFrame(
    per_seed_winner_rows
)


# ============================================================
# 26. ROBUSTNESS SUMMARY FUNCTION
# ============================================================

def summarize_metrics(
    df
):
    summary = (
        df
        .groupby(
            "ProbabilityMethod"
        )
        .agg(
            LogLoss_Mean=(
                "LogLoss",
                "mean",
            ),
            LogLoss_Std=(
                "LogLoss",
                "std",
            ),
            Brier_Mean=(
                "MulticlassBrier",
                "mean",
            ),
            Brier_Std=(
                "MulticlassBrier",
                "std",
            ),
            ECE_Mean=(
                "ECE",
                "mean",
            ),
            ECE_Std=(
                "ECE",
                "std",
            ),
            ArgmaxAccuracy_Mean=(
                "ArgmaxAccuracy",
                "mean",
            ),
            ArgmaxAccuracy_Std=(
                "ArgmaxAccuracy",
                "std",
            ),
        )
        .reset_index()
        .sort_values(
            "LogLoss_Mean",
            ascending=True,
        )
    )

    return summary


selection_dev_summary_df = (
    summarize_metrics(
        selection_dev_df
    )
)

validation_summary_df = (
    summarize_metrics(
        validation_df
    )
)


# ============================================================
# 27. METHOD WIN COUNTS
# ============================================================

winner_counts_df = (
    per_seed_winners_df[
        "SelectedProbabilityMethod"
    ]
    .value_counts()
    .rename_axis(
        "ProbabilityMethod"
    )
    .reset_index(
        name="WinCount"
    )
)

winner_counts_df[
    "WinRate"
] = (
    winner_counts_df[
        "WinCount"
    ]
    /
    len(
        SEEDS
    )
)


# ============================================================
# 28. ROBUST SELECTED METHOD
# ============================================================
#
# Primary rule:
#   lowest MEAN Selection-Dev LogLoss across seeds.
#
# This is selected WITHOUT using official validation.
# ============================================================

robust_selected_row = (
    selection_dev_summary_df
    .sort_values(
        "LogLoss_Mean",
        ascending=True,
    )
    .iloc[0]
)

ROBUST_SELECTED_METHOD = (
    robust_selected_row[
        "ProbabilityMethod"
    ]
)


# ============================================================
# 29. PRINT RESULTS
# ============================================================

print("\n" + "=" * 110)
print("SELECTION DEV — PER-SEED RESULTS")
print("=" * 110)

print(
    selection_dev_df[
        [
            "Seed",
            "ProbabilityMethod",
            "LogLoss",
            "MulticlassBrier",
            "ECE",
            "ArgmaxAccuracy",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("SELECTION DEV — ROBUSTNESS SUMMARY")
print("=" * 110)

print(
    selection_dev_summary_df
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("PER-SEED WINNERS")
print("=" * 110)

print(
    per_seed_winners_df
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("METHOD WIN COUNTS")
print("=" * 110)

print(
    winner_counts_df
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("ROBUST SELECTED PROBABILITY METHOD")
print("=" * 110)

print(
    "Selected by lowest MEAN Selection-Dev LogLoss:",
    ROBUST_SELECTED_METHOD,
)

print(
    "Mean Selection-Dev LogLoss:",
    round(
        float(
            robust_selected_row[
                "LogLoss_Mean"
            ]
        ),
        6,
    )
)

print(
    "Std Selection-Dev LogLoss:",
    round(
        float(
            robust_selected_row[
                "LogLoss_Std"
            ]
        ),
        6,
    )
)


print("\n" + "=" * 110)
print("OFFICIAL VALIDATION — EXTERNAL DIAGNOSTIC SUMMARY")
print("=" * 110)

print(
    validation_summary_df
    .round(6)
    .to_string(
        index=False
    )
)

print(
    "\nIMPORTANT: Official validation was NOT used to choose "
    "the calibration method."
)


# ============================================================
# 30. SAVE OUTPUTS
# ============================================================

SELECTION_BY_SEED_PATH = (
    OUTPUT_DIR
    /
    "model_07_selection_dev_metrics_by_seed.csv"
)

SELECTION_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_07_selection_dev_summary.csv"
)

VALIDATION_BY_SEED_PATH = (
    OUTPUT_DIR
    /
    "model_07_validation_metrics_by_seed.csv"
)

VALIDATION_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_07_validation_summary.csv"
)

SPLIT_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_07_split_summary.csv"
)

PRIOR_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_07_prior_summary_by_seed.csv"
)

PER_SEED_WINNERS_PATH = (
    OUTPUT_DIR
    /
    "model_07_per_seed_winners.csv"
)

WIN_COUNTS_PATH = (
    OUTPUT_DIR
    /
    "model_07_method_win_counts.csv"
)

ROBUST_SELECTED_METHOD_PATH = (
    OUTPUT_DIR
    /
    "model_07_selected_probability_method.txt"
)


selection_dev_df.to_csv(
    SELECTION_BY_SEED_PATH,
    index=False,
)

selection_dev_summary_df.to_csv(
    SELECTION_SUMMARY_PATH,
    index=False,
)

validation_df.to_csv(
    VALIDATION_BY_SEED_PATH,
    index=False,
)

validation_summary_df.to_csv(
    VALIDATION_SUMMARY_PATH,
    index=False,
)

split_df.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False,
)

prior_df.to_csv(
    PRIOR_SUMMARY_PATH,
    index=False,
)

per_seed_winners_df.to_csv(
    PER_SEED_WINNERS_PATH,
    index=False,
)

winner_counts_df.to_csv(
    WIN_COUNTS_PATH,
    index=False,
)

ROBUST_SELECTED_METHOD_PATH.write_text(
    (
        "FleetCare AI — Model 07\n"
        "Robust probability calibration selection\n\n"
        "Selection rule:\n"
        "Lowest mean Selection-Dev LogLoss across all seeds.\n\n"
        "Official validation was NOT used for method selection.\n\n"
        f"Selected method: {ROBUST_SELECTED_METHOD}\n"
        f"Mean Selection-Dev LogLoss: "
        f"{robust_selected_row['LogLoss_Mean']:.8f}\n"
        f"Std Selection-Dev LogLoss: "
        f"{robust_selected_row['LogLoss_Std']:.8f}\n"
    ),
    encoding="utf-8",
)


# ============================================================
# 31. FINAL INTERPRETATION GUIDE
# ============================================================

print("\n" + "=" * 110)
print("MODEL 7 COMPLETE")
print("=" * 110)

print(
    "\nUse these rules when interpreting Test 07:"
)

print(
    "1) The MAIN table is model_07_selection_dev_summary.csv."
)

print(
    "2) Prefer a calibration method that has BOTH low mean "
    "LogLoss and low standard deviation across seeds."
)

print(
    "3) Check model_07_method_win_counts.csv. A method that wins "
    "most seeds is more convincing than a method that wins only "
    "because of one unusually favorable split."
)

print(
    "4) Official-validation metrics are external diagnostics only. "
    "Do not change the selected calibration method simply because "
    "another method happens to look better on official validation."
)

print(
    "5) If PriorCorrected and LogitCalibrated are extremely close "
    "internally, prefer the simpler / more stable method rather than "
    "over-claiming a tiny difference."
)

print(
    "6) Test 08 should use the robustly selected calibrated risk "
    "score to build a capacity-constrained inspection policy."
)

print("\nSaved files:")

for path in [
    SELECTION_BY_SEED_PATH,
    SELECTION_SUMMARY_PATH,
    VALIDATION_BY_SEED_PATH,
    VALIDATION_SUMMARY_PATH,
    SPLIT_SUMMARY_PATH,
    PRIOR_SUMMARY_PATH,
    PER_SEED_WINNERS_PATH,
    WIN_COUNTS_PATH,
    ROBUST_SELECTED_METHOD_PATH,
]:
    print(" ", path)
