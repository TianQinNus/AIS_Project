# ============================================================
# FleetCare AI - FINAL TEMPORAL REASONING SYSTEM
#
# Vehicle-Aware Multi-History XGBoost
# + Historical Risk Trajectory
# + Fleet-Relative Risk
# + Risk Persistence
# + Relative Deterioration
# + Prediction Stability
# + Temporal Rule Engine
# + Evaluation
# + Frontend Data Generation
#
# IMPORTANT:
# Raw XGBoost scores are used.
#
# Calibration was experimentally evaluated but NOT adopted:
#
# Raw LogLoss        = 0.8911
# Calibrated LogLoss = 0.9707
#
# Raw Macro-F1        = 0.2031
# Calibrated Macro-F1 = 0.1923
#
# Therefore Risk48 / Risk24 / Risk12 / Risk6 are treated
# as MODEL RISK SCORES, not calibrated real-world probabilities.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import gc
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = Path("../data/raw")

MODEL_DIR = Path(
    "../data/processed/final_model"
)

OUTPUT_DIR = Path(
    "../data/processed/system_output"
)

PLOT_DIR = (
    OUTPUT_DIR
    / "plots"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CLASS_IDS = [
    0,
    1,
    2,
    3,
    4
]


CLASS_NAMES = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}


CLASS_DESCRIPTIONS = {

    0:
        "No repair predicted within 48 operational units",

    1:
        "Repair proximity: 24-48 operational units",

    2:
        "Repair proximity: 12-24 operational units",

    3:
        "Repair proximity: 6-12 operational units",

    4:
        "Repair proximity: 0-6 operational units",
}


HORIZONS = {

    "Risk48": [
        1,
        2,
        3,
        4
    ],

    "Risk24": [
        2,
        3,
        4
    ],

    "Risk12": [
        3,
        4
    ],

    "Risk6": [
        4
    ],
}


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

print("=" * 80)
print("FLEETCARE AI - FINAL TEMPORAL REASONING SYSTEM")
print("=" * 80)


print(
    "\n[1/15] Loading final model..."
)


MODEL_PATH = (
    MODEL_DIR
    / "fleetcare_vehicle_aware_history_xgb.joblib"
)

FEATURE_PATH = (
    MODEL_DIR
    / "fleetcare_history_feature_columns.joblib"
)

RAW_FEATURE_PATH = (
    MODEL_DIR
    / "fleetcare_raw_feature_columns.joblib"
)


model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURE_PATH
)

raw_feature_columns = joblib.load(
    RAW_FEATURE_PATH
)


print(
    "Model loaded successfully."
)

print(
    "Model features:",
    len(feature_columns)
)

print(
    "Raw operational features:",
    len(raw_feature_columns)
)


# ============================================================
# 4. LOAD VALIDATION DATA
# ============================================================

print(
    "\n[2/15] Loading validation data..."
)


operational = pd.read_csv(
    DATA_DIR
    / "validation_operational_readouts.csv"
)


specifications = pd.read_csv(
    DATA_DIR
    / "validation_specifications.csv"
)


labels = pd.read_csv(
    DATA_DIR
    / "validation_labels.csv"
)


print(
    "Operational:",
    operational.shape
)

print(
    "Specifications:",
    specifications.shape
)

print(
    "Labels:",
    labels.shape
)


# ============================================================
# 5. MULTI-HISTORY FEATURE ENGINEERING
# ============================================================

def create_history_features(
    df,
    feature_cols
):

    print(
        "  Sorting history..."
    )


    df = (
        df
        .sort_values(
            [
                "vehicle_id",
                "time_step"
            ]
        )
        .reset_index(
            drop=True
        )
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
        grouped[
            "time_step"
        ]
        .diff()
        .astype(
            np.float32
        )
    )


    safe_gap = (
        df[
            "time_gap"
        ]
        .replace(
            0,
            np.nan
        )
    )


    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    previous = (
        grouped[
            feature_cols
        ]
        .shift(1)
        .astype(
            np.float32
        )
    )


    previous.columns = [
        f"{column}__prev"
        for column in feature_cols
    ]


    current_array = (
        df[
            feature_cols
        ]
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


    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta_array = (
        current_array
        -
        previous_array
    )


    delta = pd.DataFrame(
        delta_array,
        index=df.index,
        columns=[
            f"{column}__delta"
            for column in feature_cols
        ]
    )


    # --------------------------------------------------------
    # Rate
    # --------------------------------------------------------

    gap_array = (
        safe_gap
        .to_numpy(
            dtype=np.float32
        )
        [:, None]
    )


    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        rate_array = (
            delta_array
            /
            gap_array
        )


    rate = pd.DataFrame(
        rate_array,
        index=df.index,
        columns=[
            f"{column}__rate"
            for column in feature_cols
        ]
    )


    rate.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )


    # --------------------------------------------------------
    # Change from 3 readouts ago
    # --------------------------------------------------------

    lag3 = (
        grouped[
            feature_cols
        ]
        .shift(3)
        .astype(
            np.float32
        )
    )


    change3_array = (
        current_array
        -
        lag3.to_numpy(
            dtype=np.float32
        )
    )


    change3 = pd.DataFrame(
        change3_array,
        index=df.index,
        columns=[
            f"{column}__change3"
            for column in feature_cols
        ]
    )


    del lag3
    del change3_array

    gc.collect()


    # --------------------------------------------------------
    # Change from 5 readouts ago
    # --------------------------------------------------------

    lag5 = (
        grouped[
            feature_cols
        ]
        .shift(5)
        .astype(
            np.float32
        )
    )


    change5_array = (
        current_array
        -
        lag5.to_numpy(
            dtype=np.float32
        )
    )


    change5 = pd.DataFrame(
        change5_array,
        index=df.index,
        columns=[
            f"{column}__change5"
            for column in feature_cols
        ]
    )


    del lag5
    del change5_array

    gc.collect()


    # --------------------------------------------------------
    # Rolling mean 3
    # --------------------------------------------------------

    rolling_mean3 = (
        grouped[
            feature_cols
        ]
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
        .reindex(
            df.index
        )
        .astype(
            np.float32
        )
    )


    rolling_mean3.columns = [
        f"{column}__mean3"
        for column in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling std 3
    # --------------------------------------------------------

    rolling_std3 = (
        grouped[
            feature_cols
        ]
        .rolling(
            window=3,
            min_periods=2
        )
        .std()
        .reset_index(
            level=0,
            drop=True
        )
        .reindex(
            df.index
        )
        .astype(
            np.float32
        )
    )


    rolling_std3.columns = [
        f"{column}__std3"
        for column in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling mean 5
    # --------------------------------------------------------

    rolling_mean5 = (
        grouped[
            feature_cols
        ]
        .rolling(
            window=5,
            min_periods=1
        )
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
        .reindex(
            df.index
        )
        .astype(
            np.float32
        )
    )


    rolling_mean5.columns = [
        f"{column}__mean5"
        for column in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling std 5
    # --------------------------------------------------------

    rolling_std5 = (
        grouped[
            feature_cols
        ]
        .rolling(
            window=5,
            min_periods=2
        )
        .std()
        .reset_index(
            level=0,
            drop=True
        )
        .reindex(
            df.index
        )
        .astype(
            np.float32
        )
    )


    rolling_std5.columns = [
        f"{column}__std5"
        for column in feature_cols
    ]


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(
        [
            df,
            previous,
            delta,
            rate,
            change3,
            change5,
            rolling_mean3,
            rolling_std3,
            rolling_mean5,
            rolling_std5
        ],
        axis=1
    )


    result.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )


    return result


# ============================================================
# 6. CREATE HISTORICAL FEATURES
# ============================================================

print(
    "\n[3/15] Creating multi-history features..."
)


history = create_history_features(
    operational,
    raw_feature_columns
)


# ============================================================
# 7. ADD VEHICLE SPECIFICATIONS
# ============================================================

print(
    "\n[4/15] Adding vehicle specifications..."
)


history = history.merge(
    specifications,
    on="vehicle_id",
    how="left",
    validate="many_to_one"
)


history = (
    history
    .sort_values(
        [
            "vehicle_id",
            "time_step"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Historical snapshots:",
    len(history)
)

print(
    "Vehicles:",
    history[
        "vehicle_id"
    ].nunique()
)


# ============================================================
# 8. MODEL INFERENCE ON ALL HISTORICAL SNAPSHOTS
# ============================================================

print(
    "\n[5/15] Running historical model inference..."
)


X_history = (
    history
    .drop(
        columns=[
            "vehicle_id"
        ],
        errors="ignore"
    )
    .reindex(
        columns=feature_columns
    )
)


history_proba = model.predict_proba(
    X_history
)


history[
    "predicted_class"
] = np.argmax(
    history_proba,
    axis=1
)


history[
    "confidence"
] = np.max(
    history_proba,
    axis=1
)


history["P_C0"] = (
    history_proba[:, 0]
)

history["P_C1"] = (
    history_proba[:, 1]
)

history["P_C2"] = (
    history_proba[:, 2]
)

history["P_C3"] = (
    history_proba[:, 3]
)

history["P_C4"] = (
    history_proba[:, 4]
)


# ------------------------------------------------------------
# Cumulative risk scores
# ------------------------------------------------------------

history[
    "Risk48"
] = (
    history_proba[
        :,
        1:
    ]
    .sum(
        axis=1
    )
)


history[
    "Risk24"
] = (
    history_proba[
        :,
        2:
    ]
    .sum(
        axis=1
    )
)


history[
    "Risk12"
] = (
    history_proba[
        :,
        3:
    ]
    .sum(
        axis=1
    )
)


history[
    "Risk6"
] = (
    history_proba[
        :,
        4
    ]
)


print(
    "Historical inference complete."
)


# ============================================================
# 9. BUILD TEMPORAL REASONING FEATURES
# ============================================================

print(
    "\n[6/15] Building temporal reasoning evidence..."
)


def build_temporal_reasoning(
    group
):

    group = (
        group
        .sort_values(
            "time_step"
        )
        .copy()
    )


    n = len(group)

    current = group.iloc[-1]


    # --------------------------------------------------------
    # History sufficiency
    # --------------------------------------------------------

    history_count = n


    if history_count >= 5:

        history_quality = "STRONG"

    elif history_count >= 3:

        history_quality = "MODERATE"

    else:

        history_quality = "LIMITED"


    # --------------------------------------------------------
    # Previous snapshot
    # --------------------------------------------------------

    if n >= 2:

        previous = group.iloc[-2]

        previous_class = int(
            previous[
                "predicted_class"
            ]
        )

        previous_risk48 = float(
            previous[
                "Risk48"
            ]
        )

        previous_risk12 = float(
            previous[
                "Risk12"
            ]
        )

        previous_risk6 = float(
            previous[
                "Risk6"
            ]
        )

        time_gap = (
            float(
                current[
                    "time_step"
                ]
            )
            -
            float(
                previous[
                    "time_step"
                ]
            )
        )

    else:

        previous_class = np.nan

        previous_risk48 = np.nan

        previous_risk12 = np.nan

        previous_risk6 = np.nan

        time_gap = np.nan


    # --------------------------------------------------------
    # Current - previous changes
    # --------------------------------------------------------

    if n >= 2:

        risk48_change = (
            float(
                current[
                    "Risk48"
                ]
            )
            -
            previous_risk48
        )

        risk12_change = (
            float(
                current[
                    "Risk12"
                ]
            )
            -
            previous_risk12
        )

        risk6_change = (
            float(
                current[
                    "Risk6"
                ]
            )
            -
            previous_risk6
        )

    else:

        risk48_change = 0.0
        risk12_change = 0.0
        risk6_change = 0.0


    # --------------------------------------------------------
    # Current / previous slope
    # --------------------------------------------------------

    if (
        n >= 2
        and pd.notna(
            time_gap
        )
        and time_gap > 0
    ):

        risk48_slope = (
            risk48_change
            /
            time_gap
        )

        risk12_slope = (
            risk12_change
            /
            time_gap
        )

        risk6_slope = (
            risk6_change
            /
            time_gap
        )

    else:

        risk48_slope = 0.0
        risk12_slope = 0.0
        risk6_slope = 0.0


    # --------------------------------------------------------
    # Recent history
    # --------------------------------------------------------

    recent3 = group.tail(
        3
    )

    recent5 = group.tail(
        5
    )


    # --------------------------------------------------------
    # Trend using actual operational time
    # --------------------------------------------------------

    def temporal_slope(
        frame,
        column
    ):

        if len(frame) < 2:

            return 0.0


        x = (
            frame[
                "time_step"
            ]
            .to_numpy(
                dtype=float
            )
        )


        y = (
            frame[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )


        if np.ptp(
            x
        ) == 0:

            return 0.0


        return float(
            np.polyfit(
                x,
                y,
                1
            )[0]
        )


    risk48_trend = temporal_slope(
        recent5,
        "Risk48"
    )


    risk12_trend = temporal_slope(
        recent5,
        "Risk12"
    )


    risk6_trend = temporal_slope(
        recent5,
        "Risk6"
    )


    # --------------------------------------------------------
    # Recent class history
    # --------------------------------------------------------

    recent_classes = (
        recent5[
            "predicted_class"
        ]
        .astype(int)
        .tolist()
    )


    if len(
        recent_classes
    ) >= 2:

        class_change = (
            recent_classes[-1]
            -
            recent_classes[0]
        )

    else:

        class_change = 0


    current_class = int(
        current[
            "predicted_class"
        ]
    )


    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    class_stability = float(
        (
            recent5[
                "predicted_class"
            ]
            ==
            current_class
        ).mean()
    )


    unique_recent_classes = int(
        recent5[
            "predicted_class"
        ]
        .nunique()
    )


    # --------------------------------------------------------
    # Recent mean risk
    # --------------------------------------------------------

    recent_risk48_mean = float(
        recent3[
            "Risk48"
        ].mean()
    )


    recent_risk12_mean = float(
        recent3[
            "Risk12"
        ].mean()
    )


    recent_risk6_mean = float(
        recent3[
            "Risk6"
        ].mean()
    )


    # --------------------------------------------------------
    # Descriptive trend direction
    #
    # This is retained for frontend display only.
    # It is NOT used as the main escalation criterion.
    # --------------------------------------------------------

    positive_trends = sum(
        [
            risk48_trend > 0,
            risk12_trend > 0,
            risk6_trend > 0
        ]
    )


    if positive_trends >= 2:

        trend_direction = (
            "RISING"
        )

    elif positive_trends == 0:

        trend_direction = (
            "STABLE_OR_FALLING"
        )

    else:

        trend_direction = (
            "MIXED"
        )


    return pd.Series({

        "time_step":
            current[
                "time_step"
            ],

        "history_count":
            history_count,

        "history_quality":
            history_quality,

        "previous_class":
            previous_class,

        "previous_Risk48":
            previous_risk48,

        "previous_Risk12":
            previous_risk12,

        "previous_Risk6":
            previous_risk6,

        "Risk48_change":
            risk48_change,

        "Risk12_change":
            risk12_change,

        "Risk6_change":
            risk6_change,

        "Risk48_slope":
            risk48_slope,

        "Risk12_slope":
            risk12_slope,

        "Risk6_slope":
            risk6_slope,

        "Risk48_trend":
            risk48_trend,

        "Risk12_trend":
            risk12_trend,

        "Risk6_trend":
            risk6_trend,

        "trend_direction":
            trend_direction,

        "class_change":
            class_change,

        "class_stability":
            class_stability,

        "unique_recent_classes":
            unique_recent_classes,

        "recent_Risk48_mean":
            recent_risk48_mean,

        "recent_Risk12_mean":
            recent_risk12_mean,

        "recent_Risk6_mean":
            recent_risk6_mean,

        "recent_class_history":
            " -> ".join(
                [
                    f"C{x}"
                    for x
                    in recent_classes
                ]
            )
    })


# ------------------------------------------------------------
# Explicit group loop.
#
# This avoids the pandas include_groups=False issue that
# previously removed vehicle_id.
# ------------------------------------------------------------

temporal_reasoning_rows = []


for (
    vehicle_id,
    group
) in history.groupby(
    "vehicle_id",
    sort=False
):

    result = (
        build_temporal_reasoning(
            group
        )
    )

    result[
        "vehicle_id"
    ] = vehicle_id

    temporal_reasoning_rows.append(
        result
    )


temporal_reasoning = pd.DataFrame(
    temporal_reasoning_rows
)


temporal_reasoning = temporal_reasoning[
    [
        "vehicle_id"
    ]
    +
    [
        column
        for column
        in temporal_reasoning.columns
        if column
        != "vehicle_id"
    ]
]


temporal_reasoning = (
    temporal_reasoning
    .reset_index(
        drop=True
    )
)


print(
    "Temporal reasoning rows:",
    len(
        temporal_reasoning
    )
)


# ============================================================
# 10. GET CURRENT / LATEST SNAPSHOT
# ============================================================

latest = (
    history
    .groupby(
        "vehicle_id",
        sort=False
    )
    .tail(1)
    .copy()
)


latest = (
    latest
    .sort_values(
        "vehicle_id"
    )
    .reset_index(
        drop=True
    )
)


latest = latest[
    [
        "vehicle_id",
        "time_step",
        "predicted_class",
        "confidence",
        "P_C0",
        "P_C1",
        "P_C2",
        "P_C3",
        "P_C4",
        "Risk48",
        "Risk24",
        "Risk12",
        "Risk6",
    ]
]


latest = latest.merge(
    temporal_reasoning,
    on=[
        "vehicle_id",
        "time_step"
    ],
    how="left",
    validate="one_to_one"
)


assert latest[
    "vehicle_id"
].is_unique


print(
    "Latest snapshots:",
    len(latest)
)


# ============================================================
# 11. CURRENT FLEET RISK PERCENTILES
# ============================================================

print(
    "\n[7/15] Calculating current fleet risk percentiles..."
)


for risk_column in [
    "Risk48",
    "Risk24",
    "Risk12",
    "Risk6"
]:

    latest[
        f"{risk_column}_percentile"
    ] = (
        latest[
            risk_column
        ]
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )


# ============================================================
# 12. HISTORICAL FLEET RISK PERCENTILES
# ============================================================
#
# Used to measure persistence.
#
# NOTE:
# These are relative risk rankings,
# not calibrated repair probabilities.
# ============================================================

for risk_column in [
    "Risk48",
    "Risk12",
    "Risk6"
]:

    history[
        f"{risk_column}_percentile"
    ] = (
        history[
            risk_column
        ]
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )


# ============================================================
# 13. HISTORICAL RISK PERSISTENCE
# ============================================================

print(
    "\n[8/15] Calculating historical risk persistence..."
)


def calculate_persistence(
    group
):

    group = (
        group
        .sort_values(
            "time_step"
        )
        .tail(5)
    )


    n = len(group)


    if n == 0:

        return pd.Series({

            "Risk48_high_count_5":
                0,

            "Risk12_high_count_5":
                0,

            "Risk6_high_count_5":
                0,

            "Risk48_persistence":
                0.0,

            "Risk12_persistence":
                0.0,

            "Risk6_persistence":
                0.0,
        })


    # --------------------------------------------------------
    # Historical high-risk definitions
    # --------------------------------------------------------

    risk48_count = int(
        (
            group[
                "Risk48_percentile"
            ]
            >= 80
        ).sum()
    )


    risk12_count = int(
        (
            group[
                "Risk12_percentile"
            ]
            >= 90
        ).sum()
    )


    risk6_count = int(
        (
            group[
                "Risk6_percentile"
            ]
            >= 95
        ).sum()
    )


    return pd.Series({

        "Risk48_high_count_5":
            risk48_count,

        "Risk12_high_count_5":
            risk12_count,

        "Risk6_high_count_5":
            risk6_count,

        "Risk48_persistence":
            risk48_count
            / n,

        "Risk12_persistence":
            risk12_count
            / n,

        "Risk6_persistence":
            risk6_count
            / n,
    })


persistence_rows = []


for (
    vehicle_id,
    group
) in history.groupby(
    "vehicle_id",
    sort=False
):

    result = (
        calculate_persistence(
            group
        )
    )

    result[
        "vehicle_id"
    ] = vehicle_id

    persistence_rows.append(
        result
    )


persistence = pd.DataFrame(
    persistence_rows
)


persistence = persistence[
    [
        "vehicle_id"
    ]
    +
    [
        column
        for column
        in persistence.columns
        if column
        != "vehicle_id"
    ]
]


latest = latest.merge(
    persistence,
    on="vehicle_id",
    how="left",
    validate="one_to_one"
)


print(
    "Persistence rows:",
    len(
        persistence
    )
)


# ============================================================
# 14. RELATIVE DETERIORATION PERCENTILES
# ============================================================
#
# IMPORTANT IMPROVEMENT:
#
# risk_trend > 0 alone is NOT treated as abnormal.
#
# Instead, each vehicle's risk deterioration rate is compared
# against the rest of the fleet.
#
# Top 10% deterioration = unusually rapid deterioration.
# ============================================================

print(
    "\n[9/15] Calculating relative deterioration..."
)


for trend_column in [
    "Risk48_trend",
    "Risk12_trend",
    "Risk6_trend"
]:

    latest[
        f"{trend_column}_percentile"
    ] = (
        latest[
            trend_column
        ]
        .rank(
            pct=True,
            method="average"
        )
        * 100
    )


latest[
    "fast_Risk48_deterioration"
] = (
    latest[
        "Risk48_trend_percentile"
    ]
    >= 90
)


latest[
    "fast_Risk12_deterioration"
] = (
    latest[
        "Risk12_trend_percentile"
    ]
    >= 90
)


latest[
    "fast_Risk6_deterioration"
] = (
    latest[
        "Risk6_trend_percentile"
    ]
    >= 90
)


latest[
    "fast_near_term_deterioration"
] = (
    latest[
        "fast_Risk6_deterioration"
    ]
    |
    latest[
        "fast_Risk12_deterioration"
    ]
)


print(
    "Fast Risk48 deterioration:",
    int(
        latest[
            "fast_Risk48_deterioration"
        ].sum()
    )
)


print(
    "Fast Risk12 deterioration:",
    int(
        latest[
            "fast_Risk12_deterioration"
        ].sum()
    )
)


print(
    "Fast Risk6 deterioration:",
    int(
        latest[
            "fast_Risk6_deterioration"
        ].sum()
    )
)


# ============================================================
# 15. REASONING FLAGS
# ============================================================

latest[
    "is_class_worsening"
] = (
    latest[
        "class_change"
    ]
    > 0
)


latest[
    "is_prediction_stable"
] = (
    latest[
        "class_stability"
    ]
    >= 0.60
)


# ------------------------------------------------------------
# Persistence now requires >= 3 of recent 5 observations
# ------------------------------------------------------------

latest[
    "is_risk48_persistent"
] = (
    latest[
        "Risk48_persistence"
    ]
    >= 0.60
)


latest[
    "is_risk12_persistent"
] = (
    latest[
        "Risk12_persistence"
    ]
    >= 0.60
)


latest[
    "is_risk6_persistent"
] = (
    latest[
        "Risk6_persistence"
    ]
    >= 0.60
)


latest[
    "limited_history"
] = (
    latest[
        "history_count"
    ]
    < 3
)


# Descriptive only

latest[
    "is_rising"
] = (
    latest[
        "trend_direction"
    ]
    == "RISING"
)


# ============================================================
# 16. FINAL TEMPORAL RULE ENGINE
# ============================================================
#
# Decisions use explicit conditions rather than an arbitrary
# evidence-score threshold.
#
# Inputs:
#
# - Current C0-C4 model state
# - Fleet-relative Risk48 / Risk24 / Risk12 / Risk6
# - Historical persistence
# - Relative deterioration rate
# - Class trajectory
# - Prediction stability
# - History sufficiency
#
# Evidence score is retained only for explanation / frontend.
# ============================================================

print(
    "\n[10/15] Applying final temporal rule engine..."
)


def temporal_rule_engine(
    row
):

    # --------------------------------------------------------
    # Current model state
    # --------------------------------------------------------

    cls = int(
        row[
            "predicted_class"
        ]
    )


    confidence = float(
        row[
            "confidence"
        ]
    )


    # --------------------------------------------------------
    # Current relative risk
    # --------------------------------------------------------

    risk48_pct = float(
        row[
            "Risk48_percentile"
        ]
    )


    risk24_pct = float(
        row[
            "Risk24_percentile"
        ]
    )


    risk12_pct = float(
        row[
            "Risk12_percentile"
        ]
    )


    risk6_pct = float(
        row[
            "Risk6_percentile"
        ]
    )


    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistent48 = bool(
        row[
            "is_risk48_persistent"
        ]
    )


    persistent12 = bool(
        row[
            "is_risk12_persistent"
        ]
    )


    persistent6 = bool(
        row[
            "is_risk6_persistent"
        ]
    )


    # --------------------------------------------------------
    # Relative deterioration
    # --------------------------------------------------------

    fast48 = bool(
        row[
            "fast_Risk48_deterioration"
        ]
    )


    fast12 = bool(
        row[
            "fast_Risk12_deterioration"
        ]
    )


    fast6 = bool(
        row[
            "fast_Risk6_deterioration"
        ]
    )


    fast_near_term = bool(
        row[
            "fast_near_term_deterioration"
        ]
    )


    # --------------------------------------------------------
    # Class history
    # --------------------------------------------------------

    worsening = bool(
        row[
            "is_class_worsening"
        ]
    )


    stable = bool(
        row[
            "is_prediction_stable"
        ]
    )


    limited_history = bool(
        row[
            "limited_history"
        ]
    )


    # ========================================================
    # EXPLANATION EVIDENCE
    #
    # Descriptive only.
    # Does NOT determine final action.
    # ========================================================

    evidence_score = 0

    evidence = []


    if cls == 4:

        evidence_score += 3

        evidence.append(
            "current model state is C4"
        )

    elif cls == 3:

        evidence_score += 2

        evidence.append(
            "current model state is C3"
        )

    elif cls == 2:

        evidence_score += 1

        evidence.append(
            "current model state is C2"
        )


    if risk6_pct >= 95:

        evidence_score += 2

        evidence.append(
            "Risk6 is in the top 5% of the fleet"
        )

    elif risk12_pct >= 90:

        evidence_score += 1

        evidence.append(
            "Risk12 is in the top 10% of the fleet"
        )


    if persistent6:

        evidence_score += 2

        evidence.append(
            "Risk6 has remained high in at least "
            "3 of the last 5 observations"
        )

    elif persistent12:

        evidence_score += 1

        evidence.append(
            "Risk12 has remained high in at least "
            "3 of the last 5 observations"
        )


    if fast6:

        evidence_score += 2

        evidence.append(
            "Risk6 deterioration is within the "
            "fastest 10% of the fleet"
        )

    elif fast12:

        evidence_score += 1

        evidence.append(
            "Risk12 deterioration is within the "
            "fastest 10% of the fleet"
        )


    if worsening:

        evidence_score += 1

        evidence.append(
            "recent predicted repair state is worsening"
        )


    if stable:

        evidence_score += 1

        evidence.append(
            "recent model state is relatively stable"
        )


    # ========================================================
    # TR0 — MANUAL REVIEW
    #
    # Elevated current prediction but insufficient history.
    # ========================================================

    if (
        limited_history
        and cls >= 3
    ):

        rule_id = "TR0"

        priority = "REVIEW"

        action = "MANUAL_REVIEW"

        reason = (
            "The current model indicates elevated repair "
            "proximity, but fewer than three historical "
            "snapshots are available. Human review is "
            "recommended before escalation."
        )


    # ========================================================
    # TR1 — PRIORITY INSPECT
    #
    # Strict critical condition:
    #
    # C4
    # +
    # Risk6 top 5%
    # +
    # persistent Risk6 OR unusually rapid deterioration
    # ========================================================

    elif (
        cls == 4
        and risk6_pct >= 95
        and (
            persistent6
            or fast_near_term
        )
    ):

        rule_id = "TR1"

        priority = "CRITICAL"

        action = "PRIORITY_INSPECT"

        reason = (
            "The vehicle is currently classified as C4 and "
            "is within the highest 5% of fleet Risk6 scores. "
            "The urgent signal is supported by persistent "
            "near-term risk or unusually rapid deterioration."
        )


    # ========================================================
    # TR2 — INSPECT
    #
    # Current C3/C4 plus supporting evidence.
    # ========================================================

    elif (
        cls >= 3
        and (
            risk12_pct >= 90
            or persistent12
            or fast_near_term
            or worsening
        )
    ):

        rule_id = "TR2"

        priority = "HIGH"

        action = "INSPECT"

        reason = (
            "The current model indicates near-term repair "
            "proximity and at least one additional historical "
            "or fleet-relative risk signal supports inspection."
        )


    # ========================================================
    # TR3 — TEMPORAL OVERRIDE
    #
    # Current class < C3,
    # but longitudinal Risk6 evidence is unusually strong.
    # ========================================================

    elif (
        cls < 3
        and risk6_pct >= 95
        and persistent6
        and fast6
    ):

        rule_id = "TR3"

        priority = "HIGH"

        action = "INSPECT"

        reason = (
            "The hard classifier does not indicate C3/C4, "
            "but Risk6 is in the fleet's highest 5%, has "
            "remained persistently high, and is deteriorating "
            "at an unusually fast rate. Temporal evidence "
            "therefore overrides the lower hard class."
        )


    # ========================================================
    # TR4 — ENHANCED MONITOR
    # ========================================================

    elif (
        cls == 2
        or (
            risk12_pct >= 90
            and (
                persistent12
                or fast12
            )
        )
    ):

        rule_id = "TR4"

        priority = "MEDIUM"

        action = "ENHANCED_MONITOR"

        reason = (
            "The vehicle shows elevated medium-term repair "
            "proximity. Historical persistence or relatively "
            "rapid Risk12 deterioration supports enhanced "
            "monitoring."
        )


    # ========================================================
    # TR5 — MONITOR
    # ========================================================

    elif (
        cls == 1
        or (
            risk48_pct >= 80
            and (
                persistent48
                or fast48
            )
        )
    ):

        rule_id = "TR5"

        priority = "LOW"

        action = "MONITOR"

        reason = (
            "An early repair-risk signal is present and is "
            "supported by historical persistence or faster "
            "than typical Risk48 deterioration."
        )


    # ========================================================
    # TR6 — CONTINUE
    # ========================================================

    else:

        rule_id = "TR6"

        priority = "NORMAL"

        action = "CONTINUE"

        reason = (
            "Current prediction, fleet-relative risk and "
            "historical evidence do not jointly support a "
            "higher maintenance-priority action."
        )


    # --------------------------------------------------------
    # Explanation text
    # --------------------------------------------------------

    if len(
        evidence
    ) == 0:

        evidence_text = (
            "No strong elevated-risk evidence detected"
        )

    else:

        evidence_text = "; ".join(
            evidence
        )


    # --------------------------------------------------------
    # Transparent rule trace
    # --------------------------------------------------------

    rule_trace = (

        f"{rule_id} | "

        f"class=C{cls} | "

        f"confidence_score={confidence:.3f} | "

        f"Risk6_pct={risk6_pct:.1f} | "

        f"Risk12_pct={risk12_pct:.1f} | "

        f"Risk48_pct={risk48_pct:.1f} | "

        f"Risk6_persistence="
        f"{row['Risk6_persistence']:.2f} | "

        f"Risk12_persistence="
        f"{row['Risk12_persistence']:.2f} | "

        f"Risk6_deterioration_pct="
        f"{row['Risk6_trend_percentile']:.1f} | "

        f"Risk12_deterioration_pct="
        f"{row['Risk12_trend_percentile']:.1f} | "

        f"class_stability="
        f"{row['class_stability']:.2f} | "

        f"class_worsening={worsening} | "

        f"action={action}"
    )


    return pd.Series({

        "evidence_score":
            evidence_score,

        "evidence":
            evidence_text,

        "rule_id":
            rule_id,

        "priority":
            priority,

        "action":
            action,

        "reason":
            reason,

        "rule_trace":
            rule_trace
    })


reasoning_results = latest.apply(
    temporal_rule_engine,
    axis=1
)


system_df = pd.concat(
    [
        latest,
        reasoning_results
    ],
    axis=1
)


# ============================================================
# 17. DISPLAY FIELDS
# ============================================================

system_df[
    "predicted_state"
] = (
    system_df[
        "predicted_class"
    ]
    .map(
        CLASS_NAMES
    )
)


system_df[
    "state_description"
] = (
    system_df[
        "predicted_class"
    ]
    .map(
        CLASS_DESCRIPTIONS
    )
)


# ============================================================
# 18. EVALUATE PREDICTIVE MODEL
# ============================================================

print(
    "\n[11/15] Evaluating predictive model..."
)


evaluation = system_df.merge(
    labels,
    on="vehicle_id",
    how="inner",
    validate="one_to_one"
)


y_true = (
    evaluation[
        "class_label"
    ]
    .astype(int)
    .to_numpy()
)


y_pred = (
    evaluation[
        "predicted_class"
    ]
    .astype(int)
    .to_numpy()
)


accuracy = accuracy_score(
    y_true,
    y_pred
)


macro_precision = precision_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="weighted",
    zero_division=0
)


report = classification_report(
    y_true,
    y_pred,
    labels=CLASS_IDS,
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


print("\n" + "=" * 80)
print("FINAL CLASSIFICATION REPORT")
print("=" * 80)


print(
    report
)


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Macro Precision:",
    round(
        macro_precision,
        4
    )
)


print(
    "Macro Recall:",
    round(
        macro_recall,
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


with open(
    OUTPUT_DIR
    / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report
    )


# ============================================================
# 19. CONFUSION MATRICES
# ============================================================

print(
    "\n[12/15] Creating confusion matrices..."
)


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_IDS
)


fig, ax = plt.subplots(
    figsize=(
        8,
        7
    )
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)


display.plot(
    ax=ax,
    values_format="d",
    colorbar=False
)


ax.set_title(
    "FleetCare AI - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    PLOT_DIR
    / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# Normalized confusion matrix
# ------------------------------------------------------------

cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    normalize="true"
)


fig, ax = plt.subplots(
    figsize=(
        8,
        7
    )
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)


display.plot(
    ax=ax,
    values_format=".2f",
    colorbar=False
)


ax.set_title(
    "FleetCare AI - Normalized Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    PLOT_DIR
    / "confusion_matrix_normalized.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 20. RISK RANKING EVALUATION
# ============================================================

print(
    "\n[13/15] Evaluating risk ranking..."
)


risk_metrics_rows = []


fig, ax = plt.subplots(
    figsize=(
        9,
        7
    )
)


for (
    horizon,
    positive_classes
) in HORIZONS.items():


    binary = np.isin(
        y_true,
        positive_classes
    ).astype(int)


    scores = (
        evaluation[
            horizon
        ]
        .to_numpy()
    )


    ap = average_precision_score(
        binary,
        scores
    )


    auc = roc_auc_score(
        binary,
        scores
    )


    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            binary,
            scores
        )
    )


    ax.plot(
        recall_curve,
        precision_curve,
        label=(
            f"{horizon} "
            f"(AP={ap:.3f})"
        )
    )


    # --------------------------------------------------------
    # Top 5%
    # --------------------------------------------------------

    k5 = max(
        1,
        int(
            np.ceil(
                len(scores)
                * 0.05
            )
        )
    )


    top5 = np.argsort(
        -scores,
        kind="stable"
    )[:k5]


    top5_recall = (
        binary[
            top5
        ].sum()
        /
        binary.sum()
    )


    top5_precision = (
        binary[
            top5
        ].mean()
    )


    # --------------------------------------------------------
    # Top 10%
    # --------------------------------------------------------

    k10 = max(
        1,
        int(
            np.ceil(
                len(scores)
                * 0.10
            )
        )
    )


    top10 = np.argsort(
        -scores,
        kind="stable"
    )[:k10]


    top10_recall = (
        binary[
            top10
        ].sum()
        /
        binary.sum()
    )


    top10_precision = (
        binary[
            top10
        ].mean()
    )


    risk_metrics_rows.append({

        "Horizon":
            horizon,

        "AveragePrecision":
            float(
                ap
            ),

        "ROCAUC":
            float(
                auc
            ),

        "Top5Recall":
            float(
                top5_recall
            ),

        "Top5Precision":
            float(
                top5_precision
            ),

        "Top10Recall":
            float(
                top10_recall
            ),

        "Top10Precision":
            float(
                top10_precision
            )
    })


ax.set_xlabel(
    "Recall"
)


ax.set_ylabel(
    "Precision"
)


ax.set_title(
    "FleetCare AI - Precision-Recall Curves"
)


ax.legend()


ax.grid(
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    PLOT_DIR
    / "risk_precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


risk_metrics_df = pd.DataFrame(
    risk_metrics_rows
)


risk_metrics_df.to_csv(
    OUTPUT_DIR
    / "risk_metrics.csv",
    index=False
)


print(
    risk_metrics_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. RULE ENGINE EVALUATION
# ============================================================

print(
    "\n[14/15] Evaluating temporal reasoning engine..."
)


rule_evaluation_rows = []


intervention_definitions = {

    "Priority Inspect Only": [
        "PRIORITY_INSPECT"
    ],

    "Inspect or Higher": [
        "INSPECT",
        "PRIORITY_INSPECT"
    ],

    "Manual Review / Inspect or Higher": [
        "MANUAL_REVIEW",
        "INSPECT",
        "PRIORITY_INSPECT"
    ],

    "Enhanced Monitor or Higher": [
        "ENHANCED_MONITOR",
        "MANUAL_REVIEW",
        "INSPECT",
        "PRIORITY_INSPECT"
    ],
}


for (
    intervention_name,
    actions
) in intervention_definitions.items():


    selected = (
        evaluation[
            "action"
        ]
        .isin(
            actions
        )
    )


    selected_count = int(
        selected.sum()
    )


    # --------------------------------------------------------
    # C4
    # --------------------------------------------------------

    actual_c4 = (
        evaluation[
            "class_label"
        ]
        == 4
    )


    c4_found = int(
        (
            selected
            &
            actual_c4
        ).sum()
    )


    total_c4 = int(
        actual_c4.sum()
    )


    c4_recall = (
        c4_found
        /
        total_c4
        if total_c4 > 0
        else np.nan
    )


    c4_precision = (
        c4_found
        /
        selected_count
        if selected_count > 0
        else np.nan
    )


    # --------------------------------------------------------
    # Repair within 48
    # --------------------------------------------------------

    actual_risk48 = (
        evaluation[
            "class_label"
        ]
        .isin(
            [
                1,
                2,
                3,
                4
            ]
        )
    )


    risk48_found = int(
        (
            selected
            &
            actual_risk48
        ).sum()
    )


    total_risk48 = int(
        actual_risk48.sum()
    )


    risk48_recall = (
        risk48_found
        /
        total_risk48
        if total_risk48 > 0
        else np.nan
    )


    risk48_precision = (
        risk48_found
        /
        selected_count
        if selected_count > 0
        else np.nan
    )


    rule_evaluation_rows.append({

        "Intervention":
            intervention_name,

        "VehiclesSelected":
            selected_count,

        "FleetPercent":
            selected_count
            /
            len(
                evaluation
            ),

        "C4Found":
            c4_found,

        "C4Recall":
            c4_recall,

        "C4Precision":
            c4_precision,

        "Risk48Found":
            risk48_found,

        "Risk48Recall":
            risk48_recall,

        "Risk48Precision":
            risk48_precision
    })


rule_evaluation_df = pd.DataFrame(
    rule_evaluation_rows
)


print("\n" + "=" * 80)
print("RULE ENGINE EVALUATION")
print("=" * 80)


print(
    rule_evaluation_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


rule_evaluation_df.to_csv(
    OUTPUT_DIR
    / "rule_engine_evaluation.csv",
    index=False
)


# ============================================================
# 22. SORT FOR FRONTEND
# ============================================================

PRIORITY_RANK = {

    "CRITICAL": 5,

    "HIGH": 4,

    "REVIEW": 3,

    "MEDIUM": 2,

    "LOW": 1,

    "NORMAL": 0,
}


system_df[
    "priority_rank"
] = (
    system_df[
        "priority"
    ]
    .map(
        PRIORITY_RANK
    )
)


system_df = (
    system_df
    .sort_values(
        [
            "priority_rank",
            "Risk6_percentile",
            "Risk6_trend_percentile",
            "Risk12_percentile",
            "Risk48_percentile"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 23. FLEET SUMMARY FOR FRONTEND
# ============================================================

fleet_summary = {

    "total_vehicles":
        int(
            len(
                system_df
            )
        ),

    "predicted_classes": {

        f"C{x}":
            int(
                (
                    system_df[
                        "predicted_class"
                    ]
                    == x
                ).sum()
            )

        for x in range(
            5
        )
    },

    "actions": {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value
        ) in (
            system_df[
                "action"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "priorities": {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value
        ) in (
            system_df[
                "priority"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "history_quality": {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value
        ) in (
            system_df[
                "history_quality"
            ]
            .value_counts()
            .to_dict()
            .items()
        )
    },

    "temporal_evidence": {

        "ordinary_rising_risk":
            int(
                system_df[
                    "is_rising"
                ].sum()
            ),

        "fast_Risk48_deterioration":
            int(
                system_df[
                    "fast_Risk48_deterioration"
                ].sum()
            ),

        "fast_Risk12_deterioration":
            int(
                system_df[
                    "fast_Risk12_deterioration"
                ].sum()
            ),

        "fast_Risk6_deterioration":
            int(
                system_df[
                    "fast_Risk6_deterioration"
                ].sum()
            ),

        "persistent_Risk6":
            int(
                system_df[
                    "is_risk6_persistent"
                ].sum()
            ),

        "worsening_class":
            int(
                system_df[
                    "is_class_worsening"
                ].sum()
            ),

        "limited_history":
            int(
                system_df[
                    "limited_history"
                ].sum()
            )
    },

    "model_metrics": {

        "accuracy":
            float(
                accuracy
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "weighted_f1":
            float(
                weighted_f1
            )
    }
}


# ============================================================
# 24. FRONTEND VEHICLE RECORDS
# ============================================================

print(
    "\n[15/15] Generating frontend data..."
)


frontend_records = []


for _, row in system_df.iterrows():


    vehicle_id = row[
        "vehicle_id"
    ]


    if hasattr(
        vehicle_id,
        "item"
    ):

        vehicle_id = (
            vehicle_id
            .item()
        )


    record = {

        # ====================================================
        # ID
        # ====================================================

        "vehicle_id":
            vehicle_id,


        # ====================================================
        # CURRENT ML PREDICTION
        # ====================================================

        "current_prediction": {

            "class_id":
                int(
                    row[
                        "predicted_class"
                    ]
                ),

            "state":
                row[
                    "predicted_state"
                ],

            "description":
                row[
                    "state_description"
                ],

            # Model confidence score,
            # NOT calibrated real-world probability.
            "confidence_score":
                round(
                    float(
                        row[
                            "confidence"
                        ]
                    ),
                    6
                ),

            "class_scores": {

                "C0":
                    round(
                        float(
                            row[
                                "P_C0"
                            ]
                        ),
                        6
                    ),

                "C1":
                    round(
                        float(
                            row[
                                "P_C1"
                            ]
                        ),
                        6
                    ),

                "C2":
                    round(
                        float(
                            row[
                                "P_C2"
                            ]
                        ),
                        6
                    ),

                "C3":
                    round(
                        float(
                            row[
                                "P_C3"
                            ]
                        ),
                        6
                    ),

                "C4":
                    round(
                        float(
                            row[
                                "P_C4"
                            ]
                        ),
                        6
                    )
            }
        },


        # ====================================================
        # CURRENT RISK
        # ====================================================

        "risk": {

            "Risk48": {

                "score":
                    round(
                        float(
                            row[
                                "Risk48"
                            ]
                        ),
                        6
                    ),

                "percentile":
                    round(
                        float(
                            row[
                                "Risk48_percentile"
                            ]
                        ),
                        2
                    )
            },

            "Risk24": {

                "score":
                    round(
                        float(
                            row[
                                "Risk24"
                            ]
                        ),
                        6
                    ),

                "percentile":
                    round(
                        float(
                            row[
                                "Risk24_percentile"
                            ]
                        ),
                        2
                    )
            },

            "Risk12": {

                "score":
                    round(
                        float(
                            row[
                                "Risk12"
                            ]
                        ),
                        6
                    ),

                "percentile":
                    round(
                        float(
                            row[
                                "Risk12_percentile"
                            ]
                        ),
                        2
                    )
            },

            "Risk6": {

                "score":
                    round(
                        float(
                            row[
                                "Risk6"
                            ]
                        ),
                        6
                    ),

                "percentile":
                    round(
                        float(
                            row[
                                "Risk6_percentile"
                            ]
                        ),
                        2
                    )
            }
        },


        # ====================================================
        # HISTORICAL REASONING
        # ====================================================

        "history": {

            "readout_count":
                int(
                    row[
                        "history_count"
                    ]
                ),

            "quality":
                row[
                    "history_quality"
                ],

            "recent_class_history":
                row[
                    "recent_class_history"
                ],

            "class_stability":
                round(
                    float(
                        row[
                            "class_stability"
                        ]
                    ),
                    4
                ),

            "class_worsening":
                bool(
                    row[
                        "is_class_worsening"
                    ]
                ),

            "descriptive_trend":
                row[
                    "trend_direction"
                ],


            # ------------------------------------------------
            # Persistence
            # ------------------------------------------------

            "persistence": {

                "Risk48":
                    round(
                        float(
                            row[
                                "Risk48_persistence"
                            ]
                        ),
                        4
                    ),

                "Risk12":
                    round(
                        float(
                            row[
                                "Risk12_persistence"
                            ]
                        ),
                        4
                    ),

                "Risk6":
                    round(
                        float(
                            row[
                                "Risk6_persistence"
                            ]
                        ),
                        4
                    )
            },


            # ------------------------------------------------
            # Deterioration
            # ------------------------------------------------

            "deterioration": {

                "Risk48_percentile":
                    round(
                        float(
                            row[
                                "Risk48_trend_percentile"
                            ]
                        ),
                        2
                    ),

                "Risk12_percentile":
                    round(
                        float(
                            row[
                                "Risk12_trend_percentile"
                            ]
                        ),
                        2
                    ),

                "Risk6_percentile":
                    round(
                        float(
                            row[
                                "Risk6_trend_percentile"
                            ]
                        ),
                        2
                    ),

                "fast_Risk48":
                    bool(
                        row[
                            "fast_Risk48_deterioration"
                        ]
                    ),

                "fast_Risk12":
                    bool(
                        row[
                            "fast_Risk12_deterioration"
                        ]
                    ),

                "fast_Risk6":
                    bool(
                        row[
                            "fast_Risk6_deterioration"
                        ]
                    )
            }
        },


        # ====================================================
        # RULE-BASED REASONING
        # ====================================================

        "reasoning": {

            "evidence_score":
                int(
                    row[
                        "evidence_score"
                    ]
                ),

            "evidence":
                row[
                    "evidence"
                ],

            "rule_id":
                row[
                    "rule_id"
                ],

            "priority":
                row[
                    "priority"
                ],

            "action":
                row[
                    "action"
                ],

            "reason":
                row[
                    "reason"
                ],

            "trace":
                row[
                    "rule_trace"
                ]
        }
    }


    frontend_records.append(
        record
    )


# ============================================================
# 25. SAVE FRONTEND / SYSTEM OUTPUTS
# ============================================================

system_df.to_csv(
    OUTPUT_DIR
    / "frontend_vehicle_data.csv",
    index=False
)


with open(
    OUTPUT_DIR
    / "frontend_vehicle_data.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        frontend_records,
        file,
        indent=2,
        ensure_ascii=False
    )


with open(
    OUTPUT_DIR
    / "fleet_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        fleet_summary,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 26. SAVE HISTORICAL RISK TRAJECTORIES
# ============================================================
#
# Used by frontend for line charts.
# ============================================================

history_output_columns = [

    "vehicle_id",

    "time_step",

    "predicted_class",

    "confidence",

    "Risk48",

    "Risk24",

    "Risk12",

    "Risk6",

    "Risk48_percentile",

    "Risk12_percentile",

    "Risk6_percentile",
]


history[
    history_output_columns
].to_csv(
    OUTPUT_DIR
    / "vehicle_risk_history.csv",
    index=False
)


# ============================================================
# 27. SAVE REASONING DATA SEPARATELY
# ============================================================

reasoning_columns = [

    "vehicle_id",

    "predicted_class",

    "confidence",

    "Risk48",

    "Risk24",

    "Risk12",

    "Risk6",

    "Risk48_percentile",

    "Risk24_percentile",

    "Risk12_percentile",

    "Risk6_percentile",

    "history_count",

    "history_quality",

    "recent_class_history",

    "class_stability",

    "class_change",

    "Risk48_persistence",

    "Risk12_persistence",

    "Risk6_persistence",

    "Risk48_trend",

    "Risk12_trend",

    "Risk6_trend",

    "Risk48_trend_percentile",

    "Risk12_trend_percentile",

    "Risk6_trend_percentile",

    "fast_Risk48_deterioration",

    "fast_Risk12_deterioration",

    "fast_Risk6_deterioration",

    "evidence_score",

    "evidence",

    "rule_id",

    "priority",

    "action",

    "reason",

    "rule_trace"
]


system_df[
    reasoning_columns
].to_csv(
    OUTPUT_DIR
    / "reasoning_evidence.csv",
    index=False
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEMPORAL REASONING SUMMARY")
print("=" * 80)


print(
    "\nMaintenance actions:"
)


print(
    system_df[
        "action"
    ]
    .value_counts()
)


print(
    "\nRule distribution:"
)


print(
    system_df[
        "rule_id"
    ]
    .value_counts()
)


print(
    "\nHistory quality:"
)


print(
    system_df[
        "history_quality"
    ]
    .value_counts()
)


print(
    "\nOrdinary trend direction "
    "(descriptive only):"
)


print(
    system_df[
        "trend_direction"
    ]
    .value_counts()
)


print(
    "\nRelative deterioration:"
)


print(
    "Fast Risk48:",
    int(
        system_df[
            "fast_Risk48_deterioration"
        ].sum()
    )
)


print(
    "Fast Risk12:",
    int(
        system_df[
            "fast_Risk12_deterioration"
        ].sum()
    )
)


print(
    "Fast Risk6:",
    int(
        system_df[
            "fast_Risk6_deterioration"
        ].sum()
    )
)


print(
    "\nPersistent risk:"
)


print(
    "Persistent Risk48:",
    int(
        system_df[
            "is_risk48_persistent"
        ].sum()
    )
)


print(
    "Persistent Risk12:",
    int(
        system_df[
            "is_risk12_persistent"
        ].sum()
    )
)


print(
    "Persistent Risk6:",
    int(
        system_df[
            "is_risk6_persistent"
        ].sum()
    )
)


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)


output_files = [

    OUTPUT_DIR
    / "frontend_vehicle_data.csv",

    OUTPUT_DIR
    / "frontend_vehicle_data.json",

    OUTPUT_DIR
    / "fleet_summary.json",

    OUTPUT_DIR
    / "vehicle_risk_history.csv",

    OUTPUT_DIR
    / "reasoning_evidence.csv",

    OUTPUT_DIR
    / "classification_report.txt",

    OUTPUT_DIR
    / "risk_metrics.csv",

    OUTPUT_DIR
    / "rule_engine_evaluation.csv",

    PLOT_DIR
    / "confusion_matrix.png",

    PLOT_DIR
    / "confusion_matrix_normalized.png",

    PLOT_DIR
    / "risk_precision_recall_curves.png",
]


for path in output_files:

    print(
        path
    )


print("\n" + "=" * 80)
print("FLEETCARE FINAL SYSTEM COMPLETE")
print("=" * 80)