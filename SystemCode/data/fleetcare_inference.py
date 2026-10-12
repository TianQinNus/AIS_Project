from pathlib import Path
import gc
import joblib
import sqlite3
import numpy as np
import pandas as pd

CLASS_NAMES = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}
CLASS_DESCRIPTIONS = {
    0: "No repair predicted within 48 operational units",
    1: "Repair proximity: 24-48 operational units",
    2: "Repair proximity: 12-24 operational units",
    3: "Repair proximity: 6-12 operational units",
    4: "Repair proximity: 0-6 operational units",
}
PRIORITY_RANK = {"CRITICAL": 5, "HIGH": 4, "REVIEW": 3, "MEDIUM": 2, "LOW": 1, "NORMAL": 0}


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

class FleetCarePredictor:
    """Production-style FleetCare inference pipeline.

    Input:
        operational: historical operational readouts for the fleet
        specifications: one specification row per vehicle

    Output:
        one flat row per vehicle, ready to save into a SQL table.
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "fleetcare_vehicle_aware_history_xgb.joblib")
        self.feature_columns = joblib.load(self.model_dir / "fleetcare_history_feature_columns.joblib")
        self.raw_feature_columns = joblib.load(self.model_dir / "fleetcare_raw_feature_columns.joblib")

    def predict(self, operational, specifications):
        operational = operational.copy()
        specifications = specifications.copy()
        self._validate_inputs(operational, specifications)

        history = create_history_features(operational, self.raw_feature_columns)
        history = history.merge(
            specifications,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
        history = history.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

        x_history = (
            history.drop(columns=["vehicle_id"], errors="ignore")
            .reindex(columns=self.feature_columns)
        )
        proba = self.model.predict_proba(x_history)

        # ============================================================
        # Save latest raw model inputs for SHAP
        # ============================================================

        latest_indices = (
            history
            .groupby(
                "vehicle_id",
                sort=False,
            )
            .tail(1)
            .index
        )


        self._latest_model_input = (
            x_history
            .loc[
                latest_indices
            ]
            .copy()
        )


        self._latest_vehicle_ids = (
            history
            .loc[
                latest_indices,
                "vehicle_id"
            ]
            .astype(int)
            .to_numpy()
        )

        history["predicted_class"] = np.argmax(proba, axis=1)
        history["confidence"] = np.max(proba, axis=1)
        for i in range(5):
            history[f"P_C{i}"] = proba[:, i]
        history["Risk48"] = proba[:, 1:].sum(axis=1)
        history["Risk24"] = proba[:, 2:].sum(axis=1)
        history["Risk12"] = proba[:, 3:].sum(axis=1)
        history["Risk6"] = proba[:, 4]

        temporal_rows = []
        for vehicle_id, group in history.groupby("vehicle_id", sort=False):
            row = build_temporal_reasoning(group)
            row["vehicle_id"] = vehicle_id
            temporal_rows.append(row)
        temporal = pd.DataFrame(temporal_rows)

        latest = (
            history.groupby("vehicle_id", sort=False).tail(1)
            .sort_values("vehicle_id").reset_index(drop=True)
        )
        latest = latest[[
            "vehicle_id", "time_step", "predicted_class", "confidence",
            "P_C0", "P_C1", "P_C2", "P_C3", "P_C4",
            "Risk48", "Risk24", "Risk12", "Risk6",
        ]]
        latest = latest.merge(
            temporal,
            on=["vehicle_id", "time_step"],
            how="left",
            validate="one_to_one",
        )

        # Current fleet-relative risk percentiles.
        for col in ["Risk48", "Risk24", "Risk12", "Risk6"]:
            latest[f"{col}_percentile"] = latest[col].rank(pct=True, method="average") * 100

        # Historical percentiles used by the frozen persistence definition.
        for col in ["Risk48", "Risk12", "Risk6"]:
            history[f"{col}_percentile"] = history[col].rank(pct=True, method="average") * 100

        persistence_rows = []
        for vehicle_id, group in history.groupby("vehicle_id", sort=False):
            row = calculate_persistence(group)
            row["vehicle_id"] = vehicle_id
            persistence_rows.append(row)
        persistence = pd.DataFrame(persistence_rows)
        latest = latest.merge(persistence, on="vehicle_id", how="left", validate="one_to_one")

        # Relative deterioration.
        for col in ["Risk48_trend", "Risk12_trend", "Risk6_trend"]:
            latest[f"{col}_percentile"] = latest[col].rank(pct=True, method="average") * 100

        latest["fast_Risk48_deterioration"] = latest["Risk48_trend_percentile"] >= 90
        latest["fast_Risk12_deterioration"] = latest["Risk12_trend_percentile"] >= 90
        latest["fast_Risk6_deterioration"] = latest["Risk6_trend_percentile"] >= 90
        latest["fast_near_term_deterioration"] = (
            latest["fast_Risk6_deterioration"] | latest["fast_Risk12_deterioration"]
        )

        # Frozen reasoning flags.
        latest["is_class_worsening"] = latest["class_change"] > 0
        latest["is_prediction_stable"] = latest["class_stability"] >= 0.60
        latest["is_risk48_persistent"] = latest["Risk48_persistence"] >= 0.60
        latest["is_risk12_persistent"] = latest["Risk12_persistence"] >= 0.60
        latest["is_risk6_persistent"] = latest["Risk6_persistence"] >= 0.60
        latest["limited_history"] = latest["history_count"] < 3
        latest["is_rising"] = latest["trend_direction"].eq("RISING")

        reasoning = latest.apply(temporal_rule_engine, axis=1)
        output = pd.concat([latest, reasoning], axis=1)
        output["predicted_state"] = output["predicted_class"].map(CLASS_NAMES)
        output["state_description"] = output["predicted_class"].map(CLASS_DESCRIPTIONS)
        output["priority_rank"] = output["priority"].map(PRIORITY_RANK)
        output["priority_score"] = output["priority_rank"] / 5 * 100

        # Frozen hybrid ranking used in the evaluated final system.
        output["hybrid_ranking_score"] = (
            0.50 * output["Risk6_percentile"]
            + 0.20 * output["priority_score"]
            + 0.15 * output["Risk6_trend_percentile"]
            + 0.10 * output["Risk6_persistence"] * 100
            + 0.05 * output["Risk12_percentile"]
        )

        # Keep the exact latest model inputs for SHAP.
        latest_model_rows = (
            history.groupby("vehicle_id", sort=False).tail(1)
            .sort_values("vehicle_id").reset_index(drop=True)
        )
        self._latest_shap_vehicle_ids = latest_model_rows["vehicle_id"].astype(int).to_numpy()
        self._latest_shap_x = (
            latest_model_rows.drop(columns=["vehicle_id"], errors="ignore")
            .reindex(columns=self.feature_columns)
        )

        output = output.sort_values(
            "hybrid_ranking_score", ascending=False, kind="stable"
        ).reset_index(drop=True)
        output.insert(0, "ranking", np.arange(1, len(output) + 1))
        return self._table_columns(output)

    def predict_from_csv(self, operational_path, specifications_path):
        return self.predict(
            pd.read_csv(operational_path),
            pd.read_csv(specifications_path),
        )

    @staticmethod
    def save_to_sqlite(df, database_path, table_name="fleetcare_predictions", if_exists="replace"):
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path) as connection:
            df.to_sql(table_name, connection, if_exists=if_exists, index=False)
        return database_path

    @staticmethod
    def _validate_inputs(operational, specifications):
        required_op = {"vehicle_id", "time_step"}
        missing_op = required_op - set(operational.columns)
        if missing_op:
            raise ValueError(f"Operational data missing columns: {sorted(missing_op)}")
        if "vehicle_id" not in specifications.columns:
            raise ValueError("Specifications data missing column: vehicle_id")
        if specifications["vehicle_id"].duplicated().any():
            raise ValueError("Specifications must contain one row per vehicle_id")

    @staticmethod
    def _table_columns(df):
        # Flat, database-friendly final output. Keep trace/reason text for explainability.
        columns = [
            "ranking", "vehicle_id", "time_step",
            "predicted_class", "predicted_state", "state_description", "confidence",
            "P_C0", "P_C1", "P_C2", "P_C3", "P_C4",
            "Risk48", "Risk24", "Risk12", "Risk6", "trend_direction",
            "Risk48_percentile", "Risk24_percentile", "Risk12_percentile", "Risk6_percentile",
            "history_count", "history_quality", "recent_class_history",
            "class_stability", "class_change", "is_class_worsening",
            "Risk48_persistence", "Risk12_persistence", "Risk6_persistence",
            "is_risk48_persistent", "is_risk12_persistent", "is_risk6_persistent",
            "Risk48_trend", "Risk12_trend", "Risk6_trend",
            "Risk48_trend_percentile", "Risk12_trend_percentile", "Risk6_trend_percentile",
            "fast_Risk48_deterioration", "fast_Risk12_deterioration", "fast_Risk6_deterioration",
            "evidence_score", "evidence", "rule_id", "priority", "action", "reason", "rule_trace",
            "priority_rank", "priority_score", "hybrid_ranking_score",
        ]
        return df[[c for c in columns if c in df.columns]].copy()



def _create_postgres_engine(username, password, host, port, database):
    from database import create_postgres_engine
    return create_postgres_engine(username=username, password=password, host=host, port=port, database=database)


def _load_fleet_from_postgres(engine):
    operational = pd.read_sql(
        "SELECT * FROM public.truck_readout ORDER BY vehicle_id, time_step",
        engine,
    ).drop(columns=["readout_id", "collected_at"], errors="ignore")

    specifications = pd.read_sql(
        """
        SELECT vehicle_id, spec_0, spec_1, spec_2, spec_3,
               spec_4, spec_5, spec_6, spec_7
        FROM public.truck
        """,
        engine,
    ).rename(columns={f"spec_{i}": f"Spec_{i}" for i in range(8)})

    if operational.empty:
        raise RuntimeError("truck_readout is empty")
    if specifications.empty:
        raise RuntimeError("truck is empty")

    print(f"Loaded {len(operational):,} readouts for {operational['vehicle_id'].nunique():,} vehicles")
    return operational, specifications

def _build_analysis_dataframe(result):
    """
    Convert full FleetCare inference output into the
    PostgreSQL `public.analysis` table schema.

    Persist the core fields and available risk/time/trace fields;
    other internal variables remain in the model output.
    """

    required_columns = [
        "vehicle_id",
        "predicted_state",
        "confidence",
        "Risk48_percentile",
        "Risk12_percentile",
        "Risk6",
        "Risk6_percentile",
        "history_quality",
        "Risk6_persistence",
        "Risk6_trend_percentile",
        "is_class_worsening",
        "rule_id",
        "priority",
        "action",
        "evidence",
        "reason",
    ]

    missing = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise RuntimeError(
            f"Prediction result missing required columns: {missing}"
        )

    analysis_df = pd.DataFrame({

        # ====================================================
        # Vehicle
        # ====================================================

        "vehicle_id":
            result["vehicle_id"].astype(int),

        # ====================================================
        # ML prediction
        # ====================================================

        "state":
            result["predicted_state"],

        "confidence_score":
            result["confidence"].round(6),

        # ====================================================
        # Risk
        # ====================================================

        "risk48_percentile":
            result["Risk48_percentile"].round(2),

        "risk12_percentile":
            result["Risk12_percentile"].round(2),

        "risk6_score":
            result["Risk6"].round(6),

        "risk6_percentile":
            result["Risk6_percentile"].round(2),

        # ====================================================
        # Temporal evidence
        # ====================================================

        "history_quality":
            result["history_quality"],

        "risk6_persistence":
            result["Risk6_persistence"].round(4),

        "risk6_deterioration_percentile":
            result["Risk6_trend_percentile"].round(2),

        "class_worsening":
            result["is_class_worsening"].astype(int),

        # ====================================================
        # Rule engine
        # ====================================================

        "rule_id":
            result["rule_id"],

        "priority":
            result["priority"],

        "action":
            result["action"],

        "evidence":
            result["evidence"],

        "reason":
            result["reason"],
    })

    extra_fields = {
        "Risk48":"risk48_score", "Risk24":"risk24_score", "Risk12":"risk12_score",
        "Risk24_percentile":"risk24_percentile", "time_step":"time_step",
        "history_count":"history_count", "rule_trace":"rule_trace",
        "hybrid_ranking_score":"hybrid_ranking_score", "priority_score":"priority_score",
    }
    for source, target in extra_fields.items():
        if source in result.columns:
            analysis_df[target] = result[source]

    return analysis_df
def _insert_analysis_and_get_ids(analysis_df, engine, table_name):
    # PostgreSQL RETURNING obtains exact IDs without timestamp-based recovery.
    from sqlalchemy import MetaData, Table
    db_df = analysis_df.copy()
    db_df["prediction_at"] = pd.Timestamp.now(tz="UTC").to_pydatetime()
    records = db_df.astype(object).where(pd.notna(db_df), None).to_dict("records")
    if not records:
        return pd.DataFrame(columns=["analysis_id", "vehicle_id", "prediction_at"])
    with engine.begin() as conn:
        table = Table(table_name, MetaData(), schema="public", autoload_with=conn)
        result = conn.execute(table.insert().returning(table.c.analysis_id, table.c.vehicle_id,
                             table.c.prediction_at, sort_by_parameter_order=True), records)
        rows = result.mappings().all()
        if len(rows) != len(records):
            raise RuntimeError("PostgreSQL insert did not return every analysis ID")
    return pd.DataFrame(rows)


def _calculate_top_shap(
    predictor,
    result,
    top_n=5,
):
    """
    Calculate Top-N SHAP explanations for each vehicle.

    predictor.model is a sklearn Pipeline:
        preprocessing -> XGBoost

    Therefore SHAP must explain the final XGBoost estimator
    using the transformed feature matrix.
    """

    import shap

    print(
        f"\nCalculating Top-{top_n} SHAP explanations..."
    )

    # ========================================================
    # 1. GET PIPELINE
    # ========================================================

    pipeline = predictor.model

    if not hasattr(pipeline, "steps"):
        raise TypeError(
            "Expected predictor.model to be a sklearn Pipeline, "
            f"but got {type(pipeline)}"
        )

    print(
        "Pipeline steps:",
        [name for name, _ in pipeline.steps]
    )


    # ========================================================
    # 2. SPLIT PREPROCESSOR AND XGBOOST
    #
    # Everything except last step = preprocessing
    # Last step = XGBoost
    # ========================================================

    preprocessor = pipeline[:-1]

    xgb_model = pipeline.steps[-1][1]

    print(
        "Final estimator:",
        type(xgb_model)
    )


    # ========================================================
    # 3. GET EXACT LATEST INPUT
    #
    # This should already have been stored during predict()
    # ========================================================

    if not hasattr(
        predictor,
        "_latest_model_input"
    ):
        raise RuntimeError(
            "predictor._latest_model_input not found. "
            "Run predictor.predict() before calculating SHAP."
        )

    if not hasattr(
        predictor,
        "_latest_vehicle_ids"
    ):
        raise RuntimeError(
            "predictor._latest_vehicle_ids not found."
        )

    X_latest_raw = (
        predictor
        ._latest_model_input
        .copy()
    )

    vehicle_ids = np.asarray(
        predictor._latest_vehicle_ids
    )


    print(
        "Raw latest input:",
        X_latest_raw.shape
    )


    # ========================================================
    # 4. APPLY PIPELINE PREPROCESSING
    # ========================================================

    X_transformed = (
        preprocessor
        .transform(
            X_latest_raw
        )
    )


    print(
        "Transformed SHAP input:",
        X_transformed.shape
    )


    # ========================================================
    # 5. GET TRANSFORMED FEATURE NAMES
    # ========================================================

    try:

        shap_feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception as exc:

        print(
            "Could not retrieve transformed feature names:",
            exc
        )

        shap_feature_names = [
            f"feature_{i}"
            for i in range(
                X_transformed.shape[1]
            )
        ]


    if (
        len(shap_feature_names)
        !=
        X_transformed.shape[1]
    ):

        print(
            "Feature-name count does not match "
            "transformed matrix. Using generic names."
        )

        shap_feature_names = [
            f"feature_{i}"
            for i in range(
                X_transformed.shape[1]
            )
        ]


    # Make names cleaner for frontend
    shap_feature_names = [
        str(name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        for name in shap_feature_names
    ]


    # ========================================================
    # 6. CREATE TREE EXPLAINER
    #
    # IMPORTANT:
    # Explain XGBoost, NOT sklearn Pipeline
    # ========================================================

    explainer = shap.TreeExplainer(
        xgb_model
    )


    # ========================================================
    # 7. CALCULATE SHAP
    # ========================================================

    shap_output = explainer(
        X_transformed
    )

    shap_values = (
        shap_output.values
    )


    print(
        "SHAP values shape:",
        np.asarray(
            shap_values
        ).shape
    )


    # ========================================================
    # 8. NORMALIZE MULTICLASS SHAPE
    #
    # New SHAP commonly:
    #   (samples, features, classes)
    #
    # Older SHAP may return:
    #   list[class] of (samples, features)
    # ========================================================

    if isinstance(
        shap_values,
        list
    ):

        # list of 5 arrays:
        #
        # [
        #   C0: samples x features,
        #   C1: samples x features,
        #   ...
        # ]
        #
        # Convert to:
        # samples x features x classes

        shap_values = np.stack(
            shap_values,
            axis=-1
        )


    else:

        shap_values = np.asarray(
            shap_values
        )


    # ========================================================
    # 9. CHECK SHAPE
    # ========================================================

    if shap_values.ndim == 2:

        # Binary / single-output fallback
        #
        # samples x features

        shap_values = (
            shap_values[
                :,
                :,
                None
            ]
        )


    elif shap_values.ndim == 3:

        # Could be:
        #
        # samples x features x classes
        #
        # OR
        #
        # samples x classes x features

        n_features = (
            X_transformed.shape[1]
        )

        if (
            shap_values.shape[1]
            ==
            n_features
        ):

            # Already correct
            pass

        elif (
            shap_values.shape[2]
            ==
            n_features
        ):

            shap_values = np.transpose(
                shap_values,
                (
                    0,
                    2,
                    1,
                )
            )

        else:

            raise RuntimeError(
                "Cannot determine SHAP feature dimension. "
                f"SHAP shape={shap_values.shape}, "
                f"expected features={n_features}"
            )


    else:

        raise RuntimeError(
            "Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )


    print(
        "Normalized SHAP shape:",
        shap_values.shape
    )


    # ========================================================
    # 10. MAP VEHICLE -> PREDICTED CLASS
    # ========================================================

    predicted_class_map = (
        result
        .set_index(
            "vehicle_id"
        )[
            "predicted_class"
        ]
        .astype(int)
        .to_dict()
    )


    # ========================================================
    # 11. BUILD TOP-N SHAP ROWS
    # ========================================================

    shap_rows = []


    for i, vehicle_id in enumerate(
        vehicle_ids
    ):

        vehicle_id = int(
            vehicle_id
        )


        predicted_class = int(
            predicted_class_map[
                vehicle_id
            ]
        )


        # ----------------------------------------------------
        # Select SHAP values for predicted class
        # ----------------------------------------------------

        if (
            shap_values.shape[2]
            ==
            1
        ):

            vehicle_shap = (
                shap_values[
                    i,
                    :,
                    0
                ]
            )

        else:

            vehicle_shap = (
                shap_values[
                    i,
                    :,
                    predicted_class
                ]
            )


        # ----------------------------------------------------
        # Top absolute SHAP contributions
        # ----------------------------------------------------

        top_indices = (
            np.argsort(
                np.abs(
                    vehicle_shap
                )
            )[
                -top_n:
            ][::-1]
        )


        # ----------------------------------------------------
        # Get transformed feature values
        # ----------------------------------------------------

        if hasattr(
            X_transformed,
            "toarray"
        ):

            transformed_row = (
                X_transformed[
                    i
                ]
                .toarray()
                .ravel()
            )

        else:

            transformed_row = (
                np.asarray(
                    X_transformed[
                        i
                    ]
                )
                .ravel()
            )


        # ----------------------------------------------------
        # Save Top N
        # ----------------------------------------------------

        for (
            contribution_rank,
            feature_index
        ) in enumerate(
            top_indices,
            start=1,
        ):

            feature_name = (
                shap_feature_names[
                    feature_index
                ]
            )


            feature_value = (
                transformed_row[
                    feature_index
                ]
            )


            shap_value = (
                vehicle_shap[
                    feature_index
                ]
            )


            if pd.isna(
                feature_value
            ):

                feature_value_db = None

            else:

                feature_value_db = float(
                    feature_value
                )


            shap_rows.append({

                "vehicle_id":
                    vehicle_id,

                "predicted_class":
                    predicted_class,

                "feature_name":
                    feature_name,

                "feature_value":
                    feature_value_db,

                "shap_value":
                    float(
                        shap_value
                    ),

                "contribution_rank":
                    contribution_rank,
            })


    shap_df = pd.DataFrame(
        shap_rows
    )


    print(
        "SHAP rows generated:",
        len(shap_df)
    )


    expected_rows = (
        len(vehicle_ids)
        *
        top_n
    )


    if len(shap_df) != expected_rows:

        print(
            "WARNING: expected",
            expected_rows,
            "SHAP rows but generated",
            len(shap_df),
        )


    return shap_df

def _save_shap(shap_df, inserted_analysis, engine, shap_table):
    shap_db = shap_df.merge(
        inserted_analysis[["analysis_id", "vehicle_id"]],
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
    if shap_db["analysis_id"].isna().any():
        raise RuntimeError("Some SHAP rows could not be linked to analysis_id")

    shap_db = shap_db[[
        "analysis_id", "feature_name", "feature_value",
        "shap_value", "contribution_rank",
    ]].copy()
    shap_db["analysis_id"] = shap_db["analysis_id"].astype(int)

    shap_db.to_sql(
        shap_table,
        engine,
        if_exists="append",
        index=False,
        chunksize=500,
    )
    return shap_db


def run_fleetcare_prediction(
    model_dir,
    username=None,
    password=None,
    host=None,
    port=None,
    database=None,
    analysis_table="analysis",
    shap_table="prediction_shap",
    top_shap=5,
    
):
    """PostgreSQL -> FleetCare inference -> analysis -> Top-N SHAP -> prediction_shap."""
    print("=" * 80)
    print("FLEETCARE AI - POSTGRESQL INFERENCE + SHAP")
    print("=" * 80)

    engine = _create_postgres_engine(username, password, host, port, database)
    operational, specifications = _load_fleet_from_postgres(engine)

    predictor = FleetCarePredictor(model_dir=model_dir)
    result = predictor.predict(operational, specifications)

    print("\nTop predictions:")
    print(
        result[[
            "ranking", "vehicle_id", "predicted_state", "confidence",
            "Risk6", "Risk6_percentile", "rule_id", "priority", "action",
        ]].head(20).round(4).to_string(index=False)
    )

    # Calculate SHAP before insert; if SHAP fails, analysis is not partially written.
    print(f"\nCalculating Top-{top_shap} SHAP explanations...")
    shap_df = _calculate_top_shap(predictor, result, top_n=top_shap)
    print(f"Generated {len(shap_df):,} SHAP rows")

    analysis_df = _build_analysis_dataframe(result)
    print(f"\nWriting {len(analysis_df):,} analysis rows...")
    inserted = _insert_analysis_and_get_ids(analysis_df, engine, analysis_table)

    print(f"Writing SHAP rows to `{shap_table}`...")
    shap_db = _save_shap(shap_df, inserted, engine, shap_table)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Analysis rows: {len(analysis_df):,}")
    print(f"SHAP rows:     {len(shap_db):,}")

    return analysis_df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    result = run_fleetcare_prediction(
        model_dir=BASE_DIR / "processed" / "final_model",
        username=None,
        password=None,
        host=None,
        port=None,
        database=None,
        analysis_table="analysis",
        shap_table="prediction_shap",
        top_shap=5,
    )
    print(result.head().to_string(index=False))
