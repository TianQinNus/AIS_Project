import contextlib
import io
import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "final_model"
)
_DATA_DIR = _MODEL_DIR.parents[1]


@lru_cache(maxsize=1)
def _predictor():
    if str(_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_DIR))

    from fleetcare_inference import FleetCarePredictor

    return FleetCarePredictor(_MODEL_DIR)


def _time_label(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if number.is_integer():
        return str(int(number))
    text_value = f"{number:.2f}".rstrip("0").rstrip(".")
    return text_value


def _clock_label(value: object) -> str | None:
    """Calendar time for one readout, as YYYY-MM-DD HH:MM."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None

    converted = pd.to_datetime(value, errors="coerce")
    if pd.isna(converted):
        return None
    return converted.strftime("%Y-%m-%d %H:%M")


def _round_risk(value: float) -> float:
    return round(float(value), 3)


def operational_risk_curve(
    db: Session,
    vehicle_id: int,
) -> list[dict] | None:
    """Risk probabilities at each readout time for one vehicle.

    The label is the readout's calendar time. Risk48, Risk24, Risk12 and
    Risk6 are nested class probabilities from the same prediction, so
    Risk48 >= Risk24 >= Risk12 >= Risk6. At a given time, the elevated
    curves show which repair window the vehicle is in.
    """

    try:
        return _operational_risk_curve(db, vehicle_id)
    except Exception:
        logger.exception(
            "Could not build the operational risk curve for vehicle %s",
            vehicle_id,
        )
        return None


def _operational_risk_curve(
    db: Session,
    vehicle_id: int,
) -> list[dict] | None:
    readouts = pd.read_sql(
        text(
            """
            SELECT *
            FROM truck_readout
            WHERE vehicle_id = :vehicle_id
            ORDER BY time_step
            """
        ),
        db.connection(),
        params={"vehicle_id": vehicle_id},
    )
    if readouts.empty:
        return None

    specifications = pd.read_sql(
        text(
            """
            SELECT vehicle_id, spec_0, spec_1, spec_2, spec_3,
                   spec_4, spec_5, spec_6, spec_7
            FROM truck
            WHERE vehicle_id = :vehicle_id
            """
        ),
        db.connection(),
        params={"vehicle_id": vehicle_id},
    )
    specifications = specifications.rename(
        columns={f"spec_{index}": f"Spec_{index}" for index in range(8)}
    )

    predictor = _predictor()
    readouts = readouts.drop(
        columns=["readout_id"],
        errors="ignore",
    )

    from fleetcare_inference import create_history_features

    with contextlib.redirect_stdout(io.StringIO()):
        history = create_history_features(
            readouts,
            predictor.raw_feature_columns,
        )
    history = history.merge(
        specifications,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
    model_input = (
        history.drop(columns=["vehicle_id"], errors="ignore")
        .reindex(columns=predictor.feature_columns)
    )
    probabilities = predictor.model.predict_proba(model_input)

    risk48 = probabilities[:, 1:].sum(axis=1)
    risk24 = probabilities[:, 2:].sum(axis=1)
    risk12 = probabilities[:, 3:].sum(axis=1)
    risk6 = probabilities[:, 4]

    clocks = (
        history["collected_at"]
        if "collected_at" in history.columns
        else None
    )
    labels = []
    for index, time_step in enumerate(history["time_step"]):
        clock = _clock_label(clocks.iloc[index]) if clocks is not None else None
        labels.append(clock or _time_label(time_step))

    return [
        {
            "label": labels[index],
            "risk48": _round_risk(risk48[index]),
            "risk24": _round_risk(risk24[index]),
            "risk12": _round_risk(risk12[index]),
            "risk6": _round_risk(risk6[index]),
        }
        for index in range(len(labels))
    ]
