from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundError
from app.repositories.vehicle_repository import VehicleRepository
from app.services.operational_risk import operational_risk_curve
from app.schemas.vehicle import (
    VehicleActionListResponse,
    VehicleActionOption,
    VehicleDetailResponse,
    VehicleFactor,
    VehicleListItem,
    VehicleListResponse,
    VehicleRiskReading,
    VehicleTrendPoint,
)


class VehicleQueryError(Exception):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


_VEHICLE_PREFIX = "VHC-"

_HISTORY_LABELS = {
    "STRONG": "Strong",
    "MODERATE": "Moderate",
    "LIMITED": "Limited",
}

_STATE_WINDOWS = {
    "C0": "48+ units",
    "C1": "24–48 units",
    "C2": "12–24 units",
    "C3": "6–12 units",
    "C4": "0–6 units",
}

_STATE_RISK_CLASS = {
    "C0": "Beyond 48",
    "C1": "Risk48",
    "C2": "Risk24",
    "C3": "Risk12",
    "C4": "Risk6",
}

@dataclass(frozen=True)
class ActionPresentation:
    predicted_class: str
    badge: str
    tag_color: str
    action_label: str


_ACTIONS = {
    "PRIORITY_INSPECT": ActionPresentation(
        "Priority Inspect",
        "PRIORITY INSPECT",
        "red",
        "Priority Inspect",
    ),
    "INSPECT": ActionPresentation(
        "Inspect",
        "INSPECT",
        "orange",
        "Plan Inspection",
    ),
    "ENHANCED_MONITOR": ActionPresentation(
        "Enhanced Monitor",
        "ENHANCED MONITOR",
        "gold",
        "Enhanced Monitor",
    ),
    "MONITOR": ActionPresentation(
        "Monitor",
        "MONITOR",
        "blue",
        "Monitor",
    ),
    "CONTINUE": ActionPresentation(
        "Continue",
        "CONTINUE",
        "green",
        "Continue",
    ),
}


def format_vehicle_id(vehicle_id: int) -> str:
    return f"{_VEHICLE_PREFIX}{vehicle_id}"


def parse_vehicle_id(value: str) -> int | None:
    text = value.strip()
    if text.upper().startswith(_VEHICLE_PREFIX):
        text = text[len(_VEHICLE_PREFIX) :]
    if not text.isdigit():
        return None
    return int(text)


def action_label(action: str) -> str:
    return action.replace("_", " ").title()


def resolve_action_filter(value: str | None) -> str | None:
    """Match a filter to the action stored in analysis, without a fixed list."""
    if value is None:
        return None

    key = value.strip()
    if not key or key.lower() in {"all", "all classes"}:
        return None

    return key.upper().replace(" ", "_")


def vehicle_search_digits(value: str | None) -> tuple[str | None, bool]:
    """Return ``(digits, force_empty)``.

    ``digits`` is the vehicle-id fragment to match. ``force_empty`` means the
    text cannot match an id, so the list should be empty without querying.
    """

    if value is None:
        return None, False

    text = value.strip()
    if not text:
        return None, False

    upper = text.upper()
    if upper.startswith(_VEHICLE_PREFIX):
        text = text[len(_VEHICLE_PREFIX) :].strip()
        if not text:
            return None, False
    elif upper in {"V", "VH", "VHC"}:
        return None, False

    if not text.isdigit():
        return None, True

    return text, False


def present_action(action: str | None) -> ActionPresentation:
    if action and action in _ACTIONS:
        return _ACTIONS[action]

    label = action_label(action) if action else "Unknown"
    return ActionPresentation(label, label.upper(), "default", label)


def history_label(value: str | None) -> str:
    if not value:
        return "Unknown"
    return _HISTORY_LABELS.get(value.strip().upper(), value.strip().title())


def predicted_state_label(state: str | None) -> str:
    if not state:
        return "Unknown"

    code = state.strip().upper()
    window = _STATE_WINDOWS.get(code)
    if window:
        return f"{code} · {window}"
    return state.strip()


def split_reasons(evidence: str | None, reason: str | None) -> list[str]:
    """Why-this-action lines: the reason sentence, then each evidence item."""
    parts: list[str] = []
    if reason and reason.strip():
        parts.append(reason.strip())
    if evidence:
        parts.extend(
            part.strip()
            for part in evidence.split(";")
            if part.strip()
        )
    return parts


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    return float(value)


def _round(value: object, digits: int) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return round(number, digits)


def risk_score(percentile: object) -> float:
    number = _round(percentile, 2)
    if number is None:
        return 0.0
    return number


def _unit_interval(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return round(number, 3)


def _percentile_unit(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return round(number / 100, 3)


def _format_unit(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _trend_label(value: object) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _daily_class_trend(history: list[dict]) -> list[VehicleTrendPoint]:
    """One stored analysis row per calendar day, including its risk scores."""
    by_day: dict[str, VehicleTrendPoint] = {}
    for row in history:
        label = _trend_label(row["prediction_at"])
        by_day[label[:10]] = VehicleTrendPoint(
            label=label,
            risk_class=_STATE_RISK_CLASS.get(
                (row["state"] or "").strip().upper(),
                "Beyond 48",
            ),
            time_step=_as_float(row.get("time_step")),
            trend_direction=row.get("trend_direction"),
            risk48=_unit_interval(row.get("risk48_score")),
            risk24=_unit_interval(row.get("risk24_score")),
            risk12=_unit_interval(row.get("risk12_score")),
            risk6=_unit_interval(row.get("risk6_score")),
            risk6_percentile=_round(row.get("risk6_percentile"), 2),
        )
    return list(by_day.values())


def _sql_round(value: Decimal) -> Decimal:
    """Match PostgreSQL ROUND(numeric, 2): half away from zero."""
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _as_decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return None


def _interval_risk_readings(
    risk6: object,
    risk12: object,
    risk24: object,
    risk48: object,
) -> list[VehicleRiskReading] | None:
    """Split cumulative scores into the five time windows.

    Within 6 units is risk6. Each later window is the difference between
    neighboring cumulative scores. Beyond 48 units is the remainder.
    """
    risk6_score = _as_decimal(risk6)
    risk12_score = _as_decimal(risk12)
    risk24_score = _as_decimal(risk24)
    risk48_score = _as_decimal(risk48)
    if None in (risk6_score, risk12_score, risk24_score, risk48_score):
        return None

    buckets = (
        ("Within 6 units", risk6_score),
        ("6–12 units", risk12_score - risk6_score),
        ("12–24 units", risk24_score - risk12_score),
        ("24–48 units", risk48_score - risk24_score),
        ("Beyond 48 units", Decimal(1) - risk48_score),
    )
    return [
        VehicleRiskReading(label=label, value=f"{_sql_round(score):.2f}")
        for label, score in buckets
    ]


def _stored_risk_readings(latest: dict) -> list[VehicleRiskReading] | None:
    """Window shares from the latest analysis row, Within 6 units first."""
    return _interval_risk_readings(
        latest.get("risk6_score"),
        latest.get("risk12_score"),
        latest.get("risk24_score"),
        latest.get("risk48_score"),
    )


def build_factors(rows: list[dict]) -> list[VehicleFactor]:
    magnitudes = [abs(float(row["shap_value"])) for row in rows]
    peak = max(magnitudes) if magnitudes else 0.0
    factors: list[VehicleFactor] = []

    for row, magnitude in zip(rows, magnitudes, strict=True):
        impact = 0.0 if peak == 0 else round(magnitude / peak, 2)

        factors.append(
            VehicleFactor(
                name=row["feature_name"],
                impact=impact,
                shap_value=float(row["shap_value"]),
            )
        )

    return factors


class VehicleService:

    @staticmethod
    def list_vehicles(
        db: Session,
        *,
        q: str | None,
        predicted_class: str | None,
        page: int,
        page_size: int,
    ) -> VehicleListResponse:
        action = resolve_action_filter(predicted_class)
        digits, force_empty = vehicle_search_digits(q)

        if force_empty:
            return VehicleListResponse(
                total=0,
                page=page,
                page_size=page_size,
                items=[],
            )

        total, rows = VehicleRepository.list_latest(
            db,
            action=action,
            vehicle_digits=digits,
            page=page,
            page_size=page_size,
        )

        items = [
            VehicleListItem(
                vehicle_id=format_vehicle_id(row["vehicle_id"]),
                state=(row["state"] or "Unknown").strip().upper(),
                predicted_class=present_action(row["action"]).predicted_class,
                risk_score=risk_score(row["risk6_percentile"]),
                in_maintenance=bool(row["in_maintenance"]),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

        return VehicleListResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items,
        )

    @staticmethod
    def list_actions(db: Session) -> VehicleActionListResponse:
        items = [
            VehicleActionOption(
                action=row["action"],
                label=action_label(row["action"]),
                count=int(row["count"]),
            )
            for row in VehicleRepository.list_actions(db)
            if row["action"]
        ]
        return VehicleActionListResponse(items=items)

    @staticmethod
    def set_maintenance(
        db: Session,
        vehicle_id: int,
        in_maintenance: bool,
    ) -> bool | None:
        return VehicleRepository.set_maintenance(
            db,
            vehicle_id,
            in_maintenance,
        )

    @staticmethod
    def get_vehicle(
        db: Session,
        vehicle_id: int,
    ) -> VehicleDetailResponse:
        history = VehicleRepository.get_history(db, vehicle_id)
        if not history:
            raise ResourceNotFoundError(
                "Vehicle",
                format_vehicle_id(vehicle_id),
            )

        latest = history[-1]
        factors = VehicleRepository.get_factors(
            db,
            latest["analysis_id"],
        )
        presentation = present_action(latest["action"])

        trend = _daily_class_trend(history)
        risk_readings = _stored_risk_readings(latest)
        risk6 = _unit_interval(latest["risk6_score"])
        if risk_readings is None:
            curve = operational_risk_curve(db, vehicle_id)
            if curve:
                latest_curve = curve[-1]
                risk48 = latest_curve["risk48"]
                risk24 = latest_curve["risk24"]
                risk12 = latest_curve["risk12"]
                risk6 = latest_curve["risk6"]
                risk_readings = _interval_risk_readings(
                    risk6,
                    risk12,
                    risk24,
                    risk48,
                ) or []
            else:
                risk48 = _percentile_unit(latest["risk48_percentile"])
                risk12 = _percentile_unit(latest["risk12_percentile"])
                risk6 = _unit_interval(latest["risk6_score"])
                risk_readings = [
                    VehicleRiskReading(label="Risk48", value=_format_unit(risk48)),
                    VehicleRiskReading(label="Risk12", value=_format_unit(risk12)),
                    VehicleRiskReading(label="Risk6", value=_format_unit(risk6)),
                ]

        return VehicleDetailResponse(
            vehicle_id=format_vehicle_id(vehicle_id),
            badge=presentation.badge,
            tag_color=presentation.tag_color,
            predicted_state=predicted_state_label(latest["state"]),
            action=presentation.action_label,
            history=history_label(latest["history_quality"]),
            risks=risk_readings,
            trend=trend,
            rule=latest["rule_id"] or "",
            reasons=split_reasons(latest["evidence"], latest["reason"]),
            factors=build_factors(factors),
            evidence=latest["evidence"],
            reason=latest["reason"],
            priority=latest["priority"],
            confidence=_round(latest["confidence_score"], 6),
            risk48_percentile=_round(latest["risk48_percentile"], 2),
            risk24_percentile=_round(latest.get("risk24_percentile"), 2),
            risk12_percentile=_round(latest["risk12_percentile"], 2),
            risk48_score=_unit_interval(latest.get("risk48_score")),
            risk24_score=_unit_interval(latest.get("risk24_score")),
            risk12_score=_unit_interval(latest.get("risk12_score")),
            risk6_score=risk6,
            risk6_percentile=_round(latest["risk6_percentile"], 2),
            time_step=_as_float(latest.get("time_step")),
            trend_direction=latest.get("trend_direction"),
        )
