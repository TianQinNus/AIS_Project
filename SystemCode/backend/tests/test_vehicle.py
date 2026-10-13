import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("DATABASE_URL", "sqlite://")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.dependencies import get_db
from app.main import app
from app.models.analysis import Analysis
from app.models.prediction_shap import PredictionShap
from app.models.truck import Truck
from app.services.vehicle_service import (
    build_factors,
    history_label,
    predicted_state_label,
    present_action,
    split_reasons,
)


engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def _seed() -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    now = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)
    earlier = now - timedelta(days=7)

    with TestingSession() as db:
        db.add_all(
            [
                Truck(vehicle_id=89321),
                Truck(vehicle_id=66107),
                Truck(vehicle_id=10001),
            ]
        )
        db.add_all(
            [
                Analysis(
                    analysis_id=1,
                    vehicle_id=89321,
                    prediction_at=earlier,
                    state="C3",
                    confidence_score=0.71,
                    risk48_percentile=70,
                    risk12_percentile=55,
                    risk6_score=0.21,
                    risk6_percentile=60,
                    history_quality="MODERATE",
                    rule_id="TR2",
                    priority="HIGH",
                    action="INSPECT",
                    evidence="current model state is C3",
                    reason="Earlier inspection signal.",
                ),
                Analysis(
                    analysis_id=2,
                    vehicle_id=89321,
                    prediction_at=now,
                    state="C4",
                    confidence_score=0.91,
                    risk48_percentile=89.8,
                    risk12_percentile=67.9,
                    risk6_score=0.431,
                    risk6_percentile=92.4,
                    history_quality="STRONG",
                    rule_id="TR1",
                    priority="CRITICAL",
                    action="PRIORITY_INSPECT",
                    evidence=(
                        "current model state is C4; "
                        "Risk6 is in the top 5% of the fleet"
                    ),
                    reason="Priority inspection is recommended.",
                ),
                Analysis(
                    analysis_id=3,
                    vehicle_id=66107,
                    prediction_at=now,
                    state="C3",
                    confidence_score=0.8,
                    risk48_percentile=70.4,
                    risk12_percentile=44.6,
                    risk6_score=0.268,
                    risk6_percentile=71,
                    history_quality="MODERATE",
                    rule_id="TR2",
                    priority="HIGH",
                    action="INSPECT",
                    evidence="current model state is C3",
                    reason="Plan an inspection.",
                ),
                Analysis(
                    analysis_id=4,
                    vehicle_id=10001,
                    prediction_at=now,
                    state="C0",
                    confidence_score=0.66,
                    risk48_percentile=20,
                    risk12_percentile=10,
                    risk6_score=0.04,
                    risk6_percentile=15,
                    history_quality="LIMITED",
                    rule_id="TR6",
                    priority="NORMAL",
                    action="CONTINUE",
                    evidence=None,
                    reason="No strong elevated-risk evidence detected",
                ),
            ]
        )
        db.add_all(
            [
                PredictionShap(
                    shap_id=1,
                    analysis_id=2,
                    feature_name="397_12 rate",
                    feature_value=1.2,
                    shap_value=0.86,
                    contribution_rank=1,
                ),
                PredictionShap(
                    shap_id=2,
                    analysis_id=2,
                    feature_name="167_3 change5",
                    feature_value=0.4,
                    shap_value=-0.43,
                    contribution_rank=2,
                ),
            ]
        )
        db.commit()


def test_risk_curve_clock_label():
    from app.services.operational_risk import _clock_label

    moment = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    assert _clock_label(moment) == "2026-09-27 12:00"
    assert _clock_label(None) is None


def test_label_mapping():
    assert predicted_state_label("C4") == "C4 · 0–6 units"
    assert history_label("STRONG") == "Strong"
    assert present_action("INSPECT").action_label == "Plan Inspection"
    assert split_reasons(
        "current model state is C4; Risk6 is in the top 5% of the fleet",
        "ignored when evidence exists",
    ) == [
        "current model state is C4",
        "Risk6 is in the top 5% of the fleet",
    ]

    factors = build_factors(
        [
            {"feature_name": "a", "shap_value": 0.5},
            {"feature_name": "b", "shap_value": -0.25},
        ]
    )
    assert factors[0].impact == 1
    assert factors[1].impact == 0.5


def test_vehicle_list_and_detail():
    _seed()
    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)

    try:
        response = client.get("/api/v1/vehicles")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["pageSize"] == 20
        assert [item["vehicleId"] for item in data["items"]] == [
            "VHC-89321",
            "VHC-66107",
            "VHC-10001",
        ]
        assert data["items"][0]["state"] == "C4"
        assert data["items"][0]["predictedClass"] == "Priority Inspect"
        assert data["items"][0]["riskScore"] == 92
        assert data["items"][0]["inMaintenance"] is False

        turned_on = client.patch(
            "/api/v1/vehicles/VHC-89321/maintenance",
            json={"inMaintenance": True},
        )
        assert turned_on.status_code == 200
        assert turned_on.json()["inMaintenance"] is True

        listed = client.get("/api/v1/vehicles", params={"q": "89321"})
        assert listed.json()["items"][0]["inMaintenance"] is True

        filtered = client.get(
            "/api/v1/vehicles",
            params={"predicted_class": "Inspect", "q": "66107"},
        )
        assert filtered.status_code == 200
        assert filtered.json()["total"] == 1
        assert filtered.json()["items"][0]["vehicleId"] == "VHC-66107"

        empty = client.get("/api/v1/vehicles", params={"q": "not-an-id"})
        assert empty.status_code == 200
        assert empty.json()["total"] == 0

        paged = client.get(
            "/api/v1/vehicles",
            params={"page": 2, "page_size": 1},
        )
        assert paged.status_code == 200
        assert paged.json()["items"][0]["vehicleId"] == "VHC-66107"

        invalid_class = client.get(
            "/api/v1/vehicles",
            params={"predicted_class": "Broken"},
        )
        assert invalid_class.status_code == 422

        detail = client.get("/api/v1/vehicles/VHC-89321")
        assert detail.status_code == 200
        body = detail.json()
        assert body["vehicleId"] == "VHC-89321"
        assert body["badge"] == "PRIORITY INSPECT"
        assert body["tagColor"] == "red"
        assert body["predictedState"] == "C4 · 0–6 units"
        assert body["action"] == "Priority Inspect"
        assert body["history"] == "Strong"
        assert body["rule"] == "TR1"
        assert body["risks"] == [
            {"label": "Risk48", "value": "0.898"},
            {"label": "Risk12", "value": "0.679"},
            {"label": "Risk6", "value": "0.431"},
        ]
        assert len(body["trend"]) == 2
        assert body["trend"][0]["label"] == "2026-09-24 08:00"
        assert body["trend"][0]["riskClass"] == "Risk12"
        assert body["trend"][1]["label"] == "2026-10-01 08:00"
        assert body["trend"][1]["riskClass"] == "Risk6"
        assert body["reasons"][0] == "current model state is C4"
        assert body["factors"][0] == {"name": "397_12 rate", "impact": 1.0}
        assert body["factors"][1]["impact"] == 0.5

        numeric = client.get("/api/v1/vehicles/66107")
        assert numeric.status_code == 200
        assert numeric.json()["action"] == "Plan Inspection"

        missing = client.get("/api/v1/vehicles/VHC-999")
        assert missing.status_code == 404

        invalid = client.get("/api/v1/vehicles/VHC-abc")
        assert invalid.status_code == 422
    finally:
        app.dependency_overrides.clear()
