from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_dashboard():
    response = client.get(
        "/api/v1/dashboard"
    )

    assert response.status_code == 200

    data = response.json()

    assert "fleet" in data
    assert "risk" in data
    assert "risk_snapshot" in data
    assert "top_risk_alerts" in data
    assert "manual_review" in data["maintenance_plan"]

    assert isinstance(
        data["fleet"]["total"],
        int,
    )

    assert isinstance(
        data["risk_snapshot"],
        list,
    )