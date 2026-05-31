from fastapi.testclient import TestClient

from src.app.main import create_app


def test_health():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_shape():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/query", json={"question": "What is LCEL?"})
    assert resp.status_code == 200
    data = resp.json()
    # only shape checks (content depends on docs/model)
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "iterations" in data
    assert "request_id" in data