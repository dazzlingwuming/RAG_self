from fastapi.testclient import TestClient

from hybrid_graph_rag_app.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["routing"] == "enabled"
    assert data["verifier"] == "enabled"
    assert data["response_schema"] == "trusted_rag_v1"


def test_index_page_renders():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Hybrid Graph RAG" in response.text
    assert "可信校验" in response.text
