import pytest
from fastapi.testclient import TestClient

from litebench.server.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_root_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text
    assert "LiteBench" in resp.text


def test_api_tasks_lists_builtins(client):
    resp = client.get("/api/tasks")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    for expected in {"humaneval", "gsm8k", "gsm8k-agent", "mmlu", "truthfulqa", "math", "arc"}:
        assert expected in names


def test_api_runs_returns_list(client):
    resp = client.get("/api/runs?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_api_runs_detail_404_for_unknown(client):
    resp = client.get("/api/runs/nonexistent-prefix")
    assert resp.status_code == 404


def test_api_compare_shape(client):
    resp = client.get("/api/compare")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"tasks", "models", "cells"}
    assert isinstance(data["tasks"], list)
    assert isinstance(data["models"], list)
    assert isinstance(data["cells"], list)
