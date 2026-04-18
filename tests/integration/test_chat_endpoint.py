"""Integration tests for the FastAPI /chat endpoint.

The real FastAPI app is instantiated, but the OpenAI client and the FAISS
index are replaced with in-process stubs before app startup. Auth is not
exercised here because API Gateway enforces the JWT authorizer outside the
function (the Lambda never sees unauthenticated requests).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tests.conftest import StubOpenAI


@pytest.fixture
def client(monkeypatch, tiny_bundle):
    # Short-circuit key resolution so lifespan doesn't reach for SSM/boto3.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PALAV_CORS_ORIGINS", "https://frontend.example.com")

    import app as app_module

    stub = StubOpenAI(
        chat_completion_text=(
            'A good latch means the baby covers the areola.\n'
            'USED_URLS: ["https://example.com/latch"]'
        )
    )

    # Replace lifespan side-effects: avoid building an index from network.
    def fake_build_or_load(**kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            index=tiny_bundle.index,
            vectors=tiny_bundle.vectors,
            chunks=tiny_bundle.chunks,
            report={"ok": 3, "failed": []},
            key="test",
            paths={},
            loaded_from_cache=True,
        )

    monkeypatch.setattr(app_module, "build_or_load", fake_build_or_load)
    monkeypatch.setattr(app_module, "OpenAI", lambda api_key: stub)

    with TestClient(app_module.app) as c:
        c._stub = stub  # type: ignore[attr-defined] - handy in a couple tests
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["chunks"] == 3


def test_chat_happy_path(client):
    r = client.post("/chat", json={"message": "how do I latch?", "history": []})
    assert r.status_code == 200
    body = r.json()
    assert "areola" in body["answer"]
    assert body["sources"] == [
        {"url": "https://example.com/latch", "title": "Latching basics"}
    ]
    assert body["rejected"] is False
    assert body["external_knowledge"] is False


def test_chat_rejects_unrelated(client):
    client._stub._chat_text = "I do not have required information. Please try different question"
    r = client.post("/chat", json={"message": "how do I fix my car?", "history": []})
    assert r.status_code == 200
    body = r.json()
    assert body["rejected"] is True
    assert body["sources"] == []


def test_chat_history_forwarded(client):
    r = client.post(
        "/chat",
        json={
            "message": "and at night?",
            "history": [
                {"role": "user", "content": "how often should I feed?"},
                {"role": "assistant", "content": "about every 2-3 hours"},
            ],
        },
    )
    assert r.status_code == 200
    roles = [m["role"] for m in client._stub.last_messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_chat_rejects_empty_message(client):
    r = client.post("/chat", json={"message": "", "history": []})
    assert r.status_code == 422


def test_chat_rejects_malformed_body(client):
    r = client.post("/chat", json={"not_the_right_field": "hi"})
    assert r.status_code == 422


def test_chat_rejects_invalid_role(client):
    r = client.post(
        "/chat",
        json={"message": "hi", "history": [{"role": "system", "content": "evil"}]},
    )
    assert r.status_code == 422


def test_cors_preflight(client):
    r = client.options(
        "/chat",
        headers={
            "Origin": "https://frontend.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://frontend.example.com"
