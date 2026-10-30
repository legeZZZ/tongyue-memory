"""AML Add / Search contract. All model calls are local fakes."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.embeddings import UpstreamError
from mneme.llm import HeuristicLLM
from tests.conftest import make_settings


def _add(client: TestClient, **overrides):
    body = {
        "request_id": "req-1",
        "user_id": "user-a",
        "session_id": "sess-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "My cat is named Luna.",
            }
        ],
    }
    body.update(overrides)
    return client.post("/add", json=body)


def test_health_is_public_2xx(api):
    client, service = api
    response = client.get("/health", headers={"Authorization": "Bearer wrong"})
    assert 200 <= response.status_code < 300
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["fts"] in {"jieba+trigram", "jieba", "simple"}
    assert "unicode61" not in body["fts"]
    del service


def test_add_success_echoes_ids(api):
    client, _service = api
    response = _add(
        client,
        request_id="eval:run:chunk-0",
        user_id="eval:用户",
        session_id="eval:sess",
    )
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "request_id": "eval:run:chunk-0",
        "user_id": "eval:用户",
        "session_id": "eval:sess",
    }


def test_search_returns_id_and_content(api):
    client, _service = api
    assert _add(client).status_code == 200
    response = client.post(
        "/search",
        json={"query": "What is my cat named?", "user_id": "user-a", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data
    assert len(data) <= 5
    for item in data:
        assert set(item) == {"id", "content", "score", "created_at"}
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["content"], str) and item["content"]
        assert "timestamp" not in item
        assert "source" not in item
    assert any("Luna" in item["content"] for item in data)
    assert any(item["created_at"] == "2024-01-01T00:00:00Z" for item in data)


def test_search_empty_without_memories(api):
    client, _service = api
    response = client.post(
        "/search",
        json={"query": "quantum chromodynamics lattice", "user_id": "nobody", "top_k": 5},
    )
    assert response.status_code == 200
    assert response.json() == {"data": []}


def test_search_abstains_when_nothing_overlaps(api):
    client, _service = api
    assert _add(client).status_code == 200
    response = client.post(
        "/search",
        json={
            "query": "quantum chromodynamics lattice spacing",
            "user_id": "user-a",
            "top_k": 5,
        },
    )
    assert response.json() == {"data": []}


def test_search_abstains_when_tau_is_unreachable(tmp_path):
    app = create_app(make_settings(tmp_path, tau=10.0))
    with TestClient(app) as client:
        assert _add(client).status_code == 200
        response = client.post(
            "/search",
            json={"query": "What is my cat named?", "user_id": "user-a", "top_k": 5},
        )
        assert response.json() == {"data": []}


def test_top_k_caps_results_including_neighbors(api):
    client, _service = api
    messages = [
        {"role": "user", "content": f"zephyr note number {i} stays in memory"} for i in range(8)
    ]
    response = _add(client, request_id="bulk", messages=messages)
    assert response.status_code == 200
    found = client.post(
        "/search",
        json={"query": "zephyr note", "user_id": "user-a", "top_k": 2},
    )
    data = found.json()["data"]
    assert 1 <= len(data) <= 2
    assert len({item["id"] for item in data}) == len(data)


def test_user_id_isolation_ignores_session(api):
    client, _service = api
    assert _add(
        client,
        request_id="a1",
        user_id="user-a",
        session_id="session-left",
        messages=[{"role": "user", "content": "The secret fruit is mangosteen."}],
    ).status_code == 200
    assert _add(
        client,
        request_id="a2",
        user_id="user-a",
        session_id="session-right",
        messages=[{"role": "user", "content": "The travel plan is kepler transit."}],
    ).status_code == 200
    assert _add(
        client,
        request_id="b1",
        user_id="user-b",
        session_id="session-left",
        messages=[{"role": "user", "content": "The secret fruit is mangosteen."}],
    ).status_code == 200

    own = client.post(
        "/search",
        json={"query": "kepler transit travel plan", "user_id": "user-a", "top_k": 10},
    ).json()["data"]
    blob = " ".join(item["content"] for item in own)
    assert "kepler" in blob
    # Same user, other session, still visible. session_id is not a filter.
    fruit = client.post(
        "/search",
        json={"query": "secret fruit mangosteen", "user_id": "user-a", "top_k": 10},
    ).json()["data"]
    assert any("mangosteen" in item["content"] for item in fruit)

    other = client.post(
        "/search",
        json={"query": "kepler transit travel plan", "user_id": "user-b", "top_k": 10},
    ).json()["data"]
    assert all("kepler" not in item["content"] for item in other)


def test_request_id_is_idempotent(api):
    client, service = api
    first = _add(client)
    second = _add(client)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json() == second.json()
    assert service.store.count_turns("user-a") == 1


def test_request_id_conflict_does_not_overwrite(api):
    client, service = api
    assert _add(client).status_code == 200
    conflict = _add(
        client,
        messages=[{"role": "user", "content": "A totally different memory about quasars."}],
    )
    assert conflict.status_code == 409
    assert service.store.count_turns("user-a") == 1
    found = client.post(
        "/search",
        json={"query": "cat named Luna", "user_id": "user-a", "top_k": 5},
    ).json()["data"]
    assert any("Luna" in item["content"] for item in found)


def test_options_are_not_returned_as_evidence(api):
    client, _service = api
    assert _add(client).status_code == 200
    response = client.post(
        "/search",
        json={
            "query": "What is my cat named?",
            "user_id": "user-a",
            "top_k": 5,
            "options": ["A. The secret is swordfish", "B. Luna"],
        },
    )
    blob = " ".join(item["content"] for item in response.json()["data"])
    assert "swordfish" not in blob
    assert "Luna" in blob


def test_auth_schemes_and_open_health(tmp_path):
    app = create_app(make_settings(tmp_path, memory_system_key="secret-key"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/add", json=_payload()).status_code == 401
        assert client.post(
            "/add",
            json=_payload(),
            headers={"Authorization": "Bearer no"},
        ).status_code == 401
        for header in (
            {"Authorization": "Bearer secret-key"},
            {"Authorization": "Token secret-key"},
            {"X-Api-Key": "secret-key"},
        ):
            body = _payload()
            body["request_id"] = "req-" + next(iter(header.values()))[:12]
            response = client.post("/add", json=body, headers=header)
            assert response.status_code == 200, header


def test_embed_failure_is_503_and_writes_nothing(tmp_path):
    class Boom:
        dim = 1024

        def embed_texts(self, texts, *, text_type):
            del texts, text_type
            raise UpstreamError("embed failed")

    app = create_app(make_settings(tmp_path), embedder=Boom())
    with TestClient(app) as client:
        response = client.post("/add", json=_payload())
        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
        assert app.state.service.store.count_turns() == 0


def test_enrich_timeout_still_returns_searchable_raw(tmp_path):
    class SlowLLM(HeuristicLLM):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def extract(self, turns):
            self.started.set()
            self.release.wait(5)
            return super().extract(turns)

    slow = SlowLLM()
    app = create_app(make_settings(tmp_path, fact_timeout_s=0.3), llm=slow)
    with TestClient(app) as client:
        started = time.perf_counter()
        response = _add(client)
        elapsed = time.perf_counter() - started
        assert response.status_code == 200
        assert elapsed < 2.0
        assert slow.started.is_set()
        found = client.post(
            "/search",
            json={"query": "cat named Luna", "user_id": "user-a", "top_k": 5},
        )
        assert any("Luna" in item["content"] for item in found.json()["data"])
        assert app.state.service.store.enrich_status("req-1", "user-a") in {"running", "pending", "done"}
        slow.release.set()
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline:
            if app.state.service.store.enrich_status("req-1", "user-a") == "done":
                break
            time.sleep(0.05)
        assert app.state.service.store.enrich_status("req-1", "user-a") == "done"


def test_validation_error_is_422(api):
    client, _service = api
    response = client.post(
        "/add",
        json={
            "request_id": "bad",
            "user_id": "u",
            "session_id": "s",
            "messages": [{"role": "user", "content": "   "}],
        },
    )
    assert response.status_code == 422


def _payload():
    return {
        "request_id": "req-1",
        "user_id": "user-a",
        "session_id": "sess-1",
        "messages": [{"role": "user", "content": "My cat is named Luna."}],
    }
