"""Search must answer under a wall-clock budget when upstream calls stall.

Empty-user Search returns before any model call, so these tests store a turn
first. That is the Full-eval shape: a user who already has memory.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.config import Settings
from mneme.embeddings import HashEmbedder, UpstreamError
from mneme.llm import Analysis, HeuristicLLM, OpenAILLM
from mneme.schemas import AddRequest, SearchRequest
from mneme.service import MemoryService
from tests.conftest import make_settings


def _search(service: MemoryService, query: str, user_id: str = "user-a"):
    return service.search(SearchRequest(query=query, user_id=user_id, top_k=5))


def _add_luna(service: MemoryService, request_id: str = "req-1", user_id: str = "user-a") -> None:
    service.add(
        AddRequest(
            request_id=request_id,
            user_id=user_id,
            session_id="sess-1",
            messages=[{"role": "user", "content": "My cat is named Luna."}],
        )
    )


class _GateLLM(HeuristicLLM):
    """Blocks in analyze until released. Extract stays instant so Add can commit."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def analyze(self, query: str) -> Analysis:
        self.calls += 1
        self.started.set()
        self.release.wait(30)
        return super().analyze(query)


class _GateEmbedder:
    """Document embeds succeed; query embeds block. Add can store, Search cannot wait."""

    def __init__(self) -> None:
        self.dim = 1024
        self.started = threading.Event()
        self.release = threading.Event()
        self._hash = HashEmbedder()

    def embed_texts(self, texts, *, text_type):
        if text_type == "query":
            self.started.set()
            self.release.wait(30)
        return self._hash.embed_texts(texts, text_type=text_type)


class _Overlap:
    def __init__(self) -> None:
        self.dim = 1024
        self._hash = HashEmbedder()
        self.analyze_start = 0.0
        self.analyze_end = 0.0
        self.embed_start = 0.0
        self.embed_end = 0.0
        self._delay = 0.45

    def analyze(self, query: str) -> Analysis:
        self.analyze_start = time.perf_counter()
        time.sleep(self._delay)
        self.analyze_end = time.perf_counter()
        return Analysis(tense="none", as_of=None, hops=1, recall_q=query)

    def extract(self, turns):
        return HeuristicLLM().extract(turns)

    def judge(self, new, candidates):
        return HeuristicLLM().judge(new, candidates)

    def embed_texts(self, texts, *, text_type):
        if text_type == "query":
            self.embed_start = time.perf_counter()
            time.sleep(self._delay)
            self.embed_end = time.perf_counter()
        return self._hash.embed_texts(texts, text_type=text_type)


def test_search_returns_stored_turn_when_analyze_and_embed_block(tmp_path):
    llm = _GateLLM()
    embedder = _GateEmbedder()
    settings = make_settings(
        tmp_path,
        search_budget_s=0.4,
        search_llm_timeout_s=0.25,
        embedding_timeout_s=0.25,
    )
    app = create_app(settings, embedder=embedder, llm=llm)
    try:
        with TestClient(app) as client:
            _add_luna(app.state.service)
            started = time.perf_counter()
            response = client.post(
                "/search",
                json={"query": "cat named Luna", "user_id": "user-a", "top_k": 5},
            )
            elapsed = time.perf_counter() - started
        assert response.status_code == 200, response.text
        assert elapsed < 2.0
        assert any("Luna" in item["content"] for item in response.json()["data"])
        assert llm.started.is_set()
        assert embedder.started.is_set()
    finally:
        llm.release.set()
        embedder.release.set()


def test_analyze_and_query_embed_overlap(tmp_path):
    models = _Overlap()
    settings = make_settings(tmp_path, search_budget_s=3.0, search_llm_timeout_s=2.0, embedding_timeout_s=2.0)
    service = MemoryService(settings, embedder=models, llm=models)
    _add_luna(service)
    started = time.perf_counter()
    found = _search(service, "cat named Luna")
    elapsed = time.perf_counter() - started
    assert any("Luna" in hit.content for hit in found.data)
    assert models.analyze_start and models.embed_start
    # Serial would start the second call only after the first 0.45s sleep ends.
    assert min(models.analyze_end, models.embed_end) > max(models.analyze_start, models.embed_start)
    assert abs(models.analyze_start - models.embed_start) < 0.3
    assert elapsed < 0.9


def test_failed_analyze_is_not_cached(tmp_path):
    class Flaky(HeuristicLLM):
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, query: str) -> Analysis:
            self.calls += 1
            if self.calls == 1:
                raise UpstreamError("llm request failed: TimeoutException")
            return Analysis(tense="current", as_of=None, hops=1, recall_q="我记得猫的名字")

    llm = Flaky()
    service = MemoryService(make_settings(tmp_path), embedder=HashEmbedder(), llm=llm)
    _add_luna(service)
    query = "cat named Luna"
    _search(service, query)
    assert query not in service._cache
    _search(service, query)
    assert llm.calls == 2
    assert service._cache[query].recall_q == "我记得猫的名字"


def test_rate_limited_query_embed_still_returns_fts(tmp_path):
    class RateLimit:
        def __init__(self) -> None:
            self.dim = 1024
            self._hash = HashEmbedder()

        def embed_texts(self, texts, *, text_type):
            if text_type == "query":
                raise UpstreamError("embedding upstream HTTP 429")
            return self._hash.embed_texts(texts, text_type=text_type)

    app = create_app(make_settings(tmp_path), embedder=RateLimit(), llm=HeuristicLLM())
    with TestClient(app) as client:
        _add_luna(app.state.service)
        response = client.post(
            "/search",
            json={"query": "cat named Luna", "user_id": "user-a", "top_k": 5},
        )
    assert response.status_code == 200, response.text
    assert any("Luna" in item["content"] for item in response.json()["data"])


def test_missing_embed_key_on_search_stays_503(tmp_path):
    class MissingQueryKey:
        def __init__(self) -> None:
            self.dim = 1024
            self._hash = HashEmbedder()

        def embed_texts(self, texts, *, text_type):
            if text_type == "query":
                raise UpstreamError(
                    "embedding API key not configured (set DASHSCOPE_API_KEY or EMBEDDING_API_KEY)"
                )
            return self._hash.embed_texts(texts, text_type=text_type)

    app = create_app(make_settings(tmp_path), embedder=MissingQueryKey(), llm=HeuristicLLM())
    with TestClient(app) as client:
        _add_luna(app.state.service)
        response = client.post(
            "/search",
            json={"query": "cat named Luna", "user_id": "user-a", "top_k": 5},
        )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"


def test_enrich_concurrency_defers_extra_chunks(tmp_path):
    class GateExtract(HeuristicLLM):
        def __init__(self) -> None:
            self.entered = threading.Semaphore(0)
            self.release = threading.Event()
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def extract(self, turns):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.entered.release()
            assert self.release.wait(5)
            with self.lock:
                self.active -= 1
            return super().extract(turns)

    gate = GateExtract()
    service = MemoryService(
        make_settings(tmp_path, fact_timeout_s=0, enrich_concurrency=1, disable_worker=True),
        embedder=HashEmbedder(),
        llm=gate,
    )
    try:
        _add_luna(service, request_id="chunk-1")
        assert gate.entered.acquire(timeout=2)
        _add_luna(service, request_id="chunk-2")
        time.sleep(0.15)
        assert gate.max_active == 1
        assert service.store.enrich_status("chunk-2", "user-a") == "pending"
    finally:
        gate.release.set()
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        if service.store.enrich_status("chunk-1", "user-a") == "done":
            break
        time.sleep(0.05)
    assert service.store.enrich_status("chunk-1", "user-a") == "done"
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        if service.store.enrich_status("chunk-2", "user-a") == "done":
            break
        service.drain_once()
        time.sleep(0.05)
    assert service.store.enrich_status("chunk-2", "user-a") == "done"
    assert gate.max_active == 1


def test_openai_analyze_http_failure_is_not_cached(tmp_path):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    llm = OpenAILLM(
        base_url="https://llm.example/v1",
        api_key="test-key",
        timeout_s=2.0,
        transport=httpx.MockTransport(handler),
    )
    service = MemoryService(make_settings(tmp_path), embedder=HashEmbedder(), llm=llm)
    _add_luna(service)
    query = "cat named Luna"
    _search(service, query)
    _search(service, query)
    assert query not in service._cache
    llm.close()


def test_search_passes_sliced_timeout_to_embed_batches(tmp_path):
    """Query embeds must not keep the client's full timeout after the search slice."""
    import httpx
    import numpy as np

    from mneme.embeddings import DashScopeEmbedder, effective_timeout, l2_normalize

    class Spy(DashScopeEmbedder):
        def __init__(self) -> None:
            super().__init__(
                base_url="https://embed.example/v1",
                api_key="k",
                timeout_s=40.0,
                transport=httpx.MockTransport(lambda request: httpx.Response(500)),
            )
            self.seen: list[float] = []

        def _embed_batch(self, texts, text_type):
            if text_type == "query":
                self.seen.append(effective_timeout(self.timeout_s))
            vec = l2_normalize(np.ones(self.dim, dtype=np.float32))
            return [vec for _ in texts]

    spy = Spy()
    service = MemoryService(
        make_settings(tmp_path, search_budget_s=2.0, search_llm_timeout_s=2.0, embedding_timeout_s=40.0),
        embedder=spy,
        llm=HeuristicLLM(),
    )
    _add_luna(service)
    found = _search(service, "cat named Luna")
    assert any("Luna" in hit.content for hit in found.data)
    assert spy.seen
    assert all(0 < timeout <= 2.5 for timeout in spy.seen)
    spy.close()


def test_search_budget_settings_from_env(monkeypatch):
    monkeypatch.setenv("MNEME_SEARCH_BUDGET_S", "22")
    monkeypatch.setenv("MNEME_SEARCH_LLM_TIMEOUT_S", "6")
    monkeypatch.setenv("MNEME_ENRICH_CONCURRENCY", "3")
    monkeypatch.setenv("MNEME_SEARCH_WORKERS", "24")
    monkeypatch.setenv("LLM_TIMEOUT_S", "11")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_S", "14")
    cfg = Settings.from_env()
    assert cfg.search_budget_s == 22
    assert cfg.search_llm_timeout_s == 6
    assert cfg.enrich_concurrency == 3
    assert cfg.search_workers == 24
    assert cfg.llm_timeout_s == 11
    assert cfg.embedding_timeout_s == 14
    assert os.environ["MNEME_SEARCH_BUDGET_S"] == "22"
