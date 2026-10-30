"""Regressions for current-tense leaks, similarity supersede, and abstention.

These tests must fail if the corresponding fix is reverted.
"""

from __future__ import annotations

import math

import numpy as np
from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.llm import JUDGE_SYSTEM, FactDraft, HeuristicLLM, Verdict
from tests.conftest import make_settings

# Pinned with the heuristic. Production code must use this cutoff.
_SIM_SAME = 0.82

_OLD_TS = 1704067200000  # 2024-01-01
_NEW_TS = 1735689600000  # 2025-01-01
_OLD = "用户的最喜欢导演是诺兰"
_NEW = "用户的最喜欢导演是维伦纽瓦"


class _NoFacts(HeuristicLLM):
    def extract(self, turns):
        del turns
        return []


class _DriftExtract(HeuristicLLM):
    """Same referent, different predicate strings. Judge must use sim, not SPO equality."""

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def extract(self, turns):
        drafts: list[FactDraft] = []
        for turn in turns:
            if "诺兰" in turn.content:
                drafts.append(
                    FactDraft(
                        statement="用户最喜欢的导演是诺兰",
                        subject="用户",
                        predicate="最喜欢的导演",
                        object="诺兰",
                        ts_event=turn.ts_event,
                        provenance="stated",
                        confidence=1.0,
                        sensitivity="normal",
                        src_turn_id=turn.id,
                    )
                )
            elif "维伦纽瓦" in turn.content:
                drafts.append(
                    FactDraft(
                        statement="用户目前最爱的导演是维伦纽瓦",
                        subject="用户",
                        predicate="目前最爱的导演",
                        object="维伦纽瓦",
                        ts_event=turn.ts_event,
                        provenance="stated",
                        confidence=1.0,
                        sensitivity="normal",
                        src_turn_id=turn.id,
                    )
                )
        return drafts

    def judge(self, new, candidates):
        self.seen.append(candidates)
        return super().judge(new, candidates)


class _CosineEmbedder:
    """Unit vectors whose dot product is ``cosine`` for the two director statements."""

    def __init__(self, cosine: float) -> None:
        self.dim = 8
        self._old = np.zeros(self.dim, dtype=np.float32)
        self._old[0] = 1.0
        self._new = np.zeros(self.dim, dtype=np.float32)
        self._new[0] = float(cosine)
        self._new[1] = math.sqrt(max(0.0, 1.0 - float(cosine) * float(cosine)))

    def embed_texts(self, texts, *, text_type):
        if text_type not in {"query", "document"}:
            raise ValueError(text_type)
        out = []
        for text in texts:
            if "维伦纽瓦" in text:
                out.append(self._new.copy())
            else:
                out.append(self._old.copy())
        return out


def _client(tmp_path, **settings):
    llm = settings.pop("llm", None)
    embedder = settings.pop("embedder", None)
    app = create_app(make_settings(tmp_path, **settings), llm=llm, embedder=embedder)
    return TestClient(app), app


def _add(client: TestClient, request_id: str, user_id: str, messages: list[dict]) -> None:
    response = client.post(
        "/add",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "session_id": "s",
            "messages": messages,
        },
    )
    assert response.status_code == 200


def _blob(client: TestClient, user_id: str, query: str, top_k: int = 10) -> str:
    found = client.post(
        "/search",
        json={"query": query, "user_id": user_id, "top_k": top_k},
    )
    assert found.status_code == 200
    return " ".join(item["content"] for item in found.json()["data"])


def _facts(service, user_id: str) -> list:
    with service.store._lock:
        return service.store.conn.execute(
            "SELECT subject, predicate, object, status, statement FROM facts WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()


def test_current_hides_old_turn_when_extract_returns_no_facts(tmp_path):
    """Enrich never writes facts, so suppressed_turn_ids stays empty.

    The two lines share a chunk, so neighbor expansion would put 诺兰 back
    unless the current-tense dedupe also hides that turn.
    """
    client, _app = _client(tmp_path, llm=_NoFacts())
    with client:
        _add(
            client,
            "nofacts-same",
            "u-nofacts",
            [
                {"role": "user", "content": _OLD, "timestamp": _OLD_TS},
                {"role": "user", "content": _NEW, "timestamp": _NEW_TS},
            ],
        )
        blob = _blob(client, "u-nofacts", "现在最喜欢的导演是谁")
    assert "维伦纽瓦" in blob
    assert "诺兰" not in blob


def test_current_hides_old_turn_across_chunks_without_facts(tmp_path):
    client, _app = _client(tmp_path, llm=_NoFacts())
    with client:
        _add(
            client,
            "nofacts-a",
            "u-chunks",
            [{"role": "user", "content": _OLD, "timestamp": _OLD_TS}],
        )
        _add(
            client,
            "nofacts-b",
            "u-chunks",
            [{"role": "user", "content": _NEW, "timestamp": _NEW_TS}],
        )
        blob = _blob(client, "u-chunks", "现在最喜欢的导演是谁")
    assert "维伦纽瓦" in blob
    assert "诺兰" not in blob


def test_current_blocks_stale_neighbor_outside_recall(tmp_path):
    """recall_limit=1 keeps only the newer turn. Expansion must not revive 诺兰."""
    client, _app = _client(tmp_path, llm=_NoFacts(), recall_limit=1)
    with client:
        _add(
            client,
            "nofacts-recall",
            "u-recall",
            [
                {"role": "user", "content": _OLD, "timestamp": _OLD_TS},
                {"role": "user", "content": _NEW, "timestamp": _NEW_TS},
            ],
        )
        blob = _blob(client, "u-recall", "现在最喜欢的导演是维伦纽瓦吗", top_k=5)
    assert "维伦纽瓦" in blob
    assert "诺兰" not in blob


def test_judge_prompt_tells_the_model_to_use_similarity():
    assert "Do not answer" in JUDGE_SYSTEM
    assert "same_spo" in JUDGE_SYSTEM
    assert "sim" in JUDGE_SYSTEM
    assert "0.82" in JUDGE_SYSTEM


def test_judge_updates_when_similarity_is_high_and_predicates_differ():
    llm = HeuristicLLM()
    new = FactDraft(
        statement="用户目前最爱的导演是维伦纽瓦",
        subject="用户",
        predicate="目前最爱的导演",
        object="维伦纽瓦",
        ts_event="2025-01-01T00:00:00Z",
        provenance="stated",
        confidence=1.0,
        sensitivity="normal",
    )
    old = {
        "id": 7,
        "subject": "用户",
        "predicate": "最喜欢的导演",
        "object": "诺兰",
        "status": "active",
        "valid_to": None,
        "same_spo": False,
    }
    assert llm.judge(new, [{**old, "sim": _SIM_SAME}]) == Verdict("updates", 7)
    assert llm.judge(new, [{**old, "sim": _SIM_SAME - 0.01}]) == Verdict("unrelated", None)


def test_reconcile_passes_sim_and_supersedes_on_predicate_drift(tmp_path):
    llm = _DriftExtract()
    client, app = _client(tmp_path, llm=llm, embedder=_CosineEmbedder(0.9))
    with client:
        _add(
            client,
            "sim-old",
            "u-sim",
            [{"role": "user", "content": "我以前说用户最喜欢的导演是诺兰", "timestamp": _OLD_TS}],
        )
        _add(
            client,
            "sim-new",
            "u-sim",
            [{"role": "user", "content": "我现在说用户目前最爱的导演是维伦纽瓦", "timestamp": _NEW_TS}],
        )
        rows = _facts(app.state.service, "u-sim")
    assert llm.seen, "judge was not called"
    passed = llm.seen[-1]
    assert passed, "judge received no candidates"
    assert passed[0]["same_spo"] is False
    assert passed[0]["sim"] >= _SIM_SAME
    by_object = {row["object"]: row["status"] for row in rows}
    assert by_object["诺兰"] == "superseded"
    assert by_object["维伦纽瓦"] == "active"


class _UnrelatedJudge:
    """Stands in for a real model that ignores sim and returns unrelated."""

    def extract(self, turns):
        return _DriftExtract().extract(turns)

    def analyze(self, query):
        return HeuristicLLM().analyze(query)

    def judge(self, new, candidates):
        del new, candidates
        return Verdict("unrelated", None)


def test_real_judge_unrelated_still_supersedes_on_similarity(tmp_path):
    client, app = _client(tmp_path, llm=_UnrelatedJudge(), embedder=_CosineEmbedder(0.9))
    with client:
        _add(
            client,
            "fb-old",
            "u-fb",
            [{"role": "user", "content": "以前的说法是用户最喜欢的导演是诺兰", "timestamp": _OLD_TS}],
        )
        _add(
            client,
            "fb-new",
            "u-fb",
            [{"role": "user", "content": "新的说法是用户目前最爱的导演是维伦纽瓦", "timestamp": _NEW_TS}],
        )
        rows = _facts(app.state.service, "u-fb")
    by_object = {row["object"]: row["status"] for row in rows}
    assert by_object["诺兰"] == "superseded"
    assert by_object["维伦纽瓦"] == "active"


def test_reconcile_does_not_supersede_below_similarity_threshold(tmp_path):
    llm = _DriftExtract()
    client, app = _client(tmp_path, llm=llm, embedder=_CosineEmbedder(0.5))
    with client:
        _add(
            client,
            "low-old",
            "u-low",
            [{"role": "user", "content": "用户最喜欢的导演是诺兰", "timestamp": _OLD_TS}],
        )
        _add(
            client,
            "low-new",
            "u-low",
            [{"role": "user", "content": "用户目前最爱的导演是维伦纽瓦", "timestamp": _NEW_TS}],
        )
        rows = _facts(app.state.service, "u-low")
    assert {row["object"] for row in rows} == {"诺兰", "维伦纽瓦"}
    assert all(row["status"] == "active" for row in rows)


def test_abstain_uses_relevance_not_src_bonus(tmp_path):
    """One shared token is above the rerank floor and below TAU.

    src_bonus on a stated turn lifts ``final`` over TAU. Abstention must still
    return an empty list. A real overlap still retrieves the memory.
    """
    client, _app = _client(tmp_path)
    with client:
        _add(
            client,
            "abs-1",
            "u-abs",
            [{"role": "user", "content": "My cat is named Luna.", "timestamp": _OLD_TS}],
        )
        weak = client.post(
            "/search",
            json={
                "query": "luna zircon neutrino boson phonon plasmon muon gluon lepton photon",
                "user_id": "u-abs",
                "top_k": 5,
            },
        )
        assert weak.status_code == 200
        assert weak.json()["data"] == []
        strong = _blob(client, "u-abs", "What is my cat named?")
    assert "Luna" in strong
