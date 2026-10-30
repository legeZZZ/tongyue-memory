"""Offline stand-ins for the six gpt-4o-mini failures.

Live ``pytest -m real_llm`` needs ``OPENAI_API_KEY`` and ``DASHSCOPE_API_KEY``
(or ``EMBEDDING_API_KEY``). See the module docstring in
``tests/test_real_llm_regression.py`` for the tryallapi + DashScope command.
These tests use fake models and must fail if supersede, current-tense hiding,
or the job retrieval path regresses.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.llm import Analysis, FactDraft, HeuristicLLM, Verdict
from mneme.rerank import LexicalReranker
from tests.conftest import make_settings


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def _msg(content: str, year: int, month: int, day: int) -> dict:
    return {"role": "user", "content": content, "timestamp": _ms(year, month, day)}


class _Orthogonal:
    """Distinct texts get orthogonal vectors, so cosine cannot clear 0.82."""

    dim = 32

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def embed_texts(self, texts, *, text_type):
        if text_type not in {"query", "document"}:
            raise ValueError(text_type)
        out = []
        for text in texts:
            if text not in self._ids:
                self._ids[text] = len(self._ids)
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[self._ids[text] % self.dim] = 1.0
            out.append(vec)
        return out


class _Scripted(HeuristicLLM):
    """Extract the drifted SPO the live model actually stored."""

    def __init__(self, rows: list[tuple[str, str, str, str]]) -> None:
        self.rows = rows

    def extract(self, turns):
        drafts: list[FactDraft] = []
        for turn in turns:
            for needle, subject, predicate, obj in self.rows:
                if needle in turn.content:
                    drafts.append(
                        FactDraft(
                            statement=turn.content.strip(),
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            ts_event=turn.ts_event,
                            provenance="stated",
                            confidence=1.0,
                            sensitivity="normal",
                            src_turn_id=turn.id,
                        )
                    )
                    break
        return drafts


class _Labeled(_Scripted):
    def __init__(self, rows: list[tuple[str, str, str, str]], tense: str) -> None:
        super().__init__(rows)
        self._tense = tense

    def analyze(self, query: str) -> Analysis:
        base = super().analyze(query)
        return Analysis(tense=self._tense, as_of=None, hops=base.hops, recall_q=base.recall_q)


class _NoSupersede(HeuristicLLM):
    """Heuristic subclass: the service must not replace this judge with the backup."""

    def judge(self, new, candidates):
        del new, candidates
        return Verdict("unrelated", None)


class _DenyJudge:
    """Not a HeuristicLLM. Backup heuristic still has to supersede."""

    def __init__(self, inner: _Scripted) -> None:
        self.inner = inner

    def extract(self, turns):
        return self.inner.extract(turns)

    def analyze(self, query: str) -> Analysis:
        return self.inner.analyze(query)

    def judge(self, new, candidates):
        del new, candidates
        return Verdict("unrelated", None)


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
    assert response.status_code == 200, response.text


def _search(client: TestClient, user_id: str, query: str) -> str:
    found = client.post("/search", json={"query": query, "user_id": user_id, "top_k": 10})
    assert found.status_code == 200, found.text
    return " ".join(item["content"] for item in found.json()["data"])


def _rows(service, user_id: str):
    with service.store._lock:
        return service.store.conn.execute(
            """
            SELECT subject, predicate, object, status, statement
            FROM facts WHERE user_id = ? ORDER BY id
            """,
            (user_id,),
        ).fetchall()


def _blob(row) -> str:
    return " ".join(str(row[key] or "") for key in ("statement", "object", "predicate", "subject"))


def _assert_closed(rows, old: str, new: str) -> None:
    superseded = [_blob(row) for row in rows if row["status"] == "superseded"]
    active = [_blob(row) for row in rows if row["status"] == "active"]
    assert superseded, [dict(row) for row in rows]
    assert any(old in text for text in superseded), [dict(row) for row in rows]
    assert any(new in text for text in active), [dict(row) for row in rows]
    assert not any(old in text for text in active), [dict(row) for row in rows]


def _pair(client, user: str, messages: list[dict]) -> None:
    for index, message in enumerate(messages):
        _add(client, f"{user}-{index}", user, [message])


def test_judge_updates_pet_and_residence_drift_at_zero_similarity():
    llm = HeuristicLLM()
    pets = llm.judge(
        FactDraft(
            statement="I have two cats.",
            subject="User",
            predicate="has",
            object="two cats",
            ts_event="2025-06-01T00:00:00Z",
            provenance="stated",
            confidence=1.0,
            sensitivity="normal",
        ),
        [
            {
                "id": 3,
                "subject": "I",
                "predicate": "have",
                "object": "one cat",
                "statement": "I have one cat.",
                "status": "active",
                "valid_to": None,
                "same_spo": False,
                "sim": 0.0,
            }
        ],
    )
    assert pets == Verdict("updates", 3)
    cities = llm.judge(
        FactDraft(
            statement="我搬到了杭州，现在住在杭州",
            subject="",
            predicate="",
            object="",
            ts_event="2024-08-01T00:00:00Z",
            provenance="stated",
            confidence=1.0,
            sensitivity="normal",
        ),
        [
            {
                "id": 4,
                "subject": "",
                "predicate": "",
                "object": "",
                "statement": "我住在上海",
                "status": "active",
                "valid_to": None,
                "same_spo": False,
                "sim": 0.0,
            }
        ],
    )
    assert cities == Verdict("updates", 4)


def test_judge_keeps_sim_gate_for_unrecognized_director_paraphrase():
    """Pinned pair: 最爱 vs 喜欢 is not a known fold. sim 0.81 must stay unrelated."""
    llm = HeuristicLLM()
    verdict = llm.judge(
        FactDraft(
            statement="用户目前最爱的导演是维伦纽瓦",
            subject="用户",
            predicate="目前最爱的导演",
            object="维伦纽瓦",
            ts_event="2025-01-01T00:00:00Z",
            provenance="stated",
            confidence=1.0,
            sensitivity="normal",
        ),
        [
            {
                "id": 7,
                "subject": "用户",
                "predicate": "最喜欢的导演",
                "object": "诺兰",
                "statement": "用户最喜欢的导演是诺兰",
                "status": "active",
                "valid_to": None,
                "same_spo": False,
                "sim": 0.81,
            }
        ],
    )
    assert verdict == Verdict("unrelated", None)


def test_job_query_clears_tau_without_lifting_irrelevant_overlap():
    reranker = LexicalReranker()
    job = reranker.score("What is my job right now?", ["I work as a doctor."])[0]
    assert job >= 0.25
    junk = reranker.score(
        "luna zircon neutrino boson phonon plasmon muon gluon lepton photon",
        ["My cat is named Luna."],
    )[0]
    assert junk < 0.25


_DRIFT = [
    {
        "id": "current-city-zh",
        "messages": [
            _msg("我住在上海", 2023, 3, 1),
            _msg("我搬到了杭州，现在住在杭州", 2024, 8, 1),
        ],
        "rows": [
            ("上海", "我", "住在", "上海"),
            ("杭州", "我", "搬到", "杭州"),
        ],
        "old": "上海",
        "new": "杭州",
        "query": "我现在住在哪个城市",
        "exclude": "上海",
    },
    {
        "id": "current-pets-en",
        "messages": [
            _msg("I have one cat.", 2024, 1, 1),
            _msg("I have two cats.", 2025, 6, 1),
        ],
        "rows": [
            ("one cat", "I", "have", "one cat"),
            ("two cats", "User", "has", "two cats"),
        ],
        "old": "one cat",
        "new": "two cats",
        "query": "How many cats do I have right now?",
        "exclude": "one cat",
    },
    {
        "id": "current-city-wording-en",
        "messages": [
            _msg("I used to live in Boston.", 2023, 1, 15),
            _msg("I now live in Seattle.", 2025, 1, 15),
        ],
        "rows": [
            ("Boston", "User", "used to live in", "Boston"),
            ("Seattle", "User", "lives in", "Seattle"),
        ],
        "old": "Boston",
        "new": "Seattle",
        "query": "Where do I live right now?",
        "exclude": "Boston",
    },
    {
        "id": "drift-director-zh",
        "messages": [
            _msg("我最喜欢的导演是诺兰", 2024, 1, 1),
            _msg("我现在最推崇的电影导演是维伦纽瓦", 2025, 3, 1),
        ],
        "rows": [
            ("诺兰", "我", "最喜欢的导演是", "诺兰"),
            ("维伦纽瓦", "用户", "最推崇的电影导演", "维伦纽瓦"),
        ],
        "old": "诺兰",
        "new": "维伦纽瓦",
        "query": "现在我最喜欢的导演是谁",
        "exclude": "诺兰",
    },
    {
        "id": "asof-city-after-en",
        "messages": [
            _msg("I live in Boston.", 2023, 3, 1),
            _msg("I live in Seattle.", 2024, 8, 1),
        ],
        "rows": [
            ("Boston", "I", "live in", "Boston"),
            ("Seattle", "User", "lives in", "Seattle"),
        ],
        "old": "Boston",
        "new": "Seattle",
        "query": "As of 2025-03-01, which city did I live in?",
        "exclude": "Boston",
        "also": [
            ("As of 2023-06-01, which city did I live in?", "Boston", "Seattle"),
        ],
    },
    {
        "id": "current-job-en",
        "messages": [
            _msg("I work as a teacher.", 2022, 5, 1),
            _msg("I work as a doctor.", 2025, 2, 1),
        ],
        "rows": [
            ("teacher", "I", "work as", "teacher"),
            ("doctor", "User", "works as", "doctor"),
        ],
        "old": "teacher",
        "new": "doctor",
        "query": "What is my job right now?",
        "exclude": "teacher",
    },
]


@pytest.mark.parametrize("case", _DRIFT, ids=[case["id"] for case in _DRIFT])
def test_drifted_spo_supersedes_and_search_hides_old_value(tmp_path, case):
    llm = _Scripted(case["rows"])
    client, app = _client(tmp_path, llm=llm, embedder=_Orthogonal())
    user = case["id"]
    with client:
        _pair(client, user, case["messages"])
        rows = _rows(app.state.service, user)
        _assert_closed(rows, case["old"], case["new"])
        text = _search(client, user, case["query"])
        assert case["new"] in text, text
        assert case["exclude"] not in text, text
        for query, include, exclude in case.get("also", []):
            earlier = _search(client, user, query)
            assert include in earlier, earlier
            assert exclude not in earlier, earlier


def test_current_job_with_heuristic_extract_is_not_empty(tmp_path):
    """Default extractor stores the raw sentence. ``job`` does not occur in it."""
    client, app = _client(tmp_path)
    user = "job-heuristic"
    with client:
        _pair(
            client,
            user,
            [
                _msg("I work as a teacher.", 2022, 5, 1),
                _msg("I work as a doctor.", 2025, 2, 1),
            ],
        )
        rows = _rows(app.state.service, user)
        _assert_closed(rows, "teacher", "doctor")
        text = _search(client, user, "What is my job right now?")
    assert text.strip()
    assert "doctor" in text
    assert "teacher" not in text


@pytest.mark.parametrize("label", ["none", "delta"])
def test_current_city_hides_shanghai_when_tense_label_is_wrong(tmp_path, label):
    rows = [
        ("上海", "我", "住在", "上海"),
        ("杭州", "我", "搬到", "杭州"),
    ]
    client, _app = _client(tmp_path, llm=_Labeled(rows, label), embedder=_Orthogonal())
    user = f"city-{label}"
    with client:
        _pair(
            client,
            user,
            [
                _msg("我住在上海", 2023, 3, 1),
                _msg("我搬到了杭州，现在住在杭州", 2024, 8, 1),
            ],
        )
        text = _search(client, user, "我现在住在哪个城市")
    assert "杭州" in text, text
    assert "上海" not in text, text


def test_current_city_hides_stale_value_when_judge_refuses(tmp_path):
    client, app = _client(tmp_path, llm=_NoSupersede())
    user = "city-refuse"
    with client:
        _add(
            client,
            "city-refuse",
            user,
            [
                _msg("我住在上海", 2023, 3, 1),
                _msg("我搬到了杭州，现在住在杭州", 2024, 8, 1),
            ],
        )
        rows = _rows(app.state.service, user)
        assert rows
        assert all(row["status"] == "active" for row in rows)
        text = _search(client, user, "我现在住在哪个城市")
    assert "杭州" in text, text
    assert "上海" not in text, text


def test_unrelated_judge_backup_supersedes_pet_drift(tmp_path):
    inner = _Scripted(
        [
            ("one cat", "I", "have", "one cat"),
            ("two cats", "User", "has", "two cats"),
        ]
    )
    client, app = _client(tmp_path, llm=_DenyJudge(inner), embedder=_Orthogonal())
    user = "pets-backup"
    with client:
        _pair(
            client,
            user,
            [
                _msg("I have one cat.", 2024, 1, 1),
                _msg("I have two cats.", 2025, 6, 1),
            ],
        )
        rows = _rows(app.state.service, user)
    _assert_closed(rows, "one cat", "two cats")


def test_cat_name_city_and_dog_do_not_collapse(tmp_path):
    client, app = _client(tmp_path)
    user = "unrelated"
    with client:
        _pair(
            client,
            user,
            [
                _msg("我的猫叫露娜", 2024, 2, 2),
                _msg("我住在巴黎", 2024, 2, 2),
            ],
        )
        _pair(
            client,
            user + "-pets",
            [
                _msg("I have one cat.", 2024, 1, 1),
                _msg("I have one dog.", 2024, 3, 1),
                _msg("My cat is named Luna.", 2024, 4, 1),
            ],
        )
        home = _rows(app.state.service, user)
        pets = _rows(app.state.service, user + "-pets")
        cat = _search(client, user, "我的猫叫什么")
        city = _search(client, user, "我住在哪个城市")
    assert all(row["status"] == "active" for row in home), [dict(row) for row in home]
    assert all(row["status"] == "active" for row in pets), [dict(row) for row in pets]
    assert "露娜" in cat
    assert "巴黎" in city
