"""Live memory regressions against gpt-4o-mini and text-embedding-v4.

Skipped unless both keys are exported. ``MNEME_FAKE_MODELS`` is ignored;
this module builds real clients. No secrets belong in the repo.

tryallapi (chat) plus DashScope (embeddings)::

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=https://tryallapi.com/v1
    export LLM_MODEL=gpt-4o-mini
    export DASHSCOPE_API_KEY=...
    export EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    export EMBEDDING_MODEL=text-embedding-v4
    pytest -m real_llm

Omit ``OPENAI_BASE_URL`` to call ``https://api.openai.com/v1``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.config import Settings

pytestmark = pytest.mark.real_llm


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


def _msg(content: str, year: int, month: int, day: int) -> dict:
    return {"role": "user", "content": content, "timestamp": _ms(year, month, day)}


# Each message is its own /add so the second fact is judged against the first.
# ``superseded`` asks the store for a closed fact plus an active fact that
# still carries ``active_has`` (extract + supersede, not only the turn dedupe).
CASES: list[dict] = [
    {
        "id": "current-director-zh",
        "messages": [
            _msg("我最喜欢的导演是诺兰", 2024, 1, 1),
            _msg("我最喜欢的导演是维伦纽瓦", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "维伦纽瓦",
        "checks": [
            {
                "query": "现在我最喜欢的导演是谁",
                "include": ["维伦纽瓦"],
                "exclude": ["诺兰"],
            }
        ],
    },
    {
        "id": "current-color-en",
        "messages": [
            _msg("My favorite color is blue.", 2024, 1, 1),
            _msg("My favorite color is green.", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "green",
        "checks": [
            {
                "query": "What is my favorite color right now?",
                "include": ["green"],
                "exclude": ["blue"],
            }
        ],
    },
    {
        "id": "current-city-zh",
        "messages": [
            _msg("我住在上海", 2023, 3, 1),
            _msg("我搬到了杭州，现在住在杭州", 2024, 8, 1),
        ],
        "superseded": True,
        "active_has": "杭州",
        "checks": [
            {"query": "我现在住在哪个城市", "include": ["杭州"], "exclude": ["上海"]}
        ],
    },
    {
        "id": "current-job-en",
        "messages": [
            _msg("I work as a teacher.", 2022, 5, 1),
            _msg("I work as a doctor.", 2025, 2, 1),
        ],
        "superseded": True,
        "active_has": "doctor",
        "checks": [
            {
                "query": "What is my job right now?",
                "include": ["doctor"],
                "exclude": ["teacher"],
            }
        ],
    },
    {
        "id": "current-pets-en",
        "messages": [
            _msg("I have one cat.", 2024, 1, 1),
            _msg("I have two cats.", 2025, 6, 1),
        ],
        "superseded": True,
        "active_has": "two",
        "checks": [
            {
                "query": "How many cats do I have right now?",
                "include": ["two"],
                "exclude": ["one cat"],
            }
        ],
    },
    {
        "id": "current-drink-zh",
        "messages": [
            _msg("我喝的是咖啡", 2024, 1, 1),
            _msg("我喝的是茶", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "茶",
        "checks": [{"query": "我现在喝什么", "include": ["茶"], "exclude": ["咖啡"]}],
    },
    {
        "id": "current-dog-en",
        "messages": [
            _msg("My dog is named Max.", 2024, 4, 1),
            _msg("My dog is named Coco.", 2025, 4, 1),
        ],
        "superseded": True,
        "active_has": "Coco",
        "checks": [
            {
                "query": "What is my dog named right now?",
                "include": ["Coco"],
                "exclude": ["Max"],
            }
        ],
    },
    {
        "id": "current-city-wording-en",
        "messages": [
            _msg("I used to live in Boston.", 2023, 1, 15),
            _msg("I now live in Seattle.", 2025, 1, 15),
        ],
        "superseded": True,
        "active_has": "Seattle",
        "checks": [
            {
                "query": "Where do I live right now?",
                "include": ["Seattle"],
                "exclude": ["Boston"],
            }
        ],
    },
    {
        "id": "drift-director-zh",
        "messages": [
            _msg("我最喜欢的导演是诺兰", 2024, 1, 1),
            _msg("我现在最推崇的电影导演是维伦纽瓦", 2025, 3, 1),
        ],
        "superseded": True,
        "active_has": "维伦纽瓦",
        "checks": [
            {
                "query": "现在我最喜欢的导演是谁",
                "include": ["维伦纽瓦"],
                "exclude": ["诺兰"],
            }
        ],
    },
    {
        "id": "asof-director-zh",
        "messages": [
            _msg("我最喜欢的导演是诺兰", 2024, 1, 1),
            _msg("我最喜欢的导演是维伦纽瓦", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "维伦纽瓦",
        "checks": [
            {
                "query": "在2024年6月1日的时候，我最喜欢的导演是谁",
                "include": ["诺兰"],
                "exclude": ["维伦纽瓦"],
            }
        ],
    },
    {
        "id": "asof-city-before-en",
        "messages": [
            _msg("I live in Boston.", 2023, 3, 1),
            _msg("I live in Seattle.", 2024, 8, 1),
        ],
        "superseded": True,
        "active_has": "Seattle",
        "checks": [
            {
                "query": "As of 2023-06-01, which city did I live in?",
                "include": ["Boston"],
                "exclude": ["Seattle"],
            }
        ],
    },
    {
        "id": "asof-city-after-en",
        "messages": [
            _msg("I live in Boston.", 2023, 3, 1),
            _msg("I live in Seattle.", 2024, 8, 1),
        ],
        "superseded": True,
        "active_has": "Seattle",
        "checks": [
            {
                "query": "As of 2025-03-01, which city did I live in?",
                "include": ["Seattle"],
                "exclude": ["Boston"],
            }
        ],
    },
    {
        "id": "asof-drink-zh",
        "messages": [
            _msg("我喝的是咖啡", 2024, 1, 1),
            _msg("我喝的是茶", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "茶",
        "checks": [
            {
                "query": "在2024年6月1日的时候我喝什么",
                "include": ["咖啡"],
                "exclude": ["茶"],
            }
        ],
    },
    {
        "id": "delta-director-zh",
        "messages": [
            _msg("我最喜欢的导演是诺兰", 2024, 1, 1),
            _msg("我最喜欢的导演是维伦纽瓦", 2025, 1, 1),
        ],
        "superseded": True,
        "active_has": "维伦纽瓦",
        "checks": [
            {
                "query": "我对导演的喜好有什么变化吗",
                "include": ["诺兰", "维伦纽瓦"],
                "exclude": [],
            }
        ],
    },
    {
        "id": "unrelated-cat-and-city",
        "messages": [
            _msg("我的猫叫露娜", 2024, 2, 2),
            _msg("我住在巴黎", 2024, 2, 2),
        ],
        "checks": [
            {"query": "我的猫叫什么", "include": ["露娜"], "exclude": []},
            {"query": "我住在哪个城市", "include": ["巴黎"], "exclude": []},
        ],
    },
    {
        "id": "order-tokyo-paris",
        "messages": [
            _msg("我先去了东京", 2024, 1, 1),
            _msg("然后我去了巴黎", 2024, 5, 1),
        ],
        "checks": [{"query": "我是先去东京还是先去巴黎", "order": ["东京", "巴黎"]}],
    },
    {
        "id": "repeat-luna-still-found",
        "messages": [
            _msg("My cat is named Luna.", 2024, 1, 1),
            _msg("My cat is named Luna.", 2024, 6, 1),
        ],
        "checks": [{"query": "What is my cat named?", "include": ["Luna"], "exclude": []}],
    },
]


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    embed_key = (
        os.environ.get("DASHSCOPE_API_KEY", "").strip()
        or os.environ.get("EMBEDDING_API_KEY", "").strip()
    )
    if not openai_key or not embed_key:
        pytest.skip(
            "real LLM regression needs OPENAI_API_KEY and DASHSCOPE_API_KEY or EMBEDDING_API_KEY"
        )
    root = tmp_path_factory.mktemp("mneme-live")
    settings = Settings(
        db_path=str(root / "mneme.db"),
        fake_models=False,
        disable_worker=True,
        fact_timeout_s=120.0,
        llm_timeout_s=60.0,
        embedding_timeout_s=60.0,
        memory_system_key=None,
        openai_api_key=openai_key,
        openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
        embedding_api_key=embed_key,
        embedding_base_url=_env(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-v4"),
        embedding_dim=int(_env("EMBEDDING_DIM", "1024")),
        tau=0.25,
        rerank_floor=0.08,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, app.state.service


def _facts(service, user_id: str):
    with service.store._lock:
        return service.store.conn.execute(
            """
            SELECT subject, predicate, object, status, statement
            FROM facts WHERE user_id = ? ORDER BY id
            """,
            (user_id,),
        ).fetchall()


def _has(haystack: str, needle: str) -> bool:
    if needle.isascii():
        return needle.casefold() in haystack.casefold()
    return needle in haystack


def _fact_blob(rows, status: str) -> str:
    parts: list[str] = []
    for row in rows:
        if row["status"] != status:
            continue
        parts.extend(str(row[key] or "") for key in ("statement", "object", "predicate", "subject"))
    return " ".join(parts)


def _search(client: TestClient, user_id: str, query: str) -> list[dict]:
    response = client.post(
        "/search",
        json={"query": query, "user_id": user_id, "top_k": 10},
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_live_memory_case(live, case):
    client, service = live
    user = case["id"]
    for index, message in enumerate(case["messages"]):
        response = client.post(
            "/add",
            json={
                "request_id": f"{user}-{index}",
                "user_id": user,
                "session_id": "live",
                "messages": [message],
            },
        )
        assert response.status_code == 200, response.text
    if case.get("superseded"):
        rows = _facts(service, user)
        rendered = [dict(row) for row in rows]
        assert any(row["status"] == "superseded" for row in rows), rendered
        assert _has(_fact_blob(rows, "active"), case["active_has"]), rendered
    for check in case["checks"]:
        data = _search(client, user, check["query"])
        text = " ".join(item["content"] for item in data)
        for needle in check.get("include", []):
            assert _has(text, needle), f"{case['id']} missing {needle!r} in {text!r}"
        for needle in check.get("exclude", []):
            assert not _has(text, needle), f"{case['id']} leaked {needle!r} in {text!r}"
        order = check.get("order")
        if order:
            contents = [item["content"] for item in data]
            earlier = next(i for i, item in enumerate(contents) if order[0] in item)
            later = next(i for i, item in enumerate(contents) if order[1] in item)
            assert earlier < later, contents
