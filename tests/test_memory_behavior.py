"""Bitemporal filter, CJK index, disclosure, and ordering."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from mneme.store import Store


def test_neighbor_expand_then_truncate(api):
    client, _service = api
    assert client.post(
        "/add",
        json={
            "request_id": "nbr-1",
            "user_id": "u-nbr",
            "session_id": "s",
            "messages": [
                {"role": "user", "content": "alpha preamble unique"},
                {"role": "user", "content": "the codeword is xylophone"},
                {"role": "user", "content": "omega epilogue unique"},
            ],
        },
    ).status_code == 200
    tight = client.post(
        "/search",
        json={"query": "xylophone", "user_id": "u-nbr", "top_k": 1},
    ).json()["data"]
    assert len(tight) == 1
    assert "xylophone" in tight[0]["content"]
    wide = client.post(
        "/search",
        json={"query": "xylophone", "user_id": "u-nbr", "top_k": 6},
    ).json()["data"]
    assert len(wide) <= 6
    blob = " ".join(item["content"] for item in wide)
    assert "xylophone" in blob
    assert "alpha preamble" in blob
    assert "omega epilogue" in blob
    assert len({item["id"] for item in wide}) == len(wide)


def test_chinese_query_hits_jieba_index_not_a_single_cjk_blob(api):
    client, service = api
    response = client.post(
        "/add",
        json={
            "request_id": "cn-1",
            "user_id": "u-cn",
            "session_id": "s",
            "messages": [{"role": "user", "content": "用户最喜欢的导演是诺兰", "timestamp": 1704067200000}],
        },
    )
    assert response.status_code == 200
    assert service.store.fts_mode in {"jieba+trigram", "jieba", "simple"}
    if service.store.fts_mode != "simple":
        tokens = service.store.fts_tokens("t:1")
        assert tokens is not None
        parts = tokens.split()
        assert "用户最喜欢的导演是诺兰" not in parts
        assert any("导演" in part for part in parts)
    found = client.post(
        "/search",
        json={"query": "导演", "user_id": "u-cn", "top_k": 5},
    )
    assert any("诺兰" in item["content"] for item in found.json()["data"])


def test_current_hides_superseded_value(api):
    client, _service = api
    _directors(client)
    found = client.post(
        "/search",
        json={"query": "现在最喜欢的导演是谁", "user_id": "u-time", "top_k": 10},
    )
    blob = " ".join(item["content"] for item in found.json()["data"])
    assert "维伦纽瓦" in blob
    assert "诺兰" not in blob


def test_as_of_returns_the_value_inside_the_interval(api):
    client, _service = api
    _directors(client)
    found = client.post(
        "/search",
        json={"query": "在2024-06-01时喜欢的导演", "user_id": "u-time", "top_k": 10},
    )
    blob = " ".join(item["content"] for item in found.json()["data"])
    assert "诺兰" in blob
    assert "维伦纽瓦" not in blob


def test_delta_returns_both_sides_of_the_update(api):
    client, _service = api
    _directors(client)
    found = client.post(
        "/search",
        json={"query": "对导演的看法变了吗", "user_id": "u-time", "top_k": 10},
    )
    blob = " ".join(item["content"] for item in found.json()["data"])
    assert "诺兰" in blob
    assert "维伦纽瓦" in blob


def test_order_lists_earlier_event_first(api):
    client, _service = api
    assert client.post(
        "/add",
        json={
            "request_id": "ord-1",
            "user_id": "u-ord",
            "session_id": "s",
            "messages": [{"role": "user", "content": "我先去了东京", "timestamp": 1704067200000}],
        },
    ).status_code == 200
    assert client.post(
        "/add",
        json={
            "request_id": "ord-2",
            "user_id": "u-ord",
            "session_id": "s",
            "messages": [{"role": "user", "content": "然后我去了巴黎", "timestamp": 1706745600000}],
        },
    ).status_code == 200
    data = client.post(
        "/search",
        json={"query": "我是先去东京还是先去巴黎", "user_id": "u-ord", "top_k": 10},
    ).json()["data"]
    contents = [item["content"] for item in data]
    tokyo = next(i for i, text in enumerate(contents) if "东京" in text)
    paris = next(i for i, text in enumerate(contents) if "巴黎" in text)
    assert tokyo < paris


def test_sensitive_fact_is_withheld_when_off_topic(api):
    client, _service = api
    assert client.post(
        "/add",
        json={
            "request_id": "sec-1",
            "user_id": "u-sec",
            "session_id": "s",
            "messages": [
                {"role": "user", "content": "我的猫叫露娜"},
                {"role": "user", "content": "我的身份证号是110101199001011234"},
            ],
        },
    ).status_code == 200
    about_cat = client.post(
        "/search",
        json={"query": "我的猫叫什么", "user_id": "u-sec", "top_k": 10},
    ).json()["data"]
    blob = " ".join(item["content"] for item in about_cat)
    assert "露娜" in blob
    assert "身份证" not in blob

    about_id = client.post(
        "/search",
        json={"query": "身份证号是多少", "user_id": "u-sec", "top_k": 10},
    ).json()["data"]
    assert any("身份证" in item["content"] for item in about_id)


def test_parallel_same_request_writes_once(tmp_path):
    from mneme.embeddings import HashEmbedder
    from mneme.service import MemoryService

    settings = __import__("tests.conftest", fromlist=["make_settings"]).make_settings(tmp_path)
    service = MemoryService(settings, embedder=HashEmbedder())
    body = {
        "request_id": "same",
        "user_id": "u",
        "session_id": "s",
        "messages": [{"role": "user", "content": "parallel write of one memory"}],
    }
    from mneme.schemas import AddRequest

    req = AddRequest.model_validate(body)
    errors = []

    def once():
        try:
            service.add(req)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=once) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert service.store.count_turns("u") == 1
    service.close()


def test_store_opens_fts_without_unicode61_raw_cjk(tmp_path):
    store = Store(str(tmp_path / "fts.db"))
    try:
        assert store.fts_mode != "unicode61"
        assert store.fts_mode in {"jieba+trigram", "jieba", "simple"}
    finally:
        store.close()


def _directors(client: TestClient) -> None:
    response = client.post(
        "/add",
        json={
            "request_id": "dir-1",
            "user_id": "u-time",
            "session_id": "s",
            "messages": [
                {
                    "role": "user",
                    "content": "用户的最喜欢导演是诺兰",
                    "timestamp": 1704067200000,
                },
                {
                    "role": "user",
                    "content": "用户的最喜欢导演是维伦纽瓦",
                    "timestamp": 1735689600000,
                },
            ],
        },
    )
    assert response.status_code == 200
