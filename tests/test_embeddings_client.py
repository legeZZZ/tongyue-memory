import json

import httpx

from mneme.embeddings import DashScopeEmbedder
from mneme.llm import ANALYZE_SYSTEM, EXTRACT_SYSTEM, JUDGE_SYSTEM


def test_v4_payload_sends_text_type_and_dimensions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.2] * 1024}, {"index": 1, "embedding": [0.1] * 1024}]},
        )

    embedder = DashScopeEmbedder(
        base_url="https://dashscope.example/compatible-mode/v1",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    vectors = embedder.embed_texts(["hello", "world"], text_type="document")
    embedder.close()
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "text-embedding-v4"
    assert seen["body"]["dimensions"] == 1024
    assert seen["body"]["text_type"] == "document"
    assert seen["body"]["input"] == ["hello", "world"]
    assert len(vectors) == 2
    assert vectors[0].shape == (1024,)


def test_prompts_do_not_ask_the_model_to_answer():
    for prompt in (EXTRACT_SYSTEM, ANALYZE_SYSTEM, JUDGE_SYSTEM):
        assert "Do not answer" in prompt or "do not answer" in prompt
    assert "gold" not in ANALYZE_SYSTEM.lower()
