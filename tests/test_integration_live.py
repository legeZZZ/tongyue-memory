"""Optional live calls. Skipped unless both provider keys are present."""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("EMBEDDING_API_KEY"),
    reason="no embedding key",
)
def test_live_embedding_shape():
    from mneme.config import Settings
    from mneme.embeddings import DashScopeEmbedder

    settings = Settings.from_env()
    embedder = DashScopeEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key or "",
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    try:
        query = embedder.embed_texts(["where did I leave the spare key"], text_type="query")
        document = embedder.embed_texts(["I left the spare key under the blue pot."], text_type="document")
    finally:
        embedder.close()
    assert query[0].shape == (settings.embedding_dim,)
    assert document[0].shape == (settings.embedding_dim,)
