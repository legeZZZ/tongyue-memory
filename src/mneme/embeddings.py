"""Dense embeddings. Academic lock: text-embedding-v4, 1024 dimensions.

``text_type`` is ``document`` on Add and ``query`` on Search. The default client
speaks the DashScope OpenAI-compatible embeddings endpoint and always sends
``text_type``. Tests use :class:`HashEmbedder`, which needs no network.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

import httpx
import numpy as np

from mneme.tokenize import tokenize

log = logging.getLogger(__name__)


class UpstreamError(Exception):
    """Embedding or LLM dependency failed before a durable write."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


class Embedder(Protocol):
    dim: int

    def embed_texts(self, texts: list[str], *, text_type: str) -> list[np.ndarray]:
        """Return one L2-normalized vector per text. ``text_type`` is query or document."""


class HashEmbedder:
    """Deterministic bag-of-tokens embedder for tests and offline smoke.

    Shared tokens land in the same coordinates, so cosine similarity tracks
    lexical overlap. This is not a substitute for text-embedding-v4 in eval.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str], *, text_type: str) -> list[np.ndarray]:
        if text_type not in {"query", "document"}:
            raise ValueError("text_type must be query or document")
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return l2_normalize(vec)


class DashScopeEmbedder:
    """OpenAI-compatible text-embedding-v4 client (DashScope by default)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "text-embedding-v4",
        dim: int = 1024,
        batch_size: int = 10,
        timeout_s: float = 40.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise UpstreamError("embedding API key not configured (DASHSCOPE_API_KEY or EMBEDDING_API_KEY)")
        self.dim = dim
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout_s = timeout_s
        self._url = base_url.rstrip("/") + "/embeddings"
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def embed_texts(self, texts: list[str], *, text_type: str) -> list[np.ndarray]:
        if text_type not in {"query", "document"}:
            raise ValueError("text_type must be query or document")
        if not texts:
            return []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        ordered: list[list[np.ndarray] | None] = [None] * len(batches)
        workers = min(16, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._embed_batch, batch, text_type): index for index, batch in enumerate(batches)
            }
            for fut in as_completed(futures):
                ordered[futures[fut]] = fut.result()
        out: list[np.ndarray] = []
        for batch_vecs in ordered:
            if batch_vecs is None:
                raise UpstreamError("embedding batch missing")
            out.extend(batch_vecs)
        return out

    def _embed_batch(self, texts: list[str], text_type: str) -> list[np.ndarray]:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dim,
            "encoding_format": "float",
            "text_type": text_type,
        }
        try:
            response = self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"embedding request failed: {exc.__class__.__name__}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise UpstreamError(f"embedding upstream HTTP {response.status_code}")
        if response.status_code >= 400:
            raise UpstreamError(f"embedding upstream HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("embedding response was not JSON") from exc
        return _parse_embeddings(body, expected=len(texts), dim=self.dim)


def _parse_embeddings(payload: dict, *, expected: int, dim: int) -> list[np.ndarray]:
    vectors: list[list[float]] | None = None
    data = payload.get("data")
    if isinstance(data, list):
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
    else:
        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("embeddings"), list):
            ordered = sorted(output["embeddings"], key=lambda item: item.get("text_index", 0))
            vectors = [item["embedding"] for item in ordered]
    if vectors is None:
        raise UpstreamError("unexpected embedding response shape")
    if len(vectors) != expected:
        raise UpstreamError("embedding count mismatch")
    out: list[np.ndarray] = []
    for vec in vectors:
        if len(vec) != dim:
            raise UpstreamError(f"expected {dim} dimensions, got {len(vec)}")
        out.append(l2_normalize(np.asarray(vec, dtype=np.float32)))
    return out


class MissingKeyEmbedder:
    """Keeps /health up when the embedding key is not configured yet."""

    dim = 1024

    def embed_texts(self, texts: list[str], *, text_type: str) -> list[np.ndarray]:
        del texts, text_type
        raise UpstreamError(
            "embedding API key not configured (set DASHSCOPE_API_KEY or EMBEDDING_API_KEY)"
        )


def build_embedder(settings) -> Embedder:
    if settings.fake_models:
        return HashEmbedder(dim=settings.embedding_dim)
    if not settings.embedding_api_key:
        return MissingKeyEmbedder()
    return DashScopeEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
        timeout_s=settings.embedding_timeout_s,
    )
