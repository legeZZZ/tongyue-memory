"""Runtime configuration from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


@dataclass(frozen=True)
class Settings:
    """Process settings. Academic locks: text-embedding-v4 (1024d) and gpt-4o-mini."""

    db_path: str = "mneme.db"
    memory_system_key: str | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    embedding_batch_size: int = 10
    embedding_timeout_s: float = 40.0
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0
    reranker: str = "lexical"
    qwen_model: str = "Qwen/Qwen3-Reranker-0.6B"
    tau: float = 0.25
    beta: float = 0.5
    gamma: float = 0.2
    rrf_k: int = 60
    rerank_floor: float = 0.08
    recall_limit: int = 100
    rerank_max: int = 80
    neighbor_radius: int = 1
    fact_timeout_s: float = 12.0
    fake_models: bool = False
    fts_simple_path: str | None = None
    disable_worker: bool = False
    host: str = "0.0.0.0"
    port: int = 8097

    @classmethod
    def from_env(cls) -> Settings:
        embedding_key = _str("EMBEDDING_API_KEY") or _str("DASHSCOPE_API_KEY")
        return cls(
            db_path=_str("MNEME_DB", "mneme.db") or "mneme.db",
            memory_system_key=_str("MEMORY_SYSTEM_KEY"),
            embedding_base_url=_str(
                "EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_api_key=embedding_key,
            embedding_model=_str("EMBEDDING_MODEL", "text-embedding-v4") or "text-embedding-v4",
            embedding_dim=_int("EMBEDDING_DIM", 1024),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 10),
            embedding_timeout_s=_float("EMBEDDING_TIMEOUT_S", 40.0),
            openai_base_url=_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
            or "https://api.openai.com/v1",
            openai_api_key=_str("OPENAI_API_KEY"),
            llm_model=_str("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            llm_timeout_s=_float("LLM_TIMEOUT_S", 30.0),
            reranker=(_str("MNEME_RERANKER", "lexical") or "lexical").lower(),
            qwen_model=_str("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
            or "Qwen/Qwen3-Reranker-0.6B",
            tau=_float("MNEME_TAU", 0.25),
            beta=_float("MNEME_BETA", 0.5),
            gamma=_float("MNEME_GAMMA", 0.2),
            rrf_k=_int("MNEME_RRF_K", 60),
            rerank_floor=_float("MNEME_RERANK_FLOOR", 0.08),
            recall_limit=_int("MNEME_RECALL_LIMIT", 100),
            rerank_max=_int("MNEME_RERANK_MAX", 80),
            neighbor_radius=_int("MNEME_NEIGHBOR_RADIUS", 1),
            fact_timeout_s=_float("MNEME_FACT_TIMEOUT_S", 12.0),
            fake_models=_flag("MNEME_FAKE_MODELS", False),
            fts_simple_path=_str("MNEME_FTS_SIMPLE"),
            disable_worker=_flag("MNEME_DISABLE_WORKER", False),
            host=_str("HOST", "0.0.0.0") or "0.0.0.0",
            port=_int("PORT", 8097),
        )
