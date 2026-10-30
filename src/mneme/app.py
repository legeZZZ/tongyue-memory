"""AML HTTP surface: POST /add, POST /search, GET /health."""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from mneme import __version__
from mneme.config import Settings
from mneme.embeddings import UpstreamError
from mneme.schemas import AddRequest, SearchRequest
from mneme.service import MemoryService
from mneme.store import Conflict

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    embedder=None,
    llm=None,
    reranker=None,
) -> FastAPI:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    cfg = settings or Settings.from_env()
    service = MemoryService(cfg, embedder=embedder, llm=llm, reranker=reranker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service.start()
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(
        title="同月记忆",
        description="tongyue-memory textual memory service (academic track). Import package: mneme.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        # Unauthenticated on purpose. Any 2xx means the process can serve Add.
        return {"status": "ok", "version": __version__, "fts": service.store.fts_mode}

    @app.post("/add")
    def add(body: AddRequest, request: Request):
        _authorize(request, cfg)
        try:
            return service.add(body)
        except Conflict:
            raise HTTPException(
                status_code=409,
                detail={"reason": "request_id already stored with a different body"},
            ) from None
        except UpstreamError as exc:
            return JSONResponse(
                status_code=503,
                content={"detail": {"reason": exc.message}},
                headers={"Retry-After": "2"},
            )

    @app.post("/search")
    def search(body: SearchRequest, request: Request):
        _authorize(request, cfg)
        try:
            return service.search(body)
        except UpstreamError as exc:
            return JSONResponse(
                status_code=503,
                content={"detail": {"reason": exc.message}},
                headers={"Retry-After": "2"},
            )

    return app


def _presented(request: Request) -> list[str]:
    tokens: list[str] = []
    header = request.headers.get("authorization", "")
    lowered = header.lower()
    if lowered.startswith("bearer "):
        tokens.append(header[7:].strip())
    elif lowered.startswith("token "):
        tokens.append(header[6:].strip())
    api_key = request.headers.get("x-api-key")
    if api_key:
        tokens.append(api_key.strip())
    return [token for token in tokens if token]


def _authorize(request: Request, settings: Settings) -> None:
    expected = settings.memory_system_key
    if not expected:
        return
    for token in _presented(request):
        try:
            if hmac.compare_digest(token, expected):
                return
        except (TypeError, ValueError):
            continue
    raise HTTPException(status_code=401, detail={"reason": "invalid credentials"})
