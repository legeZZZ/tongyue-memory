"""App fixtures. Models are fakes; no network and no API keys."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from mneme.app import create_app
from mneme.config import Settings
from mneme.service import MemoryService


def make_settings(tmp_path, **overrides) -> Settings:
    params = dict(
        db_path=str(tmp_path / "mneme.db"),
        fake_models=True,
        disable_worker=True,
        fact_timeout_s=5.0,
        memory_system_key=None,
        tau=0.25,
        rerank_floor=0.08,
        recall_limit=100,
        neighbor_radius=1,
    )
    params.update(overrides)
    return Settings(**params)


@pytest.fixture
def api(tmp_path) -> Iterator[tuple[TestClient, MemoryService]]:
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as test_client:
        yield test_client, app.state.service
