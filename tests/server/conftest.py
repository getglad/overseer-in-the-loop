"""Fixtures for src/server/ tests — the FastAPI TestClient with lifespan mocked out."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.loop.agent import LLM_API_KEY_ENV, LLM_BASE_URL_ENV, LLM_MODEL_ENV
from src.server.app import app

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="module")
def client() -> Generator[TestClient]:
    """TestClient with lifespan mocked.

    Stubs the LLM_* env vars so ``AgentSettings()`` can instantiate, and
    mocks the NAT pipeline (WorkflowBuilder, configure_builder,
    SessionManager) so no real API key, network, or NAT runtime is needed.
    """
    env_stub = {
        LLM_API_KEY_ENV: "test",
        LLM_BASE_URL_ENV: "http://test",
        LLM_MODEL_ENV: "test-model",
    }
    # NAT imports are deferred into ``lifespan`` (see src/server/app.py), so
    # we patch at the source modules — the function-local imports inside
    # ``lifespan`` will resolve to these mocks at call time.
    with (
        patch.dict(os.environ, env_stub, clear=False),
        patch("nat.builder.workflow_builder.WorkflowBuilder") as mock_wb,
        patch("src.loop.agent.configure_builder", new_callable=AsyncMock) as mock_cb,
        patch("nat.runtime.session.SessionManager") as mock_sm,
    ):
        mock_wb.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_wb.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_cb.return_value = MagicMock()

        mock_session_manager = AsyncMock()
        mock_session_manager.shutdown = AsyncMock()
        mock_sm.create = AsyncMock(return_value=mock_session_manager)

        with TestClient(app) as c:
            yield c
