"""FastAPI application — NAT-as-a-library WebSocket gateway.

The agent loop is wired in via ``configure_builder`` during lifespan startup;
the workflow lives on ``app.state.session_manager`` and is shared across
WebSocket connections.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from src.server.router import router as server_router
from src.server.router import warn_if_ws_token_malformed

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from src.loop.agent import AgentSettings

logger = structlog.get_logger()


async def _warm_llm(settings: AgentSettings) -> None:
    """Fire a 1-token request to NIM so the first real query isn't a cold start.

    Opportunistic — runs as a background task during lifespan startup.
    Never blocks startup and never raises: if the warm-up fails (network
    glitch, bad key, model unavailable), the first real query will surface
    the same error via the gateway's regular path.
    """
    try:
        # Deferred so the uvicorn watcher process doesn't pull NAT/openai.
        from openai import AsyncOpenAI

        # `async with` so the underlying httpx connection pool is closed —
        # a bare client would leak its pool for the server's lifetime.
        async with AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url) as client:
            await client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        logger.info("llm_warmup_complete", model=settings.model_name)
    except Exception as exc:  # noqa: BLE001 — warmup is best-effort
        logger.warning(
            "llm_warmup_failed",
            model=settings.model_name,
            error=str(exc),
        )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    """Build the NAT workflow on startup, tear down on shutdown."""
    # NAT imports deferred so `uvicorn --reload`'s watcher doesn't pay them.
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.session import SessionManager

    from src.loop.agent import AgentSettings, configure_builder

    settings = AgentSettings()
    warn_if_ws_token_malformed()

    async with WorkflowBuilder() as builder:
        config = await configure_builder(builder, settings)

        application.state.session_manager = await SessionManager.create(
            config=config,
            shared_builder=builder,
        )
        logger.info("server_ready", model=settings.model_name)

        # Fire-and-forget warm-up: doesn't block startup, doesn't bubble errors.
        # The first real user query then skips the NIM cold-start latency.
        warmup_task = asyncio.create_task(_warm_llm(settings))

        try:
            yield
        finally:
            # `_warm_llm`'s `except Exception` doesn't catch `CancelledError`
            # (a `BaseException`), so a cancellation mid-warmup would surface
            # as "Task exception was never retrieved". Awaiting collects it.
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)

        if application.state.session_manager is not None:
            await application.state.session_manager.shutdown()
        logger.info("server_shutdown")


app = FastAPI(
    title="Overseer-in-the-loop",
    description="Code-first NAT agent loop with HITL approval, exposed over WebSocket.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(server_router)
