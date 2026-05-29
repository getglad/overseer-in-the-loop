"""Fixtures for src/tools/ tests."""

from __future__ import annotations

import pytest
from nat.runtime.loader import PluginTypes, discover_and_register_plugins


@pytest.fixture(autouse=True, scope="module")
def _discover_nat_plugins() -> None:
    """Register NAT config-object plugins once per module.

    `discover_and_register_plugins` is idempotent process-wide, but pulling
    it into an autouse fixture removes the boilerplate from every test.
    """
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)


@pytest.fixture(autouse=True)
def _stub_llm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set ``LLM_API_KEY`` so the classifier middleware's build_rails succeeds.

    ``classifier_middleware_builder`` reads ``LLM_API_KEY`` and passes it to
    LangChain's OpenAI client, which validates the key at construction time
    (rejects empty string). Tests never make real API calls — ``classify``
    is monkeypatched — but the rails construction has to succeed for the
    FunctionGroup builder to register the middleware.
    """
    monkeypatch.setenv("LLM_API_KEY", "test-key-for-classifier-init")
