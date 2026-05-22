"""Fixtures for src/loop/ tests — NAT Context mocks and HITL helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from nat.data_models.interactive import BinaryHumanPromptOption

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def mock_context() -> MagicMock:
    """A mock NAT Context whose user_interaction_manager.prompt_user_input is controllable."""
    ctx = MagicMock()
    ctx.user_interaction_manager = MagicMock()
    ctx.user_interaction_manager.prompt_user_input = AsyncMock()
    return ctx


@pytest.fixture
def mock_builder() -> AsyncMock:
    """A mock WorkflowBuilder with async methods stubbed."""
    builder = AsyncMock()
    builder.add_telemetry_exporter = AsyncMock()
    return builder


@pytest.fixture
def make_binary_response() -> Callable[[str], MagicMock]:
    """Factory that builds a mock InteractionResponse for a given selected option value."""

    def _make(selected_value: str) -> MagicMock:
        response = MagicMock()
        response.content.selected_option = BinaryHumanPromptOption(
            id=selected_value,
            label=selected_value.title(),
            value=selected_value,
        )
        return response

    return _make
