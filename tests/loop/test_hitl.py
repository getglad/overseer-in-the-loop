"""Tests for HITL approval flow in the agent loop.

Verifies that:
- prompt_binary_approval correctly prompts with binary choice and returns the result
- The HITL-wrapped tool calls prompt_binary_approval before executing tool logic
- Rejected tool calls return a rejection message without executing
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nat.data_models.interactive import HumanPromptBinary

from src.loop.agent import HITLCurrentTimeConfig, hitl_current_datetime
from src.loop.hitl import prompt_binary_approval


class TestApproveAction:
    """Tests for prompt_binary_approval — the core HITL approval logic."""

    @pytest.mark.parametrize(
        ("selected", "expected"),
        [("yes", True), ("no", False)],
    )
    async def test_returns_user_decision(
        self, mock_context, make_binary_response, selected, expected,
    ):
        """prompt_binary_approval returns True for 'yes' and False for 'no'."""
        mock_context.user_interaction_manager.prompt_user_input.return_value = (
            make_binary_response(selected)
        )

        with patch("src.loop.hitl.Context.get", return_value=mock_context):
            result = await prompt_binary_approval("Approve this?")

        assert result is expected

    async def test_sends_binary_prompt_with_approve_reject_options(
        self, mock_context, make_binary_response,
    ):
        """WHEN approval is requested THEN prompt has two options: Approve and Reject."""
        mock_context.user_interaction_manager.prompt_user_input.return_value = (
            make_binary_response("yes")
        )

        with patch("src.loop.hitl.Context.get", return_value=mock_context):
            await prompt_binary_approval("Do you approve this action?")

        call_args = mock_context.user_interaction_manager.prompt_user_input.call_args
        prompt = call_args[0][0]

        assert isinstance(prompt, HumanPromptBinary)
        assert prompt.text == "Do you approve this action?"
        assert len(prompt.options) == 2

        option_values = {opt.value for opt in prompt.options}
        assert option_values == {"yes", "no"}

        option_labels = {opt.label for opt in prompt.options}
        assert "Approve" in option_labels
        assert "Reject" in option_labels


@pytest.fixture
async def tool_fn():
    """The callable inside the HITL-wrapped current_datetime tool."""
    config = HITLCurrentTimeConfig()
    stub_builder = MagicMock()
    async with hitl_current_datetime(config, stub_builder) as fn_info:
        yield fn_info.single_fn


class TestHITLWrappedTool:
    """Tests that the HITL-wrapped tool calls prompt_binary_approval before executing.

    These are the integration tests that would have caught the original bug:
    prompt_binary_approval existed but was never called before tool execution.
    """

    async def test_approved_tool_calls_prompt_binary_approval_then_executes(self, tool_fn):
        """WHEN user approves THEN prompt_binary_approval is called AND tool returns the time."""
        with patch(
            "src.loop.agent.prompt_binary_approval",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_approve:
            result = await tool_fn("")

        mock_approve.assert_called_once()
        assert "current time" in result.lower()

    async def test_rejected_tool_returns_rejection_without_executing(self, tool_fn):
        """WHEN user rejects THEN tool returns rejection message, not the time."""
        with patch(
            "src.loop.agent.prompt_binary_approval",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_approve:
            result = await tool_fn("")

        mock_approve.assert_called_once()
        assert "rejected" in result.lower()
        assert "current time" not in result.lower()

    async def test_approval_prompt_includes_tool_name(self, tool_fn):
        """WHEN approval is requested THEN the prompt text identifies the tool."""
        with patch(
            "src.loop.agent.prompt_binary_approval",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_approve:
            await tool_fn("")

        prompt_text = mock_approve.call_args[0][0]
        assert "current_datetime" in prompt_text

    @pytest.mark.parametrize(
        ("approved", "expected_sequence"),
        [
            (True, ["approve", "datetime_now"]),
            (False, ["approve"]),
        ],
        ids=["approved-executes", "rejected-skips"],
    )
    async def test_trajectory_execution_order(self, tool_fn, approved, expected_sequence):
        """Trajectory: call-sequence spy verifies prompt_binary_approval fires first.

        On approval, datetime logic runs after. On rejection, it never runs.
        """
        call_sequence: list[str] = []

        async def spy_approve(_prompt: str) -> bool:
            call_sequence.append("approve")
            return approved

        original_now = datetime.now

        def spy_now(*args, **kwargs):
            call_sequence.append("datetime_now")
            return original_now(*args, **kwargs)

        with (
            patch("src.loop.agent.prompt_binary_approval", side_effect=spy_approve),
            patch("src.loop.agent.datetime") as mock_dt,
        ):
            mock_dt.now = spy_now
            mock_dt.UTC = UTC
            await tool_fn("")

        assert call_sequence == expected_sequence
