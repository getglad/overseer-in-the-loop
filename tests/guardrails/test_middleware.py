"""Unit tests for ``src.guardrails.middleware``.

Covers ``_extract_tool_args`` (both NAT arg shapes) and the
``function_middleware_invoke`` fast-path: the security-critical contract
that a deterministic ALLOW runs without the LLM, while a credential read
or a guardrails self-write skips the fast-path and escalates to HITL.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.core.conversation import reset_recent_user_prompts, set_recent_user_prompts
from src.guardrails import middleware as mw
from src.guardrails.middleware import ClassifierMiddleware, _extract_tool_args

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import pytest


@dataclass(frozen=True)
class _FakeContext:
    """Stand-in for NAT's FunctionMiddlewareContext.

    The real one is frozen with many fields; we only need
    ``input_schema`` for the helper under test.
    """

    input_schema: type[BaseModel] | None


class _ListDirSchema(BaseModel):
    """Mimics NAT's InputArgsSchema for a single-arg function."""

    dir_path: str


class _WriteFileSchema(BaseModel):
    """Mimics NAT's InputArgsSchema for a multi-arg function."""

    file_path: str
    text: str
    append: str = "false"


class TestExtractToolArgs:
    """Verify both NAT arg shapes produce usable ``tool_args`` dicts.

    Without the schema-aware path, single-arg tools (``list_directory``,
    ``read_file``) reach ``classify()`` with ``tool_args={}`` because
    NAT doesn't wrap them in a BaseModel — leaving the HITL prompt with
    no args to show the human.
    """

    def test_basemodel_first_arg_is_model_dumped(self) -> None:
        """The multi-arg path: NAT passes a BaseModel, we use ``model_dump``."""
        ctx = _FakeContext(input_schema=_WriteFileSchema)
        model = _WriteFileSchema(file_path="x.txt", text="hello")
        result = _extract_tool_args((model,), ctx)  # type: ignore[arg-type]
        assert result == {"file_path": "x.txt", "text": "hello", "append": "false"}

    def test_positional_scalar_recovered_via_schema(self) -> None:
        """The single-arg path: NAT passes a raw value, schema gives the field name."""
        ctx = _FakeContext(input_schema=_ListDirSchema)
        result = _extract_tool_args((".",), ctx)  # type: ignore[arg-type]
        assert result == {"dir_path": "."}

    def test_empty_args_returns_empty_dict(self) -> None:
        """No args, no work — return an empty dict rather than raising."""
        ctx = _FakeContext(input_schema=_ListDirSchema)
        result = _extract_tool_args((), ctx)  # type: ignore[arg-type]
        assert result == {}

    def test_no_schema_falls_back_to_empty(self) -> None:
        """If NAT supplies neither a BaseModel nor a schema, degrade safely."""
        ctx = _FakeContext(input_schema=None)
        result = _extract_tool_args(("anything",), ctx)  # type: ignore[arg-type]
        assert result == {}

    def test_positional_multiple_zipped_with_schema(self) -> None:
        """A multi-arg function passed positionally — fields zip in order."""
        ctx = _FakeContext(input_schema=_WriteFileSchema)
        result = _extract_tool_args(("x.txt", "hello"), ctx)  # type: ignore[arg-type]
        assert result == {"file_path": "x.txt", "text": "hello"}


@dataclass(frozen=True)
class _InvokeContext:
    """Minimal stand-in for the FunctionMiddlewareContext the invoke path reads.

    ``name`` is the FunctionGroup-qualified tool name (``group__tool``);
    ``input_schema`` recovers field names for single-arg positional calls.
    """

    name: str
    input_schema: type[BaseModel] | None


class _ReadSchema(BaseModel):
    """InputArgsSchema for the single-arg ``read_file``."""

    file_path: str


class _GrepSchema(BaseModel):
    """InputArgsSchema for the multi-arg ``grep_search`` (credential in ``include``)."""

    pattern: str
    path: str = "/"
    include: str = ""


class _StubRails:
    """Stand-in for LLMRails: records whether the LLM tier was consulted.

    ``check_async`` returns a ``blocked`` status so an escalated call lands
    in the HITL branch; ``called`` proves whether the fast-path was skipped,
    and ``last_messages`` captures what the classifier sent for inspection.
    """

    config = SimpleNamespace(models=[SimpleNamespace(model="stub")])

    def __init__(self) -> None:
        self.called = False
        self.last_messages: list[dict[str, str]] | None = None

    async def check_async(self, **kwargs: Any) -> SimpleNamespace:
        self.called = True
        self.last_messages = kwargs.get("messages")
        return SimpleNamespace(status=SimpleNamespace(value="blocked"))


def _recording_call_next() -> tuple[Callable[..., Awaitable[str]], dict[str, bool]]:
    """A ``call_next`` that records whether the wrapped tool actually ran."""
    ran: dict[str, bool] = {}

    async def call_next(*_args: Any, **_kwargs: Any) -> str:
        ran["ran"] = True
        return "TOOL RAN"

    return call_next, ran


class TestFastPathInvoke:
    """The fast-path contract: ALLOW runs bare, BLOCK/credential escalates to HITL.

    This is the regression guard for the cycle-2 finding that the fast-path
    was untested — if someone reverts it to a name-only allowlist or inverts
    the ``Decision`` comparison, a credential read or self-write would be
    silently auto-approved, and one of these tests would fail.
    """

    async def test_safe_read_runs_without_llm(self) -> None:
        """A non-credential read takes the rules fast-path — the LLM is never consulted."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, ran = _recording_call_next()
        ctx = _InvokeContext(name="g__read_file", input_schema=_ReadSchema)

        result = await middleware.function_middleware_invoke(
            "README.md", call_next=call_next, context=ctx,  # type: ignore[arg-type]
        )

        assert result == "TOOL RAN"
        assert ran.get("ran") is True
        assert rails.called is False

    async def test_credential_read_skips_fast_path_and_escalates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A credential read is NOT auto-allowed: it reaches the LLM tier, then HITL."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, ran = _recording_call_next()

        async def reject(_text: str) -> bool:
            return False

        monkeypatch.setattr(mw, "prompt_binary_approval", reject)
        ctx = _InvokeContext(name="g__read_file", input_schema=_ReadSchema)

        result = await middleware.function_middleware_invoke(
            ".env", call_next=call_next, context=ctx,  # type: ignore[arg-type]
        )

        assert result == mw.REJECTION_MESSAGE
        assert ran.get("ran") is None  # tool did not auto-run
        assert rails.called is True  # proves the fast-path was skipped

    async def test_list_directory_of_credential_dir_escalates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_directory of .ssh/ is recon — it skips the fast-path end-to-end."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, ran = _recording_call_next()

        async def reject(_text: str) -> bool:
            return False

        monkeypatch.setattr(mw, "prompt_binary_approval", reject)
        ctx = _InvokeContext(name="g__list_directory", input_schema=_ListDirSchema)

        result = await middleware.function_middleware_invoke(
            ".ssh/", call_next=call_next, context=ctx,  # type: ignore[arg-type]
        )

        assert result == mw.REJECTION_MESSAGE
        assert ran.get("ran") is None
        assert rails.called is True

    async def test_grep_credential_in_nonfirst_field_escalates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A credential target in grep's ``include`` (not the first arg) still escalates."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, ran = _recording_call_next()

        async def reject(_text: str) -> bool:
            return False

        monkeypatch.setattr(mw, "prompt_binary_approval", reject)
        model = _GrepSchema(pattern="BEGIN", include="*.pem")
        ctx = _InvokeContext(name="g__grep_search", input_schema=_GrepSchema)

        result = await middleware.function_middleware_invoke(
            model, call_next=call_next, context=ctx,  # type: ignore[arg-type]
        )

        assert result == mw.REJECTION_MESSAGE
        assert ran.get("ran") is None
        assert rails.called is True

    async def test_self_write_blocks_at_rules_without_llm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A write to src/guardrails/ blocks deterministically — escalates, no LLM call."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, ran = _recording_call_next()

        async def reject(_text: str) -> bool:
            return False

        monkeypatch.setattr(mw, "prompt_binary_approval", reject)
        model = _WriteFileSchema(file_path="src/guardrails/rules.py", text="x = 1")
        ctx = _InvokeContext(name="g__write_file", input_schema=_WriteFileSchema)

        result = await middleware.function_middleware_invoke(
            model, call_next=call_next, context=ctx,  # type: ignore[arg-type]
        )

        assert result == mw.REJECTION_MESSAGE
        assert ran.get("ran") is None
        assert rails.called is False  # rules-tier BLOCK short-circuits before the LLM

    async def test_bound_conversation_window_reaches_the_classifier(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The prior-prompt window bound for the run appears in the classifier's user message."""
        rails = _StubRails()
        middleware = ClassifierMiddleware(rails)  # type: ignore[arg-type]
        call_next, _ran = _recording_call_next()

        async def reject(_text: str) -> bool:
            return False

        monkeypatch.setattr(mw, "prompt_binary_approval", reject)
        # A credential read escalates past the fast-path into the LLM tier.
        ctx = _InvokeContext(name="g__read_file", input_schema=_ReadSchema)
        token = set_recent_user_prompts(["read the database config"])
        try:
            await middleware.function_middleware_invoke(
                ".env", call_next=call_next, context=ctx,  # type: ignore[arg-type]
            )
        finally:
            reset_recent_user_prompts(token)

        assert rails.called is True
        assert rails.last_messages is not None
        user_msg = next(m["content"] for m in rails.last_messages if m["role"] == "user")
        assert "read the database config" in user_msg
