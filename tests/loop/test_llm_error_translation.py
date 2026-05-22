"""Tests for ``_explain_llm_failure`` — graceful LLM error translation.

``run_agent`` catches every exception via a broad ``except Exception``
because NAT/LangGraph wraps LLM errors several layers deep. The
``_explain_llm_failure`` helper inspects the exception chain and
returns an actionable client message when it recognizes a known
LLM-side failure (so the UI shows "model X is gone" instead of
"check server logs").

These tests pin three things:
1. Status errors (e.g., 410 model EOL) get translated with the code + detail.
2. Connection errors get translated separately.
3. Unrelated exceptions return None so the caller's generic fallback fires.
"""

from __future__ import annotations

import httpx
from openai import APIConnectionError, APIStatusError

from src.loop.service import _explain_llm_failure


def _fake_status_error(status: int, body_detail: str) -> APIStatusError:
    """Construct an APIStatusError suitable for asserting against."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status_code=status, request=request)
    return APIStatusError(message=body_detail, response=response, body={"detail": body_detail})


class TestExplainLLMFailure:
    """Translation of openai-shaped errors into user-facing strings."""

    def test_status_error_includes_code_and_detail(self) -> None:
        """A 410 EOL surfaces the status code and the trimmed message text."""
        err = _fake_status_error(410, "The model 'foo' has reached its end of life")
        message = _explain_llm_failure(err)
        assert message is not None
        assert "410" in message
        assert "end of life" in message
        assert "LLM_MODEL" in message

    def test_connection_error_returns_connectivity_hint(self) -> None:
        """An APIConnectionError points the user at LLM_BASE_URL and the network."""
        request = httpx.Request("POST", "https://unreachable.invalid/v1")
        err = APIConnectionError(request=request)
        message = _explain_llm_failure(err)
        assert message is not None
        assert "Could not reach" in message
        assert "LLM_BASE_URL" in message

    def test_unrelated_exception_returns_none(self) -> None:
        """Caller's generic fallback fires when we can't translate."""
        assert _explain_llm_failure(ValueError("not an LLM thing")) is None

    def test_walks_exception_chain_via_cause(self) -> None:
        """Wrapped errors (NAT/LangGraph) still get translated via __cause__."""
        inner = _fake_status_error(401, "Invalid API key")
        outer = RuntimeError("agent step failed")
        outer.__cause__ = inner
        message = _explain_llm_failure(outer)
        assert message is not None
        assert "401" in message
        assert "Invalid API key" in message

    def test_walks_exception_chain_via_context(self) -> None:
        """Implicit `raise ... from None` chains via __context__ also get translated."""
        inner = _fake_status_error(429, "Too many requests")
        outer = RuntimeError("aggregator wrapper")
        outer.__context__ = inner
        message = _explain_llm_failure(outer)
        assert message is not None
        assert "429" in message

    def test_cycle_in_chain_does_not_loop_forever(self) -> None:
        """A pathological cycle is detected via the seen-set; we return rather than hang."""
        a: BaseException = ValueError("a")
        b: BaseException = ValueError("b")
        a.__cause__ = b
        b.__cause__ = a  # cycle
        # No assertion on result content — just that this returns at all.
        _explain_llm_failure(a)

    def test_redacts_api_key_from_upstream_message(self) -> None:
        """If a misconfigured upstream echoes a bearer token back in the body, redact it."""
        err = _fake_status_error(401, "Authentication failed: sk-abc1234567890XYZ rejected")
        message = _explain_llm_failure(err)
        assert message is not None
        assert "sk-abc1234567890XYZ" not in message
        assert "[REDACTED]" in message

    def test_redacts_bearer_header_from_upstream_message(self) -> None:
        """Bearer tokens leaked via upstream error bodies are redacted too."""
        err = _fake_status_error(403, "Bearer eyJhbGciOiJIUzI1NiJ9.invalidtoken denied")
        message = _explain_llm_failure(err)
        assert message is not None
        assert "Bearer eyJhbGciOiJIUzI1NiJ9.invalidtoken" not in message
        assert "[REDACTED]" in message
