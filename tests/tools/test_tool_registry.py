"""Tests for the tool registry — FunctionGroup registration, scoping, classifier wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

from src.guardrails.classifier import ClassifyResult
from src.guardrails.middleware import ClassifierConfig, set_evil_toggle
from src.tools.tool_registry import GetgladToolsConfig

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

GROUP_NAME = "getglad_tools"

EXPECTED_TOOLS = {
    f"{GROUP_NAME}__read_file",
    f"{GROUP_NAME}__write_file",
    f"{GROUP_NAME}__list_directory",
    f"{GROUP_NAME}__glob_search",
    f"{GROUP_NAME}__grep_search",
    f"{GROUP_NAME}__edit_file",
}


async def _setup_builder(builder: WorkflowBuilder, tmp_path: Path) -> None:
    """Register classifier middleware + function group on builder. Shared by all tests."""
    await builder.add_middleware("classifier", ClassifierConfig())
    await builder.add_function_group(
        GROUP_NAME,
        GetgladToolsConfig(
            workspace_root=str(tmp_path),
            middleware=["classifier"],
        ),
    )


@pytest.fixture
def _auto_classify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass classifier so non-fast-path tools execute without HITL."""

    async def _always_allow(*_args: Any, **_kwargs: Any) -> ClassifyResult:
        return ClassifyResult(
            allowed=True, layer="guardrail-agent", reason="test fixture",
        )

    monkeypatch.setattr("src.guardrails.middleware.classify", _always_allow)


@pytest.fixture
def _evil_on() -> Generator[None]:
    """Enable evil toggle for one test; always reset on teardown."""
    set_evil_toggle(enabled=True)
    yield
    set_evil_toggle(enabled=False)


class TestToolRegistration:
    """Tests that the tool catalog is complete and correctly configured."""

    async def test_all_expected_tools_registered(self, tmp_path: Path) -> None:
        """Every tool in EXPECTED_TOOLS resolves through the FunctionGroup."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            group_result = builder._function_groups[GROUP_NAME]
            functions = await group_result.instance.get_all_functions()
            assert set(functions.keys()) >= EXPECTED_TOOLS

    async def test_tool_count(self, tmp_path: Path) -> None:
        """FunctionGroup advertises at least the catalog Blog expects."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            group_result = builder._function_groups[GROUP_NAME]
            functions = await group_result.instance.get_all_functions()
            assert len(functions) >= len(EXPECTED_TOOLS)

    async def test_tools_resolve_as_langchain(self, tmp_path: Path) -> None:
        """Tools resolve as LangChain `BaseTool` instances ready for the ReAct agent."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            resolved_names = {t.name for t in tools}
            assert resolved_names >= EXPECTED_TOOLS


class TestClassifierFires:
    """Verify the classifier middleware actually intercepts tool calls."""

    async def test_always_allow_tool_skips_classify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A read-only tool (ALWAYS_ALLOW) executes without ever calling classify()."""
        call_log: list[str] = []

        async def _track_classify(
            _rails: object, tool_name: str, _args: dict[str, Any], **_kwargs: Any,
        ) -> ClassifyResult:
            call_log.append(tool_name)
            return ClassifyResult(allowed=True, layer="guardrail-agent", reason="test")

        monkeypatch.setattr("src.guardrails.middleware.classify", _track_classify)

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            list_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__list_directory"
            )
            await list_tool.ainvoke({"dir_path": "."})

        assert call_log == [], (
            f"list_directory is in ALWAYS_ALLOW but classify() was called: {call_log}"
        )

    @pytest.mark.usefixtures("_evil_on")
    async def test_evil_toggle_bypasses_fast_path_for_always_allow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WHEN evil toggle is on THEN classify() is called with skip_rules=True.

        Without this, the most natural reader prompts (``list files``,
        ``read README``) either short-circuit on the middleware fast-path
        OR get short-circuited by classify()'s own rules check — the evil
        payload never reaches the guardrail agent. The demo's whole point
        is to make the LLM-tier block visible, so the toggle must bypass
        BOTH fast-paths and force LLM evaluation.
        """
        call_log: list[tuple[str, bool]] = []

        async def _track_classify(
            _rails: object,
            tool_name: str,
            _args: dict[str, Any],
            *,
            skip_rules: bool = False,
            **_kwargs: Any,
        ) -> ClassifyResult:
            call_log.append((tool_name, skip_rules))
            return ClassifyResult(
                allowed=False, layer="guardrail-agent",
                reason="test: evil toggle should reach the LLM tier",
            )

        async def _approve(_text: str) -> bool:
            return True  # accept the HITL override so the tool still runs

        monkeypatch.setattr("src.guardrails.middleware.classify", _track_classify)
        monkeypatch.setattr(
            "src.guardrails.middleware.prompt_binary_approval", _approve,
        )

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            list_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__list_directory"
            )
            await list_tool.ainvoke({"dir_path": "."})

        assert call_log == [("list_directory", True)], (
            f"Evil toggle on but classify() did not receive skip_rules=True "
            f"— fast-path leaked: {call_log}"
        )

    async def test_write_tool_consults_classifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A write tool — outside the fast-path — invokes classify() once."""
        call_log: list[str] = []

        async def _track_classify(
            _rails: object, tool_name: str, _args: dict[str, Any], **_kwargs: Any,
        ) -> ClassifyResult:
            call_log.append(tool_name)
            return ClassifyResult(allowed=True, layer="guardrail-agent", reason="test")

        monkeypatch.setattr("src.guardrails.middleware.classify", _track_classify)

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            write_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__write_file"
            )
            await write_tool.ainvoke({
                "file_path": "out.txt",
                "text": "hello",
            })

        assert call_log == ["write_file"], (
            f"Expected classifier consulted for write_file, got {call_log}"
        )

    async def test_write_tool_args_flow_through_to_classifier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multi-arg tool: write_file's args reach classify() via BaseModel.model_dump().

        End-to-end check that NAT's InputArgsSchema unwrapping works in
        the live middleware path. ``_extract_tool_args`` is also covered
        directly in test_middleware_args_extraction below for the
        single-arg case (which the fast-path would otherwise hide).
        """
        captured: list[dict[str, Any]] = []

        async def _capture_args(
            _rails: object, _name: str, args: dict[str, Any], **_kwargs: Any,
        ) -> ClassifyResult:
            captured.append(args)
            return ClassifyResult(allowed=True, layer="guardrail-agent", reason="test")

        monkeypatch.setattr("src.guardrails.middleware.classify", _capture_args)

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            write_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__write_file"
            )
            await write_tool.ainvoke({"file_path": "out.txt", "text": "hello"})

        assert len(captured) == 1
        assert set(captured[0].keys()) >= {"file_path", "text"}, captured[0]

    async def test_blocked_classifier_with_hitl_rejection_returns_rejection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When classifier blocks AND HITL rejects, the tool returns REJECTION_MESSAGE."""

        async def _always_block(
            _rails: object, _tool_name: str, _args: dict[str, Any], **_kwargs: Any,
        ) -> ClassifyResult:
            return ClassifyResult(
                allowed=False, layer="guardrail-agent",
                reason="too risky for test",
            )

        async def _always_reject(_prompt_text: str) -> bool:
            return False

        monkeypatch.setattr("src.guardrails.middleware.classify", _always_block)
        monkeypatch.setattr(
            "src.guardrails.middleware.prompt_binary_approval", _always_reject,
        )

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            write_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__write_file"
            )
            result = await write_tool.ainvoke({
                "file_path": "out.txt",
                "text": "hello",
            })

        assert "rejected" in str(result).lower(), (
            f"Expected REJECTION_MESSAGE on block+reject, got: {result}"
        )

    async def test_blocked_classifier_with_hitl_override_runs_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When classifier blocks but HITL overrides, the tool still executes."""

        async def _always_block(
            _rails: object, _tool_name: str, _args: dict[str, Any], **_kwargs: Any,
        ) -> ClassifyResult:
            return ClassifyResult(
                allowed=False, layer="guardrail-agent",
                reason="too risky for test",
            )

        async def _always_approve(_prompt_text: str) -> bool:
            return True

        monkeypatch.setattr("src.guardrails.middleware.classify", _always_block)
        monkeypatch.setattr(
            "src.guardrails.middleware.prompt_binary_approval", _always_approve,
        )

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            write_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__write_file"
            )
            result = await write_tool.ainvoke({
                "file_path": "out.txt",
                "text": "hello",
            })

        # LangChain's WriteFileTool returns "File written successfully..."
        # on success — the key invariant is no rejection message.
        assert "rejected" not in str(result).lower(), (
            f"Override should have allowed the write; got rejection: {result}"
        )


@pytest.mark.usefixtures("_auto_classify")
class TestEditToolWorkspaceScoping:
    """Tests that the edit tool rejects paths outside the workspace."""

    async def test_edit_rejects_path_outside_workspace(self, tmp_path: Path) -> None:
        """Edit tool refuses absolute paths that escape the configured workspace root."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            edit_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__edit_file"
            )
            result = await edit_tool.ainvoke({
                "file_path": "/etc/passwd",
                "old_string": "root",
                "new_string": "hacked",
            })
            assert "outside the workspace" in str(result)

    async def test_edit_works_within_workspace(self, tmp_path: Path) -> None:
        """Edit tool succeeds for paths inside the workspace root."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")

        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            edit_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__edit_file"
            )
            result = await edit_tool.ainvoke({
                "file_path": str(f),
                "old_string": "hello",
                "new_string": "goodbye",
            })
            assert "Successfully edited" in str(result)
            assert f.read_text() == "goodbye world"


@pytest.mark.usefixtures("_auto_classify")
class TestLangChainToolContainment:
    """Pin LangChain's path-containment behavior for read/write/list tools.

    The edit tool enforces its own boundary check (see
    `TestEditToolWorkspaceScoping`); for the other tools we trust
    LangChain's `root_dir`. These tests catch silent regressions if a
    future LangChain version changes the rejection semantics.
    """

    async def test_read_file_rejects_path_outside_workspace(
        self, tmp_path: Path,
    ) -> None:
        """ReadFileTool with root_dir set must refuse absolute paths outside it."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            read_tool = next(t for t in tools if t.name == f"{GROUP_NAME}__read_file")
            result = await read_tool.ainvoke({"file_path": "/etc/passwd"})
            # LangChain emits "Access denied" or similar; the key invariant
            # is that the file's contents are NOT in the result.
            assert "root:" not in str(result), (
                "ReadFileTool returned /etc/passwd contents — workspace boundary broken"
            )

    async def test_write_file_rejects_absolute_path_outside_workspace(
        self, tmp_path: Path,
    ) -> None:
        """WriteFileTool with root_dir set must refuse absolute paths outside it."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            write_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__write_file"
            )
            result = await write_tool.ainvoke({
                "file_path": "/etc/passwd",
                "text": "should not land",
            })
            # LangChain returns an error string; the invariant is that
            # the result does NOT report success.
            assert "successfully" not in str(result).lower(), (
                f"WriteFileTool accepted /etc/passwd write: {result}"
            )

    async def test_list_directory_rejects_path_outside_workspace(
        self, tmp_path: Path,
    ) -> None:
        """ListDirectoryTool with root_dir set must refuse listings outside it."""
        async with WorkflowBuilder() as builder:
            await _setup_builder(builder, tmp_path)
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            list_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__list_directory"
            )
            result = await list_tool.ainvoke({"dir_path": "/etc"})
            # /etc on macOS/Linux contains "passwd"; if it leaks, containment failed.
            assert "passwd" not in str(result).lower(), (
                "ListDirectoryTool exposed /etc contents — boundary broken"
            )
