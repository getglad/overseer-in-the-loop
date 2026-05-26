"""Tests for the tool registry — FunctionGroup registration, scoping, HITL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

from src.tools.tool_registry import (
    GetgladToolsConfig,
    HITLApprovalConfig,
    HITLApprovalMiddleware,
)

if TYPE_CHECKING:
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
    """Register middleware + function group on builder. Shared by all tests."""
    await builder.add_middleware("hitl_approval", HITLApprovalConfig())
    await builder.add_function_group(
        GROUP_NAME,
        GetgladToolsConfig(
            workspace_root=str(tmp_path),
            middleware=["hitl_approval"],
        ),
    )


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


@pytest.fixture
def _auto_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass HITL approval middleware so tools execute in tests."""

    async def _always_approve(_tool_name: str, _description: str) -> bool:
        return True

    monkeypatch.setattr(
        HITLApprovalMiddleware,
        "_prompt_approval",
        staticmethod(_always_approve),
    )


class TestMiddlewareFires:
    """Verify that HITL middleware actually intercepts tool calls."""

    async def test_middleware_called_on_tool_invoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WHEN a tool is invoked THEN the HITL middleware fires."""
        call_log: list[str] = []

        async def _tracking_approve(_tool_name: str, _description: str) -> bool:
            call_log.append(_tool_name)
            return True

        monkeypatch.setattr(
            HITLApprovalMiddleware,
            "_prompt_approval",
            staticmethod(_tracking_approve),
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

        assert "list_directory" in call_log, (
            "HITL middleware did not fire — approval was not called"
        )

    async def test_middleware_blocks_on_rejection(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WHEN the user rejects THEN the tool returns rejection message."""

        async def _always_reject(_tool_name: str, _description: str) -> bool:
            return False

        monkeypatch.setattr(
            HITLApprovalMiddleware,
            "_prompt_approval",
            staticmethod(_always_reject),
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
            result = await list_tool.ainvoke({"dir_path": "."})

        assert "rejected" in str(result).lower()

    async def test_disabled_middleware_skips_approval(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """WHEN HITLApprovalConfig.enabled is False THEN approval is not requested.

        NAT's framework checks ``Middleware.enabled`` before invoking it. We
        expose this through ``HITLApprovalConfig.enabled`` so blog readers
        can run unattended demos by flipping a config flag.
        """
        call_log: list[str] = []

        async def _track(tool_name: str, _description: str) -> bool:
            call_log.append(tool_name)
            return True

        monkeypatch.setattr(
            HITLApprovalMiddleware,
            "_prompt_approval",
            staticmethod(_track),
        )

        async with WorkflowBuilder() as builder:
            await builder.add_middleware(
                "hitl_approval", HITLApprovalConfig(enabled=False),
            )
            await builder.add_function_group(
                GROUP_NAME,
                GetgladToolsConfig(
                    workspace_root=str(tmp_path),
                    middleware=["hitl_approval"],
                ),
            )
            tools = await builder.get_tools(
                [GROUP_NAME],
                wrapper_type=LLMFrameworkEnum.LANGCHAIN,
            )
            list_tool = next(
                t for t in tools if t.name == f"{GROUP_NAME}__list_directory"
            )
            await list_tool.ainvoke({"dir_path": "."})

        assert call_log == [], (
            f"Expected no approval prompts when enabled=False, got {call_log}"
        )


@pytest.mark.usefixtures("_auto_approve")
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


@pytest.mark.usefixtures("_auto_approve")
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
