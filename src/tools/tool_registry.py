"""Tool registry — registers system access tools as a NAT FunctionGroup.

All tools operate on the server's file system within a workspace root.
Tools are registered via @register_function_group so NAT's builder resolves
them by group name ("getglad_tools").

HITL approval is applied as FunctionGroup middleware — a single middleware
instance gates all tools in the group, eliminating per-tool approval boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_community.tools.file_management import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group, register_middleware
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from pydantic import Field

from src.loop.hitl import REJECTION_MESSAGE, prompt_binary_approval
from src.loop.prompts import tool_approval_prompt
from src.tools.edit import EditError, edit_file

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nat.builder.builder import Builder
    from nat.middleware.middleware import CallNext, FunctionMiddlewareContext

logger = structlog.get_logger()

# Truthy strings the LLM might emit for a boolean-ish flag. Strict — anything
# outside this set is False — but wide enough that "1"/"yes" don't silently
# turn an intended append into a destructive overwrite.
_TRUTHY = frozenset({"true", "1", "yes", "y", "on"})


def _is_truthy(value: str) -> bool:
    """Coerce an LLM-supplied string flag to bool without silent data loss."""
    return value.strip().lower() in _TRUTHY


# FunctionMiddlewareBaseConfig uses metaclass __init_subclass__ kwargs
class HITLApprovalConfig(FunctionMiddlewareBaseConfig, name="hitl_approval"):  # type: ignore[call-arg,misc]
    """Configuration for the HITL approval middleware."""

    enabled: bool = True


class HITLApprovalMiddleware(FunctionMiddleware):  # type: ignore[misc]
    """FunctionGroup middleware that gates every tool call behind HITL approval.

    Overrides function_middleware_invoke to short-circuit on rejection —
    the tool function never executes if the user says no. NAT's framework
    skips this middleware entirely when ``enabled`` is False.
    """

    def __init__(self, config: HITLApprovalConfig) -> None:
        """Store the config so ``enabled`` and ``invoke`` can read it."""
        super().__init__()
        self._config = config

    @property
    def enabled(self) -> bool:
        """Whether the middleware should run; framework checks before invoke."""
        return self._config.enabled

    async def function_middleware_invoke(
        self,
        *args: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext,
        **kwargs: Any,
    ) -> Any:
        """Prompt for approval, short-circuit if rejected."""
        _, fn_name = FunctionGroup.decompose(context.name)
        # Prefer the description registered with `group.add_function`; fall
        # back to the bare tool name. We intentionally don't reflect LLM-
        # supplied args into the prompt — those values can contain
        # whitespace or content-spoofing characters that would mislead the
        # user about what they're approving.
        desc = context.description or fn_name

        if not await self._prompt_approval(fn_name, desc):
            return REJECTION_MESSAGE

        return await call_next(*args, **kwargs)

    @staticmethod
    async def _prompt_approval(tool_name: str, description: str) -> bool:
        """Prompt the user via shared HITL primitive."""
        return await prompt_binary_approval(
            tool_approval_prompt(tool_name, description),
        )


@register_middleware(config_type=HITLApprovalConfig)  # type: ignore[untyped-decorator]
async def hitl_approval_middleware(
    config: HITLApprovalConfig,
    _builder: Builder,
) -> AsyncIterator[HITLApprovalMiddleware]:
    """Build the HITL approval middleware instance."""
    if not config.enabled:
        logger.warning(
            "hitl_approval_disabled",
            reason="HITL approval middleware disabled — tools run without prompting",
        )
    yield HITLApprovalMiddleware(config)


# FunctionGroupBaseConfig uses metaclass __init_subclass__ kwargs
class GetgladToolsConfig(FunctionGroupBaseConfig, name="getglad_tools"):  # type: ignore[call-arg,misc]
    """Configuration for the project's system access tool group."""

    workspace_root: str = Field(description="Root directory for file operations")


def _add_file_tools(group: FunctionGroup, root: str) -> None:
    """Add file management tools (read, write, list) to the group."""
    read_lc = ReadFileTool(root_dir=root)
    write_lc = WriteFileTool(root_dir=root)
    list_lc = ListDirectoryTool(root_dir=root)

    async def read_file(file_path: str) -> str:
        """Read file from disk."""
        # ainvoke runs the sync tool off the event loop — these are blocking I/O.
        result: str = await read_lc.ainvoke({"file_path": file_path})
        return result

    async def write_file(file_path: str, text: str, append: str = "false") -> str:
        """Write file to disk. Set append to 'true' to append."""
        result: str = await write_lc.ainvoke(
            {"file_path": file_path, "text": text, "append": _is_truthy(append)},
        )
        return result

    async def list_directory(dir_path: str = ".") -> str:
        """List files and directories."""
        result: str = await list_lc.ainvoke({"dir_path": dir_path})
        return result

    group.add_function(
        "read_file", read_file,
        description="Read a file's contents. Use before editing to understand the current state.",
    )
    group.add_function(
        "write_file", write_file,
        description=(
            "Write content to a file, replacing its contents entirely. "
            "Read first to avoid losing data."
        ),
    )
    group.add_function(
        "list_directory", list_directory,
        description="List files and directories at a path. Use to explore project structure.",
    )


def _add_search_tools(group: FunctionGroup, root: str) -> None:
    """Add search tools (glob, grep) to the group."""
    search = FilesystemFileSearchMiddleware(root_path=root)
    glob_lc = next(t for t in search.tools if t.name == "glob_search")
    grep_lc = next(t for t in search.tools if t.name == "grep_search")

    async def glob_search(pattern: str, path: str = "/") -> str:
        """Fast file pattern matching."""
        # ainvoke: a glob/grep can walk the whole tree — never block the loop.
        result: str = await glob_lc.ainvoke({"pattern": pattern, "path": path})
        return result

    async def grep_search(pattern: str, path: str = "/", include: str = "") -> str:
        """Fast content search across the codebase."""
        result: str = await grep_lc.ainvoke(
            {"pattern": pattern, "path": path, "include": include or None},
        )
        return result

    group.add_function(
        "glob_search", glob_search,
        description=(
            "Find files by name pattern (e.g. '**/*.py'). "
            "Use to locate files before reading."
        ),
    )
    group.add_function(
        "grep_search", grep_search,
        description=(
            "Search file contents for a pattern. "
            "Use to find specific code, functions, or text."
        ),
    )


def _add_edit_tool(group: FunctionGroup, workspace: Path) -> None:
    """Add the edit tool (9-strategy targeted replacement) to the group."""
    # Resolve once at registration — the workspace doesn't change for the
    # group's lifetime, and resolve() can hit the filesystem.
    workspace_resolved = workspace.resolve()

    async def edit_file_tool(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: str = "false",
    ) -> str:
        """Edit a file using targeted string replacement."""
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = workspace_resolved / candidate

        target = candidate.resolve()
        if not target.is_relative_to(workspace_resolved):
            return f"Error: path {file_path} is outside the workspace"

        # Refuse to edit through a symlink even if its target is inside the
        # workspace — narrows the TOCTOU race where the link is swapped
        # between check and write to point outside.
        if candidate.is_symlink():
            return f"Error: path {file_path} is a symlink; edits must target real files"

        try:
            result = edit_file(
                target,
                old_string,
                new_string,
                replace_all=_is_truthy(replace_all),
            )
        except EditError as e:
            return f"Error: {e}"
        else:
            return f"Successfully edited {file_path} (strategy: {result.strategy})"

    group.add_function(
        "edit_file",
        edit_file_tool,
        description=(
            "Edit a file using targeted string replacement. "
            "Tolerant of minor whitespace differences. "
            "Prefer over write_file for targeted changes — preserves surrounding content."
        ),
    )


@register_function_group(config_type=GetgladToolsConfig)  # type: ignore[untyped-decorator]
async def getglad_tools(
    config: GetgladToolsConfig,
    _builder: Builder,
) -> AsyncIterator[FunctionGroup]:
    """Build the system access tool group.

    Creates file management, search, and edit tools scoped to the
    configured workspace root. HITL approval is applied as group-level
    middleware — all tools share the same approval gate.
    """
    # Middleware is resolved by the builder from config.middleware (name-based)
    group = FunctionGroup(config=config)

    _add_file_tools(group, config.workspace_root)
    _add_search_tools(group, config.workspace_root)
    _add_edit_tool(group, Path(config.workspace_root))

    logger.info(
        "getglad_tools_registered",
        tool_count=len(await group.get_all_functions()),
        workspace_root=config.workspace_root,
    )

    yield group
