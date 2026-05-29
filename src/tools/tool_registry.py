"""Tool registry — registers system access tools as a NAT FunctionGroup.

All tools operate on the server's file system within a workspace root.
Tools are registered via @register_function_group so NAT's builder resolves
them by group name ("getglad_tools").

The classifier middleware (see src/guardrails/middleware.py) wraps every
tool in the group: rules-fast-path allow for read-only tools, LLM judgment
for ambiguous cases, HITL fallback when the classifier escalates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_community.tools.file_management import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig
from pydantic import Field

from src.tools.edit import EditError, edit_file

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nat.builder.builder import Builder

logger = structlog.get_logger()

# Truthy strings the LLM might emit for a boolean-ish flag. Strict — anything
# outside this set is False — but wide enough that "1"/"yes" don't silently
# turn an intended append into a destructive overwrite.
_TRUTHY = frozenset({"true", "1", "yes", "y", "on"})


def _is_truthy(value: str) -> bool:
    """Coerce an LLM-supplied string flag to bool without silent data loss."""
    return value.strip().lower() in _TRUTHY


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
    configured workspace root. Approval is handled by the classifier
    middleware registered separately and referenced by name via
    `config.middleware`.
    """
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
