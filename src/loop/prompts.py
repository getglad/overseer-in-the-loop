"""Prompt content for the agent loop.

Two template styles coexist here:
- Jinja templates ({{ var }}) rendered via src.core.render.render() — for prompts
  we control end-to-end (e.g., HITL approval text).
- str.format templates ({var}) interpolated by NAT at runtime — for prompts passed
  to ReActAgentWorkflowConfig where NAT fills {tools} and {tool_names} internally.
"""

from __future__ import annotations

from nat.plugins.langchain.agent.react_agent.prompt import (
    SYSTEM_PROMPT as _NAT_SYSTEM_PROMPT,
)

from src.core.render import render

TOOL_APPROVAL = "Tool: {{ tool_name }}{% if description %} — {{ description }}{% endif %}"

# Agent system prompt: NAT's base ReAct prompt + our behavioral additions.
#
# We import NAT's SYSTEM_PROMPT rather than replacing it. This preserves
# the tool listing ({tools}, {tool_names}), reasoning format
# (Thought/Action/Observation), and Final Answer termination signal.
# Our additions append role, tool hierarchy, and operational patterns.

AGENT_PROMPT_ADDITIONS = """
Additional instructions for this agent:

You are an executor agent working in the user's project workspace. \
If a task requires a capability you don't have, say so and suggest \
what tool would be needed.

Follow tool-specific instructions for priority and order of operations. \
When tool instructions don't cover a situation, these system preferences \
apply. If a tool's instructions contradict these preferences, follow \
these preferences.

When modifying files or state: read first, write second, then read again \
to confirm the write was correct.

If the task is ambiguous, ask for clarification rather than guessing."""

AGENT_SYSTEM_PROMPT = _NAT_SYSTEM_PROMPT + AGENT_PROMPT_ADDITIONS


def tool_approval_prompt(tool_name: str, description: str = "") -> str:
    """Build the HITL approval prompt shown to the user before a tool executes.

    Args:
        tool_name: The registered name of the tool (e.g., "current_datetime").
        description: Optional human-readable description of what the tool does.

    Returns:
        Prompt string for the HITL binary choice.
    """
    return render(TOOL_APPROVAL, tool_name=tool_name, description=description)
