"""HITL (Human-in-the-Loop) approval primitives.

Shared across all domains — any tool or middleware that needs human
approval imports from here.
"""

from nat.builder.context import Context
from nat.data_models.interactive import BinaryHumanPromptOption, HumanPromptBinary

APPROVE_OPTION = BinaryHumanPromptOption(id="yes", label="Approve", value="yes")
REJECT_OPTION = BinaryHumanPromptOption(id="no", label="Reject", value="no")

APPROVAL_TIMEOUT = 60

REJECTION_MESSAGE = "Tool call was rejected by the user."


async def prompt_binary_approval(prompt_text: str) -> bool:
    """Prompt the user for binary approval via NAT's Context.

    Routes to the appropriate transport (console, WebSocket, HTTP)
    depending on how the session's user_input_callback is configured.

    Args:
        prompt_text: The text to display to the user.

    Returns:
        True if the user approved, False if rejected.
    """
    ctx = Context.get()
    response = await ctx.user_interaction_manager.prompt_user_input(
        HumanPromptBinary(
            text=prompt_text,
            options=[APPROVE_OPTION, REJECT_OPTION],
            timeout=APPROVAL_TIMEOUT,
        ),
    )
    selected: str = response.content.selected_option.value
    expected: str = APPROVE_OPTION.value
    return selected == expected
