"""ReAct agent workflow that emits LLM-tier IntermediateSteps natively.

NAT's stock ``react_agent`` register builds the agent graph WITHOUT a callback
handler, so the LLM tier emits no ``LLM_START`` / ``LLM_NEW_TOKEN`` / ``LLM_END``
events — only the FunctionGroup's tool events reach the stream. This register is
NAT's own (non-streaming) ReAct workflow with one change: it attaches NAT's
``LangchainProfilerHandler`` per run — the exact pattern
``nat.plugins.langchain.control_flow.sequential_executor`` uses — so the gateway
and UI see model latency + token-by-token streaming through the canonical event
stream, with no ``_stream_llm`` monkey-patch.

The handler is built INSIDE the per-run function (not at workflow-build time)
because ``LangchainProfilerHandler.__init__`` captures
``Context.get().intermediate_step_manager`` — which only exists once a run's
context is active. Only the non-streaming ``single_fn`` is provided; the gateway
consumes ``SessionManager.run().result()``, not the SSE ``stream_fn``.

Maintenance: the graph construction below mirrors NAT 1.7.x's
``react_agent_workflow`` register (``nat.plugins.langchain.agent.react_agent``).
On a NAT upgrade, diff this against that register — if NAT adds a
``ReActAgentGraph`` constructor argument or a config field, wire it here too,
or the fork silently drifts from upstream behavior.
"""

# No ``from __future__ import annotations`` here: NAT validates the workflow
# function's type hints via ``get_type_hints`` at registration, and string
# (forward-ref) annotations fail to resolve in that context — NAT's own
# register uses real annotations, so we match it. That makes the hint names
# (Builder, AsyncGenerator) runtime dependencies, so they cannot move under
# TYPE_CHECKING — hence the TC002/TC003 suppressions below.
from collections.abc import AsyncGenerator  # noqa: TC003 — runtime hint (no __future__ annotations)

from langchain_core.messages import trim_messages
from nat.builder.builder import Builder  # noqa: TC002 — runtime hint (no __future__ annotations)
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import (
    ChatRequest,
    ChatRequestOrMessage,
    ChatResponse,
    Usage,
)
from nat.plugins.langchain.agent.react_agent.agent import (
    ReActAgentGraph,
    ReActGraphState,
    create_react_agent_prompt,
)
from nat.plugins.langchain.agent.react_agent.register import ReActAgentWorkflowConfig
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
from nat.utils.type_converter import GlobalTypeConverter

# Workflow type name used in the Config and by builder.set_workflow.
REACT_WITH_STEPS_TYPE = "react_agent_with_steps"


# NAT's ReActAgentWorkflowConfig resolves to Any (untyped plugin import), so
# mypy flags subclassing it (misc); the metaclass `name=` kwarg is NAT's
# registration mechanism.
class ReActWithStepsConfig(ReActAgentWorkflowConfig, name=REACT_WITH_STEPS_TYPE):  # type: ignore[call-arg,misc]
    """ReAct workflow config that also emits LLM-tier intermediate steps.

    Inherits every field of NAT's ``ReActAgentWorkflowConfig`` — only the
    registered workflow type differs, so the agent behaves identically apart
    from the added LLM event stream.
    """


@register_function(  # type: ignore[untyped-decorator]  # NAT decorator is untyped
    config_type=ReActWithStepsConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def react_agent_with_steps(
    config: ReActWithStepsConfig,
    builder: Builder,
) -> AsyncGenerator[FunctionInfo]:
    """Build the ReAct agent graph and run it with a per-run profiler callback."""
    prompt = create_react_agent_prompt(config)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = await builder.get_tools(
        tool_names=config.tool_names,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN,
    )
    if not tools:
        msg = f"No tools specified for ReAct Agent '{config.llm_name}'"
        raise ValueError(msg)

    graph = await ReActAgentGraph(
        llm=llm,
        prompt=prompt,
        tools=tools,
        use_tool_schema=config.include_tool_input_schema_in_tool_description,
        detailed_logs=config.verbose,
        log_response_max_chars=config.log_response_max_chars,
        retry_agent_response_parsing_errors=config.retry_agent_response_parsing_errors,
        parse_agent_response_max_retries=config.parse_agent_response_max_retries,
        tool_call_max_retries=config.tool_call_max_retries,
        pass_tool_call_errors_to_agent=config.pass_tool_call_errors_to_agent,
        normalize_tool_input_quotes=config.normalize_tool_input_quotes,
        raise_on_parsing_failure=config.raise_on_parsing_failure,
        use_native_tool_calling=config.use_native_tool_calling,
    ).build_graph()

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """Invoke the graph for one turn, with LLM events flowing via the profiler."""
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)
        messages = trim_messages(
            messages=[m.model_dump() for m in message.messages],
            max_tokens=config.max_history,
            strategy="last",
            token_counter=len,
            start_on="human",
            include_system=True,
        )
        state = ReActGraphState(messages=messages)

        # Per-run: Context is active here, so the handler binds THIS run's step
        # manager (the one run_agent subscribes to). Passing it via the ainvoke
        # config propagates to the LLM's _stream_llm, which merges run + profiler
        # callbacks, so LLM_START/LLM_NEW_TOKEN/LLM_END reach the stream.
        profiler = LangchainProfilerHandler()
        result_state = await graph.ainvoke(
            state,
            config={
                "recursion_limit": (config.max_tool_calls + 1) * 2,
                "callbacks": [profiler],
            },
        )

        content = str(ReActGraphState(**result_state).messages[-1].content)
        prompt_tokens = sum(len(str(m.content).split()) for m in message.messages)
        completion_tokens = len(content.split()) if content else 0
        response = ChatResponse.from_string(
            content,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        if chat_request_or_message.is_string:
            return GlobalTypeConverter.get().convert(response, to_type=str)
        return response

    yield FunctionInfo.create(single_fn=_response_fn, description=config.description)
