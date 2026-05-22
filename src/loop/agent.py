"""Workflow construction via WorkflowBuilder — no YAML.

Builds a NAT ReAct agent with:
- LLM setup against any OpenAI-compatible provider (configured via env)
- Function/tool registration
- HITL approval on every tool call
- OTel tracing (optional)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef, LLMRef
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.plugins.opentelemetry.register import OtelCollectorTelemetryExporter
from nat.runtime.loader import PluginTypes, discover_and_register_plugins

from src.loop.hitl import prompt_binary_approval
from src.loop.prompts import AGENT_SYSTEM_PROMPT, tool_approval_prompt
from src.loop.react_steps import REACT_WITH_STEPS_TYPE, ReActWithStepsConfig

if TYPE_CHECKING:
    from nat.builder.builder import Builder
    from nat.builder.workflow_builder import WorkflowBuilder

logger = structlog.get_logger()

REJECTION_MESSAGE = "Tool call was rejected by the user."

# Typed catalog refs — one name per site, mypy-typo-safe.
MAIN_LLM = LLMRef("main_llm")
CURRENT_DATETIME = FunctionRef("current_datetime")

# Env var names. Exported so test fixtures and downstream domains can
# reference them by name instead of duplicating string literals.
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_MODEL_ENV = "LLM_MODEL"

# Defaults used when LLM_BASE_URL / LLM_MODEL are not set. Overriding via env
# is how readers swap to any OpenAI-compatible provider without code changes.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL_NAME = "z-ai/glm-5.1"


def _require_env(name: str) -> str:
    """Read a required env var. Raises RuntimeError with a pointer to mise.local.toml."""
    value = os.environ.get(name)
    if not value:
        msg = (
            f"{name} is not set. Configure it in mise.local.toml under [env] "
            "(see mise.local.toml.example for the shape) and restart the server."
        )
        raise RuntimeError(msg)
    return value


def _default_base_url() -> str:
    """Read ``LLM_BASE_URL`` or fall back to the NVIDIA NIM endpoint."""
    return os.environ.get(LLM_BASE_URL_ENV) or DEFAULT_BASE_URL


def _default_model_name() -> str:
    """Read ``LLM_MODEL`` or fall back to ``DEFAULT_MODEL_NAME``."""
    return os.environ.get(LLM_MODEL_ENV) or DEFAULT_MODEL_NAME


def _default_otel_endpoint() -> str | None:
    """Read ``OTEL_ENDPOINT`` from env — optional OTLP collector URL."""
    return os.environ.get("OTEL_ENDPOINT") or None


@dataclass(frozen=True)
class AgentSettings:
    """Configuration for building the agent workflow.

    Only ``LLM_API_KEY`` is required. ``LLM_BASE_URL`` and ``LLM_MODEL``
    default to NVIDIA NIM + ``DEFAULT_MODEL_NAME`` — set them in
    ``mise.local.toml`` to swap to any other OpenAI-compatible provider.
    Sampling knobs (``temperature``, ``top_p``, ``max_tokens``) are
    project tuning and stay as code defaults.
    """

    model_name: str = field(default_factory=_default_model_name)
    base_url: str = field(default_factory=_default_base_url)
    api_key: str = field(
        default_factory=lambda: _require_env(LLM_API_KEY_ENV),
        repr=False,
    )
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    otel_endpoint: str | None = field(default_factory=_default_otel_endpoint)
    otel_project: str = "agent-auto-mode"


# ---------------------------------------------------------------------------
# HITL-wrapped tools — approval required before execution
# ---------------------------------------------------------------------------


# FunctionBaseConfig uses metaclass __init_subclass__ kwargs that mypy can't resolve
class HITLCurrentTimeConfig(FunctionBaseConfig, name="hitl_current_datetime"):  # type: ignore[call-arg,misc]
    """Current datetime tool that requires HITL approval before executing."""


# NAT's register_function decorator lacks type stubs
@register_function(config_type=HITLCurrentTimeConfig)  # type: ignore[untyped-decorator]
async def hitl_current_datetime(
    _config: HITLCurrentTimeConfig,
    _builder: Builder,
) -> FunctionInfo:
    """Current datetime tool wrapped with HITL approval.

    Matches NAT's official HITL pattern: approval happens inside the tool
    function body, before the actual logic executes. On rejection, returns
    a message the agent can use to replan.
    """

    async def _get_current_time(query: str) -> str:
        # NAT requires one positional param; we don't use it.
        del query
        approved = await prompt_binary_approval(
            tool_approval_prompt("current_datetime", "get current date and time"),
        )
        if not approved:
            return REJECTION_MESSAGE

        now = datetime.now(tz=UTC)
        return f"The current time of day is {now.strftime('%Y-%m-%d %H:%M:%S %z')}"

    yield FunctionInfo.from_fn(
        _get_current_time,
        description=(
            "Returns the current date and time in human readable format. "
            "Requires user approval before executing."
        ),
    )


async def configure_telemetry(
    builder: WorkflowBuilder,
    *,
    endpoint: str | None,
    project: str = "agent-auto-mode",
) -> None:
    """Add OTel tracing to the workflow builder if an endpoint is configured.

    NAT uses its own span pipeline (IntermediateStep -> Span -> OtelSpan -> OTLP),
    not OpenTelemetry auto-instrumentation. This function wires that pipeline.

    Args:
        builder: The NAT WorkflowBuilder to configure.
        endpoint: OTLP endpoint URL. None skips telemetry entirely.
        project: Project name for trace grouping.
    """
    if endpoint is None:
        logger.info("otel_skipped", reason="no endpoint configured")
        return

    otel_config = OtelCollectorTelemetryExporter(
        project=project,
        endpoint=endpoint,
    )
    await builder.add_telemetry_exporter("otel", otel_config)
    logger.info("otel_configured", endpoint=endpoint, project=project)


async def configure_builder(
    builder: WorkflowBuilder,
    settings: AgentSettings | None = None,
) -> Config:
    """Configure a WorkflowBuilder with LLM, tools, HITL, and OTel.

    The builder must already be entered as an async context manager.
    Constructs the workflow entirely in Python — no YAML files.
    Returns the Config for SessionManager.create().

    Args:
        builder: An active WorkflowBuilder (inside `async with`).
        settings: Agent configuration. Uses defaults if None.

    Returns:
        Config object for creating a SessionManager with this workflow.
    """
    if settings is None:
        settings = AgentSettings()

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    await builder.add_llm(
        MAIN_LLM,
        OpenAIModelConfig(
            model_name=settings.model_name,
            base_url=settings.base_url,
            api_key=settings.api_key,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            # GLM-5.x requires thinking mode for reliable tool calling.
            # Other models may ignore or reject this parameter.
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "clear_thinking": False,
                },
            },
        ),
    )

    await builder.add_function(CURRENT_DATETIME, HITLCurrentTimeConfig())

    tool_names: list[FunctionRef] = [CURRENT_DATETIME]

    await configure_telemetry(
        builder,
        endpoint=settings.otel_endpoint,
        project=settings.otel_project,
    )

    await builder.set_workflow(
        ReActWithStepsConfig(
            tool_names=tool_names,
            llm_name=MAIN_LLM,
            verbose=True,
            use_native_tool_calling=True,
            system_prompt=AGENT_SYSTEM_PROMPT,
            description="A code-first ReAct agent with HITL approval",
            # Some models (e.g. GLM-5.x) occasionally blend native tool-call
            # XML with ReAct prose on their first attempt; allow a few retries.
            parse_agent_response_max_retries=3,
        ),
    )

    logger.info(
        "workflow_built",
        model=settings.model_name,
        tools=tool_names,
        otel=settings.otel_endpoint is not None,
    )

    return Config(
        workflow={
            "_type": REACT_WITH_STEPS_TYPE,
            "tool_names": tool_names,
            "llm_name": MAIN_LLM,
        },
    )
