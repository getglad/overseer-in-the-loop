"""Fixtures for src/tools/ tests."""

from __future__ import annotations

import pytest
from nat.runtime.loader import PluginTypes, discover_and_register_plugins


@pytest.fixture(autouse=True, scope="module")
def _discover_nat_plugins() -> None:
    """Register NAT config-object plugins once per module.

    `discover_and_register_plugins` is idempotent process-wide, but pulling
    it into an autouse fixture removes the boilerplate from every test.
    """
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
