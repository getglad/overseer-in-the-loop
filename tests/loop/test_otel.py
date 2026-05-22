"""Tests for OTel span emission configuration in the agent loop.

Verifies that the workflow builder:
- Configures a telemetry exporter with the correct endpoint when one is provided
- Skips telemetry when no endpoint is given
"""

from nat.plugins.opentelemetry.register import OtelCollectorTelemetryExporter

from src.loop.agent import configure_telemetry


class TestOTelConfiguration:
    """Tests for OTel telemetry wiring in configure_telemetry."""

    async def test_otel_exporter_uses_correct_endpoint(self, mock_builder):
        """Registers the exporter with the correct name and endpoint when provided."""
        endpoint = "http://jaeger:4318/v1/traces"
        await configure_telemetry(mock_builder, endpoint=endpoint)

        mock_builder.add_telemetry_exporter.assert_called_once()
        call_args = mock_builder.add_telemetry_exporter.call_args
        assert call_args[0][0] == "otel"
        exporter_config = call_args[0][1]
        assert isinstance(exporter_config, OtelCollectorTelemetryExporter)
        assert exporter_config.endpoint == endpoint

    async def test_skips_otel_when_no_endpoint(self, mock_builder):
        """WHEN otel_endpoint is None THEN no telemetry exporter is added."""
        await configure_telemetry(mock_builder, endpoint=None)

        mock_builder.add_telemetry_exporter.assert_not_called()
