"""Decision tracing: one OTEL span per policy decision, env-gated.

Active only when OTEL_EXPORTER_OTLP_ENDPOINT is set; standard OTEL_*
env vars configure the SDK, which is inert otherwise. Deployments that
record decisions for fine-tuning should set OTEL_TRACES_SAMPLER=always_on:
the SDK default (parentbased_always_on) drops spans under unsampled
parents, and front-proxies (e.g. toolhive) sample aggressively — that
would silently shrink the shadow dataset.

Spans carry gen_ai.* semantic attributes so collector pipelines filtering
on gen_ai.system admit them and langfuse renders them as generations.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

tracer = None


def init_tracing() -> None:
    """Install the TracerProvider; no-op without an OTLP endpoint. Idempotent."""
    global tracer
    if tracer is not None or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "reflex")})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("reflex")


def shutdown_tracing() -> None:
    """Flush pending spans; call on app shutdown or BatchSpanProcessor loses them."""
    global tracer
    if tracer is None:
        return
    from opentelemetry import trace

    trace.get_tracer_provider().shutdown()
    tracer = None


@contextmanager
def decision_span(policy: str, surface: str, tool: str, arguments: dict, model: str | None):
    """Wrap one policy decision; yields a recorder for the answer.

    Raw-schema decisions (/decide, reflex_decide) are intentionally not
    traced: they carry no policy label and their question schema varies
    per call, which does not fit the fine-tune dataset contract.
    """
    if tracer is None:
        yield _Recorder(None)
        return
    with tracer.start_as_current_span(f"reflex.decide.{policy}") as span:
        span.set_attribute("gen_ai.system", "reflex")
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", tool)
        span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))
        span.set_attribute("reflex.policy", policy)
        span.set_attribute("reflex.surface", surface)
        if model:
            span.set_attribute("gen_ai.request.model", model)
        yield _Recorder(span)


class _Recorder:
    def __init__(self, span):
        self._span = span

    def record(self, answer) -> None:
        if self._span is None:
            return
        self._span.set_attribute("reflex.answer", json.dumps(answer))
        if isinstance(answer, dict):
            if answer.get("choice") is not None:
                self._span.set_attribute("reflex.answer.choice", str(answer["choice"]))
            if answer.get("probability") is not None:
                self._span.set_attribute("reflex.answer.probability", float(answer["probability"]))
