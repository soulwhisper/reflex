"""MCP tools over Streamable HTTP, mountable into the FastAPI app.

Exposes the routing policies as MCP tools for MCP-native consumers (e.g.
toolhive MCPServer CR). Served in-process at /mcp by app.server; can also
run standalone (python -m app.mcp).
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from .config import load_policies
from .telemetry import decision_span, init_tracing

POLICIES_PATH = os.environ.get("REFLEX_POLICIES", "policies/routes.yaml")
MODEL_DIR = os.environ.get("REFLEX_MODEL_DIR", "model")
policies = load_policies(POLICIES_PATH)

mcp = MCPServer("reflex")
_model = None


def get_model():
    global _model
    if _model is None:
        from .model import ONNXModel

        _model = ONNXModel(MODEL_DIR)
    return _model


def _decide(policy: str, text: str) -> dict:
    pol = policies[policy]
    model = get_model()
    with decision_span(policy, "mcp", f"route_{policy}", {"text": text}, model.repo) as rec:
        res = model.predict({"text": text}, pol.question())
        answer = res["answers"][policy]
        rec.record(answer)
    return {"policy": policy, "answer": answer, "routing": res.get("routing", {})}


@mcp.tool()
def route_mcp(text: str) -> dict:
    """Decide which MCP backend should handle a request."""
    return _decide("mcp", text)


@mcp.tool()
def route_tool(text: str) -> dict:
    """Decide which tool to call for a request."""
    return _decide("tool", text)


@mcp.tool()
def route_profile(text: str) -> dict:
    """Decide which agent profile should handle a message."""
    return _decide("profile", text)


@mcp.tool()
def reflex_decide(state: dict, questions: dict) -> dict:
    """Raw typed-question decision: pass a state and a question schema."""
    return get_model().predict(state, questions)


def create_mcp_app():
    """ASGI app serving MCP at the mount prefix (path moved here in mcp v2).

    Host-header protection (mcp v2): the SDK auto-enables DNS-rebinding
    protection with a localhost-only allowlist when no settings are passed.
    REFLEX_ALLOWED_HOSTS controls it:
      - unset: SDK default (localhost-only — right for local dev)
      - comma-separated host[:port] list: protection on, those hosts allowed
      - "*": protection explicitly OFF — for proxy-fronted deploys where the
        front (toolhive) dials dynamic endpoint IPs, so no static allowlist
        can be correct. Only use behind a trusted in-cluster proxy.
    """
    allowed = [h.strip() for h in os.environ.get("REFLEX_ALLOWED_HOSTS", "").split(",") if h.strip()]
    kwargs = {}
    if allowed == ["*"]:
        from mcp.server.transport_security import TransportSecuritySettings

        kwargs["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    elif allowed:
        from mcp.server.transport_security import TransportSecuritySettings

        kwargs["transport_security"] = TransportSecuritySettings(allowed_hosts=allowed)
    return mcp.streamable_http_app(streamable_http_path="/", **kwargs)


if __name__ == "__main__":
    init_tracing()
    mcp.run(transport="streamable-http")
