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
CHECKPOINTS = [c.strip() for c in os.environ.get("REFLEX_CHECKPOINTS", "english").split(",") if c.strip()]
policies = load_policies(POLICIES_PATH)

mcp = MCPServer("reflex")
_router = None


def get_router():
    global _router
    if _router is None:
        import laya

        _router = laya.Router(preload=CHECKPOINTS)
    return _router


def _decide(policy: str, text: str) -> dict:
    pol = policies[policy]
    model = ",".join(CHECKPOINTS)
    with decision_span(policy, "mcp", f"route_{policy}", {"text": text}, model) as rec:
        res = get_router().predict({"text": text}, pol.question())
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
    return get_router().predict(state, questions)

def create_mcp_app():
    """ASGI app serving MCP at the mount prefix (path moved here in mcp v2)."""
    return mcp.streamable_http_app(streamable_http_path="/")


if __name__ == "__main__":
    init_tracing()
    mcp.run(transport="streamable-http")
