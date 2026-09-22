"""MCP tools over Streamable HTTP, mountable into the FastAPI app.

Exposes the routing policies as MCP tools for MCP-native consumers (e.g.
toolhive MCPServer CR). Served in-process at /mcp by app.server; can also
run standalone (python -m app.mcp).
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .config import load_policies

POLICIES_PATH = os.environ.get("REFLEX_POLICIES", "policies/routes.yaml")
CHECKPOINTS = [c.strip() for c in os.environ.get("REFLEX_CHECKPOINTS", "english").split(",") if c.strip()]

policies = load_policies(POLICIES_PATH)
mcp = FastMCP("reflex", streamable_http_path="/")
_router = None


def get_router():
    global _router
    if _router is None:
        import laya

        _router = laya.Router(preload=CHECKPOINTS)
    return _router


def _decide(policy: str, text: str) -> dict:
    pol = policies[policy]
    res = get_router().predict({"text": text}, pol.question())
    return {"policy": policy, "answer": res["answers"][policy], "routing": res.get("routing", {})}


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
    """ASGI app serving MCP at the mount prefix."""
    return mcp.streamable_http_app()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
