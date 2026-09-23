"""reflex app: System-1 decisions over HTTP and MCP."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_policies
from .mcp import create_mcp_app
from .telemetry import decision_span, init_tracing, shutdown_tracing

POLICIES_PATH = os.environ.get("REFLEX_POLICIES", "policies/routes.yaml")
MODEL_DIR = os.environ.get("REFLEX_MODEL_DIR", "model")

policies = load_policies(POLICIES_PATH)
mcp_app = create_mcp_app()
router = None  # ONNXModel, built at startup
ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, ready
    init_tracing()
    from .model import ONNXModel

    router = ONNXModel(MODEL_DIR)
    ready = True
    # run the MCP session manager alongside the HTTP app
    async with mcp_app.router.lifespan_context(mcp_app):
        yield
    shutdown_tracing()

app = FastAPI(title="reflex", version="0.1.0", lifespan=lifespan)
app.mount("/mcp", mcp_app)


class DecideRequest(BaseModel):
    state: dict


class RawRequest(BaseModel):
    state: dict
    questions: dict


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    if not ready:
        raise HTTPException(503, "model loading")
    return {"ready": True, "model": router.repo}


@app.post("/decide/{policy}")
def decide(policy: str, req: DecideRequest) -> dict:
    if not ready:
        raise HTTPException(503, "checkpoints loading")
    pol = policies.get(policy)
    if pol is None:
        raise HTTPException(404, f"unknown policy {policy!r}; have {sorted(policies)}")
    t0 = time.monotonic()
    with decision_span(policy, "http", policy, {"state": req.state}, router.repo) as rec:
        res = router.predict(req.state, pol.question())
        answer = res["answers"][policy]
        rec.record(answer)
    return {
        "policy": policy,
        "answer": answer,
        "routing": res.get("routing", {}),
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    }


@app.post("/decide")
def decide_raw(req: RawRequest) -> dict:
    if not ready:
        raise HTTPException(503, "checkpoints loading")
    t0 = time.monotonic()
    res = router.predict(req.state, req.questions)
    res["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return res
