"""reflex serving: System-1 decisions over HTTP."""
from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_policies

POLICIES_PATH = os.environ.get("REFLEX_POLICIES", "policies/routes.yaml")
CHECKPOINTS = [c.strip() for c in os.environ.get("REFLEX_CHECKPOINTS", "english").split(",") if c.strip()]

app = FastAPI(title="reflex", version="0.1.0")
policies = load_policies(POLICIES_PATH)
router = None  # laya Router, built at startup
ready = False


class DecideRequest(BaseModel):
    state: dict
    model: str | None = None  # explicit checkpoint override


class RawRequest(BaseModel):
    state: dict
    questions: dict
    model: str | None = None


@app.on_event("startup")
def startup() -> None:
    global router, ready
    import laya

    router = laya.Router(preload=CHECKPOINTS)
    ready = True


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    if not ready:
        raise HTTPException(503, "checkpoints loading")
    return {"ready": True, "checkpoints": CHECKPOINTS}


@app.post("/decide/{policy}")
def decide(policy: str, req: DecideRequest) -> dict:
    if not ready:
        raise HTTPException(503, "checkpoints loading")
    pol = policies.get(policy)
    if pol is None:
        raise HTTPException(404, f"unknown policy {policy!r}; have {sorted(policies)}")
    kwargs = {"model": req.model} if req.model else {}
    t0 = time.monotonic()
    res = router.predict(req.state, pol.question(), **kwargs)
    answer = res["answers"][policy]
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
    kwargs = {"model": req.model} if req.model else {}
    t0 = time.monotonic()
    res = router.predict(req.state, req.questions, **kwargs)
    res["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return res
