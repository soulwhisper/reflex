# Roadmap

## Goals

1. **Production routing as a service**: MCP/tool/profile decisions with
   calibrated probabilities, consumable over HTTP and MCP.
2. **Config-driven first**: new routes are YAML, shipped without retraining.
3. **Evidence-gated fine-tuning**: base checkpoint until shadow data proves a
   fine-tuned head is better; fine-tuning is an operational stage, not research.
4. **Homelab-shaped**: CPU-only, baked weights, HA-by-replicas-not-scale,
   never in a deny path.

## Phases

| Phase | Deliverable | Gate to exit |
|---|---|---|
| **0 — Scaffold** ✅ | repo, HTTP+MCP serving, policies, CI, k8s bundle | smoke-verified end-to-end (done) |
| **1 — v0.1.0** | tagged release, ghcr image (public) | CI green, image pulls |
| **2 — Cluster mount** | `MCPServer` CR into toolhive `internal-ro` group (home-ops PR) | tools visible via vmcp, live decision call succeeds |
| **3 — First consumer** | one real consumer wired (KB federation search via vmcp, or hermes profile routing) | decision log shows real traffic |
| **4 — Shadow mode** | decision+outcome logging to the platform log pipeline; weekly accuracy report | ≥2 weeks of shadow data |
| **5 — Fine-tune v1** | domain JSONL → RLCD fine-tune → temperature refit → packaged checkpoint | beats base on holdout + calibration ECE < 0.1 |
| **6 — Promote** | shadow new checkpoint vs base; enforce per policy | agreement ≥ base, no regression |
| **7 — Later (candidates)** | score-based KB merge rerank, typed-decision workflows (alert triage), multilingual checkpoint, ONNX export for latency | one at a time, with evidence |

## Non-goals (permanent)

- No embeddings, no generation, no inline chat-lane gating.
- No guardrail deny path (fail-closed lives elsewhere).
- No high-cardinality (>20-option) policies without the card's mitigations
  (raise `head_max_len` or coarse-to-fine two-step).
