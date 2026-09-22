# Evaluation

Findings from the design/eval cycle that shaped this project. All claims were
verified against live systems (home-ops cluster, HuggingFace model card, or
local smoke tests on Intel N305 CPU).

## Model selection: why System-1 (`convaiinnovations/laya`)

| Candidate | Verdict | Evidence |
|---|---|---|
| TypeSafe Jev (closed API) | **Rejected** | Closed weights, $0.042/1M tokens, external dependency (violates the platform's no-external-LLM rule), p50 236–276 ms |
| **`convaiinnovations/laya`** | **Selected** | Apache-2.0 self-hosted; beats Jev on published benchmarks (typed-decisions 0.766 vs 0.727, AG News 0.950 vs 0.910, ECE 0.081 vs 0.246, ~7.8× faster); built-in `Router` with script detection |
| Generic classifier (self-trained ModernBERT) | Backup only | No calibrated probabilities, no request-time schema, no router — rebuild everything |

Key model-card facts that shaped the design:

- **Not an embedding/rerank/LLM model.** No vectors, no generation. Early
  drafts assigned embedding/rerank fallback to this service; that was wrong
  and was removed. Vectors belong to TEI-style servers; generation to the LLM
  gateway.
- **Request-time answer space** (option markers): new routes are YAML, not
  retraining. The policy file IS the question schema.
- **Base checkpoints are near-chance zero-shot on typed workflows** (0.362 vs
  0.318 random) and ship over-confident (ECE 0.466 raw → 0.081 after domain
  temperature refit). Fine-tuning + temperature refit is a production
  requirement, not an enhancement.
- **High-cardinality weakness**: >20 options per question degrades accuracy at
  default token budgets. All current policies have ≤6 options — safe.

## Hosting: node runtime vs TEI

TEI was rejected on architecture grounds: it serves embedding/rerank models
and does not understand laya's decision head, option markers, or the `Router`
script-dispatch logic. The model runs via its own pip package inside a
FastAPI process ("node runtime" in-cluster). ONNX export (ort stack) is a
later optimization, not a prerequisite.

## Smoke-test results (Intel N305, CPU-only, no GPU)

| Check | Result |
|---|---|
| Checkpoint load + preload (first run, cold cache) | ~340 s incl. ~1.5 GB download |
| `/decide/mcp`: "what did I bookmark about cilium networking last week" | `karakeep` p=0.918 ✓ |
| `/decide/mcp`: "which vlan is the frigate camera on" | `netbox` p=0.560 ✓ (appropriately lower confidence) |
| `/decide/profile`: "run the nightly backup now" | `ops` p=0.905 ✓ |
| Out-of-domain ask against KB policies | spread with `confidence ≈ 0.02` — honest no-route signal ✓ |
| First-call latency per question shape | 0.95–1.5 s cold; lower steady-state |
| Load warning | `checkpoint ships temperatures outside [0.5,5]` — confirms the refit requirement |

Latency verdict: fine for async routing (tool/KB/profile), **not** acceptable
for inline chat-lane gating — which stays out of scope.

## Architecture decisions record

1. **Decision layer only.** No embeddings (can't), no generation (can't), no
   deny path (fail-closed belongs to guardrails with local models).
2. **Config-first routing.** YAML policies = typed question schemas; the same
   file is the fine-tune dataset seed later. Fine-tuning is a refinement
   stage, gated by shadow-mode evidence.
3. **MCP + HTTP dual surface.** FastAPI for direct consumers; MCP (Streamable
   HTTP) for toolhive-native consumers, mounted in-process at `/mcp`.
4. **Replicas for HA, not throughput.** One replica at day 1 (routing is
   advisory); two with pod anti-affinity when a consumer lands on a critical
   path. CPU capped at 2 cores — control-plane protection first.
5. **Weights baked at build time.** No runtime downloads; image is the model
   artifact (OCI distribution fits the platform's registry habits).
