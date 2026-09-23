# Evaluation

Findings from the design/eval cycle that shaped this project. All claims were
verified against live systems (home-ops cluster — Intel 13900H/96 GB nodes,
HuggingFace model card, or local smoke tests on Intel N305 CPU).

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

## ONNX runtime equivalence & hardware benchmarks (2026-09-23)

Measured during the v0.4.0 ONNX migration. **Production nodes are Intel
13900H (6P+8E, 96 GB RAM); the i3-N305 figures below are the dev box — a
conservative lower bound, not the deployment target.**

### Decision fidelity: ONNX fp32 vs live torch deployment

Same policies, same inputs, torch (0.3.0 pod, 13900H) vs the new ONNX path —
choice, per-option probabilities and confidence **identical to 4 decimals on
all cases**, plus batched multi-question calls (dynamic batch axis):

| case | torch (live) | ONNX fp32 |
|---|---|---|
| `mcp`: "show me the cilium daemonset pods" | kubernetes 0.5498 / conf 0.1506 | identical |
| `mcp`: "what did I bookmark about BGP" | fluxcd 0.2841 / conf 0.0031 (honest no-route) | identical |
| `tool`: "list the pods that are not running" | pods_list 1.0 / conf 1.0 | identical |
| `profile`: "hey good morning" | default 0.5054 / conf 0.1715 | identical |

### Latency and footprint

| runtime | 13900H (cluster) | N305 (dev box) | warm RSS | image |
|---|---|---|---|---|
| torch 0.3.0 (previous) | **0.3–0.6 s** (measured live, 2-CPU limit) | — | ~1.5 GiB | 5.36 GB |
| ONNX fp32 (default) | ~1.5–3 s *(inferred)* | 9 s (measured) | 2.4 GiB → 3 Gi limit | ~2 GB |
| ONNX int8 (build-arg) | ~1–1.7 s *(inferred)* | 5 s (measured) | 0.75 GiB → 1 Gi limit | ~800 MB |

### Findings that shaped the release

- **Static seq axis**: the export constant-folds seq_len at 512 (legacy
  TorchScript exporter), so short inputs pay full-length compute; torch pads
  to batch-max and is faster. marker/batch axes are truly dynamic (probed).
  Truly dynamic shapes need a `dynamo=True` re-export — open follow-up.
- **int8 is not free**: 2× faster and ¼ the size, but flipped 2/4 close-call
  choices and inflated calibration (0.51 → 0.75 observed). fp32 stays
  default; the shadow dataset feeds temperature refitting, so fidelity wins.
- **`act` head saturated**: raw outputs ±4000, `act_probability` pins at 1.0
  on every live answer in both runtimes. Model-quality issue for the
  fine-tune path; consumers must not gate on it.
- **96 GB nodes make fp32 sizing free**: int8's memory win is irrelevant on
  the production hardware; 3 Gi fp32 pods are trivial.

### Accelerator positioning

CPU on 13900H is homelab-adequate for advisory/shadow traffic (requests per
minute). Inline high-QPS enforcement (~33 ms/decision on the model card's
T4) wants an accelerator: 13900H Iris Xe iGPU (OpenVINO path, unverified),
an Apple-silicon host, or a cloud burst lane. No accelerator is required for
the shadow → fine-tune cycle itself (training is already kaggle-T4-planned).

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
