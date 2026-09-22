# Architecture

Design deep dive. For usage see [README.md](README.md).

## Positioning

`reflex` hosts System-1 decision models
([`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)) as a
production routing service. The model type defines the boundaries:

- **Typed questions in, calibrated decisions out** — `choice` (pick an option),
  `score` (ordinal rating), `noul` (probability of yes). One forward pass per
  call, ~33 ms on GPU, ~200–500 ms on CPU.
- **Answer space is request-time** — option markers mean new routes are config,
  not training.
- **No text generation ever** — nothing to parse, nothing to hallucinate.

This makes it the decision layer, sitting beside — never replacing — the
vector layer (embedding servers) and the generation layer (LLM gateways).

## Request flow

```
consumer → /decide/{policy} {state}
         → policies/{policy}.yaml → typed question schema
         → laya Router.preloaded[checkpoint]
         → {choice|score|noul, probability, routing metadata}
```

- `state` is free-form JSON (`{"text": ...}` or structured fields).
- The policy file owns the criteria (option names + descriptions); the model
  owns the judgment; probabilities let callers gate on confidence
  (`if probability < threshold: escalate/fallback`).

## Policies as the contract

`policies/*.yaml` maps a policy name to one typed question:

```yaml
mcp:
  type: choice
  instructions: Which MCP backend should handle this request?
  criteria:
    karakeep: bookmarks and saved articles
    hindsight: agent long-term memory
    netbox: infrastructure inventory, IPs, devices
    obsidian: personal knowledge notes
```

The same file is later reused as the **seed dataset schema** for fine-tuning:
one source of truth, two consumption modes (runtime config → training format).

## Checkpoints

| Checkpoint | Size | Use |
|---|---|---|
| `laya` (English, ModernBERT-large) | 808 MB | default |
| `laya/multilingual` (mmBERT-base) | 647 MB | non-English traffic |
| `laya/typed-decisions` | 808 MB | typed-workflow reference |
| *your fine-tune* | — | produced by [finetune/](finetune/) |

`Router(preload=True)` is mandatory: lazy loading costs 7–10 s per checkpoint
switch; preloaded switching is sub-millisecond. Bake weights into the image or
pull from an OCI registry at init — never at request time.

## Serving

FastAPI on uvicorn. Endpoints:

- `POST /decide/{policy}` — decision for a state against one policy
- `POST /decide` — raw `{state, questions}` passthrough (power users)
- `GET /healthz`, `GET /readyz` — liveness vs model-warm readiness
- MCP (Streamable HTTP, `/mcp`) — the same policies exposed as tools for
  MCP-native consumers (e.g. toolhive `MCPServer` CR)

Concurrency: ONNX/torch sessions allow concurrent inference; throughput needs
are trivial next to homelab traffic. Scale for HA (2 replicas, anti-affinity),
not throughput.

## Production fine-tune path

See [finetune/README.md](finetune/README.md) for the dataset contract. Stages:

1. **Shadow** — every `/decide` logs `{state, decision, probability, route_taken}`
   to stdout (collected by the platform log pipeline).
2. **Distill** — join decisions with outcomes (logs/traces), label via teacher
   LLM or rules.
3. **Fine-tune** — upstream RLCD notebook; refit temperature per
   (question type, option count) on your data — the base checkpoints are
   over-confident and near-chance zero-shot on typed workflows; this step is
   not optional for production trust.
4. **Repackage** — checkpoint versioned as an OCI artifact; promote
   shadow → enforce per policy.

## Failure posture

Reflex never denies traffic. On model load failure, bad input, or low
confidence it answers honestly: HTTP 503 (not ready), 400 (schema), or a
decision with its probability attached. Callers own their fallback. This is
the deliberate inverse of fail-closed guardrails: routing is advisory,
guarding is not.

## Cost tradeoffs (homelab)

- CPU-only is the default: 200–500 ms/decision is invisible for tool/KB
  routing and unacceptable only for inline chat-lane gating — which is
  deliberately out of scope.

### Measured on CPU (smoke-verified)

Validated end-to-end on an Intel N305 (no GPU): policy load → checkpoint
preload → `/decide/{policy}` and `/decide` answers with routing metadata.

- First calls per question shape: ~1–1.5 s; steady-state is lower. Inline
  chat-lane gating stays out of scope, as designed.
- `Router(preload=[...])` pulls tokenizer/config files for every checkpoint
  in the family even when only one is preloaded — budget image size for all
  of them.
- At load, the package warns: `checkpoint ships temperatures outside [0.5, 5]
  … treat confidence from the affected buckets as uncalibrated`. Observed
  live; reinforces that temperature refit on your data is a production
  requirement, not an option (see finetune/).
- Calibration behaves: a KB query routes to the right backend with high
  probability (karakeep 0.92), while an ops action against KB policies
  returns a spread with `confidence ≈ 0.02` — the honest "no route" signal
  callers should gate on.
- Memory: ~1.5 GB resident for both published checkpoints; single-checkpoint
  deployments halve that.
- Two replicas max, only when a consumer moves onto a critical path.
