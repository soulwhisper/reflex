# Fine-tuning pipeline (production)

Fine-tuning here is an operational stage, not research. The output is a
versioned checkpoint promoted per policy — the same relationship
`convaiinnovations/laya-typed-decisions` has to `laya`.

## Dataset contract

One JSONL per policy, one decision per line:

```json
{"state": {"text": "what did I bookmark about cilium"}, "policy": "mcp", "label": "karakeep"}
```

Datasets land in `datasets/` (one `<policy>.jsonl` per policy, plus the
`<policy>.jsonl.sha256` receipt at packaging time). Datasets are artifacts,
not source — keep raw traffic-derived files out of git.

Sources (all collectable from platform logs): shadow-mode `/decide` logs joined
with the route actually taken; agent/tool-call logs; teacher-LLM labels for
unlabeled traffic.

## Stages

1. **Shadow** — log `{state, decision, probability, route_taken}` per request.
2. **Distill** — build the JSONL; teacher = production LLM for unlabeled rows,
   rule labels where rules exist.
3. **Fine-tune** — upstream RLCD notebook:
   <https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb>
   Then **refit temperature per (question type, option count) on your data** —
   base checkpoints ship over-confident (ECE 0.466 raw → 0.081 after refit);
   skipping this makes probabilities untrustworthy.
4. **Repackage** — checkpoint as versioned OCI artifact; bake via
   `docker build --build-arg CHECKPOINTS=...` or pull at init.
5. **Promote** — shadow the new checkpoint against the incumbent; enforce only
   when agreement and calibration both improve.

## Hard rules

- Never fine-tune into the deny path (guardrails stay fail-closed and local).
- Never train on secrets; scrub with the same redaction the platform uses.
- Keep the JSONL with the model artifact — a checkpoint without its dataset
  receipt is undeployable.
