# Fine-tuning: plans and methods

Fine-tuning here is **production work**: a repeatable pipeline that turns
platform traffic into a better checkpoint. The reference shape is
`convaiinnovations/laya-typed-decisions` — a fine-tuned head on the same base.

## Why it is required (not optional)

Measured facts (see [EVALUATION.md](EVALUATION.md)):

- Base checkpoints are near-chance zero-shot on typed workflows (0.362 vs
  0.318 random on the upstream typed-decisions benchmark).
- They ship over-confident: raw ECE 0.466; domain temperature refit brings it
  to ~0.081. The package itself warns at load (`temperatures outside [0.5,5]`)
  — observed in our smoke test.
- Our policies are domain-specific (homelab MCP backends, tools, profiles) —
  generic calibration does not transfer.

## Method

### 1. Shadow mode (data collection)

Every `/decide` response already carries `policy`, `answer`, `confidence`,
`routing`. Log one line per request:

```json
{"ts": "...", "policy": "mcp", "state": {"text": "..."},
 "decision": "karakeep", "probability": 0.91, "confidence": 0.74,
 "route_taken": null}
```

`route_taken` is filled by the consumer (what actually happened downstream:
which tool was invoked, which profile answered, whether it succeeded). The
platform log pipeline (fluent-bit → VictoriaLogs) collects these; a weekly
job computes per-policy accuracy and confidence histograms.

### 2. Dataset (JSONL contract)

One file per policy, one decision per line:

```json
{"state": {"text": "what did I bookmark about cilium"}, "policy": "mcp", "label": "karakeep"}
```

Sources, in priority order:

1. **Outcome-joined shadow logs** (route_taken + success) — highest trust.
2. **Teacher labels** — the production LLM grades unlabeled states
   (cheap: only disagreements and low-confidence rows need the teacher).
3. **Rule labels** — where deterministic rules already exist (e.g. profile
   routing heuristics), mine them for bulk labels.
4. **Policy examples** — the `criteria` descriptions in `policies/*.yaml`
   seed few-shot rows for cold-start classes.

Quality bars: ≥200 rows per policy before training; class balance within 2×;
scrubbed with the platform's redaction rules before leaving the cluster.

### 3. Training

Local scripted pipeline (ported from the upstream RLCD Kaggle notebook —
[provenance](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)):
`finetune/` holds the stages (`preprocess.py` → `train.py` → `evaluate.py`
→ optional `export_onnx.py`); see `finetune/README.md`.

- Base: `laya` (English) for now; `laya-multilingual` when non-English
  traffic matters.
- Hardware: `finetune/run_nvidia.sh` for CUDA hosts (2×T4-class suffices),
  `finetune/run_k100ai.sh` for Hygon K100AI (DTK) hosts. Small head-only
  fine-tune first (`--lr-encoder 0`); full fine-tune only if head-only
  underperforms.
- Holdout: 20% stratified, never seen by teacher or rules.

### 4. Calibration (mandatory)

Refit temperature per (question type, option count) on the holdout — this is
what moves ECE into the trustworthy range. Ship the temperature map next to
the checkpoint; the loader applies it.

### 5. Evaluation gates

A fine-tuned checkpoint promotes only if **all** hold:

- holdout accuracy ≥ base + 5 points (or ≥ shadow-measured production accuracy
  of the incumbent),
- ECE ≤ 0.10 on holdout,
- no per-class regression > 3 points,
- shadow-vs-incumbent agreement review on 1 week of live traffic.

### 6. Packaging & promotion

- Checkpoint directory + `DATASET.jsonl.sha256` receipt + temperature map →
  versioned OCI artifact (registry of choice).
- Bake via `docker build --build-arg CHECKPOINTS=...`; tag image with the
  checkpoint version (`0.2.0-ft1`, …).
- Promote per policy: shadow → enforce; rollback = redeploy previous image.

## What we will NOT fine-tune

- Guardrail/deny-path decisions (fail-closed sidecars own those, locally).
- High-cardinality option spaces without the model card's mitigations.
- Anything whose dataset receipt can't be produced.
