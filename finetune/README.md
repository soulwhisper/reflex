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

Optional soft targets for calibration-sensitive rows: `"probabilities": {"karakeep": 0.9, ...}`
(choice), `{"true": 0.8}` (noul), `{"0": 0.1, "1": 0.9, ...}` (score levels).
Without it, `label` becomes a one-hot target.

Sources (all collectable from platform logs): shadow-mode `/decide` logs joined
with the route actually taken; agent/tool-call logs; teacher-LLM labels for
unlabeled traffic.

## Pipeline stages

Scripted port of the upstream RLCD Kaggle notebook
([provenance](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)),
with calibration fitted on the holdout split (the notebook fits on train
data, which leaks calibration).

```bash
pip install -r finetune/requirements.txt

bash finetune/run_nvidia.sh    # CUDA hosts (1-2 GPUs, e.g. Kaggle 2xT4)
bash finetune/run_k100ai.sh    # Hygon K100AI hosts (DTK stack, gfx928)
```

Both launchers run the same four stages; only preflight and accelerator env
differ (`DIST_BACKEND=gloo` override if a DTK build rejects `nccl`):

1. **`preprocess.py`** — `datasets/*.jsonl` + `policies/routes.yaml` →
   tokenized train/holdout items + `manifest.json` with per-file sha256
   receipts. Question schemas come from the same policies file the service
   loads — one source of truth.
2. **`train.py`** (via `torchrun`) — RLCD: GRPO-style group baseline over a
   strictly proper scoring-rule reward + soft cross-entropy. Full fine-tune
   (encoder lr 2.5e-5, head lr 1e-4, 4 epochs, σ 0.4→0.1). Rolling
   `checkpoint_latest/` per epoch; final save includes refit temperatures.
   Head-only fine-tune: freeze by setting `--lr-encoder 0` as the cheap first
   pass; full run only if it underperforms.
3. **`evaluate.py`** — holdout accuracy / ECE / Brier per policy and overall,
   optional `--baseline` delta. Promotion gates (`--gate`): accuracy gain
   ≥ +5 points over incumbent, ECE ≤ 0.10.
4. **`export_onnx.py`** (optional optimization) — full-graph ONNX export
   (backbone + decision head, dynamic batch/seq/markers), `--int8` dynamic
   per-channel quantization, validation on holdout items with drift gates
   (argmax agreement ≥ 0.99, max |Δp| ≤ 0.02).

## Promotion

- Checkpoint directory + `temperatures.json` + `manifest.json` receipts →
  versioned OCI artifact; bake via `docker build --build-arg CHECKPOINTS=...`.
- Promote per policy: shadow → enforce; rollback = redeploy previous image.
- Remaining gates from docs/FINETUNING.md apply: no per-class regression > 3
  points, one week of shadow-vs-incumbent agreement review.

## Hard rules

- Never fine-tune into the deny path (guardrails stay fail-closed and local).
- Never train on secrets; scrub with the same redaction the platform uses.
- Keep the JSONL with the model artifact — a checkpoint without its dataset
  receipt is undeployable.
