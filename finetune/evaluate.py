"""Evaluate a checkpoint on holdout items; print metrics and gate verdicts.

Metrics per policy and overall: argmax accuracy, ECE (laya.common.ece_score),
Brier, mean confidence. Compare against a baseline checkpoint with
--baseline to judge the promotion gates in docs/FINETUNING.md (+5 points,
ECE <= 0.10). Exits 1 when --gate is given and any gate fails.

Run from the repo root: `python finetune/evaluate.py --checkpoint ... --items ...`
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for items.py
from items import collate_batch, load_items


def load_model(checkpoint: str, device: torch.device):
    from huggingface_hub import snapshot_download
    from laya.common import build_model
    from safetensors.torch import load_file

    model_dir = checkpoint if Path(checkpoint).is_dir() else snapshot_download(checkpoint)
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    model = build_model(cfg, encoder_dir=Path(model_dir) / "encoder")
    model.load_state_dict(load_file(Path(model_dir) / "model.safetensors"), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(model, items: list[dict], device: torch.device, batch_size: int = 16):
    """Per item: masked softmax probabilities over its option markers."""
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        batch = collate_batch(chunk, pad_id=0)
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
        logits = logits.float().cpu()
        for r, it in zip(chunk, logits):
            k = len(it["markers"])
            probs = torch.softmax(r[:k], -1).tolist()
            out.append({"policy": it["policy"], "qtype": it["qtype"], "probs": probs,
                        "label": it["label"], "target": it["target"]})
    return out


def metrics(preds: list[dict]) -> dict:
    from laya.common import ece_score

    correct = np.array([float(np.argmax(p["probs"]) == p["label"]) for p in preds])
    conf = np.array([max(p["probs"]) for p in preds])
    brier = float(np.mean([np.sum((np.array(p["probs"]) - np.array(p["target"])) ** 2) for p in preds]))
    return {
        "n": len(preds),
        "accuracy": round(float(correct.mean()), 4),
        "ece": round(float(ece_score(conf, correct)), 4),
        "brier": round(brier, 4),
        "mean_conf": round(float(conf.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir or hub id")
    ap.add_argument("--items", required=True, help="holdout_items.pt from preprocess.py")
    ap.add_argument("--baseline", help="optional baseline checkpoint for delta comparison")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gate", action="store_true", help="exit 1 if promotion gates fail")
    ap.add_argument("--gate-ece", type=float, default=0.10)
    ap.add_argument("--gate-acc-gain", type=float, default=0.05, help="min accuracy gain vs --baseline")
    args = ap.parse_args()

    device = torch.device(args.device)
    items = load_items(args.items)
    if not items:
        sys.exit("empty holdout; preprocess with enough rows (>=5 per policy)")

    model = load_model(args.checkpoint, device)
    preds = predict_probs(model, items, device)
    overall = metrics(preds)

    per_policy = defaultdict(list)
    for p in preds:
        per_policy[p["policy"]].append(p)
    print(f"== {args.checkpoint} on {overall['n']} holdout items ({device}) ==")
    for name in sorted(per_policy):
        print(f"  {name}: {metrics(per_policy[name])}")
    print(f"  OVERALL: {overall}")

    failures = []
    base_overall = None
    if args.baseline:
        base_preds = predict_probs(load_model(args.baseline, device), items, device)
        base_overall = metrics(base_preds)
        gain = overall["accuracy"] - base_overall["accuracy"]
        print(f"  BASELINE ({args.baseline}): {base_overall} | acc gain {gain:+.4f}")
        if gain < args.gate_acc_gain:
            failures.append(f"accuracy gain {gain:+.4f} < {args.gate_acc_gain}")
    if overall["ece"] > args.gate_ece:
        failures.append(f"ECE {overall['ece']} > {args.gate_ece}")

    if failures:
        print("GATES FAIL: " + "; ".join(failures))
        if args.gate:
            sys.exit(1)
    else:
        print("GATES PASS" if (args.baseline or args.gate) else "no gates evaluated (pass --baseline for full gates)")


if __name__ == "__main__":
    main()
