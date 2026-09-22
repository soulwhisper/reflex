"""Tokenize policy JSONL datasets into laya training items.

Reads one `<policy>.jsonl` per policy from --datasets (dataset contract:
docs/FINETUNING.md), builds typed questions from the policies file (one
source of truth, same as runtime), and writes train/holdout tensors plus a
manifest with sha256 receipts.

Row schema: {"state": {...}, "policy": "mcp", "label": "karakeep"}
Optional soft targets via "probabilities": option-keyed (choice),
{"true": p} (noul), or {"0": p, ...} (score level index).

Run from the repo root: `python finetune/preprocess.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # script dir is sys.path[0]; add root for `app`

from app.config import load_policies


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_target(pol, row: dict) -> tuple[list[float], int]:
    """(target distribution, argmax label) for one labeled row."""
    t = pol.type
    crit = pol.criteria
    label = row["label"]
    probs = row.get("probabilities")

    if t == "choice":
        keys = list(crit.keys())
        if probs:
            target = [float(probs.get(k, 0.0)) for k in keys]
        else:
            if label not in crit:
                raise ValueError(f"label {label!r} not in {pol.name} criteria {sorted(crit)}")
            target = [float(k == label) for k in keys]
    elif t == "noul":
        if probs:
            target = [float(probs.get("false", 0.5)), float(probs.get("true", 0.5))]
        else:
            truth = label if isinstance(label, bool) else str(label).lower() == "true"
            target = [float(not truth), float(truth)]
    elif t == "score":
        levels = len(crit) if isinstance(crit, list) else 4
        if probs:
            target = [float(probs.get(str(i), 0.0)) for i in range(levels)]
        else:
            lvl = int(label)
            if not 0 <= lvl < levels:
                raise ValueError(f"score label {label!r} outside 0..{levels - 1} for {pol.name}")
            target = [float(i == lvl) for i in range(levels)]
    else:
        raise ValueError(f"unknown policy type {t!r} for {pol.name}")

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    return target, target.index(max(target))


def build_item(tok, cfg, pol, row: dict) -> dict | None:
    from laya.common import QTYPES, build_sequence, render_options

    target, label = gold_target(pol, row)
    q = {"t": pol.type, "crit": pol.criteria}
    if len(render_options(q)) != len(target):
        return None
    seq, markers = build_sequence(
        tok, row["state"], {"t": pol.type, "ins": pol.instructions, "crit": pol.criteria},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != len(target):
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[pol.type],
        "target": target,
        "label": label,
        "policy": pol.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="convaiinnovations/laya", help="base checkpoint (hub id or local dir)")
    ap.add_argument("--datasets", default="finetune/datasets", help="directory of <policy>.jsonl files")
    ap.add_argument("--policies", default="policies/routes.yaml", help="policy schema source of truth")
    ap.add_argument("--out", default="finetune/build", help="output directory for tensors + manifest")
    ap.add_argument("--holdout", type=float, default=0.2, help="stratified holdout fraction per policy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    policies = load_policies(ROOT / args.policies)
    model_dir = snapshot_download(args.model) if not Path(args.model).is_dir() else args.model
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())

    rng = random.Random(args.seed)
    train, holdout, manifest_files, dropped = [], [], [], 0
    for path in sorted((ROOT / args.datasets).glob("*.jsonl")):
        policy = path.stem
        if policy not in policies:
            print(f"skip {path.name}: no policy {policy!r} in {args.policies}")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        items = []
        for row in rows:
            it = build_item(tok, cfg, policies[policy], row)
            if it is None:
                dropped += 1
            else:
                items.append(it)
        rng.shuffle(items)
        n_hold = max(1, round(len(items) * args.holdout)) if len(items) >= 5 else 0
        holdout.extend(items[:n_hold])
        train.extend(items[n_hold:])
        manifest_files.append({"file": path.name, "sha256": sha256(path), "rows": len(rows),
                               "items": len(items), "holdout": n_hold})
        print(f"{policy}: {len(rows)} rows -> {len(items)} items ({n_hold} holdout)")

    if not train:
        sys.exit(f"no training items built from {args.datasets}; see dataset contract in docs/FINETUNING.md")
    if dropped:
        print(f"warning: {dropped} rows dropped (marker/option mismatch)")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train, out / "train_items.pt")
    torch.save(holdout, out / "holdout_items.pt")
    manifest = {"model": args.model, "policies": args.policies, "created": int(time.time()),
                "seed": args.seed, "holdout_fraction": args.holdout, "files": manifest_files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(train)} train / {len(holdout)} holdout items -> {out}")


if __name__ == "__main__":
    main()
