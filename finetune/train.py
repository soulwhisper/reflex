"""RLCD fine-tune of a laya checkpoint on policy routing items.

Scripted port of the upstream Kaggle notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb in
NandhaKishorM/laya): GRPO-style policy gradient over strictly proper
scoring-rule reward + soft cross-entropy guidance, DDP via torchrun.

Platform-agnostic: CUDA and Hygon DTK both expose the "cuda" device API and
an nccl-compatible backend; platform env lives in run_nvidia.sh /
run_k100ai.sh. Override the collectives backend with DIST_BACKEND if a DTK
build rejects "nccl".

Deviation from the notebook: calibration temperatures are fitted on the
holdout split, not on training data — fitting on train leaks calibration.

Launch: `torchrun --standalone --nproc_per_node=N finetune/train.py ...`
(they launchers do this; single process works too: --nproc_per_node=1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for items.py
from items import collate_batch, load_items

QTEMP_NAMES = None  # inverse QTYPES map, built in main


def fit_one_temp(sel: list[tuple[list, list]]) -> float:
    """LBFGS temperature fit on (logits, target) pairs; 1.0 if data is thin."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def save_checkpoint(model, tok, cfg, out_dir: str, extra_meta: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if extra_meta is not None:
        with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
            json.dump(extra_meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="base checkpoint (hub id or local dir)")
    ap.add_argument("items", help="train_items.pt from preprocess.py")
    ap.add_argument("out", help="output checkpoint directory")
    ap.add_argument("--holdout", help="holdout_items.pt for calibration (default: fit on train subset)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4, help="GRPO baseline samples")
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from laya.common import QTYPES, build_model, proper_reward
    from transformers import AutoTokenizer

    global QTEMP_NAMES
    # Allow direct `python finetune/train.py` single-process runs (torchrun
    # sets these itself; single-card K100AI/DTK fallback path).
    for k, v in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
                 "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
        os.environ.setdefault(k, v)

    QTEMP_NAMES = {v: k for k, v in QTYPES.items()}

    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        sys.exit("no accelerator visible; run on a CUDA/DTK host (see run_nvidia.sh / run_k100ai.sh)")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir = args.model if Path(args.model).is_dir() else snapshot_download(args.model)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = args.max_len
    cfg["head_max_len"] = args.head_max_len

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True
    ) if world_size > 1 else model

    all_items = load_items(args.items)
    my_items = all_items[rank::world_size]

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, len(my_items) // (args.micro_batch * args.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        print(f"RLCD training: {len(all_items)} items | {world_size} proc(s) | {args.epochs} epochs")
    t0 = time.time()

    for epoch in range(args.epochs):
        random.seed(args.seed + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress

        for b_idx in range(0, len(my_items), args.micro_batch):
            chunk = my_items[b_idx:b_idx + args.micro_batch]
            if not chunk:
                continue
            batch = collate_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # G noisy logit distributions with zero-mean projection (exploration)
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # strictly proper scoring-rule reward, group-mean advantage
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % args.grad_accum == 0 or (b_idx + args.micro_batch) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if rank == 0 and (n_batches % 50) == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch + 1}/{args.epochs} | step {n_batches} | "
                      f"loss {loss.item() * args.grad_accum:.4f} | reward {r.mean().item():.3f} | lr {lr:.2e}")

        if rank == 0:
            avg = epoch_loss / max(1, n_batches)
            print(f"=== epoch {epoch + 1}/{args.epochs} in {time.time() - t0:.1f}s | avg loss {avg:.4f} ===")
            save_checkpoint(model, tok, cfg, os.path.join(args.out, "checkpoint_latest"),
                            {"epoch": epoch + 1, "total_epochs": args.epochs, "avg_loss": avg})
        if world_size > 1:
            dist.barrier()

    # Post-training calibration on the holdout split (never on train data).
    if rank == 0:
        print("fitting calibration temperatures on holdout...")
        model.eval()
        calib_items = load_items(args.holdout) if args.holdout else all_items[::15][:400]
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device), cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r_i, it in enumerate(c_chunk):
                    calib_preds.append((it["qtype"], list(l_np[r_i, : len(it["markers"])]), it["target"]))

        # per-question-type temperatures, stored as the loader-compatible list
        temps = [1.0] * (max(QTEMP_NAMES) + 1)
        detailed = {}
        for qt, name in QTEMP_NAMES.items():
            sel = [(z, t) for q, z, t in calib_preds if q == qt]
            if len(sel) < 10:
                print(f"warning: only {len(sel)} holdout items for {name!r}; "
                      "temperature stays 1.0 — treat its probabilities as uncalibrated")
            temps[qt] = fit_one_temp(sel)
            detailed[name] = temps[qt]
        print("fitted temperatures:", {k: round(v, 3) for k, v in detailed.items()})

        cfg["fine_tuned"] = True
        cfg["temperature"] = temps
        save_checkpoint(model, tok, cfg, args.out)
        with open(os.path.join(args.out, "temperatures.json"), "w") as f:
            json.dump(detailed, f, indent=2)
        print(f"saved checkpoint -> {args.out}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
