"""Export a laya checkpoint to ONNX (full graph: backbone + decision head),
optionally quantize to INT8, and validate against the PyTorch reference.

Approach adopted from tozp/laya-onnx (HF): legacy TorchScript exporter
(dynamo=False) with dynamic batch/seq/marker axes — one artifact serves every
policy shape. Unlike that repo we validate on real pipeline items (policy
sequences), not random token IDs, and gate on argmax agreement and
probability drift rather than per-output relative error.

Run from the repo root:
  python finetune/export_onnx.py --checkpoint finetune/build/out --int8 \
      --validate-with finetune/build/holdout_items.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for items.py


def load_torch_model(checkpoint: str):
    from huggingface_hub import snapshot_download
    from laya.common import build_model
    from safetensors.torch import load_file

    model_dir = checkpoint if Path(checkpoint).is_dir() else snapshot_download(checkpoint)
    cfg = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    model = build_model(cfg, encoder_dir=Path(model_dir) / "encoder")
    model.load_state_dict(load_file(Path(model_dir) / "model.safetensors"), strict=True)
    return model.cpu().eval()


def export(model, out_path: Path, opset: int, seq_len: int, markers: int) -> None:
    input_ids = torch.randint(0, 1000, (1, seq_len))
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    marker_pos = torch.arange(markers).unsqueeze(0) * 5 + 10
    marker_mask = torch.ones(1, markers, dtype=torch.bool)
    qtype = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        model(input_ids, attention_mask, marker_pos, marker_mask, qtype)  # trace sanity pass
    torch.onnx.export(
        model,
        args=(input_ids, attention_mask, marker_pos, marker_mask, qtype),
        f=str(out_path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "markers"},
            "marker_mask": {0: "batch", 1: "markers"},
            "qtype": {0: "batch"},
            "logits": {0: "batch"},
            "act": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(out_path)))


def quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=str(src), model_output=str(dst),
                     weight_type=QuantType.QInt8, per_channel=True, reduce_range=False)


def validate(model, onnx_path: Path, items: list[dict], device: torch.device) -> tuple[float, float]:
    """Returns (argmax agreement, max |Δp|) over masked option probabilities."""
    import onnxruntime as ort
    from items import collate_batch

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    agree, max_dp, n = 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            batch = collate_batch(chunk, pad_id=0)
            ref_logits, _ = model(batch["input_ids"], batch["attention_mask"],
                                  batch["marker_pos"], batch["marker_mask"], batch["qtype"])
            (ort_logits,) = sess.run(["logits"], {
                "input_ids": batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
                "marker_pos": batch["marker_pos"].numpy(),
                "marker_mask": batch["marker_mask"].numpy(),
                "qtype": batch["qtype"].numpy(),
            })
            for r, it, o in zip(chunk, ref_logits, ort_logits):
                k = len(it["markers"])
                p_ref = torch.softmax(r[:k].float(), -1)
                p_ort = torch.softmax(torch.tensor(o[:k], dtype=torch.float32), -1)
                agree += int(torch.argmax(p_ref) == torch.argmax(p_ort))
                max_dp = max(max_dp, float((p_ref - p_ort).abs().max()))
                n += 1
    return agree / max(1, n), max_dp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="convaiinnovations/laya", help="checkpoint dir or hub id")
    ap.add_argument("--out", default="finetune/build/onnx", help="output directory")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--int8", action="store_true", help="also emit dynamic INT8 (per-channel) variant")
    ap.add_argument("--validate-with", help="holdout_items.pt; default: dummy-shaped smoke pass only")
    ap.add_argument("--gate", action="store_true", help="exit 1 if validation drift exceeds bounds")
    ap.add_argument("--min-agree", type=float, default=0.99)
    ap.add_argument("--max-dp", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_torch_model(args.checkpoint)

    fp32_path = out / "model.onnx"
    print(f"exporting {args.checkpoint} -> {fp32_path} (opset {args.opset})")
    export(model, fp32_path, args.opset, seq_len=512, markers=16)
    print(f"  fp32: {fp32_path.stat().st_size / 1e6:.1f} MB")

    variants = [fp32_path]
    if args.int8:
        int8_path = out / "model_int8.onnx"
        quantize_int8(fp32_path, int8_path)
        print(f"  int8: {int8_path.stat().st_size / 1e6:.1f} MB")
        variants.append(int8_path)

    if args.validate_with:
        from items import load_items

        items = load_items(args.validate_with)
        device = torch.device("cpu")
        failed = False
        for path in variants:
            agree, max_dp = validate(model, path, items, device)
            ok = agree >= args.min_agree and max_dp <= args.max_dp
            failed |= not ok
            print(f"  validate {path.name}: argmax agree {agree:.4f}, max |Δp| {max_dp:.4f} "
                  f"-> {'OK' if ok else 'DRIFT'}")
        if failed and args.gate:
            sys.exit(1)


if __name__ == "__main__":
    main()
