"""ONNX runtime for laya typed-decision checkpoints (tozp/laya-onnx layout).

Replaces the torch stack (laya.Router) with a single onnxruntime session:
ModernBERT backbone + decision head exported as one graph
(input_ids, attention_mask, marker_pos, marker_mask, qtype -> logits, act).

The encode/decode contract mirrors laya.common/laya.agent exactly — sequence
layout, marker placement, temperature clamping, confidence — so choices and
calibrated probabilities are interchangeable with the torch reference. Model
dir layout (as shipped by tozp/laya-onnx and by finetune/export_onnx.py):

    model.onnx            full-graph export (backbone + decision head)
    tokenizer.json        ModernBERT fast tokenizer
    rl_agent_config.json  max_len / head_max_len / fitted temperatures
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

# See laya.common: fitted temperatures below 0.5 sharpen a coin flip into a
# published certainty; outside [0.5, 5.0] is treated as uncalibrated.
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def clamp_temperature(t) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(t) or math.isinf(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: dict) -> list[str]:
    """Option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {render_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


class ONNXModel:
    """System-1 decision model on onnxruntime; drop-in for laya's predict."""

    def __init__(self, model_dir: str, repo: str | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = Path(model_dir)
        self.repo = repo or os.environ.get("REFLEX_MODEL_REPO", "local")
        self.cfg = json.loads((self.dir / "rl_agent_config.json").read_text())
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        for tok in ("[CLS]", "[SEP]", "[MASK]", "[PAD]"):
            if self.tok.token_to_id(tok) is None:
                raise ValueError(f"tokenizer at {self.dir} lacks required special token {tok!r}")
        self.cls_id = self.tok.token_to_id("[CLS]")
        self.sep_id = self.tok.token_to_id("[SEP]")
        self.mask_id = self.tok.token_to_id("[MASK]")
        self.pad_id = self.tok.token_to_id("[PAD]")

        self.temperature = [clamp_temperature(t) for t in self.cfg.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.cfg.get("temperature_by_options", {}).items()
        }
        self.sess = ort.InferenceSession(str(self.dir / "model.onnx"), providers=["CPUExecutionProvider"])

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def build_sequence(self, state, q: dict, max_len: int, head_max_len: int):
        """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]"""
        mask_str = "[MASK]"
        opts = render_options(q)
        ins = str(q["ins"]).replace(mask_str, " ")
        head_ids = self._encode(f"{q['t']} question: {ins}")
        opt_ids = []
        for opt in opts:
            opt_ids.append([self.mask_id] + self._encode(" " + opt.replace(mask_str, " "))[:48])
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, max_len - len(ids) - 1)
        st = self._encode(serialize_state(state).replace(mask_str, " "))[:room]
        ids = ids + st + [self.sep_id]
        return ids[:max_len], [m for m in markers if m < max_len]

    @staticmethod
    def _to_internal(qdef: dict) -> dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    def predict(self, state, questions: dict, model: str | None = None) -> dict:
        """Evaluate typed questions across state in one batched forward pass.

        `model` is accepted for laya API compatibility; the single bundled
        checkpoint always serves (multi-checkpoint routing was a torch
        Router feature — bundle the checkpoint you want at image build).
        """
        ids_q = list(questions.keys())
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)

        items = []
        for qid in ids_q:
            q = self._to_internal(questions[qid])
            seq, markers = self.build_sequence(state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r} options exceed head_max_len={head_max_len}")
            items.append({"markers": markers, "qtype": QTYPES[q["t"]], "ids": seq})

        # The exported graph has seq_len constant-folded at max_len (legacy
        # TorchScript exporter, do_constant_folding=True): pad every row to
        # exactly max_len. marker/batch axes are truly dynamic — verified
        # against tozp/laya-onnx.
        n, L = len(items), max_len
        kmax = max(len(it["markers"]) for it in items)
        input_ids = np.full((n, L), self.pad_id, dtype=np.int64)
        att = np.zeros((n, L), dtype=np.int64)
        mpos = np.zeros((n, kmax), dtype=np.int64)
        mmask = np.zeros((n, kmax), dtype=bool)
        qtype = np.array([it["qtype"] for it in items], dtype=np.int64)
        for i, it in enumerate(items):
            input_ids[i, : len(it["ids"])] = it["ids"]
            att[i, : len(it["ids"])] = 1
            k = len(it["markers"])
            mpos[i, :k] = it["markers"]
            mmask[i, :k] = True

        logits, act = self.sess.run(
            ["logits", "act"],
            {
                "input_ids": input_ids,
                "attention_mask": att,
                "marker_pos": mpos,
                "marker_mask": mmask,
                "qtype": qtype,
            },
        )
        logits = logits.astype(np.float32)
        act = act.astype(np.float32)
        act = np.exp(act - act.max(-1, keepdims=True))
        act = act / act.sum(-1, keepdims=True)

        answers = {}
        n_tokens = int(att.sum())
        for r, qid in enumerate(ids_q):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                top = int(p.argmax())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[top],
                    "probability": round(float(p[top]), 4),
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probability": round(float(p[int(p.argmax())]), 4),
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "probability": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }

        return {
            "model": "laya-onnx",
            "answers": answers,
            "routing": {"model": self.cfg.get("model_name", "rl-agent"), "repo": self.repo},
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }
