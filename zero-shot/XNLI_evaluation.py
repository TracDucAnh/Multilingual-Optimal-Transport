"""
XNLI_evaluation.py
==================
Zero-shot evaluation của Llama-3-8B (vanilla, không fine-tune) trên toàn bộ
tập dữ liệu XNLI — tất cả 15 ngôn ngữ.

Scoring Strategy — PMI Log-prob:
    Mỗi sample được score bằng PMI (Pointwise Mutual Information):

        pmi_score(c) = mean_nll(prompt + candidate_c)
                     - mean_nll(null_prefix + candidate_c)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                prior NLL (không có context)

    Prediction = candidate có PMI score thấp nhất (log-prob cao nhất
    sau khi đã trừ đi bias prior của từng label).

    Tại sao cần PMI?
    ----------------
    Llama-3 có prior mạnh cho " entailment" (token ngắn, common token).
    Nếu chỉ dùng mean_nll thô → model luôn predict "entailment" (~33%
    random) bất kể nội dung. PMI normalization loại bỏ bias này bằng
    cách trừ đi NLL của candidate khi không có context.

Pipeline:
    1. Load XNLIDataLoader  (log-prob mode, right-padded batches)
    2. Tính prior NLL của mỗi candidate label (1 forward pass nhỏ)
    3. Forward pass toàn bộ dataset → per-token CrossEntropyLoss
    4. pmi_score = mean_nll - prior_nll  → argmin → predicted label
    5. Tính Accuracy per-lang + overall

Kết quả lưu vào:
    results/xnli_vanilla/
        summary.json       — overall + per_lang accuracy
        per_sample.jsonl   — từng sample: pred_label, gold_label, correct,
                             nll (raw), pmi (normalized)

Usage:
    python XNLI_evaluation.py
    python XNLI_evaluation.py --batch_size 16 --max_length 256
    python XNLI_evaluation.py --data_root ../raw_data/ --output_dir results/xnli_vanilla/
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
# Cau truc project:  Multilingual OT/
#                        dataloader/downstream_dataloader.py
#                        zero-shot/XNLI_evaluation.py
# parent.parent = project root (Multilingual OT/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import XNLIDataLoader, XNLI_CANDIDATES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# XNLI_CANDIDATES = ["entailment", "neutral", "contradiction"]
# cand_id:           0              1          2
LABEL_NAMES = XNLI_CANDIDATES   # ["entailment", "neutral", "contradiction"]

# Null prefix dùng để tính prior NLL — phải khớp với đuôi prompt trong dataloader
# _build_xnli_prompt() kết thúc bằng "Relationship:" nên dùng chuỗi này làm prefix
NULL_PREFIX = "Relationship:"


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dtype_str: str = "bf16"):
    """Load Llama-3-8B vanilla."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"[Model] Loading {MODEL_NAME}  dtype={dtype_str} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded. Parameters: {n_params / 1e9:.2f}B\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# PMI Prior Computation
# ---------------------------------------------------------------------------

def compute_prior_nll(
    model,
    tokenizer,
    candidates: List[str],
    null_prefix: str = NULL_PREFIX,
) -> torch.Tensor:
    """
    Tính mean NLL của mỗi candidate khi không có premise/hypothesis.

    Dùng null_prefix = "Relationship:" (đuôi của prompt template) để
    prior context khớp với vị trí candidate trong full prompt.

    Token layout:
        [BOS] <null_prefix tokens> <candidate tokens> [EOS]
        labels: -100 ... -100       <candidate>       [EOS]

    Returns
    -------
    prior_nll : FloatTensor [C]  — mean NLL per candidate
    """
    device   = next(model.parameters()).device
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    prefix_ids = tokenizer.encode(null_prefix, add_special_tokens=False)
    prior_nlls = []

    print(f"\n[Prior] Computing prior NLL with null_prefix='{null_prefix}'")

    for cand in candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        full_ids = bos_ids + prefix_ids + cand_ids + eos_ids

        # Labels: mask prefix, score candidate + EOS
        prompt_len = len(bos_ids) + len(prefix_ids)
        labels     = [-100] * prompt_len + cand_ids + eos_ids

        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        label_tensor = torch.tensor([labels],   dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids=input_tensor)

        shift_logits = out.logits[:, :-1].contiguous()          # [1, L-1, V]
        shift_labels = label_tensor[:, 1:].contiguous()         # [1, L-1]

        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                              # [1, L-1]

        n_cand   = (shift_labels != -100).sum().clamp(min=1)
        mean_nll = (token_loss.sum() / n_cand).item()
        prior_nlls.append(mean_nll)

        cand_decoded = tokenizer.decode(cand_ids)
        print(f"  prior_nll('{cand_decoded}') = {mean_nll:.6f}  "
              f"[{len(cand_ids)} token(s): {cand_ids}]")

    prior_tensor = torch.tensor(prior_nlls, dtype=torch.float32)
    print(f"[Prior] Done: {prior_tensor.tolist()}\n")
    return prior_tensor   # [C]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    data_root: str  = "../raw_data/",
    output_dir: str = "results/xnli_vanilla/",
    batch_size: int = 16,
    max_length: int = 256,
) -> Dict:
    """
    Chạy toàn bộ PMI log-prob evaluation trên XNLI.

    PMI scoring:
        - Tính prior_nll[c] cho mỗi candidate (1 forward pass nhỏ)
        - Mỗi sample được mở rộng thành 3 rows (C=3 candidates)
        - Forward → shift logits/labels → CrossEntropyLoss per token
        - mean_nll[i] = mean loss trên candidate tokens (ignore -100)
        - pmi_score[i] = mean_nll[i] - prior_nll[cand_id[i]]
        - pred = argmin(pmi_score) trên C candidates của cùng sample_id

    Returns
    -------
    summary : dict — overall + per_lang accuracy
    """
    device = next(model.parameters()).device

    # ── Tính prior NLL trước (chỉ C=3 forward passes nhỏ) ────────────────
    # Candidates trong dataloader có leading space: " entailment", ...
    # (xem XNLIDataset.__init__: candidates = [f" {w}" for w in XNLI_CANDIDATES])
    candidates_with_space = [f" {w}" for w in LABEL_NAMES]
    prior_nll = compute_prior_nll(
        model, tokenizer, candidates_with_space, null_prefix=NULL_PREFIX
    )  # FloatTensor [C]

    # ── DataLoader ────────────────────────────────────────────────────────
    loader = XNLIDataLoader(
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    C = loader.num_candidates   # 3
    print(f"[Eval] Total batches: {len(loader)}  batch_size={batch_size}  C={C}\n")

    # ── Output setup ──────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.jsonl"

    # ── Collectors ────────────────────────────────────────────────────────
    all_nll       : List[torch.Tensor] = []
    all_sample_id : List[torch.Tensor] = []
    all_cand_id   : List[torch.Tensor] = []
    all_gold      : List[torch.Tensor] = []
    all_lang      : List[str]          = []

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    t0 = time.time()

    for batch in tqdm(loader, desc="XNLI zero-shot eval"):
        input_ids      = batch["input_ids"].to(device)       # [B*C, L]
        attention_mask = batch["attention_mask"].to(device)  # [B*C, L]
        labels         = batch["labels"].to(device)          # [B*C, L]

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Shift: logits[t] predicts token[t+1]
        logits       = out.logits                          # [B*C, L, V]
        shift_logits = logits[:, :-1].contiguous()         # [B*C, L-1, V]
        shift_labels = labels[:, 1:].contiguous()          # [B*C, L-1]

        # Per-token loss, -100 positions auto-zeroed by ignore_index
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # [B*C*(L-1), V]
            shift_labels.view(-1),                          # [B*C*(L-1)]
        ).view(shift_labels.size())                         # [B*C, L-1]

        # Mean NLL over candidate tokens only (non -100)
        n_ans    = (shift_labels != -100).sum(-1).clamp(min=1)  # [B*C]
        mean_nll = token_loss.sum(-1) / n_ans                    # [B*C]

        all_nll.append(mean_nll.cpu())
        all_sample_id.append(batch["sample_id"])   # [B*C]
        all_cand_id.append(batch["cand_id"])        # [B*C]
        all_gold.append(batch["gold_cand_id"])      # [B*C]
        all_lang.extend(batch["lang"])              # List[str] len B*C

    elapsed = time.time() - t0

    # ── Aggregate NLL → PMI scores → predictions ──────────────────────────
    all_nll       = torch.cat(all_nll)        # [N_total]
    all_sample_id = torch.cat(all_sample_id)  # [N_total]
    all_cand_id   = torch.cat(all_cand_id)    # [N_total]
    all_gold      = torch.cat(all_gold)       # [N_total]

    n_samples = int(all_sample_id.max().item()) + 1

    # Raw NLL matrix [n_samples, C]
    nll_mat = torch.full((n_samples, C), float("inf"))
    nll_mat[all_sample_id, all_cand_id] = all_nll.float()

    # PMI score = mean_nll - prior_nll  (broadcast prior [C] across n_samples)
    # prior_nll [C] → [1, C] → broadcast to [n_samples, C]
    pmi_mat = nll_mat - prior_nll.unsqueeze(0)   # [n_samples, C]

    # Gold labels [n_samples]
    gold_mat = torch.zeros(n_samples, dtype=torch.long)
    gold_mat[all_sample_id] = all_gold

    # Lang per sample (first occurrence)
    lang_of: List[str] = [""] * n_samples
    for sid, lg in zip(all_sample_id.tolist(), all_lang):
        if lang_of[sid] == "":
            lang_of[sid] = lg

    # Predict bằng PMI (thấp nhất = tốt nhất)
    pred    = pmi_mat.argmin(-1)              # [n_samples]
    correct = (pred == gold_mat)              # [n_samples] bool

    print(f"\n[Eval] Finished {n_samples:,} samples in {elapsed:.1f}s "
          f"({n_samples / max(elapsed, 1):.1f} samples/s)")

    # ── Write per_sample.jsonl ────────────────────────────────────────────
    with open(per_sample_path, "w", encoding="utf-8") as fout:
        for sid in range(n_samples):
            record = {
                "sample_id":  sid,
                "lang":       lang_of[sid],
                "pred_label": LABEL_NAMES[int(pred[sid].item())],
                "gold_label": LABEL_NAMES[int(gold_mat[sid].item())],
                "correct":    bool(correct[sid].item()),
                # Raw NLL scores
                "nll":  [round(nll_mat[sid, c].item(), 6) for c in range(C)],
                # PMI-normalized scores (dùng để predict)
                "pmi":  [round(pmi_mat[sid, c].item(), 6) for c in range(C)],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Per-language accuracy ─────────────────────────────────────────────
    per_lang: Dict[str, List[bool]] = defaultdict(list)
    for sid in range(n_samples):
        per_lang[lang_of[sid]].append(bool(correct[sid].item()))

    per_lang_summary: Dict[str, Dict] = {}

    print(f"\n{'═'*58}")
    print(f"  XNLI Zero-shot Results (PMI) — {MODEL_NAME}")
    print(f"{'═'*58}")
    print(f"  {'Language':<14}  {'N':>6}  {'Accuracy (%)':>13}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*13}")

    for lang in sorted(per_lang.keys()):
        vals = per_lang[lang]
        n    = len(vals)
        acc  = 100.0 * sum(vals) / n
        per_lang_summary[lang] = {"n": n, "accuracy": round(acc, 2)}
        print(f"  {lang:<14}  {n:>6,}  {acc:>13.2f}")

    overall_acc = 100.0 * correct.float().mean().item()
    print(f"  {'─'*14}  {'─'*6}  {'─'*13}")
    print(f"  {'OVERALL':<14}  {n_samples:>6,}  {overall_acc:>13.2f}")
    print(f"{'═'*58}\n")

    # ── Confusion matrix ──────────────────────────────────────────────────
    conf_overall = [[0] * C for _ in range(C)]
    for sid in range(n_samples):
        p = int(pred[sid].item())
        g = int(gold_mat[sid].item())
        conf_overall[p][g] += 1

    # In confusion matrix để debug distribution
    print(f"  Confusion matrix (row=pred, col=gold):")
    print(f"  {'':18s}  " + "  ".join(f"{l:>13}" for l in LABEL_NAMES))
    for pi, row in enumerate(conf_overall):
        print(f"  {LABEL_NAMES[pi]:18s}  " + "  ".join(f"{v:>13,}" for v in row))
    print()

    # ── Build summary ─────────────────────────────────────────────────────
    summary = {
        "model":    MODEL_NAME,
        "task":     "XNLI",
        "mode":     "zero-shot PMI log-prob scoring",
        "n_samples": n_samples,
        "candidates": LABEL_NAMES,
        "prior_nll": {
            LABEL_NAMES[c]: round(prior_nll[c].item(), 6) for c in range(C)
        },
        "overall": {
            "accuracy": round(overall_acc, 2),
        },
        "per_lang": per_lang_summary,
        "confusion_matrix": {
            "labels": LABEL_NAMES,
            "matrix": conf_overall,   # conf[pred][gold]
        },
        "eval_config": {
            "max_length":  max_length,
            "batch_size":  batch_size,
            "null_prefix": NULL_PREFIX,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    # ── Save summary.json ─────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Output] summary    → {summary_path}")
    print(f"[Output] per_sample → {per_sample_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XNLI zero-shot PMI evaluation — Llama-3-8B vanilla baseline"
    )
    parser.add_argument(
        "--data_root", type=str, default="../raw_data/",
        help="Path đến thư mục raw_data/ (default: ../raw_data/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/xnli_vanilla/",
        help="Thư mục lưu kết quả (default: results/xnli_vanilla/)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Số sample gốc per batch — thực tế rows = batch_size × 3 (default: 16)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Max sequence length in tokens (default: 256)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype (default: bf16)"
    )
    parser.add_argument(
        "--null_prefix", type=str, default=NULL_PREFIX,
        help=f"Null prefix để tính prior NLL (default: '{NULL_PREFIX}')"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Override NULL_PREFIX nếu user truyền vào
    import XNLI_evaluation as _self
    _self.NULL_PREFIX = args.null_prefix

    # HuggingFace login nếu cần
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[HF] Logged in with HF_TOKEN from .env\n")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(dtype_str=args.dtype)

    # Run evaluation
    summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("✓ Evaluation complete.")
    print(f"  Overall Accuracy = {summary['overall']['accuracy']:.2f}%")