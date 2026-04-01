"""
MMMLU_evaluation.py
===================
Zero-shot evaluation của Llama-3-8B (vanilla, không fine-tune) trên toàn bộ
tập dữ liệu MMMLU — tất cả ngôn ngữ có sẵn.

Scoring Strategy — Log-prob:
    Mỗi sample được score bằng cách tính mean NLL của 4 candidate labels:
        " A" / " B" / " C" / " D"
    Prediction = candidate có NLL thấp nhất (log-prob cao nhất).

Pipeline:
    1. Load MMLUDownstreamDataLoader  (log-prob mode, right-padded batches)
    2. Forward pass → per-token CrossEntropyLoss
    3. mean NLL per candidate → argmin → predicted label
    4. Tính Accuracy per-lang + overall

Kết quả lưu vào:
    results/mmmlu_vanilla/
        summary.json       — overall + per_lang accuracy
        per_sample.jsonl   — từng sample: pred_label, gold_label, correct, nll scores

Usage:
    python MMMLU_evaluation.py
    python MMMLU_evaluation.py --batch_size 8 --max_length 512
    python MMMLU_evaluation.py --data_root ../raw_data/ --output_dir results/mmmlu_vanilla/
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
#                        zero-shot/MMMLU_evaluation.py
# parent.parent = project root (Multilingual OT/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import MMLUDownstreamDataLoader, MCQ_OPTIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# MCQ_OPTIONS = ["A", "B", "C", "D"]
# cand_id:       0    1    2    3
LABEL_NAMES = MCQ_OPTIONS  # ["A", "B", "C", "D"]


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
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    data_root: str  = "../raw_data/",
    output_dir: str = "results/mmmlu_vanilla/",
    batch_size: int = 8,
    max_length: int = 512,
) -> Dict:
    """
    Chạy toàn bộ log-prob evaluation trên MMMLU.

    Log-prob scoring:
        - Mỗi sample mở rộng thành 4 rows (C=4 candidates: A/B/C/D)
        - Forward pass → shift logits/labels → CrossEntropyLoss per token
        - mean_nll[i] = mean loss trên candidate tokens (ignore_index=-100)
        - pred = argmin(mean_nll) trên C=4 candidates của cùng sample_id

    Returns
    -------
    summary : dict — overall + per_lang accuracy
    """
    device = next(model.parameters()).device

    # ── DataLoader ────────────────────────────────────────────────────────
    loader = MMLUDownstreamDataLoader(
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    C = loader.num_candidates   # 4
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

    for batch in tqdm(loader, desc="MMMLU zero-shot eval"):
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

        # Per-token loss; -100 positions auto-zeroed via ignore_index
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # [B*C*(L-1), V]
            shift_labels.view(-1),                          # [B*C*(L-1)]
        ).view(shift_labels.size())                         # [B*C, L-1]

        # Mean NLL trên candidate tokens only (non -100)
        n_ans    = (shift_labels != -100).sum(-1).clamp(min=1)  # [B*C]
        mean_nll = token_loss.sum(-1) / n_ans                    # [B*C]

        all_nll.append(mean_nll.cpu())
        all_sample_id.append(batch["sample_id"])    # [B*C]
        all_cand_id.append(batch["cand_id"])         # [B*C]
        all_gold.append(batch["gold_cand_id"])       # [B*C]
        all_lang.extend(batch["lang"])               # List[str] len B*C

    elapsed = time.time() - t0

    # ── Aggregate NLL → predictions ───────────────────────────────────────
    all_nll       = torch.cat(all_nll)        # [N_total]
    all_sample_id = torch.cat(all_sample_id)  # [N_total]
    all_cand_id   = torch.cat(all_cand_id)    # [N_total]
    all_gold      = torch.cat(all_gold)       # [N_total]

    n_samples = int(all_sample_id.max().item()) + 1

    # NLL matrix [n_samples, C]
    nll_mat = torch.full((n_samples, C), float("inf"))
    nll_mat[all_sample_id, all_cand_id] = all_nll.float()

    # Gold labels [n_samples]
    gold_mat = torch.zeros(n_samples, dtype=torch.long)
    gold_mat[all_sample_id] = all_gold

    # Lang per sample (first occurrence)
    lang_of: List[str] = [""] * n_samples
    for sid, lg in zip(all_sample_id.tolist(), all_lang):
        if lang_of[sid] == "":
            lang_of[sid] = lg

    pred    = nll_mat.argmin(-1)          # [n_samples]
    correct = (pred == gold_mat)          # [n_samples] bool

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
                "nll":        {
                    LABEL_NAMES[c]: round(nll_mat[sid, c].item(), 6)
                    for c in range(C)
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Per-language accuracy ─────────────────────────────────────────────
    per_lang: Dict[str, List[bool]] = defaultdict(list)
    for sid in range(n_samples):
        per_lang[lang_of[sid]].append(bool(correct[sid].item()))

    per_lang_summary: Dict[str, Dict] = {}

    print(f"\n{'═'*58}")
    print(f"  MMMLU Zero-shot Results — {MODEL_NAME}")
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
    print(f"{'═'*58}")
    print(f"  Random baseline: 25.00%  |  Evaluated: {overall_acc:.2f}%")
    print(f"{'═'*58}\n")

    # ── Label distribution of predictions (bias check) ───────────────────
    pred_dist = {
        LABEL_NAMES[c]: int((pred == c).sum().item())
        for c in range(C)
    }
    gold_dist = {
        LABEL_NAMES[c]: int((gold_mat == c).sum().item())
        for c in range(C)
    }

    # ── Build summary ─────────────────────────────────────────────────────
    summary = {
        "model":     MODEL_NAME,
        "task":      "MMMLU",
        "mode":      "zero-shot log-prob scoring",
        "n_samples": n_samples,
        "candidates": LABEL_NAMES,
        "random_baseline": 25.0,
        "overall": {
            "accuracy": round(overall_acc, 2),
        },
        "per_lang": per_lang_summary,
        "prediction_distribution": pred_dist,
        "gold_distribution":       gold_dist,
        "eval_config": {
            "max_length": max_length,
            "batch_size": batch_size,
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
        description="MMMLU zero-shot evaluation — Llama-3-8B vanilla baseline"
    )
    parser.add_argument(
        "--data_root", type=str, default="../raw_data/",
        help="Path đến thư mục raw_data/ (default: ../raw_data/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/mmmlu_vanilla/",
        help="Thư mục lưu kết quả (default: results/mmmlu_vanilla/)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Số sample gốc per batch — thực tế rows = batch_size × 4 (default: 8)"
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max sequence length in tokens (default: 512)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype (default: bf16)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

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
    print(f"  Random Baseline  = 25.00%")