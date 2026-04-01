"""
XSQuAD_evaluation.py
====================
Zero-shot evaluation của Llama-3-8B (vanilla, không fine-tune) trên toàn bộ
tập dữ liệu XSQuAD — tất cả ngôn ngữ.

Pipeline:
    1. Load XSQuADDataLoader  (generation mode, left-padded batches)
    2. model.generate()  →  decode  →  normalize
    3. Tính F1 / EM per-sample  →  aggregate per-lang + overall

Normalize strategy (SQuAD chuẩn):
    - Lowercase
    - Loại bỏ articles (a, an, the)
    - Loại bỏ punctuation
    - Collapse whitespace
    F1 tính ở mức token (unigram overlap).

Kết quả lưu vào:
    results/xsquad_vanilla/
        summary.json       — overall + per_lang metrics
        per_sample.jsonl   — từng sample: prediction, gold, em, f1

Usage:
    python XSQuAD_evaluation.py
    python XSQuAD_evaluation.py --batch_size 8 --max_new_tokens 64
    python XSQuAD_evaluation.py --data_root ../raw_data/ --output_dir results/xsquad_vanilla/
"""

import argparse
import json
import os
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
# Cau truc project:  Multilingual OT/
#                        dataloader/downstream_dataloader.py
#                        zero-shot/XSQuAD_evaluation.py
# parent.parent = project root (Multilingual OT/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import XSQuADDataLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "meta-llama/Meta-Llama-3-8B"
ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Normalization & Scoring  (SQuAD-style)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """
    Chuẩn hoá câu trả lời theo chuẩn SQuAD:
        1. Lowercase
        2. Bỏ punctuation
        3. Bỏ articles (a / an / the)
        4. Collapse whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _token_f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    """Unigram F1 giữa hai danh sách token."""
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counter: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1

    gold_counter: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1

    overlap = sum(
        min(pred_counter.get(t, 0), gold_counter.get(t, 0))
        for t in gold_counter
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, gold_answers: List[str]) -> Tuple[float, float]:
    """
    Tính (EM, F1) giữa prediction và danh sách gold answers.
    Lấy max trên tất cả gold answers (theo chuẩn SQuAD evaluation).

    Returns
    -------
    em : float  — 1.0 nếu exact match với bất kỳ gold nào, else 0.0
    f1 : float  — max F1 trên tất cả gold answers
    """
    norm_pred = normalize_answer(prediction)
    pred_toks = norm_pred.split()

    best_em = 0.0
    best_f1 = 0.0

    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        gold_toks = norm_gold.split()

        em = 1.0 if norm_pred == norm_gold else 0.0
        f1 = _token_f1(pred_toks, gold_toks)

        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)

    return best_em, best_f1


# ---------------------------------------------------------------------------
# Post-processing prediction
# ---------------------------------------------------------------------------

def extract_answer(generated_text: str) -> str:
    """
    Trích xuất phần answer từ text đã generate (phần mới sau prompt).

    Chiến lược:
        1. Lấy text sau "Answer:" cuối cùng nếu model echo lại.
        2. Lấy dòng đầu tiên non-empty.
        3. Truncate tại dấu câu kết thúc câu / newline ẩn.
    """
    # Lấy phần sau "Answer:" cuối cùng nếu model vô tình echo lại
    marker = "Answer:"
    if marker in generated_text:
        generated_text = generated_text.split(marker)[-1]

    # Lấy dòng đầu tiên có nội dung
    for line in generated_text.split("\n"):
        line = line.strip()
        if line:
            # Truncate tại dấu chấm / hỏi / chấm than kết thúc câu
            line = re.split(r"(?<=[.!?])\s", line)[0].strip()
            return line

    return generated_text.strip()


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dtype_str: str = "bf16"):
    """Load Llama-3-8B vanilla với dtype tuỳ chọn."""
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
    output_dir: str = "results/xsquad_vanilla/",
    batch_size: int = 4,
    max_length: int = 1024,
    max_new_tokens: int = 64,
    num_beams: int = 1,
) -> Dict:
    """
    Chạy toàn bộ evaluation pipeline trên XSQuAD.

    Returns
    -------
    summary : dict — overall + per_lang EM/F1
    """
    device = next(model.parameters()).device

    # ── DataLoader ────────────────────────────────────────────────────────
    loader = XSQuADDataLoader(
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    print(f"[Eval] Total batches: {len(loader)}  batch_size={batch_size}\n")

    # ── Output setup ──────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.jsonl"

    # ── Accumulators per language ─────────────────────────────────────────
    # per_lang[lang] = {"em": [...], "f1": [...]}
    per_lang: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"em": [], "f1": []}
    )
    global_em: List[float] = []
    global_f1: List[float] = []
    sample_idx = 0
    t0 = time.time()

    with open(per_sample_path, "w", encoding="utf-8") as fout:
        for batch in tqdm(loader, desc="XSQuAD zero-shot eval"):
            input_ids      = batch["input_ids"].to(device)       # [B, L]
            attention_mask = batch["attention_mask"].to(device)  # [B, L]
            langs          = batch["lang"]                        # List[str]
            gold_list      = batch["answers"]                     # List[List[str]]

            # ── Generate ──────────────────────────────────────────────────
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # output_ids: [B, L + new_tokens]

            # ── Decode only newly generated tokens ───────────────────────
            # attention_mask.sum(1) = real prompt length per sample (excluding left-pad)
            prompt_lens = attention_mask.sum(dim=1).tolist()

            for i in range(len(langs)):
                lang         = langs[i]
                gold_answers = gold_list[i]

                prompt_len = int(prompt_lens[i])
                new_ids    = output_ids[i][prompt_len:]
                raw_pred   = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                # Post-process
                prediction = extract_answer(raw_pred)

                # Score vs gold (max over all gold answers)
                em, f1 = score_prediction(prediction, gold_answers)

                # Accumulate
                per_lang[lang]["em"].append(em)
                per_lang[lang]["f1"].append(f1)
                global_em.append(em)
                global_f1.append(f1)

                # Ghi per-sample record
                record = {
                    "sample_id":  sample_idx,
                    "lang":       lang,
                    "prediction": prediction,
                    "gold":       gold_answers,
                    "em":         em,
                    "f1":         round(f1, 6),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                sample_idx += 1

    elapsed = time.time() - t0
    print(f"\n[Eval] Finished {sample_idx:,} samples in {elapsed:.1f}s "
          f"({sample_idx / max(elapsed, 1):.1f} samples/s)")

    # ── Per-language aggregation ──────────────────────────────────────────
    per_lang_summary: Dict[str, Dict] = {}

    print(f"\n{'═'*60}")
    print(f"  XSQuAD Zero-shot Results — {MODEL_NAME}")
    print(f"{'═'*60}")
    print(f"  {'Language':<14}  {'N':>6}  {'EM (%)':>9}  {'F1 (%)':>9}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*9}  {'─'*9}")

    for lang in sorted(per_lang.keys()):
        ems = per_lang[lang]["em"]
        f1s = per_lang[lang]["f1"]
        n   = len(ems)
        avg_em = 100.0 * sum(ems) / n
        avg_f1 = 100.0 * sum(f1s) / n
        per_lang_summary[lang] = {
            "n":  n,
            "em": round(avg_em, 2),
            "f1": round(avg_f1, 2),
        }
        print(f"  {lang:<14}  {n:>6,}  {avg_em:>9.2f}  {avg_f1:>9.2f}")

    n_total    = len(global_em)
    overall_em = 100.0 * sum(global_em) / n_total
    overall_f1 = 100.0 * sum(global_f1) / n_total

    print(f"  {'─'*14}  {'─'*6}  {'─'*9}  {'─'*9}")
    print(f"  {'OVERALL':<14}  {n_total:>6,}  {overall_em:>9.2f}  {overall_f1:>9.2f}")
    print(f"{'═'*60}\n")

    # ── Build summary dict ────────────────────────────────────────────────
    summary = {
        "model":   MODEL_NAME,
        "task":    "XSQuAD",
        "mode":    "zero-shot generation",
        "n_samples": n_total,
        "overall": {
            "em": round(overall_em, 2),
            "f1": round(overall_f1, 2),
        },
        "per_lang": per_lang_summary,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "num_beams":      num_beams,
            "do_sample":      False,
            "max_length":     max_length,
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
        description="XSQuAD zero-shot evaluation — Llama-3-8B vanilla baseline"
    )
    parser.add_argument(
        "--data_root", type=str, default="../raw_data/",
        help="Path đến thư mục raw_data/ (default: ../raw_data/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/xsquad_vanilla/",
        help="Thư mục lưu kết quả (default: results/xsquad_vanilla/)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Số sample per batch (default: 4)"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Max input sequence length in tokens (default: 1024)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=64,
        help="Max số token mới sinh ra per sample (default: 64)"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1,
        help="Beam search width — 1 = greedy decoding (default: 1)"
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

    # HuggingFace login nếu cần (đọc HF_TOKEN từ .env)
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[HF] Logged in with HF_TOKEN from .env\n")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(dtype_str=args.dtype)

    # Run full evaluation
    summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print("✓ Evaluation complete.")
    print(f"  Overall EM = {summary['overall']['em']:.2f}%")
    print(f"  Overall F1 = {summary['overall']['f1']:.2f}%")