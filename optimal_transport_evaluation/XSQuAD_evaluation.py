"""
XSQuAD_evaluation_OT.py
========================
Zero-shot evaluation của Llama-3-8B-OT (Stage 2 OT-aligned) trên XSQuAD — generation mode.

Khác biệt duy nhất so với XSQuAD_evaluation.py (Instruct baseline):
    - Load base model + LoRA adapter từ HuggingFace Hub (ducanhdinh/Llama3-8B-OT)
    - attn_implementation="eager" (bắt buộc theo kiến trúc OT training)
    - Merge LoRA vào base trước khi inference (model.merge_and_unload())

Toàn bộ logic evaluation, normalize, score, extract_answer, output format giữ nguyên
y hệt XSQuAD_evaluation.py — không thay đổi một dòng nào.

Usage:
    python XSQuAD_evaluation_OT.py
    python XSQuAD_evaluation_OT.py --hub_repo ducanhdinh/Llama3-8B-OT
    python XSQuAD_evaluation_OT.py --batch_size 4 --max_new_tokens 64
    python XSQuAD_evaluation_OT.py --data_root ../raw_data/ --output_dir results/xsquad_ot/
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
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import XSQuADDataLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "ducanhdinh/Llama3-8B-Finetune"
DEFAULT_HUB_REPO   = "ducanhdinh/Llama3-8B-OT"
LORA_ADAPTER_DIR   = "lora_adapter"
ARTICLES_RE        = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Normalization & Scoring  (SQuAD-style)
# ── GIỮ NGUYÊN 100% từ XSQuAD_evaluation.py ──
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
# Post-processing prediction (Instruct-aware)
# ── GIỮ NGUYÊN 100% từ XSQuAD_evaluation.py ──
# ---------------------------------------------------------------------------

def extract_answer(generated_text: str) -> str:
    """
    Trích xuất câu trả lời từ phần mới sinh (new tokens only — không có prompt echo).

    Với Instruct model, output thường là:
        - Đúng span ngắn: "Albert Einstein"
        - Câu ngắn: "The answer is Albert Einstein."
        - Đôi khi kèm giải thích: "Albert Einstein, a physicist born in Germany."

    Chiến lược:
        1. Lấy dòng đầu tiên non-empty (Instruct thường output đúng span ở dòng 1).
        2. Nếu có prefix "The answer is:" / "Answer:" → bỏ prefix đó.
        3. KHÔNG truncate tại dấu câu nội tuyến — tránh cắt nhầm entity names
           như "St. Mary's", "3.5 million", "U.S.A.", v.v.

    Note: Không cần tìm "Answer:" echo từ prompt vì chúng ta chỉ decode new tokens.
    """
    text = generated_text.strip()

    # Lấy dòng đầu tiên non-empty
    first_line = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return text  # fallback: toàn bộ output

    # Bỏ các prefix phổ biến của instruct model
    for prefix in ("The answer is:", "The answer is", "Answer:", "Answer"):
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip(" .,")
            break

    return first_line if first_line else text


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ── THAY ĐỔI DUY NHẤT so với XSQuAD_evaluation.py ──
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str = DEFAULT_BASE_MODEL,
    hub_repo:   str = DEFAULT_HUB_REPO,
    dtype_str:  str = "bf16",
):
    """
    Load OT-aligned model đúng kiến trúc:
      1. Base model với attn_implementation='eager'
         (bắt buộc: OT training dùng eager để output_attentions hoạt động)
      2. Wrap PeftModel với LoRA adapter từ Hub subfolder 'lora_adapter'
      3. merge_and_unload() để inference không cần adapter overhead
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"[Model] Loading base model: {base_model}  dtype={dtype_str}")
    print(f"[Model] LoRA adapter from:  {hub_repo}/{LORA_ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Step 1: Load base model
    # attn_implementation="eager" bắt buộc — giống build_lora_model() trong OT training
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Step 2: Gắn LoRA adapter từ Hub
    model = PeftModel.from_pretrained(
        base,
        hub_repo,
        subfolder=LORA_ADAPTER_DIR,
        is_trainable=False,
    )

    # Step 3: Merge LoRA vào base weights → inference thuần, không overhead adapter
    print("[Model] Merging LoRA adapter into base model ...")
    model = model.merge_and_unload()

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded & merged. Parameters: {n_params / 1e9:.2f}B\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation loop  ── GIỮ NGUYÊN 100% từ XSQuAD_evaluation.py ──
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    data_root: str  = "../raw_data/",
    output_dir: str = "results/xsquad_ot/",
    batch_size: int = 4,
    max_length: int = 1024,
    max_new_tokens: int = 64,
    num_beams: int = 1,
) -> Dict:
    """
    Chạy toàn bộ evaluation pipeline trên XSQuAD với OT-aligned backbone.

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

    # ── Accumulators ──────────────────────────────────────────────────────
    per_lang: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"em": [], "f1": []}
    )
    global_em: List[float] = []
    global_f1: List[float] = []
    sample_idx = 0
    t0 = time.time()

    with open(per_sample_path, "w", encoding="utf-8") as fout:
        for batch in tqdm(loader, desc="XSQuAD zero-shot eval (OT-aligned)"):
            input_ids      = batch["input_ids"].to(device)       # [B, L]
            attention_mask = batch["attention_mask"].to(device)  # [B, L]
            langs          = batch["lang"]                        # List[str]
            gold_list      = batch["answers"]                     # List[List[str]]

            # Ghi nhớ input_len để slice đúng phần new tokens
            # Left-padding: tất cả sequences trong batch có cùng độ dài L
            # → output_ids[i][input_len:] là phần model sinh ra
            input_len = input_ids.shape[1]

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
            # output_ids: [B, input_len + new_tokens]

            # ── Decode only newly generated tokens ───────────────────────
            for i in range(len(langs)):
                lang         = langs[i]
                gold_answers = gold_list[i]

                # New tokens only — không có prompt echo (instruct template)
                new_ids  = output_ids[i][input_len:]
                raw_pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

                # Post-process: lấy dòng đầu, bỏ prefix nếu có
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
                    "raw_output": raw_pred,
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

    print(f"\n{'═'*62}")
    print(f"  XSQuAD Zero-shot Results — OT-aligned Llama3-8B")
    print(f"{'═'*62}")
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
    print(f"{'═'*62}\n")

    # ── Build summary dict ────────────────────────────────────────────────
    summary = {
        "model":      f"{DEFAULT_HUB_REPO}/{LORA_ADAPTER_DIR} (merged)",
        "base_model": DEFAULT_BASE_MODEL,
        "task":       "XSQuAD",
        "mode":       "zero-shot generation (OT-aligned)",
        "n_samples":  n_total,
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
        description="XSQuAD zero-shot evaluation — Llama-3-8B OT-aligned"
    )
    parser.add_argument(
        "--base_model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base model HF repo (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--hub_repo", type=str, default=DEFAULT_HUB_REPO,
        help=f"HF repo chứa LoRA adapter (subfolder: {LORA_ADAPTER_DIR}) (default: {DEFAULT_HUB_REPO})"
    )
    parser.add_argument(
        "--data_root", type=str, default="../raw_data/",
        help="Path đến thư mục raw_data/ (default: ../raw_data/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/xsquad_ot/",
        help="Thư mục lưu kết quả (default: results/xsquad_ot/)"
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

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[HF] Logged in with HF_TOKEN from .env\n")

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        hub_repo=args.hub_repo,
        dtype_str=args.dtype,
    )

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