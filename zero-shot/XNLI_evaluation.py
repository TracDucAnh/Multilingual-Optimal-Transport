"""
XNLI_evaluation.py
==================
Zero-shot evaluation của Llama-3-8B (vanilla) trên XNLI — generation mode.

Scoring Strategy — Generation + Exact Match:
    model.generate() → decode → parse first valid keyword →
    compare với gold label (entailment / neutral / contradiction)

Parse strategy:
    Lấy dòng đầu tiên của output, lowercase, tìm keyword đầu tiên
    xuất hiện trong {"entailment", "neutral", "contradiction"}.
    Nếu không tìm được → "unknown" → incorrect.

Usage:
    python XNLI_evaluation.py
    python XNLI_evaluation.py --batch_size 16 --max_new_tokens 16
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
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import XNLIDataLoader, XNLI_CANDIDATES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME   = "meta-llama/Meta-Llama-3-8B"
VALID_LABELS = set(XNLI_CANDIDATES)  # {"entailment", "neutral", "contradiction"}


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(dtype_str: str = "bf16"):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"[Model] Loading {MODEL_NAME}  dtype={dtype_str} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Loaded. Parameters: {n_params / 1e9:.2f}B\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_xnli_output(raw_text: str) -> str:
    """
    Parse output của model thành một trong 3 nhãn XNLI.

    Chiến lược:
        1. Lấy dòng đầu tiên non-empty.
        2. Lowercase.
        3. Tìm vị trí xuất hiện đầu tiên của mỗi keyword trong string.
        4. Trả về keyword xuất hiện sớm nhất.
        5. Nếu không tìm được → "unknown".

    Returns
    -------
    str : "entailment" | "neutral" | "contradiction" | "unknown"
    """
    first_line = ""
    for line in raw_text.split("\n"):
        line = line.strip()
        if line:
            first_line = line.lower()
            break

    if not first_line:
        return "unknown"

    positions = {}
    for label in XNLI_CANDIDATES:
        pos = first_line.find(label)
        if pos != -1:
            positions[label] = pos

    if not positions:
        return "unknown"

    return min(positions, key=positions.get)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    data_root: str      = "../raw_data/",
    output_dir: str     = "results/xnli_vanilla/",
    batch_size: int     = 16,
    max_length: int     = 256,
    max_new_tokens: int = 16,
    num_beams: int      = 1,
) -> Dict:
    """
    Generation-mode evaluation trên XNLI.
    max_new_tokens=16 là đủ — model chỉ cần output 1 từ.
    """
    device = next(model.parameters()).device

    loader = XNLIDataLoader(
        data_root=data_root,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )
    print(f"[Eval] Total batches: {len(loader)}  batch_size={batch_size}\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "per_sample.jsonl"

    per_lang: Dict[str, List[bool]] = defaultdict(list)
    pred_dist: Dict[str, int] = {l: 0 for l in XNLI_CANDIDATES}
    pred_dist["unknown"] = 0
    unknown_count = 0
    sample_idx    = 0
    t0            = time.time()

    with open(per_sample_path, "w", encoding="utf-8") as fout:
        for batch in tqdm(loader, desc="XNLI generation eval"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_labels    = batch["gold_label"]
            langs          = batch["lang"]
            input_len      = input_ids.shape[1]

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

            for i in range(len(langs)):
                lang       = langs[i]
                gold_label = gold_labels[i]

                new_ids    = output_ids[i][input_len:]
                raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                pred_label = parse_xnli_output(raw_output)
                is_correct = (pred_label == gold_label)

                if pred_label == "unknown":
                    unknown_count += 1
                pred_dist[pred_label] = pred_dist.get(pred_label, 0) + 1
                per_lang[lang].append(is_correct)

                fout.write(json.dumps({
                    "sample_id":  sample_idx,
                    "lang":       lang,
                    "raw_output": raw_output,
                    "pred_label": pred_label,
                    "gold_label": gold_label,
                    "correct":    is_correct,
                }, ensure_ascii=False) + "\n")
                sample_idx += 1

    elapsed   = time.time() - t0
    n_samples = sample_idx

    print(f"\n[Eval] Finished {n_samples:,} samples in {elapsed:.1f}s "
          f"({n_samples / max(elapsed, 1):.1f} samples/s)")
    print(f"[Eval] Unknown/unparseable: {unknown_count:,} "
          f"({100*unknown_count/max(n_samples,1):.1f}%)\n")

    # ── Per-language accuracy ─────────────────────────────────────────────
    per_lang_summary: Dict[str, Dict] = {}

    print(f"\n{'═'*58}")
    print(f"  XNLI Zero-shot Results (Generation) — {MODEL_NAME}")
    print(f"{'═'*58}")
    print(f"  {'Language':<14}  {'N':>6}  {'Accuracy (%)':>13}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*13}")

    for lang in sorted(per_lang.keys()):
        vals = per_lang[lang]
        n    = len(vals)
        acc  = 100.0 * sum(vals) / n
        per_lang_summary[lang] = {"n": n, "accuracy": round(acc, 2)}
        print(f"  {lang:<14}  {n:>6,}  {acc:>13.2f}")

    overall_correct = sum(sum(v) for v in per_lang.values())
    overall_acc     = 100.0 * overall_correct / n_samples

    print(f"  {'─'*14}  {'─'*6}  {'─'*13}")
    print(f"  {'OVERALL':<14}  {n_samples:>6,}  {overall_acc:>13.2f}")
    print(f"{'═'*58}")
    print(f"  Random baseline: 33.33%")
    print(f"{'═'*58}\n")

    print(f"  Prediction distribution (bias check):")
    print(f"  {'Label':<15}  {'Count':>8}  {'%':>7}")
    for lbl in XNLI_CANDIDATES + ["unknown"]:
        cnt = pred_dist.get(lbl, 0)
        print(f"  {lbl:<15}  {cnt:>8,}  {100*cnt/max(n_samples,1):>6.1f}%")
    print()

    summary = {
        "model":     MODEL_NAME,
        "task":      "XNLI",
        "mode":      "zero-shot generation",
        "n_samples": n_samples,
        "valid_labels": XNLI_CANDIDATES,
        "overall":   {"accuracy": round(overall_acc, 2)},
        "per_lang":  per_lang_summary,
        "prediction_distribution": pred_dist,
        "unknown_count": unknown_count,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "num_beams":      num_beams,
            "do_sample":      False,
            "max_length":     max_length,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Output] summary    → {summary_path}")
    print(f"[Output] per_sample → {per_sample_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XNLI zero-shot generation evaluation — Llama-3-8B"
    )
    parser.add_argument("--data_root",      type=str, default="../raw_data/")
    parser.add_argument("--output_dir",     type=str, default="results/xnli_vanilla/")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_length",     type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="16 là đủ cho 1 label word")
    parser.add_argument("--num_beams",      type=int, default=1)
    parser.add_argument("--dtype",          type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[HF] Logged in with HF_TOKEN from .env\n")

    model, tokenizer = load_model_and_tokenizer(dtype_str=args.dtype)

    summary = evaluate(
        model=model, tokenizer=tokenizer,
        data_root=args.data_root, output_dir=args.output_dir,
        batch_size=args.batch_size, max_length=args.max_length,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
    )

    print("✓ Evaluation complete.")
    print(f"  Overall Accuracy = {summary['overall']['accuracy']:.2f}%")
    print(f"  Random Baseline  = 33.33%")