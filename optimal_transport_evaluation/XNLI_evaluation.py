"""
XNLI_evaluation_OT.py
=====================
Zero-shot evaluation của Llama-3-8B-OT (Stage 2 OT-aligned) trên XNLI.

Khác biệt duy nhất so với XNLI_evaluation.py (Instruct baseline):
    - Load base model + LoRA adapter từ HuggingFace Hub (ducanhdinh/Llama3-8B-OT)
    - attn_implementation="eager" (bắt buộc theo kiến trúc OT training)
    - Merge LoRA vào base trước khi inference (model.merge_and_unload())

Toàn bộ logic evaluation, parse, generate, output format giữ nguyên
y hệt XNLI_evaluation.py — không thay đổi một dòng nào.

Usage:
    python XNLI_evaluation_OT.py
    python XNLI_evaluation_OT.py --hub_repo ducanhdinh/Llama3-8B-OT
    python XNLI_evaluation_OT.py --batch_size 16 --max_new_tokens 24
    python XNLI_evaluation_OT.py --data_root ../raw_data/ --output_dir results/xnli_ot/
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
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataloader"))
from downstream_dataloader import XNLIDataLoader, XNLI_CANDIDATES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "ducanhdinh/Llama3-8B-Finetune"
DEFAULT_HUB_REPO   = "ducanhdinh/Llama3-8B-OT"
LORA_ADAPTER_DIR   = "lora_adapter"
VALID_LABELS       = set(XNLI_CANDIDATES)  # {"entailment", "neutral", "contradiction"}


# ---------------------------------------------------------------------------
# Model & Tokenizer
# ── THAY ĐỔI DUY NHẤT so với XNLI_evaluation.py ──
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model:  str = DEFAULT_BASE_MODEL,
    hub_repo:    str = DEFAULT_HUB_REPO,
    dtype_str:   str = "bf16",
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
# Output parser  ── GIỮ NGUYÊN 100% từ XNLI_evaluation.py ──
# ---------------------------------------------------------------------------

def parse_xnli_output(raw_text: str) -> str:
    """
    Parse output của model thành một trong 3 nhãn XNLI.

    Chiến lược:
        1. Lấy dòng đầu tiên non-empty.
        2. Lowercase.
        3. Tìm vị trí xuất hiện đầu tiên của mỗi keyword.
        4. Trả về keyword xuất hiện sớm nhất trong string.
        5. Nếu không tìm được → "unknown".

    Returns
    -------
    str : "entailment" | "neutral" | "contradiction" | "unknown"
    """
    first_line = ""
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped.lower()
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
# Evaluation loop  ── GIỮ NGUYÊN 100% từ XNLI_evaluation.py ──
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    data_root:      str = "../raw_data/",
    output_dir:     str = "results/xnli_ot/",
    batch_size:     int = 16,
    max_length:     int = 512,
    max_new_tokens: int = 24,
    num_beams:      int = 1,
) -> Dict:
    """
    Generation-mode evaluation trên XNLI.
    Logic giữ nguyên y hệt XNLI_evaluation.py.
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
        for batch in tqdm(loader, desc="XNLI generation eval (OT-aligned)"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_labels    = batch["gold_label"]
            langs          = batch["lang"]

            input_len = input_ids.shape[1]

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

    per_lang_summary: Dict[str, Dict] = {}

    print(f"\n{'═'*60}")
    print(f"  XNLI Zero-shot Results (Generation) — OT-aligned Llama3-8B")
    print(f"{'═'*60}")
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
    print(f"{'═'*60}")
    print(f"  Random baseline: 33.33%")
    print(f"{'═'*60}\n")

    print(f"  Prediction distribution (bias check):")
    print(f"  {'Label':<15}  {'Count':>8}  {'%':>7}")
    for lbl in XNLI_CANDIDATES + ["unknown"]:
        cnt = pred_dist.get(lbl, 0)
        print(f"  {lbl:<15}  {cnt:>8,}  {100*cnt/max(n_samples,1):>6.1f}%")
    print()

    summary = {
        "model":           f"{DEFAULT_HUB_REPO}/{LORA_ADAPTER_DIR} (merged)",
        "base_model":      DEFAULT_BASE_MODEL,
        "task":            "XNLI",
        "mode":            "zero-shot generation (OT-aligned)",
        "n_samples":       n_samples,
        "valid_labels":    XNLI_CANDIDATES,
        "random_baseline": 33.33,
        "overall":         {"accuracy": round(overall_acc, 2)},
        "per_lang":        per_lang_summary,
        "prediction_distribution": pred_dist,
        "unknown_count":   unknown_count,
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
        description="XNLI zero-shot generation evaluation — Llama-3-8B OT-aligned"
    )
    parser.add_argument("--base_model",     type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--hub_repo",       type=str, default=DEFAULT_HUB_REPO,
                        help="HF repo chứa LoRA adapter (subfolder: lora_adapter)")
    parser.add_argument("--data_root",      type=str, default="../raw_data/")
    parser.add_argument("--output_dir",     type=str, default="results/xnli_ot/")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_length",     type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=24)
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

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        hub_repo=args.hub_repo,
        dtype_str=args.dtype,
    )

    summary = evaluate(
        model=model, tokenizer=tokenizer,
        data_root=args.data_root, output_dir=args.output_dir,
        batch_size=args.batch_size, max_length=args.max_length,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
    )

    print("✓ Evaluation complete.")
    print(f"  Overall Accuracy = {summary['overall']['accuracy']:.2f}%")
    print(f"  Random Baseline  = 33.33%")