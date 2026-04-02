"""
Llama3-8B-Finetuning.py
========================
Full-model supervised fine-tuning của meta-llama/Meta-Llama-3-8B-Instruct
trên ba English benchmark datasets (MMLU, SQuAD, SNLI) với:

  - Generate mode + L_LM loss (CrossEntropy trên answer tokens)
  - 3 epochs, batch_size=64 (với gradient accumulation)
  - Per-iteration logging ra file .jsonl
  - Sau mỗi epoch: push lên HuggingFace Hub, chạy evaluation
  - Đồ thị acc / F1 / EM theo epoch (matplotlib)
  - Report file JSON per-epoch: metrics từng task

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_LM Loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_LM = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)

Trong SFT batch:
  - labels = -100 trên toàn bộ prompt (system + user turns)
  - labels = answer token ids + EOS trên phần trả lời của assistant
  → Model chỉ học mapping prompt → answer đúng
  → Gradient KHÔNG chảy ngược qua prompt tokens

Shift để align logits với targets:
  shift_logits = logits[:, :-1, :]   # dự đoán token t+1 từ token t
  shift_labels = labels[:, 1:]       # target là token thực tế tại t+1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluation — Generate Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- MMLU : model.generate(prompt) → parse letter → Accuracy
- SNLI : model.generate(prompt) → parse label word → Accuracy
- SQuAD: model.generate(prompt) → decode new tokens → F1 + EM vs all gold answers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python Llama3-8B-Finetuning.py \
        --data_root   ../raw_data/english/ \
        --output_dir  ./checkpoints \
        --hub_repo    Llama3-8B-Finetune \
        --epochs      3 \
        --batch_size  64 \
        --micro_batch 4 \
        --lr          2e-5 \
        --max_new_tokens 32
"""

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, login

# Import dataloaders từ cùng thư mục
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dataloader"))

from finetune_dataloader import (
    MixedFinetuneDataLoader,
    MMLUValDataLoader,
    SNLIValDataLoader,
    SQuADValDataLoader,
    parse_mmlu_output,
    parse_snli_output,
    compute_exact_match,
    compute_f1_score,
    MMLU_OPTIONS,
    SNLI_CANDIDATES,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-3-8B Full Finetune")

    # Data
    parser.add_argument("--data_root",   type=str, default="../raw_data/english/")
    parser.add_argument("--output_dir",  type=str, default="./checkpoints")

    # HuggingFace Hub
    parser.add_argument("--hub_repo",    type=str, default="Llama3-8B-Finetune",
                        help="HF repo name (tổ chức/user/repo hoặc chỉ repo)")
    parser.add_argument("--hub_private", action="store_true",
                        help="Tạo private repo")

    # Training
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Effective batch size (dùng gradient accumulation)")
    parser.add_argument("--micro_batch", type=int,   default=4,
                        help="Micro batch size per forward pass")
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed",        type=int,   default=42)

    # Model
    parser.add_argument("--model_name",  type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mmlu_max_len",  type=int, default=512)
    parser.add_argument("--squad_max_len", type=int, default=1024)
    parser.add_argument("--snli_max_len",  type=int, default=256)

    # Evaluation
    parser.add_argument("--eval_batch",  type=int,   default=8,
                        help="Batch size cho evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max new tokens cho model.generate()")
    parser.add_argument("--skip_eval",   action="store_true",
                        help="Bỏ qua evaluation (chỉ train)")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16",        action="store_true", default=True,
                        help="Dùng bfloat16 (A100/H100); nếu không có thì dùng fp16")
    parser.add_argument("--fp16",        action="store_true",
                        help="Dùng float16 mixed precision")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# L_LM loss computation
# ---------------------------------------------------------------------------

def compute_lm_loss(
    logits: torch.Tensor,   # [B, L, V]
    labels: torch.Tensor,   # [B, L]   -100 on prompt tokens
) -> torch.Tensor:
    """
    L_LM = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)

    Shift:
        shift_logits[b, t] = logits[b, t]     → dự đoán token t+1
        shift_labels[b, t] = labels[b, t+1]   → target tại vị trí t+1

    Chỉ tính loss trên các vị trí có labels != -100 (answer + EOS tokens).
    Trả về scalar loss (mean qua tất cả non-masked tokens trong batch).
    """
    # Shift logits và labels để align
    shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, L-1]

    vocab_size = shift_logits.size(-1)
    loss_fct   = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    loss = loss_fct(
        shift_logits.view(-1, vocab_size),  # [(B*(L-1)), V]
        shift_labels.view(-1),              # [(B*(L-1))]
    )
    return loss


# ---------------------------------------------------------------------------
# HuggingFace Hub push
# ---------------------------------------------------------------------------

def push_to_hub(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    epoch: int,
    private: bool = False,
) -> None:
    """
    Push model + tokenizer lên HuggingFace Hub.
    Tạo repo mới nếu chưa tồn tại.
    """
    logger.info(f"[Hub] Pushing epoch {epoch} → {repo_id}")
    api = HfApi()

    # Tạo repo nếu chưa có
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"[Hub] Repo '{repo_id}' đã tồn tại.")
    except Exception:
        logger.info(f"[Hub] Tạo repo mới: '{repo_id}' (private={private})")
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    commit_msg = f"Epoch {epoch} checkpoint"
    try:
        model.push_to_hub(repo_id, commit_message=commit_msg, private=private)
        tokenizer.push_to_hub(repo_id, commit_message=commit_msg, private=private)
        logger.info(f"[Hub] ✓ Push thành công: {repo_id} (epoch={epoch})")
    except Exception as e:
        logger.error(f"[Hub] Push thất bại: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mmlu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: MMLUValDataLoader,
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> Dict:
    """
    Đánh giá MMLU bằng generate mode.
    Parse letter đầu tiên trong {A,B,C,D} từ output.
    Trả về: {"accuracy": float, "n_correct": int, "n_total": int, "unknown_rate": float}
    """
    model.eval()
    n_correct, n_total, n_unknown = 0, 0, 0

    pbar = tqdm(val_loader, desc="  [Eval] MMLU", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_labels    = batch["gold_label"]
        prompt_len     = input_ids.size(1)

        with autocast(dtype=dtype):
            gen_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode chỉ phần new tokens
        new_ids = gen_ids[:, prompt_len:]
        for i, (gen, gold) in enumerate(zip(new_ids, gold_labels)):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_mmlu_output(decoded)
            if pred == "":
                n_unknown += 1
            if pred == gold:
                n_correct += 1
            n_total += 1

        # Cập nhật postfix tqdm với acc hiện tại
        acc_so_far = n_correct / n_total if n_total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.4f}", correct=n_correct, total=n_total)

    pbar.close()

    accuracy      = n_correct / n_total if n_total > 0 else 0.0
    unknown_rate  = n_unknown / n_total if n_total > 0 else 0.0
    return {
        "accuracy":     accuracy,
        "n_correct":    n_correct,
        "n_total":      n_total,
        "unknown_rate": unknown_rate,
    }


@torch.no_grad()
def evaluate_snli(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: SNLIValDataLoader,
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> Dict:
    """
    Đánh giá SNLI bằng generate mode.
    Parse label word: entailment / neutral / contradiction.
    """
    model.eval()
    n_correct, n_total, n_unknown = 0, 0, 0

    pbar = tqdm(val_loader, desc="  [Eval] SNLI", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_labels    = batch["gold_label"]
        prompt_len     = input_ids.size(1)

        with autocast(dtype=dtype):
            gen_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = gen_ids[:, prompt_len:]
        for gen, gold in zip(new_ids, gold_labels):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_snli_output(decoded)
            if pred == "":
                n_unknown += 1
            if pred == gold:
                n_correct += 1
            n_total += 1

        # Cập nhật postfix tqdm với acc hiện tại
        acc_so_far = n_correct / n_total if n_total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.4f}", correct=n_correct, total=n_total)

    pbar.close()

    accuracy     = n_correct / n_total if n_total > 0 else 0.0
    unknown_rate = n_unknown / n_total if n_total > 0 else 0.0
    return {
        "accuracy":     accuracy,
        "n_correct":    n_correct,
        "n_total":      n_total,
        "unknown_rate": unknown_rate,
    }


@torch.no_grad()
def evaluate_squad(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_loader: SQuADValDataLoader,
    device: torch.device,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> Dict:
    """
    Đánh giá SQuAD bằng generate mode.
    Decode new tokens → F1 + EM so với ALL gold answers.
    """
    model.eval()
    total_f1, total_em, n_total = 0.0, 0.0, 0

    pbar = tqdm(val_loader, desc="  [Eval] SQuAD", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        all_answers    = batch["answers"]   # List[List[str]]
        prompt_len     = input_ids.size(1)

        with autocast(dtype=dtype):
            gen_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = gen_ids[:, prompt_len:]
        for gen, gold_list in zip(new_ids, all_answers):
            pred     = tokenizer.decode(gen, skip_special_tokens=True).strip()
            total_em += compute_exact_match(pred, gold_list)
            total_f1 += compute_f1_score(pred, gold_list)
            n_total  += 1

        # Cập nhật postfix tqdm với F1/EM hiện tại
        f1_so_far = total_f1 / n_total if n_total > 0 else 0.0
        em_so_far = total_em / n_total if n_total > 0 else 0.0
        pbar.set_postfix(F1=f"{f1_so_far:.4f}", EM=f"{em_so_far:.4f}", n=n_total)

    pbar.close()

    f1 = total_f1 / n_total if n_total > 0 else 0.0
    em = total_em / n_total if n_total > 0 else 0.0
    return {"f1": f1, "em": em, "n_total": n_total}


def run_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    epoch: int,
    output_dir: Path,
) -> Dict:
    """
    Chạy evaluation trên tất cả 3 tasks.
    Ghi report ra file JSON.
    Trả về dict chứa tất cả metrics.
    """
    logger.info(f"[Eval] Bắt đầu evaluation sau epoch {epoch}...")

    # Tokenizer cho val phải là left-padded
    tokenizer.padding_side = "left"

    # ── MMLU val ────────────────────────────────────────────────────────────
    logger.info("[Eval] MMLU...")
    mmlu_val = MMLUValDataLoader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        batch_size=args.eval_batch,
        max_length=args.mmlu_max_len,
        shuffle=False,
        num_workers=0,
    )
    mmlu_res = evaluate_mmlu(model, tokenizer, mmlu_val, device, args.max_new_tokens, dtype)
    logger.info(
        f"[Eval] MMLU  acc={mmlu_res['accuracy']:.4f}  "
        f"({mmlu_res['n_correct']}/{mmlu_res['n_total']})  "
        f"unknown_rate={mmlu_res['unknown_rate']:.3f}"
    )

    # ── SNLI val ─────────────────────────────────────────────────────────────
    logger.info("[Eval] SNLI...")
    snli_val = SNLIValDataLoader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        batch_size=args.eval_batch,
        max_length=args.snli_max_len,
        shuffle=False,
        num_workers=0,
    )
    snli_res = evaluate_snli(model, tokenizer, snli_val, device, args.max_new_tokens, dtype)
    logger.info(
        f"[Eval] SNLI  acc={snli_res['accuracy']:.4f}  "
        f"({snli_res['n_correct']}/{snli_res['n_total']})  "
        f"unknown_rate={snli_res['unknown_rate']:.3f}"
    )

    # ── SQuAD val ─────────────────────────────────────────────────────────────
    logger.info("[Eval] SQuAD...")
    squad_val = SQuADValDataLoader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        batch_size=args.eval_batch,
        max_length=args.squad_max_len,
        shuffle=False,
        num_workers=0,
    )
    squad_res = evaluate_squad(model, tokenizer, squad_val, device, args.max_new_tokens, dtype)
    logger.info(
        f"[Eval] SQuAD F1={squad_res['f1']:.4f}  EM={squad_res['em']:.4f}  "
        f"n={squad_res['n_total']}"
    )

    # Reset padding side về right cho training
    tokenizer.padding_side = "right"

    # Tổng hợp results
    results = {
        "epoch": epoch,
        "mmlu":  mmlu_res,
        "snli":  snli_res,
        "squad": squad_res,
    }

    # Ghi report JSON
    report_path = output_dir / f"report_epoch{epoch}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"[Eval] Report saved → {report_path}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(
    epoch_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Vẽ đồ thị metrics theo epoch:
      - MMLU accuracy
      - SNLI accuracy
      - SQuAD F1 + EM
    Lưu vào output_dir/metrics_plot.png
    """
    epochs = [r["epoch"] for r in epoch_results]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evaluation Metrics per Epoch", fontsize=14, fontweight="bold")

    # MMLU accuracy
    ax = axes[0]
    mmlu_acc = [r["mmlu"]["accuracy"] for r in epoch_results]
    ax.plot(epochs, mmlu_acc, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_title("MMLU Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for e, v in zip(epochs, mmlu_acc):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # SNLI accuracy
    ax = axes[1]
    snli_acc = [r["snli"]["accuracy"] for r in epoch_results]
    ax.plot(epochs, snli_acc, "o-", color="darkorange", linewidth=2, markersize=8)
    ax.set_title("SNLI Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    for e, v in zip(epochs, snli_acc):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # SQuAD F1 + EM
    ax = axes[2]
    squad_f1 = [r["squad"]["f1"] for r in epoch_results]
    squad_em = [r["squad"]["em"] for r in epoch_results]
    ax.plot(epochs, squad_f1, "o-", color="forestgreen",  linewidth=2,
            markersize=8, label="F1")
    ax.plot(epochs, squad_em, "s--", color="crimson", linewidth=2,
            markersize=8, label="EM")
    ax.set_title("SQuAD F1 / EM")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for e, f1, em in zip(epochs, squad_f1, squad_em):
        ax.annotate(f"F1={f1:.3f}", (e, f1), textcoords="offset points",
                    xytext=(-15, 8), ha="center", fontsize=8, color="forestgreen")
        ax.annotate(f"EM={em:.3f}", (e, em), textcoords="offset points",
                    xytext=(15, -15), ha="center", fontsize=8, color="crimson")

    plt.tight_layout()
    plot_path = output_dir / "metrics_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Plot] Saved → {plot_path}")


def plot_training_loss(
    loss_log: List[Dict],
    output_dir: Path,
) -> None:
    """
    Vẽ đồ thị training loss theo iteration.
    Lưu vào output_dir/training_loss.png
    """
    if not loss_log:
        return
    steps  = [e["step"]       for e in loss_log]
    losses = [e["loss"]       for e in loss_log]
    lrs    = [e["lr"]         for e in loss_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(steps, losses, color="steelblue", linewidth=0.8, alpha=0.7)
    # Smooth loss (moving average window=50)
    if len(losses) >= 50:
        window = 50
        smooth = [sum(losses[max(0,i-window):i+1]) / min(i+1, window)
                  for i in range(len(losses))]
        ax1.plot(steps, smooth, color="red", linewidth=1.5, label="MA-50")
        ax1.legend()
    ax1.set_ylabel("L_LM Loss")
    ax1.set_xlabel("Global Step")
    ax1.grid(True, alpha=0.3)

    # LR
    ax2.plot(steps, lrs, color="darkorange", linewidth=1.0)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Global Step")
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Plot] Training loss saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device & dtype ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
        use_amp = True
        logger.info("[Setup] Using bfloat16 mixed precision")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16
        amp_dtype = torch.float16
        use_amp = True
        logger.info("[Setup] Using float16 mixed precision")
    else:
        dtype = torch.float32
        amp_dtype = torch.float32
        use_amp = False
        logger.info("[Setup] Using float32 (no mixed precision)")

    # ── Gradient accumulation ─────────────────────────────────────────────────
    accum_steps = max(1, args.batch_size // args.micro_batch)
    logger.info(
        f"[Setup] effective_batch={args.batch_size}  "
        f"micro_batch={args.micro_batch}  accum_steps={accum_steps}"
    )

    # ── HF Hub login ──────────────────────────────────────────────────────────
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("[Hub] Logged in with HF_TOKEN")
    else:
        logger.warning("[Hub] HF_TOKEN không được set — push hub có thể thất bại")

    # Xác định full repo_id
    hub_repo = args.hub_repo
    if "/" not in hub_repo:
        # Lấy username từ API
        try:
            api      = HfApi()
            whoami   = api.whoami()
            username = "ducanhdinh"
            hub_repo = f"{username}/{hub_repo}"
            logger.info(f"[Hub] Full repo_id: {hub_repo}")
        except Exception as e:
            logger.warning(f"[Hub] Không lấy được username: {e}. Dùng repo_id = '{hub_repo}'")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"[Setup] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # SFT dùng right-padding

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"[Setup] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to(device)

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Setup] Trainable params: {n_params/1e9:.2f}B")

    # ── DataLoader ────────────────────────────────────────────────────────────
    logger.info("[Setup] Building MixedFinetuneDataLoader...")
    train_loader = MixedFinetuneDataLoader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        batch_size=args.micro_batch,
        mmlu_max_length=args.mmlu_max_len,
        squad_max_length=args.squad_max_len,
        snli_max_length=args.snli_max_len,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # Tách weight decay: không apply cho bias và LayerNorm
    no_decay     = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95))

    # Số optimizer steps thực tế
    steps_per_epoch  = math.ceil(len(train_loader) / accum_steps)
    total_opt_steps  = steps_per_epoch * args.epochs
    warmup_steps     = max(1, int(total_opt_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(
        f"[Setup] total_opt_steps={total_opt_steps}  "
        f"warmup_steps={warmup_steps}  "
        f"steps_per_epoch={steps_per_epoch}"
    )

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_path  = output_dir / "train_log.jsonl"
    log_file  = open(log_path, "w", encoding="utf-8")
    loss_log  : List[Dict] = []   # in-memory cho plotting
    epoch_results: List[Dict] = []

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step     = 0   # optimizer steps
    micro_step      = 0   # forward pass count
    running_loss    = 0.0
    t_start         = time.time()

    # tqdm progress bar toàn bộ training (theo optimizer steps)
    overall_pbar = tqdm(
        total=total_opt_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        colour="green",
    )

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        # tqdm progress bar cho từng epoch (theo micro-batches)
        epoch_pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,       # xoá bar sau khi epoch xong
        )

        for batch in epoch_pbar:
            micro_step += 1

            # Move to device
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # ── Forward pass ──────────────────────────────────────────────────
            with autocast(dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,   # Tắt cache khi training để tiết kiệm memory
                )
                # Tính L_LM loss thủ công để đảm bảo chính xác
                loss = compute_lm_loss(outputs.logits, labels)
                # Scale loss cho gradient accumulation
                loss_scaled = loss / accum_steps

            # ── Backward ──────────────────────────────────────────────────────
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            running_loss += loss.item()

            # ── Optimizer step (sau accum_steps micro-batches) ───────────────
            is_accum_step = (micro_step % accum_steps == 0)
            is_last_batch = (micro_step == len(train_loader))

            if is_accum_step or is_last_batch:
                global_step += 1

                # Gradient clipping
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )

                # Optimizer step
                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # Avg loss qua các micro-batches tích lũy
                actual_accum = (micro_step - 1) % accum_steps + 1
                avg_loss     = running_loss / actual_accum
                current_lr   = scheduler.get_last_lr()[0]

                # Elapsed time
                elapsed  = time.time() - t_start
                tok_sec  = (input_ids.numel() * actual_accum) / max(elapsed, 1e-9)
                t_start  = time.time()

                # Logging
                log_entry = {
                    "epoch":      epoch,
                    "step":       global_step,
                    "micro_step": micro_step,
                    "loss":       round(avg_loss, 6),
                    "lr":         current_lr,
                    "grad_norm":  round(float(grad_norm), 4),
                    "tok_sec":    round(tok_sec, 0),
                    "tasks":      list(set(batch["task"])),
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()
                loss_log.append({"step": global_step, "loss": avg_loss, "lr": current_lr})

                # Cập nhật overall progress bar
                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    epoch=f"{epoch}/{args.epochs}",
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    grad=f"{float(grad_norm):.3f}",
                )

                # Cập nhật epoch progress bar
                epoch_pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=f"{global_step}",
                )

                running_loss = 0.0

        epoch_pbar.close()

        # ── End of epoch ──────────────────────────────────────────────────────
        logger.info(f"\n[Epoch {epoch}] Training done.")

        # 1. Push to Hub
        push_to_hub(model, tokenizer, hub_repo, epoch, private=args.hub_private)

        # 2. Save local checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        logger.info(f"[Checkpoint] Saved → {ckpt_path}")

        # 3. Evaluation
        if not args.skip_eval:
            eval_results = run_evaluation(
                model, tokenizer, args, device, amp_dtype, epoch, output_dir
            )
            epoch_results.append(eval_results)

            # Update plots
            plot_metrics(epoch_results, output_dir)
            plot_training_loss(loss_log, output_dir)

    overall_pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    log_file.close()
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if epoch_results:
        logger.info("\nFinal Metrics Summary:")
        for res in epoch_results:
            ep = res["epoch"]
            logger.info(
                f"  Epoch {ep} | "
                f"MMLU acc={res['mmlu']['accuracy']:.4f} | "
                f"SNLI acc={res['snli']['accuracy']:.4f} | "
                f"SQuAD F1={res['squad']['f1']:.4f} EM={res['squad']['em']:.4f}"
            )

    # Save final full report
    final_report = {
        "model":          args.model_name,
        "hub_repo":       hub_repo,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "epoch_results":  epoch_results,
    }
    with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.info(f"[Done] Final report → {output_dir / 'final_report.json'}")

    # Final plots
    plot_metrics(epoch_results, output_dir)
    plot_training_loss(loss_log, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Llama-3-8B Full Finetuning")
    logger.info("=" * 60)
    logger.info(f"  model:       {args.model_name}")
    logger.info(f"  data_root:   {args.data_root}")
    logger.info(f"  output_dir:  {args.output_dir}")
    logger.info(f"  hub_repo:    {args.hub_repo}")
    logger.info(f"  epochs:      {args.epochs}")
    logger.info(f"  batch_size:  {args.batch_size} (micro={args.micro_batch})")
    logger.info(f"  lr:          {args.lr}")
    logger.info(f"  bf16:        {args.bf16}")
    logger.info(f"  fp16:        {args.fp16}")
    logger.info("=" * 60)

    train(args)