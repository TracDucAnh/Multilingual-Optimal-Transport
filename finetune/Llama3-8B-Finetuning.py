"""
Llama3-8B-Finetuning.py
========================
Full-model supervised fine-tuning của meta-llama/Meta-Llama-3-8B-Instruct
trên ba English benchmark datasets (MMLU, SQuAD, SNLI) với:

  - Generate mode + L_LM loss (CrossEntropy trên answer tokens)
  - Chạy TUẦN TỰ từng task: MMLU → SQuAD → SNLI (không mix)
  - 1 SequentialTaskDataLoader duy nhất với adaptive batch size (OOM → giảm ½)
  - Resume tự động: check HF Hub → load training_state.json → skip epoch đã xong
  - Mỗi epoch push HF Hub kèm training_state.json
  - Per-iteration logging ra file .jsonl
  - Đồ thị acc / F1 / EM theo epoch (matplotlib)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_LM Loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_LM = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)

  shift_logits = logits[:, :-1, :]
  shift_labels = labels[:, 1:]      (-100 trên prompt, answer+EOS có loss)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOM Handling (Fixed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Catch cả torch.cuda.OutOfMemoryError lẫn RuntimeError("CUDA error: out of memory")
    (torch.AcceleratorError kế thừa RuntimeError, không phải OutOfMemoryError)
  - Sau OOM: del tensors + gc.collect() để giải phóng GPU memory triệt để
  - oom_skip_count cooldown: bỏ qua N batch sau OOM cho GPU kịp ổn định

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resume
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Khi start: check HF Hub có repo chưa
  - Nếu chưa: bắt đầu từ epoch 1
  - Nếu có: download training_state.json → resume từ epoch tiếp theo
  - training_state.json chứa: completed_epochs, batch_sizes, epoch_results, ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python Llama3-8B-Finetuning.py \
        --data_root   ../raw_data/english/ \
        --output_dir  ./checkpoints \
        --hub_repo    Llama3-8B-Finetune \
        --epochs      3 \
        --batch_size  64 \
        --lr          2e-5 \
        --max_new_tokens 32
"""

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dataloader"))

from finetune_dataloader import (
    SequentialTaskDataLoader,
    MMLUValDataLoader,
    SNLIValDataLoader,
    SQuADValDataLoader,
    parse_mmlu_output,
    parse_snli_output,
    compute_exact_match,
    compute_f1_score,
    MMLU_OPTIONS,
    SNLI_CANDIDATES,
    TASK_ORDER,
    DEFAULT_MODEL,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tên file state trên HF Hub
TRAINING_STATE_FILE = "training_state.json"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-3-8B Full Finetune — Sequential Tasks")

    parser.add_argument("--data_root",       type=str,   default="../raw_data/english/")
    parser.add_argument("--output_dir",      type=str,   default="./checkpoints")
    parser.add_argument("--hub_repo",        type=str,   default="Llama3-8B-Finetune")
    parser.add_argument("--hub_private",     action="store_true")

    # Training hyper-params — cố định mỗi run, không đổi khi resume
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=64,
                        help="Initial batch size mỗi task (tự giảm khi OOM)")
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",    type=float, default=0.03)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--max_grad_norm",   type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--oom_skip_batches", type=int,  default=3,
                        help="Số batch bỏ qua sau mỗi OOM để GPU kịp giải phóng memory")

    # Model
    parser.add_argument("--model_name",      type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--mmlu_max_len",    type=int,   default=512)
    parser.add_argument("--squad_max_len",   type=int,   default=1024)
    parser.add_argument("--snli_max_len",    type=int,   default=256)

    # Evaluation
    parser.add_argument("--eval_batch",      type=int,   default=8)
    parser.add_argument("--max_new_tokens",  type=int,   default=32)
    parser.add_argument("--skip_eval",       action="store_true")

    # Misc
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--bf16",            action="store_true", default=True)
    parser.add_argument("--fp16",            action="store_true")

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
# L_LM loss
# ---------------------------------------------------------------------------

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size   = shift_logits.size(-1)
    loss_fct     = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    return loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    )


# ---------------------------------------------------------------------------
# OOM detection helper
# ---------------------------------------------------------------------------

def _is_oom_error(e: Exception) -> bool:
    """
    Trả True nếu exception là OOM — bao gồm:
      - torch.cuda.OutOfMemoryError  (PyTorch gốc)
      - RuntimeError / torch.AcceleratorError có chứa "out of memory"
        (xảy ra khi lỗi được raise từ CUDA kernel bất đồng bộ)
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _cleanup_after_oom(
    optimizer: torch.optim.Optimizer,
    *tensors_to_delete,
) -> None:
    """
    Dọn dẹp GPU memory sau OOM:
      1. del tất cả tensors được truyền vào
      2. zero_grad(set_to_none=True) — giải phóng grad buffers
      3. empty_cache() — trả memory về CUDA allocator
      4. gc.collect() — Python garbage collection
    """
    for t in tensors_to_delete:
        try:
            del t
        except Exception:
            pass
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# HF Hub — Resume helpers
# ---------------------------------------------------------------------------

def _resolve_hub_repo(hub_repo: str) -> str:
    """Nếu repo chưa có '/', thêm username vào trước."""
    if "/" not in hub_repo:
        try:
            api      = HfApi()
            username = "ducanhdinh"
            hub_repo = f"{username}/{hub_repo}"
            logger.info(f"[Hub] Full repo_id: {hub_repo}")
        except Exception as e:
            logger.warning(f"[Hub] Không lấy được username: {e}")
    return hub_repo


def _ensure_repo_exists(hub_repo: str, private: bool) -> None:
    api = HfApi()
    try:
        api.repo_info(repo_id=hub_repo, repo_type="model")
        logger.info(f"[Hub] Repo '{hub_repo}' đã tồn tại.")
    except RepositoryNotFoundError:
        logger.info(f"[Hub] Tạo repo mới: '{hub_repo}' (private={private})")
        api.create_repo(repo_id=hub_repo, repo_type="model", private=private, exist_ok=True)


def load_training_state(hub_repo: str) -> Optional[Dict]:
    """
    Cố gắng download training_state.json từ HF Hub.
    Trả về dict nếu tìm thấy, None nếu repo chưa có hoặc file không tồn tại.
    """
    try:
        local_path = hf_hub_download(
            repo_id=hub_repo,
            filename=TRAINING_STATE_FILE,
            repo_type="model",
        )
        with open(local_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(f"[Resume] Tìm thấy {TRAINING_STATE_FILE} trên HF Hub.")
        logger.info(f"[Resume] Đã hoàn thành epochs: {state.get('completed_epochs', [])}")
        return state
    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.info(f"[Resume] Không tìm thấy {TRAINING_STATE_FILE} — bắt đầu từ đầu.")
        return None
    except Exception as e:
        logger.warning(f"[Resume] Lỗi khi load state: {e} — bắt đầu từ đầu.")
        return None


def save_and_push_state(
    hub_repo: str,
    output_dir: Path,
    state: Dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    epoch: int,
    private: bool,
) -> None:
    """
    1. Lưu training_state.json local
    2. Push model + tokenizer + state lên HF Hub
    """
    # Lưu state local
    state_path = output_dir / TRAINING_STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    logger.info(f"[State] Saved local → {state_path}")

    logger.info(f"[Hub] Pushing epoch {epoch} → {hub_repo}")
    commit_msg = f"Epoch {epoch} checkpoint — completed_epochs={state['completed_epochs']}"

    try:
        _ensure_repo_exists(hub_repo, private)
        model.push_to_hub(hub_repo, commit_message=commit_msg, private=private)
        tokenizer.push_to_hub(hub_repo, commit_message=commit_msg, private=private)

        # Upload training_state.json riêng
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(state_path),
            path_in_repo=TRAINING_STATE_FILE,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=f"Update {TRAINING_STATE_FILE} — epoch {epoch}",
        )
        logger.info(f"[Hub] ✓ Push thành công (epoch={epoch}, state đã upload)")
    except Exception as e:
        logger.error(f"[Hub] Push thất bại: {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mmlu(model, tokenizer, val_loader, device, max_new_tokens, amp_dtype, use_amp) -> Dict:
    model.eval()
    n_correct, n_total, n_unknown = 0, 0, 0
    pbar = tqdm(val_loader, desc="  [Eval] MMLU", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_labels    = batch["gold_label"]
        prompt_len     = input_ids.size(1)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            gen_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen_ids[:, prompt_len:]
        for gen, gold in zip(new_ids, gold_labels):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_mmlu_output(decoded)
            if pred == "": n_unknown += 1
            if pred == gold: n_correct += 1
            n_total += 1
        acc_so_far = n_correct / n_total if n_total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.4f}", correct=n_correct, total=n_total)
    pbar.close()
    accuracy     = n_correct / n_total if n_total > 0 else 0.0
    unknown_rate = n_unknown / n_total if n_total > 0 else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": n_total,
            "unknown_rate": unknown_rate}


@torch.no_grad()
def evaluate_snli(model, tokenizer, val_loader, device, max_new_tokens, amp_dtype, use_amp) -> Dict:
    model.eval()
    n_correct, n_total, n_unknown = 0, 0, 0
    pbar = tqdm(val_loader, desc="  [Eval] SNLI", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_labels    = batch["gold_label"]
        prompt_len     = input_ids.size(1)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            gen_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen_ids[:, prompt_len:]
        for gen, gold in zip(new_ids, gold_labels):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_snli_output(decoded)
            if pred == "": n_unknown += 1
            if pred == gold: n_correct += 1
            n_total += 1
        acc_so_far = n_correct / n_total if n_total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.4f}", correct=n_correct, total=n_total)
    pbar.close()
    accuracy     = n_correct / n_total if n_total > 0 else 0.0
    unknown_rate = n_unknown / n_total if n_total > 0 else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": n_total,
            "unknown_rate": unknown_rate}


@torch.no_grad()
def evaluate_squad(model, tokenizer, val_loader, device, max_new_tokens, amp_dtype, use_amp) -> Dict:
    model.eval()
    total_f1, total_em, n_total = 0.0, 0.0, 0
    pbar = tqdm(val_loader, desc="  [Eval] SQuAD", unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        all_answers    = batch["answers"]
        prompt_len     = input_ids.size(1)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            gen_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = gen_ids[:, prompt_len:]
        for gen, gold_list in zip(new_ids, all_answers):
            pred      = tokenizer.decode(gen, skip_special_tokens=True).strip()
            total_em += compute_exact_match(pred, gold_list)
            total_f1 += compute_f1_score(pred, gold_list)
            n_total  += 1
        f1_so_far = total_f1 / n_total if n_total > 0 else 0.0
        em_so_far = total_em / n_total if n_total > 0 else 0.0
        pbar.set_postfix(F1=f"{f1_so_far:.4f}", EM=f"{em_so_far:.4f}", n=n_total)
    pbar.close()
    f1 = total_f1 / n_total if n_total > 0 else 0.0
    em = total_em / n_total if n_total > 0 else 0.0
    return {"f1": f1, "em": em, "n_total": n_total}


def run_evaluation(model, tokenizer, args, device, amp_dtype, use_amp, epoch, output_dir) -> Dict:
    logger.info(f"[Eval] Bắt đầu evaluation sau epoch {epoch}...")
    tokenizer.padding_side = "left"

    logger.info("[Eval] MMLU...")
    mmlu_val = MMLUValDataLoader(
        data_root=args.data_root, tokenizer=tokenizer,
        batch_size=args.eval_batch, max_length=args.mmlu_max_len,
        shuffle=False, num_workers=0,
    )
    mmlu_res = evaluate_mmlu(model, tokenizer, mmlu_val, device, args.max_new_tokens, amp_dtype, use_amp)
    logger.info(f"[Eval] MMLU  acc={mmlu_res['accuracy']:.4f}  "
                f"({mmlu_res['n_correct']}/{mmlu_res['n_total']})  "
                f"unknown_rate={mmlu_res['unknown_rate']:.3f}")

    logger.info("[Eval] SNLI...")
    snli_val = SNLIValDataLoader(
        data_root=args.data_root, tokenizer=tokenizer,
        batch_size=args.eval_batch, max_length=args.snli_max_len,
        shuffle=False, num_workers=0,
    )
    snli_res = evaluate_snli(model, tokenizer, snli_val, device, args.max_new_tokens, amp_dtype, use_amp)
    logger.info(f"[Eval] SNLI  acc={snli_res['accuracy']:.4f}  "
                f"({snli_res['n_correct']}/{snli_res['n_total']})  "
                f"unknown_rate={snli_res['unknown_rate']:.3f}")

    logger.info("[Eval] SQuAD...")
    squad_val = SQuADValDataLoader(
        data_root=args.data_root, tokenizer=tokenizer,
        batch_size=args.eval_batch, max_length=args.squad_max_len,
        shuffle=False, num_workers=0,
    )
    squad_res = evaluate_squad(model, tokenizer, squad_val, device, args.max_new_tokens, amp_dtype, use_amp)
    logger.info(f"[Eval] SQuAD F1={squad_res['f1']:.4f}  EM={squad_res['em']:.4f}  "
                f"n={squad_res['n_total']}")

    tokenizer.padding_side = "right"
    results = {"epoch": epoch, "mmlu": mmlu_res, "snli": snli_res, "squad": squad_res}

    report_path = output_dir / f"report_epoch{epoch}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"[Eval] Report saved → {report_path}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(epoch_results: List[Dict], output_dir: Path) -> None:
    epochs = [r["epoch"] for r in epoch_results]
    if not epochs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Evaluation Metrics per Epoch", fontsize=14, fontweight="bold")

    ax = axes[0]
    mmlu_acc = [r["mmlu"]["accuracy"] for r in epoch_results]
    ax.plot(epochs, mmlu_acc, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_title("MMLU Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    for e, v in zip(epochs, mmlu_acc):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax = axes[1]
    snli_acc = [r["snli"]["accuracy"] for r in epoch_results]
    ax.plot(epochs, snli_acc, "o-", color="darkorange", linewidth=2, markersize=8)
    ax.set_title("SNLI Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    for e, v in zip(epochs, snli_acc):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax = axes[2]
    squad_f1 = [r["squad"]["f1"] for r in epoch_results]
    squad_em = [r["squad"]["em"] for r in epoch_results]
    ax.plot(epochs, squad_f1, "o-", color="forestgreen", linewidth=2, markersize=8, label="F1")
    ax.plot(epochs, squad_em, "s--", color="crimson", linewidth=2, markersize=8, label="EM")
    ax.set_title("SQuAD F1 / EM"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
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


def plot_training_loss(loss_log: List[Dict], output_dir: Path) -> None:
    if not loss_log:
        return
    steps  = [e["step"]  for e in loss_log]
    losses = [e["loss"]  for e in loss_log]
    lrs    = [e["lr"]    for e in loss_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")

    ax1.plot(steps, losses, color="steelblue", linewidth=0.8, alpha=0.7)
    if len(losses) >= 50:
        window = 50
        smooth = [sum(losses[max(0,i-window):i+1]) / min(i+1, window)
                  for i in range(len(losses))]
        ax1.plot(steps, smooth, color="red", linewidth=1.5, label="MA-50")
        ax1.legend()
    ax1.set_ylabel("L_LM Loss"); ax1.set_xlabel("Global Step"); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, lrs, color="darkorange", linewidth=1.0)
    ax2.set_ylabel("Learning Rate"); ax2.set_xlabel("Global Step"); ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plot_path = output_dir / "training_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Plot] Training loss saved → {plot_path}")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device & dtype ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16; amp_dtype = torch.bfloat16; use_amp = True
        logger.info("[Setup] Using bfloat16 mixed precision")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16; amp_dtype = torch.float16; use_amp = True
        logger.info("[Setup] Using float16 mixed precision")
    else:
        dtype = torch.float32; amp_dtype = torch.float32; use_amp = False
        logger.info("[Setup] Using float32 (no mixed precision)")

    # ── HF Hub login ──────────────────────────────────────────────────────────
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("[Hub] Logged in with HF_TOKEN")
    else:
        logger.warning("[Hub] HF_TOKEN không được set — push hub có thể thất bại")

    hub_repo = _resolve_hub_repo(args.hub_repo)

    # ── Check resume ──────────────────────────────────────────────────────────
    training_state = load_training_state(hub_repo)

    if training_state is not None:
        completed_epochs   = training_state.get("completed_epochs", [])
        epoch_results      = training_state.get("epoch_results", [])
        saved_batch_sizes  = training_state.get("batch_sizes", {})
        logger.info(
            f"[Resume] ✓ Resume từ epoch {max(completed_epochs)+1 if completed_epochs else 1}\n"
            f"[Resume]   Đã xong epochs: {completed_epochs}\n"
            f"[Resume]   Batch sizes đã tìm: {saved_batch_sizes}"
        )
        # Load model từ Hub
        logger.info(f"[Resume] Loading model từ {hub_repo}...")
        tokenizer = AutoTokenizer.from_pretrained(hub_repo, use_fast=True)
        model     = AutoModelForCausalLM.from_pretrained(
            hub_repo,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        logger.info(f"[Resume] ✓ Model loaded từ Hub")
    else:
        completed_epochs  = []
        epoch_results     = []
        saved_batch_sizes = {}
        logger.info("[Setup] Bắt đầu fine-tune mới từ đầu.")
        logger.info(f"[Setup] Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        logger.info(f"[Setup] Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Setup] Trainable params: {n_params/1e9:.2f}B")

    # ── DataLoader ────────────────────────────────────────────────────────────
    logger.info("[Setup] Building SequentialTaskDataLoader...")
    train_loader = SequentialTaskDataLoader(
        data_root=args.data_root,
        tokenizer=tokenizer,
        initial_batch_size=args.batch_size,
        mmlu_max_length=args.mmlu_max_len,
        squad_max_length=args.squad_max_len,
        snli_max_length=args.snli_max_len,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Restore batch sizes đã tìm được trước đó (tránh OOM lại)
    if saved_batch_sizes:
        for task, bs in saved_batch_sizes.items():
            if task in train_loader.current_batch_sizes:
                old = train_loader.current_batch_sizes[task]
                train_loader.current_batch_sizes[task] = bs
                if bs != old:
                    logger.info(f"[Resume] Restored batch_size[{task}]: {old} → {bs}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    no_decay     = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95))

    # Ước tính tổng steps (có thể thay đổi nếu OOM → bs nhỏ hơn → nhiều batch hơn)
    steps_per_epoch = len(train_loader)   # ước tính với bs hiện tại
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_opt_steps * args.warmup_ratio))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    # GradScaler chỉ dùng cho fp16 (bf16 không cần scaler)
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(f"[Setup] steps_per_epoch≈{steps_per_epoch}  "
                f"total_opt_steps≈{total_opt_steps}  warmup={warmup_steps}")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path  = output_dir / "train_log.jsonl"
    # Append mode nếu resume
    log_mode  = "a" if completed_epochs else "w"
    log_file  = open(log_path, log_mode, encoding="utf-8")
    loss_log: List[Dict] = []

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step  = 0
    t_start      = time.time()

    # Overall progress bar (theo epochs)
    epoch_pbar = tqdm(
        range(1, args.epochs + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
        colour="green",
    )

    for epoch in epoch_pbar:
        # ── Kiểm tra có cần skip epoch này không (resume) ─────────────────────
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} đã hoàn thành — bỏ qua.")
            epoch_pbar.set_postfix(status="skipped (resumed)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}  (task order: {' → '.join(TASK_ORDER)})")
        logger.info(f"{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Progress bar cho epoch (theo batches)
        batch_pbar = tqdm(
            desc=f"  Epoch {epoch} | task=?",
            total=len(train_loader),
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        running_loss   = 0.0
        running_count  = 0
        current_task   = None
        oom_skip_count = 0   # số batch còn phải skip sau OOM gần nhất

        for batch in train_loader:
            task = batch["task"][0]

            # Thông báo khi đổi task
            if task != current_task:
                if current_task is not None:
                    logger.info(
                        f"\n[Train] ✓ Task '{current_task.upper()}' hoàn thành trong epoch {epoch}."
                    )
                current_task = task
                bs_now = train_loader.current_batch_sizes[task]
                logger.info(
                    f"\n[Train] ▶ Chuyển sang task: {task.upper()}  "
                    f"batch_size={bs_now}  epoch={epoch}"
                )
                batch_pbar.set_description(f"  Epoch {epoch} | task={task.upper()}")
                # Cập nhật total vì bs có thể đã đổi
                batch_pbar.total = len(train_loader)
                batch_pbar.refresh()

            # ── Cooldown sau OOM: skip N batch để GPU kịp "nguội" ─────────────
            if oom_skip_count > 0:
                oom_skip_count -= 1
                batch_pbar.update(1)
                batch_pbar.set_postfix(status=f"OOM cooldown ({oom_skip_count} left)")
                continue

            # ── Move batch lên device ─────────────────────────────────────────
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # ── Forward pass với OOM handling ─────────────────────────────────
            outputs = None
            loss    = None
            try:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    loss = compute_lm_loss(outputs.logits, labels)

                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss  += loss.item()
                running_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom_error(e):
                    # Không phải OOM → re-raise nguyên vẹn
                    raise

                # ── Dọn sạch GPU memory triệt để ──────────────────────────────
                _cleanup_after_oom(optimizer, outputs, loss, input_ids, attention_mask, labels)

                try:
                    new_bs = train_loader.report_oom(task)
                    oom_skip_count = args.oom_skip_batches
                    logger.warning(
                        f"[OOM] Epoch {epoch} | task={task} | "
                        f"bs → {new_bs} | skip {oom_skip_count} batch kế tiếp"
                    )
                    # Cập nhật tqdm total vì bs thay đổi → số batch thay đổi
                    batch_pbar.total = len(train_loader)
                    batch_pbar.refresh()
                except RuntimeError as re:
                    logger.error(f"[OOM] Không thể giảm batch size thêm: {re}")
                    raise

                batch_pbar.update(1)
                continue

            # ── Optimizer step ─────────────────────────────────────────────────
            if use_amp and amp_dtype == torch.float16:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            if use_amp and amp_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            avg_loss     = running_loss / running_count
            current_lr   = scheduler.get_last_lr()[0]
            running_loss = running_count = 0

            # Logging
            elapsed = time.time() - t_start
            t_start = time.time()
            log_entry = {
                "epoch":     epoch,
                "step":      global_step,
                "task":      task,
                "loss":      round(avg_loss, 6),
                "lr":        current_lr,
                "grad_norm": round(float(grad_norm), 4),
                "bs":        train_loader.current_batch_sizes[task],
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            loss_log.append({"step": global_step, "loss": avg_loss, "lr": current_lr})

            batch_pbar.update(1)
            batch_pbar.set_postfix(
                task=task,
                loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}",
                grad=f"{float(grad_norm):.3f}",
                bs=train_loader.current_batch_sizes[task],
            )
            epoch_pbar.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                step=global_step,
                loss=f"{avg_loss:.4f}",
            )

        batch_pbar.close()
        if current_task:
            logger.info(f"\n[Train] ✓ Task '{current_task.upper()}' hoàn thành (epoch {epoch}).")

        # ── End of epoch ───────────────────────────────────────────────────────
        logger.info(f"\n[Epoch {epoch}] Training done.")

        # Evaluation
        eval_res = None
        if not args.skip_eval:
            eval_res = run_evaluation(
                model, tokenizer, args, device, amp_dtype, use_amp, epoch, output_dir
            )
            epoch_results.append(eval_res)
            plot_metrics(epoch_results, output_dir)
            plot_training_loss(loss_log, output_dir)

        # Cập nhật training state
        completed_epochs.append(epoch)
        state = {
            "model_name":        args.model_name,
            "hub_repo":          hub_repo,
            "total_epochs":      args.epochs,
            "completed_epochs":  completed_epochs,
            "batch_sizes":       dict(train_loader.current_batch_sizes),
            "lr":                args.lr,
            "seed":              args.seed,
            "epoch_results":     epoch_results,
            "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Lưu local checkpoint
        ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        logger.info(f"[Checkpoint] Saved → {ckpt_path}")

        # Push lên HF Hub (model + tokenizer + state)
        save_and_push_state(
            hub_repo=hub_repo,
            output_dir=output_dir,
            state=state,
            model=model,
            tokenizer=tokenizer,
            epoch=epoch,
            private=args.hub_private,
        )

    epoch_pbar.close()
    log_file.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Completed epochs : {completed_epochs}")
    logger.info(f"  Final batch sizes: {train_loader.current_batch_sizes}")

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

    final_report = {
        "model":           args.model_name,
        "hub_repo":        hub_repo,
        "epochs":          args.epochs,
        "completed_epochs": completed_epochs,
        "batch_sizes":     dict(train_loader.current_batch_sizes),
        "lr":              args.lr,
        "epoch_results":   epoch_results,
    }
    with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.info(f"[Done] Final report → {output_dir / 'final_report.json'}")

    plot_metrics(epoch_results, output_dir)
    plot_training_loss(loss_log, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Llama-3-8B Full Finetuning — Sequential Tasks")
    logger.info("=" * 60)
    logger.info(f"  model:        {args.model_name}")
    logger.info(f"  data_root:    {args.data_root}")
    logger.info(f"  output_dir:   {args.output_dir}")
    logger.info(f"  hub_repo:     {args.hub_repo}")
    logger.info(f"  epochs:       {args.epochs}")
    logger.info(f"  batch_size:   {args.batch_size} (auto-reduce on OOM)")
    logger.info(f"  oom_skip:     {args.oom_skip_batches} batches after each OOM")
    logger.info(f"  task_order:   {' → '.join(TASK_ORDER)}")
    logger.info(f"  lr:           {args.lr}")
    logger.info(f"  bf16:         {args.bf16}  fp16: {args.fp16}")
    logger.info("=" * 60)

    train(args)