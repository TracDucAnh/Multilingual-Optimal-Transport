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
Last-N Layers Finetuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Llama-3-8B có 32 transformer layers (model.layers[0..31]).

  Với --freeze_layers N:
    - Freeze : embed_tokens + layers[0 .. 31-N-1]
    - Train  : layers[31-N .. 31] + norm + lm_head

  Memory estimate (bf16, AdamW, 80GB GPU):
    N=4  → trainable ~1.0B → optimizer ~8GB  → tổng ~40GB  ✓ bs=32
    N=8  → trainable ~2.0B → optimizer ~16GB → tổng ~48GB  ✓ bs=16
    N=16 → trainable ~4.0B → optimizer ~32GB → tổng ~64GB  ✓ bs=8
    N=32 → full finetune   → optimizer ~64GB → tổng ~96GB  ✗ OOM

  embed_tokens luôn bị freeze (tiết kiệm ~1GB optimizer state, ít ảnh hưởng perf).
  Gradient checkpointing bật mặc định → tiết kiệm thêm ~10GB activation memory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recommended cho 80GB — train 8 layers cuối
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python Llama3-8B-Finetuning.py \
        --data_root   ../raw_data/english/ \
        --output_dir  ./checkpoints \
        --hub_repo    Llama3-8B-Finetune \
        --epochs      3 \
        --batch_size  16 \
        --lr          2e-5 \
        --freeze_layers 8
"""

import argparse
import gc
import json
import logging
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

# Giảm CUDA memory fragmentation — phải set trước mọi CUDA call
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAINING_STATE_FILE = "training_state.json"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-3-8B Last-N Layers Finetune")

    parser.add_argument("--data_root",        type=str,   default="../raw_data/english/")
    parser.add_argument("--output_dir",       type=str,   default="./checkpoints")
    parser.add_argument("--hub_repo",         type=str,   default="Llama3-8B-Finetune")
    parser.add_argument("--hub_private",      action="store_true")

    # Training hyper-params
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--weight_decay",     type=float, default=0.01)
    parser.add_argument("--max_grad_norm",    type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--oom_skip_batches", type=int,   default=3)

    # ── Last-N layers (thay đổi chính) ────────────────────────────────────────
    parser.add_argument(
        "--freeze_layers", type=int, default=8,
        help=(
            "Số transformer layers CUỐI được train (còn lại bị freeze).\n"
            "  0  → full finetune (không freeze gì, dễ OOM)\n"
            "  8  → train layers[24..31] + norm + lm_head  (recommended 80GB)\n"
            "  16 → train layers[16..31] + norm + lm_head\n"
            "  32 → train tất cả layers + norm + lm_head (= full nhưng freeze embed)"
        ),
    )

    # Memory
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Bật gradient checkpointing — tiết kiệm ~10GB activation memory, chậm hơn ~20%%",
    )
    parser.add_argument(
        "--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false",
    )

    # Model
    parser.add_argument("--model_name",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mmlu_max_len",  type=int, default=512)
    parser.add_argument("--squad_max_len", type=int, default=1024)
    parser.add_argument("--snli_max_len",  type=int, default=256)

    # Evaluation
    parser.add_argument("--eval_batch",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--skip_eval",      action="store_true")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")

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
# Last-N layers freeze  ← THAY ĐỔI CHÍNH
# ---------------------------------------------------------------------------

def freeze_model_layers(model: AutoModelForCausalLM, n_train_layers: int) -> None:
    """
    Freeze toàn bộ model rồi unfreeze N transformer layers cuối + norm + lm_head.

    Llama-3-8B structure:
        model.model.embed_tokens          ← luôn freeze
        model.model.layers[0..31]         ← freeze layers[0..31-N], train layers[31-N+1..31]
        model.model.norm                  ← luôn train
        model.lm_head                     ← luôn train

    Args:
        n_train_layers: số layers cuối được train.
                        0 = không train gì (vô nghĩa).
                        32 = train tất cả layers (nhưng embed vẫn frozen).
    """
    # Bước 1: Freeze toàn bộ
    for param in model.parameters():
        param.requires_grad = False

    if n_train_layers <= 0:
        logger.warning("[Freeze] n_train_layers=0 → toàn bộ model frozen, không có gì để train!")
        return

    transformer_layers = model.model.layers          # ModuleList
    total_layers       = len(transformer_layers)     # 32 với Llama-3-8B
    n_train            = min(n_train_layers, total_layers)
    first_train_idx    = total_layers - n_train      # ví dụ: 32-8 = 24

    logger.info(f"[Freeze] Tổng transformer layers : {total_layers}")
    logger.info(f"[Freeze] Frozen  : embed_tokens + layers[0..{first_train_idx - 1}]")
    logger.info(f"[Freeze] Trainable: layers[{first_train_idx}..{total_layers - 1}] + norm + lm_head")

    # Bước 2: Unfreeze N layers cuối
    for layer in transformer_layers[first_train_idx:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Bước 3: Unfreeze final norm + lm_head (luôn cần train để output đúng)
    for param in model.model.norm.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Bước 4: Log thống kê + memory estimate
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = total_p - trainable_p
    opt_mem_gb  = trainable_p * 4 * 2 / 1e9   # AdamW: 2 fp32 buffers per param

    logger.info(f"[Freeze] Total params    : {total_p / 1e9:.3f}B")
    logger.info(f"[Freeze] Trainable params: {trainable_p / 1e9:.3f}B  "
                f"({100 * trainable_p / total_p:.1f}%)")
    logger.info(f"[Freeze] Frozen params   : {frozen_p / 1e9:.3f}B  "
                f"({100 * frozen_p / total_p:.1f}%)")
    logger.info(f"[Freeze] Est. optimizer memory (AdamW fp32): ~{opt_mem_gb:.1f} GB")


# ---------------------------------------------------------------------------
# L_LM loss
# ---------------------------------------------------------------------------

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size   = shift_logits.size(-1)
    loss_fct     = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    return loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))


# ---------------------------------------------------------------------------
# OOM helpers
# ---------------------------------------------------------------------------

def _is_oom_error(e: Exception) -> bool:
    """Catch cả OutOfMemoryError lẫn RuntimeError('CUDA error: out of memory')."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _cleanup_after_oom(optimizer: torch.optim.Optimizer, *tensors) -> None:
    """Xoá tensors, zero grad, empty cache, gc.collect."""
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# HF Hub helpers
# ---------------------------------------------------------------------------

def _resolve_hub_repo(hub_repo: str) -> str:
    if "/" not in hub_repo:
        username = "ducanhdinh"
        hub_repo = f"{username}/{hub_repo}"
        logger.info(f"[Hub] Full repo_id: {hub_repo}")
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
    try:
        local_path = hf_hub_download(
            repo_id=hub_repo, filename=TRAINING_STATE_FILE, repo_type="model",
        )
        with open(local_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(f"[Resume] Tìm thấy {TRAINING_STATE_FILE}. "
                    f"Completed epochs: {state.get('completed_epochs', [])}")
        return state
    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.info("[Resume] Không tìm thấy state — bắt đầu từ đầu.")
        return None
    except Exception as e:
        logger.warning(f"[Resume] Lỗi load state: {e} — bắt đầu từ đầu.")
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
    state_path = output_dir / TRAINING_STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    logger.info(f"[State] Saved → {state_path}")

    commit_msg = f"Epoch {epoch} — completed={state['completed_epochs']}"
    try:
        _ensure_repo_exists(hub_repo, private)
        model.push_to_hub(hub_repo, commit_message=commit_msg, private=private)
        tokenizer.push_to_hub(hub_repo, commit_message=commit_msg, private=private)
        HfApi().upload_file(
            path_or_fileobj=str(state_path),
            path_in_repo=TRAINING_STATE_FILE,
            repo_id=hub_repo, repo_type="model",
            commit_message=f"Update state — epoch {epoch}",
        )
        logger.info(f"[Hub] ✓ Push thành công epoch {epoch}")
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
        for gen, gold in zip(gen_ids[:, prompt_len:], gold_labels):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_mmlu_output(decoded)
            if pred == "": n_unknown += 1
            if pred == gold: n_correct += 1
            n_total += 1
        pbar.set_postfix(acc=f"{n_correct/max(n_total,1):.4f}", n=n_total)
    pbar.close()
    return {
        "accuracy":     n_correct / max(n_total, 1),
        "n_correct":    n_correct,
        "n_total":      n_total,
        "unknown_rate": n_unknown / max(n_total, 1),
    }


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
        for gen, gold in zip(gen_ids[:, prompt_len:], gold_labels):
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            pred    = parse_snli_output(decoded)
            if pred == "": n_unknown += 1
            if pred == gold: n_correct += 1
            n_total += 1
        pbar.set_postfix(acc=f"{n_correct/max(n_total,1):.4f}", n=n_total)
    pbar.close()
    return {
        "accuracy":     n_correct / max(n_total, 1),
        "n_correct":    n_correct,
        "n_total":      n_total,
        "unknown_rate": n_unknown / max(n_total, 1),
    }


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
        for gen, gold_list in zip(gen_ids[:, prompt_len:], all_answers):
            pred      = tokenizer.decode(gen, skip_special_tokens=True).strip()
            total_em += compute_exact_match(pred, gold_list)
            total_f1 += compute_f1_score(pred, gold_list)
            n_total  += 1
        pbar.set_postfix(F1=f"{total_f1/max(n_total,1):.4f}", EM=f"{total_em/max(n_total,1):.4f}")
    pbar.close()
    return {"f1": total_f1 / max(n_total, 1), "em": total_em / max(n_total, 1), "n_total": n_total}


def run_evaluation(model, tokenizer, args, device, amp_dtype, use_amp, epoch, output_dir) -> Dict:
    logger.info(f"[Eval] Epoch {epoch} evaluation...")
    tokenizer.padding_side = "left"

    mmlu_res = evaluate_mmlu(
        model, tokenizer,
        MMLUValDataLoader(data_root=args.data_root, tokenizer=tokenizer,
                          batch_size=args.eval_batch, max_length=args.mmlu_max_len,
                          shuffle=False, num_workers=0),
        device, args.max_new_tokens, amp_dtype, use_amp,
    )
    logger.info(f"[Eval] MMLU  acc={mmlu_res['accuracy']:.4f} "
                f"({mmlu_res['n_correct']}/{mmlu_res['n_total']}) "
                f"unk={mmlu_res['unknown_rate']:.3f}")

    snli_res = evaluate_snli(
        model, tokenizer,
        SNLIValDataLoader(data_root=args.data_root, tokenizer=tokenizer,
                          batch_size=args.eval_batch, max_length=args.snli_max_len,
                          shuffle=False, num_workers=0),
        device, args.max_new_tokens, amp_dtype, use_amp,
    )
    logger.info(f"[Eval] SNLI  acc={snli_res['accuracy']:.4f} "
                f"({snli_res['n_correct']}/{snli_res['n_total']}) "
                f"unk={snli_res['unknown_rate']:.3f}")

    squad_res = evaluate_squad(
        model, tokenizer,
        SQuADValDataLoader(data_root=args.data_root, tokenizer=tokenizer,
                           batch_size=args.eval_batch, max_length=args.squad_max_len,
                           shuffle=False, num_workers=0),
        device, args.max_new_tokens, amp_dtype, use_amp,
    )
    logger.info(f"[Eval] SQuAD F1={squad_res['f1']:.4f} EM={squad_res['em']:.4f} "
                f"n={squad_res['n_total']}")

    tokenizer.padding_side = "right"
    results = {"epoch": epoch, "mmlu": mmlu_res, "snli": snli_res, "squad": squad_res}
    with open(output_dir / f"report_epoch{epoch}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
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

    axes[0].plot(epochs, [r["mmlu"]["accuracy"] for r in epoch_results],
                 "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_title("MMLU Accuracy"); axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3)
    for e, v in zip(epochs, [r["mmlu"]["accuracy"] for r in epoch_results]):
        axes[0].annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    axes[1].plot(epochs, [r["snli"]["accuracy"] for r in epoch_results],
                 "o-", color="darkorange", linewidth=2, markersize=8)
    axes[1].set_title("SNLI Accuracy"); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
    for e, v in zip(epochs, [r["snli"]["accuracy"] for r in epoch_results]):
        axes[1].annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)

    squad_f1 = [r["squad"]["f1"] for r in epoch_results]
    squad_em = [r["squad"]["em"] for r in epoch_results]
    axes[2].plot(epochs, squad_f1, "o-", color="forestgreen", linewidth=2, markersize=8, label="F1")
    axes[2].plot(epochs, squad_em, "s--", color="crimson",    linewidth=2, markersize=8, label="EM")
    axes[2].set_title("SQuAD F1 / EM"); axes[2].set_ylim(0, 1)
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Plot] metrics_plot.png saved")


def plot_training_loss(loss_log: List[Dict], output_dir: Path) -> None:
    if not loss_log:
        return
    steps  = [e["step"] for e in loss_log]
    losses = [e["loss"] for e in loss_log]
    lrs    = [e["lr"]   for e in loss_log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")
    ax1.plot(steps, losses, color="steelblue", linewidth=0.8, alpha=0.7)
    if len(losses) >= 50:
        window = 50
        smooth = [sum(losses[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(losses))]
        ax1.plot(steps, smooth, color="red", linewidth=1.5, label="MA-50")
        ax1.legend()
    ax1.set_ylabel("L_LM Loss"); ax1.set_xlabel("Global Step"); ax1.grid(True, alpha=0.3)
    ax2.plot(steps, lrs, color="darkorange", linewidth=1.0)
    ax2.set_ylabel("Learning Rate"); ax2.set_xlabel("Global Step"); ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[Plot] training_loss.png saved")


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
        logger.info("[Setup] bfloat16 mixed precision")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16; amp_dtype = torch.float16; use_amp = True
        logger.info("[Setup] float16 mixed precision")
    else:
        dtype = torch.float32; amp_dtype = torch.float32; use_amp = False
        logger.info("[Setup] float32")

    # ── HF login ──────────────────────────────────────────────────────────────
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("[Hub] Logged in")
    else:
        logger.warning("[Hub] HF_TOKEN không set")

    hub_repo = _resolve_hub_repo(args.hub_repo)

    # ── Resume ────────────────────────────────────────────────────────────────
    training_state = load_training_state(hub_repo)
    if training_state is not None:
        completed_epochs  = training_state.get("completed_epochs", [])
        epoch_results     = training_state.get("epoch_results", [])
        saved_batch_sizes = training_state.get("batch_sizes", {})
        logger.info(f"[Resume] Completed: {completed_epochs} | BS: {saved_batch_sizes}")
        tokenizer = AutoTokenizer.from_pretrained(hub_repo, use_fast=True)
        model     = AutoModelForCausalLM.from_pretrained(
            hub_repo, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    else:
        completed_epochs  = []
        epoch_results     = []
        saved_batch_sizes = {}
        logger.info(f"[Setup] Loading {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model     = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        model = model.to(device)

    # ── Freeze layers ─────────────────────────────────────────────────────────
    # Phải gọi TRƯỚC gradient_checkpointing_enable()
    if args.freeze_layers > 0:
        freeze_model_layers(model, n_train_layers=args.freeze_layers)
    else:
        logger.info("[Setup] freeze_layers=0 → full finetune")
        total_p = sum(p.numel() for p in model.parameters())
        logger.info(f"[Setup] Trainable: {total_p/1e9:.2f}B")

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if args.gradient_checkpointing:
        # enable_input_require_grads() bắt buộc khi có frozen layers:
        # gradient cần flow từ frozen input → vào trainable layers đầu tiên
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("[Setup] Gradient checkpointing: ON (use_reentrant=False)")
    else:
        logger.info("[Setup] Gradient checkpointing: OFF")

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

    if saved_batch_sizes:
        for task, bs in saved_batch_sizes.items():
            if task in train_loader.current_batch_sizes:
                old = train_loader.current_batch_sizes[task]
                train_loader.current_batch_sizes[task] = bs
                if bs != old:
                    logger.info(f"[Resume] batch_size[{task}]: {old} → {bs}")

    # ── Optimizer — chỉ pass trainable params ─────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    total_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    if total_trainable == 0:
        raise ValueError("Không có param nào được train! Kiểm tra lại --freeze_layers.")
    logger.info(f"[Optim] Trainable params in optimizer: {total_trainable/1e9:.3f}B")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_opt_steps * args.warmup_ratio))
    scheduler       = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(f"[Setup] steps/epoch≈{steps_per_epoch}  total≈{total_opt_steps}  warmup={warmup_steps}")

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_file = open(output_dir / "train_log.jsonl",
                    "a" if completed_epochs else "w", encoding="utf-8")
    loss_log: List[Dict] = []

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs",
                      unit="epoch", dynamic_ncols=True, colour="green")

    for epoch in epoch_pbar:
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} done — skip.")
            epoch_pbar.set_postfix(status="skipped")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}  ({' → '.join(TASK_ORDER)})")
        logger.info(f"{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_pbar = tqdm(desc=f"  Epoch {epoch} | task=?",
                          total=len(train_loader), unit="batch",
                          dynamic_ncols=True, leave=False)

        running_loss   = 0.0
        running_count  = 0
        current_task   = None
        oom_skip_count = 0

        for batch in train_loader:
            task = batch["task"][0]

            # Task switch log
            if task != current_task:
                if current_task is not None:
                    logger.info(f"\n[Train] ✓ '{current_task.upper()}' done epoch {epoch}.")
                current_task = task
                logger.info(f"\n[Train] ▶ {task.upper()}  "
                             f"bs={train_loader.current_batch_sizes[task]}  epoch={epoch}")
                batch_pbar.set_description(f"  Epoch {epoch} | task={task.upper()}")
                batch_pbar.total = len(train_loader)
                batch_pbar.refresh()

            # OOM cooldown
            if oom_skip_count > 0:
                oom_skip_count -= 1
                batch_pbar.update(1)
                batch_pbar.set_postfix(status=f"cooldown ({oom_skip_count} left)")
                continue

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = None
            loss    = None
            try:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=False)
                    loss    = compute_lm_loss(outputs.logits, labels)

                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss  += loss.item()
                running_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom_error(e):
                    raise
                _cleanup_after_oom(optimizer, outputs, loss, input_ids, attention_mask, labels)
                try:
                    new_bs = train_loader.report_oom(task)
                    oom_skip_count = args.oom_skip_batches
                    logger.warning(f"[OOM] task={task} bs→{new_bs} skip={oom_skip_count}")
                    batch_pbar.total = len(train_loader)
                    batch_pbar.refresh()
                except RuntimeError as re:
                    logger.error(f"[OOM] Không thể giảm bs thêm: {re}")
                    raise
                batch_pbar.update(1)
                continue

            # Optimizer step
            if use_amp and amp_dtype == torch.float16:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )

            if use_amp and amp_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step  += 1
            avg_loss      = running_loss / running_count
            current_lr    = scheduler.get_last_lr()[0]
            running_loss  = running_count = 0

            log_entry = {
                "epoch": epoch, "step": global_step, "task": task,
                "loss": round(avg_loss, 6), "lr": current_lr,
                "grad_norm": round(float(grad_norm), 4),
                "bs": train_loader.current_batch_sizes[task],
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            loss_log.append({"step": global_step, "loss": avg_loss, "lr": current_lr})

            batch_pbar.update(1)
            batch_pbar.set_postfix(
                task=task, loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}",
                grad=f"{float(grad_norm):.3f}", bs=train_loader.current_batch_sizes[task],
            )
            epoch_pbar.set_postfix(epoch=f"{epoch}/{args.epochs}",
                                   step=global_step, loss=f"{avg_loss:.4f}")

        batch_pbar.close()
        if current_task:
            logger.info(f"\n[Train] ✓ '{current_task.upper()}' done (epoch {epoch}).")
        logger.info(f"\n[Epoch {epoch}] Training done.")

        # Eval
        if not args.skip_eval:
            eval_res = run_evaluation(
                model, tokenizer, args, device, amp_dtype, use_amp, epoch, output_dir,
            )
            epoch_results.append(eval_res)
            plot_metrics(epoch_results, output_dir)
            plot_training_loss(loss_log, output_dir)

        # Save state & push
        completed_epochs.append(epoch)
        state = {
            "model_name":       args.model_name,
            "hub_repo":         hub_repo,
            "total_epochs":     args.epochs,
            "completed_epochs": completed_epochs,
            "freeze_layers":    args.freeze_layers,
            "batch_sizes":      dict(train_loader.current_batch_sizes),
            "lr":               args.lr,
            "seed":             args.seed,
            "epoch_results":    epoch_results,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        logger.info(f"[Checkpoint] → {ckpt_path}")

        save_and_push_state(hub_repo, output_dir, state, model, tokenizer, epoch, args.hub_private)

    epoch_pbar.close()
    log_file.close()

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Completed epochs : {completed_epochs}")
    logger.info(f"  Trained layers   : last {args.freeze_layers} of {len(model.model.layers)}")
    logger.info(f"  Final batch sizes: {train_loader.current_batch_sizes}")

    if epoch_results:
        logger.info("\nFinal Metrics:")
        for res in epoch_results:
            logger.info(
                f"  Epoch {res['epoch']} | "
                f"MMLU={res['mmlu']['accuracy']:.4f} | "
                f"SNLI={res['snli']['accuracy']:.4f} | "
                f"SQuAD F1={res['squad']['f1']:.4f} EM={res['squad']['em']:.4f}"
            )

    with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_name, "hub_repo": hub_repo,
            "epochs": args.epochs, "freeze_layers": args.freeze_layers,
            "completed_epochs": completed_epochs,
            "batch_sizes": dict(train_loader.current_batch_sizes),
            "lr": args.lr, "epoch_results": epoch_results,
        }, f, indent=2, ensure_ascii=False)

    plot_metrics(epoch_results, output_dir)
    plot_training_loss(loss_log, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Llama-3-8B — Last-N Layers Finetuning")
    logger.info("=" * 60)
    logger.info(f"  model          : {args.model_name}")
    logger.info(f"  data_root      : {args.data_root}")
    logger.info(f"  output_dir     : {args.output_dir}")
    logger.info(f"  hub_repo       : {args.hub_repo}")
    logger.info(f"  epochs         : {args.epochs}")
    logger.info(f"  batch_size     : {args.batch_size}")
    logger.info(f"  freeze_layers  : {args.freeze_layers}  "
                f"(train last {args.freeze_layers}/32 layers + norm + lm_head)")
    logger.info(f"  grad_ckpt      : {args.gradient_checkpointing}")
    logger.info(f"  task_order     : {' → '.join(TASK_ORDER)}")
    logger.info(f"  lr             : {args.lr}")
    logger.info(f"  bf16/fp16      : {args.bf16}/{args.fp16}")
    logger.info("=" * 60)

    train(args)