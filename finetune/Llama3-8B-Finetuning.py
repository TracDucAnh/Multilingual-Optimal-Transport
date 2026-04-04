"""
Llama3-8B-Finetuning.py
========================
Full-model supervised fine-tuning của meta-llama/Meta-Llama-3-8B-Instruct
trên ba English benchmark datasets (MMLU, SQuAD, SNLI) với:

  - Generate mode + L_LM loss (CrossEntropy trên answer tokens)
  - Mixed batch: mỗi batch chứa samples từ CẢ 3 task cùng lúc
  - MixedTaskDataLoader với adaptive batch size (OOM → giảm ½ toàn bộ)
  - Resume tự động: check HF Hub → load training_state.json → skip epoch đã xong
  - --save_iter N: cứ N iteration push checkpoint lên Hub 1 lần (mid-epoch resume)
  - Mỗi epoch push HF Hub kèm training_state.json + optimizer checkpoint
  - Per-iteration logging ra file .jsonl
  - Đồ thị acc / F1 / EM theo epoch (matplotlib)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resume đầy đủ bao gồm:
  - Model weights  (push_to_hub)
  - Optimizer state dict  → optimizer_state.pt  (upload lên Hub)
  - Scheduler state dict  → scheduler_state.pt  (upload lên Hub)
  - GradScaler state dict → scaler_state.pt     (upload lên Hub)
  - training_state.json   → global_step, loss_log, epoch_results, batch_size
                            current_epoch, steps_done_in_epoch  ← mid-epoch resume

  Khi resume mid-epoch:
    - Load model/optimizer/scheduler/scaler từ Hub
    - Skip (steps_done_in_epoch) batches đầu của epoch đó
    - Tiếp tục training từ iteration kế tiếp
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recommended cho 80GB — train 8 layers cuối, save mỗi 500 iter
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python Llama3-8B-Finetuning.py \\
        --data_root   ../raw_data/english/ \\
        --output_dir  ./checkpoints \\
        --hub_repo    Llama3-8B-Finetune \\
        --epochs      3 \\
        --batch_size  16 \\
        --lr          2e-5 \\
        --freeze_layers 8 \\
        --save_iter   500
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
    MixedTaskDataLoader,
    SequentialTaskDataLoader,   # alias, same class
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

TRAINING_STATE_FILE  = "training_state.json"
OPTIMIZER_STATE_FILE = "optimizer_state.pt"
SCHEDULER_STATE_FILE = "scheduler_state.pt"
SCALER_STATE_FILE    = "scaler_state.pt"
LOSS_LOG_FILE        = "loss_log.json"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-3-8B Last-N Layers Finetune (Mixed Batch)")

    parser.add_argument("--data_root",        type=str,   default="../raw_data/english/")
    parser.add_argument("--output_dir",       type=str,   default="./checkpoints")
    parser.add_argument("--hub_repo",         type=str,   default="Llama3-8B-Finetune")
    parser.add_argument("--hub_private",      action="store_true")

    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--weight_decay",     type=float, default=0.01)
    parser.add_argument("--max_grad_norm",    type=float, default=1.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--oom_skip_batches", type=int,   default=3)

    # ── NEW: mid-epoch checkpoint ──────────────────────────────────────────
    parser.add_argument(
        "--save_iter", type=int, default=0,
        help=(
            "Nếu > 0: cứ mỗi save_iter iteration (global_step) sẽ push checkpoint "
            "lên HuggingFace Hub, lưu đủ state để resume lại từ giữa epoch. "
            "Ví dụ: --save_iter 500"
        ),
    )
    # ──────────────────────────────────────────────────────────────────────

    parser.add_argument(
        "--freeze_layers", type=int, default=8,
        help=(
            "Số transformer layers CUỐI được train (còn lại bị freeze).\n"
            "  0  → full finetune\n"
            "  8  → train layers[24..31] + norm + lm_head  (recommended 80GB)\n"
            "  16 → train layers[16..31] + norm + lm_head\n"
            "  32 → train tất cả layers + norm + lm_head (freeze embed)"
        ),
    )

    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
    )
    parser.add_argument(
        "--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false",
    )

    parser.add_argument("--model_name",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mmlu_max_len",  type=int, default=512)
    parser.add_argument("--squad_max_len", type=int, default=1024)
    parser.add_argument("--snli_max_len",  type=int, default=256)

    parser.add_argument("--eval_batch",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--skip_eval",      action="store_true")

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
# Last-N layers freeze
# ---------------------------------------------------------------------------

def freeze_model_layers(model: AutoModelForCausalLM, n_train_layers: int) -> None:
    """
    Freeze toàn bộ model rồi unfreeze N transformer layers cuối + norm + lm_head.
    embed_tokens luôn bị freeze.
    """
    for param in model.parameters():
        param.requires_grad = False

    if n_train_layers <= 0:
        logger.warning("[Freeze] n_train_layers=0 → toàn bộ model frozen!")
        return

    transformer_layers = model.model.layers
    total_layers       = len(transformer_layers)
    n_train            = min(n_train_layers, total_layers)
    first_train_idx    = total_layers - n_train

    logger.info(f"[Freeze] Tổng transformer layers : {total_layers}")
    logger.info(f"[Freeze] Frozen  : embed_tokens + layers[0..{first_train_idx - 1}]")
    logger.info(f"[Freeze] Trainable: layers[{first_train_idx}..{total_layers - 1}] + norm + lm_head")

    for layer in transformer_layers[first_train_idx:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.model.norm.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p    = total_p - trainable_p
    opt_mem_gb  = trainable_p * 4 * 2 / 1e9

    logger.info(f"[Freeze] Total params    : {total_p / 1e9:.3f}B")
    logger.info(f"[Freeze] Trainable params: {trainable_p / 1e9:.3f}B  ({100 * trainable_p / total_p:.1f}%)")
    logger.info(f"[Freeze] Frozen params   : {frozen_p / 1e9:.3f}B  ({100 * frozen_p / total_p:.1f}%)")
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
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _cleanup_after_oom(optimizer: torch.optim.Optimizer, *tensors) -> None:
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


def _download_hub_file(hub_repo: str, filename: str) -> Optional[str]:
    """Download một file từ Hub, trả về local path hoặc None nếu không tồn tại."""
    try:
        local_path = hf_hub_download(
            repo_id=hub_repo, filename=filename, repo_type="model",
        )
        return local_path
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        logger.warning(f"[Hub] Lỗi download '{filename}': {e}")
        return None


def load_training_state(hub_repo: str) -> Optional[Dict]:
    """
    Load training_state.json từ Hub để xác định resume point.

    State có thể chứa:
      - completed_epochs      : list[int]  — các epoch đã hoàn thành
      - current_epoch         : int        — epoch đang chạy dở (nếu có mid-epoch save)
      - steps_done_in_epoch   : int        — số iteration đã hoàn thành trong current_epoch
      - global_step           : int
    """
    local_path = _download_hub_file(hub_repo, TRAINING_STATE_FILE)
    if local_path is None:
        logger.info("[Resume] Không tìm thấy state — bắt đầu từ đầu.")
        return None
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            f"[Resume] Tìm thấy {TRAINING_STATE_FILE}. "
            f"Completed epochs: {state.get('completed_epochs', [])}  "
            f"current_epoch: {state.get('current_epoch', None)}  "
            f"steps_done_in_epoch: {state.get('steps_done_in_epoch', 0)}  "
            f"global_step: {state.get('global_step', 0)}"
        )
        return state
    except Exception as e:
        logger.warning(f"[Resume] Lỗi parse state: {e} — bắt đầu từ đầu.")
        return None


def load_optimizer_states(
    hub_repo: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> List[Dict]:
    """
    Download và load optimizer / scheduler / scaler state dicts từ Hub.
    Trả về loss_log (list) nếu có, ngược lại trả về [].
    """
    # --- Optimizer ---
    opt_path = _download_hub_file(hub_repo, OPTIMIZER_STATE_FILE)
    if opt_path:
        try:
            opt_state = torch.load(opt_path, map_location=device)
            optimizer.load_state_dict(opt_state)
            logger.info("[Resume] ✓ Optimizer state loaded.")
        except Exception as e:
            logger.warning(f"[Resume] Không load được optimizer state: {e}")
    else:
        logger.warning("[Resume] Không tìm thấy optimizer_state.pt — optimizer fresh start.")

    # --- Scheduler ---
    sch_path = _download_hub_file(hub_repo, SCHEDULER_STATE_FILE)
    if sch_path:
        try:
            sch_state = torch.load(sch_path, map_location="cpu")
            scheduler.load_state_dict(sch_state)
            logger.info(
                f"[Resume] ✓ Scheduler state loaded. "
                f"last_epoch={scheduler.last_epoch}  "
                f"lr={scheduler.get_last_lr()}"
            )
        except Exception as e:
            logger.warning(f"[Resume] Không load được scheduler state: {e}")
    else:
        logger.warning("[Resume] Không tìm thấy scheduler_state.pt — scheduler fresh start.")

    # --- GradScaler (chỉ cần thiết khi dùng fp16) ---
    scl_path = _download_hub_file(hub_repo, SCALER_STATE_FILE)
    if scl_path:
        try:
            scl_state = torch.load(scl_path, map_location="cpu")
            scaler.load_state_dict(scl_state)
            logger.info(f"[Resume] ✓ GradScaler state loaded. scale={scaler.get_scale():.1f}")
        except Exception as e:
            logger.warning(f"[Resume] Không load được scaler state: {e}")

    # --- Loss log ---
    ll_path = _download_hub_file(hub_repo, LOSS_LOG_FILE)
    if ll_path:
        try:
            with open(ll_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[Resume] Không load được loss_log: {e}")
    return []


def _upload_file_to_hub(
    hub_repo: str,
    local_path: Path,
    repo_filename: str,
    commit_msg: str,
) -> None:
    try:
        HfApi().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=commit_msg,
        )
    except Exception as e:
        logger.error(f"[Hub] Upload '{repo_filename}' thất bại: {e}")


def save_and_push_checkpoint(
    hub_repo: str,
    output_dir: Path,
    state: Dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    loss_log: List[Dict],
    epoch: int,
    private: bool,
    commit_suffix: str = "",
) -> None:
    """
    Push model + tất cả state files lên Hub.

    commit_suffix: chuỗi phụ thêm vào commit message, vd "step-1500" hoặc "" (end-of-epoch).
    """
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo_exists(hub_repo, private)

    # 1. training_state.json
    state_path = output_dir / TRAINING_STATE_FILE
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    logger.info(f"[State] Saved → {state_path}")

    # 2. optimizer_state.pt
    opt_path = output_dir / OPTIMIZER_STATE_FILE
    torch.save(optimizer.state_dict(), opt_path)
    logger.info(f"[State] Optimizer state → {opt_path}  ({opt_path.stat().st_size / 1e6:.1f} MB)")

    # 3. scheduler_state.pt
    sch_path = output_dir / SCHEDULER_STATE_FILE
    torch.save(scheduler.state_dict(), sch_path)

    # 4. scaler_state.pt
    scl_path = output_dir / SCALER_STATE_FILE
    torch.save(scaler.state_dict(), scl_path)

    # 5. loss_log.json
    ll_path = output_dir / LOSS_LOG_FILE
    with open(ll_path, "w", encoding="utf-8") as f:
        json.dump(loss_log, f)

    # 6. Push model + tokenizer
    try:
        model.push_to_hub(hub_repo, commit_message=f"model {commit_base}", private=private)
        tokenizer.push_to_hub(hub_repo, commit_message=f"tokenizer {commit_base}", private=private)
        logger.info(f"[Hub] ✓ Model/tokenizer pushed — {commit_base}")
    except Exception as e:
        logger.error(f"[Hub] Model push thất bại: {e}")

    # 7. Upload aux files
    for local_f, repo_f in [
        (state_path, TRAINING_STATE_FILE),
        (opt_path,   OPTIMIZER_STATE_FILE),
        (sch_path,   SCHEDULER_STATE_FILE),
        (scl_path,   SCALER_STATE_FILE),
        (ll_path,    LOSS_LOG_FILE),
    ]:
        _upload_file_to_hub(hub_repo, local_f, repo_f, f"state {commit_base}")

    logger.info(f"[Hub] ✓ Tất cả state files pushed — {commit_base}")


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
    logger.info("[Plot] metrics_plot.png saved")


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
# Mid-epoch state builder (dùng chung cho save_iter và end-of-epoch)
# ---------------------------------------------------------------------------

def _build_training_state(
    args,
    hub_repo: str,
    completed_epochs: List[int],
    epoch_results: List[Dict],
    global_step: int,
    train_loader,
    current_epoch: int,
    steps_done_in_epoch: int,
    is_epoch_complete: bool,
) -> Dict:
    """
    Tạo dict state để dump ra training_state.json.

    - is_epoch_complete=True  → current_epoch / steps_done_in_epoch không cần resume
    - is_epoch_complete=False → mid-epoch snapshot; resume sẽ skip steps_done_in_epoch batches
    """
    state = {
        "model_name":           args.model_name,
        "hub_repo":             hub_repo,
        "total_epochs":         args.epochs,
        "completed_epochs":     completed_epochs,
        "freeze_layers":        args.freeze_layers,
        "batch_size":           args.batch_size,
        "sub_batch_sizes":      dict(train_loader.current_batch_sizes),
        "lr":                   args.lr,
        "seed":                 args.seed,
        "epoch_results":        epoch_results,
        "global_step":          global_step,
        # ── mid-epoch resume fields ──────────────────────────────────────
        "current_epoch":        current_epoch,
        "steps_done_in_epoch":  steps_done_in_epoch,
        "is_epoch_complete":    is_epoch_complete,
        # ────────────────────────────────────────────────────────────────
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


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

    # ── Resume: load training_state trước để biết có resume không ─────────────
    training_state = load_training_state(hub_repo)
    is_resuming    = training_state is not None

    # ── Các biến resume ────────────────────────────────────────────────────────
    if is_resuming:
        completed_epochs    = training_state.get("completed_epochs", [])
        epoch_results       = training_state.get("epoch_results", [])
        saved_batch_size    = training_state.get("batch_size", args.batch_size)
        saved_global_step   = training_state.get("global_step", 0)

        # Mid-epoch resume fields
        resume_epoch        = training_state.get("current_epoch", None)
        resume_skip_steps   = training_state.get("steps_done_in_epoch", 0)
        is_epoch_complete   = training_state.get("is_epoch_complete", True)

        # Nếu epoch đó đã complete rồi thì không cần mid-epoch resume
        if is_epoch_complete or resume_epoch is None:
            resume_epoch      = None
            resume_skip_steps = 0

        logger.info(
            f"[Resume] Completed: {completed_epochs} | "
            f"global_step: {saved_global_step} | "
            f"global_batch_size: {saved_batch_size}"
        )
        if resume_epoch is not None:
            logger.info(
                f"[Resume] Mid-epoch resume: epoch={resume_epoch}  "
                f"skip first {resume_skip_steps} batches"
            )

        tokenizer = AutoTokenizer.from_pretrained(hub_repo, use_fast=True)
        model     = AutoModelForCausalLM.from_pretrained(
            hub_repo, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        args.batch_size = saved_batch_size
    else:
        completed_epochs    = []
        epoch_results       = []
        saved_global_step   = 0
        resume_epoch        = None
        resume_skip_steps   = 0

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

    # ── Freeze layers ──────────────────────────────────────────────────────────
    if args.freeze_layers > 0:
        freeze_model_layers(model, n_train_layers=args.freeze_layers)
    else:
        logger.info("[Setup] freeze_layers=0 → full finetune")
        total_p = sum(p.numel() for p in model.parameters())
        logger.info(f"[Setup] Trainable: {total_p/1e9:.2f}B")

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("[Setup] Gradient checkpointing: ON (use_reentrant=False)")
    else:
        logger.info("[Setup] Gradient checkpointing: OFF")

    # ── DataLoader (Mixed Batch) ───────────────────────────────────────────────
    logger.info("[Setup] Building MixedTaskDataLoader...")
    train_loader = MixedTaskDataLoader(
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
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

    steps_per_epoch = len(train_loader)   # min-limited by shortest task
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_opt_steps * args.warmup_ratio))
    scheduler       = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(
        f"[Setup] steps/epoch≈{steps_per_epoch}  total≈{total_opt_steps}  warmup={warmup_steps}\n"
        f"        sub-batch sizes: "
        + ", ".join(f"{t}={train_loader.current_batch_sizes[t]}" for t in TASK_ORDER)
    )
    if args.save_iter > 0:
        logger.info(f"[Setup] save_iter={args.save_iter}  → mid-epoch push mỗi {args.save_iter} steps")

    # ── Resume optimizer/scheduler/scaler states ──────────────────────────────
    if is_resuming:
        loss_log = load_optimizer_states(hub_repo, optimizer, scheduler, scaler, device)
        if not isinstance(loss_log, list):
            loss_log = []
        logger.info(
            f"[Resume] Optimizer LR after load: "
            f"{[pg['lr'] for pg in optimizer.param_groups]}"
        )
    else:
        loss_log = []

    global_step = saved_global_step

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_file = open(
        output_dir / "train_log.jsonl",
        "a" if is_resuming else "w",
        encoding="utf-8",
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs",
                      unit="epoch", dynamic_ncols=True, colour="green")

    for epoch in epoch_pbar:
        # ── Skip completed epochs ─────────────────────────────────────────────
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} done — skip.")
            epoch_pbar.set_postfix(status="skipped")
            continue

        # ── Determine how many batches to skip for mid-epoch resume ───────────
        #   Chỉ skip khi đây đúng là epoch bị ngắt giữa chừng.
        skip_batches = resume_skip_steps if (epoch == resume_epoch) else 0
        if skip_batches > 0:
            logger.info(
                f"[Resume] Mid-epoch resume: epoch={epoch}, "
                f"fast-forwarding {skip_batches} batches..."
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}  (mixed: {' + '.join(TASK_ORDER)})")
        logger.info(f"{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_pbar = tqdm(
            desc=f"  Epoch {epoch} | mixed batch",
            total=len(train_loader),
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        running_loss    = 0.0
        running_count   = 0
        oom_skip_count  = 0
        # steps_in_epoch đếm số batch thực sự đã xử lý trong epoch này
        # (bao gồm cả OOM-skip; khớp với resume_skip_steps đã lưu)
        steps_in_epoch  = 0

        for batch in train_loader:
            task_list = batch["task"]

            # ── Fast-forward: skip batches đã hoàn thành trước khi resume ─────
            if steps_in_epoch < skip_batches:
                steps_in_epoch += 1
                batch_pbar.update(1)
                batch_pbar.set_postfix(status=f"fast-fwd {steps_in_epoch}/{skip_batches}")
                continue

            # ── OOM cooldown ──────────────────────────────────────────────────
            if oom_skip_count > 0:
                oom_skip_count -= 1
                steps_in_epoch += 1
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
                    raise
                _cleanup_after_oom(optimizer, outputs, loss, input_ids, attention_mask, labels)
                try:
                    new_global_bs  = train_loader.report_oom(task_list[0] if task_list else "mixed")
                    oom_skip_count = args.oom_skip_batches
                    logger.warning(
                        f"[OOM] global_bs→{new_global_bs}  "
                        f"sub-bs: "
                        + ", ".join(f"{t}={train_loader.current_batch_sizes[t]}" for t in TASK_ORDER)
                        + f"  skip={oom_skip_count}"
                    )
                    args.batch_size = new_global_bs
                except RuntimeError as re:
                    logger.error(f"[OOM] Không thể giảm batch_size thêm: {re}")
                    raise
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            # ── Optimizer step ────────────────────────────────────────────────
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

            global_step    += 1
            steps_in_epoch += 1
            avg_loss        = running_loss / running_count
            current_lr      = scheduler.get_last_lr()[0]
            running_loss    = running_count = 0

            # Count tasks in this batch for logging
            from collections import Counter as _Counter
            task_counts = dict(_Counter(task_list))

            log_entry = {
                "epoch":      epoch,
                "step":       global_step,
                "tasks":      task_counts,
                "batch_size": len(task_list),
                "loss":       round(avg_loss, 6),
                "lr":         current_lr,
                "grad_norm":  round(float(grad_norm), 4),
                "sub_bs":     dict(train_loader.current_batch_sizes),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            loss_log.append({"step": global_step, "loss": avg_loss, "lr": current_lr})

            batch_pbar.update(1)
            batch_pbar.set_postfix(
                bs=len(task_list),
                loss=f"{avg_loss:.4f}",
                lr=f"{current_lr:.2e}",
                grad=f"{float(grad_norm):.3f}",
            )
            epoch_pbar.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                step=global_step,
                loss=f"{avg_loss:.4f}",
            )

            # ── Mid-epoch checkpoint (save_iter) ──────────────────────────────
            if args.save_iter > 0 and global_step % args.save_iter == 0:
                logger.info(
                    f"\n[SaveIter] global_step={global_step}  "
                    f"(epoch={epoch}, steps_in_epoch={steps_in_epoch}) — pushing to Hub..."
                )
                mid_state = _build_training_state(
                    args=args,
                    hub_repo=hub_repo,
                    completed_epochs=completed_epochs,
                    epoch_results=epoch_results,
                    global_step=global_step,
                    train_loader=train_loader,
                    current_epoch=epoch,
                    steps_done_in_epoch=steps_in_epoch,  # số batch đã đi qua (kể cả skip)
                    is_epoch_complete=False,
                )
                save_and_push_checkpoint(
                    hub_repo=hub_repo,
                    output_dir=output_dir,
                    state=mid_state,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    loss_log=loss_log,
                    epoch=epoch,
                    private=args.hub_private,
                    commit_suffix=f"step-{global_step}",
                )
                model.train()   # đảm bảo model về train mode sau push
            # ─────────────────────────────────────────────────────────────────

        batch_pbar.close()
        logger.info(f"\n[Epoch {epoch}] Training done.  global_step={global_step}")

        # ── Eval ──────────────────────────────────────────────────────────────
        if not args.skip_eval:
            eval_res = run_evaluation(
                model, tokenizer, args, device, amp_dtype, use_amp, epoch, output_dir,
            )
            epoch_results.append(eval_res)
            plot_metrics(epoch_results, output_dir)
            plot_training_loss(loss_log, output_dir)

        # ── End-of-epoch Save & push ──────────────────────────────────────────
        completed_epochs.append(epoch)
        state = _build_training_state(
            args=args,
            hub_repo=hub_repo,
            completed_epochs=completed_epochs,
            epoch_results=epoch_results,
            global_step=global_step,
            train_loader=train_loader,
            current_epoch=epoch,
            steps_done_in_epoch=steps_in_epoch,
            is_epoch_complete=True,   # ← epoch hoàn thành, không cần mid-epoch resume
        )

        ckpt_path = output_dir / f"checkpoint_epoch{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        torch.save(optimizer.state_dict(), ckpt_path / OPTIMIZER_STATE_FILE)
        torch.save(scheduler.state_dict(), ckpt_path / SCHEDULER_STATE_FILE)
        torch.save(scaler.state_dict(),    ckpt_path / SCALER_STATE_FILE)
        logger.info(f"[Checkpoint] → {ckpt_path}")

        save_and_push_checkpoint(
            hub_repo=hub_repo,
            output_dir=output_dir,
            state=state,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_log=loss_log,
            epoch=epoch,
            private=args.hub_private,
            commit_suffix="",   # end-of-epoch: không có suffix
        )

    epoch_pbar.close()
    log_file.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Completed epochs  : {completed_epochs}")
    logger.info(f"  Trained layers    : last {args.freeze_layers} of {len(model.model.layers)}")
    logger.info(
        f"  Final sub-bs      : "
        + ", ".join(f"{t}={train_loader.current_batch_sizes[t]}" for t in TASK_ORDER)
    )
    logger.info(f"  Total steps       : {global_step}")

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
            "model":             args.model_name,
            "hub_repo":          hub_repo,
            "epochs":            args.epochs,
            "freeze_layers":     args.freeze_layers,
            "completed_epochs":  completed_epochs,
            "global_step":       global_step,
            "batch_size":        args.batch_size,
            "sub_batch_sizes":   dict(train_loader.current_batch_sizes),
            "lr":                args.lr,
            "epoch_results":     epoch_results,
        }, f, indent=2, ensure_ascii=False)

    plot_metrics(epoch_results, output_dir)
    plot_training_loss(loss_log, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Llama-3-8B — Last-N Layers Finetuning (Mixed Batch)")
    logger.info("=" * 60)
    logger.info(f"  model          : {args.model_name}")
    logger.info(f"  data_root      : {args.data_root}")
    logger.info(f"  output_dir     : {args.output_dir}")
    logger.info(f"  hub_repo       : {args.hub_repo}")
    logger.info(f"  epochs         : {args.epochs}")
    logger.info(f"  batch_size     : {args.batch_size}  "
                f"(mixed: ~{args.batch_size//3}+{args.batch_size//3}+{args.batch_size - 2*(args.batch_size//3)} per task)")
    logger.info(f"  freeze_layers  : {args.freeze_layers}  "
                f"(train last {args.freeze_layers}/32 layers + norm + lm_head)")
    logger.info(f"  grad_ckpt      : {args.gradient_checkpointing}")
    logger.info(f"  lr             : {args.lr}")
    logger.info(f"  bf16/fp16      : {args.bf16}/{args.fp16}")
    logger.info(f"  save_iter      : {args.save_iter if args.save_iter > 0 else 'disabled'}")
    logger.info("=" * 60)

    train(args)