"""
Llama3-8B-OT.py
================
Stage 2 — Cross-lingual Optimal Transport Alignment
cho meta-llama/Meta-Llama-3-8B-Instruct / ducanhdinh/Llama3-8B-Finetune.

Kiến trúc
---------
  Một backbone duy nhất (Llama-3-8B), hai forward pass bất đối xứng:

    [English branch]  s_en  → M(Θ)          [frozen, no_grad]
    [Target  branch]  s_tgt → M(Θ + ΔΘ_LoRA) [trainable]

  Với mỗi middle layer l ∈ L:
    1. Attention-weighted pooling (eq. 13) → normalize unit sphere (eq. 14)
    2. Cosine cost matrix C^(l) (eq. 16)
    3. Log-domain Sinkhorn (eq. 9-10, ε=0.1, T_iter=50)
    4. OT loss = <C^(l), T^(l)*>_F  (eq. 17)
  Tổng hợp:
    L_OT = Σ_l softmax(w)_l · OT^(l)  (eq. 18, w learnable)
    L     = L_LM + λ · L_OT            (eq. 19, default λ=0.5)

  Chỉ ΔΘ (LoRA adapters tại middle layers) và w được train.
  Dominant branch M(·; Θ) hoàn toàn frozen.

LoRA placement
--------------
  LoRA attach vào {q_proj, k_proj, v_proj, o_proj} của các middle layers.
  Mặc định middle_layers = layers[8..23] (Llama-3-8B có 32 layers).

Data
----
  OPUS-100 : sampling theo --opus_ratio (mặc định 5%)
  FLORES-200: full (ratio=1.0)
  Eng-Eng pairs: thêm vào theo --eng_eng_ratio (mặc định 0.0 = tắt)
  Dùng AlignmentDataLoader từ dataloader/alignment_dataloader.py

Resume & Checkpoint
-------------------
  Push lên HF Hub: LoRA adapter + optimizer + scheduler + scaler +
                   training_state.json + layer_weights.pt
  Sau khi push xong → XÓA local checkpoint files (tiết kiệm disk).
  Resume: tự download từ Hub.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python Llama3-8B-OT.py \\
        --base_model    ducanhdinh/Llama3-8B-Finetune \\
        --data_root     ../raw_data/alignment/ \\
        --output_dir    ./ot_checkpoints \\
        --hub_repo      Llama3-8B-OT \\
        --epochs        3 \\
        --batch_size    8 \\
        --lr            2e-5 \\
        --lambda_ot     0.5 \\
        --sinkhorn_eps  0.1 \\
        --opus_ratio    0.05 \\
        --eng_eng_ratio 0.30 \\
        --save_iter     200
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from peft import LoraConfig, get_peft_model, PeftModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dataloader"))

from alignment_dataloader import (
    AlignmentDataset,
    AlignmentDataLoader,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAINING_STATE_FILE  = "ot_training_state.json"
OPTIMIZER_STATE_FILE = "ot_optimizer_state.pt"
SCHEDULER_STATE_FILE = "ot_scheduler_state.pt"
SCALER_STATE_FILE    = "ot_scaler_state.pt"
LAYER_WEIGHTS_FILE   = "ot_layer_weights.pt"
LORA_ADAPTER_DIR     = "lora_adapter"
LOSS_LOG_FILE        = "ot_loss_log.json"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-3-8B OT Alignment (Stage 2)")

    p.add_argument("--base_model",      type=str, default="ducanhdinh/Llama3-8B-Finetune")
    p.add_argument("--data_root",       type=str, default="../raw_data/alignment/")
    p.add_argument("--output_dir",      type=str, default="./ot_checkpoints")
    p.add_argument("--hub_repo",        type=str, default="Llama3-8B-OT")
    p.add_argument("--hub_private",     action="store_true")

    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    # OT hyper-params
    p.add_argument("--lambda_ot",       type=float, default=0.5,
                   help="Trọng số OT loss: L = L_LM + lambda_ot * L_OT")
    p.add_argument("--sinkhorn_eps",    type=float, default=0.1,
                   help="Entropy regularisation ε cho Sinkhorn (eq. 6)")
    p.add_argument("--sinkhorn_iters",  type=int,   default=50,
                   help="Số vòng lặp Sinkhorn–Knopp")

    # Middle layers: comma-separated hoặc dùng auto (default)
    p.add_argument(
        "--middle_layers", type=str, default="auto",
        help=(
            "Các middle layers để tính OT. "
            "'auto' → lấy layers[8..23] (Llama-3-8B 32 layers). "
            "Hoặc truyền comma-separated list: '8,12,16,20'"
        ),
    )

    # LoRA config
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    # ── Data sampling ────────────────────────────────────────────────────────
    # --opus_ratio   : tỷ lệ dữ liệu OPUS-100 được dùng, mặc định 5%
    #                  tương ứng với opus_sample_ratio trong AlignmentDataset
    # --eng_eng_ratio: tỷ lệ cặp eng-eng thêm vào so với joint records
    #                  mặc định 0.0 = tắt hoàn toàn
    p.add_argument(
        "--opus_ratio",
        type=float,
        default=0.05,
        help="Fraction của OPUS-100 để dùng, trong (0.0, 1.0] (mặc định 5%%)",
    )
    p.add_argument(
        "--eng_eng_ratio",
        type=float,
        default=0.0,
        help=(
            "Tỷ lệ cặp eng-eng identity thêm vào so với joint records, "
            "trong [0.0, 1.0]. 0.0 = tắt (mặc định). "
            "Ví dụ: 0.30 → thêm eng-eng = 30%% số joint records."
        ),
    )

    # Mid-epoch save
    p.add_argument("--save_iter",       type=int, default=0,
                   help="Push checkpoint lên Hub mỗi N global steps (0 = tắt)")

    # misc
    p.add_argument("--max_length",      type=int, default=256)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--bf16",            action="store_true", default=True)
    p.add_argument("--fp16",            action="store_true")
    p.add_argument("--oom_skip_batches",type=int, default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Parse middle layers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_middle_layers(spec: str, n_total: int = 32) -> List[int]:
    """
    Trả về list các layer indices (0-indexed) sẽ được dùng cho OT.

    'auto'         → layers[8..23]  (16 layers giữa của Llama-3-8B 32-layer)
    '8,12,16,20'   → [8, 12, 16, 20]
    '8:24'         → list(range(8, 24))
    """
    if spec == "auto":
        start = n_total // 4
        end   = n_total * 3 // 4
        layers = list(range(start, end))
    elif ":" in spec:
        a, b = spec.split(":")
        layers = list(range(int(a), int(b)))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    logger.info(f"[OT] Middle layers ({len(layers)}): {layers}")
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Log-domain Sinkhorn  (differentiable, eq. 9–10)
# ─────────────────────────────────────────────────────────────────────────────

def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.1,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Log-domain Sinkhorn–Knopp (eq. 9–10).

    Parameters
    ----------
    a : [m]  source weights (uniform = 1/m)
    b : [n]  target weights (uniform = 1/n)
    C : [m, n] cost matrix (cosine distance, values in [0, 2])
    eps   : entropy regularisation ε
    max_iter : Sinkhorn iterations

    Returns
    -------
    T : [m, n] approximate transport plan
    """
    m, n   = C.shape
    log_a  = torch.log(a + 1e-8)   # [m]
    log_b  = torch.log(b + 1e-8)   # [n]

    f = torch.zeros(m, dtype=C.dtype, device=C.device)  # Kantorovich potentials
    g = torch.zeros(n, dtype=C.dtype, device=C.device)

    for _ in range(max_iter):
        # eq. 9:  f_i = ε log a_i - ε log Σ_j exp((g_j - C_ij)/ε)
        log_sum = torch.logsumexp((g.unsqueeze(0) - C) / eps, dim=1)  # [m]
        f = eps * log_a - eps * log_sum

        # eq. 10:  g_j = ε log b_j - ε log Σ_i exp((f_i - C_ij)/ε)
        log_sum = torch.logsumexp((f.unsqueeze(1) - C) / eps, dim=0)  # [n]
        g = eps * log_b - eps * log_sum

    # T_ij = exp((f_i + g_j - C_ij) / ε)
    log_T = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
    T = torch.exp(log_T)
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Attention-weighted pooling + L2 normalisation  (eq. 13–14)
# ─────────────────────────────────────────────────────────────────────────────

def attention_pooled_representations(
    hidden: torch.Tensor,          # [B, s, d]
    attn_weights: torch.Tensor,    # [B, H, s, s]  raw attentions at this layer
    attention_mask: torch.Tensor,  # [B, s]
) -> torch.Tensor:
    """
    Returns H̃^(l) ∈ R^{B×s×d} where each row is α_i · h_i,
    then each row normalized to unit sphere  (eq. 13–14).

    α^(l) = mean over heads of last-token row of attention matrix.
    We use mean over heads of the attention distribution (averaged over
    query positions that are not padding).
    """
    # attn_weights: [B, H, s, s] → mean over heads → [B, s, s]
    alpha_matrix = attn_weights.mean(dim=1)              # [B, s, s]

    # Mean over query positions (masked)
    mask_float = attention_mask.float()                  # [B, s]
    # weight only non-pad query positions
    alpha = (alpha_matrix * mask_float.unsqueeze(-1)).sum(dim=1)   # [B, s]
    alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)       # [B, s] normalise

    # H̃_i = α_i · h_i
    pooled = alpha.unsqueeze(-1) * hidden                # [B, s, d]

    # Row-wise L2 normalise
    pooled = F.normalize(pooled, p=2, dim=-1)            # [B, s, d]
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer OT loss  (eq. 16–17)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_loss_layer(
    h_tgt: torch.Tensor,           # [s_tgt, d]  (single sample, unit-sphere rows)
    h_dom: torch.Tensor,           # [s_dom, d]
    eps: float,
    max_iter: int,
) -> torch.Tensor:
    """
    Cosine distance cost → Sinkhorn → OT loss scalar.
    """
    # Cosine distance: C_ij = 1 - <h_tgt_i, h_dom_j>  (eq. 16)
    # h_tgt, h_dom already unit-sphere → dot product = cosine similarity
    C = 1.0 - torch.mm(h_tgt, h_dom.t())  # [s_tgt, s_dom]
    C = C.clamp(0.0, 2.0)

    s_tgt = h_tgt.size(0)
    s_dom = h_dom.size(0)
    a = torch.full((s_tgt,), 1.0 / s_tgt, dtype=h_tgt.dtype, device=h_tgt.device)
    b = torch.full((s_dom,), 1.0 / s_dom, dtype=h_dom.dtype, device=h_dom.device)

    T = sinkhorn_log(a, b, C, eps=eps, max_iter=max_iter)  # [s_tgt, s_dom]

    # OT loss = <C, T>_F  (eq. 17)
    ot_loss = (C * T).sum()
    return ot_loss


# ─────────────────────────────────────────────────────────────────────────────
# LayerWeights module  (learnable w, eq. 18)
# ─────────────────────────────────────────────────────────────────────────────

class LayerWeights(nn.Module):
    """
    Learnable scalar weights w = (w_1, …, w_|L|).
    L_OT = Σ_l softmax(w)_l · OT^(l)  (eq. 18)
    """
    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, ot_losses: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.w, dim=0)             # [|L|]
        stacked = torch.stack(ot_losses)               # [|L|]
        return (weights * stacked).sum()


# ─────────────────────────────────────────────────────────────────────────────
# L_LM — causal LM loss on target branch (eq. 20)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# OOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _cleanup_oom(optimizer, *tensors):
    for t in tensors:
        try: del t
        except Exception: pass
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# HF Hub helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_repo(hub_repo: str) -> str:
    if "/" not in hub_repo:
        username = "ducanhdinh"
        hub_repo = f"{username}/{hub_repo}"
        logger.info(f"[Hub] Full repo_id: {hub_repo}")
    return hub_repo


def _ensure_repo(hub_repo: str, private: bool) -> None:
    api = HfApi()
    try:
        api.repo_info(repo_id=hub_repo, repo_type="model")
    except RepositoryNotFoundError:
        logger.info(f"[Hub] Tạo repo: {hub_repo}")
        api.create_repo(repo_id=hub_repo, repo_type="model",
                        private=private, exist_ok=True)


def _download_hub_file(hub_repo: str, filename: str) -> Optional[str]:
    try:
        return hf_hub_download(repo_id=hub_repo, filename=filename,
                                repo_type="model")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        logger.warning(f"[Hub] Không download được '{filename}': {e}")
        return None


def _upload_file(hub_repo: str, local_path: Path, repo_filename: str,
                  commit_msg: str) -> None:
    try:
        HfApi().upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=commit_msg,
        )
        logger.info(f"[Hub] ✓ Uploaded '{repo_filename}'")
    except Exception as e:
        logger.error(f"[Hub] Upload '{repo_filename}' thất bại: {e}")


def _upload_folder(hub_repo: str, local_dir: Path, repo_subfolder: str,
                    commit_msg: str) -> None:
    """Upload toàn bộ folder lên Hub (dùng cho LoRA adapter)."""
    try:
        HfApi().upload_folder(
            folder_path=str(local_dir),
            path_in_repo=repo_subfolder,
            repo_id=hub_repo,
            repo_type="model",
            commit_message=commit_msg,
        )
        logger.info(f"[Hub] ✓ Uploaded folder '{repo_subfolder}'")
    except Exception as e:
        logger.error(f"[Hub] Upload folder '{repo_subfolder}' thất bại: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Training state I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_training_state(hub_repo: str) -> Optional[Dict]:
    local = _download_hub_file(hub_repo, TRAINING_STATE_FILE)
    if local is None:
        logger.info("[Resume] Không tìm thấy state — bắt đầu từ đầu.")
        return None
    try:
        with open(local, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            f"[Resume] Tìm thấy {TRAINING_STATE_FILE}. "
            f"Completed epochs: {state.get('completed_epochs', [])}  "
            f"current_epoch: {state.get('current_epoch')}  "
            f"steps_done: {state.get('steps_done_in_epoch', 0)}  "
            f"global_step: {state.get('global_step', 0)}"
        )
        return state
    except Exception as e:
        logger.warning(f"[Resume] Lỗi parse state: {e}")
        return None


def load_optimizer_states(
    hub_repo: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    layer_weights_module: LayerWeights,
    device: torch.device,
) -> List[Dict]:
    """Download và load optimizer/scheduler/scaler/layer_weights từ Hub."""

    opt_path = _download_hub_file(hub_repo, OPTIMIZER_STATE_FILE)
    if opt_path:
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            logger.info("[Resume] ✓ Optimizer state loaded.")
        except Exception as e:
            logger.warning(f"[Resume] Optimizer load lỗi: {e}")

    sch_path = _download_hub_file(hub_repo, SCHEDULER_STATE_FILE)
    if sch_path:
        try:
            scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
            logger.info(f"[Resume] ✓ Scheduler loaded. last_epoch={scheduler.last_epoch}")
        except Exception as e:
            logger.warning(f"[Resume] Scheduler load lỗi: {e}")

    scl_path = _download_hub_file(hub_repo, SCALER_STATE_FILE)
    if scl_path:
        try:
            scaler.load_state_dict(torch.load(scl_path, map_location="cpu"))
            logger.info(f"[Resume] ✓ Scaler loaded. scale={scaler.get_scale():.1f}")
        except Exception as e:
            logger.warning(f"[Resume] Scaler load lỗi: {e}")

    lw_path = _download_hub_file(hub_repo, LAYER_WEIGHTS_FILE)
    if lw_path:
        try:
            lw_state = torch.load(lw_path, map_location=device)
            layer_weights_module.load_state_dict(lw_state)
            logger.info(f"[Resume] ✓ LayerWeights loaded: {layer_weights_module.w.data.tolist()}")
        except Exception as e:
            logger.warning(f"[Resume] LayerWeights load lỗi: {e}")

    ll_path = _download_hub_file(hub_repo, LOSS_LOG_FILE)
    if ll_path:
        try:
            with open(ll_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[Resume] loss_log load lỗi: {e}")
    return []


def save_and_push_checkpoint(
    hub_repo: str,
    output_dir: Path,
    state: Dict,
    model,                          # PEFT model
    tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    layer_weights_module: LayerWeights,
    loss_log: List[Dict],
    epoch: int,
    private: bool,
    commit_suffix: str = "",
    delete_local: bool = True,
) -> None:
    """
    Push LoRA adapter + tất cả state files lên Hub.
    Sau khi push xong → XÓA local checkpoint files nếu delete_local=True.
    """
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo(hub_repo, private)

    # 1. Save LoRA adapter
    lora_dir = output_dir / LORA_ADAPTER_DIR
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    # 2. Save aux state files
    state_path = output_dir / TRAINING_STATE_FILE
    opt_path   = output_dir / OPTIMIZER_STATE_FILE
    sch_path   = output_dir / SCHEDULER_STATE_FILE
    scl_path   = output_dir / SCALER_STATE_FILE
    lw_path    = output_dir / LAYER_WEIGHTS_FILE
    ll_path    = output_dir / LOSS_LOG_FILE

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    torch.save(optimizer.state_dict(), opt_path)
    torch.save(scheduler.state_dict(), sch_path)
    torch.save(scaler.state_dict(), scl_path)
    torch.save(layer_weights_module.state_dict(), lw_path)
    with open(ll_path, "w", encoding="utf-8") as f:
        json.dump(loss_log, f)

    logger.info(f"[Hub] Đang push checkpoint: {commit_base} ...")

    # 3. Upload LoRA adapter folder
    _upload_folder(hub_repo, lora_dir, LORA_ADAPTER_DIR, f"lora {commit_base}")

    # 4. Upload aux files
    for local_f, repo_f in [
        (state_path, TRAINING_STATE_FILE),
        (opt_path,   OPTIMIZER_STATE_FILE),
        (sch_path,   SCHEDULER_STATE_FILE),
        (scl_path,   SCALER_STATE_FILE),
        (lw_path,    LAYER_WEIGHTS_FILE),
        (ll_path,    LOSS_LOG_FILE),
    ]:
        _upload_file(hub_repo, local_f, repo_f, f"state {commit_base}")

    logger.info(f"[Hub] ✓ Checkpoint pushed — {commit_base}")

    # 5. XÓA local checkpoint files để tiết kiệm disk
    if delete_local:
        _delete_local_checkpoint(output_dir, lora_dir,
                                  [state_path, opt_path, sch_path,
                                   scl_path, lw_path, ll_path])


def _delete_local_checkpoint(
    output_dir: Path,
    lora_dir: Path,
    aux_files: List[Path],
) -> None:
    """Xóa local checkpoint files sau khi đã push lên Hub."""
    # Xóa LoRA adapter folder
    if lora_dir.exists():
        shutil.rmtree(lora_dir)
        logger.info(f"[Cleanup] Đã xóa local LoRA adapter: {lora_dir}")

    # Xóa aux state files
    for f in aux_files:
        if f.exists():
            f.unlink()
            logger.info(f"[Cleanup] Đã xóa local file: {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Resume LoRA model từ Hub
# ─────────────────────────────────────────────────────────────────────────────

def load_lora_from_hub(hub_repo: str, base_model_name: str,
                        dtype, device_map) -> Optional[object]:
    """
    Download LoRA adapter từ Hub và load vào base model.
    Trả về PEFT model hoặc None nếu không tìm thấy.
    """
    api = HfApi()
    try:
        # Kiểm tra xem LORA_ADAPTER_DIR có trên Hub không
        files = api.list_repo_files(repo_id=hub_repo, repo_type="model")
        adapter_config_file = f"{LORA_ADAPTER_DIR}/adapter_config.json"
        if adapter_config_file not in files:
            logger.info(f"[Resume] Không tìm thấy LoRA adapter trên Hub.")
            return None
    except RepositoryNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"[Resume] Lỗi kiểm tra Hub files: {e}")
        return None

    logger.info(f"[Resume] Loading base model từ: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        output_attentions=True,  # cần cho OT pooling
    )

    logger.info(f"[Resume] Loading LoRA adapter từ Hub: {hub_repo}/{LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(
        base_model,
        hub_repo,
        subfolder=LORA_ADAPTER_DIR,
        is_trainable=True,
    )
    logger.info("[Resume] ✓ LoRA model loaded từ Hub.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build LoRA model từ đầu
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(
    base_model_name: str,
    middle_layers: List[int],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype,
    device_map,
) -> object:
    """
    Load base model rồi attach LoRA adapters chỉ vào middle layers.
    Base model được freeze hoàn toàn; chỉ LoRA params được train.
    """
    logger.info(f"[Setup] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        output_attentions=True,
    )

    # Target modules: chỉ các middle layers
    target_modules = []
    for l in middle_layers:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            target_modules.append(f"model.layers.{l}.self_attn.{proj}")

    logger.info(f"[LoRA] Target modules ({len(target_modules)}): {target_modules[:4]}...")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[LoRA] Trainable: {trainable/1e6:.2f}M / {total/1e9:.3f}B "
                f"({100*trainable/total:.2f}%)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Extract hidden states + attentions cho một forward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward_and_extract(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers: List[int],
    use_lora: bool,
    amp_dtype,
    use_amp: bool,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Một forward pass qua model.

    Parameters
    ----------
    use_lora : True → training branch (LoRA active, get logits)
               False → frozen branch (no LoRA, no_grad)

    Returns
    -------
    logits  : [B, s, V] hoặc None (nếu frozen branch)
    layer_data : {l: (hidden [B, s, d], attn [B, H, s, s])}
    """
    ctx = torch.no_grad() if not use_lora else torch.enable_grad()

    with ctx:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )

    # out.hidden_states: tuple of (n_layers+1) tensors [B, s, d]
    # hidden_states[0] = embedding, hidden_states[l+1] = layer l output
    # out.attentions:    tuple of n_layers tensors [B, H, s, s]
    layer_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]   # layer l output, [B, s, d]
        attn   = out.attentions[l]          # [B, H, s, s]
        layer_data[l] = (hidden, attn)

    logits = out.logits if use_lora else None
    return logits, layer_data


# ─────────────────────────────────────────────────────────────────────────────
# Compute full L = L_LM + lambda * L_OT for a batch
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    model,
    batch: Dict,
    middle_layers: List[int],
    layer_weights_module: LayerWeights,
    lambda_ot: float,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
    amp_dtype,
    use_amp: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float]:
    """
    Returns (total_loss, lm_loss_item, ot_loss_item).
    """
    tgt_ids  = batch["tgt_input_ids"].to(device)
    tgt_mask = batch["tgt_attention_mask"].to(device)
    tgt_lbl  = batch["tgt_labels"].to(device)
    en_ids   = batch["en_input_ids"].to(device)
    en_mask  = batch["en_attention_mask"].to(device)

    # ── Target branch: forward với LoRA ──────────────────────────────────────
    logits, tgt_layer_data = forward_and_extract(
        model, tgt_ids, tgt_mask, middle_layers,
        use_lora=True, amp_dtype=amp_dtype, use_amp=use_amp, device=device,
    )

    # L_LM (eq. 20)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        loss_lm = compute_lm_loss(logits, tgt_lbl)

    # ── English branch: forward frozen (no LoRA) ──────────────────────────────
    # Disable LoRA adapters tạm thời cho pass này
    model.disable_adapter_layers()
    _, en_layer_data = forward_and_extract(
        model, en_ids, en_mask, middle_layers,
        use_lora=False, amp_dtype=amp_dtype, use_amp=use_amp, device=device,
    )
    model.enable_adapter_layers()

    # ── Compute L_OT across middle layers ────────────────────────────────────
    B = tgt_ids.size(0)
    ot_losses_per_layer: List[torch.Tensor] = []

    for l in middle_layers:
        h_tgt_batch, attn_tgt = tgt_layer_data[l]   # [B, s_tgt, d], [B, H, s_tgt, s_tgt]
        h_en_batch,  attn_en  = en_layer_data[l]    # [B, s_en,  d], [B, H, s_en,  s_en]

        # Attention-weighted pooling + normalize (eq. 13-14)
        pooled_tgt = attention_pooled_representations(h_tgt_batch, attn_tgt, tgt_mask)  # [B, s_tgt, d]
        pooled_en  = attention_pooled_representations(h_en_batch,  attn_en,  en_mask)   # [B, s_en,  d]

        # Per-sample OT loss, averaged over batch
        batch_ot = []
        for b in range(B):
            # Get non-pad tokens
            tgt_len = tgt_mask[b].sum().item()
            en_len  = en_mask[b].sum().item()
            if tgt_len == 0 or en_len == 0:
                continue

            h_t = pooled_tgt[b, :int(tgt_len), :]  # [s_tgt, d]
            h_e = pooled_en[b,  :int(en_len),  :]  # [s_en,  d]

            ot_l = compute_ot_loss_layer(
                h_t.float(), h_e.float(),   # cast to float32 cho Sinkhorn stability
                eps=sinkhorn_eps, max_iter=sinkhorn_iters,
            )
            batch_ot.append(ot_l)

        if batch_ot:
            layer_ot_mean = torch.stack(batch_ot).mean()
            ot_losses_per_layer.append(layer_ot_mean)

    # Aggregated multi-layer OT loss (eq. 18)
    if ot_losses_per_layer:
        loss_ot = layer_weights_module(ot_losses_per_layer)
    else:
        loss_ot = torch.tensor(0.0, device=device, requires_grad=True)

    # L = L_LM + λ * L_OT  (eq. 19)
    loss = loss_lm + lambda_ot * loss_ot

    return loss, loss_lm.item(), loss_ot.item()


# ─────────────────────────────────────────────────────────────────────────────
# State builder
# ─────────────────────────────────────────────────────────────────────────────

def build_state(
    args,
    hub_repo: str,
    completed_epochs: List[int],
    global_step: int,
    current_epoch: int,
    steps_done_in_epoch: int,
    is_epoch_complete: bool,
    loss_log: List[Dict],
    middle_layers: List[int],
) -> Dict:
    return {
        "base_model":           args.base_model,
        "hub_repo":             hub_repo,
        "total_epochs":         args.epochs,
        "completed_epochs":     completed_epochs,
        "global_step":          global_step,
        "current_epoch":        current_epoch,
        "steps_done_in_epoch":  steps_done_in_epoch,
        "is_epoch_complete":    is_epoch_complete,
        "middle_layers":        middle_layers,
        "lambda_ot":            args.lambda_ot,
        "sinkhorn_eps":         args.sinkhorn_eps,
        "lora_r":               args.lora_r,
        "lora_alpha":           args.lora_alpha,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        # ── data sampling params ─────────────────────────────────────────────
        "opus_ratio":           args.opus_ratio,
        "eng_eng_ratio":        args.eng_eng_ratio,
        # ────────────────────────────────────────────────────────────────────
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss(loss_log: List[Dict], output_dir: Path) -> None:
    if not loss_log:
        return
    steps   = [e["step"]     for e in loss_log]
    losses  = [e["loss"]     for e in loss_log]
    lm_l    = [e["lm_loss"]  for e in loss_log]
    ot_l    = [e["ot_loss"]  for e in loss_log]
    lrs     = [e["lr"]       for e in loss_log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("OT Alignment Training Progress", fontsize=14, fontweight="bold")

    for ax, vals, title, color in [
        (axes[0, 0], losses, "Total Loss",  "steelblue"),
        (axes[0, 1], lm_l,  "L_LM",        "darkorange"),
        (axes[1, 0], ot_l,  "L_OT",        "forestgreen"),
        (axes[1, 1], lrs,   "Learning Rate","crimson"),
    ]:
        ax.plot(steps, vals, color=color, linewidth=0.8, alpha=0.7)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        if title == "Learning Rate":
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output_dir / "ot_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[Plot] ot_training_loss.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

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

    device_map = "auto" if torch.cuda.is_available() else None

    # ── HF login ─────────────────────────────────────────────────────────────
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("[Hub] Logged in")
    else:
        logger.warning("[Hub] HF_TOKEN không set")

    hub_repo = _resolve_repo(args.hub_repo)

    # ── Middle layers ─────────────────────────────────────────────────────────
    middle_layers = resolve_middle_layers(args.middle_layers, n_total=32)

    # ── Resume: check Hub ─────────────────────────────────────────────────────
    training_state = load_training_state(hub_repo)
    is_resuming    = training_state is not None

    if is_resuming:
        completed_epochs  = training_state.get("completed_epochs", [])
        saved_global_step = training_state.get("global_step", 0)
        resume_epoch      = training_state.get("current_epoch", None)
        resume_skip_steps = training_state.get("steps_done_in_epoch", 0)
        is_epoch_complete = training_state.get("is_epoch_complete", True)

        if is_epoch_complete or resume_epoch is None:
            resume_epoch = None; resume_skip_steps = 0

        logger.info(
            f"[Resume] Completed: {completed_epochs} | "
            f"global_step: {saved_global_step}"
        )

        # Load LoRA model từ Hub
        model = load_lora_from_hub(hub_repo, args.base_model, dtype, device_map)
        if model is None:
            logger.warning("[Resume] Không tìm thấy LoRA checkpoint → build mới")
            model = build_lora_model(
                args.base_model, middle_layers,
                args.lora_r, args.lora_alpha, args.lora_dropout,
                dtype, device_map,
            )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    else:
        completed_epochs  = []
        saved_global_step = 0
        resume_epoch      = None
        resume_skip_steps = 0

        model = build_lora_model(
            args.base_model, middle_layers,
            args.lora_r, args.lora_alpha, args.lora_dropout,
            dtype, device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        model = model.to(device)

    # ── Gradient checkpointing ────────────────────────────────────────────────
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("[Setup] Gradient checkpointing: ON")

    # ── Learnable layer weights (eq. 18) ──────────────────────────────────────
    layer_weights_module = LayerWeights(n_layers=len(middle_layers)).to(device)

    # ── Alignment dataset ─────────────────────────────────────────────────────
    # Dùng args.opus_ratio  → truyền vào opus_sample_ratio của AlignmentDataset
    # Dùng args.eng_eng_ratio → truyền vào eng_eng_ratio của AlignmentDataset
    logger.info(
        f"[Data] Loading alignment dataset  "
        f"(opus_ratio={args.opus_ratio:.1%}, eng_eng_ratio={args.eng_eng_ratio:.1%}) ..."
    )
    align_dataset = AlignmentDataset(
        alignment_data_path=args.data_root,
        opus_sample_ratio=args.opus_ratio,
        eng_eng_ratio=args.eng_eng_ratio,
    ).load()
    align_dataset.stats()

    train_loader = AlignmentDataLoader(
        dataset=align_dataset,
        split="train",
        source="joint",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"[Data] Train batches: {len(train_loader):,}")

    # ── Optimizer: LoRA params + layer_weights ────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    lora_params_no_decay = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
    ]
    lora_params_decay = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
    ]
    param_groups = [
        {"params": lora_params_no_decay, "weight_decay": args.weight_decay},
        {"params": lora_params_decay,    "weight_decay": 0.0},
        {"params": list(layer_weights_module.parameters()), "weight_decay": 0.0,
         "lr": args.lr * 10},  # layer weights có thể học nhanh hơn
    ]
    total_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    logger.info(f"[Optim] Trainable params in optimizer: {total_trainable/1e6:.2f}M")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95))

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))
    scheduler       = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(
        f"[Setup] steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}\n"
        f"        middle_layers={len(middle_layers)}  lambda_ot={args.lambda_ot}  "
        f"eps={args.sinkhorn_eps}  sinkhorn_iters={args.sinkhorn_iters}"
    )

    # ── Resume aux states ─────────────────────────────────────────────────────
    if is_resuming:
        loss_log = load_optimizer_states(
            hub_repo, optimizer, scheduler, scaler, layer_weights_module, device,
        )
    else:
        loss_log = []

    global_step = saved_global_step

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file = open(
        output_dir / "ot_train_log.jsonl",
        "a" if is_resuming else "w",
        encoding="utf-8",
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_pbar = tqdm(
        range(1, args.epochs + 1), desc="Epochs",
        unit="epoch", dynamic_ncols=True, colour="green",
    )

    for epoch in epoch_pbar:
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} done — skip.")
            continue

        skip_batches = resume_skip_steps if (epoch == resume_epoch) else 0
        if skip_batches > 0:
            logger.info(f"[Resume] Epoch {epoch}: fast-forward {skip_batches} batches...")

        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}/{args.epochs}  — OT Alignment")
        logger.info(f"{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        layer_weights_module.train()
        optimizer.zero_grad(set_to_none=True)

        batch_pbar = tqdm(
            desc=f"  Epoch {epoch} | OT align",
            total=len(train_loader),
            unit="batch", dynamic_ncols=True, leave=False,
        )

        running_loss  = 0.0
        running_lm    = 0.0
        running_ot    = 0.0
        running_count = 0
        oom_skip_count = 0
        steps_in_epoch = 0

        for batch in train_loader:
            # ── Fast-forward ──────────────────────────────────────────────────
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

            # ── Forward + Loss ────────────────────────────────────────────────
            loss = None
            try:
                loss, lm_item, ot_item = compute_total_loss(
                    model=model,
                    batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    lambda_ot=args.lambda_ot,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    device=device,
                )

                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss  += loss.item()
                running_lm    += lm_item
                running_ot    += ot_item
                running_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                _cleanup_oom(optimizer, loss)
                oom_skip_count = args.oom_skip_batches
                # Reduce batch size
                new_bs = max(1, args.batch_size // 2)
                if new_bs < args.batch_size:
                    logger.warning(f"[OOM] batch_size {args.batch_size} → {new_bs}")
                    args.batch_size = new_bs
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            # ── Optimizer step ────────────────────────────────────────────────
            if use_amp and amp_dtype == torch.float16:
                scaler.unscale_(optimizer)

            # Gradient clipping — only trainable params
            all_trainable = (
                [p for p in model.parameters() if p.requires_grad]
                + list(layer_weights_module.parameters())
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.max_grad_norm)

            if use_amp and amp_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step    += 1
            steps_in_epoch += 1
            avg_loss  = running_loss  / running_count
            avg_lm    = running_lm    / running_count
            avg_ot    = running_ot    / running_count
            current_lr = scheduler.get_last_lr()[0]
            running_loss = running_lm = running_ot = running_count = 0

            log_entry = {
                "epoch":     epoch,
                "step":      global_step,
                "loss":      round(avg_loss, 6),
                "lm_loss":   round(avg_lm,   6),
                "ot_loss":   round(avg_ot,   6),
                "lr":        current_lr,
                "grad_norm": round(float(grad_norm), 4),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            loss_log.append(log_entry)

            batch_pbar.update(1)
            batch_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lm=f"{avg_lm:.4f}",
                ot=f"{avg_ot:.4f}",
                lr=f"{current_lr:.2e}",
            )
            epoch_pbar.set_postfix(
                epoch=f"{epoch}/{args.epochs}",
                step=global_step,
                loss=f"{avg_loss:.4f}",
            )

            # ── Mid-epoch checkpoint ──────────────────────────────────────────
            if args.save_iter > 0 and global_step % args.save_iter == 0:
                logger.info(
                    f"\n[SaveIter] step={global_step} (epoch={epoch}, "
                    f"steps_in_epoch={steps_in_epoch}) — pushing..."
                )
                mid_state = build_state(
                    args, hub_repo, completed_epochs, global_step,
                    current_epoch=epoch,
                    steps_done_in_epoch=steps_in_epoch,
                    is_epoch_complete=False,
                    loss_log=loss_log,
                    middle_layers=middle_layers,
                )
                save_and_push_checkpoint(
                    hub_repo=hub_repo, output_dir=output_dir,
                    state=mid_state, model=model, tokenizer=tokenizer,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    layer_weights_module=layer_weights_module,
                    loss_log=loss_log, epoch=epoch,
                    private=args.hub_private,
                    commit_suffix=f"step-{global_step}",
                    delete_local=True,
                )
                model.train()
                layer_weights_module.train()
            # ─────────────────────────────────────────────────────────────────

        batch_pbar.close()
        logger.info(f"\n[Epoch {epoch}] Done. global_step={global_step}")

        # Log layer weights at end of epoch
        lw_softmax = F.softmax(layer_weights_module.w.detach(), dim=0)
        top_k = min(5, len(middle_layers))
        top_indices = lw_softmax.topk(top_k).indices.tolist()
        logger.info(
            f"[Epoch {epoch}] Layer weights (top-{top_k}): "
            + ", ".join(f"layer{middle_layers[i]}={lw_softmax[i].item():.4f}"
                        for i in top_indices)
        )

        # ── End-of-epoch checkpoint ───────────────────────────────────────────
        completed_epochs.append(epoch)
        state = build_state(
            args, hub_repo, completed_epochs, global_step,
            current_epoch=epoch,
            steps_done_in_epoch=steps_in_epoch,
            is_epoch_complete=True,
            loss_log=loss_log,
            middle_layers=middle_layers,
        )
        save_and_push_checkpoint(
            hub_repo=hub_repo, output_dir=output_dir,
            state=state, model=model, tokenizer=tokenizer,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            layer_weights_module=layer_weights_module,
            loss_log=loss_log, epoch=epoch,
            private=args.hub_private,
            commit_suffix="",
            delete_local=True,
        )

        plot_training_loss(loss_log, output_dir)

    epoch_pbar.close()
    log_file.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("OT ALIGNMENT TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"  Completed epochs : {completed_epochs}")
    logger.info(f"  Total steps      : {global_step}")
    logger.info(f"  Middle layers    : {middle_layers}")
    lw_final = F.softmax(layer_weights_module.w.detach(), dim=0).tolist()
    logger.info(f"  Final layer weights (softmax): {[round(v, 4) for v in lw_final]}")
    logger.info(f"  Hub repo         : {hub_repo}/{LORA_ADAPTER_DIR}")

    with open(output_dir / "ot_final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model":        args.base_model,
            "hub_repo":          hub_repo,
            "epochs":            args.epochs,
            "completed_epochs":  completed_epochs,
            "global_step":       global_step,
            "middle_layers":     middle_layers,
            "lambda_ot":         args.lambda_ot,
            "sinkhorn_eps":      args.sinkhorn_eps,
            "lora_r":            args.lora_r,
            "lora_alpha":        args.lora_alpha,
            # ── data sampling params ─────────────────────────────────────────
            "opus_ratio":        args.opus_ratio,
            "eng_eng_ratio":     args.eng_eng_ratio,
            # ────────────────────────────────────────────────────────────────
            "final_layer_weights": lw_final,
        }, f, indent=2, ensure_ascii=False)

    plot_training_loss(loss_log, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 65)
    logger.info("Llama-3-8B — Stage 2: Optimal Transport Alignment")
    logger.info("=" * 65)
    logger.info(f"  base_model       : {args.base_model}")
    logger.info(f"  data_root        : {args.data_root}")
    logger.info(f"  output_dir       : {args.output_dir}")
    logger.info(f"  hub_repo         : {args.hub_repo}")
    logger.info(f"  epochs           : {args.epochs}")
    logger.info(f"  batch_size       : {args.batch_size}")
    logger.info(f"  lr               : {args.lr}")
    logger.info(f"  lambda_ot        : {args.lambda_ot}")
    logger.info(f"  sinkhorn_eps     : {args.sinkhorn_eps}")
    logger.info(f"  sinkhorn_iters   : {args.sinkhorn_iters}")
    logger.info(f"  middle_layers    : {args.middle_layers}")
    logger.info(f"  lora_r / alpha   : {args.lora_r} / {args.lora_alpha}")
    logger.info(f"  opus_ratio       : {args.opus_ratio:.1%}")
    logger.info(f"  eng_eng_ratio    : {args.eng_eng_ratio:.1%}")
    logger.info(f"  save_iter        : {args.save_iter if args.save_iter > 0 else 'disabled'}")
    logger.info(f"  bf16/fp16        : {args.bf16}/{args.fp16}")
    logger.info("=" * 65)

    train(args)