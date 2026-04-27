"""
Llama3-8B-OT-eng-reg.py
========================
Stage 2 — Cross-lingual Optimal Transport Alignment
Distributed Data Parallel (DDP) version

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS FORMULATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three forward passes per step:

  Pass 1: EN  → LoRA OFF (frozen)  → h_en_frozen   [inference_mode]
  Pass 2: EN  → LoRA ON  (grad)    → h_en_lora      [gradient flows]
  Pass 3: TGT → LoRA ON  (grad)    → h_tgt_lora     [gradient flows]

Three losses:

  L_LM  = CausalLM CrossEntropy on target branch (Pass 3 logits)
  L_OT  = Sinkhorn OT cost between h_tgt_lora and h_en_frozen   (eqs. 13-18)
  L_Reg = L2 per-token between h_en_lora and h_en_frozen         (paper eq. 2)
          = λ Σ_t Σ_j ||h_en_lora(j,t) - h_en_frozen(j,t)||²₂

  L = L_LM + lambda_ot * L_OT + lambda_reg * L_Reg

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUTED CHANGES (inherited from OT-only-distributed.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[DIST-1]  --gpus N validated against torch.cuda.device_count()
[DIST-2]  Launch via torchrun --nproc_per_node=N
[DIST-3]  --batch_size is GLOBAL; per-GPU = batch_size // world_size
[DIST-4]  OOM coordinated across all GPUs via all_reduce MAX
[DIST-5]  DDP wraps LoRA model; LayerWeights manually all-reduced
[DIST-6]  Frozen EN branch runs under inference_mode on every rank
[DIST-7]  DistributedSortedSampler shards data across ranks
[DIST-8]  Only rank-0 saves/uploads checkpoints
[DIST-9]  Only rank-0 logs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAUNCH COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single node, 2 GPUs:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc_per_node=2 Llama3-8B-OT-eng-reg.py \\
        --gpus 2 \\
        --base_model  ducanhdinh/Llama3-8B-Finetune \\
        --data_root   ../raw_data/alignment/ \\
        --output_dir  ./ot_eng_reg_checkpoints \\
        --hub_repo    Llama3-8B-OT-EngReg \\
        --epochs      3 \\
        --batch_size  32 \\
        --grad_accum  4 \\
        --lr          2e-5 \\
        --lambda_ot   1.0 \\
        --lambda_reg  1.0 \\
        --sinkhorn_eps 0.1 \\
        --opus_ratio  0.05 \\
        --max_length  256 \\
        --save_iter   200
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dataloader"))
from alignment_dataloader import AlignmentDataset, AlignmentDataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> Tuple[int, int, int]:
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int, device: torch.device = None) -> None:
    if world_size > 1:
        if device is not None:
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return tensor


def all_reduce_max(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def reduce_gradients_manual(module: nn.Module, world_size: int) -> None:
    if world_size <= 1:
        return
    for param in module.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


# ─────────────────────────────────────────────────────────────────────────────
# Logger — rank-0 only
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_base_logger = logging.getLogger(__name__)


class RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


def setup_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger(f"ot_eng_reg.rank{rank}")
    logger.addFilter(RankFilter(rank))
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

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
    p = argparse.ArgumentParser(description="Llama-3-8B OT + EN Reg — Distributed")

    p.add_argument("--gpus",            type=int,   default=None)
    p.add_argument("--base_model",      type=str,   default="ducanhdinh/Llama3-8B-Finetune")
    p.add_argument("--data_root",       type=str,   default="../raw_data/alignment/")
    p.add_argument("--output_dir",      type=str,   default="./ot_eng_reg_checkpoints")
    p.add_argument("--hub_repo",        type=str,   default="Llama3-8B-OT-EngReg")
    p.add_argument("--hub_private",     action="store_true")

    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=32,
                   help="GLOBAL batch size. Divided equally across GPUs.")
    p.add_argument("--grad_accum",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    # Loss weights
    p.add_argument("--lambda_ot",       type=float, default=1.0,
                   help="Weight for OT alignment loss.")
    p.add_argument("--lambda_reg",      type=float, default=1.0,
                   help="Weight for EN L2 regularization loss.")

    # Sinkhorn
    p.add_argument("--sinkhorn_eps",    type=float, default=0.1)
    p.add_argument("--sinkhorn_iters",  type=int,   default=50)

    p.add_argument("--middle_layers",   type=str,   default="auto",
                   help="'auto' → layers[N/4 .. 3N/4). Or comma list: '8,12,16,20'")

    # LoRA
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    p.add_argument("--opus_ratio",      type=float, default=0.05)
    p.add_argument("--eng_eng_ratio",   type=float, default=0.0)

    p.add_argument("--save_iter",       type=int,   default=0)
    p.add_argument("--seq_length",      type=int,   default=None)
    p.add_argument("--max_length",      type=int,   default=256)

    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--bf16",            action="store_true", default=True)
    p.add_argument("--fp16",            action="store_true")
    p.add_argument("--oom_skip_batches",type=int,   default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")
    p.add_argument(
        "--mean_layer",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        metavar="BOOL",
    )
    p.add_argument("--plot_smooth",     type=int,   default=20)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GPU validation  [DIST-1]
# ─────────────────────────────────────────────────────────────────────────────

def validate_gpus(requested: Optional[int], world_size: int, rank: int) -> int:
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("[GPU] No CUDA GPUs detected.")
    if requested is not None:
        if requested > available:
            raise RuntimeError(
                f"[GPU] --gpus {requested} requested but only {available} available."
            )
        if requested != world_size:
            raise RuntimeError(
                f"[GPU] --gpus {requested} does not match torchrun world_size={world_size}."
            )
    if rank == 0:
        _base_logger.info(f"[GPU] Detected {available} GPU(s) | Using {world_size} GPU(s)")
    return world_size


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU batch size  [DIST-3]
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_gpu_batch(global_batch: int, world_size: int, rank: int) -> int:
    per_gpu = global_batch // world_size
    if per_gpu == 0:
        raise ValueError(f"batch_size={global_batch} < world_size={world_size}.")
    remainder = global_batch % world_size
    if remainder != 0 and rank == 0:
        _base_logger.warning(
            f"[Batch] batch_size={global_batch} not divisible by {world_size}. "
            f"Using per_gpu={per_gpu} → effective global={per_gpu * world_size}."
        )
    return per_gpu


# ─────────────────────────────────────────────────────────────────────────────
# OOM coordinator  [DIST-4]
# ─────────────────────────────────────────────────────────────────────────────

class OOMCoordinator:
    def __init__(self, per_gpu_batch: int, world_size: int, device: torch.device,
                 min_batch: int = 1):
        self.per_gpu_batch = per_gpu_batch
        self.world_size    = world_size
        self.device        = device
        self.min_batch     = min_batch

    def report_oom(self) -> None:
        self._flag().fill_(1)

    def sync_and_check(self) -> bool:
        flag = self._flag()
        all_reduce_max(flag, self.world_size)
        hit = flag.item() > 0
        flag.zero_()
        return bool(hit)

    def halve_batch(self, logger) -> int:
        old = self.per_gpu_batch
        self.per_gpu_batch = max(self.min_batch, old // 2)
        if logger:
            logger.warning(
                f"[OOM] per_gpu_batch {old} → {self.per_gpu_batch} "
                f"(global: {old * self.world_size} → {self.per_gpu_batch * self.world_size})"
            )
        return self.per_gpu_batch

    def _flag(self) -> torch.Tensor:
        if not hasattr(self, "_flag_tensor"):
            self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=self.device)
        return self._flag_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int, rank: int = 0) -> None:
    import random
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Middle layers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_middle_layers(spec: str, n_total: int = 32) -> List[int]:
    if spec == "auto":
        start = n_total // 4
        end   = (n_total * 3) // 4
        return list(range(start, end))
    elif ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    else:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Sinkhorn (batched, GPU-vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log_batched_inner(
    C: torch.Tensor,
    eps: float,
    max_iter: int,
) -> torch.Tensor:
    B, m, n = C.shape
    device  = C.device
    dtype   = C.dtype

    log_a = torch.full((B, m), -math.log(m), dtype=dtype, device=device)
    log_b = torch.full((B, n), -math.log(n), dtype=dtype, device=device)
    f = torch.zeros(B, m, dtype=dtype, device=device)
    g = torch.zeros(B, n, dtype=dtype, device=device)

    for _ in range(max_iter):
        log_sum_f = torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=2)
        f = eps * log_a - eps * log_sum_f
        log_sum_g = torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=1)
        g = eps * log_b - eps * log_sum_g

    log_T = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps
    return torch.exp(log_T)


def sinkhorn_log_batched(C: torch.Tensor, eps: float, max_iter: int) -> torch.Tensor:
    return _sinkhorn_log_batched_inner(C.float(), eps, max_iter)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-weighted pooling
# ─────────────────────────────────────────────────────────────────────────────

def attention_weighted_pool_from_attn(
    hidden: torch.Tensor,
    attn_weights: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    alpha = attn_weights.float().mean(dim=1).mean(dim=1)
    alpha = alpha * attention_mask.float()
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)
    h_tilde = hidden * alpha.unsqueeze(-1)
    return F.normalize(h_tilde, p=2, dim=-1)


def attention_weighted_pool_uniform(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask   = attention_mask.float()
    n_real = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    alpha  = mask / n_real
    h_tilde = hidden * alpha.unsqueeze(-1)
    return F.normalize(h_tilde, p=2, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer OT loss  (eqs. 16-17)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_loss_single_layer(
    h_tgt: torch.Tensor,
    h_en:  torch.Tensor,
    tgt_mask: torch.Tensor,
    en_mask:  torch.Tensor,
    eps:      float,
    max_iter: int,
) -> torch.Tensor:
    C = 1.0 - torch.bmm(h_tgt.float(), h_en.float().transpose(1, 2))
    C = C.clamp(0.0, 2.0)
    valid    = tgt_mask.float().unsqueeze(2) * en_mask.float().unsqueeze(1)
    C_masked = C * valid + (1.0 - valid) * 2.0
    T = sinkhorn_log_batched(C_masked, eps=eps, max_iter=max_iter)
    ot_per_sample = (C * T * valid).sum(dim=(1, 2))
    return ot_per_sample.mean()


# ─────────────────────────────────────────────────────────────────────────────
# LayerWeights  (eq. 18)
# ─────────────────────────────────────────────────────────────────────────────

class LayerWeights(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, ot_losses: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.w, dim=0)
        stacked = torch.stack(ot_losses)
        return (weights * stacked).sum()


# ─────────────────────────────────────────────────────────────────────────────
# LM loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EN L2 regularization loss  (paper eq. 2)
# Per-token L2 between EN-LoRA-ON and EN-LoRA-OFF hidden states
# ─────────────────────────────────────────────────────────────────────────────

def compute_en_reg_loss(
    en_lora_layer_data:   Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    en_frozen_layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    en_mask:              torch.Tensor,
    middle_layers:        List[int],
    mean_layer:           bool,
    layer_weights_module: LayerWeights,
) -> torch.Tensor:
    """
    L_Reg = mean_over_layers( mean_over_batch( sum_over_valid_tokens( ||h_lora - h_frozen||^2 ) ) )

    This is the per-token L2 regularization from Alqahtani et al. (2021) eq. 2,
    adapted to work across middle layers.
    """
    reg_losses: List[torch.Tensor] = []
    mask = en_mask.float()  # [B, s]

    for l in middle_layers:
        h_lora,   _ = en_lora_layer_data[l]    # [B, s, d]
        h_frozen, _ = en_frozen_layer_data[l]  # [B, s, d]  — detached (inference_mode)

        # Per-token squared L2 distance
        diff_sq = ((h_lora.float() - h_frozen.float()) ** 2).sum(dim=-1)  # [B, s]

        # Mask out padding tokens
        diff_sq_masked = diff_sq * mask  # [B, s]

        # Sum over valid tokens, mean over batch
        n_valid = mask.sum(dim=1).clamp(min=1.0)  # [B]
        reg_per_sample = diff_sq_masked.sum(dim=1) / n_valid  # [B]
        reg_losses.append(reg_per_sample.mean())

    if not reg_losses:
        return torch.tensor(0.0, device=en_mask.device)

    if mean_layer:
        return torch.stack(reg_losses).mean()
    else:
        return layer_weights_module(reg_losses)


# ─────────────────────────────────────────────────────────────────────────────
# Forward passes
# ─────────────────────────────────────────────────────────────────────────────

def forward_lora_on(
    model,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers:  List[int],
    amp_dtype:      torch.dtype,
    use_amp:        bool,
    device:         torch.device,
    output_logits:  bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    """
    Forward pass with LoRA ON (trainable branch).
    Returns (logits or None, layer_data).
    Used for both EN-LoRA and TGT-LoRA branches.
    """
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    has_attentions = out.attentions is not None and len(out.attentions) > 0

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        attn   = out.attentions[l] if has_attentions else None
        layer_data[l] = (hidden, attn)

    logits = out.logits if output_logits else None
    return logits, layer_data


@torch.inference_mode()
def forward_lora_off(
    model,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers:  List[int],
    amp_dtype:      torch.dtype,
    use_amp:        bool,
    device:         torch.device,
) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Forward pass with LoRA OFF (frozen EN branch).  [DIST-6]
    Runs under inference_mode — no gradient, no DDP sync needed.
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.disable_adapter_layers()
    try:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = raw_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                output_attentions=False,  # not needed for frozen branch
                use_cache=False,
            )
    finally:
        raw_model.enable_adapter_layers()

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        layer_data[l] = (hidden, None)

    return layer_data


# ─────────────────────────────────────────────────────────────────────────────
# Total loss
# L = L_LM + lambda_ot * L_OT + lambda_reg * L_Reg
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    model,
    batch:                  dict,
    middle_layers:          List[int],
    layer_weights_module:   LayerWeights,
    lambda_ot:              float,
    lambda_reg:             float,
    sinkhorn_eps:           float,
    sinkhorn_iters:         int,
    mean_layer:             bool,
    amp_dtype:              torch.dtype,
    use_amp:                bool,
    device:                 torch.device,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Three forward passes:
      Pass 1: EN  → LoRA OFF → h_en_frozen   (inference_mode, no grad)
      Pass 2: EN  → LoRA ON  → h_en_lora     (gradient flows for L_Reg)
      Pass 3: TGT → LoRA ON  → h_tgt + logits (gradient flows for L_LM + L_OT)

    Returns: (total_loss, lm_item, ot_item, reg_item)
    """
    tgt_ids  = batch["tgt_input_ids"].to(device, non_blocking=True)
    tgt_mask = batch["tgt_attention_mask"].to(device, non_blocking=True)
    tgt_lbl  = batch["tgt_labels"].to(device, non_blocking=True)
    en_ids   = batch["en_input_ids"].to(device, non_blocking=True)
    en_mask  = batch["en_attention_mask"].to(device, non_blocking=True)

    # ── Pass 1: EN → LoRA OFF  ───────────────────────────────────────────────
    en_frozen_layer_data = forward_lora_off(
        model, en_ids, en_mask, middle_layers, amp_dtype, use_amp, device,
    )

    # ── Pass 2: EN → LoRA ON  ────────────────────────────────────────────────
    _, en_lora_layer_data = forward_lora_on(
        model, en_ids, en_mask, middle_layers, amp_dtype, use_amp, device,
        output_logits=False,
    )

    # ── Pass 3: TGT → LoRA ON  ───────────────────────────────────────────────
    logits, tgt_layer_data = forward_lora_on(
        model, tgt_ids, tgt_mask, middle_layers, amp_dtype, use_amp, device,
        output_logits=True,
    )

    # ── L_LM  ────────────────────────────────────────────────────────────────
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        loss_lm = compute_lm_loss(logits, tgt_lbl)

    # ── L_OT: OT between h_tgt_lora and h_en_frozen  ─────────────────────────
    ot_losses: List[torch.Tensor] = []
    for l in middle_layers:
        tgt_hidden, tgt_attn = tgt_layer_data[l]
        en_hidden,  _        = en_frozen_layer_data[l]

        if tgt_attn is not None:
            h_tgt = attention_weighted_pool_from_attn(tgt_hidden, tgt_attn, tgt_mask)
        else:
            h_tgt = attention_weighted_pool_uniform(tgt_hidden, tgt_mask)

        h_en = attention_weighted_pool_uniform(en_hidden, en_mask)

        ot_l = compute_ot_loss_single_layer(
            h_tgt, h_en, tgt_mask, en_mask,
            eps=sinkhorn_eps, max_iter=sinkhorn_iters,
        )
        ot_losses.append(ot_l)

    if ot_losses:
        if mean_layer:
            loss_ot = torch.stack(ot_losses).mean()
        else:
            loss_ot = layer_weights_module(ot_losses)
    else:
        loss_ot = torch.tensor(0.0, device=device, requires_grad=True)

    # ── L_Reg: per-token L2 between h_en_lora and h_en_frozen  ───────────────
    loss_reg = compute_en_reg_loss(
        en_lora_layer_data=en_lora_layer_data,
        en_frozen_layer_data=en_frozen_layer_data,
        en_mask=en_mask,
        middle_layers=middle_layers,
        mean_layer=mean_layer,
        layer_weights_module=layer_weights_module,
    )

    # ── Total  ────────────────────────────────────────────────────────────────
    loss = loss_lm + lambda_ot * loss_ot + lambda_reg * loss_reg

    return loss, loss_lm.item(), loss_ot.item(), loss_reg.item()


# ─────────────────────────────────────────────────────────────────────────────
# OOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def _cleanup_after_oom(optimizer, *tensors):
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder  [DIST-7]
# ─────────────────────────────────────────────────────────────────────────────

def build_distributed_dataloader(
    align_dataset,
    tokenizer,
    per_gpu_batch: int,
    max_length:    int,
    seed:          int,
    num_workers:   int,
    rank:          int,
    world_size:    int,
    epoch:         int = 0,
):
    loader = AlignmentDataLoader(
        dataset=align_dataset,
        split="train",
        source="joint",
        tokenizer=tokenizer,
        batch_size=per_gpu_batch,
        max_length=max_length,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        rank=rank,
        world_size=world_size,
    )
    if hasattr(loader, "sampler") and isinstance(
        getattr(loader, "sampler", None), DistributedSampler
    ):
        loader.sampler.set_epoch(epoch)
    elif hasattr(loader, "dataloader") and hasattr(loader.dataloader, "sampler"):
        if isinstance(loader.dataloader.sampler, DistributedSampler):
            loader.dataloader.sampler.set_epoch(epoch)
    return loader


def rebuild_dataloader(
    align_dataset, tokenizer, per_gpu_batch, max_length,
    seed, num_workers, rank, world_size, epoch, logger,
):
    if logger:
        logger.info(
            f"[DataLoader] Rebuilding with per_gpu_batch={per_gpu_batch} "
            f"(global={per_gpu_batch * world_size})"
        )
    return build_distributed_dataloader(
        align_dataset, tokenizer, per_gpu_batch, max_length,
        seed, num_workers, rank, world_size, epoch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss plotting — 4 panels + 3 raw losses
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 2:
        return values
    arr    = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    padded = np.concatenate([arr[:window - 1], arr])
    return np.convolve(padded, kernel, mode="valid").tolist()


def plot_training_loss(loss_log: List[Dict], output_dir: Path, smooth: int = 20,
                       lambda_ot: float = 1.0, lambda_reg: float = 1.0) -> None:
    if not loss_log:
        return

    steps      = [e["step"]      for e in loss_log]
    total_loss = [e["loss"]      for e in loss_log]
    lm_loss    = [e["lm_loss"]   for e in loss_log]
    ot_raw     = [e["ot_loss"]   for e in loss_log]
    reg_raw    = [e["reg_loss"]  for e in loss_log]
    lam_ot     = [v * lambda_ot  for v in ot_raw]
    lam_reg    = [v * lambda_reg for v in reg_raw]

    panels = [
        # (row, col), values, title, ylabel, colour
        ((0, 0), total_loss, "Total Loss  L = L_LM + λ_OT·L_OT + λ_Reg·L_Reg", "loss",    "#2f7fc1"),
        ((0, 1), lm_loss,    "L_LM  (causal LM loss)",                           "loss",    "#e07b39"),
        ((1, 0), lam_ot,     f"λ_OT·L_OT  (λ={lambda_ot})",                      "λ·loss",  "#3aaa6b"),
        ((1, 1), lam_reg,    f"λ_Reg·L_Reg  (λ={lambda_reg})",                   "λ·loss",  "#9b59b6"),
        ((2, 0), ot_raw,     "L_OT raw  (OT alignment loss)",                    "loss",    "#27ae60"),
        ((2, 1), reg_raw,    "L_Reg raw  (EN L2 regularization)",                "loss",    "#8e44ad"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        f"OT + EN-Reg Alignment — Training Progress  (step {steps[-1]})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for (r, c), vals, title, ylabel, colour in panels:
        ax = axes[r][c]
        ax.plot(steps, vals, color=colour, alpha=0.20, linewidth=0.7, label="raw")
        smoothed = _rolling_mean(vals, smooth)
        ax.plot(steps, smoothed, color=colour, linewidth=1.8, label=f"smooth (w={smooth})")
        ax.scatter([steps[-1]], [smoothed[-1]], color=colour, s=30, zorder=5)
        ax.annotate(
            f"{smoothed[-1]:.4f}",
            xy=(steps[-1], smoothed[-1]), xytext=(5, 4),
            textcoords="offset points", fontsize=7, color=colour,
        )
        ax.set_title(title, fontsize=10, fontweight="semibold")
        ax.set_xlabel("Optimizer step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper right")

        prev_ep = None
        for entry in loss_log:
            ep = entry.get("epoch")
            if ep != prev_ep and prev_ep is not None:
                ax.axvline(x=entry["step"], color="gray",
                           linestyle="--", linewidth=0.7, alpha=0.5)
            prev_ep = ep

    plt.tight_layout()
    out_path = output_dir / "ot_training_loss.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HF Hub helpers  [DIST-8]
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_repo(hub_repo: str) -> str:
    if "/" not in hub_repo:
        hub_repo = f"ducanhdinh/{hub_repo}"
    return hub_repo


def _ensure_repo(hub_repo: str, private: bool) -> None:
    api = HfApi()
    try:
        api.repo_info(repo_id=hub_repo, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=hub_repo, repo_type="model",
                        private=private, exist_ok=True)


def _download_hub_file(hub_repo: str, filename: str) -> Optional[str]:
    try:
        return hf_hub_download(repo_id=hub_repo, filename=filename, repo_type="model")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        _base_logger.warning(f"[Hub] Cannot download '{filename}': {e}")
        return None


def _upload_file(hub_repo, local_path, repo_filename, commit_msg):
    try:
        HfApi().upload_file(
            path_or_fileobj=str(local_path), path_in_repo=repo_filename,
            repo_id=hub_repo, repo_type="model", commit_message=commit_msg,
        )
    except Exception as e:
        _base_logger.error(f"[Hub] Upload '{repo_filename}' failed: {e}")


def _upload_folder(hub_repo, local_dir, repo_subfolder, commit_msg):
    try:
        HfApi().upload_folder(
            folder_path=str(local_dir), path_in_repo=repo_subfolder,
            repo_id=hub_repo, repo_type="model", commit_message=commit_msg,
        )
    except Exception as e:
        _base_logger.error(f"[Hub] Upload folder '{repo_subfolder}' failed: {e}")


def load_training_state(hub_repo: str) -> Optional[Dict]:
    local = _download_hub_file(hub_repo, TRAINING_STATE_FILE)
    if local is None:
        return None
    try:
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _base_logger.warning(f"[Resume] Parse error: {e}")
        return None


def load_optimizer_states(hub_repo, optimizer, scheduler, scaler,
                           layer_weights_module, mean_layer, device) -> List[Dict]:
    loaders = [
        (OPTIMIZER_STATE_FILE, lambda p: optimizer.load_state_dict(
            torch.load(p, map_location=device))),
        (SCHEDULER_STATE_FILE, lambda p: scheduler.load_state_dict(
            torch.load(p, map_location="cpu"))),
        (SCALER_STATE_FILE,    lambda p: scaler.load_state_dict(
            torch.load(p, map_location="cpu"))),
    ]
    if not mean_layer:
        loaders.append(
            (LAYER_WEIGHTS_FILE, lambda p: layer_weights_module.load_state_dict(
                torch.load(p, map_location=device)))
        )
    for fname, loader in loaders:
        path = _download_hub_file(hub_repo, fname)
        if path:
            try:
                loader(path)
            except Exception as e:
                _base_logger.warning(f"[Resume] {fname} error: {e}")

    ll_path = _download_hub_file(hub_repo, LOSS_LOG_FILE)
    if ll_path:
        try:
            with open(ll_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_and_push_checkpoint(
    hub_repo, output_dir, state, model, tokenizer,
    optimizer, scheduler, scaler, layer_weights_module,
    loss_log, epoch, private, smooth, mean_layer,
    lambda_ot, lambda_reg,
    commit_suffix="", delete_local=True,
):
    """Only called from rank-0.  [DIST-8]"""
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo(hub_repo, private)

    raw_model = model.module if hasattr(model, "module") else model

    lora_dir = output_dir / LORA_ADAPTER_DIR
    lora_dir.mkdir(parents=True, exist_ok=True)
    raw_model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    state_path = output_dir / TRAINING_STATE_FILE
    opt_path   = output_dir / OPTIMIZER_STATE_FILE
    sch_path   = output_dir / SCHEDULER_STATE_FILE
    scl_path   = output_dir / SCALER_STATE_FILE
    lw_path    = output_dir / LAYER_WEIGHTS_FILE
    ll_path    = output_dir / LOSS_LOG_FILE
    plot_path  = output_dir / "ot_training_loss.png"

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    torch.save(optimizer.state_dict(), opt_path)
    torch.save(scheduler.state_dict(), sch_path)
    torch.save(scaler.state_dict(),    scl_path)
    if not mean_layer:
        torch.save(layer_weights_module.state_dict(), lw_path)
    with open(ll_path, "w", encoding="utf-8") as f:
        json.dump(loss_log, f)

    plot_training_loss(loss_log, output_dir, smooth=smooth,
                       lambda_ot=lambda_ot, lambda_reg=lambda_reg)

    _upload_folder(hub_repo, lora_dir, LORA_ADAPTER_DIR, f"lora {commit_base}")

    files_to_upload = [
        (state_path, TRAINING_STATE_FILE),
        (opt_path,   OPTIMIZER_STATE_FILE),
        (sch_path,   SCHEDULER_STATE_FILE),
        (scl_path,   SCALER_STATE_FILE),
        (ll_path,    LOSS_LOG_FILE),
        (plot_path,  "ot_training_loss.png"),
    ]
    if not mean_layer and lw_path.exists():
        files_to_upload.append((lw_path, LAYER_WEIGHTS_FILE))

    for local_f, repo_f in files_to_upload:
        if local_f.exists():
            _upload_file(hub_repo, local_f, repo_f, f"state {commit_base}")

    _base_logger.info(f"[Hub] ✓ Checkpoint pushed — {commit_base}")

    if delete_local:
        if lora_dir.exists():
            shutil.rmtree(lora_dir)
        cleanup = [state_path, opt_path, sch_path, scl_path, ll_path]
        if not mean_layer:
            cleanup.append(lw_path)
        for f in cleanup:
            if f.exists():
                f.unlink()


def load_lora_from_hub(hub_repo, base_model_name, dtype, device_map):
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=hub_repo, repo_type="model")
        if f"{LORA_ADAPTER_DIR}/adapter_config.json" not in files:
            return None
    except Exception:
        return None
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        base_model, hub_repo, subfolder=LORA_ADAPTER_DIR, is_trainable=True,
    )
    return model


def build_lora_model(base_model_name, middle_layers, lora_r, lora_alpha,
                     lora_dropout, dtype, device_map):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True, attn_implementation="eager",
    )
    target_modules = [
        f"model.layers.{l}.self_attn.{proj}"
        for l in middle_layers
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    _base_logger.info(
        f"[LoRA] Trainable: {trainable/1e6:.2f}M / {total/1e9:.3f}B "
        f"({100*trainable/total:.2f}%)"
    )
    return model


def build_state(args, hub_repo, completed_epochs, global_step,
                current_epoch, steps_done_in_epoch, is_epoch_complete,
                loss_log, middle_layers, world_size, per_gpu_batch):
    return {
        "base_model":         args.base_model,
        "hub_repo":           hub_repo,
        "total_epochs":       args.epochs,
        "completed_epochs":   completed_epochs,
        "global_step":        global_step,
        "current_epoch":      current_epoch,
        "steps_done_in_epoch": steps_done_in_epoch,
        "is_epoch_complete":  is_epoch_complete,
        "middle_layers":      middle_layers,
        "lambda_ot":          args.lambda_ot,
        "lambda_reg":         args.lambda_reg,
        "sinkhorn_eps":       args.sinkhorn_eps,
        "lora_r":             args.lora_r,
        "lora_alpha":         args.lora_alpha,
        "batch_size_global":  args.batch_size,
        "per_gpu_batch":      per_gpu_batch,
        "world_size":         world_size,
        "grad_accum":         args.grad_accum,
        "lr":                 args.lr,
        "opus_ratio":         args.opus_ratio,
        "eng_eng_ratio":      args.eng_eng_ratio,
        "seq_length":         args.max_length,
        "mean_layer":         args.mean_layer,
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    # ── Distributed init ─────────────────────────────────────────────────────
    rank, local_rank, world_size = init_distributed()
    logger = setup_logger(rank)
    main   = is_main_process(rank)

    validate_gpus(args.gpus, world_size, rank)
    set_seed(args.seed, rank)

    if args.seq_length is not None:
        args.max_length = args.seq_length

    output_dir = Path(args.output_dir)
    if main:
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    barrier(world_size, device)

    # ── Precision ────────────────────────────────────────────────────────────
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16; amp_dtype = torch.bfloat16; use_amp = True
        logger.info("[Setup] bfloat16 AMP")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16; amp_dtype = torch.float16; use_amp = True
        logger.info("[Setup] float16 AMP")
    else:
        dtype = torch.float32; amp_dtype = torch.float32; use_amp = False

    per_gpu_batch = compute_per_gpu_batch(args.batch_size, world_size, rank)
    logger.info(
        f"[Batch] global={args.batch_size} | per_gpu={per_gpu_batch} | world_size={world_size}"
    )

    device_map = None

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    hub_repo      = _resolve_repo(args.hub_repo)
    middle_layers = resolve_middle_layers(args.middle_layers, n_total=32)
    logger.info(f"[OT] Middle layers ({len(middle_layers)}): {middle_layers}")
    logger.info(f"[Loss] lambda_ot={args.lambda_ot}  lambda_reg={args.lambda_reg}")

    # ── Resume logic (rank-0 downloads, broadcasts) ───────────────────────────
    training_state = None
    if main:
        training_state = load_training_state(hub_repo)

    is_resuming_tensor = torch.tensor(
        [1 if training_state is not None else 0], dtype=torch.int32, device=device
    )
    if world_size > 1:
        dist.broadcast(is_resuming_tensor, src=0)
    is_resuming = bool(is_resuming_tensor.item())

    if is_resuming and training_state is not None:
        completed_epochs  = training_state.get("completed_epochs", [])
        saved_global_step = training_state.get("global_step", 0)
        resume_epoch      = training_state.get("current_epoch")
        resume_skip_steps = training_state.get("steps_done_in_epoch", 0)
        is_epoch_complete = training_state.get("is_epoch_complete", True)
        if is_epoch_complete or resume_epoch is None:
            resume_epoch = None; resume_skip_steps = 0
        if main:
            logger.info(
                f"[Resume] completed={completed_epochs} step={saved_global_step}"
            )
    else:
        completed_epochs  = []
        saved_global_step = 0
        resume_epoch      = None
        resume_skip_steps = 0

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_resuming:
        model = load_lora_from_hub(hub_repo, args.base_model, dtype, device_map)
        if model is None:
            model = build_lora_model(
                args.base_model, middle_layers,
                args.lora_r, args.lora_alpha, args.lora_dropout, dtype, device_map,
            )
    else:
        model = build_lora_model(
            args.base_model, middle_layers,
            args.lora_r, args.lora_alpha, args.lora_dropout, dtype, device_map,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = model.to(device)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("[Setup] Gradient checkpointing: ON")

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    logger.info(f"[Setup] Model on rank {rank}")

    # ── LayerWeights ──────────────────────────────────────────────────────────
    layer_weights_module = LayerWeights(n_layers=len(middle_layers)).to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    align_dataset = AlignmentDataset(
        alignment_data_path=args.data_root,
        opus_sample_ratio=args.opus_ratio,
        eng_eng_ratio=args.eng_eng_ratio,
    ).load()
    if main:
        align_dataset.stats()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    raw_model_for_params = model.module if hasattr(model, "module") else model
    param_groups = [
        {"params": [p for n, p in raw_model_for_params.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in raw_model_for_params.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    if not args.mean_layer:
        param_groups.append(
            {"params": list(layer_weights_module.parameters()),
             "weight_decay": 0.0, "lr": args.lr * 10}
        )

    use_fused = (
        torch.cuda.is_available()
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95), fused=use_fused,
    )
    if use_fused:
        logger.info("[Setup] Fused AdamW enabled.")

    steps_per_epoch_est = 1000
    total_steps         = steps_per_epoch_est * args.epochs
    warmup_steps        = max(1, int(total_steps * args.warmup_ratio))
    scheduler           = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ── Resume optimizer states ───────────────────────────────────────────────
    loss_log: List[Dict] = []
    if is_resuming and main:
        loss_log = load_optimizer_states(
            hub_repo, optimizer, scheduler, scaler,
            layer_weights_module, args.mean_layer, device,
        )

    global_step = saved_global_step

    oom_coord = OOMCoordinator(
        per_gpu_batch=per_gpu_batch,
        world_size=world_size,
        device=device,
    )

    log_file = None
    if main:
        log_file = open(
            output_dir / "ot_train_log.jsonl",
            "a" if is_resuming else "w", encoding="utf-8",
        )

    epoch_iter = range(1, args.epochs + 1)
    if main:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", unit="epoch",
                          dynamic_ncols=True, colour="green")

    for epoch in epoch_iter:
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} already done — skip.")
            continue

        skip_batches = resume_skip_steps if (epoch == resume_epoch) else 0
        logger.info(f"EPOCH {epoch}/{args.epochs} | rank={rank}")

        # ── DataLoader ────────────────────────────────────────────────────────
        train_loader = build_distributed_dataloader(
            align_dataset=align_dataset,
            tokenizer=tokenizer,
            per_gpu_batch=oom_coord.per_gpu_batch,
            max_length=args.max_length,
            seed=args.seed,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )
        steps_per_epoch = len(train_loader)

        if skip_batches >= steps_per_epoch:
            logger.info(
                f"[Resume] skip_batches={skip_batches} >= steps_per_epoch={steps_per_epoch}, reset to 0."
            )
            skip_batches = 0

        model.train()
        if not args.mean_layer:
            layer_weights_module.train()
        optimizer.zero_grad(set_to_none=True)

        accum_loss = accum_lm = accum_ot = accum_reg = 0.0
        accum_count    = 0
        steps_in_epoch = 0
        loader_needs_rebuild = False

        batch_iter = enumerate(train_loader)
        if main:
            batch_iter_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"  Epoch {epoch}", unit="batch",
                dynamic_ncols=True, leave=False,
            )

        for batch_idx, batch in batch_iter:

            if steps_in_epoch < skip_batches:
                steps_in_epoch += 1
                if main:
                    batch_iter_pbar.update(1)
                continue

            if loader_needs_rebuild:
                break

            is_last_accum = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1 == steps_per_epoch)
            )

            loss       = None
            oom_this_step = False

            try:
                loss, lm_item, ot_item, reg_item = compute_total_loss(
                    model=model,
                    batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    lambda_ot=args.lambda_ot,
                    lambda_reg=args.lambda_reg,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    mean_layer=args.mean_layer,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    device=device,
                )

                scaled_loss = loss / args.grad_accum
                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if not args.mean_layer and is_last_accum:
                    reduce_gradients_manual(layer_weights_module, world_size)

                accum_loss += loss.item()
                accum_lm   += lm_item
                accum_ot   += ot_item
                accum_reg  += reg_item
                accum_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                oom_this_step = True
                oom_coord.report_oom()
                _cleanup_after_oom(optimizer, loss)

            # ── OOM sync  [DIST-4] ────────────────────────────────────────────
            any_oom = oom_coord.sync_and_check()
            if any_oom:
                if not oom_this_step:
                    optimizer.zero_grad(set_to_none=True)
                oom_coord.halve_batch(logger if main else None)
                torch.cuda.empty_cache()
                gc.collect()
                loader_needs_rebuild = True
                steps_in_epoch += 1
                if main:
                    batch_iter_pbar.update(1)
                continue

            # ── Optimizer step ────────────────────────────────────────────────
            if is_last_accum:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)

                raw_m = model.module if hasattr(model, "module") else model
                all_trainable = [p for p in raw_m.parameters() if p.requires_grad]
                if not args.mean_layer:
                    all_trainable += list(layer_weights_module.parameters())

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_trainable, args.max_grad_norm,
                )

                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if accum_count > 0:
                    avg_loss = accum_loss / accum_count
                    avg_lm   = accum_lm   / accum_count
                    avg_ot   = accum_ot   / accum_count
                    avg_reg  = accum_reg  / accum_count

                    # All-reduce losses for logging
                    for t_val in [avg_loss, avg_lm, avg_ot, avg_reg]:
                        lt = torch.tensor(t_val, device=device)
                        all_reduce_mean(lt, world_size)

                    loss_tensor = torch.tensor(avg_loss, device=device)
                    lm_tensor   = torch.tensor(avg_lm,   device=device)
                    ot_tensor   = torch.tensor(avg_ot,   device=device)
                    reg_tensor  = torch.tensor(avg_reg,  device=device)
                    for t in [loss_tensor, lm_tensor, ot_tensor, reg_tensor]:
                        all_reduce_mean(t, world_size)

                    current_lr = scheduler.get_last_lr()[0]

                    if main:
                        log_entry = {
                            "epoch":     epoch,
                            "step":      global_step,
                            "loss":      round(loss_tensor.item(), 6),
                            "lm_loss":   round(lm_tensor.item(),  6),
                            "ot_loss":   round(ot_tensor.item(),  6),
                            "reg_loss":  round(reg_tensor.item(), 6),
                            "lam_ot":    round(args.lambda_ot * ot_tensor.item(),  6),
                            "lam_reg":   round(args.lambda_reg * reg_tensor.item(), 6),
                            "lr":        current_lr,
                            "grad_norm": round(float(grad_norm), 4),
                            "world_size": world_size,
                            "per_gpu_batch": oom_coord.per_gpu_batch,
                        }
                        log_file.write(json.dumps(log_entry) + "\n")
                        log_file.flush()
                        loss_log.append(log_entry)

                        batch_iter_pbar.set_postfix({
                            "L":     f"{loss_tensor.item():.4f}",
                            "LM":    f"{lm_tensor.item():.4f}",
                            "OT":    f"{ot_tensor.item():.4f}",
                            "Reg":   f"{reg_tensor.item():.4f}",
                            "lr":    f"{current_lr:.2e}",
                            "gnorm": f"{float(grad_norm):.3f}",
                        })

                    accum_loss = accum_lm = accum_ot = accum_reg = accum_count = 0.0

                # Mid-epoch checkpoint  [DIST-8]
                if args.save_iter > 0 and global_step % args.save_iter == 0:
                    barrier(world_size, device)
                    if main:
                        plot_training_loss(
                            loss_log, output_dir, smooth=args.plot_smooth,
                            lambda_ot=args.lambda_ot, lambda_reg=args.lambda_reg,
                        )
                        mid_state = build_state(
                            args, hub_repo, completed_epochs, global_step,
                            current_epoch=epoch,
                            steps_done_in_epoch=steps_in_epoch,
                            is_epoch_complete=False,
                            loss_log=loss_log,
                            middle_layers=middle_layers,
                            world_size=world_size,
                            per_gpu_batch=oom_coord.per_gpu_batch,
                        )
                        save_and_push_checkpoint(
                            hub_repo=hub_repo, output_dir=output_dir,
                            state=mid_state, model=model, tokenizer=tokenizer,
                            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                            layer_weights_module=layer_weights_module,
                            loss_log=loss_log, epoch=epoch,
                            private=args.hub_private,
                            smooth=args.plot_smooth, mean_layer=args.mean_layer,
                            lambda_ot=args.lambda_ot, lambda_reg=args.lambda_reg,
                            commit_suffix=f"step-{global_step}", delete_local=True,
                        )
                    barrier(world_size, device)
                    model.train()
                    if not args.mean_layer:
                        layer_weights_module.train()

            steps_in_epoch += 1
            if main:
                batch_iter_pbar.update(1)

        if main:
            batch_iter_pbar.close()

        # ── OOM rebuild and retry epoch ───────────────────────────────────────
        if loader_needs_rebuild:
            logger.info(
                f"[OOM] Rebuilding DataLoader with per_gpu_batch={oom_coord.per_gpu_batch}."
            )
            barrier(world_size, device)
            skip_batches = steps_in_epoch

            train_loader = rebuild_dataloader(
                align_dataset=align_dataset,
                tokenizer=tokenizer,
                per_gpu_batch=oom_coord.per_gpu_batch,
                max_length=args.max_length,
                seed=args.seed,
                num_workers=args.num_workers,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                logger=logger if main else None,
            )
            steps_per_epoch = len(train_loader)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            if main:
                batch_iter_pbar = tqdm(
                    total=steps_per_epoch,
                    desc=f"  Epoch {epoch} [retry]", unit="batch",
                    dynamic_ncols=True, leave=False,
                )

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx < skip_batches:
                    if main:
                        batch_iter_pbar.update(1)
                    continue

                is_last_accum = (
                    (batch_idx + 1) % args.grad_accum == 0
                    or (batch_idx + 1 == steps_per_epoch)
                )

                loss, lm_item, ot_item, reg_item = compute_total_loss(
                    model=model, batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    lambda_ot=args.lambda_ot,
                    lambda_reg=args.lambda_reg,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    mean_layer=args.mean_layer,
                    amp_dtype=amp_dtype, use_amp=use_amp, device=device,
                )

                scaled_loss = loss / args.grad_accum
                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if not args.mean_layer and is_last_accum:
                    reduce_gradients_manual(layer_weights_module, world_size)

                accum_loss += loss.item()
                accum_lm   += lm_item
                accum_ot   += ot_item
                accum_reg  += reg_item
                accum_count += 1

                if is_last_accum:
                    if use_amp and amp_dtype == torch.float16:
                        scaler.unscale_(optimizer)

                    raw_m = model.module if hasattr(model, "module") else model
                    all_trainable = [p for p in raw_m.parameters() if p.requires_grad]
                    if not args.mean_layer:
                        all_trainable += list(layer_weights_module.parameters())

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_trainable, args.max_grad_norm,
                    )

                    if use_amp and amp_dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if accum_count > 0:
                        avg_loss = accum_loss / accum_count
                        avg_lm   = accum_lm   / accum_count
                        avg_ot   = accum_ot   / accum_count
                        avg_reg  = accum_reg  / accum_count

                        loss_tensor = torch.tensor(avg_loss, device=device)
                        lm_tensor   = torch.tensor(avg_lm,   device=device)
                        ot_tensor   = torch.tensor(avg_ot,   device=device)
                        reg_tensor  = torch.tensor(avg_reg,  device=device)
                        for t in [loss_tensor, lm_tensor, ot_tensor, reg_tensor]:
                            all_reduce_mean(t, world_size)

                        current_lr = scheduler.get_last_lr()[0]

                        if main:
                            log_entry = {
                                "epoch":     epoch,
                                "step":      global_step,
                                "loss":      round(loss_tensor.item(), 6),
                                "lm_loss":   round(lm_tensor.item(),  6),
                                "ot_loss":   round(ot_tensor.item(),  6),
                                "reg_loss":  round(reg_tensor.item(), 6),
                                "lam_ot":    round(args.lambda_ot  * ot_tensor.item(),  6),
                                "lam_reg":   round(args.lambda_reg * reg_tensor.item(), 6),
                                "lr":        current_lr,
                                "grad_norm": round(float(grad_norm), 4),
                                "world_size": world_size,
                                "per_gpu_batch": oom_coord.per_gpu_batch,
                            }
                            log_file.write(json.dumps(log_entry) + "\n")
                            log_file.flush()
                            loss_log.append(log_entry)

                        accum_loss = accum_lm = accum_ot = accum_reg = accum_count = 0.0

                if main:
                    batch_iter_pbar.update(1)

            if main:
                batch_iter_pbar.close()

        # ── End of epoch ──────────────────────────────────────────────────────
        logger.info(f"[Epoch {epoch}] Done. global_step={global_step}")

        completed_epochs.append(epoch)
        barrier(world_size, device)

        if main:
            state = build_state(
                args, hub_repo, completed_epochs, global_step,
                current_epoch=epoch, steps_done_in_epoch=steps_in_epoch,
                is_epoch_complete=True, loss_log=loss_log,
                middle_layers=middle_layers,
                world_size=world_size, per_gpu_batch=oom_coord.per_gpu_batch,
            )
            save_and_push_checkpoint(
                hub_repo=hub_repo, output_dir=output_dir, state=state,
                model=model, tokenizer=tokenizer, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                layer_weights_module=layer_weights_module,
                loss_log=loss_log, epoch=epoch, private=args.hub_private,
                smooth=args.plot_smooth, mean_layer=args.mean_layer,
                lambda_ot=args.lambda_ot, lambda_reg=args.lambda_reg,
                commit_suffix="", delete_local=True,
            )

        barrier(world_size)

    if main and log_file:
        log_file.close()

    logger.info("OT + EN-REG DISTRIBUTED ALIGNMENT TRAINING COMPLETE")
    logger.info(f"  Completed: {completed_epochs}  |  Steps: {global_step}")

    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    available_gpus = torch.cuda.device_count()
    if args.gpus is not None and args.gpus > available_gpus:
        raise SystemExit(
            f"[ERROR] --gpus {args.gpus} requested but only "
            f"{available_gpus} GPU(s) detected. Aborting."
        )

    _base_logger.info("=" * 65)
    _base_logger.info("Llama-3-8B  Stage 2: OT + EN-Reg Alignment  [DISTRIBUTED]")
    _base_logger.info(f"Detected GPUs: {available_gpus} | Requested: {args.gpus}")
    _base_logger.info(f"Loss: L_LM + {args.lambda_ot}·L_OT + {args.lambda_reg}·L_Reg")
    _base_logger.info("=" * 65)

    train(args)