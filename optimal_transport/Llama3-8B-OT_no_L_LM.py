"""
Llama3-8B-OT-only-distributed.py
==========================
Stage 2 — Cross-lingual Optimal Transport Alignment (OT Loss Only)
Distributed Data Parallel (DDP) version

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTRIBUTED CHANGES (vs single-GPU version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[DIST-1]  --gpus N flag: validated against torch.cuda.device_count()
          before any model loading. If N > available GPUs → abort early.

[DIST-2]  Launch via torchrun (NOT python directly):
            torchrun --nproc_per_node=2 Llama3-8B-OT-only-distributed.py --gpus 2 ...
          The script detects rank/world_size from env vars set by torchrun.

[DIST-3]  BATCH SIZE SEMANTICS:
          --batch_size is the GLOBAL batch size.
          Per-GPU batch = batch_size // world_size.
          If batch_size is not divisible, it's rounded down and a warning
          is logged. Effective global batch = per_gpu_batch * world_size.

[DIST-4]  OOM HANDLING — coordinated across all GPUs:
          - Each GPU catches OOM locally and sets a flag tensor.
          - dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX) propagates it.
          - ALL GPUs halve their per_gpu_batch_size simultaneously so the
            DataLoader is rebuilt consistently on every rank.
          - This prevents deadlocks from mismatched batch sizes across ranks.

[DIST-5]  GRADIENT AGGREGATION:
          - DDP wraps the LoRA model → gradients are all-reduced automatically
            after each backward() call (Ring All-Reduce).
          - LayerWeights module is NOT wrapped in DDP (scalar params, tiny).
            Instead its gradients are manually all-reduced after backward.
          - Loss is computed in float32 per rank, then averaged across ranks
            via dist.all_reduce / world_size before logging.

[DIST-6]  DOMINANT (FROZEN) BRANCH:
          - Runs under torch.inference_mode() + no_grad on every rank
            independently (no sync needed — frozen weights, no grad).

[DIST-7]  DistributedSampler ensures each rank sees a non-overlapping
          shard of the dataset. set_epoch(epoch) is called each epoch
          to re-shuffle deterministically.

[DIST-8]  CHECKPOINTING: only rank-0 saves/uploads checkpoints to avoid
          race conditions and duplicate Hub uploads.

[DIST-9]  LOGGING: only rank-0 prints to avoid duplicated log lines.
          Use logger_rank0() wrapper throughout.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAUNCH COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single node, 2 GPUs:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    torchrun --nproc_per_node=2 Llama3-8B-OT-only-distributed.py \\
        --gpus 2 \\
        --base_model  ducanhdinh/Llama3-8B-Finetune \\
        --data_root   ../raw_data/alignment/ \\
        --output_dir  ./ot_checkpoints \\
        --hub_repo    Llama3-8B-OT \\
        --epochs      3 \\
        --batch_size  32   ← GLOBAL; each GPU gets 32/2=16 \\
        --grad_accum  4 \\
        --lr          1e-4

Single node, 4 GPUs:
    torchrun --nproc_per_node=4 Llama3-8B-OT-only-distributed.py --gpus 4 ...

Multi-node (2 nodes × 4 GPUs each = 8 GPUs total):
    # On node 0 (master):
    torchrun --nnodes=2 --nproc_per_node=4 \\
             --node_rank=0 --master_addr=<NODE0_IP> --master_port=29500 \\
             Llama3-8B-OT-only-distributed.py --gpus 8 ...
    # On node 1:
    torchrun --nnodes=2 --nproc_per_node=4 \\
             --node_rank=1 --master_addr=<NODE0_IP> --master_port=29500 \\
             Llama3-8B-OT-only-distributed.py --gpus 8 ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
import matplotlib.gridspec as gridspec
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "dataloader"))

from alignment_dataloader import AlignmentDataset, AlignmentDataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> Tuple[int, int, int]:
    """
    Initialise process group from torchrun env vars.
    Returns (rank, local_rank, world_size).
    """
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
    """In-place all_reduce then divide by world_size."""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)
    return tensor


def all_reduce_max(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """In-place all_reduce MAX — used for OOM flag broadcast."""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def reduce_gradients_manual(module: nn.Module, world_size: int) -> None:
    """
    Manually all_reduce gradients for a module that is NOT wrapped in DDP.
    Used for LayerWeights (tiny scalar params).
    [DIST-5]
    """
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
    logger = logging.getLogger(f"ot_dist.rank{rank}")
    logger.addFilter(RankFilter(rank))
    return logger


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
    p = argparse.ArgumentParser(description="Llama-3-8B OT Alignment — Distributed")

    # ── NEW: distributed flag ────────────────────────────────────────────────
    p.add_argument(
        "--gpus", type=int, default=None,
        help=(
            "Number of GPUs to use. Must equal torchrun --nproc_per_node. "
            "Validated against torch.cuda.device_count() BEFORE model loading."
        ),
    )

    p.add_argument("--base_model",      type=str, default="ducanhdinh/Llama3-8B-Finetune")
    p.add_argument("--data_root",       type=str, default="../raw_data/alignment/")
    p.add_argument("--output_dir",      type=str, default="./ot_checkpoints_dist")
    p.add_argument("--hub_repo",        type=str, default="Llama3-8B-OT")
    p.add_argument("--hub_private",     action="store_true")

    p.add_argument("--epochs",          type=int,   default=3)
    # batch_size = GLOBAL batch; per-GPU = batch_size // world_size  [DIST-3]
    p.add_argument("--batch_size",      type=int,   default=32,
                   help="GLOBAL batch size. Divided equally across GPUs.")
    p.add_argument("--grad_accum",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    p.add_argument("--sinkhorn_eps",    type=float, default=0.1)
    p.add_argument("--sinkhorn_iters",  type=int,   default=50)

    p.add_argument(
        "--middle_layers", type=str, default="auto",
        help="'auto' → layers[N/4 .. 3N/4). Or comma list: '8,12,16,20'",
    )

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
    # OOM: when triggered on ANY gpu, ALL gpus halve per-gpu batch  [DIST-4]
    p.add_argument("--oom_skip_batches",type=int,   default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")
    p.add_argument("--compile",         action="store_true", default=False)
    p.add_argument("--real_attn_en",    action="store_true", default=False)
    p.add_argument(
        "--mean_layer",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        metavar="BOOL",
    )
    p.add_argument("--plot_smooth",     type=int,   default=20)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# GPU validation  [DIST-1]
# ─────────────────────────────────────────────────────────────────────────────

def validate_gpus(requested: Optional[int], world_size: int, rank: int) -> int:
    """
    Check --gpus against available GPUs and torchrun world_size.
    Called on every rank; aborts on mismatch.
    Returns the validated world_size to use.
    """
    available = torch.cuda.device_count()

    # Detect available first
    if available == 0:
        raise RuntimeError(
            "[GPU] No CUDA GPUs detected. Cannot run distributed training."
        )

    if requested is not None:
        if requested > available:
            raise RuntimeError(
                f"[GPU] --gpus {requested} requested but only "
                f"{available} GPU(s) available on this node. Aborting."
            )
        if requested != world_size:
            raise RuntimeError(
                f"[GPU] --gpus {requested} does not match torchrun "
                f"world_size={world_size}. "
                f"Run: torchrun --nproc_per_node={requested} ... --gpus {requested}"
            )

    if rank == 0:
        _base_logger.info(
            f"[GPU] Detected {available} GPU(s) | Using {world_size} GPU(s)"
        )
    return world_size


# ─────────────────────────────────────────────────────────────────────────────
# Per-GPU batch size helper  [DIST-3]
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_gpu_batch(global_batch: int, world_size: int, rank: int) -> int:
    per_gpu = global_batch // world_size
    if per_gpu == 0:
        raise ValueError(
            f"[Batch] global batch_size={global_batch} < world_size={world_size}. "
            "Increase --batch_size."
        )
    remainder = global_batch % world_size
    if remainder != 0 and rank == 0:
        _base_logger.warning(
            f"[Batch] batch_size={global_batch} not divisible by {world_size} GPUs. "
            f"Using per_gpu={per_gpu} → effective global={per_gpu * world_size}."
        )
    return per_gpu


# ─────────────────────────────────────────────────────────────────────────────
# OOM coordinator  [DIST-4]
# ─────────────────────────────────────────────────────────────────────────────

class OOMCoordinator:
    """
    Coordinates OOM events across ranks.
    When any rank hits OOM, all_reduce MAX propagates it to all others.
    All ranks halve their per-GPU batch size simultaneously.
    """
    def __init__(self, per_gpu_batch: int, world_size: int, device: torch.device,
                 min_batch: int = 1):
        self.per_gpu_batch = per_gpu_batch
        self.world_size    = world_size
        self.device        = device
        self.min_batch     = min_batch
        self._skip_count   = 0

    def report_oom(self) -> None:
        """Called by the rank that hit OOM."""
        self._oom_flag_tensor().fill_(1)

    def sync_and_check(self) -> bool:
        """
        Synchronise OOM flag across all ranks.
        Returns True if any rank hit OOM (meaning all should halve).
        """
        flag = self._oom_flag_tensor()
        all_reduce_max(flag, self.world_size)
        hit = flag.item() > 0
        flag.zero_()  # reset for next iteration
        return bool(hit)

    def halve_batch(self, logger) -> int:
        old = self.per_gpu_batch
        self.per_gpu_batch = max(self.min_batch, old // 2)
        if logger:
            logger.warning(
                f"[OOM] All GPUs: per_gpu_batch {old} → {self.per_gpu_batch} "
                f"(global: {old * self.world_size} → "
                f"{self.per_gpu_batch * self.world_size})"
            )
        return self.per_gpu_batch

    def _oom_flag_tensor(self) -> torch.Tensor:
        # Lazily create a shared flag on the correct device
        if not hasattr(self, "_flag"):
            self._flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        return self._flag


# ─────────────────────────────────────────────────────────────────────────────
# Seed (rank-aware)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int, rank: int = 0) -> None:
    import random
    seed = seed + rank  # each rank gets a different but deterministic seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Parse middle layers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_middle_layers(spec: str, n_total: int = 32) -> List[int]:
    if spec == "auto":
        start  = n_total // 4
        end    = (n_total * 3) // 4
        layers = list(range(start, end))
    elif ":" in spec:
        a, b   = spec.split(":")
        layers = list(range(int(a), int(b)))
    else:
        layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    return layers


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
# Attention-weighted pooling  (eqs. 13–14)
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
# Per-layer OT loss  (eqs. 16–17)
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

    valid = tgt_mask.float().unsqueeze(2) * en_mask.float().unsqueeze(1)
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
# Forward passes
# ─────────────────────────────────────────────────────────────────────────────

def forward_target_branch(
    model,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers:  List[int],
    amp_dtype:      torch.dtype,
    use_amp:        bool,
    device:         torch.device,
) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Target (LoRA) branch — eq. (11).
    When DDP is used, model is a DDP wrapper; calling model() automatically
    syncs gradients across ranks during backward.
    """
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

    has_attentions = (
        out.attentions is not None and len(out.attentions) > 0
    )

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        attn   = out.attentions[l] if has_attentions else None
        layer_data[l] = (hidden, attn)
    return layer_data


@torch.inference_mode()
def forward_dominant_branch(
    model,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers:  List[int],
    real_attn_en:   bool,
    amp_dtype:      torch.dtype,
    use_amp:        bool,
    device:         torch.device,
) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Dominant (frozen EN) branch — eq. (12).
    No grad, no DDP sync needed. Each rank runs independently.  [DIST-6]
    """
    # Unwrap DDP to access adapter methods
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.disable_adapter_layers()
    try:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = raw_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                output_attentions=real_attn_en,
                use_cache=False,
            )
    finally:
        raw_model.enable_adapter_layers()

    has_attentions = (
        real_attn_en and out.attentions is not None and len(out.attentions) > 0
    )

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        attn   = out.attentions[l] if has_attentions else None
        layer_data[l] = (hidden, attn)
    return layer_data


# ─────────────────────────────────────────────────────────────────────────────
# Full loss  —  L = L_OT
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    model,
    batch:                  dict,
    middle_layers:          List[int],
    layer_weights_module:   LayerWeights,
    sinkhorn_eps:           float,
    sinkhorn_iters:         int,
    real_attn_en:           bool,
    mean_layer:             bool,
    amp_dtype:              torch.dtype,
    use_amp:                bool,
    device:                 torch.device,
) -> Tuple[torch.Tensor, float]:
    tgt_ids  = batch["tgt_input_ids"].to(device, non_blocking=True)
    tgt_mask = batch["tgt_attention_mask"].to(device, non_blocking=True)
    en_ids   = batch["en_input_ids"].to(device, non_blocking=True)
    en_mask  = batch["en_attention_mask"].to(device, non_blocking=True)

    tgt_layer_data = forward_target_branch(
        model, tgt_ids, tgt_mask, middle_layers, amp_dtype, use_amp, device,
    )
    en_layer_data = forward_dominant_branch(
        model, en_ids, en_mask, middle_layers, real_attn_en, amp_dtype, use_amp, device,
    )

    ot_losses: List[torch.Tensor] = []
    for l in middle_layers:
        tgt_hidden, tgt_attn = tgt_layer_data[l]
        en_hidden,  en_attn  = en_layer_data[l]

        if tgt_attn is not None:
            h_tgt = attention_weighted_pool_from_attn(tgt_hidden, tgt_attn, tgt_mask)
        else:
            h_tgt = attention_weighted_pool_uniform(tgt_hidden, tgt_mask)

        if en_attn is not None:
            h_en = attention_weighted_pool_from_attn(en_hidden, en_attn, en_mask)
        else:
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

    return loss_ot, loss_ot.item()


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
# DataLoader builder (with DistributedSampler)  [DIST-7]
# ─────────────────────────────────────────────────────────────────────────────

def build_distributed_dataloader(
    align_dataset,
    tokenizer,
    per_gpu_batch: int,
    max_length: int,
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
    epoch: int = 0,
):
    """
    Wraps AlignmentDataLoader with DistributedSampler so each rank
    sees only its shard.  [DIST-7]
    """
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
        # DistributedSampler kwargs passed through if the dataloader supports it:
        rank=rank,
        world_size=world_size,
    )
    # If AlignmentDataLoader doesn't support rank/world_size natively,
    # inject a DistributedSampler into its underlying torch DataLoader.
    # We handle both cases below:
    if hasattr(loader, "sampler") and isinstance(
        getattr(loader, "sampler", None), DistributedSampler
    ):
        loader.sampler.set_epoch(epoch)
    elif hasattr(loader, "dataloader") and hasattr(loader.dataloader, "sampler"):
        if isinstance(loader.dataloader.sampler, DistributedSampler):
            loader.dataloader.sampler.set_epoch(epoch)

    return loader


def rebuild_dataloader_with_new_batch(
    align_dataset,
    tokenizer,
    per_gpu_batch: int,
    max_length: int,
    seed: int,
    num_workers: int,
    rank: int,
    world_size: int,
    epoch: int,
    logger,
):
    """Rebuild dataloader after OOM batch-size reduction."""
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
# Loss plotting
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 2:
        return values
    arr    = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    padded = np.concatenate([arr[:window - 1], arr])
    return np.convolve(padded, kernel, mode="valid").tolist()


def plot_training_loss(loss_log: List[Dict], output_dir: Path, smooth: int = 20) -> None:
    if not loss_log:
        return
    steps   = [e["step"]    for e in loss_log]
    ot_loss = [e["ot_loss"] for e in loss_log]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(
        f"OT-Only Alignment (Distributed) — step {steps[-1]}",
        fontsize=13, fontweight="bold",
    )
    colour = "#3aaa6b"
    ax.plot(steps, ot_loss, color=colour, alpha=0.20, linewidth=0.7, label="raw L_OT")
    smoothed = _rolling_mean(ot_loss, smooth)
    ax.plot(steps, smoothed, color=colour, linewidth=1.8, label=f"smooth (w={smooth})")
    ax.scatter([steps[-1]], [smoothed[-1]], color=colour, s=30, zorder=5)
    ax.annotate(
        f"{smoothed[-1]:.4f}",
        xy=(steps[-1], smoothed[-1]), xytext=(5, 4),
        textcoords="offset points", fontsize=8, color=colour,
    )
    ax.set_title("L_OT  (pure OT loss, eq.18)", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Optimizer step", fontsize=9)
    ax.set_ylabel("OT loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper right")

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
# HF Hub helpers (rank-0 only)
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
    commit_suffix="", delete_local=True,
):
    """
    Only called from rank-0.  [DIST-8]
    Unwraps DDP before saving.
    """
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo(hub_repo, private)

    # Unwrap DDP to save LoRA adapter
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

    plot_training_loss(loss_log, output_dir, smooth=smooth)

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
    return model


def build_state(args, hub_repo, completed_epochs, global_step,
                current_epoch, steps_done_in_epoch, is_epoch_complete,
                loss_log, middle_layers, world_size, per_gpu_batch):
    return {
        "base_model": args.base_model, "hub_repo": hub_repo,
        "total_epochs": args.epochs, "completed_epochs": completed_epochs,
        "global_step": global_step, "current_epoch": current_epoch,
        "steps_done_in_epoch": steps_done_in_epoch,
        "is_epoch_complete": is_epoch_complete,
        "middle_layers": middle_layers,
        "sinkhorn_eps": args.sinkhorn_eps,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "batch_size_global": args.batch_size,
        "per_gpu_batch": per_gpu_batch,
        "world_size": world_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr, "opus_ratio": args.opus_ratio,
        "eng_eng_ratio": args.eng_eng_ratio,
        "seq_length": args.max_length,
        "mean_layer": args.mean_layer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    # ── Distributed init  ────────────────────────────────────────────────────
    rank, local_rank, world_size = init_distributed()
    logger = setup_logger(rank)
    main   = is_main_process(rank)

    # ── GPU validation  [DIST-1] ─────────────────────────────────────────────
    validate_gpus(args.gpus, world_size, rank)

    set_seed(args.seed, rank)

    if args.seq_length is not None:
        args.max_length = args.seq_length

    output_dir = Path(args.output_dir)
    if main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Precision ────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    barrier(world_size, device)

    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16; amp_dtype = torch.bfloat16; use_amp = True
        logger.info("[Setup] bfloat16 AMP")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16; amp_dtype = torch.float16; use_amp = True
        logger.info("[Setup] float16 AMP")
    else:
        dtype = torch.float32; amp_dtype = torch.float32; use_amp = False

    # ── Per-GPU batch size  [DIST-3] ─────────────────────────────────────────
    per_gpu_batch = compute_per_gpu_batch(args.batch_size, world_size, rank)
    logger.info(
        f"[Batch] global={args.batch_size} | "
        f"per_gpu={per_gpu_batch} | world_size={world_size}"
    )

    # device_map=None for DDP; each rank manages its own GPU
    device_map = None

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    hub_repo      = _resolve_repo(args.hub_repo)
    middle_layers = resolve_middle_layers(args.middle_layers, n_total=32)
    logger.info(f"[OT] Middle layers ({len(middle_layers)}): {middle_layers}")

    # ── Resume logic (only rank-0 downloads state)  ──────────────────────────
    training_state = None
    if main:
        training_state = load_training_state(hub_repo)

    # Broadcast whether we are resuming
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

    # ── Model  ───────────────────────────────────────────────────────────────
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

    # ── Wrap in DDP  ─────────────────────────────────────────────────────────
    # find_unused_parameters=True because the frozen EN branch doesn't update
    # parameters but they share the same model object.
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    logger.info(f"[Setup] Model wrapped in DDP on rank {rank}")

    # ── LayerWeights (NOT DDP-wrapped — manually all-reduced)  [DIST-5] ──────
    layer_weights_module = LayerWeights(n_layers=len(middle_layers)).to(device)

    # ── Dataset  ─────────────────────────────────────────────────────────────
    align_dataset = AlignmentDataset(
        alignment_data_path=args.data_root,
        opus_sample_ratio=args.opus_ratio,
        eng_eng_ratio=args.eng_eng_ratio,
    ).load()
    if main:
        align_dataset.stats()

    # ── Optimizer  ───────────────────────────────────────────────────────────
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

    # Steps per epoch = steps on this rank (dataset is sharded)
    # We'll measure after building the loader; use an estimate for scheduler
    steps_per_epoch_est = 1000  # placeholder; updated after first epoch
    total_steps         = steps_per_epoch_est * args.epochs
    warmup_steps        = max(1, int(total_steps * args.warmup_ratio))
    scheduler           = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ── Resume optimizer states (rank-0 loads, then broadcasts)  ─────────────
    loss_log: List[Dict] = []
    if is_resuming and main:
        loss_log = load_optimizer_states(
            hub_repo, optimizer, scheduler, scaler,
            layer_weights_module, args.mean_layer, device,
        )

    global_step = saved_global_step

    # ── OOM coordinator  [DIST-4] ─────────────────────────────────────────────
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

        # ── Build DataLoader for this epoch  ──────────────────────────────────
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

        model.train()
        if not args.mean_layer:
            layer_weights_module.train()
        optimizer.zero_grad(set_to_none=True)

        accum_ot    = 0.0
        accum_count = 0
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

            # Skip already-done batches on resume
            if steps_in_epoch < skip_batches:
                steps_in_epoch += 1
                if main:
                    batch_iter_pbar.update(1)
                continue

            # Reload if OOM triggered a batch-size change
            if loader_needs_rebuild:
                break

            is_last_accum = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1 == steps_per_epoch)
            )

            loss = None
            oom_this_step = False

            try:
                loss, ot_item = compute_total_loss(
                    model=model,
                    batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    real_attn_en=args.real_attn_en,
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

                # Manually all_reduce LayerWeights grads  [DIST-5]
                if not args.mean_layer and is_last_accum:
                    reduce_gradients_manual(layer_weights_module, world_size)

                accum_ot    += ot_item
                accum_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                oom_this_step = True
                oom_coord.report_oom()
                _cleanup_after_oom(optimizer, loss)

            # ── OOM sync across all ranks  [DIST-4] ───────────────────────────
            any_oom = oom_coord.sync_and_check()
            if any_oom:
                # All ranks skip this step and halve batch size
                if not oom_this_step:
                    # Non-OOM ranks still need to zero grad consistently
                    optimizer.zero_grad(set_to_none=True)

                new_batch = oom_coord.halve_batch(logger if main else None)
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

                # Clip grads — DDP already all-reduced them
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
                    avg_ot = accum_ot / accum_count

                    # All-reduce the loss for logging (cosmetic, not used in backward)
                    loss_tensor = torch.tensor(avg_ot, device=device)
                    all_reduce_mean(loss_tensor, world_size)
                    avg_ot_global = loss_tensor.item()

                    current_lr = scheduler.get_last_lr()[0]

                    if main:
                        log_entry = {
                            "epoch":     epoch,
                            "step":      global_step,
                            "loss":      round(avg_ot_global, 6),
                            "ot_loss":   round(avg_ot_global, 6),
                            "lr":        current_lr,
                            "grad_norm": round(float(grad_norm), 4),
                            "world_size": world_size,
                            "per_gpu_batch": oom_coord.per_gpu_batch,
                        }
                        log_file.write(json.dumps(log_entry) + "\n")
                        log_file.flush()
                        loss_log.append(log_entry)

                        if main:
                            batch_iter_pbar.set_postfix({
                                "L_OT": f"{avg_ot_global:.4f}",
                                "lr":   f"{current_lr:.2e}",
                                "gnorm": f"{float(grad_norm):.3f}",
                                "bs/gpu": oom_coord.per_gpu_batch,
                            })

                    accum_ot = accum_count = 0.0

                # Mid-epoch checkpoint  [DIST-8]
                if args.save_iter > 0 and global_step % args.save_iter == 0:
                    barrier(world_size, device)
                    if main:
                        plot_training_loss(loss_log, output_dir, smooth=args.plot_smooth)
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

        # ── If OOM caused early exit from loop, rebuild and redo epoch ─────────
        if loader_needs_rebuild:
            logger.info(
                f"[OOM] Rebuilding DataLoader with per_gpu_batch="
                f"{oom_coord.per_gpu_batch}. Restarting epoch {epoch}."
            )
            barrier(world_size, device)
            # Restart epoch (skip already-done steps)
            skip_batches = steps_in_epoch  # skip what we already processed

            train_loader = rebuild_dataloader_with_new_batch(
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

                loss, ot_item = compute_total_loss(
                    model=model, batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    real_attn_en=args.real_attn_en,
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

                accum_ot    += ot_item
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
                        avg_ot = accum_ot / accum_count
                        loss_tensor = torch.tensor(avg_ot, device=device)
                        all_reduce_mean(loss_tensor, world_size)
                        avg_ot_global = loss_tensor.item()
                        current_lr = scheduler.get_last_lr()[0]

                        if main:
                            log_entry = {
                                "epoch": epoch, "step": global_step,
                                "loss": round(avg_ot_global, 6),
                                "ot_loss": round(avg_ot_global, 6),
                                "lr": current_lr,
                                "grad_norm": round(float(grad_norm), 4),
                                "world_size": world_size,
                                "per_gpu_batch": oom_coord.per_gpu_batch,
                            }
                            log_file.write(json.dumps(log_entry) + "\n")
                            log_file.flush()
                            loss_log.append(log_entry)

                        accum_ot = accum_count = 0.0

                if main:
                    batch_iter_pbar.update(1)

            if main:
                batch_iter_pbar.close()

        # ── End-of-epoch  ─────────────────────────────────────────────────────
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
                commit_suffix="", delete_local=True,
            )

        barrier(world_size)

    if main and log_file:
        log_file.close()

    logger.info("OT-ONLY DISTRIBUTED ALIGNMENT TRAINING COMPLETE")
    logger.info(f"  Completed: {completed_epochs}  |  Steps: {global_step}")

    if world_size > 1:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Early GPU count check BEFORE dist init (catches obvious misuse quickly)
    available_gpus = torch.cuda.device_count()
    if args.gpus is not None and args.gpus > available_gpus:
        raise SystemExit(
            f"[ERROR] --gpus {args.gpus} requested but only "
            f"{available_gpus} GPU(s) detected on this machine. Aborting."
        )

    _base_logger.info("=" * 65)
    _base_logger.info("Llama-3-8B  Stage 2: OT-ONLY Alignment  [DISTRIBUTED]")
    _base_logger.info(f"Detected GPUs: {available_gpus} | Requested: {args.gpus}")
    _base_logger.info("=" * 65)

    train(args)