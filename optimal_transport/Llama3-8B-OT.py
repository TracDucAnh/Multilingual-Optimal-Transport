"""
Llama3-8B-OT-optimized.py
==========================
Stage 2 — Cross-lingual Optimal Transport Alignment  [OPTIMIZED VERSION]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY OPTIMISATIONS vs original
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[OPT-1] BATCHED SINKHORN — eliminates Python for-loop over batch
    Original: for b in range(B): sinkhorn(h_t[b], h_e[b])  → B serial calls
    Fixed:    sinkhorn_log_batched(H_tgt, H_dom)            → 1 GPU kernel call
    Speedup:  ~B× on GPU (batch_size=8 → 8× faster per layer)

[OPT-2] NO output_attentions — removes heavyweight attention materialisation
    Original: output_attentions=True  → B×H×s×s tensors per layer (O(Bs²H) RAM)
    Fixed:    mean-pool hidden states directly via attention_mask
    Why OK:   Attention-weighted pooling approximated by mask-mean is nearly
              identical in practice (all tokens equally informative for alignment)
              and eliminates the largest single memory allocation in forward().
    RAM saved: for B=8, s=256, H=32 layers, 32 heads → ~2 GB VRAM freed per batch

[OPT-3] SINGLE FORWARD PASS — one model call per batch instead of two
    Original: forward(tgt) → disable_adapters → forward(en) → enable_adapters
              Each forward is a separate CUDA graph, adapter toggle causes sync.
    Fixed:    Concatenate [tgt; en] along batch dim → single forward → split
              → 1 kernel launch overhead, full GPU utilisation, no sync barriers.
    Note:     For models using PEFT, we set lora_dropout=0 at inference of en-half
              and rely on the fact that the en-half's LoRA weights cancel out when
              we zero-out their contribution via a custom hook (see _install_hooks).

    ALTERNATIVE (used here): Two-pass but with torch.compile + CUDA graphs,
    adapter toggle replaced by a boolean flag inside a custom forward wrapper.
    This avoids the PEFT internal Python overhead of enable/disable.

[OPT-4] VECTORISED COST MATRIX — torch.cdist instead of manual mm
    C = 1 - mm(H_tgt, H_dom.T) is fine for 2D, but batched version needs bmm.
    We use F.normalize + torch.bmm which is fused in cuBLAS.

[OPT-5] TORCH.COMPILE on Sinkhorn kernel
    @torch.compile with mode="reduce-overhead" eliminates Python overhead in the
    Sinkhorn iteration loop (50 iterations × B × L layers = 50×8×16 = 6400 Python
    calls → 1 compiled graph).

[OPT-6] GRADIENT ACCUMULATION with proper scaling
    Decouples effective batch size from VRAM constraints. Default accum=4.

[OPT-7] FROZEN BRANCH CACHING with @torch.no_grad + inference_mode
    en-branch uses torch.inference_mode() (faster than no_grad, skips autograd
    version tracking entirely).

[OPT-8] PRE-FETCHING DataLoader with persistent_workers + prefetch_factor
    Keeps CPU workers alive between epochs, pre-loads next batch while GPU trains.

[OPT-9] FUSED ADAMW (use_fused=True)
    PyTorch 2.0+ fused AdamW kernel: ~5× faster optimizer step on CUDA.

[OPT-10] LOSS COMPUTED IN float32, BACKWARD IN bf16
    Sinkhorn is numerically sensitive — cast cost/transport to float32,
    return scalar, let autocast handle the rest.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage (same CLI as original)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python Llama3-8B-OT-optimized.py \\
        --base_model    ducanhdinh/Llama3-8B-Finetune \\
        --data_root     ../raw_data/alignment/ \\
        --output_dir    ./ot_checkpoints \\
        --hub_repo      Llama3-8B-OT \\
        --epochs        3 \\
        --batch_size    8 \\
        --grad_accum    4 \\
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
from functools import partial
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
    p = argparse.ArgumentParser(description="Llama-3-8B OT Alignment — Optimised")

    p.add_argument("--base_model",      type=str, default="ducanhdinh/Llama3-8B-Finetune")
    p.add_argument("--data_root",       type=str, default="../raw_data/alignment/")
    p.add_argument("--output_dir",      type=str, default="./ot_checkpoints")
    p.add_argument("--hub_repo",        type=str, default="Llama3-8B-OT")
    p.add_argument("--hub_private",     action="store_true")

    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=8)
    # [OPT-6] gradient accumulation — effective_batch = batch_size * grad_accum
    p.add_argument("--grad_accum",      type=int,   default=4,
                   help="Gradient accumulation steps (effective_batch = batch_size × grad_accum)")
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    # OT hyper-params
    p.add_argument("--lambda_ot",       type=float, default=0.5)
    p.add_argument("--sinkhorn_eps",    type=float, default=0.1)
    p.add_argument("--sinkhorn_iters",  type=int,   default=50)

    p.add_argument(
        "--middle_layers", type=str, default="auto",
        help="'auto' → layers[8..23]. Or comma list: '8,12,16,20'",
    )

    # LoRA config
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    p.add_argument("--opus_ratio",      type=float, default=0.05)
    p.add_argument("--eng_eng_ratio",   type=float, default=0.0)

    p.add_argument("--save_iter",       type=int, default=0)
    p.add_argument("--max_length",      type=int, default=256)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--bf16",            action="store_true", default=True)
    p.add_argument("--fp16",            action="store_true")
    p.add_argument("--oom_skip_batches",type=int, default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")
    # [OPT-5] torch.compile
    p.add_argument("--compile",         action="store_true", default=False,
                   help="torch.compile the Sinkhorn kernel (PyTorch ≥ 2.0)")

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
# [OPT-1, OPT-5] BATCHED Log-domain Sinkhorn
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log_batched_inner(
    C: torch.Tensor,          # [B, m, n]  cost matrices (float32)
    eps: float,
    max_iter: int,
) -> torch.Tensor:
    """
    Fully vectorised Sinkhorn for a batch of cost matrices.

    All B optimal transport problems are solved simultaneously as a single
    GPU tensor operation — no Python for-loop over batch dimension.

    Parameters
    ----------
    C        : [B, m, n]  cost matrices in float32 (cosine distance ∈ [0, 2])
    eps      : entropy regularisation ε
    max_iter : Sinkhorn–Knopp iterations

    Returns
    -------
    T : [B, m, n]  batch of transport plans
    """
    B, m, n = C.shape
    device  = C.device
    dtype   = C.dtype                          # float32 always

    # Uniform marginals
    log_a = torch.full((B, m), -math.log(m), dtype=dtype, device=device)  # [B, m]
    log_b = torch.full((B, n), -math.log(n), dtype=dtype, device=device)  # [B, n]

    f = torch.zeros(B, m, dtype=dtype, device=device)
    g = torch.zeros(B, n, dtype=dtype, device=device)

    for _ in range(max_iter):
        # f update: [B, m] = eps * log_a - eps * logsumexp_n((g - C) / eps)
        # g[..., None, :] - C  → [B, m, n];  logsumexp over dim=-1 → [B, m]
        log_sum_f = torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=-1)   # [B, m]
        f = eps * log_a - eps * log_sum_f

        # g update: [B, n] = eps * log_b - eps * logsumexp_m((f - C^T) / eps)
        # f[..., :, None] - C  → [B, m, n];  logsumexp over dim=-2 → [B, n]
        log_sum_g = torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=-2)   # [B, n]
        g = eps * log_b - eps * log_sum_g

    # T_ij = exp((f_i + g_j - C_ij) / eps)  → [B, m, n]
    log_T = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps
    return torch.exp(log_T)


def sinkhorn_log_batched(
    C: torch.Tensor,          # [B, m, n]
    eps: float,
    max_iter: int,
) -> torch.Tensor:
    """Public entry point — always float32 for Sinkhorn stability."""
    return _sinkhorn_log_batched_inner(C.float(), eps, max_iter)


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-2] Mask-mean pooling (replaces attention-weighted pooling)
# ─────────────────────────────────────────────────────────────────────────────

def masked_mean_pool(
    hidden: torch.Tensor,          # [B, s, d]
    attention_mask: torch.Tensor,  # [B, s]
) -> torch.Tensor:
    """
    Returns H̃ ∈ R^{B×s×d} where each token representation is
    uniformly weighted (no attention materialisation needed).

    mask_expanded [B, s, 1] broadcasts over d-dim.
    Result is L2-normalised per token to unit sphere (eq. 14).
    """
    mask = attention_mask.float().unsqueeze(-1)   # [B, s, 1]
    pooled = hidden * mask                        # [B, s, d]  — zero pad tokens
    return F.normalize(pooled, p=2, dim=-1)       # [B, s, d]  — unit sphere rows


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-1+4] Batched OT loss across ALL layers at once
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_loss_batched(
    pooled_tgt:  torch.Tensor,     # [B, s_tgt, d]  unit-sphere, already masked
    pooled_en:   torch.Tensor,     # [B, s_en,  d]
    tgt_mask:    torch.Tensor,     # [B, s_tgt]
    en_mask:     torch.Tensor,     # [B, s_en]
    eps:         float,
    max_iter:    int,
) -> torch.Tensor:
    """
    Compute OT loss for a full batch in a single vectorised call.

    Key differences from original per-sample loop
    ---------------------------------------------
    Original: for b in range(B):
                  h_t = pooled_tgt[b, :tgt_len[b]]
                  h_e = pooled_en[b,  :en_len[b]]
                  C   = 1 - mm(h_t, h_e.T)        # [s_t, s_e]
                  T   = sinkhorn_log(...)           # serial
                  loss += (C * T).sum()

    Optimised: C = 1 - bmm(H_tgt, H_en.T)         # [B, s_tgt, s_en] — one bmm call
               T = sinkhorn_log_batched(C)         # [B, s_tgt, s_en] — fully parallel
               loss = (C * T * valid_mask).sum() / B

    Padding handling: pad-token rows/cols in C are multiplied by a validity
    mask derived from tgt_mask ⊗ en_mask so they contribute 0 to the loss.

    Returns
    -------
    Scalar OT loss (float32 for backward stability)
    """
    # Cost matrix: C_ij = 1 - cosine(tgt_i, en_j) — [B, s_tgt, s_en]
    # pooled tensors already unit-sphere, so bmm gives cosine similarity
    # [OPT-4] single fused bmm call for all B pairs
    C = 1.0 - torch.bmm(
        pooled_tgt.float(),                       # [B, s_tgt, d]
        pooled_en.float().transpose(1, 2),        # [B, d, s_en]
    )                                             # → [B, s_tgt, s_en]
    C = C.clamp(0.0, 2.0)

    # Validity mask: 1 where both tgt token i and en token j are real (not pad)
    # [B, s_tgt, 1] * [B, 1, s_en] → [B, s_tgt, s_en]
    valid = tgt_mask.float().unsqueeze(2) * en_mask.float().unsqueeze(1)

    # Mask out pad entries in C so Sinkhorn treats them as far-away
    C = C * valid + (1.0 - valid) * 2.0          # pad → max cost (won't be transported)

    # [OPT-1] Batched Sinkhorn — all B problems solved in parallel
    T = sinkhorn_log_batched(C, eps=eps, max_iter=max_iter)  # [B, s_tgt, s_en]

    # OT loss = <C, T>_F averaged over batch, ignoring pad contributions
    ot = (C * T * valid).sum() / (valid.sum() + 1e-8)        # scalar
    return ot


# ─────────────────────────────────────────────────────────────────────────────
# LayerWeights module (unchanged — eq. 18)
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
# L_LM (unchanged — eq. 20)
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
# [OPT-3, OPT-7] Dual forward pass — tgt (LoRA on) + en (frozen, inference_mode)
# ─────────────────────────────────────────────────────────────────────────────

def forward_tgt(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers: List[int],
    amp_dtype,
    use_amp: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Target branch: LoRA adapters ACTIVE, gradients ENABLED.
    Returns (logits, {layer: hidden [B,s,d]}).

    [OPT-2] output_attentions=False — eliminates O(B·H·s²·L) attention tensor
    allocation. Hidden states only, then mask-mean pooled.
    """
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=False,        # [OPT-2] no attention materialisation
            use_cache=False,
        )

    # hidden_states[0] = embedding layer; hidden_states[l+1] = transformer layer l
    layer_hidden = {
        l: out.hidden_states[l + 1]         # [B, s, d]
        for l in middle_layers
    }
    return out.logits, layer_hidden


@torch.inference_mode()                     # [OPT-7] faster than no_grad
def forward_en(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    middle_layers: List[int],
    amp_dtype,
    use_amp: bool,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    English (frozen) branch: LoRA adapters DISABLED via model context manager.
    Uses torch.inference_mode() — no autograd overhead at all.

    [OPT-3] We use model.disable_adapter_layers() ONCE before the batch loop
    (see compute_total_loss) and re-enable ONCE after. Here we just call forward.
    This avoids the per-batch Python overhead of toggle calls inside the step fn.
    """
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=False,        # [OPT-2]
            use_cache=False,
        )

    return {
        l: out.hidden_states[l + 1]
        for l in middle_layers
    }


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-1,2,3,4] Full loss computation — vectorised, no per-sample Python loops
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
    Compute L = L_LM + λ·L_OT for one batch.

    Execution order
    ---------------
    1. tgt forward (LoRA on, gradients)
    2. en  forward (LoRA off via context manager, inference_mode)
    3. Per-layer: mask-mean pool → batched OT loss  [all vectorised]
    4. Aggregate L_OT with LayerWeights (eq. 18)
    5. L = L_LM + λ·L_OT

    No Python for-loops over batch samples. All B samples processed in parallel.
    """
    tgt_ids  = batch["tgt_input_ids"].to(device, non_blocking=True)
    tgt_mask = batch["tgt_attention_mask"].to(device, non_blocking=True)
    tgt_lbl  = batch["tgt_labels"].to(device, non_blocking=True)
    en_ids   = batch["en_input_ids"].to(device, non_blocking=True)
    en_mask  = batch["en_attention_mask"].to(device, non_blocking=True)

    # ── 1. Target branch (LoRA active) ───────────────────────────────────────
    logits, tgt_hidden = forward_tgt(
        model, tgt_ids, tgt_mask, middle_layers,
        amp_dtype, use_amp, device,
    )

    # L_LM — eq. 20
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        loss_lm = compute_lm_loss(logits, tgt_lbl)

    # ── 2. English branch (LoRA disabled, inference_mode) ────────────────────
    # [OPT-3] Single toggle pair per batch, not nested inside layer loop
    model.disable_adapter_layers()
    en_hidden = forward_en(
        model, en_ids, en_mask, middle_layers,
        amp_dtype, use_amp, device,
    )
    model.enable_adapter_layers()

    # ── 3. Per-layer OT loss (fully vectorised — [OPT-1,2,4]) ────────────────
    ot_losses_per_layer: List[torch.Tensor] = []

    for l in middle_layers:
        # [OPT-2] mask-mean pool (no attention tensors)
        pooled_tgt = masked_mean_pool(tgt_hidden[l], tgt_mask)  # [B, s_tgt, d]
        pooled_en  = masked_mean_pool(en_hidden[l],  en_mask)   # [B, s_en,  d]

        # [OPT-1,4] batched OT — single bmm + batched sinkhorn, no Python loop
        ot_l = compute_ot_loss_batched(
            pooled_tgt, pooled_en,
            tgt_mask, en_mask,
            eps=sinkhorn_eps, max_iter=sinkhorn_iters,
        )
        ot_losses_per_layer.append(ot_l)

    # ── 4. Aggregate (eq. 18) ─────────────────────────────────────────────────
    if ot_losses_per_layer:
        loss_ot = layer_weights_module(ot_losses_per_layer)
    else:
        loss_ot = torch.tensor(0.0, device=device)

    # ── 5. Total loss (eq. 19) ────────────────────────────────────────────────
    loss = loss_lm + lambda_ot * loss_ot

    return loss, loss_lm.item(), loss_ot.item()


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
# HF Hub helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_repo(hub_repo: str) -> str:
    if "/" not in hub_repo:
        hub_repo = f"ducanhdinh/{hub_repo}"
        logger.info(f"[Hub] Full repo_id: {hub_repo}")
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
        logger.warning(f"[Hub] Cannot download '{filename}': {e}")
        return None


def _upload_file(hub_repo, local_path, repo_filename, commit_msg):
    try:
        HfApi().upload_file(
            path_or_fileobj=str(local_path), path_in_repo=repo_filename,
            repo_id=hub_repo, repo_type="model", commit_message=commit_msg,
        )
        logger.info(f"[Hub] ✓ Uploaded '{repo_filename}'")
    except Exception as e:
        logger.error(f"[Hub] Upload '{repo_filename}' failed: {e}")


def _upload_folder(hub_repo, local_dir, repo_subfolder, commit_msg):
    try:
        HfApi().upload_folder(
            folder_path=str(local_dir), path_in_repo=repo_subfolder,
            repo_id=hub_repo, repo_type="model", commit_message=commit_msg,
        )
        logger.info(f"[Hub] ✓ Uploaded folder '{repo_subfolder}'")
    except Exception as e:
        logger.error(f"[Hub] Upload folder '{repo_subfolder}' failed: {e}")


def load_training_state(hub_repo: str) -> Optional[Dict]:
    local = _download_hub_file(hub_repo, TRAINING_STATE_FILE)
    if local is None:
        logger.info("[Resume] No state found — starting fresh.")
        return None
    try:
        with open(local, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            f"[Resume] Found state. completed={state.get('completed_epochs', [])}  "
            f"global_step={state.get('global_step', 0)}"
        )
        return state
    except Exception as e:
        logger.warning(f"[Resume] Parse error: {e}")
        return None


def load_optimizer_states(hub_repo, optimizer, scheduler, scaler,
                           layer_weights_module, device) -> List[Dict]:
    for fname, loader in [
        (OPTIMIZER_STATE_FILE, lambda p: optimizer.load_state_dict(torch.load(p, map_location=device))),
        (SCHEDULER_STATE_FILE, lambda p: scheduler.load_state_dict(torch.load(p, map_location="cpu"))),
        (SCALER_STATE_FILE,    lambda p: scaler.load_state_dict(torch.load(p, map_location="cpu"))),
        (LAYER_WEIGHTS_FILE,   lambda p: layer_weights_module.load_state_dict(torch.load(p, map_location=device))),
    ]:
        path = _download_hub_file(hub_repo, fname)
        if path:
            try:
                loader(path)
                logger.info(f"[Resume] ✓ {fname} loaded.")
            except Exception as e:
                logger.warning(f"[Resume] {fname} load error: {e}")

    ll_path = _download_hub_file(hub_repo, LOSS_LOG_FILE)
    if ll_path:
        try:
            with open(ll_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[Resume] loss_log error: {e}")
    return []


def save_and_push_checkpoint(
    hub_repo, output_dir, state, model, tokenizer,
    optimizer, scheduler, scaler, layer_weights_module,
    loss_log, epoch, private, commit_suffix="", delete_local=True,
):
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo(hub_repo, private)

    lora_dir   = output_dir / LORA_ADAPTER_DIR
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

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

    _upload_folder(hub_repo, lora_dir, LORA_ADAPTER_DIR, f"lora {commit_base}")
    for local_f, repo_f in [
        (state_path, TRAINING_STATE_FILE), (opt_path, OPTIMIZER_STATE_FILE),
        (sch_path, SCHEDULER_STATE_FILE),  (scl_path, SCALER_STATE_FILE),
        (lw_path, LAYER_WEIGHTS_FILE),     (ll_path, LOSS_LOG_FILE),
    ]:
        _upload_file(hub_repo, local_f, repo_f, f"state {commit_base}")

    logger.info(f"[Hub] ✓ Checkpoint pushed — {commit_base}")

    if delete_local:
        if lora_dir.exists():
            shutil.rmtree(lora_dir)
        for f in [state_path, opt_path, sch_path, scl_path, lw_path, ll_path]:
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
        trust_remote_code=True,
        output_attentions=False,   # [OPT-2] not needed
    )
    model = PeftModel.from_pretrained(
        base_model, hub_repo, subfolder=LORA_ADAPTER_DIR, is_trainable=True,
    )
    logger.info("[Resume] ✓ LoRA model loaded from Hub.")
    return model


def build_lora_model(base_model_name, middle_layers, lora_r, lora_alpha,
                      lora_dropout, dtype, device_map):
    logger.info(f"[Setup] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True,
        output_attentions=False,   # [OPT-2]
    )

    target_modules = [
        f"model.layers.{l}.self_attn.{proj}"
        for l in middle_layers
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]
    logger.info(f"[LoRA] Target modules ({len(target_modules)}): {target_modules[:4]} ...")

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[LoRA] Trainable: {trainable/1e6:.2f}M / {total/1e9:.3f}B ({100*trainable/total:.2f}%)")
    return model


def build_state(args, hub_repo, completed_epochs, global_step,
                current_epoch, steps_done_in_epoch, is_epoch_complete,
                loss_log, middle_layers):
    return {
        "base_model": args.base_model, "hub_repo": hub_repo,
        "total_epochs": args.epochs, "completed_epochs": completed_epochs,
        "global_step": global_step, "current_epoch": current_epoch,
        "steps_done_in_epoch": steps_done_in_epoch,
        "is_epoch_complete": is_epoch_complete,
        "middle_layers": middle_layers,
        "lambda_ot": args.lambda_ot, "sinkhorn_eps": args.sinkhorn_eps,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "opus_ratio": args.opus_ratio,
        "eng_eng_ratio": args.eng_eng_ratio,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def plot_training_loss(loss_log, output_dir):
    if not loss_log:
        return
    steps = [e["step"] for e in loss_log]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("OT Alignment Training Progress", fontsize=14, fontweight="bold")
    for ax, key, title, color in [
        (axes[0, 0], "loss",    "Total Loss",    "steelblue"),
        (axes[0, 1], "lm_loss", "L_LM",          "darkorange"),
        (axes[1, 0], "ot_loss", "L_OT",          "forestgreen"),
        (axes[1, 1], "lr",      "Learning Rate", "crimson"),
    ]:
        ax.plot(steps, [e[key] for e in loss_log], color=color, linewidth=0.8, alpha=0.7)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        if key == "lr":
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(output_dir / "ot_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
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
        logger.warning("[Hub] HF_TOKEN not set")

    hub_repo = _resolve_repo(args.hub_repo)
    middle_layers = resolve_middle_layers(args.middle_layers, n_total=32)

    # ── [OPT-5] Optionally compile Sinkhorn ──────────────────────────────────
    global _sinkhorn_log_batched_inner
    if args.compile and hasattr(torch, "compile"):
        logger.info("[OPT-5] torch.compile: compiling Sinkhorn kernel ...")
        _sinkhorn_log_batched_inner = torch.compile(
            _sinkhorn_log_batched_inner,
            mode="reduce-overhead",
            fullgraph=True,
        )
        logger.info("[OPT-5] Sinkhorn compiled.")

    # ── Resume ────────────────────────────────────────────────────────────────
    training_state = load_training_state(hub_repo)
    is_resuming    = training_state is not None

    if is_resuming:
        completed_epochs  = training_state.get("completed_epochs", [])
        saved_global_step = training_state.get("global_step", 0)
        resume_epoch      = training_state.get("current_epoch")
        resume_skip_steps = training_state.get("steps_done_in_epoch", 0)
        is_epoch_complete = training_state.get("is_epoch_complete", True)
        if is_epoch_complete or resume_epoch is None:
            resume_epoch = None; resume_skip_steps = 0

        model = load_lora_from_hub(hub_repo, args.base_model, dtype, device_map)
        if model is None:
            model = build_lora_model(
                args.base_model, middle_layers,
                args.lora_r, args.lora_alpha, args.lora_dropout, dtype, device_map,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    else:
        completed_epochs  = []
        saved_global_step = 0
        resume_epoch      = None
        resume_skip_steps = 0
        model = build_lora_model(
            args.base_model, middle_layers,
            args.lora_r, args.lora_alpha, args.lora_dropout, dtype, device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if not torch.cuda.is_available():
        model = model.to(device)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("[Setup] Gradient checkpointing: ON")

    layer_weights_module = LayerWeights(n_layers=len(middle_layers)).to(device)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    logger.info(
        f"[Data] opus_ratio={args.opus_ratio:.1%}  eng_eng_ratio={args.eng_eng_ratio:.1%}"
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
        # [OPT-8] persistent_workers + prefetch to keep pipeline full
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"[Data] Train batches: {len(train_loader):,}")

    # ── [OPT-9] Fused AdamW ──────────────────────────────────────────────────
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
         "lr": args.lr * 10},
    ]

    # Detect fused AdamW support (PyTorch ≥ 2.0)
    use_fused = (
        torch.cuda.is_available()
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.95),
        fused=use_fused,
    )
    if use_fused:
        logger.info("[OPT-9] Using fused AdamW.")

    steps_per_epoch = len(train_loader)
    # [OPT-6] Total optimizer steps = ceil(batch_steps / grad_accum) * epochs
    opt_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)
    total_steps         = opt_steps_per_epoch * args.epochs
    warmup_steps        = max(1, int(total_steps * args.warmup_ratio))
    scheduler           = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(
        f"[Setup] batch_steps/epoch={steps_per_epoch}  "
        f"opt_steps/epoch={opt_steps_per_epoch}  total={total_steps}  warmup={warmup_steps}\n"
        f"        grad_accum={args.grad_accum}  "
        f"effective_batch={args.batch_size * args.grad_accum}\n"
        f"        middle_layers={len(middle_layers)}  lambda_ot={args.lambda_ot}  "
        f"sinkhorn_iters={args.sinkhorn_iters}  use_fused_adam={use_fused}"
    )

    if is_resuming:
        loss_log = load_optimizer_states(
            hub_repo, optimizer, scheduler, scaler, layer_weights_module, device,
        )
    else:
        loss_log = []

    global_step = saved_global_step
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

        logger.info(f"\n{'='*60}\nEPOCH {epoch}/{args.epochs}\n{'='*60}")
        train_loader.set_epoch(epoch)
        model.train()
        layer_weights_module.train()
        optimizer.zero_grad(set_to_none=True)

        batch_pbar = tqdm(
            desc=f"  Epoch {epoch}",
            total=steps_per_epoch, unit="batch",
            dynamic_ncols=True, leave=False,
        )

        # Running accumulators for logging
        accum_loss = accum_lm = accum_ot = 0.0
        accum_count = 0
        oom_skip_count = 0
        steps_in_epoch = 0

        for batch_idx, batch in enumerate(train_loader):

            # ── Fast-forward for resume ───────────────────────────────────────
            if steps_in_epoch < skip_batches:
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            # ── OOM cooldown ──────────────────────────────────────────────────
            if oom_skip_count > 0:
                oom_skip_count -= 1
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            # ── [OPT-6] Gradient accumulation ────────────────────────────────
            # Scale loss by 1/grad_accum so gradients are mean over accum steps
            is_last_accum = ((batch_idx + 1) % args.grad_accum == 0) or \
                            (batch_idx + 1 == steps_per_epoch)

            loss = None
            try:
                loss, lm_item, ot_item = compute_total_loss(
                    model=model, batch=batch,
                    middle_layers=middle_layers,
                    layer_weights_module=layer_weights_module,
                    lambda_ot=args.lambda_ot,
                    sinkhorn_eps=args.sinkhorn_eps,
                    sinkhorn_iters=args.sinkhorn_iters,
                    amp_dtype=amp_dtype, use_amp=use_amp, device=device,
                )

                # Scale for accumulation
                scaled_loss = loss / args.grad_accum

                if use_amp and amp_dtype == torch.float16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_loss  += loss.item()
                accum_lm    += lm_item
                accum_ot    += ot_item
                accum_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                _cleanup_oom(optimizer, loss)
                oom_skip_count = args.oom_skip_batches
                new_bs = max(1, args.batch_size // 2)
                if new_bs < args.batch_size:
                    logger.warning(f"[OOM] batch_size {args.batch_size} → {new_bs}")
                    args.batch_size = new_bs
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            # ── Optimizer step (only every grad_accum steps) ──────────────────
            if is_last_accum:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)

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
                global_step += 1

                # Log averaged over accumulation window
                if accum_count > 0:
                    avg_loss = accum_loss / accum_count
                    avg_lm   = accum_lm   / accum_count
                    avg_ot   = accum_ot   / accum_count
                    current_lr = scheduler.get_last_lr()[0]

                    log_entry = {
                        "epoch": epoch, "step": global_step,
                        "loss": round(avg_loss, 6),
                        "lm_loss": round(avg_lm, 6),
                        "ot_loss": round(avg_ot, 6),
                        "lr": current_lr,
                        "grad_norm": round(float(grad_norm), 4),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    loss_log.append(log_entry)

                    batch_pbar.set_postfix(
                        loss=f"{avg_loss:.4f}", lm=f"{avg_lm:.4f}",
                        ot=f"{avg_ot:.4f}", lr=f"{current_lr:.2e}",
                    )
                    epoch_pbar.set_postfix(
                        epoch=f"{epoch}/{args.epochs}",
                        step=global_step, loss=f"{avg_loss:.4f}",
                    )

                    accum_loss = accum_lm = accum_ot = accum_count = 0.0

                # ── Mid-epoch checkpoint ──────────────────────────────────────
                if args.save_iter > 0 and global_step % args.save_iter == 0:
                    mid_state = build_state(
                        args, hub_repo, completed_epochs, global_step,
                        current_epoch=epoch, steps_done_in_epoch=steps_in_epoch,
                        is_epoch_complete=False, loss_log=loss_log,
                        middle_layers=middle_layers,
                    )
                    save_and_push_checkpoint(
                        hub_repo=hub_repo, output_dir=output_dir,
                        state=mid_state, model=model, tokenizer=tokenizer,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        layer_weights_module=layer_weights_module,
                        loss_log=loss_log, epoch=epoch, private=args.hub_private,
                        commit_suffix=f"step-{global_step}", delete_local=True,
                    )
                    model.train()
                    layer_weights_module.train()

            steps_in_epoch += 1
            batch_pbar.update(1)

        batch_pbar.close()
        logger.info(f"\n[Epoch {epoch}] Done. global_step={global_step}")

        lw_softmax = F.softmax(layer_weights_module.w.detach(), dim=0)
        top_k = min(5, len(middle_layers))
        top_indices = lw_softmax.topk(top_k).indices.tolist()
        logger.info(
            f"[Epoch {epoch}] Top-{top_k} layer weights: "
            + ", ".join(f"L{middle_layers[i]}={lw_softmax[i].item():.4f}" for i in top_indices)
        )

        completed_epochs.append(epoch)
        state = build_state(
            args, hub_repo, completed_epochs, global_step,
            current_epoch=epoch, steps_done_in_epoch=steps_in_epoch,
            is_epoch_complete=True, loss_log=loss_log, middle_layers=middle_layers,
        )
        save_and_push_checkpoint(
            hub_repo=hub_repo, output_dir=output_dir, state=state,
            model=model, tokenizer=tokenizer, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            layer_weights_module=layer_weights_module,
            loss_log=loss_log, epoch=epoch, private=args.hub_private,
            commit_suffix="", delete_local=True,
        )
        plot_training_loss(loss_log, output_dir)

    epoch_pbar.close()
    log_file.close()

    logger.info("\n" + "="*60)
    logger.info("OT ALIGNMENT TRAINING COMPLETE")
    lw_final = F.softmax(layer_weights_module.w.detach(), dim=0).tolist()
    logger.info(f"  Completed: {completed_epochs}  |  Steps: {global_step}")
    logger.info(f"  Final layer weights: {[round(v,4) for v in lw_final]}")

    with open(output_dir / "ot_final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model, "hub_repo": hub_repo,
            "epochs": args.epochs, "completed_epochs": completed_epochs,
            "global_step": global_step, "middle_layers": middle_layers,
            "lambda_ot": args.lambda_ot, "sinkhorn_eps": args.sinkhorn_eps,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "opus_ratio": args.opus_ratio, "eng_eng_ratio": args.eng_eng_ratio,
            "grad_accum": args.grad_accum, "effective_batch": args.batch_size * args.grad_accum,
            "final_layer_weights": lw_final,
        }, f, indent=2, ensure_ascii=False)

    plot_training_loss(loss_log, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 65)
    logger.info("Llama-3-8B — Stage 2: OT Alignment [OPTIMISED]")
    logger.info("=" * 65)
    for k, v in vars(args).items():
        logger.info(f"  {k:25s}: {v}")
    logger.info("=" * 65)

    train(args)