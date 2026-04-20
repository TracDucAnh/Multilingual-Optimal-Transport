"""
Llama3-8B-OT-only.py
==========================
Stage 2 — Cross-lingual Optimal Transport Alignment (OT Loss Only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPER FORMULATIONS STRICTLY IMPLEMENTED (LM LOSS REMOVED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Eq. 11-12  : Two forward branches — LoRA (target) vs frozen (dominant/EN)
Eq. 13     : H̃(l) = diag(α(l)) · H(l)  where α(l) = mean attention over heads
Eq. 14     : Row-wise L2 normalise of H̃(l) to get h(l)_tgt,i  and h(l)_dom,j
Eq. 15     : Uniform marginals  a_i = 1/s_tgt,  b_j = 1/s_en
Eq. 16     : C(l)_ij = 1 − ⟨h(l)_tgt,i , h(l)_dom,j⟩  (cosine distance on unit vecs)
Eq. 17     : OT(l) = ⟨C(l), T*(l)⟩_F    (plain Frobenius inner product, no division)
Eq. 18     : L_OT = Σ_l softmax(w)_l · OT(l)    (learnable layer weights)
             OR (--mean_layer True): L_OT = mean_l OT(l)   (uniform mean, no collapse)

CHANGE vs original:
  • L_LM completely removed — no causal LM forward, no labels, no logits
  • Loss objective: L = L_OT  (no lambda, no weighting)
  • Target branch no longer needs output logits, only hidden states + attentions
  • Default lr raised to 1e-4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU OPTIMISATIONS (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[OPT-1] BATCHED SINKHORN — fully vectorised over batch dimension
[OPT-2] output_attentions=True ONLY for target branch (needed for eq.13)
        Dominant/EN branch uses output_attentions=False + uniform fallback
        to avoid doubling memory; OR set --real_attn_en to use real attn.
[OPT-3] SINGLE FORWARD PASS per branch (no redundant calls)
[OPT-4] torch.cdist / torch.bmm for cost matrix — no Python loops
[OPT-5] torch.compile on Sinkhorn kernel (PyTorch ≥ 2.0, opt-in)
[OPT-6] GRADIENT ACCUMULATION with proper 1/grad_accum scaling
[OPT-7] FROZEN BRANCH under torch.inference_mode() + no_grad
[OPT-8] DataLoader persistent_workers + prefetch_factor
[OPT-9] FUSED ADAMW
[OPT-10] Loss computed in float32; backward in bf16 (autocast)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python Llama3-8B-OT-only.py \\
        --base_model    ducanhdinh/Llama3-8B-Finetune \\
        --data_root     ../raw_data/alignment/ \\
        --output_dir    ./ot_checkpoints \\
        --hub_repo      Llama3-8B-OT \\
        --epochs        3 \\
        --batch_size    8 \\
        --grad_accum    4 \\
        --lr            1e-4 \\
        --sinkhorn_eps  0.1 \\
        --opus_ratio    0.05 \\
        --eng_eng_ratio 0.30 \\
        --seq_length    512 \\
        --save_iter     200 \\
        --mean_layer    False
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
import matplotlib.gridspec as gridspec
import numpy as np

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
    p = argparse.ArgumentParser(description="Llama-3-8B OT Alignment — OT Loss Only")

    p.add_argument("--base_model",      type=str, default="ducanhdinh/Llama3-8B-Finetune")
    p.add_argument("--data_root",       type=str, default="../raw_data/alignment/")
    p.add_argument("--output_dir",      type=str, default="./ot_checkpoints_no_L_LM")
    p.add_argument("--hub_repo",        type=str, default="Llama3-8B-OT_no_L_LM")
    p.add_argument("--hub_private",     action="store_true")

    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--grad_accum",      type=int,   default=4)
    p.add_argument("--lr",              type=float, default=1e-4)   # raised from 2e-5
    p.add_argument("--warmup_ratio",    type=float, default=0.03)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    # OT hyper-params  (lambda_ot REMOVED — loss is pure L_OT)
    p.add_argument("--sinkhorn_eps",    type=float, default=0.1)
    p.add_argument("--sinkhorn_iters",  type=int,   default=50)

    p.add_argument(
        "--middle_layers", type=str, default="auto",
        help="'auto' → layers[N/4 .. 3N/4). Or comma list: '8,12,16,20'",
    )

    # LoRA config
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)

    p.add_argument("--opus_ratio",      type=float, default=0.05)
    p.add_argument("--eng_eng_ratio",   type=float, default=0.0)

    p.add_argument("--save_iter",       type=int,   default=0)

    # Sequence length
    p.add_argument("--seq_length",      type=int,   default=None)
    p.add_argument("--max_length",      type=int,   default=256)

    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--bf16",            action="store_true", default=True)
    p.add_argument("--fp16",            action="store_true")
    p.add_argument("--oom_skip_batches",type=int,   default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing",
                   dest="gradient_checkpointing", action="store_false")
    p.add_argument("--compile",         action="store_true", default=False)

    p.add_argument("--real_attn_en",    action="store_true", default=False,
                   help="Use real attention weights for EN branch (more memory).")

    p.add_argument(
        "--mean_layer",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        metavar="BOOL",
        help=(
            "If True, replace learnable softmax layer weights (eq.18) with "
            "uniform mean pooling across middle layers. Prevents softmax "
            "collapse. Default: False (paper-faithful learnable weights)."
        ),
    )

    p.add_argument("--plot_smooth",     type=int,   default=20)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
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
    logger.info(f"[OT] Middle layers ({len(layers)}): {layers}")
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# [OPT-1, OPT-5]  Log-domain Sinkhorn — batched, GPU-vectorised
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_log_batched_inner(
    C: torch.Tensor,          # [B, m, n]  float32
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
        log_sum_f = torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=2)  # [B, m]
        f = eps * log_a - eps * log_sum_f

        log_sum_g = torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=1)  # [B, n]
        g = eps * log_b - eps * log_sum_g

    log_T = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps  # [B, m, n]
    return torch.exp(log_T)


def sinkhorn_log_batched(C: torch.Tensor, eps: float, max_iter: int) -> torch.Tensor:
    return _sinkhorn_log_batched_inner(C.float(), eps, max_iter)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-Weighted Pooling  —  eqs. (13)–(14)
# ─────────────────────────────────────────────────────────────────────────────

def attention_weighted_pool_from_attn(
    hidden: torch.Tensor,           # [B, s, d]
    attn_weights: torch.Tensor,     # [B, H, s, s]
    attention_mask: torch.Tensor,   # [B, s]
) -> torch.Tensor:
    alpha = attn_weights.float().mean(dim=1).mean(dim=1)   # [B, s]
    alpha = alpha * attention_mask.float()
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)
    h_tilde = hidden * alpha.unsqueeze(-1)                  # [B, s, d]
    h_norm  = F.normalize(h_tilde, p=2, dim=-1)
    return h_norm


def attention_weighted_pool_uniform(
    hidden: torch.Tensor,           # [B, s, d]
    attention_mask: torch.Tensor,   # [B, s]
) -> torch.Tensor:
    mask   = attention_mask.float()
    n_real = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    alpha  = mask / n_real
    h_tilde = hidden * alpha.unsqueeze(-1)
    h_norm  = F.normalize(h_tilde, p=2, dim=-1)
    return h_norm


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer OT Loss  —  eqs. (16)–(17)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ot_loss_single_layer(
    h_tgt: torch.Tensor,
    h_en:  torch.Tensor,
    tgt_mask: torch.Tensor,
    en_mask:  torch.Tensor,
    eps:      float,
    max_iter: int,
) -> torch.Tensor:
    C = 1.0 - torch.bmm(
        h_tgt.float(),
        h_en.float().transpose(1, 2),
    )
    C = C.clamp(0.0, 2.0)

    valid = tgt_mask.float().unsqueeze(2) * en_mask.float().unsqueeze(1)
    C_masked = C * valid + (1.0 - valid) * 2.0

    T = sinkhorn_log_batched(C_masked, eps=eps, max_iter=max_iter)

    ot_per_sample = (C * T * valid).sum(dim=(1, 2))
    return ot_per_sample.mean()


# ─────────────────────────────────────────────────────────────────────────────
# LayerWeights  —  eq. (18)
# Used only when --mean_layer False (default).
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
# NOTE: target branch no longer needs logits — use_cache=False kept for safety,
#       output_attentions=True kept for eq.13, but we discard logits entirely.
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
    Returns only layer_data (hidden + attn) — logits discarded (no L_LM).
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
        out.attentions is not None
        and len(out.attentions) > 0
    )
    if not has_attentions:
        logger.warning(
            "[forward_target_branch] out.attentions is empty — "
            "the model was likely loaded without attn_implementation='eager'. "
            "Falling back to uniform attention pooling for this batch."
        )

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        attn   = out.attentions[l] if has_attentions else None
        layer_data[l] = (hidden, attn)

    # logits intentionally NOT returned
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
    Dominant (frozen EN) branch — eq. (12). Unchanged.
    """
    model.disable_adapter_layers()
    try:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                output_attentions=real_attn_en,
                use_cache=False,
            )
    finally:
        model.enable_adapter_layers()

    has_attentions = (
        real_attn_en
        and out.attentions is not None
        and len(out.attentions) > 0
    )

    layer_data: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for l in middle_layers:
        hidden = out.hidden_states[l + 1]
        attn   = out.attentions[l] if has_attentions else None
        layer_data[l] = (hidden, attn)

    return layer_data


# ─────────────────────────────────────────────────────────────────────────────
# Full loss  —  L = L_OT  (L_LM removed entirely, no lambda)
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
    """
    Returns:
        loss     : L_OT scalar tensor (with grad)
        ot_item  : float for logging
    NOTE: tgt_labels no longer read — batch may still contain them (dataloader
          compat) but they are ignored here.
    """
    tgt_ids  = batch["tgt_input_ids"].to(device, non_blocking=True)
    tgt_mask = batch["tgt_attention_mask"].to(device, non_blocking=True)
    en_ids   = batch["en_input_ids"].to(device, non_blocking=True)
    en_mask  = batch["en_attention_mask"].to(device, non_blocking=True)

    # ── TARGET branch (eq. 11) — only hidden states + attentions ────────────
    tgt_layer_data = forward_target_branch(
        model, tgt_ids, tgt_mask, middle_layers, amp_dtype, use_amp, device,
    )

    # ── DOMINANT branch (eq. 12) ─────────────────────────────────────────────
    en_layer_data = forward_dominant_branch(
        model, en_ids, en_mask, middle_layers, real_attn_en, amp_dtype, use_amp, device,
    )

    # ── Per-layer OT losses (eqs. 13–17) ─────────────────────────────────────
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

    # ── Eq. 18: aggregate across layers ─────────────────────────────────────
    if ot_losses:
        if mean_layer:
            loss_ot = torch.stack(ot_losses).mean()
        else:
            loss_ot = layer_weights_module(ot_losses)
    else:
        loss_ot = torch.tensor(0.0, device=device, requires_grad=True)

    # L = L_OT  (no L_LM, no lambda)
    return loss_ot, loss_ot.item()


# ─────────────────────────────────────────────────────────────────────────────
# OOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
    )


def _cleanup_oom(optimizer, *tensors):
    for t in tensors:
        try: del t
        except Exception: pass
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Loss plotting  (lm_loss panel replaced with blank / note)
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 2:
        return values
    arr     = np.array(values, dtype=np.float64)
    kernel  = np.ones(window) / window
    padded  = np.concatenate([arr[:window - 1], arr])
    return np.convolve(padded, kernel, mode="valid").tolist()


def plot_training_loss(loss_log: List[Dict], output_dir: Path, smooth: int = 20) -> None:
    if not loss_log:
        return

    steps   = [e["step"]    for e in loss_log]
    ot_loss = [e["ot_loss"] for e in loss_log]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(
        f"OT-Only Alignment — Training Progress  (step {steps[-1]})",
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
    logger.info(f"[Plot] ✓ Loss chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# OT Transport Diagnostic  (unchanged logic, no lm references)
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_numpy(C: np.ndarray, eps: float, n_iter: int = 100) -> np.ndarray:
    m, n   = C.shape
    log_a  = np.full(m, -np.log(m))
    log_b  = np.full(n, -np.log(n))
    f, g   = np.zeros(m), np.zeros(n)
    for _ in range(n_iter):
        f = eps * log_a - eps * np.logaddexp.reduce((g[None, :] - C) / eps, axis=1)
        g = eps * log_b - eps * np.logaddexp.reduce((f[:, None] - C) / eps, axis=0)
    return np.exp((f[:, None] + g[None, :] - C) / eps)


def _decode_tokens(tokenizer, ids: torch.Tensor) -> List[str]:
    out = []
    for id_ in ids.tolist():
        tok = tokenizer.decode([id_], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False).strip()
        if not tok:
            tok = f"[{id_}]"
        out.append(tok)
    return out


@torch.no_grad()
def plot_ot_transport_diagnostic(
    model,
    tokenizer,
    batch:                Dict[str, torch.Tensor],
    middle_layers:        List[int],
    layer_weights_module: LayerWeights,
    sinkhorn_eps:         float,
    global_step:          int,
    epoch:                int,
    output_dir:           Path,
    real_attn_en:         bool,
    mean_layer:           bool,
    amp_dtype:            torch.dtype,
    use_amp:              bool,
    device:               torch.device,
    sample_idx:           int = 0,
) -> Path:
    model.eval()

    tgt_ids  = batch["tgt_input_ids"]
    tgt_mask = batch["tgt_attention_mask"]
    en_ids   = batch["en_input_ids"]
    en_mask  = batch["en_attention_mask"]

    tgt_layer_data = forward_target_branch(
        model, tgt_ids, tgt_mask, middle_layers, amp_dtype, use_amp, device,
    )
    en_layer_data = forward_dominant_branch(
        model, en_ids, en_mask, middle_layers, real_attn_en, amp_dtype, use_amp, device,
    )

    tgt_pooled: Dict[int, np.ndarray] = {}
    en_pooled:  Dict[int, np.ndarray] = {}

    for l in middle_layers:
        tgt_h, tgt_a = tgt_layer_data[l]
        en_h,  en_a  = en_layer_data[l]

        if tgt_a is not None:
            h_t = attention_weighted_pool_from_attn(
                tgt_h, tgt_a, tgt_mask.to(device),
            )[sample_idx].float().cpu().numpy()
        else:
            h_t = attention_weighted_pool_uniform(
                tgt_h, tgt_mask.to(device),
            )[sample_idx].float().cpu().numpy()

        if en_a is not None:
            h_e = attention_weighted_pool_from_attn(
                en_h, en_a, en_mask.to(device),
            )[sample_idx].float().cpu().numpy()
        else:
            h_e = attention_weighted_pool_uniform(
                en_h, en_mask.to(device),
            )[sample_idx].float().cpu().numpy()

        tgt_pooled[l] = h_t
        en_pooled[l]  = h_e

    n_layers = len(middle_layers)
    if mean_layer:
        layer_w = np.ones(n_layers, dtype=np.float32) / n_layers
    else:
        layer_w = F.softmax(layer_weights_module.w.detach().float(), dim=0).cpu().numpy()

    h_tgt_agg = sum(layer_w[li] * tgt_pooled[l]
                    for li, l in enumerate(middle_layers))
    h_en_agg  = sum(layer_w[li] * en_pooled[l]
                    for li, l in enumerate(middle_layers))

    h_tgt_agg /= np.linalg.norm(h_tgt_agg, axis=1, keepdims=True) + 1e-12
    h_en_agg  /= np.linalg.norm(h_en_agg,  axis=1, keepdims=True) + 1e-12

    tgt_real = int(tgt_mask[sample_idx].sum().item())
    en_real  = int(en_mask[sample_idx].sum().item())

    h_tv = h_tgt_agg[:tgt_real]
    h_ev = h_en_agg[:en_real]
    C_agg   = np.clip(1.0 - np.clip(h_tv @ h_ev.T, -1.0, 1.0), 0.0, 2.0)
    T_agg   = _sinkhorn_numpy(C_agg, eps=sinkhorn_eps)
    ot_agg_scalar = float(np.sum(C_agg * T_agg))

    per_layer_ot: List[float] = []
    for l in middle_layers:
        h_t = tgt_pooled[l][:tgt_real]
        h_e = en_pooled[l][:en_real]
        h_t /= np.linalg.norm(h_t, axis=1, keepdims=True) + 1e-12
        h_e /= np.linalg.norm(h_e, axis=1, keepdims=True) + 1e-12
        C_l = np.clip(1.0 - np.clip(h_t @ h_e.T, -1.0, 1.0), 0.0, 2.0)
        T_l = _sinkhorn_numpy(C_l, eps=sinkhorn_eps)
        per_layer_ot.append(float(np.sum(C_l * T_l)))

    labels_tgt = _decode_tokens(tokenizer, tgt_ids[sample_idx][:tgt_real])
    labels_en  = _decode_tokens(tokenizer, en_ids[sample_idx][:en_real])

    _BG       = "white"
    _PANEL_BG = "#f8f9fa"
    _TEXT     = "#1a1a1a"
    _GRID     = "#cccccc"
    _SPINE    = "#999999"

    n_tgt = len(labels_tgt)
    n_en  = len(labels_en)
    map_w  = max(10, n_tgt * 0.55)
    map_h  = max(8,  n_en  * 0.45)
    fig_w  = map_w + 8
    fig_h  = max(map_h, 10)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            width_ratios=[map_w, 7], height_ratios=[1, 1],
                            hspace=0.50, wspace=0.35)
    ax_map   = fig.add_subplot(gs[:, 0])
    ax_layer = fig.add_subplot(gs[0, 1])
    ax_bar   = fig.add_subplot(gs[1, 1])

    for ax in [ax_map, ax_layer, ax_bar]:
        ax.set_facecolor(_PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE)
        ax.tick_params(colors=_TEXT, labelsize=7)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)

    T_display = T_agg.T

    TICK_x = max(5, min(9, int(160 / max(n_tgt, 1))))
    TICK_y = max(5, min(9, int(160 / max(n_en,  1))))

    im = ax_map.imshow(T_display, cmap="YlOrRd", aspect="auto",
                       interpolation="nearest",
                       vmin=T_display.min(), vmax=T_display.max())
    ax_map.set_xticks(range(n_tgt))
    ax_map.set_xticklabels(labels_tgt, rotation=60, ha="right",
                            fontsize=TICK_x, color=_TEXT)
    ax_map.set_yticks(range(n_en))
    ax_map.set_yticklabels(labels_en, fontsize=TICK_y, color=_TEXT)
    ax_map.set_xlabel("Target tokens (post-LoRA, eq.11)", fontsize=9, color=_TEXT)
    ax_map.set_ylabel("Dominant EN tokens (frozen, eq.12)", fontsize=9, color=_TEXT)
    ax_map.set_title(
        f"Aggregated OT Transport Map  (step {global_step}, epoch {epoch})\n"
        f"OT = {ot_agg_scalar:.5f}   ε={sinkhorn_eps}   |L|={n_layers}",
        fontsize=9, fontweight="bold", color=_TEXT, pad=8,
    )
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, colors=_TEXT)

    if n_tgt <= 30 and n_en <= 30:
        val_fs = max(4, min(7, int(120 / max(n_tgt, n_en))))
        for ri in range(n_en):
            for ci in range(n_tgt):
                v = T_display[ri, ci]
                cell_color = "#fff" if v > 0.6 * T_display.max() else "#333"
                ax_map.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                            fontsize=val_fs, color=cell_color)

    layer_labels = [f"L{l}" for l in middle_layers]
    lw_title = (
        "mean(OT)  —  uniform  —  step {s}"
        if mean_layer else
        "softmax(w)  —  eq.18  —  step {s}"
    ).format(s=global_step)

    im_w = ax_layer.imshow(layer_w.reshape(1, -1), cmap="viridis",
                            aspect="auto", vmin=0, vmax=layer_w.max())
    ax_layer.set_xticks(range(n_layers))
    ax_layer.set_xticklabels(layer_labels, rotation=60, ha="right",
                              fontsize=6, color=_TEXT)
    ax_layer.set_yticks([])
    ax_layer.set_title(lw_title, fontsize=9, fontweight="bold", color=_TEXT, pad=6)
    for li, wv in enumerate(layer_w):
        cell_color = "white" if wv > 0.6 * layer_w.max() else "#222"
        ax_layer.text(li, 0, f"{wv:.3f}", ha="center", va="center",
                      fontsize=5, color=cell_color)
    cbar_w = fig.colorbar(im_w, ax=ax_layer, fraction=0.046, pad=0.04,
                           orientation="horizontal")
    cbar_w.ax.tick_params(labelsize=6, colors=_TEXT)

    bar_colours = plt.cm.viridis(np.linspace(0.15, 0.85, n_layers))
    bars = ax_bar.bar(range(n_layers), per_layer_ot,
                      color=bar_colours, edgecolor=_SPINE, linewidth=0.7)
    for bar, wv in zip(bars, layer_w):
        bar.set_alpha(0.55 + 0.45 * wv / (layer_w.max() + 1e-9))
    ax_bar.set_xticks(range(n_layers))
    ax_bar.set_xticklabels(layer_labels, rotation=60, ha="right",
                            fontsize=6, color=_TEXT)
    ax_bar.set_ylabel("OT(l) = ⟨C(l), T*(l)⟩_F  (eq.17)", fontsize=8, color=_TEXT)
    ax_bar.set_title("Per-layer OT scalar  (eq.17)",
                      fontsize=9, fontweight="bold", color=_TEXT, pad=6)
    ax_bar.grid(axis="y", color=_GRID, linewidth=0.5, alpha=0.8)
    for li, v in enumerate(per_layer_ot):
        ax_bar.text(li, v + 0.0005 * max(per_layer_ot, default=1),
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=5, color=_TEXT, rotation=60)

    mode_label = "mean_layer=True (uniform)" if mean_layer else "mean_layer=False (learnable)"
    fig.suptitle(
        f"OT Diagnostic  —  Transport Map  ·  Layer Weights  ·  Per-layer Loss\n"
        f"Step {global_step}  |  Epoch {epoch}  |  eqs. 13–18  |  {mode_label}",
        fontsize=11, fontweight="bold", color=_TEXT, y=1.01,
    )
    fig.patch.set_facecolor(_BG)

    out_path = output_dir / f"ot_transport_step_{global_step:06d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, pad_inches=0.2)
    plt.close(fig)
    logger.info(f"[VizOT] ✓ → {out_path}")

    model.train()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# HF Hub helpers
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
        logger.info("[Resume] No state — starting fresh.")
        return None
    try:
        with open(local, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            f"[Resume] completed={state.get('completed_epochs', [])}  "
            f"step={state.get('global_step', 0)}"
        )
        return state
    except Exception as e:
        logger.warning(f"[Resume] Parse error: {e}")
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
                logger.info(f"[Resume] ✓ {fname}")
            except Exception as e:
                logger.warning(f"[Resume] {fname} error: {e}")

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
    loss_log, epoch, private, smooth, mean_layer,
    commit_suffix="", delete_local=True,
    transport_plot_path: Optional[Path] = None,
):
    commit_base = f"epoch-{epoch}" + (f"-{commit_suffix}" if commit_suffix else "")
    _ensure_repo(hub_repo, private)

    lora_dir = output_dir / LORA_ADAPTER_DIR
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(lora_dir))
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

    if transport_plot_path is not None and transport_plot_path.exists():
        files_to_upload.append(
            (transport_plot_path, f"diagnostics/{transport_plot_path.name}")
        )
    for local_f, repo_f in files_to_upload:
        if local_f.exists():
            _upload_file(hub_repo, local_f, repo_f, f"state {commit_base}")

    logger.info(f"[Hub] ✓ Checkpoint pushed — {commit_base}")

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
        trust_remote_code=True,
        attn_implementation="eager",
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
        attn_implementation="eager",
    )
    target_modules = [
        f"model.layers.{l}.self_attn.{proj}"
        for l in middle_layers
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]
    logger.info(f"[LoRA] Target modules ({len(target_modules)}): "
                f"{target_modules[:4]} ...")
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[LoRA] Trainable: {trainable/1e6:.2f}M / {total/1e9:.3f}B "
                f"({100*trainable/total:.2f}%)")
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
        # lambda_ot removed
        "sinkhorn_eps": args.sinkhorn_eps,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
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
    set_seed(args.seed)

    if args.seq_length is not None:
        args.max_length = args.seq_length
    else:
        args.seq_length = args.max_length

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16; amp_dtype = torch.bfloat16; use_amp = True
        logger.info("[Setup] bfloat16 AMP")
    elif args.fp16 and torch.cuda.is_available():
        dtype = torch.float16; amp_dtype = torch.float16; use_amp = True
        logger.info("[Setup] float16 AMP")
    else:
        dtype = torch.float32; amp_dtype = torch.float32; use_amp = False
        logger.info("[Setup] float32")

    device_map = "auto" if torch.cuda.is_available() else None

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    hub_repo      = _resolve_repo(args.hub_repo)
    middle_layers = resolve_middle_layers(args.middle_layers, n_total=32)

    if args.mean_layer:
        logger.info("[Setup] Layer aggregation: UNIFORM MEAN (--mean_layer True)")
    else:
        logger.info("[Setup] Layer aggregation: LEARNABLE SOFTMAX (--mean_layer False)")

    logger.info("[Setup] Loss objective: L = L_OT only  (L_LM removed, no lambda)")

    global _sinkhorn_log_batched_inner
    if args.compile and hasattr(torch, "compile"):
        logger.info("[OPT-5] Compiling Sinkhorn kernel ...")
        _sinkhorn_log_batched_inner = torch.compile(
            _sinkhorn_log_batched_inner, mode="reduce-overhead", fullgraph=True,
        )

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

    align_dataset = AlignmentDataset(
        alignment_data_path=args.data_root,
        opus_sample_ratio=args.opus_ratio,
        eng_eng_ratio=args.eng_eng_ratio,
    ).load()
    align_dataset.stats()

    train_loader = AlignmentDataLoader(
        dataset=align_dataset, split="train", source="joint",
        tokenizer=tokenizer, batch_size=args.batch_size,
        max_length=args.max_length, shuffle=True, seed=args.seed,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"[Data] Train batches: {len(train_loader):,}")

    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
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
        logger.info("[OPT-9] Fused AdamW enabled.")

    steps_per_epoch     = len(train_loader)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)
    total_steps         = opt_steps_per_epoch * args.epochs
    warmup_steps        = max(1, int(total_steps * args.warmup_ratio))
    scheduler           = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    logger.info(
        f"[Setup] steps/epoch={steps_per_epoch}  opt_steps/epoch={opt_steps_per_epoch}\n"
        f"        total_steps={total_steps}  warmup={warmup_steps}\n"
        f"        grad_accum={args.grad_accum}  "
        f"eff_batch={args.batch_size * args.grad_accum}\n"
        f"        seq_length={args.max_length}  layers={len(middle_layers)}\n"
        f"        [OT-ONLY] lr={args.lr}  eps={args.sinkhorn_eps}  "
        f"iters={args.sinkhorn_iters}\n"
        f"        mean_layer={args.mean_layer}  "
        f"real_attn_en={args.real_attn_en}  save_iter={args.save_iter}"
    )

    loss_log = load_optimizer_states(
        hub_repo, optimizer, scheduler, scaler,
        layer_weights_module, args.mean_layer, device,
    ) if is_resuming else []

    global_step = saved_global_step
    log_file    = open(
        output_dir / "ot_train_log.jsonl",
        "a" if is_resuming else "w", encoding="utf-8",
    )

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs",
                      unit="epoch", dynamic_ncols=True, colour="green")

    for epoch in epoch_pbar:
        if epoch in completed_epochs:
            logger.info(f"[Resume] Epoch {epoch} already done — skip.")
            continue

        skip_batches = resume_skip_steps if (epoch == resume_epoch) else 0
        logger.info(f"\n{'='*60}\nEPOCH {epoch}/{args.epochs}\n{'='*60}")

        train_loader.set_epoch(epoch)
        model.train()
        if not args.mean_layer:
            layer_weights_module.train()
        optimizer.zero_grad(set_to_none=True)

        batch_pbar = tqdm(desc=f"  Epoch {epoch}", total=steps_per_epoch,
                          unit="batch", dynamic_ncols=True, leave=False)

        accum_ot    = 0.0
        accum_count = 0
        oom_skip_count = 0
        steps_in_epoch = 0
        last_batch_for_viz: Optional[Dict[str, torch.Tensor]] = None

        for batch_idx, batch in enumerate(train_loader):

            if steps_in_epoch < skip_batches:
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            if oom_skip_count > 0:
                oom_skip_count -= 1
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            is_last_accum = (
                (batch_idx + 1) % args.grad_accum == 0
                or (batch_idx + 1 == steps_per_epoch)
            )

            last_batch_for_viz = {k: v.cpu() for k, v in batch.items()
                                  if isinstance(v, torch.Tensor)}

            loss = None
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

                accum_ot    += ot_item
                accum_count += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                _cleanup_oom(optimizer, loss)
                oom_skip_count  = args.oom_skip_batches
                args.batch_size = max(1, args.batch_size // 2)
                logger.warning(f"[OOM] → batch_size={args.batch_size}")
                steps_in_epoch += 1
                batch_pbar.update(1)
                continue

            if is_last_accum:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)

                all_trainable = [p for p in model.parameters() if p.requires_grad]
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
                    avg_ot     = accum_ot / accum_count
                    current_lr = scheduler.get_last_lr()[0]

                    log_entry = {
                        "epoch":     epoch,
                        "step":      global_step,
                        "loss":      round(avg_ot, 6),   # loss == ot_loss here
                        "ot_loss":   round(avg_ot, 6),
                        "lr":        current_lr,
                        "grad_norm": round(float(grad_norm), 4),
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    loss_log.append(log_entry)

                    batch_pbar.set_postfix({
                        "L_OT": f"{avg_ot:.4f}",
                        "lr":   f"{current_lr:.2e}",
                        "gnorm": f"{float(grad_norm):.3f}",
                    })
                    epoch_pbar.set_postfix({
                        "ep": f"{epoch}/{args.epochs}",
                        "step": global_step,
                        "L_OT": f"{avg_ot:.4f}",
                    })
                    accum_ot = accum_count = 0.0

                if args.save_iter > 0 and global_step % args.save_iter == 0:
                    plot_training_loss(loss_log, output_dir, smooth=args.plot_smooth)

                    transport_plot = None
                    if last_batch_for_viz is not None:
                        try:
                            transport_plot = plot_ot_transport_diagnostic(
                                model=model, tokenizer=tokenizer,
                                batch=last_batch_for_viz,
                                middle_layers=middle_layers,
                                layer_weights_module=layer_weights_module,
                                sinkhorn_eps=args.sinkhorn_eps,
                                global_step=global_step, epoch=epoch,
                                output_dir=output_dir,
                                real_attn_en=args.real_attn_en,
                                mean_layer=args.mean_layer,
                                amp_dtype=amp_dtype, use_amp=use_amp, device=device,
                            )
                        except Exception as viz_err:
                            logger.warning(f"[VizOT] failed: {viz_err}")
                        finally:
                            model.train()
                            if not args.mean_layer:
                                layer_weights_module.train()

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
                        smooth=args.plot_smooth, mean_layer=args.mean_layer,
                        commit_suffix=f"step-{global_step}", delete_local=True,
                        transport_plot_path=transport_plot,
                    )
                    model.train()
                    if not args.mean_layer:
                        layer_weights_module.train()

            steps_in_epoch += 1
            batch_pbar.update(1)

        batch_pbar.close()
        logger.info(f"[Epoch {epoch}] Done. global_step={global_step}")

        if not args.mean_layer:
            lw_softmax = F.softmax(layer_weights_module.w.detach(), dim=0)
            top_k      = min(5, len(middle_layers))
            top_idx    = lw_softmax.topk(top_k).indices.tolist()
            logger.info(
                f"[Epoch {epoch}] Top-{top_k} layer weights: "
                + ", ".join(f"L{middle_layers[i]}={lw_softmax[i].item():.4f}"
                            for i in top_idx)
            )
        else:
            logger.info(
                f"[Epoch {epoch}] Layer aggregation: uniform mean over "
                f"{len(middle_layers)} layers (mean_layer=True)"
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
            smooth=args.plot_smooth, mean_layer=args.mean_layer,
            commit_suffix="", delete_local=True,
        )

    epoch_pbar.close()
    log_file.close()

    logger.info("\n" + "=" * 60)
    logger.info("OT-ONLY ALIGNMENT TRAINING COMPLETE")
    logger.info(f"  Completed: {completed_epochs}  |  Steps: {global_step}")

    final_report = {
        "base_model": args.base_model, "hub_repo": hub_repo,
        "epochs": args.epochs, "completed_epochs": completed_epochs,
        "global_step": global_step, "middle_layers": middle_layers,
        # lambda_ot removed
        "sinkhorn_eps": args.sinkhorn_eps,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "opus_ratio": args.opus_ratio, "eng_eng_ratio": args.eng_eng_ratio,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "seq_length": args.max_length, "real_attn_en": args.real_attn_en,
        "mean_layer": args.mean_layer,
        "loss_objective": "L_OT only (L_LM removed)",
    }

    if not args.mean_layer:
        lw_final = F.softmax(layer_weights_module.w.detach(), dim=0).tolist()
        logger.info(f"  Final layer weights: {[round(v, 4) for v in lw_final]}")
        final_report["final_layer_weights"] = lw_final
    else:
        n = len(middle_layers)
        final_report["final_layer_weights"] = [round(1.0 / n, 4)] * n

    with open(output_dir / "ot_final_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    plot_training_loss(loss_log, output_dir, smooth=args.plot_smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    logger.info("=" * 65)
    logger.info("Llama-3-8B  Stage 2: OT-ONLY Alignment  [L_LM removed]")
    logger.info("=" * 65)
    for k, v in vars(args).items():
        logger.info(f"  {k:28s}: {v}")
    logger.info("=" * 65)

    train(args)