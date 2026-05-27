#!/usr/bin/env python3
"""
Profile activations on the pretrained HRM to calibrate SNN thresholds.

Hooks SwiGLU gate outputs (silu activations), attention outputs, and H/L
reasoning-module state vectors; runs synthetic forward passes; saves per-layer
statistics (mean / max / p99 of |activation|) to profiling/activation_stats.json.

Run from the project root (shrm/):
    conda run -n nn python profiling/profile_activations.py
"""
import sys
import os
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent      # .../shrm/
HRM_SRC = ROOT / "HRM"
sys.path.insert(0, str(HRM_SRC))

# ── Stub flash_attn before HRM imports it ────────────────────────────────────
# MPS and CPU don't have flash_attn; replace with vanilla SDPA.
def _sdpa_compat(q, k, v, causal=False, **_):
    # q/k/v arrive as [B, S, H, D]; SDPA expects [B, H, S, D]
    orig_dtype = q.dtype
    q = q.transpose(1, 2).to(torch.float32)
    k = k.transpose(1, 2).to(torch.float32)
    v = v.transpose(1, 2).to(torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).contiguous().to(orig_dtype)   # back to [B, S, H, D]

_fa_stub = types.ModuleType("flash_attn")
_fa_stub.flash_attn_func = _sdpa_compat           # type: ignore[attr-defined]
sys.modules.setdefault("flash_attn_interface", _fa_stub)
sys.modules.setdefault("flash_attn", _fa_stub)

# ── HRM model imports ─────────────────────────────────────────────────────────
from models.hrm.hrm_act_v1 import (            # noqa: E402
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1_Inner,
)
from models.layers import SwiGLU, Attention     # noqa: E402

# ── Constants (from checkpoint inspection + dataset build script) ─────────────
VOCAB_SIZE           = 12        # PAD + EOS + colors 0-9
SEQ_LEN              = 900       # 30 × 30 flattened ARC grid
NUM_PUZZLE_IDS       = 1_045_829 # from checkpoint puzzle_emb.weights.shape[0]
HIDDEN_SIZE          = 512
H_LAYERS             = 4
L_LAYERS             = 4
H_CYCLES             = 2
L_CYCLES             = 2
NUM_HEADS            = 8
EXPANSION            = 4
HALT_MAX_STEPS       = 16

PROFILE_BATCH_SIZE   = 4
PROFILE_NUM_BATCHES  = 8

CKPT_BLOB = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sapientinc--HRM-checkpoint-ARC-2"
    / "blobs"
    / "58719e51da6cd7712eb4197f908fbfdc88403ee48c42a3a92bab9f9c968df64d"
)
OUT_PATH = ROOT / "profiling" / "activation_stats.json"


# ── Model construction ───────────────────────────────────────────────────────
def build_model(device: torch.device) -> HierarchicalReasoningModel_ACTV1_Inner:
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=PROFILE_BATCH_SIZE,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_puzzle_identifiers=NUM_PUZZLE_IDS,
        puzzle_emb_ndim=HIDDEN_SIZE,
        H_cycles=H_CYCLES,
        L_cycles=L_CYCLES,
        H_layers=H_LAYERS,
        L_layers=L_LAYERS,
        hidden_size=HIDDEN_SIZE,
        expansion=EXPANSION,
        num_heads=NUM_HEADS,
        pos_encodings="rope",
        halt_max_steps=HALT_MAX_STEPS,
        halt_exploration_prob=0.1,
    )
    with torch.device(device):
        model = HierarchicalReasoningModel_ACTV1_Inner(config)
    return model


def load_checkpoint(model: HierarchicalReasoningModel_ACTV1_Inner) -> None:
    state = torch.load(CKPT_BLOB, map_location="cpu", weights_only=False)
    prefix = "_orig_mod.model.inner."
    trimmed = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(trimmed, strict=True)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    print(f"  Loaded {len(trimmed)} tensors from checkpoint.")


# ── Activation statistics accumulator ───────────────────────────────────────
class Collector:
    """Collects flattened |activation| tensors per named layer."""

    def __init__(self):
        self._bufs: dict[str, list[torch.Tensor]] = {}

    def push(self, name: str, tensor: torch.Tensor) -> None:
        flat = tensor.detach().float().abs().flatten().cpu()
        self._bufs.setdefault(name, []).append(flat)

    _MAX_QUANTILE_ELEMS = 500_000

    def stats(self) -> dict:
        result = {}
        for name, chunks in self._bufs.items():
            vals = torch.cat(chunks)
            n = vals.numel()
            if n > self._MAX_QUANTILE_ELEMS:
                idx = torch.randperm(n)[:self._MAX_QUANTILE_ELEMS]
                sample = vals[idx]
            else:
                sample = vals
            result[name] = {
                "mean": float(vals.mean()),
                "max":  float(vals.max()),
                "p99":  float(torch.quantile(sample, 0.99)),
                "n_elements": n,
            }
        return result


# ── Hook registration ─────────────────────────────────────────────────────────
def register_hooks(model: HierarchicalReasoningModel_ACTV1_Inner, collector: Collector):
    handles = []

    for mod_name in ("H_level", "L_level"):
        reasoning_mod = getattr(model, mod_name)

        # Per-block hooks
        for i, block in enumerate(reasoning_mod.layers):
            layer_id = f"{mod_name}.layer{i}"

            # SwiGLU: capture silu(gate) — the activation that becomes a spike rate
            mlp: SwiGLU = block.mlp

            def _swiglu_hook(mod, inp, out, _tag=f"{layer_id}.mlp"):
                collector.push(f"{_tag}.output", out)
                # Recompute gate inside no_grad to capture the silu values
                with torch.no_grad():
                    gate, up = mod.gate_up_proj(inp[0]).chunk(2, dim=-1)
                    collector.push(f"{_tag}.silu_gate", F.silu(gate))
                    collector.push(f"{_tag}.gate_up_product", F.silu(gate) * up)

            handles.append(mlp.register_forward_hook(_swiglu_hook))

            # Attention output
            attn: Attention = block.self_attn

            def _attn_hook(mod, inp, out, _tag=f"{layer_id}.attn"):
                collector.push(f"{_tag}.output", out)

            handles.append(attn.register_forward_hook(_attn_hook))

        # Reasoning module state (z_H or z_L) after each full module pass
        def _state_hook(mod, inp, out, _tag=f"{mod_name}.state"):
            collector.push(_tag, out)

        handles.append(reasoning_mod.register_forward_hook(_state_hook))

    return handles


# ── Synthetic batch ───────────────────────────────────────────────────────────
def make_batch(batch_size: int, device: torch.device) -> dict:
    # Random ARC-like token ids: 0=PAD, 1=EOS, 2-11=colors
    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), dtype=torch.int32, device=device)
    puzzle_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)  # blank id=0
    return {"inputs": inputs, "puzzle_identifiers": puzzle_ids}


# ── Main ──────────────────────────────────────────────────────────────────────
def profile():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building model...")
    model = build_model(device)
    model.eval()

    print("Loading checkpoint...")
    load_checkpoint(model)
    model.to(device)

    collector = Collector()
    handles = register_hooks(model, collector)

    print(f"Running {PROFILE_NUM_BATCHES} forward passes (batch_size={PROFILE_BATCH_SIZE})...")
    with torch.no_grad():
        for i in range(PROFILE_NUM_BATCHES):
            batch = make_batch(PROFILE_BATCH_SIZE, device)

            # Initialise carry from H_init / L_init (same as training: start halted)
            carry = model.empty_carry(PROFILE_BATCH_SIZE)
            carry = HierarchicalReasoningModel_ACTV1InnerCarry(
                z_H=carry.z_H.to(device),
                z_L=carry.z_L.to(device),
            )
            reset_all = torch.ones(PROFILE_BATCH_SIZE, dtype=torch.bool, device=device)
            carry = model.reset_carry(reset_all, carry)

            _carry_out, _logits, _ = model(carry, batch)

            if (i + 1) % 2 == 0 or i == 0:
                print(f"  batch {i + 1}/{PROFILE_NUM_BATCHES}")

    for h in handles:
        h.remove()

    stats = collector.stats()
    print(f"\nActivation stats ({len(stats)} layers):")
    for name, s in sorted(stats.items()):
        print(f"  {name:<55}  mean={s['mean']:.4f}  max={s['max']:.4f}  p99={s['p99']:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    profile()
