"""
HRM → SNN converted model (rate coding, ANN-to-SNN).

Conversion strategy
-------------------
- Input embedding + RoPE positional encoding: kept as float (not spiked).
- H and L reasoning modules: each SwiGLU gate replaced by an LIF neuron.
  The `up` projection and attention remain float.
- Output projection (lm_head / q_head): kept as float.

Forward pass (inference):
1. Compute input embeddings once (float).
2. Run L-module for T_L = 8 SNN timesteps (LIF mems accumulate across steps).
3. Run H-module for T_H = 16 SNN timesteps.
4. lm_head(z_H) → logits.

L stabilises first (as in the original hierarchical design), then H reasons
over the stable L representation.

LIF thresholds
--------------
Set to p99 of |silu(gate)| per layer from profiling/activation_stats.json.
At threshold input the neuron fires every step (rate ≈ 1).
At mean input the neuron fires at rate ≈ mean/threshold < 1.
"""

import sys
import json
import types
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# ── HRM source on path ────────────────────────────────────────────────────────
HRM_SRC = Path(__file__).resolve().parent.parent / "HRM"
sys.path.insert(0, str(HRM_SRC))

# Stub flash_attn so the HRM imports work on MPS / CPU.
def _sdpa(q, k, v, causal=False, **_):
    orig = q.dtype
    q, k, v = (x.transpose(1, 2).to(torch.float32) for x in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).contiguous().to(orig)

_fa = types.ModuleType("flash_attn")
_fa.flash_attn_func = _sdpa                         # type: ignore[attr-defined]
sys.modules.setdefault("flash_attn_interface", _fa)
sys.modules.setdefault("flash_attn", _fa)

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config  # noqa: E402
from models.layers import (                          # noqa: E402
    Attention, RotaryEmbedding, rms_norm,
    CastedLinear, CastedEmbedding,
)
from models.sparse_embedding import CastedSparseEmbedding  # noqa: E402
from models.common import trunc_normal_init_         # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


# ── LIF neuron ────────────────────────────────────────────────────────────────

class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with hard reset.

    One forward call = one SNN timestep.
    Membrane potential `mem` must be threaded through calls explicitly.

    Rate coding approximation:
        Constant input u, threshold τ → fires every τ/u steps → rate = u/τ.
    """

    def __init__(self, threshold: float, beta: float = 0.9):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))
        self.register_buffer("beta",      torch.tensor(beta,      dtype=torch.float32))

    def forward(
        self, current: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mem   = self.beta * mem + current.float()
        spike = (mem >= self.threshold).to(current.dtype)
        mem   = mem * (1.0 - spike.float())   # hard reset where fired
        return spike, mem

    def init_mem(self, shape, device, dtype) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=torch.float32)


# ── SNN SwiGLU (silu gate → LIF spike) ───────────────────────────────────────

class SNN_SwiGLU(nn.Module):
    """
    SwiGLU MLP where silu(gate) is replaced by a LIF neuron.

    ANN:  out = down_proj(silu(gate) * up)
    SNN:  out = down_proj(spike      * up)   spike ∈ {0, 1} per element

    The `up` projection remains float; the spike gates it.
    Negative silu values (rare, small) are clipped to 0 as the LIF only
    integrates positive current.
    """

    def __init__(self, hidden_size: int, expansion: float, threshold: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self._inter       = inter
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)
        self.lif          = LIF(threshold=threshold)

    def forward(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate, up   = self.gate_up_proj(x).chunk(2, dim=-1)
        current    = F.silu(gate).clamp(min=0.0)   # positive silu as LIF input
        spike, mem = self.lif(current, mem)
        return self.down_proj(spike * up), mem

    def init_mem(self, shape_prefix: tuple, device, dtype) -> torch.Tensor:
        return self.lif.init_mem((*shape_prefix, self._inter), device, dtype)


# ── SNN transformer block ─────────────────────────────────────────────────────

class SNN_Block(nn.Module):
    """
    Transformer block: float multi-head attention + LIF-gated SNN MLP.

    Residual connections and RMS-norm are unchanged from the ANN.
    """

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, threshold: float):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp      = SNN_SwiGLU(config.hidden_size, config.expansion, threshold)
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin,
        hidden_states: torch.Tensor,
        mem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        mlp_out, mem = self.mlp(hidden_states, mem)
        hidden_states = rms_norm(
            hidden_states + mlp_out, variance_epsilon=self.norm_eps
        )
        return hidden_states, mem


# ── SNN reasoning module ──────────────────────────────────────────────────────

@dataclass
class ModuleMems:
    """LIF membrane potentials for every block in a reasoning module."""
    mems: List[torch.Tensor]


class SNN_ReasoningModule(nn.Module):
    """Stack of SNN_Blocks; carries LIF membrane state between timesteps."""

    def __init__(self, blocks: List[SNN_Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        mems: ModuleMems,
        cos_sin,
    ) -> Tuple[torch.Tensor, ModuleMems]:
        hidden_states = hidden_states + input_injection
        new_mems: List[torch.Tensor] = []
        for block, mem in zip(self.blocks, mems.mems):
            hidden_states, new_mem = block(cos_sin, hidden_states, mem)
            new_mems.append(new_mem)
        return hidden_states, ModuleMems(new_mems)

    def init_mems(self, shape_prefix: tuple, device, dtype) -> ModuleMems:
        return ModuleMems([
            block.mlp.init_mem(shape_prefix, device, dtype)
            for block in self.blocks
        ])


# ── Main SNN model ────────────────────────────────────────────────────────────

class HRM_SNN_Inner(nn.Module):
    """
    SNN-converted version of HierarchicalReasoningModel_ACTV1_Inner.

    Weights are loaded from the pretrained ANN checkpoint via load_checkpoint.py.
    LIF thresholds are calibrated per-layer from profiling/activation_stats.json.
    """

    T_L: int = 8    # L-module SNN timesteps (L stabilises first)
    T_H: int = 16   # H-module SNN timesteps (more steps for slower H dynamics)

    def __init__(
        self,
        config: HierarchicalReasoningModel_ACTV1Config,
        thresholds_H: List[float],
        thresholds_L: List[float],
    ):
        super().__init__()
        self.config       = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        embed_scale    = config.hidden_size ** 0.5
        embed_init_std = 1.0 / embed_scale
        self.embed_scale = embed_scale

        # I/O layers — kept as float, not spiked
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head  = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                batch_size=config.batch_size, init_std=0,
                cast_to=self.forward_dtype,
            )

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len + self.puzzle_emb_len,
            base=config.rope_theta,
        )

        # Reasoning modules with LIF gates
        self.H_level = SNN_ReasoningModule([
            SNN_Block(config, thresholds_H[i]) for i in range(config.H_layers)
        ])
        self.L_level = SNN_ReasoningModule([
            SNN_Block(config, thresholds_L[i]) for i in range(config.L_layers)
        ])

        # Learned initial hidden states (loaded from ANN checkpoint)
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _input_embeddings(
        self, inputs: torch.Tensor, puzzle_ids: torch.Tensor
    ) -> torch.Tensor:
        emb = self.embed_tokens(inputs.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            p_emb = self.puzzle_emb(puzzle_ids)
            pad = self.puzzle_emb_len * self.config.hidden_size - p_emb.shape[-1]
            if pad > 0:
                p_emb = F.pad(p_emb, (0, pad))
            emb = torch.cat(
                (p_emb.view(-1, self.puzzle_emb_len, self.config.hidden_size), emb),
                dim=-2,
            )
        return self.embed_scale * emb

    # ── Forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, batch: dict) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch: dict with keys 'inputs' [B, seq_len] and
                   'puzzle_identifiers' [B].

        Returns:
            logits [B, seq_len, vocab_size],
            (q_halt_logits [B], q_continue_logits [B])
        """
        cos_sin = self.rotary_emb()
        B       = batch["inputs"].shape[0]
        device  = batch["inputs"].device
        dtype   = self.forward_dtype

        input_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        S = input_emb.shape[1]            # seq_len + puzzle_emb_len
        shape_prefix = (B, S)

        # Broadcast learned init vectors over batch and sequence dimensions
        z_H = self.H_init.expand(B, S, self.config.hidden_size).clone().to(device)
        z_L = self.L_init.expand(B, S, self.config.hidden_size).clone().to(device)

        # ── L module: T_L timesteps (L stabilises on the input) ──────────────
        L_mems = self.L_level.init_mems(shape_prefix, device, dtype)
        for _ in range(self.T_L):
            z_L, L_mems = self.L_level(z_L, z_H + input_emb, L_mems, cos_sin)

        # ── H module: T_H timesteps (H reasons over stable L state) ──────────
        H_mems = self.H_level.init_mems(shape_prefix, device, dtype)
        for _ in range(self.T_H):
            z_H, H_mems = self.H_level(z_H, z_L, H_mems, cos_sin)

        # Float output heads — not spiked
        logits   = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return logits, (q_logits[..., 0], q_logits[..., 1])


# ── Factory ───────────────────────────────────────────────────────────────────

STATS_PATH = Path(__file__).resolve().parent.parent / "profiling" / "activation_stats.json"

VOCAB_SIZE    = 12
SEQ_LEN       = 900
NUM_PUZZLE_IDS = 1_045_829


def _thresholds_from_stats(stats: dict, module: str, n_layers: int) -> List[float]:
    thresholds = []
    for i in range(n_layers):
        key = f"{module}.layer{i}.mlp.silu_gate"
        if key in stats:
            thresholds.append(float(stats[key]["p99"]))
        else:
            thresholds.append(1.0)
            print(f"[WARN] no stats for {key}, falling back to threshold=1.0")
    return thresholds


def build_hrm_snn(
    batch_size: int = 1,
    stats_path: Path = STATS_PATH,
    threshold_overrides: dict | None = None,
) -> HRM_SNN_Inner:
    """
    Construct HRM_SNN_Inner with thresholds calibrated from activation stats.

    Args:
        threshold_overrides: optional dict of {layer_name: threshold} to
            override individual layers, e.g. {"H_level.layer3": 1.7}.

    Call load_ann_weights() afterwards to transfer pretrained ANN weights.
    """
    with open(stats_path) as f:
        stats = json.load(f)

    overrides = threshold_overrides or {}

    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_puzzle_identifiers=NUM_PUZZLE_IDS,
        puzzle_emb_ndim=512,
        H_cycles=2,
        L_cycles=2,
        H_layers=4,
        L_layers=4,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
    )

    thresholds_H = _thresholds_from_stats(stats, "H_level", config.H_layers)
    thresholds_L = _thresholds_from_stats(stats, "L_level", config.L_layers)

    for i in range(config.H_layers):
        key = f"H_level.layer{i}"
        if key in overrides:
            thresholds_H[i] = overrides[key]
    for i in range(config.L_layers):
        key = f"L_level.layer{i}"
        if key in overrides:
            thresholds_L[i] = overrides[key]

    print("LIF thresholds (p99 of silu_gate, overrides applied):")
    for i, t in enumerate(thresholds_H):
        tag = "  *overridden*" if f"H_level.layer{i}" in overrides else ""
        print(f"  H_level.layer{i}: {t:.4f}{tag}")
    for i, t in enumerate(thresholds_L):
        tag = "  *overridden*" if f"L_level.layer{i}" in overrides else ""
        print(f"  L_level.layer{i}: {t:.4f}{tag}")

    return HRM_SNN_Inner(config, thresholds_H, thresholds_L)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building SNN model...")
    model = build_hrm_snn(batch_size=2)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    batch = {
        "inputs":             torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32, device=device),
    }

    print(f"Running forward pass on {device}...")
    logits, (q_halt, q_cont) = model(batch)
    print(f"logits:   {logits.shape}  (expected [2, {SEQ_LEN}, {VOCAB_SIZE}])")
    print(f"q_halt:   {q_halt.shape}  q_cont: {q_cont.shape}")
    print("Smoke test passed.")
