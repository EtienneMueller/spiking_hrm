"""
Transfer pretrained ANN weights from the HRM checkpoint into HRM_SNN_Inner.

Mapping:
  _orig_mod.model.inner.<name>  →  <name>  in HRM_SNN_Inner

Weights that transfer directly (no change):
  embed_tokens, lm_head, q_head, puzzle_emb
  H_level.layers.{i}.self_attn.*  →  H_level.blocks.{i}.self_attn.*
  L_level.layers.{i}.self_attn.*  →  L_level.blocks.{i}.self_attn.*
  H_level.layers.{i}.mlp.{gate_up_proj,down_proj}.*  →  H_level.blocks.{i}.mlp.*
  L_level.layers.{i}.mlp.{gate_up_proj,down_proj}.*  →  L_level.blocks.{i}.mlp.*
  H_init, L_init

LIF neuron parameters (threshold, beta) are not in the ANN checkpoint; they
are initialised from activation stats in build_hrm_snn() and stay fixed.
"""

from pathlib import Path
from typing import Dict, Optional
import torch

from conversion.hrm_snn import HRM_SNN_Inner


CKPT_BLOB = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sapientinc--HRM-checkpoint-ARC-2"
    / "blobs"
    / "58719e51da6cd7712eb4197f908fbfdc88403ee48c42a3a92bab9f9c968df64d"
)


def _remap_keys(state: dict) -> Dict[str, torch.Tensor]:
    """
    Strip the torch.compile / wrapper prefixes and map ANN key names to SNN
    key names (layers → blocks).
    """
    prefix = "_orig_mod.model.inner."
    remapped = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        k = k.removeprefix(prefix)
        # layers.{i} → blocks.{i}
        k = k.replace(".layers.", ".blocks.")
        remapped[k] = v
    return remapped


def load_ann_weights(
    model: HRM_SNN_Inner,
    ckpt_path: Optional[Path] = None,
    strict: bool = True,
) -> None:
    """
    Load the pretrained ANN checkpoint into `model` in-place.

    Args:
        model:     HRM_SNN_Inner instance (from build_hrm_snn()).
        ckpt_path: path to the checkpoint blob; defaults to the cached HF blob.
        strict:    if True, raise on unexpected or missing keys (excluding
                   LIF buffers which are not in the ANN checkpoint).
    """
    path = ckpt_path or CKPT_BLOB
    print(f"Loading checkpoint: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    snn_state = _remap_keys(raw)

    # LIF buffers (threshold, beta) are not in the ANN — exclude from strict check
    snn_keys = set(model.state_dict().keys())
    lif_keys = {k for k in snn_keys if ".lif.threshold" in k or ".lif.beta" in k}

    missing, unexpected = model.load_state_dict(snn_state, strict=False)

    # Anything missing that is NOT a LIF buffer is a real problem
    real_missing = [k for k in missing if k not in lif_keys]
    if real_missing and strict:
        raise RuntimeError(f"Missing non-LIF keys: {real_missing}")
    if unexpected and strict:
        raise RuntimeError(f"Unexpected keys: {unexpected}")

    if real_missing:
        print(f"[WARN] missing non-LIF keys: {real_missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")

    n_transferred = len(snn_state) - len(unexpected)
    n_lif         = len(lif_keys)
    print(f"Transferred {n_transferred} tensors, kept {n_lif} LIF buffers as-is.")


# ── Standalone verification ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from conversion.hrm_snn import build_hrm_snn, VOCAB_SIZE, SEQ_LEN

    print("Building SNN model...")
    model = build_hrm_snn(batch_size=2)
    model.eval()

    load_ann_weights(model)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Quick forward pass to confirm weights are live
    batch = {
        "inputs":             torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32, device=device),
    }
    logits, _ = model(batch)
    print(f"logits shape: {logits.shape}  dtype: {logits.dtype}")
    print(f"logits sample (token 0): {logits[0, 0].softmax(0).tolist()}")
    print("Weight transfer verified.")
