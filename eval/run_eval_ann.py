"""
ARC-2 evaluation for the original pretrained ANN HRM model (baseline).

Identical setup to run_eval.py: raw ARC JSON → encode → predict → compare.
puzzle_identifiers = 0 (blank) for all puzzles, same as the SNN eval.

Usage
-----
conda run -n nn python eval/run_eval_ann.py [--eval-dir PATH] [--batch-size N]
"""

import sys
import json
import argparse
import time
import types
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
HRM_SRC = ROOT / "HRM"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HRM_SRC))

# Stub flash_attn for MPS
def _sdpa(q, k, v, causal=False, **_):
    orig = q.dtype
    q, k, v = (x.transpose(1, 2).to(torch.float32) for x in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).contiguous().to(orig)

_fa = types.ModuleType("flash_attn")
_fa.flash_attn_func = _sdpa                          # type: ignore[attr-defined]
sys.modules.setdefault("flash_attn_interface", _fa)
sys.modules.setdefault("flash_attn", _fa)

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1_Inner,
)
from eval.run_eval import grid_to_tokens, tokens_to_grid, _output_shape

CKPT_BLOB = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sapientinc--HRM-checkpoint-ARC-2"
    / "blobs"
    / "58719e51da6cd7712eb4197f908fbfdc88403ee48c42a3a92bab9f9c968df64d"
)
VOCAB_SIZE    = 12
SEQ_LEN       = 900
NUM_PUZZLE_IDS = 1_045_829


def build_ann(batch_size: int, device: torch.device) -> HierarchicalReasoningModel_ACTV1_Inner:
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_puzzle_identifiers=NUM_PUZZLE_IDS,
        puzzle_emb_ndim=512,
        H_cycles=2, L_cycles=2,
        H_layers=4, L_layers=4,
        hidden_size=512, expansion=4, num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16, halt_exploration_prob=0.1,
    )
    with torch.device(device):
        model = HierarchicalReasoningModel_ACTV1_Inner(config)
    return model


def load_ann(model: HierarchicalReasoningModel_ACTV1_Inner) -> None:
    state = torch.load(CKPT_BLOB, map_location="cpu", weights_only=False)
    prefix = "_orig_mod.model.inner."
    trimmed = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(trimmed, strict=True)
    print(f"Loaded {len(trimmed)} ANN tensors.")


@torch.no_grad()
def ann_forward(model, batch: dict, device: torch.device):
    carry = model.empty_carry(batch["inputs"].shape[0])
    carry = HierarchicalReasoningModel_ACTV1InnerCarry(
        z_H=carry.z_H.to(device),
        z_L=carry.z_L.to(device),
    )
    reset_all = torch.ones(batch["inputs"].shape[0], dtype=torch.bool, device=device)
    carry = model.reset_carry(reset_all, carry)
    _carry, logits, _ = model(carry, batch)
    return logits   # [B, seq_len + puzzle_emb_len, vocab_size]


def evaluate_puzzle(model, puzzle: dict, device: torch.device, batch_size: int) -> dict:
    test_cases = puzzle.get("test", [])
    all_inputs  = [grid_to_tokens(tc["input"])  for tc in test_cases]
    all_targets = [grid_to_tokens(tc["output"]) for tc in test_cases]
    results = []

    for start in range(0, len(all_inputs), batch_size):
        inp_chunk = all_inputs [start: start + batch_size]
        tgt_chunk = all_targets[start: start + batch_size]
        batch = {
            "inputs": torch.tensor(np.stack(inp_chunk), dtype=torch.int32, device=device),
            "puzzle_identifiers": torch.zeros(len(inp_chunk), dtype=torch.int32, device=device),
        }
        logits = ann_forward(model, batch, device)    # [B, seq_len, 12] — puzzle_emb already stripped
        preds  = logits.argmax(-1).cpu().numpy()       # [B, 900]

        for pred, tgt, tc in zip(preds, tgt_chunk, test_cases[start:]):
            nrow, ncol  = _output_shape(tc["output"])
            pred_grid   = tokens_to_grid(pred, nrow, ncol)
            true_grid   = np.array(tc["output"], dtype=np.int32)
            cell_match  = (pred_grid == true_grid)
            results.append({
                "cell_acc":    float(cell_match.mean()),
                "grid_solved": bool(cell_match.all()),
            })

    return {
        "examples":      results,
        "puzzle_solved": all(r["grid_solved"] for r in results),
        "mean_cell_acc": float(np.mean([r["cell_acc"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        default=str(ROOT / "HRM/dataset/raw-data/ARC-AGI-2/data/evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--out",        type=str, default=None,
                        help="Output JSON path (default: eval/results_ann_baseline.json).")
    args = parser.parse_args()

    eval_dir     = Path(args.eval_dir)
    puzzle_files = sorted(glob(str(eval_dir / "*.json")))
    if args.limit:
        puzzle_files = puzzle_files[: args.limit]
    print(f"Evaluating {len(puzzle_files)} puzzles (ANN baseline)  from {eval_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_ann(args.batch_size, device)
    model.eval()
    load_ann(model)

    all_results = {}
    n_puzzles = n_solved = n_examples = n_grids_exact = 0
    cell_accs = []
    t0 = time.time()

    for pf in puzzle_files:
        name = Path(pf).stem
        with open(pf) as f:
            puzzle = json.load(f)

        res = evaluate_puzzle(model, puzzle, device, args.batch_size)
        all_results[name] = res

        n_puzzles   += 1
        n_solved    += int(res["puzzle_solved"])
        n_examples  += len(res["examples"])
        n_grids_exact += sum(e["grid_solved"] for e in res["examples"])
        cell_accs.extend(e["cell_acc"] for e in res["examples"])

        puzzle_acc = 100.0 * n_solved / n_puzzles
        cell_acc   = 100.0 * float(np.mean(cell_accs))
        elapsed    = time.time() - t0
        marker     = "✓" if res["puzzle_solved"] else "✗"
        print(
            f"  [{n_puzzles:3d}/{len(puzzle_files)}] {marker} {name}"
            f"  cell={res['mean_cell_acc']:.1%}"
            f"  | running puzzle_acc={puzzle_acc:.1f}%  cell_acc={cell_acc:.1f}%"
            f"  ({elapsed:.0f}s)"
        )

    elapsed    = time.time() - t0
    puzzle_acc = 100.0 * n_solved    / n_puzzles
    grid_acc   = 100.0 * n_grids_exact / n_examples
    cell_acc   = 100.0 * float(np.mean(cell_accs))

    print()
    print("═" * 60)
    print(f"[ANN BASELINE]")
    print(f"Puzzles evaluated : {n_puzzles}")
    print(f"Puzzle-level acc  : {puzzle_acc:.2f}%  ({n_solved}/{n_puzzles})")
    print(f"Grid-level acc    : {grid_acc:.2f}%  ({n_grids_exact}/{n_examples})")
    print(f"Cell-level acc    : {cell_acc:.2f}%")
    print(f"Time              : {elapsed:.1f}s  ({elapsed/n_puzzles:.1f}s/puzzle)")
    print("═" * 60)

    out = {
        "model": "ANN_baseline",
        "eval_dir": str(eval_dir),
        "n_puzzles": n_puzzles,
        "puzzle_acc": round(puzzle_acc, 4),
        "grid_acc":   round(grid_acc,   4),
        "cell_acc":   round(cell_acc,   4),
        "n_solved":   n_solved,
        "per_puzzle": all_results,
    }
    out_path = Path(args.out) if args.out else ROOT / "eval" / "results_ann_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
