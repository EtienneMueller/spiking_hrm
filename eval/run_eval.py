"""
ARC-2 evaluation for the HRM SNN model.

Reads raw ARC JSON files, encodes grids to token sequences (same tokenisation
as the HRM training pipeline), runs the SNN model, and reports accuracy.

Metrics
-------
cell_acc:   fraction of individual output-grid cells predicted correctly.
grid_acc:   fraction of examples where ALL output cells match exactly.
puzzle_acc: fraction of puzzles where every test example is solved.

Usage
-----
conda run -n nn python eval/run_eval.py [--eval-dir PATH] [--batch-size N]
"""

import sys
import json
import argparse
import time
from pathlib import Path
from glob import glob
from typing import List, Tuple

import numpy as np
import torch

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conversion.hrm_snn import build_hrm_snn, VOCAB_SIZE, SEQ_LEN
from conversion.load_checkpoint import load_ann_weights


# ── ARC grid ↔ token sequence ─────────────────────────────────────────────────

ARC_MAX = 30   # maximum grid dimension

# Token map: PAD=0, EOS=1, color_k = k+2  (k in 0..9)


def grid_to_tokens(grid: List[List[int]]) -> np.ndarray:
    """
    Encode an ARC grid (list-of-lists, max 30×30) to a flat 900-token array.
    Matches the tokenisation in dataset/build_arc_dataset.py (no translation).
    """
    arr = np.array(grid, dtype=np.uint8)
    nrow, ncol = arr.shape

    # Pad to 30×30; color tokens = color + 2
    out = np.zeros((ARC_MAX, ARC_MAX), dtype=np.int32)
    out[:nrow, :ncol] = arr + 2

    # EOS markers: row below last row, column right of last column
    if nrow < ARC_MAX:
        out[nrow, :ncol] = 1       # bottom EOS row
    if ncol < ARC_MAX:
        out[:nrow, ncol] = 1       # right EOS column

    return out.flatten()           # shape (900,)


def tokens_to_grid(tokens: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    """
    Decode a 900-token array back to an (nrow, ncol) integer grid.
    Token = color + 2; PAD/EOS → 0.
    """
    mat = tokens.reshape(ARC_MAX, ARC_MAX)
    cell = mat[:nrow, :ncol]
    # Colors: token ≥ 2 → color = token - 2; EOS/PAD → 0
    return np.where(cell >= 2, cell - 2, 0).astype(np.int32)


# ── Per-puzzle evaluation ─────────────────────────────────────────────────────

def _output_shape(grid: List[List[int]]) -> Tuple[int, int]:
    return len(grid), len(grid[0])


def evaluate_puzzle(
    model,
    puzzle: dict,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """
    Run the SNN model on every test example of one ARC puzzle.

    Returns a dict with per-example results and summary metrics.
    puzzle_id = 0 (blank) — the model uses no puzzle-specific embedding.
    """
    test_cases = puzzle.get("test", [])
    results = []

    # Batch all test cases together (usually just 1)
    all_inputs  = []
    all_targets = []

    for tc in test_cases:
        inp_tokens  = grid_to_tokens(tc["input"])
        out_tokens  = grid_to_tokens(tc["output"])
        all_inputs.append(inp_tokens)
        all_targets.append(out_tokens)

    # Process in mini-batches
    for start in range(0, len(all_inputs), batch_size):
        inp_chunk = all_inputs [start: start + batch_size]
        tgt_chunk = all_targets[start: start + batch_size]

        batch = {
            "inputs": torch.tensor(
                np.stack(inp_chunk), dtype=torch.int32, device=device
            ),
            "puzzle_identifiers": torch.zeros(
                len(inp_chunk), dtype=torch.int32, device=device
            ),
        }

        logits, _ = model(batch)                          # [B, 900, 12]
        preds = logits.argmax(dim=-1).cpu().numpy()       # [B, 900]

        for pred, tgt, tc in zip(preds, tgt_chunk, test_cases[start:]):
            nrow, ncol  = _output_shape(tc["output"])
            pred_grid   = tokens_to_grid(pred, nrow, ncol)
            true_grid   = np.array(tc["output"], dtype=np.int32)

            cell_match  = (pred_grid == true_grid)
            cell_acc    = float(cell_match.mean())
            grid_solved = bool(cell_match.all())

            results.append({
                "cell_acc":    cell_acc,
                "grid_solved": grid_solved,
                "pred_grid":   pred_grid.tolist(),
                "true_grid":   true_grid.tolist(),
            })

    puzzle_solved = all(r["grid_solved"] for r in results)
    return {
        "examples":     results,
        "puzzle_solved": puzzle_solved,
        "mean_cell_acc": float(np.mean([r["cell_acc"] for r in results])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        default=str(ROOT / "HRM/dataset/raw-data/ARC-AGI-2/data/evaluation"),
        help="Directory of ARC JSON puzzle files.",
    )
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--limit",        type=int,   default=None,
                        help="Evaluate only the first N puzzles.")
    parser.add_argument("--h3-threshold", type=float, default=None,
                        help="Override LIF threshold for H_level.layer3.")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Output JSON path (default: eval/results.json).")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    puzzle_files = sorted(glob(str(eval_dir / "*.json")))
    if not puzzle_files:
        print(f"No JSON files found in {eval_dir}")
        sys.exit(1)
    if args.limit:
        puzzle_files = puzzle_files[: args.limit]

    print(f"Evaluating {len(puzzle_files)} puzzles from {eval_dir}")

    # ── Build + load model ────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    overrides = {}
    if args.h3_threshold is not None:
        overrides["H_level.layer3"] = args.h3_threshold

    model = build_hrm_snn(batch_size=args.batch_size, threshold_overrides=overrides or None)
    model.eval()
    load_ann_weights(model)
    model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results   = {}
    n_puzzles     = 0
    n_solved      = 0
    n_examples    = 0
    n_grids_exact = 0
    cell_accs     = []

    t0 = time.time()

    for pf in puzzle_files:
        name = Path(pf).stem
        with open(pf) as f:
            puzzle = json.load(f)

        res = evaluate_puzzle(model, puzzle, device, args.batch_size)
        all_results[name] = res

        n_puzzles  += 1
        n_solved   += int(res["puzzle_solved"])
        n_examples += len(res["examples"])
        n_grids_exact += sum(e["grid_solved"] for e in res["examples"])
        cell_accs.extend(e["cell_acc"] for e in res["examples"])

        # Live progress
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

    elapsed = time.time() - t0
    puzzle_acc = 100.0 * n_solved    / n_puzzles
    grid_acc   = 100.0 * n_grids_exact / n_examples
    cell_acc   = 100.0 * float(np.mean(cell_accs))

    print()
    print("═" * 60)
    print(f"Puzzles evaluated : {n_puzzles}")
    print(f"Puzzle-level acc  : {puzzle_acc:.2f}%  ({n_solved}/{n_puzzles} solved)")
    print(f"Grid-level acc    : {grid_acc:.2f}%  ({n_grids_exact}/{n_examples} exact)")
    print(f"Cell-level acc    : {cell_acc:.2f}%")
    print(f"Time              : {elapsed:.1f}s  ({elapsed/n_puzzles:.1f}s/puzzle)")
    print("═" * 60)

    summary = {
        "eval_dir":       str(eval_dir),
        "n_puzzles":      n_puzzles,
        "n_examples":     n_examples,
        "puzzle_acc":     round(puzzle_acc, 4),
        "grid_acc":       round(grid_acc,   4),
        "cell_acc":       round(cell_acc,   4),
        "n_solved":       n_solved,
        "n_grids_exact":  n_grids_exact,
        "elapsed_s":      round(elapsed, 1),
        "model": {
            "T_L": model.T_L,
            "T_H": model.T_H,
            "threshold_overrides": overrides,
        },
        "per_puzzle": all_results,
    }

    out_path = Path(args.out) if args.out else ROOT / "eval" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
