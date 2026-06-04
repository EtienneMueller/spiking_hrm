# Spiking Hierarchical Reasoning Model

Proof-of-concept direct ANN-to-SNN conversion of the [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) (HRM; Wang et al. 2025), a 27M-parameter recurrent transformer trained on ARC-AGI-2 grid reasoning tasks. Pretrained weights are transferred unchanged; SwiGLU gate activations are replaced by Leaky Integrate-and-Fire (LIF) neurons calibrated from activation statistics. No fine-tuning or surrogate gradients are used.

## Results


| Model                               | Cell Accuracy | Relative to ANN |
| ----------------------------------- | ------------- | --------------- |
| Random (12-class)                   | 8.33%         | —               |
| ANN baseline (blank puzzle id)      | 67.09%        | 100%            |
| SNN, p99 thresholds                 | 31.71%        | 47.3%           |
| SNN, H3 threshold corrected to 1.70 | **37.67%**    | **56.1%**       |

Evaluated on the 120-puzzle ARC-AGI-2 public evaluation set (167 test examples). The key finding: the per-layer p99/mean activation ratio is a useful diagnostic for outlier LIF thresholds — one miscalibrated layer (H_level.layer3, p99/mean = 8.2×) suppressed 5.95 percentage points of accuracy, recovered by a single threshold correction.

## Setup

```bash
git clone https://github.com/sapientinc/HRM
huggingface-cli download sapientinc/HRM-checkpoint-ARC-2
```

## Run

**Step 1 — Profile ANN activations** (generates `profiling/activation_stats.json`):
```bash
python profiling/profile_activations.py
```

**Steps 2+3 — Build SNN and transfer weights** (smoke test):
```bash
python -m conversion.load_checkpoint
```

**Steps 4+5 — Evaluate SNN on ARC-AGI-2**:
```bash
# Standard p99 thresholds
python eval/run_eval.py

# With H_level.layer3 threshold corrected to 1.70
python eval/run_eval.py --h3-threshold 1.70 --out eval/results_h3_1.7.json

# ANN baseline
python eval/run_eval_ann.py
```

## Reference

```
Wang et al. (2025). Hierarchical Reasoning Model. arXiv:2506.21734.
```

