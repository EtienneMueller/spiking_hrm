# spiking_hrm

Tbd

## Init

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
