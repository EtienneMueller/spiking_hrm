"""
Generate Figure 1 for the Bernstein abstract submission.

Two panels:
  A  p99/mean activation ratio per LIF layer — diagnostic for threshold outliers
  B  Cell-level accuracy across the four model conditions

Output: figures/figure1.pdf  (and figure1.png for quick preview)

Run from the repo root:
    conda run -n nn python figures/make_figure.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Load data ─────────────────────────────────────────────────────────────────

with open(ROOT / "profiling" / "activation_stats.json") as f:
    stats = json.load(f)

with open(ROOT / "eval" / "results_ann_baseline.json") as f:
    ann = json.load(f)
with open(ROOT / "eval" / "results.json") as f:
    snn = json.load(f)
with open(ROOT / "eval" / "results_h3_1.7.json") as f:
    snn_c = json.load(f)

# ── Panel A data ──────────────────────────────────────────────────────────────

layer_order = [
    ("H", 0), ("H", 1), ("H", 2), ("H", 3),
    ("L", 0), ("L", 1), ("L", 2), ("L", 3),
]
labels = [f"H{i}" for _, i in layer_order[:4]] + [f"L{i}" for _, i in layer_order[4:]]
ratios = []
for mod, i in layer_order:
    key = f"{mod}_level.layer{i}.mlp.silu_gate"
    s = stats[key]
    ratios.append(s["p99"] / s["mean"])

# ── Panel B data ──────────────────────────────────────────────────────────────

random_acc = 8.33
acc_values = [
    random_acc,
    ann["cell_acc"],
    snn["cell_acc"],
    snn_c["cell_acc"],
]
acc_labels = [
    "Random\n(chance)",
    "ANN\nbaseline",
    "SNN\np99 thresh.",
    "SNN\ncorrected",
]

# ── Style ─────────────────────────────────────────────────────────────────────

BLUE   = "#3a78b5"
RED    = "#c0392b"
GRAY   = "#aaaaaa"
GREEN  = "#2ecc71"
LIGHT  = "#d6e8f7"

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        8,
    "axes.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width":0.7,
    "ytick.major.width":0.7,
    "pdf.fonttype":     42,   # embed fonts
    "ps.fonttype":      42,
})

fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(6.5, 2.6),
    gridspec_kw={"wspace": 0.38},
)

# ── Panel A: p99/mean ratio ───────────────────────────────────────────────────

n = len(labels)
x = np.arange(n)
bar_colors = [RED if (mod == "H" and i == 3) else BLUE if mod == "H" else LIGHT
              for mod, i in layer_order]
# lighter blue for L bars
bar_colors = []
for mod, i in layer_order:
    if mod == "H" and i == 3:
        bar_colors.append(RED)
    elif mod == "H":
        bar_colors.append(BLUE)
    else:
        bar_colors.append("#8ab4d8")

bars_a = ax_a.bar(x, ratios, color=bar_colors, width=0.6, zorder=2, linewidth=0)

# Dashed reference line at group mean (excluding outlier)
normal_ratios = [r for r, (mod, i) in zip(ratios, layer_order) if not (mod == "H" and i == 3)]
mean_normal = np.mean(normal_ratios)
ax_a.axhline(mean_normal, color=GRAY, linewidth=0.9, linestyle="--", zorder=1,
             label=f"Mean (others) = {mean_normal:.1f}×")

# Annotate the outlier bar
outlier_idx = 3   # H3
ax_a.text(
    outlier_idx, ratios[outlier_idx] + 0.12,
    f"{ratios[outlier_idx]:.1f}×",
    ha="center", va="bottom", fontsize=7, color=RED, fontweight="bold",
)

ax_a.set_xticks(x)
ax_a.set_xticklabels(labels, fontsize=7.5)
ax_a.set_ylabel("p99 / mean activation ratio", fontsize=8)
ax_a.set_ylim(0, max(ratios) * 1.22)
ax_a.yaxis.set_major_locator(plt.MultipleLocator(2))
ax_a.set_axisbelow(True)
ax_a.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)

# Divider between H and L groups
ax_a.axvline(3.5, color="#cccccc", linewidth=0.8, linestyle=":")

# Group labels
ax_a.text(1.5,  ax_a.get_ylim()[1] * 0.97, "H module", ha="center",
          fontsize=7, color="#555555", style="italic")
ax_a.text(5.5,  ax_a.get_ylim()[1] * 0.97, "L module", ha="center",
          fontsize=7, color="#555555", style="italic")

ax_a.legend(fontsize=6.5, frameon=False, loc="lower right")
ax_a.set_title("A  LIF threshold diagnostic", loc="left", fontsize=8.5, fontweight="bold", pad=5)

# ── Panel B: accuracy comparison ─────────────────────────────────────────────

bar_colors_b = [GRAY, BLUE, "#e8a838", GREEN]
x_b = np.arange(len(acc_labels))
bars_b = ax_b.bar(x_b, acc_values, color=bar_colors_b, width=0.55, zorder=2, linewidth=0)

# Value labels on top of bars
for bar, val in zip(bars_b, acc_values):
    ax_b.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.8,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=7,
        color="#333333",
    )

# Annotations: arrow from SNN-p99 to SNN-corrected
x_p99  = x_b[2]
x_corr = x_b[3]
y_p99  = acc_values[2]
y_corr = acc_values[3]
# Right-side bracket: vertical span from SNN-p99 to SNN-corrected
bx = x_corr + 0.38   # x position of bracket
ax_b.plot([bx, bx], [y_p99, y_corr], color=GREEN, lw=1.2)
ax_b.plot([bx - 0.06, bx], [y_p99,  y_p99],  color=GREEN, lw=1.2)
ax_b.plot([bx - 0.06, bx], [y_corr, y_corr], color=GREEN, lw=1.2)
ax_b.text(bx + 0.07, (y_p99 + y_corr) / 2, "+5.95 pp",
          ha="left", va="center", fontsize=6.5, color=GREEN, fontweight="bold")

# ANN baseline dashed line
ax_b.axhline(acc_values[1], color=BLUE, linewidth=0.8, linestyle="--",
             alpha=0.5, zorder=1)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(acc_labels, fontsize=7.5)
ax_b.set_ylabel("Cell-level accuracy (%)", fontsize=8)
ax_b.set_ylim(0, 90)
ax_b.yaxis.set_major_locator(plt.MultipleLocator(20))
ax_b.set_axisbelow(True)
ax_b.yaxis.grid(True, linewidth=0.4, color="#e0e0e0", zorder=0)

# 56.1% annotation well above the corrected bar
ax_b.text(
    x_b[3], acc_values[3] + 12,
    "56.1% of ANN",
    ha="center", va="bottom", fontsize=6.5, color="#555555",
)

ax_b.set_title("B  Cell-level accuracy", loc="left", fontsize=8.5, fontweight="bold", pad=5)

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = Path(__file__).resolve().parent
fig.savefig(out_dir / "figure1.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out_dir / "figure1.png", bbox_inches="tight", dpi=200)
print(f"Saved → {out_dir}/figure1.pdf")
print(f"Saved → {out_dir}/figure1.png")
