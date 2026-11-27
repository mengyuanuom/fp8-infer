"""Render the published tables as a PNG; requires matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'data/comparison.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
with (root / 'data/bf16_matmul_e2e.csv').open(encoding='utf-8', newline='') as f:
    e2e_rows = list(csv.DictReader(f))
assert [r['seq_tokens'] for r in rows] == [r['seq_tokens'] for r in e2e_rows]
x = list(range(len(rows)))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
for ax, metric, title, ylabel in [
    (axes[0], 'bf16_matmul_e2e_pct', 'BF16 MatMul / E2E P50', 'MatMul mean / BF16 E2E P50 (%)'),
    (axes[1], 'latency_reduction_pct', 'MXFP8 end-to-end P50 latency reduction', 'Latency reduction vs. BF16 (%)'),
]:
    for model, label, color in [('pangu', 'PanGu 1B', '#2563eb'), ('qwen', 'Qwen 1.7B', '#ea580c')]:
        source = e2e_rows if metric == 'bf16_matmul_e2e_pct' else rows
        ax.plot(x, [float(r[f'{model}_{metric}']) for r in source], marker='o',
                linewidth=2.2, label=label, color=color)
    ax.set_xticks(x, [r['seq_tokens'] for r in rows], rotation=35)
    ax.set_xlabel('Sequence length (tokens; equally spaced categories)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
axes[0].set_ylim(0, 70)
axes[1].set_ylim(0, 35)
fig.suptitle('FP8 inference comparison | 2026-11-27', fontsize=15, weight='bold')
fig.supxlabel('Left: profiled MatMul mean / unprofiled BF16 API P50 (estimate, not a per-request decomposition).', fontsize=9)
(root / 'assets').mkdir(exist_ok=True)
fig.savefig(root / 'assets/fp8-comparison.png', dpi=170, facecolor='white')
plt.close(fig)
