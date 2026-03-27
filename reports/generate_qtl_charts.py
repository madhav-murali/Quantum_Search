#!/usr/bin/env python3
"""
Generate QTL analysis charts for presentation.
Produces 4 images in reports/ directory.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('reports')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
})

# ── Colors ──
COLORS = {
    'source':    '#2563EB',  # blue
    'frozen':    '#9333EA',  # purple
    'finetuned': '#16A34A',  # green
    'distilled': '#DC2626',  # red
    'scratch':   '#6B7280',  # gray
    'improved':  '#F59E0B',  # amber
}


def load_json(name):
    path = RESULTS_DIR / name
    if path.exists():
        return json.load(open(path))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Strategy Accuracy Comparison (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
def chart_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data: all experiments grouped
    groups = {
        'LeNet5\n(360K params, 30ep)': {
            'frozen':    0.5204,
            'finetuned': None,
            'distilled': 0.7784,
            'scratch':   0.6469,
        },
        'LeNet5 v2\n(360K params, 50ep)': {
            'frozen':    0.7247,
            'finetuned': 0.8136,
            'distilled': 0.7914,
            'scratch':   0.5926,
        },
        'LeNet Improved\n(2.2M params, 100ep)': {
            'frozen':    None,
            'finetuned': None,
            'distilled': 0.9185,
            'scratch':   None,
        },
    }

    strategies = ['frozen', 'finetuned', 'distilled', 'scratch']
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.arange(len(strategies)) - (len(strategies)-1)/2

    for i, strat in enumerate(strategies):
        vals = []
        for g in groups.values():
            vals.append(g.get(strat))
        positions = x + offsets[i] * width
        bars = []
        for j, v in enumerate(vals):
            if v is not None:
                bar = ax.bar(positions[j], v * 100, width * 0.9,
                             color=COLORS[strat], alpha=0.85, edgecolor='white',
                             linewidth=0.5)
                ax.text(positions[j], v * 100 + 0.8, f'{v*100:.1f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Source model reference line
    ax.axhline(y=94.14, color=COLORS['source'], linestyle='--', linewidth=2,
               alpha=0.7, label='Source (ResNet50): 94.1%')
    ax.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(groups.keys(), fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('QTL Strategy Comparison Across Architectures', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.2)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS[s], label=s.capitalize())
        for s in strategies
    ]
    legend_elements.append(plt.Line2D([0], [0], color=COLORS['source'],
                           linestyle='--', linewidth=2, label='Source (ResNet50)'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9)

    plt.tight_layout()
    out = OUTPUT_DIR / 'qtl_accuracy_comparison.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Training Convergence Curves
# ══════════════════════════════════════════════════════════════════════════════
def chart_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Accuracy curves
    ax = axes[0]
    curves = {
        'LeNet5 finetuned (100ep)': ('qtl_lenet5_amplitude_finetuned_results.json', COLORS['finetuned'], '--'),
        'LeNet5 v2 finetuned (50ep)': ('qtl_v2_finetuned_results.json', COLORS['finetuned'], ':'),
        'LeNet5 v2 distilled (50ep)': ('qtl_v2_distilled_results.json', COLORS['distilled'], ':'),
        'Improved distilled (100ep)': ('qtl_improved_distilled_results.json', COLORS['improved'], '-'),
    }

    max_epoch = 0
    for label, (fname, color, ls) in curves.items():
        data = load_json(fname)
        if data and 'val_acc' in data:
            accs = [a * 100 for a in data['val_acc']]
            epochs = range(1, len(accs) + 1)
            max_epoch = max(max_epoch, len(accs))
            ax.plot(epochs, accs, color=color, linestyle=ls, linewidth=2,
                    label=label, alpha=0.85)

    ax.axhline(y=94.14, color=COLORS['source'], linestyle='--', linewidth=1.5,
               alpha=0.5, label='Source (ResNet50)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Training Convergence')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_ylim(30, 100)

    # Right: Loss curves
    ax = axes[1]
    for label, (fname, color, ls) in curves.items():
        data = load_json(fname)
        if data and 'loss' in data:
            losses = data['loss']
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, color=color, linestyle=ls, linewidth=2,
                    label=label, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss Convergence')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = OUTPUT_DIR / 'qtl_convergence_curves.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: Parameter Efficiency Scatter
# ══════════════════════════════════════════════════════════════════════════════
def chart_parameter_efficiency():
    fig, ax = plt.subplots(figsize=(10, 6))

    points = [
        ('Source\n(ResNet50)', 24_032_714, 94.14, COLORS['source'], 200),
        ('LeNet5 frozen\n(30ep)', 359_578, 52.04, COLORS['frozen'], 100),
        ('LeNet5 distilled\n(30ep)', 359_578, 77.84, COLORS['distilled'], 100),
        ('LeNet5 v2 finetuned\n(50ep)', 359_578, 81.36, COLORS['finetuned'], 100),
        ('LeNet5 v2 distilled\n(50ep)', 359_578, 79.14, COLORS['distilled'], 100),
        ('Improved distilled\n(100ep)', 2_225_546, 91.85, COLORS['improved'], 150),
    ]

    for label, params, acc, color, size in points:
        ax.scatter(params / 1e6, acc, s=size, c=color, edgecolors='white',
                   linewidth=1.5, zorder=3, alpha=0.9)
        offset_y = 2.0 if 'Source' not in label else -3.0
        offset_x = 0
        ax.annotate(label, (params / 1e6, acc),
                    textcoords='offset points', xytext=(offset_x, offset_y + 8),
                    fontsize=8, ha='center', va='bottom',
                    fontweight='bold')

    # Reference lines
    ax.axhline(y=94.14, color=COLORS['source'], linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=90, color='gray', linestyle=':', linewidth=1, alpha=0.2)

    # Add arrow showing the reduction
    ax.annotate('', xy=(2.2, 91.85), xytext=(24, 94.14),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                connectionstyle='arc3,rad=0.2'))
    ax.text(13, 89, '10.8× fewer params\n2.3% accuracy drop',
            fontsize=9, ha='center', color='gray', style='italic')

    ax.set_xlabel('Parameters (millions)', fontsize=12)
    ax.set_ylabel('Best Accuracy (%)', fontsize=12)
    ax.set_title('Parameter Efficiency: Accuracy vs Model Size', fontsize=14,
                 fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.2)
    ax.set_ylim(45, 100)

    plt.tight_layout()
    out = OUTPUT_DIR / 'qtl_parameter_efficiency.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: Architecture & Transfer Flow Diagram
# ══════════════════════════════════════════════════════════════════════════════
def chart_transfer_flow():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(7, 6.6, 'Quantum Transfer Learning Pipeline',
            fontsize=16, fontweight='bold', ha='center', va='top',
            color='#1e293b')

    # ── SOURCE MODEL (top) ──
    source_y = 5.2
    # Backbone
    ax.add_patch(plt.Rectangle((0.5, source_y-0.4), 3, 0.8, facecolor='#DBEAFE',
                                edgecolor='#2563EB', linewidth=2, zorder=2))
    ax.text(2, source_y, 'ResNet50\n(23.5M params)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1e40af')

    # Projector
    ax.add_patch(plt.Rectangle((4, source_y-0.3), 2, 0.6, facecolor='#E0E7FF',
                                edgecolor='#4F46E5', linewidth=1.5, zorder=2))
    ax.text(5, source_y, 'Projector\n2048→256', ha='center', va='center',
            fontsize=8, color='#3730a3')

    # Quantum Layer
    ax.add_patch(plt.Rectangle((6.5, source_y-0.3), 2.5, 0.6, facecolor='#FEF3C7',
                                edgecolor='#D97706', linewidth=2, zorder=2))
    ax.text(7.75, source_y, '⚛ Quantum Layer\n8q, depth=2, 48 params', ha='center',
            va='center', fontsize=8, fontweight='bold', color='#92400e')

    # Classifier
    ax.add_patch(plt.Rectangle((9.5, source_y-0.3), 2, 0.6, facecolor='#DCFCE7',
                                edgecolor='#16A34A', linewidth=1.5, zorder=2))
    ax.text(10.5, source_y, 'Classifier\n8→10, 90p', ha='center', va='center',
            fontsize=8, color='#166534')

    # Result
    ax.add_patch(plt.Rectangle((12, source_y-0.3), 1.5, 0.6, facecolor='#2563EB',
                                edgecolor='#1d4ed8', linewidth=1.5, zorder=2))
    ax.text(12.75, source_y, '94.1%', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')

    # Arrows
    for x_start, x_end in [(3.5, 4), (6, 6.5), (9, 9.5), (11.5, 12)]:
        ax.annotate('', xy=(x_end, source_y), xytext=(x_start, source_y),
                    arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.5))

    ax.text(0.3, source_y + 0.55, 'SOURCE MODEL', fontsize=10, fontweight='bold',
            color='#2563EB')

    # ── TRANSFER ARROWS (middle) ──
    transfer_y = 3.8
    ax.text(7, transfer_y + 0.3, '⬇  Extended Weight Transfer (394 params)  ⬇',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#DC2626')

    # Draw transfer arrows — quantum, classifier, projector bias
    for x, label, color in [(5, 'proj bias\n(256p)', '#4F46E5'),
                             (7.75, 'quantum\n(48p)', '#D97706'),
                             (10.5, 'classifier\n(90p)', '#16A34A')]:
        ax.annotate('', xy=(x, transfer_y - 0.5), xytext=(x, source_y - 0.4),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                   linestyle='--'))
        ax.text(x, transfer_y - 0.15, label, ha='center', va='top', fontsize=7,
                color=color, fontweight='bold')

    # ── TARGET MODEL (bottom) ──
    target_y = 2.2
    # Backbone
    ax.add_patch(plt.Rectangle((0.5, target_y-0.4), 3, 0.8, facecolor='#FEF9C3',
                                edgecolor='#CA8A04', linewidth=2, zorder=2))
    ax.text(2, target_y, 'LeNet Improved\n(128K params)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#854d0e')

    # Projector
    ax.add_patch(plt.Rectangle((4, target_y-0.3), 2, 0.6, facecolor='#E0E7FF',
                                edgecolor='#4F46E5', linewidth=1.5, zorder=2,
                                linestyle='--'))
    ax.text(5, target_y, 'Projector\n8192→256', ha='center', va='center',
            fontsize=8, color='#3730a3')

    # Quantum Layer
    ax.add_patch(plt.Rectangle((6.5, target_y-0.3), 2.5, 0.6, facecolor='#FEF3C7',
                                edgecolor='#D97706', linewidth=2, zorder=2,
                                linestyle='--'))
    ax.text(7.75, target_y, '⚛ Quantum Layer\n(transferred)', ha='center',
            va='center', fontsize=8, fontweight='bold', color='#92400e')

    # Classifier
    ax.add_patch(plt.Rectangle((9.5, target_y-0.3), 2, 0.6, facecolor='#DCFCE7',
                                edgecolor='#16A34A', linewidth=1.5, zorder=2,
                                linestyle='--'))
    ax.text(10.5, target_y, 'Classifier\n(transferred)', ha='center', va='center',
            fontsize=8, color='#166534')

    # Result
    ax.add_patch(plt.Rectangle((12, target_y-0.3), 1.5, 0.6, facecolor='#F59E0B',
                                edgecolor='#d97706', linewidth=1.5, zorder=2))
    ax.text(12.75, target_y, '91.9%', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')

    # Arrows
    for x_start, x_end in [(3.5, 4), (6, 6.5), (9, 9.5), (11.5, 12)]:
        ax.annotate('', xy=(x_end, target_y), xytext=(x_start, target_y),
                    arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.5))

    ax.text(0.3, target_y + 0.55, 'TARGET MODEL', fontsize=10, fontweight='bold',
            color='#CA8A04')

    # ── Feature Distillation annotation ──
    ax.add_patch(plt.Rectangle((1, 0.7), 5.5, 0.8, facecolor='#FEE2E2',
                                edgecolor='#DC2626', linewidth=1, zorder=2,
                                alpha=0.7))
    ax.text(3.75, 1.1, 'Feature-Aligned Knowledge Distillation\n'
            'MSE on 256-dim projected features + KL-div on logits',
            ha='center', va='center', fontsize=8, color='#991b1b')

    # ── Stats box ──
    ax.add_patch(plt.Rectangle((7.5, 0.5), 6, 1.2, facecolor='#F8FAFC',
                                edgecolor='#CBD5E1', linewidth=1, zorder=2))
    stats_text = (
        'Parameter Reduction: 10.8×  (24M → 2.2M)\n'
        'Accuracy Gap:  2.3%  (94.1% → 91.9%)\n'
        'Config: 8 qubits, depth=2, amplitude enc, VQC ansatz'
    )
    ax.text(10.5, 1.1, stats_text, ha='center', va='center', fontsize=9,
            family='monospace', color='#334155')

    plt.tight_layout()
    out = OUTPUT_DIR / 'qtl_architecture_flow.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  ✓ {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating QTL analysis charts...')
    chart_accuracy_comparison()
    chart_convergence()
    chart_parameter_efficiency()
    chart_transfer_flow()
    print('\n✅ All charts generated in reports/')
