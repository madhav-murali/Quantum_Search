import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Updated data including paper results
data_comparison = {
    'Model': [
        'ResNet50\n(Classical)',
        'Our Hybrid ResNet\n+ Amplitude (8Q, D2)',
        'Paper: LeNet-5\n+ Real Amplitudes (4Q)',
        'Paper: Coarse-to-Fine\nQuantum Classifier',
        'Our Hybrid ResNet\n+ Amplitude (8Q, D3)',
        'ViT\n(Classical)',
        'Paper: LeNet-5\n+ Bellman Circuit',
        'Our Hybrid ResNet\n+ Angle (8Q)',
    ],
    'Accuracy': [96.11, 94.2, 92.0, 97.0, 92.9, 90.86, 84.0, 49.75],
    'Source': ['Our Work', 'Our Work', 'Sebastianelli 2021', 'Sebastianelli 2021', 
               'Our Work', 'Our Work', 'Sebastianelli 2021', 'Our Work'],
    'Category': ['Classical', 'Quantum (Ours)', 'Quantum (Paper)', 'Quantum (Paper)',
                 'Quantum (Ours)', 'Classical', 'Quantum (Paper)', 'Quantum (Ours)']
}

df = pd.DataFrame(data_comparison)

# Color mapping - updated for sources
colors = []
for cat in df['Category']:
    if cat == 'Classical':
        colors.append('#2ecc71')  # Green
    elif 'Ours' in cat:
        colors.append('#3498db')  # Blue (our work)
    else:
        colors.append('#9b59b6')  # Purple (paper)

# Create comprehensive comparison chart
fig, ax = plt.subplots(figsize=(16, 8))
bars = ax.bar(range(len(df)), df['Accuracy'], color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add data labels
for i, (bar, acc, source) in enumerate(zip(bars, df['Accuracy'], df['Source'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Add source label at bottom
    if 'Paper' in source:
        ax.text(bar.get_x() + bar.get_width()/2., 3,
                '[Paper]',
                ha='center', va='bottom', fontsize=8, style='italic', color='purple')

# Customize
ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Hybrid Quantum Models Performance: Our Results vs. Sebastianelli et al. (2021)\nAll experiments on EuroSAT Dataset', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Model'], rotation=20, ha='right', fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

# Add horizontal line for target
ax.axhline(y=97, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Paper Best (Coarse-to-Fine)')
ax.axhline(y=96.11, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Classical Baseline')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Classical Baseline'),
    Patch(facecolor='#3498db', edgecolor='black', label='Our Quantum Hybrids'),
    Patch(facecolor='#9b59b6', edgecolor='black', label='Sebastianelli et al. 2021')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=12, framealpha=0.9)

# Add annotation box
textstr = 'Key Insight: Our 94.2% result approaches paper\'s\n92% standard method. With coarse-to-fine\nstrategy, potential to reach 97% accuracy.'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/performance_comparison_with_paper.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved performance_comparison_with_paper.png")

# Create roadmap chart showing convergence potential
fig, ax = plt.subplots(figsize=(14, 7))

stages = ['Current\n(10 epochs)', 'Extended\nTraining\n(50 epochs)', 'Architecture\nOptimization', 
          'Coarse-to-Fine\nClassification', 'Target:\nPaper Best']
accuracies = [94.2, 95.5, 96.0, 97.0, 97.0]
colors_road = ['#3498db', '#5dade2', '#85c1e9', '#9b59b6', '#8e44ad']

bars = ax.barh(range(len(stages)), accuracies, color=colors_road, 
               edgecolor='black', linewidth=1.5, alpha=0.85)

# Add labels
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{acc:.1f}%',
            ha='left', va='center', fontsize=13, fontweight='bold')

# Add annotations
annotations = [
    'Starting Point:\nHybrid ResNet\n8Q, Depth 2',
    'Expected:\n+1.3% gain\n(from paper)',
    'Fine-tuning:\n+0.5%',
    'Multi-stage\nClassification',
    'Sebastianelli\net al. 2021'
]

for i, (bar, ann) in enumerate(zip(bars, annotations)):
    ax.text(2, bar.get_y() + bar.get_height()/2.,
            ann,
            ha='left', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Convergence Roadmap: Path to 97% Accuracy\nBased on Sebastianelli et al. (2021) Results', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_yticks(range(len(stages)))
ax.set_yticklabels(stages, fontsize=11)
ax.set_xlim(90, 100)
ax.grid(axis='x', alpha=0.3)

# Add baseline reference
ax.axvline(x=96.11, color='green', linestyle='--', linewidth=2, alpha=0.5, 
           label='Classical ResNet50 Baseline')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/convergence_roadmap.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved convergence_roadmap.png")

# Create architecture comparison
fig, ax = plt.subplots(figsize=(12, 8))

architectures = [
    'Our:\nResNet50\n+ Amplitude',
    'Paper:\nLeNet-5\n+ Real Amp',
    'Paper:\nLeNet-5\n+ Bellman',
    'Our:\nResNet50\n+ Angle'
]
params = [25.6, 0.042, 0.042, 25.6]  # Millions of parameters
accs = [94.2, 92.0, 84.0, 49.75]

scatter = ax.scatter(params, accs, s=[500, 500, 500, 500], 
                     c=['#3498db', '#9b59b6', '#9b59b6', '#e74c3c'],
                     alpha=0.7, edgecolors='black', linewidth=2)

# Add labels
for i, (arch, param, acc) in enumerate(zip(architectures, params, accs)):
    ax.annotate(f'{arch}\n{acc:.1f}%', 
                xy=(param, acc), 
                xytext=(15, 10) if i % 2 == 0 else (-15, -15),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

ax.set_xlabel('Model Parameters (Millions)', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Accuracy vs. Model Complexity\nQuantum Hybrids Performance', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(alpha=0.3)
ax.set_ylim(40, 100)

# Add insight text
textstr = 'Insight: LeNet-5 quantum hybrids achieve\n92% with only 42K parameters!\nOur ResNet hybrids: 94.2% but 25M params.'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/complexity_comparison.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved complexity_comparison.png")

print("\nAll updated charts generated successfully!")
