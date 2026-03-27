import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Hybrid Quantum-Classical Architecture', 
        fontsize=20, fontweight='bold', ha='center')
ax.text(5, 11, 'End-to-End Pipeline for Satellite Image Classification', 
        fontsize=14, ha='center', style='italic', color='gray')

# Define colors
color_input = '#95a5a6'
color_classical = '#3498db'
color_quantum = '#9b59b6'
color_output = '#2ecc71'

# Helper function to create boxes
def create_box(ax, x, y, width, height, label, sublabel, color, alpha=0.8):
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, 
                          alpha=alpha, linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2 + 0.15, label, 
            fontsize=12, fontweight='bold', ha='center', va='center')
    if sublabel:
        ax.text(x + width/2, y + height/2 - 0.2, sublabel, 
                fontsize=9, ha='center', va='center', style='italic')

def create_arrow(ax, x1, x2, y, label=''):
    arrow = FancyArrowPatch((x1, y), (x2, y),
                           arrowstyle='->', mutation_scale=30, 
                           lw=2.5, color='black')
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2, y+0.25, label, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 1. Input
create_box(ax, 0.5, 9, 1.5, 1, 'Input', 'EuroSAT\n64×64 RGB', color_input)

# Arrow to backbone
create_arrow(ax, 2.2, 2.8, 9.5)

# 2. Classical Backbone
create_box(ax, 3, 8.5, 2, 2, 'Classical Backbone', 
           'ResNet50 / ViT\n(Pre-trained)', color_classical)

# Show internal features
ax.text(4, 8.2, '2048-dim features', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Arrow to projector
create_arrow(ax, 5.2, 5.8, 9.5, 'Features')

# 3. Projector
create_box(ax, 6, 9, 1.5, 1, 'Projector', 'Linear\nReduction', color_classical, alpha=0.6)

# Dimension info
ax.text(6.75, 8.7, '2048 → n_qubits', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Arrow to quantum layer
create_arrow(ax, 7.7, 8.3, 9.5, 'Compressed')

# 4. Quantum Layer (BIG BOX)
quantum_box = FancyBboxPatch((0.5, 4.5), 8, 3.5, 
                              boxstyle="round,pad=0.15", 
                              edgecolor='purple', facecolor=color_quantum, 
                              alpha=0.2, linewidth=3)
ax.add_patch(quantum_box)
ax.text(4.5, 7.7, 'QUANTUM LAYER', fontsize=14, fontweight='bold', 
        ha='center', color='purple')

# 4a. Encoding
create_box(ax, 1, 5.8, 1.8, 0.8, 'Encoding', 
           'Angle/Amplitude/IQP', color_quantum, alpha=0.8)
ax.text(1.9, 5.4, 'x → |ψ(x)⟩', fontsize=10, ha='center', style='italic')

# 4b. Quantum State
create_box(ax, 3.2, 5.8, 1.5, 0.8, 'State', '|ψ⟩', color_quantum, alpha=0.6)

# 4c. Ansatz / VQC
create_box(ax, 5.2, 5.8, 2, 0.8, 'Variational Circuit', 
           'VQC / QAOA / QLSTM', color_quantum, alpha=0.8)

# Show circuit details
ax.text(6.2, 5.4, 'Entangling Gates', fontsize=8, ha='center',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.6))

# 4d. Measurement
create_box(ax, 7.6, 5.8, 1.2, 0.8, 'Measure', '⟨Z⟩', color_quantum, alpha=0.6)

# Internal quantum arrows
for x_start, x_end, y_pos in [(2.9, 3.1, 6.2), (4.8, 5.1, 6.2), (7.3, 7.5, 6.2)]:
    create_arrow(ax, x_start, x_end, y_pos)

# Arrow to classifier
ax.plot([4.5, 4.5], [4.3, 3.7], 'k-', lw=2.5)
arrow = FancyArrowPatch((4.5, 3.7), (4.5, 3.1),
                       arrowstyle='->', mutation_scale=30, 
                       lw=2.5, color='black')
ax.add_patch(arrow)
ax.text(5.2, 3.9, 'Quantum\nFeatures', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 5. Classifier
create_box(ax, 3.5, 1.5, 2, 1, 'Classifier', 
           'Linear(n_qubits → 10)', color_classical, alpha=0.6)

# Arrow to output
ax.plot([4.5, 4.5], [1.3, 0.7], 'k-', lw=2.5)
arrow = FancyArrowPatch((4.5, 0.7), (4.5, 0.1),
                       arrowstyle='->', mutation_scale=30, 
                       lw=2.5, color='black')
ax.add_patch(arrow)

# 6. Output
ax.text(4.5, -0.5, '10 Class Probabilities', fontsize=12, ha='center',
        fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor=color_output, alpha=0.5, pad=0.5))

# Add tensor shape annotations on the right
ax.text(9.2, 9.5, 'Shape: (B, 3, 64, 64)', fontsize=8, fontfamily='monospace')
ax.text(9.2, 9, 'Shape: (B, 2048)', fontsize=8, fontfamily='monospace')
ax.text(9.2, 6.2, 'Shape: (B, n_q)', fontsize=8, fontfamily='monospace')
ax.text(9.2, 2, 'Shape: (B, 10)', fontsize=8, fontfamily='monospace')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input Data'),
    mpatches.Patch(facecolor=color_classical, edgecolor='black', label='Classical Layers'),
    mpatches.Patch(facecolor=color_quantum, edgecolor='black', label='Quantum Layers'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
          framealpha=0.9)

# Add configuration box
config_text = """Best Configuration:
• 8 Qubits, Depth 2
• Amplitude Encoding
• VQC Ansatz
• Accuracy: 94.2%"""

ax.text(0.5, 2.5, config_text, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgreen', 
                  alpha=0.7, pad=0.5),
        verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/architecture_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved architecture_diagram.png")
