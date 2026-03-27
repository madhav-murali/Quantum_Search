import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Data from search_advantage logs
data = {
    'Model': [
        'ResNet50\n(Classical)',
        'Hybrid ResNet +\nAmplitude (8Q, D2)',
        'Hybrid ResNet +\nAmplitude (8Q, D3)',
        'ViT\n(Classical)',
        'Hybrid ResNet +\nAngle (8Q)',
        'Hybrid ViT +\nQLSTM',
        'Hybrid ResNet +\nIQP',
        'Hybrid ViT +\nIQP QAOA'
    ],
    'Accuracy': [96.11, 94.2, 92.9, 90.86, 49.75, 22.84, 21.91, 13.46],
    'Category': ['Classical', 'Quantum (Amplitude)', 'Quantum (Amplitude)', 'Classical', 
                 'Quantum (Other)', 'Quantum (Other)', 'Quantum (Other)', 'Quantum (Other)']
}

df = pd.DataFrame(data)

# Color mapping
colors = []
for cat in df['Category']:
    if cat == 'Classical':
        colors.append('#2ecc71')  # Green
    elif cat == 'Quantum (Amplitude)':
        colors.append('#3498db')  # Blue
    else:
        colors.append('#e74c3c')  # Red

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(range(len(df)), df['Accuracy'], color=colors, edgecolor='black', linewidth=1.2)

# Add data labels
for i, (bar, acc) in enumerate(zip(bars, df['Accuracy'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize
ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Comparison - EuroSAT Classification\n(10 Epochs Training)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Model'], rotation=15, ha='right', fontsize=10)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Classical Baseline'),
    Patch(facecolor='#3498db', edgecolor='black', label='Quantum (Amplitude Encoding)'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Quantum (Other Encodings)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved performance_comparison.png")

# Create time comparison chart
fig, ax = plt.subplots(figsize=(12, 6))
time_data = {
    'Model': ['ResNet50', 'Hybrid ResNet\nAmplitude (8Q, D2)', 
              'Hybrid ResNet\nAmplitude (8Q, D3)', 'ViT'],
    'Time': [25.12, 55.69, 67.17, 254.69],
    'Accuracy': [96.11, 94.2, 92.9, 90.86]
}
df_time = pd.DataFrame(time_data)

bars = ax.bar(range(len(df_time)), df_time['Time'], 
              color=['#2ecc71', '#3498db', '#5da8db', '#95a5a6'], 
              edgecolor='black', linewidth=1.2)

# Add labels
for i, (bar, time, acc) in enumerate(zip(bars, df_time['Time'], df_time['Accuracy'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{time:.1f}s\n({acc:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Time per Epoch (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Training Time Comparison\n(Accuracy shown in parentheses)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(range(len(df_time)))
ax.set_xticklabels(df_time['Model'], fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved time_comparison.png")

# Create encoding comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

encoding_data = {
    'Encoding': ['Amplitude\n(8Q, D2)', 'Amplitude\n(8Q, D3)', 'Angle\n(8Q)', 'IQP\n(8Q)'],
    'Accuracy': [94.2, 92.9, 49.75, 21.91]
}
df_enc = pd.DataFrame(encoding_data)

colors_enc = ['#3498db', '#5da8db', '#e67e22', '#e74c3c']
bars = ax1.bar(range(len(df_enc)), df_enc['Accuracy'], color=colors_enc, 
               edgecolor='black', linewidth=1.2)

for bar, acc in zip(bars, df_enc['Accuracy']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{acc:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Quantum Encoding Strategy Comparison\n(Hybrid ResNet Models)', 
              fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(df_enc)))
ax1.set_xticklabels(df_enc['Encoding'], fontsize=10)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Depth comparison
depth_data = {
    'Configuration': ['Amplitude\nDepth 2', 'Amplitude\nDepth 3'],
    'Best_Acc': [94.2, 92.9],
    'Final_Acc': [88.83, 92.9],
    'Time': [55.69, 67.17]
}
df_depth = pd.DataFrame(depth_data)

x = np.arange(len(df_depth))
width = 0.35

bars1 = ax2.bar(x - width/2, df_depth['Best_Acc'], width, label='Best Accuracy',
                color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax2.bar(x + width/2, df_depth['Final_Acc'], width, label='Final Accuracy',
                color='#5da8db', edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Circuit Depth Impact on Performance\n(8 Qubits, Amplitude Encoding)', 
              fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(df_depth['Configuration'], fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

# Add time annotations
for i, time in enumerate(df_depth['Time']):
    ax2.text(i, 5, f'{time:.0f}s/epoch', ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/madhav/projects/Quantum/reports/encoding_depth_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved encoding_depth_analysis.png")

print("\nAll charts generated successfully!")
