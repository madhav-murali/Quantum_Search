# Quantum Transfer Learning - Execution Guide

This guide walks through running the complete quantum transfer learning experiments from ResNet50 to LeNet5.

## Prerequisites

Activate your virtual environment first:
```bash
cd /home/madhav/projects/Quantum
source .venv/bin/activate
```

## Phase 1: Verification Tests (5 minutes)

Run these tests to verify the implementation is working correctly.

### Test 1: Feature Standardization Compatibility
```bash
python -c "
import sys
sys.path.insert(0, '/home/madhav/projects/Quantum')

from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel
import torch

print('Testing feature standardization compatibility...\n')

# ResNet50 with standard dim
backbone_r, feat_r = BackboneFactory.create('resnet50', pretrained=False, in_channels=3)
model_r = HybridGeoModel(backbone_r, feat_r, n_classes=10, n_qubits=8, n_qlayers=2, 
                         standard_dim=256, encoding='amplitude', ansatz='vqc')

# LeNet5 with standard dim
backbone_l, feat_l = BackboneFactory.create('lenet5', pretrained=False, in_channels=3)
model_l = HybridGeoModel(backbone_l, feat_l, n_classes=10, n_qubits=8, n_qlayers=2,
                         standard_dim=256, encoding='amplitude', ansatz='vqc')

# Test forward pass
x = torch.randn(2, 3, 64, 64)
out_r = model_r(x)
out_l = model_l(x)

assert out_r.shape == out_l.shape == (2, 10), f'Shape mismatch: {out_r.shape}, {out_l.shape}'
print('✓ Feature standardization works correctly\n')
print(f'  ResNet50 (2048-dim) -> 256-dim -> quantum -> output: {out_r.shape}')
print(f'  LeNet5 (84-dim) -> 256-dim -> quantum -> output: {out_l.shape}')
"
```

### Test 2: Quantum Weight Save/Load
```bash
python -c "
from src.models.quantum_layers import QuantumLayer
import torch
import os

print('Testing quantum weight save/load...\n')

# Create and save
q1 = QuantumLayer(n_qubits=8, n_layers=2, encoding='amplitude', ansatz='vqc')
os.makedirs('checkpoints', exist_ok=True)
q1.save_quantum_weights('checkpoints/test_weights.pth')

# Load into new layer
q2 = QuantumLayer(n_qubits=8, n_layers=2, encoding='amplitude', ansatz='vqc')
q2.load_quantum_weights('checkpoints/test_weights.pth')

print('✓ Quantum weight save/load works correctly')
"
```

### Test 3: Parameter Freezing
```bash
python -c "
from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel

print('Testing parameter freezing...\n')

backbone, feat = BackboneFactory.create('lenet5', pretrained=False, in_channels=3)
model = HybridGeoModel(backbone, feat, n_classes=10, n_qubits=8, n_qlayers=2,
                       standard_dim=256, encoding='amplitude', ansatz='vqc',
                       freeze_quantum=True)

frozen = sum(1 for p in model.quantum_layer.parameters() if not p.requires_grad)
total = sum(1 for p in model.quantum_layer.parameters())

assert frozen == total, 'Not all quantum parameters frozen'
print(f'✓ Parameter freezing works: {frozen}/{total} quantum parameters frozen')
"
```

### Test 4: Parameter Counting
```bash
python scripts/count_parameters.py --config qtl_source_resnet_amplitude --n_qubits 8 --q_depth 2
python scripts/count_parameters.py --config qtl_lenet_frozen --n_qubits 8 --q_depth 2
```

**Expected output:**
- ResNet50: ~25.6M parameters
- LeNet5: ~42K parameters  
- Reduction: ~600x fewer parameters

---

## Phase 2: Training Experiments (3-4 hours)

### Experiment 1: Train Source Model (ResNet50 with standardized features)

**Purpose:** Train the ResNet50 hybrid model that will be the source for quantum weight transfer.

```bash
python run_experiments.py \
    --config qtl_source_resnet_amplitude \
    --epochs 10 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --bands RGB
```

**Expected:**
- Training time: ~1-1.5 hours (10 epochs)
- Validation accuracy: ~88-90%
- Saves checkpoint to: `checkpoints/qtl_source_resnet_amplitude.pth`

---

### Experiment 2: Extract Quantum Weights

**Purpose:** Extract just the quantum layer weights from the trained ResNet50 model.

```bash
python scripts/extract_quantum_weights.py \
    --checkpoint checkpoints/qtl_source_resnet_amplitude.pth \
    --output checkpoints/qtl_source_quantum_weights.pth
```

**Expected output:**
```
✓ Quantum weights extracted successfully!
  Saved to: checkpoints/qtl_source_quantum_weights.pth
  Configuration:
    - n_qubits: 8
    - n_layers: 2
    - encoding: amplitude
    - ansatz: vqc
```

---

### Experiment 3A: LeNet5 with Frozen Quantum Weights

**Purpose:** Transfer quantum weights to LeNet5 and train only the classical components.

```bash
python run_experiments.py \
    --config qtl_lenet_frozen \
    --epochs 15 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --bands RGB
```

**Expected:**
- Training time: ~30-45 minutes (faster since quantum layer is frozen)
- Validation accuracy: 90-92%
- Results saved to: `results/qtl_lenet_frozen_results.json`

---

### Experiment 3B: LeNet5 with Fine-tuned Quantum Weights

**Purpose:** Transfer quantum weights to LeNet5 and fine-tune everything end-to-end.

```bash
python run_experiments.py \
    --config qtl_lenet_finetuned \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --bands RGB
```

**Expected:**
- Training time: ~1 hour
- Validation accuracy: 92-94%
- Best overall QTL performance
- Results saved to: `results/qtl_lenet_finetuned_results.json`

---

### Experiment 4: LeNet5 Baseline (No Transfer)

**Purpose:** Train LeNet5 from scratch for comparison.

```bash
python run_experiments.py \
    --config lenet5_baseline_amplitude \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --bands RGB
```

**Expected:**
- Training time: ~1 hour
- Validation accuracy: ~82-85%
- This shows the value of transfer learning (10%+ improvement)
- Results saved to: `results/lenet5_baseline_amplitude_results.json`

---

## Phase 3: Results Analysis

### Quick Results Check

View the final accuracies:
```bash
python -c "
import json
from pathlib import Path

configs = [
    'qtl_source_resnet_amplitude',
    'qtl_lenet_frozen', 
    'qtl_lenet_finetuned',
    'lenet5_baseline_amplitude'
]

print('\nQuantum Transfer Learning Results')
print('='*70)
print(f'{'Model':<35} {'Final Acc':<12} {'Best Acc':<12}')
print('-'*70)

for config in configs:
    result_file = Path(f'results/{config}_results.json')
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            final_acc = data['val_acc'][-1] * 100
            best_acc = max(data['val_acc']) * 100
            print(f'{config:<35} {final_acc:>6.2f}%     {best_acc:>6.2f}%')
    else:
        print(f'{config:<35} (not run yet)')

print('='*70)
"
```

### Generate Comparison Charts

Create a script to visualize the results:

```bash
cat > reports/generate_qtl_comparison.py << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
configs = {
    'ResNet50 Source': 'qtl_source_resnet_amplitude',
    'LeNet5 Frozen QTL': 'qtl_lenet_frozen',
    'LeNet5 Fine-tuned QTL': 'qtl_lenet_finetuned',
    'LeNet5 Baseline': 'lenet5_baseline_amplitude'
}

results = {}
for name, config in configs.items():
    result_file = Path(f'results/{config}_results.json')
    if result_file.exists():
        with open(result_file) as f:
            results[name] = json.load(f)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
names = list(results.keys())
accuracies = [max(results[name]['val_acc']) * 100 for name in names]
params = [25600000, 42000, 42000, 42000]  # Approximate parameters

colors = ['#3498db', '#2ecc71', '#27ae60', '#95a5a6']
bars = ax1.bar(range(len(names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax1.set_title('Quantum Transfer Learning: Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=15, ha='right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([75, 100])

# Add accuracy labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Parameters vs Accuracy
ax2.scatter([params[0]/1e6], [accuracies[0]], s=300, c=colors[0], 
            alpha=0.7, edgecolors='black', label=names[0])
for i in range(1, len(names)):
    ax2.scatter([params[i]/1e3], [accuracies[i]], s=300, c=colors[i],
                alpha=0.7, edgecolors='black', label=names[i])

ax2.set_xlabel('Parameters (M for ResNet, K for LeNet)', fontsize=12)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax2.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('results/qtl_comparison.png', dpi=300, bbox_inches='tight')
print('Saved: results/qtl_comparison.png')

# Print summary table
print('\n' + '='*90)
print('Quantum Transfer Learning Summary')
print('='*90)
print(f'{"Model":<30} {"Params":<15} {"Accuracy":<12} {"vs Baseline":<15}')
print('-'*90)

baseline_acc = accuracies[3] if len(accuracies) > 3 else 0
for i, (name, acc, param) in enumerate(zip(names, accuracies, params)):
    param_str = f'{param/1e6:.1f}M' if param > 1e6 else f'{param/1e3:.1f}K'
    improvement = acc - baseline_acc if i > 0 and i != 3 else 0
    improvement_str = f'+{improvement:.1f}%' if improvement > 0 else '-'
    print(f'{name:<30} {param_str:<15} {acc:>6.2f}%      {improvement_str:<15}')

print('='*90)
print(f'\nTransfer Learning Benefit: +{max(accuracies[1:3]) - baseline_acc:.1f}% accuracy improvement')
print(f'Parameter Reduction: {params[0]/params[1]:.0f}x fewer parameters (ResNet50 vs LeNet5)')
EOF

python reports/generate_qtl_comparison.py
```

---

## Success Criteria Checklist

After running all experiments, verify:

- [ ] **ResNet50 source** trained and checkpoint saved
- [ ] **Quantum weights** extracted successfully
- [ ] **LeNet5 frozen** achieves 90-92% accuracy
- [ ] **LeNet5 fine-tuned** achieves 92-94% accuracy
- [ ] **LeNet5 baseline** achieves ~82-85% accuracy
- [ ] **Transfer benefit** is >10% (fine-tuned vs baseline)
- [ ] **Parameter count** verified: LeNet5 ~42K vs ResNet50 ~25.6M
- [ ] **Comparison charts** generated

---

## Troubleshooting

### If quantum weights file not found:
```bash
# Check if checkpoint exists
ls -lh checkpoints/qtl_source_resnet_amplitude.pth

# Re-extract weights
python scripts/extract_quantum_weights.py \
    --checkpoint checkpoints/qtl_source_resnet_amplitude.pth \
    --output checkpoints/qtl_source_quantum_weights.pth
```

### If accuracy is lower than expected:
- Increase epochs (try 30 instead of 10-20)
- Adjust learning rate: `--lr 5e-5` or `--lr 2e-4`
- Use more qubits: `--n_qubits 12` or `--q_depth 3`

### If training is too slow:
- Reduce batch size: `--batch_size 16`
- Use fewer epochs for testing: `--epochs 5`
- Use smaller qubits: `--n_qubits 4 --q_depth 1`

---

## Quick Start (Run All)

If you want to run everything in sequence:

```bash
#!/bin/bash
# Run all QTL experiments

echo "Step 1: Training ResNet50 source model..."
python run_experiments.py --config qtl_source_resnet_amplitude --epochs 10 --batch_size 32 --n_qubits 8 --q_depth 2

echo "Step 2: Extracting quantum weights..."
python scripts/extract_quantum_weights.py --checkpoint checkpoints/qtl_source_resnet_amplitude.pth --output checkpoints/qtl_source_quantum_weights.pth

echo "Step 3: Training LeNet5 frozen..."
python run_experiments.py --config qtl_lenet_frozen --epochs 15 --batch_size 32 --n_qubits 8 --q_depth 2

echo "Step 4: Training LeNet5 fine-tuned..."
python run_experiments.py --config qtl_lenet_finetuned --epochs 20 --batch_size 32 --n_qubits 8 --q_depth 2

echo "Step 5: Training LeNet5 baseline..."
python run_experiments.py --config lenet5_baseline_amplitude --epochs 20 --batch_size 32 --n_qubits 8 --q_depth 2

echo "Step 6: Generating comparison charts..."
python reports/generate_qtl_comparison.py

echo "All experiments complete!"
```

Save this as `run_all_qtl.sh`, make it executable with `chmod +x run_all_qtl.sh`, and run with `./run_all_qtl.sh`.
