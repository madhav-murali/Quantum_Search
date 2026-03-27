# Quick QTL Diagnostics Guide

## Run All Diagnostics (Sequential)

```bash
chmod +x run_qtl_diagnostics.sh
./run_qtl_diagnostics.sh
```

This will run 4 experiments **one at a time** with full output visible:
1. LeNet5 Baseline (with standard_dim=256)
2. LeNet5 Original (no standardization) 
3. QTL Frozen (higher learning rate)
4. QTL Fine-tuned

**Total time:** ~2-3 hours

---

## Run Individual Experiments

### Baseline (with standardization)
```bash
python run_experiments.py \
    --config lenet5_baseline_amplitude \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2
```

### Original LeNet5 (no standardization)
```bash
python run_experiments.py \
    --config lenet5_quantum_amplitude_vqc \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2
```

### QTL Frozen (higher LR)
```bash
python run_experiments.py \
    --config qtl_lenet_frozen \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --lr 5e-4
```

### QTL Fine-tuned
```bash
python run_experiments.py \
    --config qtl_lenet_finetuned \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2
```

---

## Generate Visualizations Only

If you already have results and just want to regenerate charts:

```bash
python reports/generate_qtl_analysis.py
```

This creates:
- `qtl_convergence_comparison.png` - Training curves
- `qtl_accuracy_comparison.png` - Bar chart comparison
- `qtl_parameter_efficiency.png` - Params vs accuracy scatter
- `qtl_detailed_results.md` - Full results table

---

## Quick Results Check

```bash
python -c "
import json
from pathlib import Path

configs = ['lenet5_baseline_amplitude', 'lenet5_quantum_amplitude_vqc', 
           'qtl_lenet_frozen', 'qtl_lenet_finetuned']

print('\nQuick Results:')
print('='*60)
for c in configs:
    f = Path(f'results/{c}_results.json')
    if f.exists():
        data = json.load(f.open())
        best = max(data['val_acc']) * 100
        final = data['val_acc'][-1] * 100
        print(f'{c:40} {best:6.2f}% (best)  {final:6.2f}% (final)')
    else:
        print(f'{c:40} Not run yet')
print('='*60)
"
```
