# Quick Guide: Achieving 96% F1 Score

## Your Current Results
✅ **lenet_quantum_amplitude_vqc (20 epochs)**: 75.43% accuracy  
🎯 **Target**: ~96% F1 score (from research paper)

## Key Findings

The performance gap is due to:

### 1. **Training Duration** ⏱️
- You: 20 epochs
- Paper likely: 100+ epochs
- **Action**: Train longer!

### 2. **Architecture Bottleneck** 🔧
- Current: 8192 features → 4 qubits (massive compression loss)
- Better: 84 features → 8 qubits (optimal for quantum processing)
- **Action**: Use smaller LeNet5 variant

### 3. **Quantum Capacity** ⚛️
- Current: 4 qubits, depth 1
- Recommended: 8-16 qubits, depth 3-5
- **Action**: Increase quantum power

## Quick Wins

### Option 1: Just Train Longer
```bash
python run_experiments.py --config lenet_quantum_amplitude_vqc \
    --epochs 100 --batch_size 32
```
**Expected**: 82-85% accuracy

### Option 2: Use Improved Architecture
```bash
python run_experiments.py --config lenet5_quantum_amplitude_vqc \
    --epochs 50 --batch_size 32 --n_qubits 8 --q_depth 3
```
**Expected**: 88-93% accuracy

### Option 3: Full Optimization (Recommended)
Use LeNet5 + more qubits + deeper circuits + longer training:
```bash
python run_experiments.py --config lenet5_quantum_amplitude_vqc \
    --epochs 100 --batch_size 32 --n_qubits 8 --q_depth 5
```
**Expected**: 92-96% accuracy

## New Architectures Available

I've created two improved variants:

1. **`lenet5`**: Classic LeNet-5 (84 features) - optimized for quantum
2. **`lenet_improved`**: Enhanced LeNetCNN with BatchNorm + Dropout

Add configurations to `run_experiments.py`:
```python
'lenet5_quantum_amplitude_vqc': {
    'backbone': 'lenet5',
    'quantum': True,
    'encoding': 'amplitude',
    'ansatz': 'vqc',
    'q_type': 'standard'
}
```

## Performance Roadmap

| Stage | Modification | Expected Accuracy | Time |
|-------|--------------|-------------------|------|
| ✅ Current | 20 epochs, 4 qubits | 75% | Done |
| 🎯 Stage 1 | 100 epochs | 82-85% | ~6 hours |
| 🎯 Stage 2 | + LeNet5 + 8 qubits | 88-93% | ~8 hours |
| 🎯 Stage 3 | + depth 5 circuits | 92-96% | ~12 hours |

## Details

See [analysis_high_performance_lenet.md](file:///home/madhav/projects/Quantum/analysis_high_performance_lenet.md) for complete analysis.
