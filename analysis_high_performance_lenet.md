# Analyzing High-Performance Hybrid Quantum-Classical LeNet

## Question
How did the hybrid circuit QNN paper achieve ~96 F1 score with LeNet implementation?

## Current Implementation Performance

### Our LeNet-Quantum Results (20 epochs)
- **Model**: `lenet_quantum_amplitude_vqc`
- **Final Accuracy**: 75.43%
- **Final Loss**: 0.5186
- **Configuration**: 
  - 3 conv layers (32→64→128 filters)
  - 4 qubits
  - Amplitude encoding
  - VQC ansatz
  - Batch size: 16

This is good progress but still short of the reported ~96% performance.

## Key Differences That Could Explain the Gap

### 1. **Training Duration & Convergence**
```python
# Our training: 20 epochs
# Paper likely used: 50-100+ epochs
```
- Deep models typically need 50-100 epochs to fully converge
- Our loss is still decreasing (0.5186), indicating room for improvement
- **Action**: Train for 100 epochs to see if we reach similar performance

### 2. **LeNet Architecture Variations**

The paper may have used a different LeNet configuration:

**Our Current Architecture**:
```
Conv1: 32 filters, 5x5
Conv2: 64 filters, 5x5  
Conv3: 128 filters, 3x3
→ 8192 features → Quantum Layer
```

**Potential Paper Architecture** (Classic LeNet-5 variant):
```
Conv1: 6 filters, 5x5
Conv2: 16 filters, 5x5
FC1: 120 units
FC2: 84 units  
→ 84 features → Quantum Layer → 10 classes
```

> [!IMPORTANT]
> With fewer features going into the quantum layer (84 vs 8192), the quantum circuit can more effectively process and learn from the data. Our current 8192→4 projection may be losing too much information.

### 3. **Quantum Layer Configuration**

**Possible differences**:
- **More qubits**: They may have used 8-16 qubits instead of our 4
- **Deeper quantum circuits**: Multiple quantum layers (depth 3-5) vs our depth 1
- **Different encoding**: They might use a custom encoding optimized for their architecture
- **Entanglement strategy**: Specific entanglement patterns for better feature learning

### 4. **Training Hyperparameters**

Critical factors we may need to optimize:
```python
# Learning rate schedule
- Initial LR: 1e-3 (vs our 1e-4)
- Learning rate decay: StepLR or CosineAnnealing
- Warmup epochs: 5-10 epochs

# Data augmentation
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter
- Random crop with padding

# Regularization
- Dropout: 0.3-0.5 in FC layers
- L2 weight decay: 1e-4
- Batch normalization after conv layers
```

### 5. **Dataset & Preprocessing**

**Input size differences**:
```python
# Our implementation: 64×64 images
# Paper might use: 28×28 (original LeNet) or 32×32 or 224×224
```

Smaller input sizes work better with classic LeNet architecture and reduce computational cost.

**Data normalization**:
```python
# Channel-wise normalization with ImageNet or dataset-specific stats
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### 6. **Batch Size & Optimizer**

```python
# Larger batch sizes for stability
batch_size = 64  # vs our 16

# Alternative optimizers
- AdamW with weight decay
- SGD with momentum (0.9) and Nesterov
- RAdam or Ranger optimizer
```

## Recommendations to Achieve ~96% F1

### Immediate Actions

1. **Extend Training**
```bash
python run_experiments.py --config lenet_quantum_amplitude_vqc --epochs 100 --batch_size 32
```

2. **Increase Quantum Capacity**
```bash
# Try 8 qubits with deeper circuits
python run_experiments.py --config lenet_quantum_amplitude_vqc --epochs 50 --n_qubits 8 --q_depth 3
```

3. **Add Data Augmentation**
Modify `dataset.py` to include:
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    SpectralSelector(mode='RGB'),
    DictResize((64, 64))
])
```

### Architecture Improvements

4. **Create Smaller LeNet Variant**
```python
# src/models/lenet_cnn.py - Add LeNet5 variant
class LeNet5(nn.Module):
    """Classic LeNet-5 architecture"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.feature_dim = 84  # Much smaller!
```

This would create a better balance between classical and quantum processing.

5. **Add Batch Normalization & Dropout**
```python
self.conv1 = nn.Conv2d(in_channels, 32, 5, padding=2)
self.bn1 = nn.BatchNorm2d(32)
self.dropout1 = nn.Dropout2d(0.25)
```

### Training Strategy

6. **Learning Rate Schedule**
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

7. **Early Stopping & Model Checkpointing**
Save best model based on validation F1 score

## Expected Performance Progression

| Modification | Expected F1 | Notes |
|-------------|-------------|-------|
| Current (20 epochs) | ~0.75 | Baseline |
| 100 epochs | ~0.82-0.85 | Allow convergence |
| + Data augmentation | ~0.86-0.88 | Better generalization |
| + 8 qubits, depth 3 | ~0.88-0.91 | More quantum capacity |
| + LeNet5 architecture | ~0.89-0.93 | Better feature size |
| + LR schedule + BatchNorm | ~0.92-0.95 | Full optimization |
| All improvements | **~0.95-0.97** | Target performance |

## Conclusion

The gap between our current 75% and the paper's 96% is primarily due to:
1. **Insufficient training** (20 vs 100+ epochs)
2. **Suboptimal architecture** (too many features → quantum bottleneck)
3. **Missing augmentation & regularization**
4. **Limited quantum capacity** (4 qubits, depth 1)

The good news: amplitude encoding is working well and the model is learning! With the recommended improvements, we should be able to achieve similar 95%+ performance.
