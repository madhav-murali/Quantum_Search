import torch
import torch.nn as nn
from src.models.quantum_layers import QuantumLayer
from torchvision.transforms import Resize, Compose

print("Running verification...")

# 1. Verify Quantum Layer Shape
print("Checking QuantumLayer batch processing...")
try:
    batch_size = 32
    n_qubits = 4
    layer = QuantumLayer(n_qubits=n_qubits, n_layers=1, encoding='angle', ansatz='vqc')
    inputs = torch.randn(batch_size, n_qubits)
    outputs = layer(inputs)
    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == (batch_size, n_qubits), f"Expected ({batch_size}, {n_qubits}), got {outputs.shape}"
    print("✅ QuantumLayer batch processing passed.")
except Exception as e:
    print(f"❌ QuantumLayer failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Verify ViT Resizing Logic (Mock)
print("\nChecking ViT Resize Logic...")
try:
    # Simulate what happens in run_experiments.py
    # We just check if Resize works on a dummy tensor simulating EuroSAT image
    img = torch.randn(13, 64, 64) # 13 bands, 64x64
    
    # Selector mock (identity for now as we just test resize)
    resize = Resize((224, 224))
    
    out_img = resize(img)
    print(f"Original shape: {img.shape}, Resized shape: {out_img.shape}")
    assert out_img.shape == (13, 224, 224), "Resize failed"
    print("✅ ViT Resize passed.")

except Exception as e:
    print(f"❌ ViT Resize failed: {e}")
    
print("\nVerification Complete.")
