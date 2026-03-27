#!/usr/bin/env python3
"""
Extract quantum weights from a trained hybrid model checkpoint.

This script loads a trained hybrid model, extracts only the quantum layer weights,
and saves them to a separate file for quantum transfer learning.

Usage:
    python scripts/extract_quantum_weights.py \
        --checkpoint checkpoints/qtl_source_resnet_amplitude.pth \
        --output checkpoints/qtl_source_quantum_weights.pth
"""

import argparse
import torch
from pathlib import Path
import sys

def extract_quantum_weights(checkpoint_path, output_path):
    """
    Extract quantum weights from a model checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        output_path (str): Path to save quantum weights
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Find quantum layer weights
    quantum_weights = {}
    for key, value in state_dict.items():
        if 'quantum_layer' in key:
            quantum_weights[key] = value
    
    if not quantum_weights:
        print("ERROR: No quantum layer weights found in checkpoint!")
        print("Available keys:", list(state_dict.keys()))
        return False
    
    print(f"Found {len(quantum_weights)} quantum layer parameters:")
    for key in quantum_weights.keys():
        print(f"  - {key}: {quantum_weights[key].shape}")
    
    # Extract quantum layer config
    # The weights are in 'quantum_layer.qlayer.weights'
    if 'quantum_layer.qlayer.weights' in quantum_weights:
        weights = quantum_weights['quantum_layer.qlayer.weights']
        
        # Infer configuration from weight shape
        # For VQC: (n_layers, n_qubits, 3)
        # For QAOA: (n_layers, 2, n_qubits)
        if len(weights.shape) == 3:
            n_layers, n_qubits, third_dim = weights.shape
            if third_dim == 3:
                ansatz = 'vqc'
                weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            elif weights.shape[1] == 2:  # QAOA format
                n_layers, _, n_qubits = weights.shape
                ansatz = 'qaoa'
                weight_shapes = {"weights": (n_layers, 2, n_qubits)}
            else:
                print(f"WARNING: Unknown weight shape: {weights.shape}")
                ansatz = 'unknown'
                weight_shapes = {"weights": weights.shape}
        else:
            print(f"WARNING: Unexpected weight shape: {weights.shape}")
            ansatz = 'unknown'
            weight_shapes = {"weights": weights.shape}
            n_layers = weights.shape[0] if len(weights.shape) > 0 else 1
            n_qubits = weights.shape[1] if len(weights.shape) > 1 else 4
    else:
        print("ERROR: Could not find quantum_layer.qlayer.weights")
        return False
    
    # Try to get encoding from checkpoint config
    encoding = 'amplitude'  # default
    if 'config' in checkpoint and 'encoding' in checkpoint['config']:
        encoding = checkpoint['config']['encoding']
    
    # Create quantum weights package
    quantum_package = {
        'weights': weights,
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'encoding': encoding,
        'ansatz': ansatz,
        'weight_shapes': weight_shapes
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantum_package, output_path)
    
    print(f"\n✓ Quantum weights extracted successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Configuration:")
    print(f"    - n_qubits: {n_qubits}")
    print(f"    - n_layers: {n_layers}")
    print(f"    - encoding: {encoding}")
    print(f"    - ansatz: {ansatz}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract quantum weights from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/qtl_source_resnet_amplitude.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save quantum weights (e.g., checkpoints/qtl_source_quantum_weights.pth)')
    
    args = parser.parse_args()
    
    success = extract_quantum_weights(args.checkpoint, args.output)
    sys.exit(0 if success else 1)
