#!/usr/bin/env python3
"""
Count parameters in different model configurations.

This script helps verify parameter counts for different models,
particularly useful for comparing ResNet50 vs LeNet5 in quantum transfer learning.

Usage:
    python scripts/count_parameters.py --config qtl_lenet_frozen
    python scripts/count_parameters.py --config qtl_source_resnet_amplitude
"""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel

def count_parameters(model, verbose=True):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        verbose (bool): Print detailed breakdown
    
    Returns:
        dict: Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Parameter Count Summary")
        print(f"{'='*60}")
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Frozen Parameters:     {frozen_params:,}")
        print(f"{'='*60}\n")
        
        # Component breakdown
        print("Component Breakdown:")
        print(f"{'Component':<30} {'Total':>12} {'Trainable':>12}")
        print("-" * 60)
        
        for name, module in model.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:<30} {module_total:>12,} {module_trainable:>12,}")
        print("-" * 60)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }

def main():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration name (e.g., qtl_lenet_frozen, qtl_source_resnet_amplitude)')
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of qubits')
    parser.add_argument('--q_depth', type=int, default=2,
                        help='Quantum layer depth')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Input channels (3 for RGB, 13 for all bands)')
    
    args = parser.parse_args()
    
    # Configuration mapping (from run_experiments.py)
    configs = {
        'qtl_source_resnet_amplitude': {
            'backbone': 'resnet50',
            'encoding': 'amplitude',
            'ansatz': 'vqc',
            'standard_dim': 256,
        },
        'qtl_lenet_frozen': {
            'backbone': 'lenet5',
            'encoding': 'amplitude',
            'ansatz': 'vqc',
            'standard_dim': 256,
            'freeze_quantum': True
        },
        'qtl_lenet_finetuned': {
            'backbone': 'lenet5',
            'encoding': 'amplitude',
            'ansatz': 'vqc',
            'standard_dim': 256,
            'freeze_quantum': False
        },
        'lenet5_baseline_amplitude': {
            'backbone': 'lenet5',
            'encoding': 'amplitude',
            'ansatz': 'vqc',
            'standard_dim': 256,
        },
        'baseline_resnet50': {
            'backbone': 'resnet50',
            'classical_only': True
        },
        'baseline_lenet5': {
            'backbone': 'lenet5',
            'classical_only': True
        }
    }
    
    if args.config not in configs:
        print(f"ERROR: Unknown config '{args.config}'")
        print(f"Available configs: {list(configs.keys())}")
        return 1
    
    config = configs[args.config]
    n_classes = 10  # EuroSAT
    
    print(f"\nBuilding model for config: {args.config}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Input channels: {args.in_channels}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Quantum depth: {args.q_depth}")
    
    # Build model
    backbone, feature_dim = BackboneFactory.create(
        config['backbone'], 
        pretrained=False,  # Don't download weights for counting
        in_channels=args.in_channels
    )
    
    if config.get('classical_only', False):
        # Classical baseline
        model = backbone
        import torch.nn as nn
        if 'resnet' in config['backbone']:
            model.fc = nn.Linear(feature_dim, n_classes)
        elif 'lenet' in config['backbone']:
            # LeNet already returns features, add classifier
            model = nn.Sequential(
                backbone,
                nn.Linear(feature_dim, n_classes)
            )
    else:
        # Quantum hybrid
        model = HybridGeoModel(
            backbone=backbone,
            feature_dim=feature_dim,
            n_classes=n_classes,
            n_qubits=args.n_qubits,
            n_qlayers=args.q_depth,
            encoding=config.get('encoding', 'amplitude'),
            ansatz=config.get('ansatz', 'vqc'),
            standard_dim=config.get('standard_dim', None),
            freeze_quantum=config.get('freeze_quantum', False)
        )
    
    # Count parameters
    counts = count_parameters(model, verbose=True)
    
    # Calculate reduction if comparing to ResNet50
    if 'lenet' in config['backbone']:
        resnet_params = 25_600_000  # Approximate ResNet50 hybrid params
        reduction = resnet_params / counts['total']
        print(f"\nParameter Reduction vs ResNet50: {reduction:.1f}x fewer parameters")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
