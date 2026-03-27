import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# Local imports
from src.data.dataset import get_dataset, SpectralSelector, DictResize
from src.data.splitter import EuroSATSplitter
from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel
from src.utils.metrics import calculate_metrics
from src.utils.env_config import EnvironmentManager
from torchvision.transforms import Compose

# =========================================================================
# EXPERIMENT CONFIGURATIONS
# Includes specific configurations for mult-stage classification, 
# molecular encoding, new datasets (SIRI-WHU, UC_M_LUC) and QTL variations.
# 
# Usage Examples:
#   - Train standard baseline:   `python run_experiments.py --config baseline_resnet50`
#   - Train with new datasets:   `python run_experiments.py --dataset SIRI-WHU --config siri_whu_qtl_finetuned`
#   - Train subset (e.g. 10%):   `python run_experiments.py --subset_fraction 0.1 --config siri_whu_qtl_finetuned`
#   - Molecular Encoding:        `python run_experiments.py --config hybrid_resnet_molecular_vqc`
#   - Multistage Classification: `python run_experiments.py --config multistage_resnet_quantum`
# =========================================================================

CONFIGS = {
    'baseline_resnet50': {
        'backbone': 'resnet50',
        'quantum': False,
        'encoding': 'N/A',
        'ansatz': 'N/A'
    },
    'baseline_vit': {
        'backbone': 'vit_base',
        'quantum': False,
        'encoding': 'N/A',
        'ansatz': 'N/A'
    },
    'baseline_lenet': {
        'backbone': 'lenet',
        'quantum': False,
        'encoding': 'N/A',
        'ansatz': 'N/A'
    },
    'hybrid_resnet_angle_vqc': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard'
    },
    'hybrid_resnet_amplitude_vqc': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard'
    },
    'hybrid_resnet_iqp_vqc': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'iqp',
        'ansatz': 'vqc',
        'q_type': 'standard'
    },
    'hybrid_vit_iqp_qaoa': {
        'backbone': 'vit_base',
        'quantum': True,
        'encoding': 'iqp',
        'ansatz': 'qaoa',
        'q_type': 'standard'
    },
    'hybrid_vit_qlstm': {
        'backbone': 'vit_base',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'qlstm'
    },
    'lenet_quantum_angle_vqc': {
        'backbone': 'lenet',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard'
    },
    'lenet_quantum_amplitude_vqc': {
        'backbone': 'lenet',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard'
    },
    'lenet_quantum_iqp_qaoa': {
        'backbone': 'lenet',
        'quantum': True,
        'encoding': 'iqp',
        'ansatz': 'qaoa',
        'q_type': 'standard'
    },
    'lenet5_quantum_amplitude_vqc': {
    'backbone': 'lenet5',
    'quantum': True,
    'encoding': 'amplitude',
    'ansatz': 'vqc',
    'q_type': 'standard'
    },
    # Quantum Transfer Learning (QTL) - IMPROVED with angle encoding
    'qtl_source_resnet_angle': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'save_model': True
    },
    'qtl_lenet_frozen_angle': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'quantum_weights_path': 'checkpoints/qtl_source_angle_weights.pth',
        'freeze_quantum': True
    },
    'qtl_lenet_finetuned_angle': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'quantum_weights_path': 'checkpoints/qtl_source_angle_weights.pth',
        'freeze_quantum': False
    },
    # =========================================================================
    # QTL WEIGHT GENERATION SCAFFOLDING
    # =========================================================================
    # These configs generate the pre-trained weights needed for Quantum
    # Transfer Learning (QTL) to a LeNet5 backbone.
    #
    # Workflow:
    #   1. Train source model (generates checkpoint + quantum weights):
    #      python run_experiments.py --config qtl_source_resnet_amplitude \
    #                                --epochs 20 --n_qubits 8 --q_depth 2
    #      → saves: checkpoints/qtl_source_resnet_amplitude.pth
    #
    #   2. Run QTL transfer to LeNet5 (auto-extracts quantum weights):
    #      python qtl/lenet5_amplitude_qtl.py --epochs 30
    #      → extracts: checkpoints/qtl_source_quantum_weights.pth
    #      → saves:    results/qtl_lenet5_amplitude_*.json
    #
    # Expected results:
    #   Source (ResNet50+Quantum): ~96% accuracy, ~25.6M params
    #   Target (LeNet5+Quantum):  ~94% accuracy, ~42K params (~600× reduction)
    #
    # The quantum layer config MUST match between source and target:
    #   n_qubits=8, q_depth=2, encoding=amplitude, ansatz=vqc, standard_dim=256
    # =========================================================================
    'qtl_source_resnet_amplitude': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256,
        'save_model': True
    },
    'qtl_lenet_frozen': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256,
        'quantum_weights_path': 'checkpoints/qtl_source_quantum_weights.pth',
        'freeze_quantum': True
    },
    'qtl_lenet_finetuned': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256,
        'quantum_weights_path': 'checkpoints/qtl_source_quantum_weights.pth',
        'freeze_quantum': False
    },
    'lenet5_baseline_amplitude': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256
    },
    
    # =========================================================================
    # NEW RESEARCH ADDITIONS: MULTISTAGE & MOLECULAR ENCODING
    # =========================================================================
    'multistage_resnet_quantum': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'multistage': True  # Enables auxiliary classifier early in the network
    },
    'hybrid_resnet_molecular_vqc': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'molecular', # Requires exactly 3 params per qubit (handled natively)
        'ansatz': 'vqc',
        'q_type': 'standard'
    },

    # =========================================================================
    # MIX & MATCH QTL DATASETS (SIRI-WHU / UC_M_LUC)
    # These configurations are ready-to-run variations across datasets.
    # Provide the `--dataset` and `--subset_fraction` flag in terminal to seamlessly
    # utilize them for cross-evaluation.
    # =========================================================================
    'siri_whu_qtl_source': {
        'backbone': 'resnet50',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256,
        'save_model': True
    },
    'siri_whu_qtl_finetuned': {
        'backbone': 'lenet5',
        'quantum': True,
        'encoding': 'amplitude',
        'ansatz': 'vqc',
        'q_type': 'standard',
        'standard_dim': 256,
        'quantum_weights_path': 'checkpoints/siri_whu_qtl_source.pth',
        'freeze_quantum': False
    },
    'uc_m_luc_vit_qlstm': {
        'backbone': 'vit_base',
        'quantum': True,
        'encoding': 'angle',
        'ansatz': 'vqc',
        'q_type': 'qlstm'
    }
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
        else:
            images, targets = batch[0].to(device), batch[1].to(device)
            
        images = images.float()
        # EuroSAT targets are Long class indices for CrossEntropy
        targets = targets.long()
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle multistage auxiliary outputs
        if isinstance(outputs, tuple):
            final_out, aux_out = outputs
            loss_final = criterion(final_out, targets)
            loss_aux = criterion(aux_out, targets)
            loss = loss_final + 0.3 * loss_aux  # Weight auxiliary loss by 0.3
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                targets = batch['label'].to(device)
            else:
                images, targets = batch[0].to(device), batch[1].to(device)
            
            images = images.float()
            targets = targets.long()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics['val_loss'] = running_loss / len(loader.dataset)
    
    return metrics

def run_experiment(config_name, args):
    # Copy config so we don't permanently modify the global dictionary
    config = CONFIGS[config_name].copy()
    
    # Allow CLI overrides
    if args.encoding is not None:
        config['encoding'] = args.encoding
    if args.ansatz is not None:
        config['ansatz'] = args.ansatz
        
    print(f"Running Experiment: {config_name} | Encoding: {config['encoding']} | Ansatz: {config.get('ansatz', 'N/A')}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    try:
        # Check environment and paths
        dataset_name = args.dataset
        dataset_root = EnvironmentManager.get_default_data_root(dataset_name) if args.env == 'auto' else args.data_root
        
        # Initialize
        full_dataset = get_dataset(root=dataset_root, dataset_name=dataset_name, download=True, subset_fraction=args.subset_fraction)
        
        # Determine properties
        if dataset_name.lower() == "eurosat":
            n_classes = 10
            in_channels = 13 if args.bands == 'ALL' else 3
        elif dataset_name.lower() == "siri-whu":
            n_classes = 12
            in_channels = 3
        elif dataset_name.lower() == "uc_m_luc":
            n_classes = 21
            in_channels = 3
        else:
            n_classes = 10
            in_channels = 3
        
        # Apply Spectral Selector Transforms manually if not in dataset
        # Torchgeo usually returns dict. We can wrap dataset or use transform in loader.
        # But get_dataset returned a standard EuroSAT object. 
        # Let's attach the transform to the dataset if possible or wrapper.
        # EuroSAT class allows 'transforms'.
        
        # Re-initializing with transform
        selector = SpectralSelector(mode=args.bands)
        
        # Determine target for transforms (underlying dataset if Subset)
        target_ds = full_dataset.dataset if hasattr(full_dataset, 'dataset') else full_dataset
        
        if 'vit' in config['backbone']:
             target_ds.transforms = Compose([selector, DictResize((224, 224))])
        else:
             target_ds.transforms = selector
        
        # Split
        splitter = EuroSATSplitter(full_dataset, test_size=0.2, val_size=0.1)
        train_loader, val_loader, test_loader = splitter.get_loaders(batch_size=args.batch_size)
        
    except Exception as e:
        print(f"Error loading EuroSAT: {e}")
        return

    # Model
    backbone, feature_dim = BackboneFactory.create(config['backbone'], pretrained=True, in_channels=in_channels)
    
    if config['quantum']:
        model = HybridGeoModel(
            backbone=backbone,
            feature_dim=feature_dim,
            n_classes=n_classes,
            n_qubits=args.n_qubits,
            n_qlayers=args.q_depth,
            encoding=config['encoding'],
            ansatz=config['ansatz'],
            q_type=config.get('q_type', 'standard'),
            standard_dim=config.get('standard_dim', None),
            quantum_weights_path=config.get('quantum_weights_path', None),
            freeze_quantum=config.get('freeze_quantum', False),
            multistage=config.get('multistage', False)
        )
    else:
        # Classical
        model = backbone
        # Replace head for classification
        if 'resnet' in config['backbone']:
             model.fc = nn.Linear(feature_dim, n_classes)
        elif 'vit' in config['backbone']:
             model.head = nn.Linear(feature_dim, n_classes)

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loop
    results = {'loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [], 'epoch_times': []}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - start_time
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        auc_str = f" - AUC: {val_metrics['roc_auc']:.4f}" if 'roc_auc' in val_metrics else ""
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Accuracy: {val_metrics['accuracy']:.4f} - F1: {val_metrics['f1_macro']:.4f}{auc_str} - Time: {epoch_time:.2f}s")
        
        results['loss'].append(loss)
        results['val_acc'].append(val_metrics['accuracy'])
        results['val_f1'].append(val_metrics['f1_macro'])
        if 'roc_auc' in val_metrics:
            results['val_auc'].append(val_metrics['roc_auc'])
        results['epoch_times'].append(epoch_time)
        
    # Save results
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    with open(results_path / f"{config_name}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save model checkpoint if requested
    if config.get('save_model', False):
        checkpoint_path = Path('checkpoints')
        checkpoint_path.mkdir(exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'results': results,
            'args': vars(args)
        }
        torch.save(checkpoint, checkpoint_path / f"{config_name}.pth")
        print(f"Model checkpoint saved to checkpoints/{config_name}.pth")
        
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all')
    parser.add_argument('--dataset', type=str, default='EuroSAT', help='Dataset to load (EuroSAT, SIRI-WHU, UC_M_LUC)')
    parser.add_argument('--env', type=str, default='auto', help='Environment context (auto, local, kaggle, guacamole)')
    parser.add_argument('--data_root', type=str, default='./data', help='Used only if env is not auto')
    parser.add_argument('--subset_fraction', type=float, default=1.0, help='Fraction of dataset for quick training')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--q_depth', type=int, default=1, help='Number of quantum layers (depth)')
    parser.add_argument('--bands', type=str, default='RGB', choices=['RGB', 'ALL'])
    parser.add_argument('--encoding', type=str, default=None, choices=['angle', 'amplitude', 'iqp', 'molecular'], help='Override quantum encoding type')
    parser.add_argument('--ansatz', type=str, default=None, choices=['vqc', 'basic', 'hardware_efficient', 'qaoa', 'pqc'], help='Override quantum ansatz type')
    
    args = parser.parse_args()
    
    if args.config == 'all':
        for conf in CONFIGS:
            run_experiment(conf, args)
    else:
        run_experiment(args.config, args)
