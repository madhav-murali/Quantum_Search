#!/usr/bin/env python3
"""
Quantum Transfer Learning — LeNet5 + Amplitude Encoding
========================================================

Transfers pre-trained quantum circuit weights from a ResNet50+Amplitude source
model to a lightweight LeNet5 backbone, targeting ~94% accuracy on EuroSAT
with a ~600× parameter reduction.

Pipeline:
  Phase 0 — Verify source model accuracy on one test epoch
  Phase 1 — Load pre-trained ResNet50+Quantum source model & extract weights
  Phase 2 — Run transfer strategies (frozen / finetuned / distilled / scratch)
  Phase 3 — Compare results

Source model config (must match checkpoint):
  backbone=resnet50, encoding=amplitude, ansatz=vqc, q_type=standard,
  n_qubits=8, q_depth=2, standard_dim=256, n_classes=10

Usage:
  # Quick smoke test (~5 min)
  python qtl/lenet5_amplitude_qtl.py --epochs 2 --batch_size 16

  # Full benchmark (~2 hours, aim for ~94% accuracy)
  python qtl/lenet5_amplitude_qtl.py --epochs 30

  # Retrain source model if checkpoint missing
  python qtl/lenet5_amplitude_qtl.py --epochs 30 --retrain_source --source_epochs 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports — reuse existing models directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataset, SpectralSelector
from src.data.splitter import EuroSATSplitter
from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel
from src.models.qtl_model import QTLModel
from src.utils.metrics import calculate_metrics
from torchvision.transforms import Compose


# ===========================================================================
# Configuration
# ===========================================================================

# Source model config — must match the saved checkpoint exactly.
# This is the ResNet50 + Amplitude encoding model that achieved ~96% accuracy.
SOURCE_CONFIG = {
    'backbone': 'resnet50',
    'encoding': 'amplitude',
    'ansatz': 'vqc',
    'q_type': 'standard',
    'n_qubits': 8,
    'q_depth': 2,
    'standard_dim': 256,
    'n_classes': 10,
    'in_channels': 3,
}

# Target model config — LeNet5 backbone, same quantum layer configuration.
TARGET_CONFIG = {
    'backbone': 'lenet5',
    'encoding': 'amplitude',
    'ansatz': 'vqc',
    'q_type': 'standard',
    'n_qubits': 8,
    'q_depth': 2,
    'standard_dim': 256,
    'n_classes': 10,
    'in_channels': 3,
}

# Checkpoint paths
SOURCE_CKPT_PATH = 'checkpoints/qtl_source_resnet_amplitude.pth'
QUANTUM_WEIGHTS_PATH = 'checkpoints/qtl_source_quantum_weights.pth'


# ===========================================================================
# Model builders
# ===========================================================================

def build_source_model(device='cpu'):
    """
    Build the ResNet50+Quantum source model matching the saved checkpoint.

    Uses BackboneFactory and HybridGeoModel from src/models/.
    """
    cfg = SOURCE_CONFIG
    backbone, feature_dim = BackboneFactory.create(
        cfg['backbone'], pretrained=True, in_channels=cfg['in_channels']
    )
    model = HybridGeoModel(
        backbone=backbone,
        feature_dim=feature_dim,
        n_classes=cfg['n_classes'],
        n_qubits=cfg['n_qubits'],
        n_qlayers=cfg['q_depth'],
        encoding=cfg['encoding'],
        ansatz=cfg['ansatz'],
        q_type=cfg['q_type'],
        standard_dim=cfg['standard_dim'],
    )
    return model.to(device)


def build_target_model(quantum_weights_path=None, freeze_quantum=False,
                       device='cpu'):
    """
    Build the LeNet5+Quantum target model using the same quantum config.

    The quantum layer has identical architecture (8 qubits, depth 2, amplitude,
    VQC ansatz) so pre-trained weights transfer directly.
    """
    cfg = TARGET_CONFIG
    backbone, feature_dim = BackboneFactory.create(
        cfg['backbone'], pretrained=False, in_channels=cfg['in_channels']
    )
    model = HybridGeoModel(
        backbone=backbone,
        feature_dim=feature_dim,
        n_classes=cfg['n_classes'],
        n_qubits=cfg['n_qubits'],
        n_qlayers=cfg['q_depth'],
        encoding=cfg['encoding'],
        ansatz=cfg['ansatz'],
        q_type=cfg['q_type'],
        standard_dim=cfg['standard_dim'],
        quantum_weights_path=quantum_weights_path,
        freeze_quantum=freeze_quantum,
    )
    return model.to(device)


# ===========================================================================
# Data loading
# ===========================================================================

def get_data_loaders(data_root, batch_size, bands='RGB'):
    """Load EuroSAT and return train/val/test DataLoaders."""
    full_dataset = get_dataset(root=data_root, download=True)
    selector = SpectralSelector(mode=bands)
    full_dataset.transforms = selector
    splitter = EuroSATSplitter(full_dataset, test_size=0.2, val_size=0.1)
    return splitter.get_loaders(batch_size=batch_size)


# ===========================================================================
# Training & evaluation
# ===========================================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    qtl_wrapper=None):
    """
    Train for one epoch.

    If ``qtl_wrapper`` has a teacher, uses knowledge distillation loss.
    Otherwise uses standard cross-entropy.
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        images = batch['image'].to(device).float() if isinstance(batch, dict) \
            else batch[0].to(device).float()
        targets = batch['label'].to(device).long() if isinstance(batch, dict) \
            else batch[1].to(device).long()

        optimizer.zero_grad()

        if qtl_wrapper is not None and qtl_wrapper.teacher is not None:
            loss, _ = qtl_wrapper.distillation_loss(images, targets, criterion)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return a metrics dict (accuracy, f1_macro, etc.)."""
    model.eval()
    running_loss = 0.0
    all_targets, all_preds = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        images = batch['image'].to(device).float() if isinstance(batch, dict) \
            else batch[0].to(device).float()
        targets = batch['label'].to(device).long() if isinstance(batch, dict) \
            else batch[1].to(device).long()

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    metrics = calculate_metrics(all_targets, all_preds, None)
    metrics['val_loss'] = running_loss / len(loader.dataset)
    return metrics


# ===========================================================================
# Phase 0 — Verify source model accuracy
# ===========================================================================

def verify_source_model(model, val_loader, test_loader, device):
    """
    Run one evaluation pass on val and test sets to confirm the loaded
    source model performs as expected before starting QTL transfer.

    Returns:
        dict: {'val_acc', 'val_f1', 'test_acc', 'test_f1'}
    """
    print("\n" + "=" * 70)
    print("  PHASE 0: Verifying Source Model Accuracy")
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()

    print("  Evaluating on validation set...")
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"    Val  accuracy: {val_metrics['accuracy']:.4f} | "
          f"F1: {val_metrics['f1_macro']:.4f}")

    print("  Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"    Test accuracy: {test_metrics['accuracy']:.4f} | "
          f"F1: {test_metrics['f1_macro']:.4f}")

    result = {
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['f1_macro'],
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_macro'],
    }

    # Warn if source accuracy is unexpectedly low
    if val_metrics['accuracy'] < 0.90:
        print(f"\n  ⚠ Source model accuracy ({val_metrics['accuracy']:.4f}) is "
              f"below 90%. Consider retraining with --retrain_source --source_epochs 20")

    print(f"\n  ✓ Source model verified: {val_metrics['accuracy']:.4f} val / "
          f"{test_metrics['accuracy']:.4f} test")
    return result


# ===========================================================================
# Phase 1 — Load (or retrain) source model
# ===========================================================================

def load_source_model(device):
    """
    Load the existing pre-trained ResNet50+Quantum source model.

    Returns:
        Tuple[nn.Module | None, str | None]: (model, quantum_weights_path)
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Loading Pre-trained Source Model")
    print("=" * 70)

    if not os.path.exists(SOURCE_CKPT_PATH):
        print(f"  ✗ Checkpoint not found: {SOURCE_CKPT_PATH}")
        print(f"    Run with --retrain_source to train from scratch.")
        return None, None

    model = build_source_model(device)

    ckpt = torch.load(SOURCE_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    saved_config = ckpt.get('config', {})
    print(f"  ✓ Loaded source model from {SOURCE_CKPT_PATH}")
    print(f"    Config: {saved_config}")
    print(f"    Parameters: {total:,}")

    # Extract quantum weights if not already saved
    qw_path = QUANTUM_WEIGHTS_PATH
    if not os.path.exists(qw_path):
        print(f"  Extracting quantum weights → {qw_path}")
        Path(qw_path).parent.mkdir(exist_ok=True)
        model.quantum_layer.save_quantum_weights(qw_path)
    else:
        print(f"  ✓ Quantum weights already at {qw_path}")

    return model, qw_path


def retrain_source(args, train_loader, val_loader, device):
    """
    Retrain ResNet50+Quantum source model from scratch.

    Uses the original best config: resnet50 + amplitude + depth 2 + vqc
    with ~20 epochs to achieve ≥94% accuracy.

    Returns:
        Tuple[nn.Module, str]: (model, quantum_weights_path)
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Retraining Source Model (ResNet50 + Quantum)")
    print("=" * 70)

    model = build_source_model(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Source model parameters: {total:,}")
    print(f"  Config: {SOURCE_CONFIG}")
    print(f"  Training for {args.source_epochs} epochs...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.source_epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        acc = metrics['accuracy']
        f1 = metrics['f1_macro']
        print(f"  Epoch {epoch+1}/{args.source_epochs}  "
              f"loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}  "
              f"time={elapsed:.1f}s")

        if acc > best_acc:
            best_acc = acc

    # Save checkpoint
    ckpt_dir = Path('checkpoints')
    ckpt_dir.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': SOURCE_CONFIG,
        'best_acc': best_acc,
    }, SOURCE_CKPT_PATH)
    print(f"  Source model saved → {SOURCE_CKPT_PATH}")

    # Save quantum weights
    qw_path = QUANTUM_WEIGHTS_PATH
    model.quantum_layer.save_quantum_weights(qw_path)
    print(f"  Quantum weights → {qw_path}")
    print(f"  Best source accuracy: {best_acc:.4f}")

    return model, qw_path


# ===========================================================================
# Phase 2 — Transfer strategies
# ===========================================================================

def run_transfer_strategy(strategy, args, train_loader, val_loader, device,
                          qw_path, teacher_model=None):
    """
    Run a single transfer strategy and return training history.

    Strategies:
      frozen     — load quantum weights, freeze them entirely
      finetuned  — load quantum weights, progressive unfreeze at epoch 5
      distilled  — load quantum weights + knowledge distillation from teacher
      scratch    — no transfer, train everything from random init (control)
    """
    print(f"\n{'─'*70}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"{'─'*70}")

    # Build student model
    load_qw = qw_path if strategy != 'scratch' else None
    freeze_q = (strategy == 'frozen')

    student = build_target_model(
        quantum_weights_path=load_qw,
        freeze_quantum=freeze_q,
        device=device,
    )

    # Wrap in QTLModel (handles distillation + progressive unfreezing)
    teacher = teacher_model if strategy == 'distilled' else None
    qtl = QTLModel(
        student=student,
        teacher=teacher,
        temperature=4.0,
        alpha=0.3,
        unfreeze_epoch=5,
    )

    # For finetuned strategy: start frozen, unfreeze at epoch 5
    if strategy == 'finetuned':
        qtl.freeze_quantum()

    # Print parameter summary
    param_counts = qtl.print_parameter_summary(label=strategy)

    # Optimizer with differential learning rates for transfer strategies
    if strategy in ('finetuned', 'distilled'):
        param_groups = qtl.get_param_groups(
            backbone_lr=args.lr * 10,    # 1e-3
            quantum_lr=args.lr * 0.1,    # 1e-5
            classifier_lr=args.lr * 10,  # 1e-3
        )
    else:
        param_groups = [
            {'params': [p for p in student.parameters() if p.requires_grad],
             'lr': args.lr}
        ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        'loss': [], 'val_acc': [], 'val_f1': [], 'epoch_times': [],
        'strategy': strategy,
        'parameters': param_counts['_total'],
    }

    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Progressive unfreezing for finetuned / distilled
        if strategy in ('finetuned', 'distilled'):
            if qtl.maybe_unfreeze(epoch):
                print(f"    ↳ Quantum layer UNFROZEN at epoch {epoch+1}")
                # Rebuild optimizer to include newly unfrozen params
                param_groups = qtl.get_param_groups(
                    backbone_lr=args.lr * 10,
                    quantum_lr=args.lr * 0.1,
                    classifier_lr=args.lr * 10,
                )
                optimizer = optim.Adam(param_groups)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        loss = train_one_epoch(student, train_loader, criterion, optimizer,
                               device, qtl_wrapper=qtl)
        metrics = evaluate(student, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        acc = metrics['accuracy']
        f1 = metrics['f1_macro']
        best_acc = max(best_acc, acc)
        best_f1 = max(best_f1, f1)

        history['loss'].append(loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['epoch_times'].append(elapsed)

        print(f"  Epoch {epoch+1}/{args.epochs}  "
              f"loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}  "
              f"time={elapsed:.1f}s")

    history['best_acc'] = best_acc
    history['best_f1'] = best_f1
    history['avg_epoch_time'] = float(np.mean(history['epoch_times']))

    print(f"\n  ✓ {strategy.upper()} — Best Acc: {best_acc:.4f}, "
          f"Best F1: {best_f1:.4f}")
    return history


# ===========================================================================
# Phase 3 — Results summary
# ===========================================================================

def print_summary(results, source_metrics=None):
    """Print a comparison table of all strategies."""
    print("\n" + "=" * 80)
    print("  QTL LENET5 + AMPLITUDE — RESULTS SUMMARY")
    print("=" * 80)

    if source_metrics is not None:
        print(f"  Source (ResNet50+Quantum): "
              f"{source_metrics['val_acc']:.4f} val / "
              f"{source_metrics['test_acc']:.4f} test | ~25.6M params")
        print()

    print(f"  {'Strategy':<15} {'Params':>10} {'Trainable':>10} "
          f"{'Best Acc':>10} {'Best F1':>10} {'Avg Time':>10}")
    print(f"  {'─'*65}")

    for name, h in results.items():
        params = h['parameters']
        print(f"  {name:<15} {params['total']:>10,} {params['trainable']:>10,} "
              f"{h['best_acc']:>10.4f} {h['best_f1']:>10.4f} "
              f"{h['avg_epoch_time']:>9.1f}s")

    # Transfer benefit vs scratch baseline
    if 'scratch' in results:
        for name in ('distilled', 'finetuned', 'frozen'):
            if name in results:
                benefit = results[name]['best_acc'] - results['scratch']['best_acc']
                print(f"\n  📊 Transfer benefit ({name} vs scratch): "
                      f"{benefit:+.2%}")

    if source_metrics and results:
        best_strategy = max(results.items(), key=lambda x: x[1]['best_acc'])
        param_reduction = 25_600_000 / best_strategy[1]['parameters']['total']
        print(f"\n  🔬 Parameter reduction: {param_reduction:.0f}× fewer parameters")
        print(f"  🎯 Best transfer ({best_strategy[0]}): "
              f"{best_strategy[1]['best_acc']:.4f} vs source "
              f"{source_metrics['val_acc']:.4f}")

    print("=" * 80)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QTL: LeNet5 + Amplitude Encoding — Parameter Reduction Benchmark')
    parser.add_argument('--data_root', type=str, default='./data/EuroSAT')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs per transfer strategy')
    parser.add_argument('--source_epochs', type=int, default=20,
                        help='Epochs for source retraining (only with --retrain_source)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bands', type=str, default='RGB',
                        choices=['RGB', 'ALL'])
    parser.add_argument('--retrain_source', action='store_true',
                        help='Retrain source model from scratch instead of loading')
    parser.add_argument('--strategies', nargs='+',
                        default=['frozen', 'finetuned', 'distilled', 'scratch'],
                        help='Which strategies to run')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"  QTL: LeNet5 + Amplitude Encoding")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"  Quantum: {SOURCE_CONFIG['n_qubits']} qubits, depth "
          f"{SOURCE_CONFIG['q_depth']}, {SOURCE_CONFIG['encoding']} encoding, "
          f"standard_dim={SOURCE_CONFIG['standard_dim']}")
    print(f"  Strategies: {args.strategies}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading EuroSAT data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_root, args.batch_size, args.bands
    )
    print(f"  Train: {len(train_loader)} batches | "
          f"Val: {len(val_loader)} batches | "
          f"Test: {len(test_loader)} batches")

    # ------------------------------------------------------------------
    # Phase 1 — Load or retrain source
    # ------------------------------------------------------------------
    if args.retrain_source:
        teacher_model, qw_path = retrain_source(
            args, train_loader, val_loader, device
        )
    else:
        teacher_model, qw_path = load_source_model(device)

    # ------------------------------------------------------------------
    # Phase 0 — Verify source accuracy before QTL
    # ------------------------------------------------------------------
    source_metrics = None
    if teacher_model is not None:
        source_metrics = verify_source_model(
            teacher_model, val_loader, test_loader, device
        )
    else:
        print("\n  ✗ No source model available.")
        print("    Run with --retrain_source or place checkpoint at:")
        print(f"    {SOURCE_CKPT_PATH}")
        if 'distilled' in args.strategies:
            args.strategies = [s for s in args.strategies if s != 'distilled']
            print("    Skipping 'distilled' (no teacher)")

    # Handle missing quantum weights
    if qw_path is None or not os.path.exists(qw_path or ''):
        qw_path = None
        transfer_strategies = [s for s in args.strategies if s == 'scratch']
        skipped = [s for s in args.strategies if s != 'scratch']
        if skipped:
            print(f"  ⚠ Skipping {skipped} (no quantum weights)")
        args.strategies = transfer_strategies

    # Handle missing teacher for distilled
    if teacher_model is None and 'distilled' in args.strategies:
        args.strategies = [s for s in args.strategies if s != 'distilled']
        print("  ⚠ Skipping distilled (no teacher model)")

    # ------------------------------------------------------------------
    # Phase 2 — Transfer strategies
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2: Transfer Strategies (LeNet5 + Quantum)")
    print("=" * 70)

    results = {}
    for strategy in args.strategies:
        history = run_transfer_strategy(
            strategy, args, train_loader, val_loader, device,
            qw_path=qw_path, teacher_model=teacher_model,
        )
        results[strategy] = history

        # Save per-strategy results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"qtl_lenet5_amplitude_{strategy}_results.json"
        with open(out_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"  Results saved → {out_file}")

    # ------------------------------------------------------------------
    # Phase 3 — Summary
    # ------------------------------------------------------------------
    if results:
        print_summary(results, source_metrics=source_metrics)

        # Save combined summary
        out_dir = Path(args.output_dir)
        summary = {
            'source_metrics': source_metrics,
            'source_config': SOURCE_CONFIG,
            'target_config': TARGET_CONFIG,
            'strategies': {
                name: {
                    'best_acc': h['best_acc'],
                    'best_f1': h['best_f1'],
                    'total_params': h['parameters']['total'],
                    'trainable_params': h['parameters']['trainable'],
                    'avg_epoch_time': h['avg_epoch_time'],
                }
                for name, h in results.items()
            }
        }
        summary_file = out_dir / 'qtl_lenet5_amplitude_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary → {summary_file}")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
