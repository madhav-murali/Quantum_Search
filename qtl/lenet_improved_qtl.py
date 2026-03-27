#!/usr/bin/env python3
"""
Quantum Transfer Learning — LeNet Improved + Extended Weight Transfer
=====================================================================

Improvements over lenet5_amplitude_qtl.py:
  1. Uses LeNetCNNImproved backbone (3-conv, BatchNorm, 8192-dim features)
     instead of LeNet5Quantum (2-conv, 84-dim features) for richer representations.
  2. Transfers MORE than just quantum weights — also transfers classifier weights
     and projector bias from the source ResNet50 model.
  3. Adds a feature-alignment distillation loss that matches projected features
     (both 256-dim via standard_dim) between teacher and student.

Weight transfer breakdown (source → target):
  ✓ quantum_layer weights  (2, 8, 3) = 48 params   — same shape
  ✓ classifier weights     (10, 8)   = 90 params   — same shape
  ✓ projector bias         (256,)    = 256 params   — same shape
  ✗ projector weights      (256, 2048) vs (256, 8192) — different, can't copy
  → Total transferred: 394 params (vs 48 in the original script)

Usage:
  # Quick smoke test
  python qtl/lenet_improved_qtl.py --epochs 2 --batch_size 16

  # Full benchmark (aim for ~94%)
  python qtl/lenet_improved_qtl.py --epochs 100

  # Retrain source if checkpoint missing
  python qtl/lenet_improved_qtl.py --epochs 100 --retrain_source --source_epochs 20
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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm
import torchvision.transforms as T

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
from src.utils.env_config import EnvironmentManager


# ===========================================================================
# Configuration
# ===========================================================================

# Source model config — must match the saved checkpoint exactly.
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

# Target model config — LeNet Improved backbone, same quantum config.
TARGET_CONFIG = {
    'backbone': 'lenet_improved',
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
EXTENDED_WEIGHTS_PATH = 'checkpoints/qtl_source_extended_weights.pth'


# ===========================================================================
# Extended weight save/load (quantum + classifier + projector bias)
# ===========================================================================

def save_extended_weights(model, path):
    """
    Save quantum weights + classifier weights + projector bias from source.

    These components have identical shapes between ResNet50 and LeNet Improved
    targets (thanks to standard_dim=256 and matching quantum/classifier config).
    """
    checkpoint = {
        # Quantum layer weights
        'quantum_weights': model.quantum_layer.qlayer.weights.data,
        'quantum_config': {
            'n_qubits': model.quantum_layer.n_qubits,
            'n_layers': model.quantum_layer.n_layers,
            'encoding': model.quantum_layer.encoding,
            'ansatz': model.quantum_layer.ansatz,
        },
        # Classifier weights (Linear(8, 10))
        'classifier_weight': model.classifier.weight.data,
        'classifier_bias': model.classifier.bias.data,
        # Projector bias (256,) — same shape regardless of backbone
        'projector_bias': model.projector.bias.data,
    }

    Path(path).parent.mkdir(exist_ok=True)
    torch.save(checkpoint, path)

    total_params = (
        checkpoint['quantum_weights'].numel() +
        checkpoint['classifier_weight'].numel() +
        checkpoint['classifier_bias'].numel() +
        checkpoint['projector_bias'].numel()
    )
    print(f"  Extended weights saved → {path} ({total_params} params)")


def load_extended_weights(model, path, freeze_transferred=False):
    """
    Load quantum + classifier + projector bias into the target model.

    Args:
        model: HybridGeoModel target (LeNet Improved backbone)
        path: Path to extended weights checkpoint
        freeze_transferred: If True, freeze the transferred components
    """
    # Detect model device so loaded tensors land on the correct device
    device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device)

    # Load quantum weights
    model.quantum_layer.qlayer.weights.data = ckpt['quantum_weights'].to(device)
    print(f"    ✓ Quantum weights loaded ({ckpt['quantum_weights'].numel()} params)")

    # Load classifier weights
    model.classifier.weight.data = ckpt['classifier_weight'].to(device)
    model.classifier.bias.data = ckpt['classifier_bias'].to(device)
    print(f"    ✓ Classifier weights loaded "
          f"({ckpt['classifier_weight'].numel() + ckpt['classifier_bias'].numel()} params)")

    # Load projector bias
    model.projector.bias.data = ckpt['projector_bias'].to(device)
    print(f"    ✓ Projector bias loaded ({ckpt['projector_bias'].numel()} params)")

    if freeze_transferred:
        # Freeze quantum
        for p in model.quantum_layer.parameters():
            p.requires_grad = False
        # Freeze classifier
        for p in model.classifier.parameters():
            p.requires_grad = False
        # Note: projector bias is part of .projector so we don't freeze the
        # whole projector (its weight matrix is random, needs training)
        print("    ✓ Quantum + classifier parameters frozen")


# ===========================================================================
# Model builders
# ===========================================================================

def build_source_model(device='cpu'):
    """Build the ResNet50+Quantum source model matching the saved checkpoint."""
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


def build_target_model(device='cpu'):
    """
    Build the LeNet Improved + Quantum target model.

    Uses LeNetCNNImproved (3-conv, BatchNorm, 8192-dim features) which is
    much richer than LeNet5Quantum (84-dim features).
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
    )
    return model.to(device)


# ===========================================================================
# Data loading
# ===========================================================================

def get_data_loaders(data_root, batch_size, bands='RGB', dataset_name='EuroSAT', subset_fraction=1.0):
    """Load Dataset and return train/val/test DataLoaders."""
    full_dataset = get_dataset(root=data_root, dataset_name=dataset_name, download=True, subset_fraction=subset_fraction)
    selector = SpectralSelector(mode=bands)
    
    if dataset_name.lower() == 'eurosat':
         full_dataset.transforms = selector
    
    splitter = EuroSATSplitter(full_dataset, test_size=0.2, val_size=0.1)
    return splitter.get_loaders(batch_size=batch_size)


# ===========================================================================
# Feature alignment loss (for distillation at projector output level)
# ===========================================================================

class FeatureAlignedQTL(nn.Module):
    """
    Extended QTL wrapper that adds feature-alignment distillation.

    In addition to the standard KL-divergence on logits (from QTLModel),
    this adds an MSE loss between the 256-dim projected features of the
    teacher and student. This forces the student's projector to learn a
    similar feature space to the teacher's, even though their backbone
    dimensions differ (2048 vs 8192).

    Loss = alpha * KD_logit_loss + beta * feature_align_loss + (1-alpha-beta) * CE_loss
    """

    def __init__(self, student, teacher, temperature=4.0,
                 alpha=0.3, beta=0.2, unfreeze_epoch=5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha      # weight for logit distillation
        self.beta = beta        # weight for feature alignment
        self.unfreeze_epoch = unfreeze_epoch
        self._quantum_frozen = False
        self._classifier_frozen = False

        # Freeze teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def forward(self, x):
        """Standard forward — student logits only."""
        return self.student(x)

    def _get_projected_features(self, model, x):
        """
        Run through backbone + projector only (skip quantum + classifier).
        Returns the 256-dim projected features.
        """
        features = model.backbone(x)
        projected = model.projector(features)
        return projected

    def distillation_loss(self, x, hard_targets, criterion):
        """
        Combined loss: hard CE + logit KD + feature alignment.

        Returns:
            Tuple[Tensor, Tensor]: (total_loss, student_logits)
        """
        # Student forward
        student_logits = self.student(x)
        hard_loss = criterion(student_logits, hard_targets)

        if self.teacher is None:
            return hard_loss, student_logits

        with torch.no_grad():
            teacher_logits = self.teacher(x)
            teacher_features = self._get_projected_features(self.teacher, x)

        # Logit distillation (KL divergence with temperature)
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher,
                           reduction='batchmean') * (T * T)

        # Feature alignment (MSE on projected features)
        student_features = self._get_projected_features(self.student, x)
        feature_loss = F.mse_loss(student_features, teacher_features)

        # Combined loss
        total_loss = (self.alpha * kd_loss +
                      self.beta * feature_loss +
                      (1 - self.alpha - self.beta) * hard_loss)

        return total_loss, student_logits

    def freeze_transferred(self):
        """Freeze quantum + classifier (the transferred components)."""
        if hasattr(self.student, 'quantum_layer'):
            for p in self.student.quantum_layer.parameters():
                p.requires_grad = False
            self._quantum_frozen = True
        for p in self.student.classifier.parameters():
            p.requires_grad = False
        self._classifier_frozen = True

    def unfreeze_transferred(self):
        """Unfreeze quantum + classifier."""
        if hasattr(self.student, 'quantum_layer'):
            for p in self.student.quantum_layer.parameters():
                p.requires_grad = True
            self._quantum_frozen = False
        for p in self.student.classifier.parameters():
            p.requires_grad = True
        self._classifier_frozen = False

    def maybe_unfreeze(self, epoch):
        """Unfreeze at the configured epoch."""
        if (self._quantum_frozen or self._classifier_frozen) and epoch >= self.unfreeze_epoch:
            self.unfreeze_transferred()
            return True
        return False

    def get_param_groups(self, backbone_lr=1e-3, quantum_lr=1e-5,
                         classifier_lr=1e-4):
        """
        Build parameter groups with differential learning rates.

        Transferred components (quantum, classifier) get lower LRs to
        preserve the transferred knowledge.
        """
        groups = []

        # Backbone (highest LR — needs to learn from scratch)
        backbone_params = list(self.student.backbone.parameters())
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': backbone_lr,
                           'name': 'backbone'})

        # Projector (moderate LR — bias is transferred, weights are random)
        projector_params = list(self.student.projector.parameters())
        if projector_params:
            groups.append({'params': projector_params, 'lr': backbone_lr,
                           'name': 'projector'})

        # Quantum adapter
        if (hasattr(self.student, 'quantum_adapter') and
                not isinstance(self.student.quantum_adapter, nn.Identity)):
            adapter_params = list(self.student.quantum_adapter.parameters())
            if adapter_params:
                groups.append({'params': adapter_params, 'lr': quantum_lr,
                               'name': 'quantum_adapter'})

        # Quantum layer (low LR — transferred)
        if (hasattr(self.student, 'quantum_layer') and
                not isinstance(self.student.quantum_layer, nn.Identity)):
            q_params = list(self.student.quantum_layer.parameters())
            if q_params:
                groups.append({'params': q_params, 'lr': quantum_lr,
                               'name': 'quantum_layer'})

        # Classifier (low LR — transferred)
        classifier_params = list(self.student.classifier.parameters())
        if classifier_params:
            groups.append({'params': classifier_params, 'lr': classifier_lr,
                           'name': 'classifier'})

        return groups

    def count_parameters(self):
        """Count parameters by component."""
        report = {}
        components = {
            'backbone': self.student.backbone,
            'projector': self.student.projector,
            'classifier': self.student.classifier,
        }
        if (hasattr(self.student, 'quantum_adapter') and
                not isinstance(self.student.quantum_adapter, nn.Identity)):
            components['quantum_adapter'] = self.student.quantum_adapter
        if (hasattr(self.student, 'quantum_layer') and
                not isinstance(self.student.quantum_layer, nn.Identity)):
            components['quantum_layer'] = self.student.quantum_layer

        grand_total, grand_trainable = 0, 0
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters()
                            if p.requires_grad)
            report[name] = {'total': total, 'trainable': trainable}
            grand_total += total
            grand_trainable += trainable

        report['_total'] = {'total': grand_total, 'trainable': grand_trainable}
        return report

    def print_parameter_summary(self, label=''):
        """Pretty-print parameter counts."""
        counts = self.count_parameters()
        header = f"Parameter Summary{f' — {label}' if label else ''}"
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")
        print(f"  {'Component':<25} {'Total':>10} {'Trainable':>10}")
        print(f"  {'-'*45}")
        for name, c in counts.items():
            if name.startswith('_'):
                continue
            print(f"  {name:<25} {c['total']:>10,} {c['trainable']:>10,}")
        totals = counts['_total']
        print(f"  {'-'*45}")
        print(f"  {'TOTAL':<25} {totals['total']:>10,} {totals['trainable']:>10,}")
        print(f"{'='*60}\n")
        return counts


# ===========================================================================
# Training & evaluation
# ===========================================================================

# ---------------------------------------------------------------------------
# Training augmentation — SAFE for arbitrary value ranges
# ---------------------------------------------------------------------------
# T.ColorJitter and T.RandomErasing assume [0,1] and DESTROY these values.
# Hence i havent used them
# ---------------------------------------------------------------------------
_train_augment = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
])


def train_one_epoch(model, loader, criterion, optimizer, device,
                    qtl_wrapper=None, augment=True):
    """
    Train for one epoch with optional data augmentation.

    If qtl_wrapper is provided and has a teacher, uses the wrapper's
    distillation_loss (which may include feature alignment).
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        images = (batch['image'].to(device).float() if isinstance(batch, dict)
                  else batch[0].to(device).float())
        targets = (batch['label'].to(device).long() if isinstance(batch, dict)
                   else batch[1].to(device).long())

        # Apply augmentation on GPU (train only)
        if augment:
            images = _train_augment(images)

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
    """Evaluate model and return metrics dict."""
    model.eval()
    running_loss = 0.0
    all_targets, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        images = (batch['image'].to(device).float() if isinstance(batch, dict)
                  else batch[0].to(device).float())
        targets = (batch['label'].to(device).long() if isinstance(batch, dict)
                   else batch[1].to(device).long())

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics['val_loss'] = running_loss / len(loader.dataset)
    return metrics


# ===========================================================================
# Phase 0 — Verify source model accuracy
# ===========================================================================

def verify_source_model(model, val_loader, test_loader, device):
    """Run evaluation on val + test to confirm source model performance."""
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

    if val_metrics['accuracy'] < 0.90:
        print(f"\n  ⚠ Source accuracy ({val_metrics['accuracy']:.4f}) below 90%."
              f" Consider --retrain_source --source_epochs 20")

    print(f"\n  ✓ Source verified: {val_metrics['accuracy']:.4f} val / "
          f"{test_metrics['accuracy']:.4f} test")
    return result


# ===========================================================================
# Phase 1 — Load (or retrain) source model
# ===========================================================================

def load_source_model(device):
    """Load pre-trained ResNet50+Quantum and extract extended weights."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Loading Pre-trained Source Model")
    print("=" * 70)

    if not os.path.exists(SOURCE_CKPT_PATH):
        print(f"  ✗ Checkpoint not found: {SOURCE_CKPT_PATH}")
        return None, None

    model = build_source_model(device)
    ckpt = torch.load(SOURCE_CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Loaded source model from {SOURCE_CKPT_PATH}")
    print(f"    Parameters: {total:,}")

    # Extract quantum-only weights (for backward compatibility)
    if not os.path.exists(QUANTUM_WEIGHTS_PATH):
        model.quantum_layer.save_quantum_weights(QUANTUM_WEIGHTS_PATH)

    # Extract extended weights (quantum + classifier + projector bias)
    if not os.path.exists(EXTENDED_WEIGHTS_PATH):
        print(f"  Extracting extended weights...")
        save_extended_weights(model, EXTENDED_WEIGHTS_PATH)
    else:
        print(f"  ✓ Extended weights already at {EXTENDED_WEIGHTS_PATH}")

    return model, EXTENDED_WEIGHTS_PATH


def retrain_source(args, train_loader, val_loader, device):
    """Retrain source from scratch with resnet50 + amplitude + depth 2 + vqc."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Retraining Source Model (ResNet50 + Quantum)")
    print("=" * 70)

    model = build_source_model(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Source parameters: {total:,}")
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
    print(f"  Source saved → {SOURCE_CKPT_PATH}")

    # Save extended weights
    save_extended_weights(model, EXTENDED_WEIGHTS_PATH)

    # Save quantum-only weights (backward compat)
    model.quantum_layer.save_quantum_weights(QUANTUM_WEIGHTS_PATH)

    print(f"  Best source accuracy: {best_acc:.4f}")
    return model, EXTENDED_WEIGHTS_PATH


# ===========================================================================
# Phase 2 — Transfer strategies
# ===========================================================================

def run_transfer_strategy(strategy, args, train_loader, val_loader, device,
                          ext_weights_path, teacher_model=None,
                          augment=True):
    """
    Run a single transfer strategy with LeNet Improved backbone.

    Strategies:
      frozen      — load extended weights (quantum+classifier+proj_bias), freeze
      finetuned   — load extended weights, progressive unfreeze at epoch 5
      distilled   — load extended weights + feature-aligned KD from teacher
      scratch     — no transfer, train from random init (control)
    """
    print(f"\n{'─'*70}")
    print(f"  Strategy: {strategy.upper()}")
    print(f"{'─'*70}")

    # Build student
    student = build_target_model(device=device)

    # Load extended weights for transfer strategies
    if strategy != 'scratch' and ext_weights_path is not None:
        print(f"  Loading extended weights from {ext_weights_path}:")
        freeze = (strategy == 'frozen')
        load_extended_weights(student, ext_weights_path,
                              freeze_transferred=freeze)
    elif strategy != 'scratch':
        print(f"  ⚠ No extended weights available, running as scratch")
        strategy = 'scratch'

    # Build QTL wrapper
    if strategy == 'distilled' and teacher_model is not None:
        # Use FeatureAlignedQTL for feature-level + logit-level distillation
        # T=2.0 gives sharper soft targets for better knowledge transfer
        qtl = FeatureAlignedQTL(
            student=student,
            teacher=teacher_model,
            temperature=2.0,
            alpha=0.3,    # logit KD weight
            beta=0.2,     # feature alignment weight
            unfreeze_epoch=5,
        )
    else:
        # Use FeatureAlignedQTL without teacher (acts as regular wrapper)
        qtl = FeatureAlignedQTL(
            student=student,
            teacher=None,
            temperature=2.0,
            unfreeze_epoch=5,
        )

    # For finetuned: start with transferred components frozen
    if strategy == 'finetuned':
        qtl.freeze_transferred()

    # Print parameter summary
    param_counts = qtl.print_parameter_summary(label=strategy)

    # Optimizer with differential learning rates
    if strategy in ('finetuned', 'distilled'):
        param_groups = qtl.get_param_groups(
            backbone_lr=args.lr * 10,     # 1e-3
            quantum_lr=args.lr * 0.1,     # 1e-5
            classifier_lr=args.lr,        # 1e-4
        )
    else:
        param_groups = [
            {'params': [p for p in student.parameters() if p.requires_grad],
             'lr': args.lr}
        ]

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    # Warmup + Cosine Annealing scheduler
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    if warmup_epochs > 0 and args.epochs > warmup_epochs:
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs])
        print(f"  Scheduler: {warmup_epochs}-epoch warmup → cosine decay")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  Scheduler: cosine decay (no warmup)")

    history = {
        'loss': [], 'val_acc': [], 'val_f1': [], 'epoch_times': [],
        'strategy': strategy,
        'backbone': TARGET_CONFIG['backbone'],
        'parameters': param_counts['_total'],
    }

    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Progressive unfreezing
        if strategy in ('finetuned', 'distilled'):
            if qtl.maybe_unfreeze(epoch):
                print(f"    ↳ Transferred components UNFROZEN at epoch {epoch+1}")
                param_groups = qtl.get_param_groups(
                    backbone_lr=args.lr * 10,
                    quantum_lr=args.lr * 0.1,
                    classifier_lr=args.lr,
                )
                optimizer = optim.Adam(param_groups, weight_decay=1e-5)
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch)

        loss = train_one_epoch(student, train_loader, criterion, optimizer,
                               device, qtl_wrapper=qtl, augment=augment)
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
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("  QTL LENET IMPROVED + EXTENDED TRANSFER — RESULTS SUMMARY")
    print("=" * 80)

    if source_metrics is not None:
        print(f"  Source (ResNet50+Quantum): "
              f"{source_metrics['val_acc']:.4f} val / "
              f"{source_metrics['test_acc']:.4f} test | ~24M params")
        print()

    print(f"  {'Strategy':<15} {'Params':>10} {'Trainable':>10} "
          f"{'Best Acc':>10} {'Best F1':>10} {'Avg Time':>10}")
    print(f"  {'─'*65}")

    for name, h in results.items():
        params = h['parameters']
        print(f"  {name:<15} {params['total']:>10,} {params['trainable']:>10,} "
              f"{h['best_acc']:>10.4f} {h['best_f1']:>10.4f} "
              f"{h['avg_epoch_time']:>9.1f}s")

    # Transfer benefit
    if 'scratch' in results:
        for name in ('distilled', 'finetuned', 'frozen'):
            if name in results:
                benefit = results[name]['best_acc'] - results['scratch']['best_acc']
                print(f"\n  📊 Transfer benefit ({name} vs scratch): "
                      f"{benefit:+.2%}")

    if source_metrics and results:
        best_strat = max(results.items(), key=lambda x: x[1]['best_acc'])
        param_reduction = 24_000_000 / best_strat[1]['parameters']['total']
        print(f"\n  🔬 Parameter reduction: {param_reduction:.0f}× fewer parameters")
        print(f"  🎯 Best ({best_strat[0]}): "
              f"{best_strat[1]['best_acc']:.4f} vs source "
              f"{source_metrics['val_acc']:.4f}")

    print("=" * 80)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QTL: LeNet Improved + Extended Weight Transfer')
    parser.add_argument('--data_root', type=str, default='./data/EuroSAT')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs per transfer strategy (default 100)')
    parser.add_argument('--source_epochs', type=int, default=20,
                        help='Epochs for source retraining')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bands', type=str, default='RGB',
                        choices=['RGB', 'ALL'])
    parser.add_argument('--retrain_source', action='store_true',
                        help='Retrain source model from scratch')
    parser.add_argument('--strategies', nargs='+',
                        default=['frozen', 'finetuned', 'distilled', 'scratch'],
                        help='Which strategies to run')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Linear warmup epochs before cosine decay')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable training data augmentation')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"  QTL: LeNet Improved + Extended Weight Transfer")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Target backbone: {TARGET_CONFIG['backbone']} "
          f"(3-conv, BatchNorm, 8192-dim)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Quantum: {SOURCE_CONFIG['n_qubits']} qubits, "
          f"depth {SOURCE_CONFIG['q_depth']}, "
          f"{SOURCE_CONFIG['encoding']} encoding")
    print(f"  Extended transfer: quantum + classifier + projector_bias")
    print(f"  Strategies: {args.strategies}")
    print(f"  Improvements: augment={'ON' if not args.no_augment else 'OFF'}, "
          f"warmup={args.warmup_epochs}ep, T=2.0, "
          f"label_smooth=0.1, wd=1e-5")

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
        teacher_model, ext_path = retrain_source(
            args, train_loader, val_loader, device
        )
    else:
        teacher_model, ext_path = load_source_model(device)

    # ------------------------------------------------------------------
    # Phase 0 — Verify source accuracy
    # ------------------------------------------------------------------
    source_metrics = None
    if teacher_model is not None:
        source_metrics = verify_source_model(
            teacher_model, val_loader, test_loader, device
        )
    else:
        print("\n  ✗ No source model available.")
        print(f"    Run with --retrain_source or place checkpoint at:")
        print(f"    {SOURCE_CKPT_PATH}")
        if 'distilled' in args.strategies:
            args.strategies = [s for s in args.strategies if s != 'distilled']
            print("    Skipping 'distilled' (no teacher)")

    # Handle missing weights
    if ext_path is None or not os.path.exists(ext_path or ''):
        ext_path = None
        transfer_strategies = [s for s in args.strategies if s == 'scratch']
        skipped = [s for s in args.strategies if s != 'scratch']
        if skipped:
            print(f"  ⚠ Skipping {skipped} (no weights)")
        args.strategies = transfer_strategies

    if teacher_model is None and 'distilled' in args.strategies:
        args.strategies = [s for s in args.strategies if s != 'distilled']
        print("  ⚠ Skipping distilled (no teacher)")

    # ------------------------------------------------------------------
    # Phase 2 — Transfer strategies
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2: Transfer Strategies (LeNet Improved + Quantum)")
    print("=" * 70)

    use_augment = not getattr(args, 'no_augment', False)

    results = {}
    for strategy in args.strategies:
        history = run_transfer_strategy(
            strategy, args, train_loader, val_loader, device,
            ext_weights_path=ext_path, teacher_model=teacher_model,
            augment=use_augment,
        )
        results[strategy] = history

        # Save per-strategy results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"qtl_improved_{strategy}_results.json"
        with open(out_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"  Results saved → {out_file}")

    # ------------------------------------------------------------------
    # Phase 3 — Summary
    # ------------------------------------------------------------------
    if results:
        print_summary(results, source_metrics=source_metrics)

        out_dir = Path(args.output_dir)
        summary = {
            'source_metrics': source_metrics,
            'source_config': SOURCE_CONFIG,
            'target_config': TARGET_CONFIG,
            'extended_transfer': ['quantum_layer', 'classifier', 'projector_bias'],
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
        summary_file = out_dir / 'qtl_improved_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary → {summary_file}")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
