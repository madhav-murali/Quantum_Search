"""
Quantum Transfer Learning (QTL) Model with Knowledge Distillation.

This module provides a dedicated model class for transferring quantum circuit
knowledge from a large teacher backbone (ResNet50) to a small student backbone
(LeNet5), achieving significant parameter reduction while preserving performance.

Key features:
  - Knowledge distillation loss (KL-divergence on soft predictions)
  - Progressive unfreezing scheduler for the quantum layer
  - Differential learning rates per component
  - Detailed parameter count reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QTLModel(nn.Module):
    """
    Quantum Transfer Learning model with knowledge distillation support.

    Wraps a HybridGeoModel student and provides:
      - A distillation forward pass that returns both student logits
        and the distillation loss against a teacher's logits.
      - Progressive unfreezing of the quantum layer after a warmup period.
      - Helper methods for parameter counting and optimizer group creation.

    Args:
        student (nn.Module): The lightweight HybridGeoModel (e.g. LeNet5 backbone).
        teacher (nn.Module, optional): The large HybridGeoModel (e.g. ResNet50 backbone).
            If None, distillation is disabled and only the student trains.
        temperature (float): Softmax temperature for distillation (default 4.0).
        alpha (float): Weight of the distillation loss vs hard-label loss (default 0.3).
            Total loss = alpha * distill_loss + (1 - alpha) * hard_loss
        unfreeze_epoch (int): Epoch at which to unfreeze quantum layer (default 5).
    """

    def __init__(self, student, teacher=None, temperature=4.0, alpha=0.3,
                 unfreeze_epoch=5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.unfreeze_epoch = unfreeze_epoch
        self._quantum_frozen = False

        # Freeze teacher entirely — it is only used for inference
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Standard forward pass — returns student logits only."""
        return self.student(x)

    def distillation_loss(self, x, hard_targets, criterion):
        """
        Compute combined hard-label + distillation loss.

        Args:
            x (Tensor): Input images (B, C, H, W).
            hard_targets (Tensor): Ground-truth class indices (B,).
            criterion (nn.Module): Hard-label loss (e.g. CrossEntropyLoss).

        Returns:
            Tuple[Tensor, Tensor]: (total_loss, student_logits)
        """
        student_logits = self.student(x)
        hard_loss = criterion(student_logits, hard_targets)

        if self.teacher is None:
            return hard_loss, student_logits

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Soft targets via temperature scaling
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        return total_loss, student_logits

    # ------------------------------------------------------------------
    # Progressive unfreezing
    # ------------------------------------------------------------------

    def freeze_quantum(self):
        """Freeze quantum layer parameters."""
        if hasattr(self.student, 'quantum_layer'):
            for p in self.student.quantum_layer.parameters():
                p.requires_grad = False
            self._quantum_frozen = True

    def unfreeze_quantum(self):
        """Unfreeze quantum layer parameters."""
        if hasattr(self.student, 'quantum_layer'):
            for p in self.student.quantum_layer.parameters():
                p.requires_grad = True
            self._quantum_frozen = False

    def maybe_unfreeze(self, epoch):
        """
        Call once per epoch.  Unfreezes the quantum layer when
        ``epoch >= self.unfreeze_epoch``.

        Returns:
            bool: True if unfreezing happened on *this* call.
        """
        if self._quantum_frozen and epoch >= self.unfreeze_epoch:
            self.unfreeze_quantum()
            return True
        return False

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def get_param_groups(self, backbone_lr=1e-3, quantum_lr=1e-5, classifier_lr=1e-3):
        """
        Build parameter groups with differential learning rates.

        Args:
            backbone_lr (float): LR for backbone + projector.
            quantum_lr (float): LR for quantum layer (lower to preserve transfer).
            classifier_lr (float): LR for the final classifier head.

        Returns:
            list[dict]: Parameter groups for ``torch.optim.Adam`` etc.
        """
        groups = []

        # Backbone
        backbone_params = list(self.student.backbone.parameters())
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'})

        # Projector
        projector_params = list(self.student.projector.parameters())
        if projector_params:
            groups.append({'params': projector_params, 'lr': backbone_lr, 'name': 'projector'})

        # Quantum adapter (if present and not Identity)
        if hasattr(self.student, 'quantum_adapter') and not isinstance(self.student.quantum_adapter, nn.Identity):
            adapter_params = list(self.student.quantum_adapter.parameters())
            if adapter_params:
                groups.append({'params': adapter_params, 'lr': quantum_lr, 'name': 'quantum_adapter'})

        # Quantum layer
        if hasattr(self.student, 'quantum_layer') and not isinstance(self.student.quantum_layer, nn.Identity):
            q_params = list(self.student.quantum_layer.parameters())
            if q_params:
                groups.append({'params': q_params, 'lr': quantum_lr, 'name': 'quantum_layer'})

        # Classifier
        classifier_params = list(self.student.classifier.parameters())
        if classifier_params:
            groups.append({'params': classifier_params, 'lr': classifier_lr, 'name': 'classifier'})

        return groups

    # ------------------------------------------------------------------
    # Parameter reporting
    # ------------------------------------------------------------------

    def count_parameters(self):
        """
        Count parameters by component.

        Returns:
            dict: Mapping component name → {total, trainable}.
        """
        report = {}
        components = {
            'backbone': self.student.backbone,
            'projector': self.student.projector,
            'classifier': self.student.classifier,
        }

        if hasattr(self.student, 'quantum_adapter') and not isinstance(self.student.quantum_adapter, nn.Identity):
            components['quantum_adapter'] = self.student.quantum_adapter

        if hasattr(self.student, 'quantum_layer') and not isinstance(self.student.quantum_layer, nn.Identity):
            components['quantum_layer'] = self.student.quantum_layer

        grand_total = 0
        grand_trainable = 0

        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
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
