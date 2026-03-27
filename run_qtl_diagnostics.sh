#!/bin/bash
# Sequential QTL Diagnostics Runner
# Runs experiments one at a time so you can monitor progress

set -e

echo "=========================================="
echo "Quantum Transfer Learning Diagnostics"
echo "Running experiments SEQUENTIALLY"
echo "=========================================="
echo ""

# Experiment 1: Baseline (no transfer, with standardization)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 1/4: LeNet5 Baseline (standard_dim=256)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config lenet5_baseline_amplitude \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --lr 1e-4

echo ""
echo "✓ Experiment 1 complete"
echo ""

# Experiment 2: Original LeNet5 (no standardization)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 2/4: LeNet5 Original (no standard_dim)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config lenet5_quantum_amplitude_vqc \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --lr 1e-4

echo ""
echo "✓ Experiment 2 complete"
echo ""

# Experiment 3: QTL Frozen with higher learning rate
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 3/4: QTL Frozen (higher LR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config qtl_lenet_frozen \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --lr 5e-4

echo ""
echo "✓ Experiment 3 complete"
echo ""

# Experiment 4: QTL Fine-tuned
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 4/4: QTL Fine-tuned"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config qtl_lenet_finetuned \
    --epochs 20 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 2 \
    --lr 1e-4

echo ""
echo "✓ Experiment 4 complete"
echo ""

# Generate analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Generating Analysis and Visualizations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python reports/generate_qtl_analysis.py

echo ""
echo "=========================================="
echo "✓ All diagnostics complete!"
echo "=========================================="
echo ""
echo "Check results/ directory for outputs:"
echo "  - qtl_convergence_comparison.png"
echo "  - qtl_accuracy_comparison.png"
echo "  - qtl_parameter_efficiency.png"
echo "  - qtl_detailed_results.md"
echo ""
