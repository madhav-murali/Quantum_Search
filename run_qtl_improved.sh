#!/bin/bash
# IMPROVED Sequential QTL Diagnostics with Angle Encoding
# This uses angle encoding which works better for transfer learning

set -e

echo "=========================================="
echo "IMPROVED Quantum Transfer Learning"
echo "Using angle encoding for better transfer"
echo "=========================================="
echo ""

# Experiment 1: Train ResNet50 with angle encoding (source)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 1/6: Train ResNet50 Source (angle encoding)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config qtl_source_resnet_angle \
    --epochs 15 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 1e-4

echo ""
echo "✓ Source model trained"
echo ""

# Extract quantum weights
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Extracting quantum weights..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/extract_quantum_weights.py \
    --checkpoint checkpoints/qtl_source_resnet_angle.pth \
    --output checkpoints/qtl_source_angle_weights.pth

echo ""
echo "✓ Quantum weights extracted"
echo ""

# Experiment 2: QTL Frozen (angle)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 2/6: LeNet5 QTL Frozen (angle)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config qtl_lenet_frozen_angle \
    --epochs 25 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 1e-3

echo ""
echo "✓ QTL Frozen complete"
echo ""

# Experiment 3: QTL Fine-tuned (angle)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 3/6: LeNet5 QTL Fine-tuned (angle)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config qtl_lenet_finetuned_angle \
    --epochs 30 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 5e-4

echo ""
echo "✓ QTL Fine-tuned complete"
echo ""

# Experiment 4: LeNet5 from scratch (angle, no transfer)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 4/6: LeNet5 Baseline (angle, from scratch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config lenet_quantum_angle_vqc \
    --epochs 30 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 5e-4

echo ""
echo "✓ Baseline complete"
echo ""

# Experiment 5: Original LeNet5 (amplitude, no standard_dim)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 5/6: LeNet5 Original (amplitude, reference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config lenet5_quantum_amplitude_vqc \
    --epochs 30 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 5e-4

echo ""
echo "✓ Reference complete"
echo ""

# Experiment 6: ResNet50 baseline for comparison
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Experiment 6/6: ResNet50 Hybrid (angle, reference)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python run_experiments.py \
    --config hybrid_resnet_angle_vqc \
    --epochs 15 \
    --batch_size 32 \
    --n_qubits 8 \
    --q_depth 3 \
    --lr 1e-4

echo ""
echo "✓ ResNet50 reference complete"
echo ""

# Generate analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Generating Analysis and Visualizations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python reports/generate_qtl_analysis_v2.py

echo ""
echo "=========================================="
echo "✓ All experiments complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - qtl_convergence_comparison.png"
echo "  - qtl_accuracy_comparison.png"
echo "  - qtl_parameter_efficiency.png"
echo "  - qtl_detailed_results.md"
echo ""
