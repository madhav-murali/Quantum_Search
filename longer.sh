#!/bin/bash
source .venv/bin/activate

# Use the full dataset and provide ample epochs for the expressive models to learn
DATASET="EuroSAT" # "EuroSAT" gets by far the best results (95%+). For new datasets: SIRI-WHU (max ~45%), UC_M_LUC (max ~33%)
ENC="geospatial_patch"
ANS="vqc"
DEPTH=2
FRAC=0.3    
SOURCE_EPOCHS=30
TARGET_EPOCHS=100

RUN_NAME="${DATASET}_${ENC}_${ANS}_d${DEPTH}_extended"

# Create a dedicated log directory
mkdir -p logs/qtl_extended

echo "=================================================="
echo "Starting Extended Workflow: $RUN_NAME"
echo "=================================================="

# Step 1: Train Heavy Source Model (ResNet50 + Quantum)
echo "-> Training QTL Source..."
python run_experiments.py \
    --config qtl_source_resnet_amplitude \
    --dataset "$DATASET" \
    --encoding "$ENC" \
    --ansatz "$ANS" \
    --n_qubits 8 \
    --q_depth "$DEPTH" \
    --epochs $SOURCE_EPOCHS \
    --subset_fraction $FRAC \
    | tee "logs/qtl_extended/${RUN_NAME}_source_output.txt"

# Extract the quantum weights to avoid qubit mismatch
echo "-> Extracting Quantum Weights..."
python extract_weights.py checkpoints/qtl_source_resnet_amplitude.pth checkpoints/qtl_source_quantum_weights.pth

# Step 2: Extract weights and Finetune Lightweight Target Model (LeNet5)
echo "-> Finetuning QTL Target..."
python run_experiments.py \
    --config qtl_lenet_finetuned \
    --dataset "$DATASET" \
    --encoding "$ENC" \
    --ansatz "$ANS" \
    --n_qubits 8 \
    --q_depth "$DEPTH" \
    --epochs $TARGET_EPOCHS \
    --subset_fraction $FRAC \
    | tee "logs/qtl_extended/${RUN_NAME}_target_output.txt"

echo "Parsing logs and generating comparison plots..."
python parse_logs_and_analyze.py logs/qtl_extended
echo "All done! Logs and plots saved in logs/qtl_extended/"
