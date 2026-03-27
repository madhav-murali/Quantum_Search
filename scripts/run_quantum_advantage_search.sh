#!/bin/bash
##This should be ran in the root dir, not in the scripts dir
##changed folders for a cleanr look else need modifications 
# Configuration Parameters
DATA_ROOT="./data/EuroSAT"
EPOCHS=10
LOG_DIR="./logs/search_advantage_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Define the experiment matrix for finding Quantum Advantage
# Format: "config_name batch_size n_qubits depth lr"
EXPERIMENTS=(
    # Baseline for reference (High performance expected)
    #"baseline_resnet50 32 0 0 1e-4"
    #"baseline_vit 8 0 0 1e-4"

    # Hybrid Models - Increased Capacity & Search
    
    # 1. Deeper Quantum Circuits (Depth 2 & 3)
    "hybrid_resnet_amplitude_vqc 32 8 2 1e-4" 
    "hybrid_resnet_amplitude_vqc 32 8 3 1e-4"
    
    # 2. Angle Encoding with Higher Qubits (8) - more features
    "hybrid_resnet_angle_vqc 32 8 2 1e-3"
    
    # 3. IQP with Higher Qubits (Data Re-uploading potential?)
    "hybrid_resnet_iqp_vqc 32 8 2 1e-3"
    
    # 4. ViT Hybrid - Depth 2
    "hybrid_vit_qlstm 4 4 2 5e-5" # Lower LR for LSTM stability
    "hybrid_vit_iqp_qaoa 4 4 2 1e-4"
)

echo "Starting Quantum Advantage Search: $(date)"
echo "Logging to: $LOG_DIR"
echo "-------------------------------------------"

for exp in "${EXPERIMENTS[@]}"; do
    # Split the string
    read -r CONFIG BATCH QUBITS DEPTH LR <<< "$exp"
    
    echo "🚀 RUNNING: $CONFIG"
    echo "   -> Batch: $BATCH, Qubits: $QUBITS, Depth: $DEPTH, LR: $LR"
    
    # Run the experiment
    python3 run_experiments.py \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --n_qubits "$QUBITS" \
        --q_depth "$DEPTH" \
        --lr "$LR" \
        --bands "ALL" \
        | tee "$LOG_DIR/${CONFIG}_q${QUBITS}_d${DEPTH}_output.txt"
    
    echo "✅ FINISHED: $CONFIG"
    echo "-------------------------------------------"
done

echo "Search Complete. Results saved in $LOG_DIR"
