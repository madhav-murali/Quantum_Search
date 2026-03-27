#!/bin/bash

##This should be ran in the root dir, not in the scripts dir
##changed folders for a cleanr look

# Configuration Parameters
DATA_ROOT="./data/EuroSAT"
EPOCHS=5
QUBITS=4
BANDS="ALL"
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Define the experiment matrix
# Format: "config_name batch_size"
EXPERIMENTS=(
    #"baseline_resnet50 32"
    #"hybrid_resnet_angle_vqc 32"
    #"hybrid_resnet_amplitude_vqc 32"
   # "hybrid_resnet_iqp_vqc 32"
    "baseline_vit 8"
    "hybrid_vit_iqp_qaoa 4"
    "hybrid_vit_qlstm 4"
)

echo "Starting Research Matrix: $(date)"
echo "Logging to: $LOG_DIR"
echo "-------------------------------------------"

for exp in "${EXPERIMENTS[@]}"; do
    # Split the string into config and batch size
    read -r CONFIG BATCH <<< "$exp"
    
    echo "🚀 RUNNING: $CONFIG (Batch: $BATCH, Qubits: $QUBITS)"
    
    # Run the experiment and log output
    python3 run_experiments.py \
        --config "$CONFIG" \
        --data_root "$DATA_ROOT" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --n_qubits "$QUBITS" \
        --bands "$BANDS" \
        | tee "$LOG_DIR/${CONFIG}_output.txt"
    
    echo "✅ FINISHED: $CONFIG"
    echo "-------------------------------------------"
done

echo "Research Matrix Complete. Results saved in $LOG_DIR"