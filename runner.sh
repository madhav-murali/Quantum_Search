#!/bin/bash
source .venv/bin/activate

DATASETS=("EuroSAT" "SIRI-WHU" "UC_M_LUC")
ENCODINGS=("amplitude" "molecular")
ANSATZES=("vqc" "pqc")
DEPTHS=(1 2)
FRAC=0.2
SOURCE_EPOCHS=10
TARGET_EPOCHS=30

mkdir -p logs/qtl_grid_search

for DATASET in "${DATASETS[@]}"; do
    for ENC in "${ENCODINGS[@]}"; do
        for ANS in "${ANSATZES[@]}"; do
            for D in "${DEPTHS[@]}"; do
            
                RUN_NAME="${DATASET}_${ENC}_${ANS}_d${D}"
                echo "=================================================="
                echo "Starting Workflow: $RUN_NAME"
                echo "=================================================="
                
                # Step 1: Train Heavy Source Model (ResNet50 + Quantum)
                echo "-> Training QTL Source..."
                python run_experiments.py \
                    --config qtl_source_resnet_amplitude \
                    --dataset "$DATASET" \
                    --encoding "$ENC" \
                    --ansatz "$ANS" \
                    --q_depth "$D" \
                    --epochs $SOURCE_EPOCHS \
                    --subset_fraction $FRAC \
                    | tee "logs/qtl_grid_search/${RUN_NAME}_source_output.txt"
                
                # Step 2: Extract weights and Finetune Lightweight Target Model (LeNet5)
                # Note: Because we override encoding/ansatz/depth via CLI, the LeNet initializes perfectly 
                # to accept the pre-extracted quantum behavior.
                echo "-> Finetuning QTL Target..."
                python run_experiments.py \
                    --config qtl_lenet_finetuned \
                    --dataset "$DATASET" \
                    --encoding "$ENC" \
                    --ansatz "$ANS" \
                    --q_depth "$D" \
                    --epochs $TARGET_EPOCHS \
                    --subset_fraction $FRAC \
                    | tee "logs/qtl_grid_search/${RUN_NAME}_target_output.txt"
                    
            done
        done
    done
done

echo "Parsing logs and generating comparison plots..."
python parse_logs_and_analyze.py logs/qtl_grid_search
echo "All done! Logs and plots saved in logs/qtl_grid_search/"
