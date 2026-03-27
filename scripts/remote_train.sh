#!/bin/bash
# =========================================================================
# Remote Training Runner for Quantum Geospatial Pipeline
# =========================================================================
# This script facilitates training models explicitly on Guacamole or Kaggle.
# Ensure your environment is set up with required packages.
#
# Usage:
#   ./scripts/remote_train.sh --env <kaggle|guacamole> --config <config_name> [options]
# Example:
#   ./scripts/remote_train.sh --env guacamole --config multistage_resnet_quantum --dataset SIRI-WHU
# =========================================================================

# Defaults
ENV="auto"
CONFIG="all"
DATASET="EuroSAT"
SUBSET_FRAC="1.0"
EPOCHS=5

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --subset_fraction) SUBSET_FRAC="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Preparing environment for: $ENV"

# Set platform-specific variables
if [ "$ENV" == "guacamole" ]; then
    export GUACAMOLE_ENV="1"
elif [ "$ENV" == "kaggle" ]; then
    export KAGGLE_KERNEL_RUN_TYPE="Interactive"
fi

# Run the experiment
echo "Running experiments with Config: $CONFIG, Dataset: $DATASET, Subset: $SUBSET_FRAC"
python run_experiments.py --env "$ENV" --config "$CONFIG" --dataset "$DATASET" --subset_fraction "$SUBSET_FRAC" --epochs "$EPOCHS"

echo "Remote training script complete."
