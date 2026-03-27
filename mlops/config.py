"""
Centralized MLOps pipeline configuration.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data" / "EuroSAT"
MLOPS_RUNS_DIR = PROJECT_ROOT / "mlops_runs"


# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "quantum-mlops-artifacts")
S3_PREFIX = os.getenv("S3_PREFIX", "quantum-qtl")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# QTL-specific defaults (must match between source and target)
# ---------------------------------------------------------------------------
QTL_N_QUBITS = int(os.getenv("QTL_N_QUBITS", "8"))
QTL_Q_DEPTH = int(os.getenv("QTL_Q_DEPTH", "2"))
QTL_ENCODING = os.getenv("QTL_ENCODING", "amplitude")
QTL_ANSATZ = os.getenv("QTL_ANSATZ", "vqc")
QTL_STANDARD_DIM = int(os.getenv("QTL_STANDARD_DIM", "256"))
QTL_N_CLASSES = 10
QTL_IN_CHANNELS = 3


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_EPOCHS = int(os.getenv("SOURCE_EPOCHS", "20"))
DEFAULT_TARGET_EPOCHS = int(os.getenv("TARGET_EPOCHS", "100"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEFAULT_LR = float(os.getenv("LEARNING_RATE", "1e-3"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))


# ---------------------------------------------------------------------------
# Checkpoint paths for QTL workflow
# ---------------------------------------------------------------------------
SOURCE_CHECKPOINT = CHECKPOINTS_DIR / "qtl_source_resnet_amplitude.pth"
QUANTUM_WEIGHTS = CHECKPOINTS_DIR / "qtl_source_quantum_weights.pth"
EXTENDED_WEIGHTS = CHECKPOINTS_DIR / "qtl_source_extended_weights.pth"


# ---------------------------------------------------------------------------
# QTL transfer strategies (order matters — run in sequence)
# ---------------------------------------------------------------------------
QTL_STRATEGIES = ["frozen", "finetuned", "distilled", "scratch"]


# ---------------------------------------------------------------------------
# Pipeline stage configs
# ---------------------------------------------------------------------------
PIPELINE_STAGES = {
    "source_training": {
        "config": "qtl_source_resnet_amplitude",
        "description": "Train ResNet50 + Quantum source model",
        "script": "run_experiments.py",
        "args": {
            "--config": "qtl_source_resnet_amplitude",
            "--epochs": str(DEFAULT_SOURCE_EPOCHS),
            "--n_qubits": str(QTL_N_QUBITS),
            "--q_depth": str(QTL_Q_DEPTH),
            "--bands": "RGB",
        },
    },
    "weight_extraction": {
        "description": "Extract quantum + extended weights from source",
        "script": "scripts/extract_quantum_weights.py",
        "args": {
            "--checkpoint": str(SOURCE_CHECKPOINT),
            "--output": str(QUANTUM_WEIGHTS),
        },
    },
    "qtl_transfer": {
        "description": "Run QTL transfer to improved LeNet5",
        "script": "qtl/lenet_improved_qtl.py",
        "args": {
            "--epochs": str(DEFAULT_TARGET_EPOCHS),
        },
    },
}
