# Hybrid Quantum-Classical Geospatial Classification

This repository contains the implementation of Hybrid Quantum-Classical Neural Networks for classifying geospatial data (EuroSAT dataset). It explores the integration of Variational Quantum Circuits (VQC), QAOA, and Quantum LSTMs with classical backbones like ResNet50 and Vision Transformers (ViT).

## 📖 Architecture & Documentation

Detailed architecture diagrams, data flow descriptions, and parameter explanations can be found in **[architecture.md](architecture.md)**.

## 🚀 Features

*   **Classical Backbones**: ResNet50, ViT (Vision Transformer)
*   **Quantum Encodings**: Angle Encoding, Amplitude Encoding, IQP Encoding
*   **Quantum Ansatzes**: Strongly Entangling Layers (VQC), QAOA-inspired, Quantum LSTM
*   **Dataset**: EuroSAT (RGB)

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd Quantum
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

The main entry point is `run_experiments.py`. You can run specific configurations or all of them.

### Run a specific experiment
```bash
# Run the baseline ResNet50 (Classical)
python run_experiments.py --config baseline_resnet50

# Run a Hybrid ResNet with Angle Encoding and VQC
python run_experiments.py --config hybrid_resnet_angle_vqc
```

### Run all experiments
```bash
python run_experiments.py --config all
```

### Common Arguments
*   `--epochs`: Number of training epochs (default: 5)
*   `--batch_size`: Batch size (default: 32)
*   `--n_qubits`: Number of qubits (default: 4)
*   `--q_depth`: Depth of the quantum circuit (default: 1)
*   `--lr`: Learning rate (default: 1e-4)

## 📂 Project Structure

*   `src/`: Source code for models and data loading.
    *   `models/`: Contains `backbones.py`, `hybrid_model.py`, and `quantum_layers.py`.
    *   `data/`: Data loading and splitting logic.
    *   `utils/`: Metric calculation utilities.
*   `run_experiments.py`: Main script to train and validate models.
*   `results/`: Directory where experiment results (JSON) are saved.
*   `architecture.md`: Detailed architectural documentation.

## 📊 Results

Results are saved as JSON files in the `results/` directory. Each file contains loss curves, validation accuracy, and execution times.
