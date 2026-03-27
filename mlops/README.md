# MLOps Pipeline for QTL (Quantum Transfer Learning)

Production-grade pipeline for training, versioning, and deploying Quantum Transfer Learning models.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ Stage 1      │    │ Stage 2      │    │ Stage 3        │    │ Stage 4      │
│ Train Source │───▶│ Extract      │───▶│ QTL Transfer   │───▶│ Upload to S3 │
│ ResNet50+Q   │    │ Weights      │    │ (4 strategies) │    │ + Report     │
└──────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
```

## Quick Start

```bash
# Full pipeline (local, no S3)
make pipeline

# Skip source training if you already have a checkpoint
make pipeline SKIP_SOURCE=1

# Custom epochs
make pipeline SOURCE_EPOCHS=10 TARGET_EPOCHS=50

# Just run QTL transfer
make train-qtl EPOCHS=50

# Full pipeline with S3 upload
make pipeline-s3
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure AWS (optional, for S3)
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
export S3_BUCKET=quantum-mlops-artifacts
```

Or create a `.env` file (auto-loaded by `python-dotenv`):
```env
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET=quantum-mlops-artifacts
```

### 3. Configure DVC for data versioning
```bash
# Track the EuroSAT dataset
dvc add data/EuroSAT
dvc push

# On a new machine, pull data
dvc pull
```

## GitHub Actions

Two workflows are included:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `qtl_pipeline.yml` | Push to `main`, manual dispatch | Full QTL training pipeline |
| `lint_test.yml` | Push to any branch, PRs | Linting + syntax validation |

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_REGION` | AWS region (e.g., `us-east-1`) |
| `S3_BUCKET` | S3 bucket name |

### Manual Dispatch

You can trigger the pipeline manually from the GitHub Actions tab with custom parameters:
- **source_epochs**: Source model training epochs (default: 20)
- **target_epochs**: QTL target training epochs (default: 100)
- **strategies**: Space-separated strategies (default: all four)
- **skip_source**: Skip source training, use cached checkpoint

## Pipeline Output

Each run creates a directory under `mlops_runs/<run_id>/`:
```
mlops_runs/20260317_042800_abc1234_qtl_pipeline/
├── manifest.json           # Run metadata + stage status
├── source_metrics.json     # Source model training metrics
├── qtl_metrics.json        # Per-strategy QTL metrics
├── summary.json            # Comparison summary
├── run_report.md           # Human-readable report
└── artifacts/              # Checkpoints, results copies
```

## Makefile Targets

```bash
make help          # Show all available targets
make pipeline      # Full pipeline (no S3)
make pipeline-s3   # Full pipeline with S3
make train-source  # Train source model only
make extract-weights  # Extract quantum weights
make train-qtl     # Run QTL transfer only
make lint          # Run flake8
make test          # Run pytest
make check         # Validate Python syntax
make dvc-pull      # Pull data from DVC remote
make s3-push       # Push latest results to S3
make clean         # Clean generated files
```
