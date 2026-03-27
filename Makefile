.PHONY: train train-source train-qtl lint test dvc-pull s3-push pipeline clean help

# ============================================================================
# QTL MLOps Pipeline — Makefile
# ============================================================================
#
# Quick start:
#   make pipeline                    # Full QTL pipeline (source → extract → transfer)
#   make pipeline SKIP_SOURCE=1      # Skip source training, use cached checkpoint
#   make train-qtl EPOCHS=50         # Just run QTL transfer with 50 epochs
#

# Defaults
CONFIG          ?= qtl_source_resnet_amplitude
SOURCE_EPOCHS   ?= 20
TARGET_EPOCHS   ?= 100
BATCH_SIZE      ?= 32
STRATEGIES      ?= frozen finetuned distilled scratch
PYTHON          ?= python

# ---- Full Pipeline --------------------------------------------------------
pipeline:
	$(PYTHON) -m mlops.train_pipeline \
		--source-epochs $(SOURCE_EPOCHS) \
		--target-epochs $(TARGET_EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--strategies $(STRATEGIES) \
		--skip-s3 \
		$(if $(SKIP_SOURCE),--skip-source,)

pipeline-s3:
	$(PYTHON) -m mlops.train_pipeline \
		--source-epochs $(SOURCE_EPOCHS) \
		--target-epochs $(TARGET_EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--strategies $(STRATEGIES) \
		$(if $(SKIP_SOURCE),--skip-source,)

# ---- Individual Stages ----------------------------------------------------
train-source:
	$(PYTHON) run_experiments.py \
		--config $(CONFIG) \
		--epochs $(SOURCE_EPOCHS) \
		--n_qubits 8 --q_depth 2 --bands RGB \
		--batch_size $(BATCH_SIZE)

extract-weights:
	$(PYTHON) scripts/extract_quantum_weights.py \
		--checkpoint checkpoints/qtl_source_resnet_amplitude.pth \
		--output checkpoints/qtl_source_quantum_weights.pth

train-qtl:
	$(PYTHON) qtl/lenet_improved_qtl.py \
		--epochs $(TARGET_EPOCHS) \
		--batch_size $(BATCH_SIZE)

# ---- Code Quality ---------------------------------------------------------
lint:
	flake8 src/ mlops/ qtl/ scripts/ \
		--count --max-complexity=15 --max-line-length=120 --statistics

test:
	pytest tests/ -v --tb=short

check:
	$(PYTHON) -m py_compile mlops/config.py
	$(PYTHON) -m py_compile mlops/s3_utils.py
	$(PYTHON) -m py_compile mlops/experiment_tracker.py
	$(PYTHON) -m py_compile mlops/train_pipeline.py
	@echo "✅ All files compile successfully"

# ---- Data & Artifacts -----------------------------------------------------
dvc-pull:
	dvc pull

s3-push:
	$(PYTHON) -c "\
from mlops.s3_utils import S3Client; \
s3 = S3Client(); \
n = s3.upload_directory('results', 'latest/results'); \
print(f'Uploaded {n} result files to S3')"

# ---- Cleanup --------------------------------------------------------------
clean:
	rm -rf mlops_runs/
	rm -rf __pycache__ src/__pycache__ mlops/__pycache__ qtl/__pycache__
	find . -name "*.pyc" -delete

# ---- Help -----------------------------------------------------------------
help:
	@echo "QTL MLOps Pipeline"
	@echo ""
	@echo "  make pipeline                  Full pipeline (local, no S3)"
	@echo "  make pipeline SKIP_SOURCE=1    Skip source training"
	@echo "  make pipeline-s3               Full pipeline with S3 upload"
	@echo "  make train-source              Train ResNet50+Quantum source"
	@echo "  make extract-weights           Extract quantum weights"
	@echo "  make train-qtl                 Run QTL transfer strategies"
	@echo "  make train-qtl EPOCHS=50       Custom epochs"
	@echo "  make lint                      Run flake8"
	@echo "  make test                      Run pytest"
	@echo "  make check                     Validate Python syntax"
	@echo "  make dvc-pull                  Pull data from S3 via DVC"
	@echo "  make s3-push                   Push results to S3"
	@echo "  make clean                     Clean generated files"
