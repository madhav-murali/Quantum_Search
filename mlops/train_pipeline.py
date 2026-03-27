#!/usr/bin/env python3
"""
Orchestrated QTL MLOps Pipeline.

End-to-end pipeline that:
  1. Trains the ResNet50 + Quantum source model
  2. Extracts quantum + extended weights
  3. Runs all QTL transfer strategies on improved LeNet5
  4. Collects metrics and uploads everything to S3

Usage:
    # Full pipeline
    python -m mlops.train_pipeline

    # Skip source training (use existing checkpoint)
    python -m mlops.train_pipeline --skip-source

    # Specific strategies only
    python -m mlops.train_pipeline --strategies frozen finetuned

    # Custom epochs
    python -m mlops.train_pipeline --source-epochs 20 --target-epochs 100
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from mlops.config import (
    PROJECT_ROOT,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    SOURCE_CHECKPOINT,
    QUANTUM_WEIGHTS,
    EXTENDED_WEIGHTS,
    QTL_STRATEGIES,
    DEFAULT_SOURCE_EPOCHS,
    DEFAULT_TARGET_EPOCHS,
    DEFAULT_BATCH_SIZE,
    QTL_N_QUBITS,
    QTL_Q_DEPTH,
    ACCURACY_THRESHOLD,
    PIPELINE_STAGES,
)
from mlops.experiment_tracker import ExperimentTracker
from mlops.s3_utils import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlops.pipeline")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_source_training(tracker: ExperimentTracker,
                        source_epochs: int, batch_size: int) -> bool:
    """Stage 1: Train ResNet50 + Quantum source model."""
    tracker.start_stage("source_training",
                        "Train ResNet50 + Quantum (amplitude, 8q, depth 2)")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "run_experiments.py"),
        "--config", "qtl_source_resnet_amplitude",
        "--epochs", str(source_epochs),
        "--n_qubits", str(QTL_N_QUBITS),
        "--q_depth", str(QTL_Q_DEPTH),
        "--batch_size", str(batch_size),
        "--bands", "RGB",
    ]

    logger.info("Running source training: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    logger.info("STDOUT:\n%s", result.stdout[-2000:] if result.stdout else "(empty)")
    if result.returncode != 0:
        logger.error("STDERR:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        tracker.end_stage("source_training", status="failed")
        return False

    # Read source results
    results_file = RESULTS_DIR / "qtl_source_resnet_amplitude_results.json"
    metrics = {}
    if results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        tracker.log_source_metrics(metrics)
        tracker.copy_artifact(str(results_file))

    best_acc = max(metrics.get("val_acc", [0]))
    tracker.end_stage("source_training", status="success",
                      metrics={"best_accuracy": best_acc,
                               "epochs": source_epochs})
    return True


def run_weight_extraction(tracker: ExperimentTracker) -> bool:
    """Stage 2: Extract quantum weights from source checkpoint."""
    tracker.start_stage("weight_extraction",
                        "Extract quantum + extended weights from source")

    if not SOURCE_CHECKPOINT.exists():
        logger.error("Source checkpoint not found: %s", SOURCE_CHECKPOINT)
        tracker.end_stage("weight_extraction", status="failed",
                          metrics={"error": "source checkpoint missing"})
        return False

    # Extract quantum weights
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "extract_quantum_weights.py"),
        "--checkpoint", str(SOURCE_CHECKPOINT),
        "--output", str(QUANTUM_WEIGHTS),
    ]

    logger.info("Extracting quantum weights: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    logger.info("STDOUT:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("Weight extraction failed:\n%s", result.stderr)
        tracker.end_stage("weight_extraction", status="failed")
        return False

    tracker.copy_artifact(str(QUANTUM_WEIGHTS))
    if EXTENDED_WEIGHTS.exists():
        tracker.copy_artifact(str(EXTENDED_WEIGHTS))

    tracker.end_stage("weight_extraction", status="success",
                      metrics={"quantum_weights": str(QUANTUM_WEIGHTS)})
    return True


def run_qtl_transfer(tracker: ExperimentTracker,
                     target_epochs: int, batch_size: int,
                     strategies: list[str]) -> bool:
    """Stage 3: Run QTL transfer strategies on improved LeNet5."""
    tracker.start_stage("qtl_transfer",
                        f"QTL transfer to improved LeNet5 ({', '.join(strategies)})")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "qtl" / "lenet_improved_qtl.py"),
        "--epochs", str(target_epochs),
        "--batch_size", str(batch_size),
    ]

    logger.info("Running QTL transfer: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    logger.info("STDOUT (last 3000 chars):\n%s",
                result.stdout[-3000:] if result.stdout else "(empty)")
    if result.returncode != 0:
        logger.error("QTL transfer failed:\n%s",
                      result.stderr[-2000:] if result.stderr else "(empty)")
        tracker.end_stage("qtl_transfer", status="failed")
        return False

    # Collect strategy results
    strategy_metrics = {}
    for strategy in strategies:
        # Try improved results first, then original naming
        for prefix in ["qtl_improved", "qtl_lenet5_amplitude"]:
            results_file = RESULTS_DIR / f"{prefix}_{strategy}_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                strategy_metrics[strategy] = data
                tracker.log_qtl_metrics(strategy, data)
                tracker.copy_artifact(str(results_file))
                break

    # Build summary
    summary = {}
    for strat, m in strategy_metrics.items():
        best_acc = max(m.get("val_acc", [0]))
        best_f1 = max(m.get("val_f1", [0]))
        summary[strat] = {"best_acc": best_acc, "best_f1": best_f1}

    # Determine best strategy
    if summary:
        best_strat = max(summary, key=lambda s: summary[s]["best_acc"])
        best_acc = summary[best_strat]["best_acc"]
        passed = best_acc >= ACCURACY_THRESHOLD
    else:
        best_strat = "N/A"
        best_acc = 0
        passed = False

    tracker.end_stage("qtl_transfer", status="success" if passed else "below_threshold",
                      metrics={
                          "best_strategy": best_strat,
                          "best_accuracy": best_acc,
                          "threshold": ACCURACY_THRESHOLD,
                          "passed": passed,
                          "strategies_run": list(strategy_metrics.keys()),
                      })
    tracker.log_summary(summary)
    return True


def upload_to_s3(tracker: ExperimentTracker) -> bool:
    """Stage 4: Upload run artifacts to S3."""
    tracker.start_stage("s3_upload", "Upload run bundle to S3")

    try:
        s3 = S3Client()
        s3_prefix = f"runs/{tracker.run_id}"
        count = s3.upload_directory(str(tracker.run_dir), s3_prefix)
        logger.info("Uploaded %d files to S3 under %s", count, s3_prefix)
        tracker.end_stage("s3_upload", status="success",
                          metrics={"files_uploaded": count, "s3_prefix": s3_prefix})
        return True
    except Exception as exc:
        logger.warning("S3 upload failed (pipeline continues): %s", exc)
        tracker.end_stage("s3_upload", status="skipped",
                          metrics={"error": str(exc)})
        return False  # non-fatal


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QTL MLOps Pipeline")
    parser.add_argument("--skip-source", action="store_true",
                        help="Skip source training (use existing checkpoint)")
    parser.add_argument("--skip-s3", action="store_true",
                        help="Skip S3 upload")
    parser.add_argument("--source-epochs", type=int, default=DEFAULT_SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=DEFAULT_TARGET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--strategies", nargs="+", default=QTL_STRATEGIES,
                        choices=QTL_STRATEGIES,
                        help="QTL strategies to run")
    parser.add_argument("--run-name", type=str, default="qtl_pipeline",
                        help="Name for this pipeline run")
    args = parser.parse_args()

    # Initialize tracker
    tracker = ExperimentTracker(run_name=args.run_name)
    logger.info("=" * 70)
    logger.info("QTL MLOps Pipeline — Run ID: %s", tracker.run_id)
    logger.info("=" * 70)

    overall_ok = True

    # Stage 1: Source training
    if not args.skip_source:
        if not run_source_training(tracker, args.source_epochs, args.batch_size):
            logger.error("Source training failed — aborting pipeline")
            tracker.finish(status="failed")
            return 1
    else:
        logger.info("Skipping source training (--skip-source)")
        if not SOURCE_CHECKPOINT.exists():
            logger.error("Source checkpoint not found: %s", SOURCE_CHECKPOINT)
            tracker.finish(status="failed")
            return 1
        tracker.start_stage("source_training", "Skipped (using existing checkpoint)")
        tracker.end_stage("source_training", status="skipped")

    # Stage 2: Weight extraction
    if not run_weight_extraction(tracker):
        logger.error("Weight extraction failed — aborting pipeline")
        tracker.finish(status="failed")
        return 1

    # Stage 3: QTL transfer
    if not run_qtl_transfer(tracker, args.target_epochs, args.batch_size,
                            args.strategies):
        logger.warning("QTL transfer had issues but continuing to report")
        overall_ok = False

    # Stage 4: S3 upload
    if not args.skip_s3:
        upload_to_s3(tracker)
    else:
        logger.info("Skipping S3 upload (--skip-s3)")

    # Generate report
    report = tracker.generate_report()
    print("\n" + report)

    tracker.finish(status="success" if overall_ok else "completed_with_warnings")
    logger.info("Pipeline complete. Run directory: %s", tracker.run_dir)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
