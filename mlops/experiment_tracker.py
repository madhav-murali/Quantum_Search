"""
Lightweight experiment tracker for QTL pipeline runs.

Creates structured run directories, logs per-epoch metrics,
and uploads the run bundle to S3.
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from mlops.config import MLOPS_RUNS_DIR, QTL_STRATEGIES

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    """Return the short git SHA of the current HEAD, or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _git_branch() -> str:
    """Return the current git branch name, or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class ExperimentTracker:
    """
    Track a single pipeline run.

    Creates a directory structure:
        mlops_runs/<run_id>/
            manifest.json        — run metadata
            source_metrics.json  — source model training metrics
            qtl_metrics.json     — QTL transfer results per strategy
            artifacts/           — checkpoints, plots, etc.
    """

    def __init__(self, run_name: str = "qtl_pipeline"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sha = _git_sha()
        self.run_id = f"{ts}_{sha}_{run_name}"
        self.run_dir = MLOPS_RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)

        self.manifest = {
            "run_id": self.run_id,
            "run_name": run_name,
            "git_sha": sha,
            "git_branch": _git_branch(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
            "stages": {},
            "environment": {
                "python": os.popen("python --version 2>&1").read().strip(),
                "cuda_available": False,  # updated at runtime
            },
        }
        self._save_manifest()
        logger.info("Experiment run started: %s", self.run_id)

    # -- manifest ----------------------------------------------------------
    def _save_manifest(self):
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    def start_stage(self, stage: str, description: str = ""):
        """Record the start of a pipeline stage."""
        self.manifest["stages"][stage] = {
            "description": description,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "status": "running",
            "metrics": {},
        }
        self._save_manifest()
        logger.info("Stage started: %s", stage)

    def end_stage(self, stage: str, status: str = "success",
                  metrics: dict | None = None):
        """Record the end of a pipeline stage with optional metrics."""
        if stage in self.manifest["stages"]:
            s = self.manifest["stages"][stage]
            s["finished_at"] = datetime.now(timezone.utc).isoformat()
            s["status"] = status
            if metrics:
                s["metrics"] = metrics
        self._save_manifest()
        logger.info("Stage ended: %s — %s", stage, status)

    def finish(self, status: str = "success"):
        """Mark the entire run as finished."""
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["status"] = status
        self._save_manifest()
        logger.info("Run finished: %s — %s", self.run_id, status)

    # -- metrics logging ---------------------------------------------------
    def log_source_metrics(self, metrics: dict):
        """Save source model training metrics."""
        path = self.run_dir / "source_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Source metrics saved to %s", path)

    def log_qtl_metrics(self, strategy: str, metrics: dict):
        """Append QTL strategy metrics."""
        path = self.run_dir / "qtl_metrics.json"
        existing = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        existing[strategy] = metrics
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("QTL metrics for '%s' saved", strategy)

    def log_summary(self, summary: dict):
        """Save the final summary (comparison table, best strategy, etc.)."""
        path = self.run_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    # -- artifact management -----------------------------------------------
    def copy_artifact(self, src_path: str, artifact_name: str | None = None):
        """Copy a file into the run's artifacts/ directory."""
        import shutil

        src = Path(src_path)
        if not src.exists():
            logger.warning("Artifact source not found: %s", src)
            return None
        dest = self.run_dir / "artifacts" / (artifact_name or src.name)
        shutil.copy2(src, dest)
        logger.info("Artifact copied: %s → %s", src, dest)
        return str(dest)

    def generate_report(self) -> str:
        """
        Generate a markdown summary report from the run's metrics.

        Returns the report as a string and also saves it to the run directory.
        """
        lines = [
            f"# QTL Pipeline Run Report",
            f"",
            f"**Run ID**: `{self.run_id}`  ",
            f"**Git SHA**: `{self.manifest['git_sha']}`  ",
            f"**Branch**: `{self.manifest['git_branch']}`  ",
            f"**Status**: {self.manifest['status']}  ",
            f"**Started**: {self.manifest['started_at']}  ",
            f"**Finished**: {self.manifest.get('finished_at', 'N/A')}",
            "",
            "## Pipeline Stages",
            "",
        ]

        for name, stage in self.manifest.get("stages", {}).items():
            emoji = "✅" if stage["status"] == "success" else "❌"
            lines.append(f"- {emoji} **{name}**: {stage.get('description', '')}")
            if stage.get("metrics"):
                for k, v in stage["metrics"].items():
                    if isinstance(v, float):
                        lines.append(f"  - {k}: {v:.4f}")
                    else:
                        lines.append(f"  - {k}: {v}")

        # QTL comparison table
        qtl_path = self.run_dir / "qtl_metrics.json"
        if qtl_path.exists():
            with open(qtl_path) as f:
                qtl = json.load(f)
            lines.extend([
                "",
                "## QTL Strategy Comparison",
                "",
                "| Strategy | Best Acc | Best F1 | Final Loss |",
                "|----------|----------|---------|------------|",
            ])
            for strat, m in qtl.items():
                acc = max(m.get("val_acc", [0]))
                f1 = max(m.get("val_f1", [0]))
                loss = m.get("loss", [0])[-1] if m.get("loss") else 0
                lines.append(f"| {strat} | {acc:.4f} | {f1:.4f} | {loss:.4f} |")

        report = "\n".join(lines) + "\n"
        report_path = self.run_dir / "run_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("Report generated: %s", report_path)
        return report
