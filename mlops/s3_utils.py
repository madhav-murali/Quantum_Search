"""
AWS S3 utilities for uploading / downloading MLOps artifacts.

Usage:
    from mlops.s3_utils import S3Client
    client = S3Client()
    client.upload_artifact("checkpoints/model.pth", "runs/run123/model.pth")
"""

import os
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from mlops.config import S3_BUCKET, S3_PREFIX, AWS_REGION

logger = logging.getLogger(__name__)


class S3Client:
    """Thin wrapper around boto3 S3 operations for MLOps artifacts."""

    def __init__(self, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX,
                 region: str = AWS_REGION):
        self.bucket = bucket
        self.prefix = prefix
        self._client = None
        self._region = region

    # -- lazy init so import never fails even without creds ----------------
    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    # -- core operations ---------------------------------------------------
    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a single file to S3. Returns True on success."""
        full_key = self._full_key(s3_key)
        try:
            self.client.upload_file(str(local_path), self.bucket, full_key)
            logger.info("Uploaded %s → s3://%s/%s", local_path, self.bucket, full_key)
            return True
        except (ClientError, NoCredentialsError) as exc:
            logger.error("S3 upload failed for %s: %s", local_path, exc)
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a single file from S3. Returns True on success."""
        full_key = self._full_key(s3_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, full_key, str(local_path))
            logger.info("Downloaded s3://%s/%s → %s", self.bucket, full_key, local_path)
            return True
        except (ClientError, NoCredentialsError) as exc:
            logger.error("S3 download failed for %s: %s", s3_key, exc)
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str) -> int:
        """
        Recursively upload a directory to S3.

        Returns:
            Number of files successfully uploaded.
        """
        uploaded = 0
        local_dir = Path(local_dir)
        for path in local_dir.rglob("*"):
            if path.is_file():
                relative = path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative}"
                if self.upload_file(str(path), s3_key):
                    uploaded += 1
        return uploaded

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List object keys under the given prefix."""
        full_prefix = self._full_key(prefix)
        try:
            resp = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix=full_prefix
            )
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except (ClientError, NoCredentialsError) as exc:
            logger.error("S3 list failed: %s", exc)
            return []

    def artifact_exists(self, s3_key: str) -> bool:
        """Check whether an object exists in S3."""
        full_key = self._full_key(s3_key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False
