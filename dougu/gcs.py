from os import path
from pathlib import Path

from google.auth import exceptions
from google.cloud import storage
from logzero import logger


def upload_to_gcs(
    project: str, bucket_name: str, blob_name: str, file_path: str
) -> None:
    """Upload the file to google cloud strage
    Args:
        project:     project name
        bucket_name: bucket name
        blob_name:   blob name
        file_path:   Path to file to upload
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"not found: {file_path}")

    client = storage.Client(project=project)

    try:
        bucket = client.get_bucket(bucket_name)

        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        logger.info(f"upload: {file_path} -> {blob_name}")

    except exceptions.NotFound:
        logger.warning(f"sorry, that bucket does not exist: {bucket_name}")


def download_from_gcs(
    project: str, bucket_name: str, blob_name: str, out_path: str
) -> None:
    """Download the file from google cloud storage
    Args:
        project:     project name
        bucket_name: bucket name
        blob_name:   blob name
        out_path:   Path to output file

        project (str): project name
        bucket_name (str): bucket name
        blob_name (str): blob name
        out_path (str): Path to file to be save
    """
    assert not path.exists(out_path), f"Already exists: {out_path}"

    client = storage.Client(project=project)

    try:
        bucket = client.get_bucket(bucket_name)

        blob = bucket.blob(blob_name)
        blob.download_to_filename(out_path)
        logger.debug(f"Upload: {blob_name} -> {out_path}")
    except exceptions.NotFound:
        logger.warning(f"Sorry, that bucket does not exist: {bucket_name}")
