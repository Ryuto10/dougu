from google.auth import exceptions
from google.cloud import storage
from logzero import logger
from os import path


def upload_to_gcs(project: str, bucket_name: str, blob_name: str, file_name: str) -> None:
    """google cloud strageにファイルをアップロードします

    Args:
        project (str): projectを指定してください (e.g. os.getenv("PROJECT"))
        bucket_name (str): bucketを指定してください
        blob_name (str): blobの名前を指定してください
        file_name (str): uploadするファイルの名前を指定してください
    """
    assert path.exists(file_name), f"Not found: {file_name}"

    client = storage.Client(project=project)
    try:
        bucket = client.get_bucket(bucket_name)
    except exceptions.NotFound:
        logger.warning(f"Sorry, that bucket does not exist: {bucket_name}")

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_name)

    logger.info(f"Upload: {file_name} -> {blob_name}")


def download_from_gcs(project: str, bucket_name: str, blob_name: str, file_name: str) -> None:
    """google cloud strageからファイルをダウンロードします

    Args:
        project (str): projectを指定してください (e.g. os.getenv("PROJECT"))
        bucket_name (str): bucketを指定してください
        blob_name (str): blobの名前を指定してください
        file_name (str): downloadしたファイルを保存するための名前を指定してください
    """
    assert not path.exists(file_name), f"Already exists: {file_name}"

    client = storage.Client(project=project)
    try:
        bucket = client.get_bucket(bucket_name)
    except exceptions.NotFound:
        logger.warning(f"Sorry, that bucket does not exist: {bucket_name}")

    blob = bucket.blob(blob_name)
    blob.download_to_filename(file_name)

    logger.info(f"Upload: {blob_name} -> {file_name}")
