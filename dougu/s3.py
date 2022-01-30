from pathlib import Path

import boto3
from logzero import logger


class S3Client:
    """S3 client"""

    def __init__(self, bucket_name: str, access_key: str, secret_key: str) -> None:
        """
        Args:
            bucket_name: bucket name
            access_key: access key
            secret_key: secret key
        """
        self.__bucket_name = bucket_name
        self.__client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def download(
        self,
        remote_target_path: str,
        local_output_path: str,
    ) -> None:
        """
        Args:
            remote_target_path: Path to S3 to be downloaded
            local_output_path:  Path to local file to be saved
        """
        if Path(local_output_path).exists():
            logger.warning(f"already exists: {local_output_path}")
            logger.warning(
                f"skip downloading from {self.__bucket_name}:{remote_target_path}"
            )
        else:
            logger.debug(
                f"download: {self.__bucket_name}:{remote_target_path} -> {local_output_path}"
            )
            self.__client.download_file(
                self.__bucket_name,
                remote_target_path,
                local_output_path,
            )
            logger.debug("done")

    def upload(
        self,
        remote_save_path: str,
        local_target_path: str,
    ) -> None:
        """
        Args:
            remote_save_path:  Path to S3 to be saved
            local_target_path: Path to local file to be uploaded

        """
        if not Path(local_target_path).exists():
            raise ValueError(f"target file is not found: {local_target_path}")

        logger.info(
            f"upload: {local_target_path} -> {self.__bucket_name}:{remote_save_path}"
        )
        self.__client.upload_file(
            local_target_path, self.__bucket_name, remote_save_path
        )
        logger.debug("done")
