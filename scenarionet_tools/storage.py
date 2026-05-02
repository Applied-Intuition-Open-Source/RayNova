import os
from typing import TypedDict


class S3StorageOptions(TypedDict):
    aws_endpoint: str
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str


def make_s3_storage_options_from_env() -> S3StorageOptions:
    """Make parameters for lance to authenticate with S3 compatible storage APIs.

    Documentation: https://lancedb.github.io/lance/read_and_write.html#object-store-configuration

    """

    aws_endpoint = os.getenv("AWS_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    assert aws_endpoint is not None, "AWS_ENDPOINT_URL is not set"
    assert aws_access_key_id is not None, "AWS_ACCESS_KEY_ID is not set"
    assert aws_secret_access_key is not None, "AWS_SECRET_ACCESS_KEY is not set"
    assert aws_region is not None, "AWS_DEFAULT_REGION is not set"
    return {
        "aws_endpoint": aws_endpoint,
        "aws_region": aws_region,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }