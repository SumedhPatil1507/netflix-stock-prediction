import os
import json
import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Create a boto3 S3 client using env vars. Supports custom endpoint for MinIO."""
    endpoint = os.getenv("AWS_S3_ENDPOINT_URL") or None
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        endpoint_url=endpoint,
    )
    return client


def upload_file(file_path: str, bucket: str, key: str) -> None:
    client = get_s3_client()
    client.upload_file(file_path, bucket, key)


def download_file(bucket: str, key: str, file_path: str) -> None:
    client = get_s3_client()
    client.download_file(bucket, key, file_path)


def upload_json(data: dict, bucket: str, key: str) -> None:
    client = get_s3_client()
    client.put_object(Body=json.dumps(data), Bucket=bucket, Key=key)


def download_json(bucket: str, key: str) -> dict:
    client = get_s3_client()
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"models": [], "latest": None}
        raise
