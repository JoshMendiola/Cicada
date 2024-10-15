from app import s3_client, S3_BUCKET_NAME


def download_from_s3(s3_path, local_path):
    s3_client.download_file(S3_BUCKET_NAME, s3_path, local_path)


def upload_to_s3(local_path, s3_path):
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_path)
