import boto3

S3_BUCKET_NAME = 'cicada-data'
s3_client = boto3.client('s3')