import os
from google.cloud import storage

def download_model(bucket, bucket_path, local_path):
    if not os.path.exists(local_path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket)
        model = bucket.blob(bucket_path)
        model.download_to_filename(local_path)


