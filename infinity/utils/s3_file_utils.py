from smart_open import open
from io import BytesIO
import os
import boto3
from botocore.exceptions import ClientError
import torch
import yaml
from botocore.config import Config



def load_bytes_file(file_path: str, train_on_cluster : bool  = True):
    if not file_path.startswith("s3://"):
        assert os.path.exists(file_path), f"{file_path} cannot be found"
        with open(file_path, "rb") as f:
            data = f.read()
        return BytesIO(data)
    else:
        return load_s3_file(file_path)

def load_s3_file(s3_path: str) -> bytes:
    """
    Download a file from S3 given its S3 path (e.g., 's3://bucket/key').

    Args:
        s3_path (str): The S3 path.

    Returns:
        bytes: The downloaded file bytes.
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    # s3 = boto3.client("s3")
    # session = boto3.Session(
    #     aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
    #     aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    #     region_name=os.getenv("REGION"),
    # )

    config = Config(read_timeout=1200)

    # s3 = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"), config=config) 

    if 'chicago' in s3_path:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name='us-chicago-1',
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com", config=config) 
    else:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            
            region_name="us-phoenix-1",
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com", config=config) 

    response = s3.get_object(Bucket=bucket, Key=key)
    data_bytes = response["Body"].read()
    return BytesIO(data_bytes)
    
    # def download_streaming():
    #     response = s3.get_object(Bucket=bucket, Key=key)
    #     body = response["Body"]
    #     buf = BytesIO()
    #     while True:
    #         chunk = body.read(1024 * 1024 * 16)  # 16MB per chunk
    #         if not chunk:
    #             break
    #         buf.write(chunk)
    #     buf.seek(0)
    #     return buf
    # return download_streaming()

# def download_s3_folder(s3_path: str, local_dir: str):
#     """
#     Download all objects under an S3 folder (prefix) to a local directory.

#     Args:
#         s3_path (str): S3 folder path (e.g., 's3://my-bucket/my-folder/').
#         local_dir (str): Local directory where files will be saved.
#     """
#     assert s3_path.startswith("s3://"), "Not a valid S3 path"
    
#     # Remove 's3://' and split into bucket and prefix
#     s3_path = s3_path[5:]
#     bucket, prefix = s3_path.split("/", 1)

#     # Ensure local directory exists
#     os.makedirs(local_dir, exist_ok=True)

#     # Create S3 resource
#     s3 = boto3.resource("s3")
#     bucket_obj = s3.Bucket(bucket)

#     # Iterate over all objects with the prefix
#     for obj in bucket_obj.objects.filter(Prefix=prefix):
#         target_path = os.path.join(local_dir, obj.key[len(prefix):])
#         target_dir = os.path.dirname(target_path)
#         os.makedirs(target_dir, exist_ok=True)
#         if os.path.isfile(target_path):
#             continue

#         # Download the file
#         print(f"Downloading s3://{bucket}/{obj.key} to {target_path}", flush=True)
#         bucket_obj.download_file(obj.key, target_path)
#     return 
def download_s3_file(s3_path: str, local_folder: str, create_local_dir: bool = True):
    """
    Download a file from S3 to a local folder.
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)

    file_name = key.split("/")[-1]

    # Create local directory if it doesn't exist
    if create_local_dir:
        os.makedirs(local_folder, exist_ok=True)

    # Initialize S3 client, infra changed again
    # session = boto3.Session(
    #     aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
    #     aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    #     region_name=os.getenv("REGION"),
    # )
    # s3 = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL")) 
    if 'chicago' in s3_path:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name='us-chicago-1',
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com") 
    else:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name="us-phoenix-1",
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com") 



    local_file_path = os.path.join(local_folder, file_name)
    print(f"Downloading from {s3_path} to {local_file_path}", flush=True)

    # Download the file
    try:
        s3.download_file(bucket, key, local_file_path)
    except Exception as e:
        print(f"Error downloading from S3: {e}", flush=True)
        raise
    return local_file_path



def download_s3_folder(s3_path: str, local_path: str, create_local_dir: bool = True, skip_existing: bool = True) -> list:
    """
    Download an entire folder from S3 to a local directory, with option to skip existing files.

    Args:
        s3_path (str): Full S3 path to the folder (e.g., 's3://bucket/folder/')
        local_path (str): Local destination path where files will be downloaded
        create_local_dir (bool, optional): Create local directory if it doesn't exist. Defaults to True.
        skip_existing (bool, optional): Skip downloading files that already exist and match S3 metadata. Defaults to True.

    Returns:
        list: List of downloaded or existing file paths
    
    Raises:
        ValueError: If the S3 path is invalid
        ClientError: If there are issues accessing the S3 bucket
    """
    # Validate S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path. Must start with 's3://'")
    
    # Remove 's3://' and split into bucket and prefix
    s3_path_stripped = s3_path[5:]
    bucket, prefix = s3_path_stripped.split("/", 1)
    
    # Ensure prefix ends with a slash
    if not prefix.endswith('/'):
        prefix += '/'
    
    # Create local directory if it doesn't exist
    if create_local_dir:
        os.makedirs(local_path, exist_ok=True)
    
    # Initialize S3 client, infra changed again
    # s3 = boto3.client('s3')
    if 'chicago' in s3_path:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name='us-chicago-1',
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com") 
    else:
        session = boto3.Session(
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name="us-phoenix-1",
        )
        s3 = session.client("s3", endpoint_url="https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com") 



    # Store downloaded/existing file paths
    processed_files = []
    
    try:
        # List objects in the S3 folder
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            # Check if the folder is empty
            if 'Contents' not in page:
                print(f"No files found in the specified S3 path: {s3_path}")
                return []
            
            for obj in page['Contents']:
                # Skip if the object is the folder itself (ends with '/')
                if obj['Key'].endswith('/'):
                    continue
                
                # Construct local file path
                # Remove the prefix from the S3 key to maintain folder structure
                relative_path = obj['Key'][len(prefix):]
                local_file_path = os.path.join(local_path, relative_path)
                
                # Create local directories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Check if file exists and should be skipped
                if skip_existing and os.path.exists(local_file_path):
                    # Compare file sizes
                    local_file_size = os.path.getsize(local_file_path)
                    s3_file_size = obj['Size']
                    
                    if local_file_size == s3_file_size:
                        # Optional: Add MD5 check for more precise comparison
                        print(f"Skipping existing file: {local_file_path}", flush=True)
                        processed_files.append(local_file_path)
                        continue
                
                # Download the file
                s3.download_file(bucket, obj['Key'], local_file_path)
                processed_files.append(local_file_path)
                print(f"Downloaded: {obj['Key']} to {local_file_path}", flush=True)
        
        return processed_files
    
    except ClientError as e:
        print(f"Error downloading from S3: {e}", flush=True)
        raise


def save_state_dict_to_s3(state_dict, s3_path: str):
    """
    Save a PyTorch state dict to an S3 bucket.
    
    Args:
        state_dict: The PyTorch state dict to save
        s3_path (str): The S3 path (e.g., 's3://bucket/key.pth')
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    # Serialize the state dict
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Upload to S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
    )
    client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
    client.upload_fileobj(buffer, bucket, key)
    print(f"Successfully saved state dict to s3://{bucket}/{key}")

def save_yaml_to_s3(config_dict, s3_path: str):
    """
    Save an OrderedDict as a yaml file to an S3 bucket.
    
    Args:
        config_dict: The OrderedDict to save as yaml
        s3_path (str): The S3 path (e.g., 's3://bucket/key.yaml')
    """
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    # Convert the OrderedDict to YAML string
    yaml_str = yaml.dump(config_dict)
    
    # Create a BytesIO object
    buffer = BytesIO(yaml_str.encode('utf-8'))
    
    # Upload to S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
    )
    client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
    client.upload_fileobj(buffer, bucket, key)
    print(f"Successfully saved yaml to s3://{bucket}/{key}")