# aws_client.py
import boto3
import streamlit as st
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import logging

from config import BUCKET_NAME

logger = logging.getLogger(__name__)

def initialize_aws_clients():
    """
    Initialize and return Cognito, DynamoDB, and S3 clients 
    using credentials from st.secrets or environment variables.
    """
    try:
        cognito_client = boto3.client(
            'cognito-idp',
            region_name=st.secrets["COGNITO_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        s3_client = boto3.client(
            's3',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return cognito_client, dynamodb, s3_client
    except (NoCredentialsError, PartialCredentialsError) as e:
        st.error("AWS credentials are not configured properly.")
        logger.error(f"AWS Credentials Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during AWS client initialization: {e}")
        logger.error(f"AWS Client Initialization Error: {e}")
        st.stop()

def list_job_segments(s3_client, user_email, job_id, prefix_base="processed/"):
    """
    List segments for a given job in S3. 
    A segment is identified by a file ending with "_final_output.csv" containing "segment_".
    """
    prefix = f"{prefix_base}{user_email}/{job_id}/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith("_final_output.csv") and "segment_" in key:
                        seg_id = key.replace(prefix, "").replace("_final_output.csv", "")
                        segments.add(seg_id)
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing segments for job {job_id}: {e}")
        logger.error(f"Error listing segments: {e}")
        return []
