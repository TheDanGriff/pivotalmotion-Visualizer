import streamlit as st 
import boto3
from boto3.dynamodb.conditions import Attr
import pandas as pd
import numpy as np
import os
import logging
import base64
import hashlib
import hmac
from math import degrees, atan2
from io import StringIO
import statsmodels.api as sm
import plotly.graph_objects as go
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv
import plotly.express as px
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -------------------------------------------------------------------------------------
# Configure Logging
# -------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)  # Set to INFO for more detailed logs
logger = logging.getLogger()

# -------------------------------------------------------------------------------------
# Load Environment Variables Securely
# -------------------------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------------------------
# Define the TEAMS Dictionary
# -------------------------------------------------------------------------------------
TEAMS = {
    'hawks': {'full_name': 'Atlanta Hawks', 'logo': 'hawks_logo.png'},
    'nets': {'full_name': 'Brooklyn Nets', 'logo': 'nets_logo.png'},
    'celtics': {'full_name': 'Boston Celtics', 'logo': 'celtics_logo.png'},
    'hornets': {'full_name': 'Charlotte Hornets', 'logo': 'hornets_logo.png'},
    'bulls': {'full_name': 'Chicago Bulls', 'logo': 'bulls_logo.png'},
    'cavaliers': {'full_name': 'Cleveland Cavaliers', 'logo': 'cavaliers_logo.png'},
    'mavericks': {'full_name': 'Dallas Mavericks', 'logo': 'mavericks_logo.png'},
    'nuggets': {'full_name': 'Denver Nuggets', 'logo': 'nuggets_logo.png'},
    'pistons': {'full_name': 'Detroit Pistons', 'logo': 'pistons_logo.png'},
    'warriors': {'full_name': 'Golden State Warriors', 'logo': 'warriors_logo.png'},
    'rockets': {'full_name': 'Houston Rockets', 'logo': 'rockets_logo.png'},
    'pacers': {'full_name': 'Indiana Pacers', 'logo': 'pacers_logo.png'},
    'clippers': {'full_name': 'Los Angeles Clippers', 'logo': 'clippers_logo.png'},
    'lakers': {'full_name': 'Los Angeles Lakers', 'logo': 'lakers_logo.png'},
    'grizzlies': {'full_name': 'Memphis Grizzlies', 'logo': 'grizzlies_logo.png'},
    'heat': {'full_name': 'Miami Heat', 'logo': 'heat_logo.png'},
    'bucks': {'full_name': 'Milwaukee Bucks', 'logo': 'bucks_logo.png'},
    'timberwolves': {'full_name': 'Minnesota Timberwolves', 'logo': 'timberwolves_logo.png'},
    'pelicans': {'full_name': 'New Orleans Pelicans', 'logo': 'pelicans_logo.png'},
    'knicks': {'full_name': 'New York Knicks', 'logo': 'knicks_logo.png'},
    'thunder': {'full_name': 'Oklahoma City Thunder', 'logo': 'thunder_logo.png'},
    'magic': {'full_name': 'Orlando Magic', 'logo': 'magic_logo.png'},
    '76ers': {'full_name': 'Philadelphia 76ers', 'logo': '76ers_logo.png'},
    'suns': {'full_name': 'Phoenix Suns', 'logo': 'suns_logo.png'},
    'blazers': {'full_name': 'Portland Trail Blazers', 'logo': 'blazers_logo.png'},
    'kings': {'full_name': 'Sacramento Kings', 'logo': 'kings_logo.png'},
    'spurs': {'full_name': 'San Antonio Spurs', 'logo': 'spurs_logo.png'},
    'raptors': {'full_name': 'Toronto Raptors', 'logo': 'raptors_logo.png'},
    'jazz': {'full_name': 'Utah Jazz', 'logo': 'jazz_logo.png'},
    'wizards': {'full_name': 'Washington Wizards', 'logo': 'wizards_logo.png'}
}

# -------------------------------------------------------------------------------------
# Define DynamoDB Table Names as Constants
# -------------------------------------------------------------------------------------
DYNAMODB_TABLE_USERS = 'StreamlitUsers'      # DynamoDB table for user data
DYNAMODB_TABLE_DATA = 'ProcessingQueue'      # DynamoDB table for job data

# -------------------------------------------------------------------------------------
# Initialize AWS Clients
# -------------------------------------------------------------------------------------
def initialize_aws_clients():
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

cognito_client, dynamodb, s3_client = initialize_aws_clients()

# -------------------------------------------------------------------------------------
# Compute SECRET_HASH (for Cognito authentication)
# -------------------------------------------------------------------------------------
def get_secret_hash(username, client_id, client_secret):
    message = username + client_id
    dig = hmac.new(
        client_secret.encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

# -------------------------------------------------------------------------------------
# Cognito Authentication
# -------------------------------------------------------------------------------------
def authenticate_user(username, password):
    secret_hash = get_secret_hash(username, st.secrets["COGNITO_CLIENT_ID"], st.secrets["COGNITO_CLIENT_SECRET"])
    try:
        response = cognito_client.initiate_auth(
            ClientId=st.secrets["COGNITO_CLIENT_ID"],
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
                'SECRET_HASH': secret_hash
            }
        )
        return response['AuthenticationResult']
    except cognito_client.exceptions.NotAuthorizedException:
        st.error("Incorrect username or password.")
    except cognito_client.exceptions.UserNotFoundException:
        st.error("User does not exist.")
    except Exception as e:
        st.error(f"An error occurred during authentication: {e}")
        logger.error(f"Authentication Error: {e}")
    return None

# -------------------------------------------------------------------------------------
# Fetch Username from DynamoDB by Email
# -------------------------------------------------------------------------------------
def get_username_by_email(email):
    table = dynamodb.Table(DYNAMODB_TABLE_USERS)
    try:
        response = table.scan(FilterExpression=Attr('Email').eq(email))
        items = response.get('Items', [])
        if len(items) == 1:
            return items[0]['Username']
        elif len(items) > 1:
            st.error("Multiple users found with the same email. Contact support.")
            logger.error(f"Multiple users found for email: {email}")
            return None
        else:
            st.error("Email not found. Please register first.")
            logger.error(f"Email not found: {email}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching user data: {e}")
        logger.error(f"Error in get_username_by_email: {e}")
        return None

# -------------------------------------------------------------------------------------
# Get Metadata from DynamoDB
# -------------------------------------------------------------------------------------
def get_metadata_from_dynamodb(job_id):
    table = dynamodb.Table(DYNAMODB_TABLE_DATA)
    try:
        response = table.get_item(Key={'JobID': job_id})
        if 'Item' in response:
            logger.info(f"Metadata found for Job ID: {job_id}")
            return response['Item']
        else:
            logger.warning(f"No metadata found in DynamoDB for Job ID: {job_id}")
            return {}
    except ClientError as e:
        st.error(f"DynamoDB error: {e.response['Error']['Message']}")
        logger.error(f"DynamoDB error in get_metadata_from_dynamodb: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error retrieving metadata: {e}")
        logger.error(f"Unexpected error in get_metadata_from_dynamodb: {e}")
        return {}

# -------------------------------------------------------------------------------------
# Fetch All Completed Non-data_file Jobs (Pose/Spin)
# -------------------------------------------------------------------------------------
def fetch_user_completed_jobs(user_email):
    """
    Fetch all completed jobs for the user that are NOT of the source type 'data_file'.
    """
    try:
        prefix = f"processed/{user_email}/"
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=st.secrets["BUCKET_NAME"], Prefix=prefix, Delimiter='/')
        jobs = []
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Processing page {page_number} of S3 job listings.")
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    job_id = cp['Prefix'].replace(prefix, '').strip('/')
                    if not job_id:
                        continue

                    # Check if there is a data_file folder; if so, skip this job here.
                    response = s3_client.list_objects_v2(
                        Bucket=st.secrets["BUCKET_NAME"],
                        Prefix=f"{cp['Prefix']}data_file/",
                        MaxKeys=1
                    )
                    if 'Contents' in response:
                        continue  # This job will be handled in the data_file branch.

                    # Retrieve metadata from DynamoDB
                    job_meta = get_metadata_from_dynamodb(job_id)
                    if not job_meta:
                        continue
                    # For pose/spin jobs, we expect the status to be 'completed'
                    if job_meta.get('Status', '').lower() != 'completed':
                        logger.info(f"Job ID {job_id} is not completed. Status: {job_meta.get('Status', 'Unknown')}")
                        continue
                    jobs.append({
                        'JobID': job_id,
                        'PlayerName': job_meta.get('PlayerName', 'Unknown Player'),
                        'Team': job_meta.get('Team', 'N/A'),
                        'ShootingType': job_meta.get('ShootingType', 'N/A'),
                        'UploadTimestamp': job_meta.get('UploadTimestamp', None),
                        'Source': job_meta.get('Source', 'Unknown')  # expected: 'pose_video' or 'spin_video'
                    })
            else:
                logger.warning(f"No CommonPrefixes found on page {page_number}.")
        logger.info(f"Total completed non-data_file jobs found: {len(jobs)}")
        return jobs
    except Exception as e:
        st.error(f"Error fetching user jobs: {e}")
        logger.error(f"Error in fetch_user_completed_jobs: {e}")
        return []

# -------------------------------------------------------------------------------------
# Fetch All Completed Data_file Jobs
# -------------------------------------------------------------------------------------
def fetch_user_completed_data_file_jobs(user_email):
    """
    Fetch all jobs for the user that are of the source type 'data_file'.
    In your DynamoDB, these jobs may have a Status of 'Processed' (or 'Completed'),
    so we check the Source field rather than solely relying on status.
    """
    try:
        prefix = f"processed/{user_email}/"
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=st.secrets["BUCKET_NAME"], Prefix=prefix, Delimiter='/')
        jobs = []
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Processing page {page_number} for data_file jobs.")
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    job_id = cp['Prefix'].replace(prefix, '').strip('/')
                    if not job_id:
                        continue

                    # Look for a data_file subfolder in this job folder.
                    data_file_prefix = f"{cp['Prefix']}data_file/"
                    response = s3_client.list_objects_v2(
                        Bucket=st.secrets["BUCKET_NAME"],
                        Prefix=data_file_prefix,
                        MaxKeys=1
                    )
                    if 'Contents' not in response:
                        continue  # No data_file folder; skip.

                    # Retrieve metadata from DynamoDB
                    job_meta = get_metadata_from_dynamodb(job_id)
                    if not job_meta:
                        continue
                    # Check if the job Source is data_file (ignoring case)
                    if job_meta.get('Source', '').lower() != 'data_file':
                        continue

                    # Accept either 'processed' or 'completed' as valid statuses for data_file jobs.
                    if job_meta.get('Status', '').lower() not in ['processed', 'completed']:
                        logger.info(f"Data_file Job ID {job_id} does not have a valid status. Status: {job_meta.get('Status', 'Unknown')}")
                        continue

                    jobs.append({
                        'JobID': job_id,
                        'PlayerName': job_meta.get('PlayerName', 'Unknown Player'),
                        'Team': job_meta.get('Team', 'N/A'),
                        'ShootingType': job_meta.get('ShootingType', 'N/A'),
                        'UploadTimestamp': job_meta.get('UploadTimestamp', None),
                        'Source': job_meta.get('Source', 'Unknown')  # expected: 'data_file'
                    })
            else:
                logger.warning(f"No CommonPrefixes found on page {page_number} for data_file jobs.")
        logger.info(f"Total completed data_file jobs found: {len(jobs)}")
        return jobs
    except Exception as e:
        st.error(f"Error fetching data_file jobs: {e}")
        logger.error(f"Error in fetch_user_completed_data_file_jobs: {e}")
        return []

# -------------------------------------------------------------------------------------
# List Job Segments for Pose/Spin Jobs
# -------------------------------------------------------------------------------------
def list_job_segments(s3_client, bucket_name, user_email, job_id):
    prefix = f"processed/{user_email}/{job_id}/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Processing page {page_number} for segments.")
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    if filename.startswith("segment_") and filename.endswith("_final_output.csv"):
                        seg_id = filename.replace("_final_output.csv", "")
                        segments.add(seg_id)
            else:
                logger.warning(f"No Contents found on page {page_number}.")
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing segments for job '{job_id}': {e}")
        logger.error(f"Error listing segments for job '{job_id}': {e}")
        return []

# -------------------------------------------------------------------------------------
# List Segments for Data_file Jobs
# -------------------------------------------------------------------------------------
def list_data_file_job_segments(bucket_name, user_email, job_id):
    prefix = f"processed/{user_email}/{job_id}/data_file/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Processing page {page_number} for data_file segments.")
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    if filename.startswith("segment_") and filename.endswith("_final_output.csv"):
                        seg_id = filename.replace("_final_output.csv", "")
                        segments.add(seg_id)
            else:
                logger.warning(f"No Contents found on page {page_number} for data_file segments.")
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing segments for data_file job '{job_id}': {e}")
        logger.error(f"Error listing segments for data_file job '{job_id}': {e}")
        return []

# -------------------------------------------------------------------------------------
# Load final_output.csv for Pose/Spin Jobs
# -------------------------------------------------------------------------------------
def load_final_output_csv(s3_client, bucket_name, user_email, job_id, segment_id):
    final_output_key = f"processed/{user_email}/{job_id}/{segment_id}_final_output.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=final_output_key)
        csv_data = response['Body'].read().decode('utf-8')
        df_final = pd.read_csv(StringIO(csv_data))
        return df_final
    except s3_client.exceptions.NoSuchKey:
        st.error(f"final_output.csv not found for Job ID: {job_id}, Segment ID: {segment_id}.")
        logger.warning(f"final_output.csv not found for Job ID: {job_id}, Segment ID: {segment_id} at key: {final_output_key}")
    except Exception as e:
        st.error(f"An error occurred while fetching final_output.csv for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
        logger.error(f"Error in load_final_output_csv for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
    return pd.DataFrame()

# -------------------------------------------------------------------------------------
# Load spin_axis.csv for Spin Jobs
# -------------------------------------------------------------------------------------
def load_spin_axis_csv(s3_client, bucket_name, user_email, job_id):
    spin_axis_key = f"processed/{user_email}/{job_id}/spin_axis.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=spin_axis_key)
        csv_data = response['Body'].read().decode('utf-8')
        df_spin = pd.read_csv(StringIO(csv_data))
        logger.info(f"spin_axis.csv loaded for Job ID: {job_id}")
        return df_spin
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"spin_axis.csv not found for Job ID: {job_id}")
    except Exception as e:
        st.error(f"An error occurred while fetching spin_axis.csv: {e}")
        logger.error(f"Error in load_spin_axis_csv: {e}")
    return pd.DataFrame()

# -------------------------------------------------------------------------------------
# Load final_output.csv for Data_file Jobs
# -------------------------------------------------------------------------------------
def load_data_file_final_output_csv(bucket_name, user_email, job_id, segment_id):
    final_output_key = f"processed/{user_email}/{job_id}/data_file/{segment_id}_final_output.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=final_output_key)
        csv_data = response['Body'].read().decode('utf-8')
        df_final = pd.read_csv(StringIO(csv_data))
        logger.info(f"Loaded data_file final_output.csv for Job ID: {job_id}, Segment ID: {segment_id}")
        return df_final
    except s3_client.exceptions.NoSuchKey:
        st.error(f"final_output.csv not found for Job ID: {job_id}, Segment ID: {segment_id}.")
        logger.warning(f"final_output.csv not found at key: {final_output_key}")
    except Exception as e:
        st.error(f"An error occurred while fetching final_output.csv for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
        logger.error(f"Error in load_data_file_final_output_csv for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
    return pd.DataFrame()

# -------------------------------------------------------------------------------------
# Example of Combining and Displaying All Jobs in the App
# -------------------------------------------------------------------------------------
def display_all_jobs(user_email):
    # Fetch jobs from both branches.
    regular_jobs = fetch_user_completed_jobs(user_email)
    data_file_jobs = fetch_user_completed_data_file_jobs(user_email)
    
    # Combine both lists.
    all_jobs = regular_jobs + data_file_jobs

    for job in all_jobs:
        st.write(f"**Job ID:** {job['JobID']} | **Source:** {job['Source']} | **Player:** {job['PlayerName']}")
        if job['Source'].lower() == 'data_file':
            segments = list_data_file_job_segments(st.secrets["BUCKET_NAME"], user_email, job['JobID'])
            st.write("Data_file Segments:", segments)
            if segments:
                df = load_data_file_final_output_csv(st.secrets["BUCKET_NAME"], user_email, job['JobID'], segments[0])
                st.dataframe(df)
        elif job['Source'].lower() in ['pose_video', 'spin_video']:
            segments = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], user_email, job['JobID'])
            st.write("Segments:", segments)
            if segments:
                # For spin videos you might load spin_axis.csv instead.
                if job['Source'].lower() == 'spin_video':
                    df = load_spin_axis_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, job['JobID'])
                else:
                    df = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, job['JobID'], segments[0])
                st.dataframe(df)
        else:
            st.write("Unknown job source.")
# -------------------------------------------------------------------------------------
# Humanize Label Utility Function
# -------------------------------------------------------------------------------------
def humanize_label(label):
    """Convert 'snake_case' or 'camelCase' into Title-case, no underscores."""
    if not label or pd.isna(label):
        return "N/A"
    return ' '.join(word.capitalize() for word in label.replace('_',' ').split())

# -------------------------------------------------------------------------------------
# Show Brand Header with Team Logo
# -------------------------------------------------------------------------------------
def show_brand_header(player_names, team_abbreviation):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        players_display = ", ".join(player_names) if isinstance(player_names, list) else player_names
        st.markdown(
            f"<h1 style='text-align: center; color: #000000;'>{players_display or 'Unknown Player'}</h1>",
            unsafe_allow_html=True
        )
        if team_abbreviation and team_abbreviation.strip().lower() != "n/a":
            team_info = TEAMS.get(team_abbreviation.lower(), {'full_name': team_abbreviation, 'logo': None})
            team_full_name = team_info['full_name']
            st.markdown(
                f"<h3 style='text-align: center; color: #000000;'>{team_full_name}</h3>",
                unsafe_allow_html=True
            )
            team_logo_filename = team_info['logo']
            if team_logo_filename:
                team_logo_path = os.path.join("images", "teams", team_logo_filename)
                team_logo = load_image_from_file(team_logo_path)
                if team_logo:
                    encoded_logo = base64.b64encode(team_logo).decode()
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{encoded_logo}" style="max-width: 150px;"/>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

def load_data_file_shot_csv(bucket_name, user_email, job_id):
    """
    Loads the single-shot CSV for data_file jobs.
    This function now expects the CSV at:
        processed/{user_email}/{job_id}/data_file/segment_001_final_output.csv
    """
    shot_key = f"processed/{user_email}/{job_id}/data_file/segment_001_final_output.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=shot_key)
        csv_data = response['Body'].read().decode('utf-8')
        # Adjust parameters as needed (delimiter, etc.)
        df_shot = pd.read_csv(StringIO(csv_data))
        logger.info(f"Data_file shot CSV loaded for Job ID: {job_id}")
        return df_shot
    except s3_client.exceptions.NoSuchKey:
        st.error(f"Data file not found for Job ID: {job_id} at key: {shot_key}")
        logger.warning(f"Data file key not found: {shot_key}")
    except Exception as e:
        st.error(f"Error loading data file CSV for Job ID {job_id}: {e}")
        logger.error(f"Error loading data file CSV for Job ID {job_id}: {e}")
    return pd.DataFrame()


def load_image_from_file(filepath):
    """Loads and encodes image for display in Streamlit."""
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"Image file not found: {filepath}")
        logger.warning(f"Image file not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading image file {filepath}: {e}")
        logger.error(f"Error loading image file {filepath}: {e}")
        return None

# -------------------------------------------------------------------------------------
# Handle Login
# -------------------------------------------------------------------------------------
def handle_login(email, password):
    if email and password:
        username = get_username_by_email(email)
        if username:
            auth_result = authenticate_user(username, password)
            if auth_result:
                st.session_state['authenticated'] = True
                st.session_state['access_token'] = auth_result['AccessToken']
                st.session_state['username'] = username
                st.session_state['user_email'] = email
                st.success("Successfully logged in!")
            else:
                st.error("Authentication failed. Please check your credentials.")
        else:
            st.error("Username retrieval failed.")
    else:
        st.error("Please enter both email and password.")

# -------------------------------------------------------------------------------------
# Data Cleaning, Angles, and Trimming
# -------------------------------------------------------------------------------------
def map_body_columns(df):
    # Remove any extra whitespace from column names
    df.columns = df.columns.str.strip()
    
    rename_dict = {
        'LEYE_X': 'left_eye_x',
        'LEYE_Y': 'left_eye_y',
        'LEYE_Z': 'left_eye_z',
        'REYE_X': 'right_eye_x',
        'REYE_Y': 'right_eye_y',
        'REYE_Z': 'right_eye_z',
        'NECK_X': 'neck_x',
        'NECK_Y': 'neck_y',
        'NECK_Z': 'neck_z',
        'LSJC_X': 'left_shoulder_x',
        'LSJC_Y': 'left_shoulder_y',
        'LSJC_Z': 'left_shoulder_z',
        'LEJC_X': 'left_elbow_x',
        'LEJC_Y': 'left_elbow_y',
        'LEJC_Z': 'left_elbow_z',
        'LWJC_X': 'left_wrist_x',
        'LWJC_Y': 'left_wrist_y',
        'LWJC_Z': 'left_wrist_z',
        'LPINKY_X': 'left_pinky_x',
        'LPINKY_Y': 'left_pinky_y',
        'LPINKY_Z': 'left_pinky_z',
        'LTHUMB_X': 'left_thumb_x',
        'LTHUMB_Y': 'left_thumb_y',
        'LTHUMB_Z': 'left_thumb_z',
        'RSJC_X': 'right_shoulder_x',
        'RSJC_Y': 'right_shoulder_y',
        'RSJC_Z': 'right_shoulder_z',
        'REJC_X': 'right_elbow_x',
        'REJC_Y': 'right_elbow_y',
        'REJC_Z': 'right_elbow_z',
        'RWJC_X': 'right_wrist_x',
        'RWJC_Y': 'right_wrist_y',
        'RWJC_Z': 'right_wrist_z',
        'RPINKY_X': 'right_pinky_x',
        'RPINKY_Y': 'right_pinky_y',
        'RPINKY_Z': 'right_pinky_z',
        'RTHUMB_X': 'right_thumb_x',
        'RTHUMB_Y': 'right_thumb_y',
        'RTHUMB_Z': 'right_thumb_z',
        'MIDHIP_X': 'mid_hip_x',
        'MIDHIP_Y': 'mid_hip_y',
        'MIDHIP_Z': 'mid_hip_z',
        'LHJC_X': 'left_hip_x',
        'LHJC_Y': 'left_hip_y',
        'LHJC_Z': 'left_hip_z',
        'LKJC_X': 'left_knee_x',
        'LKJC_Y': 'left_knee_y',
        'LKJC_Z': 'left_knee_z',
        'LAJC_X': 'left_ankle_x',
        'LAJC_Y': 'left_ankle_y',
        'LAJC_Z': 'left_ankle_z',
        'RHJC_X': 'right_hip_x',
        'RHJC_Y': 'right_hip_y',
        'RHJC_Z': 'right_hip_z',
        'RJKC_X': 'right_knee_x',
        'RJKC_Y': 'right_knee_y',
        'RJKC_Z': 'right_knee_z',
        'RAJC_X': 'right_ankle_x',
        'RAJC_Y': 'right_ankle_y',
        'RAJC_Z': 'right_ankle_z',
    }
    df = df.rename(columns=rename_dict)
    return df


def angle_2d(a, b, c):
    """Compute angle at b with points a,b,c in 2D => degrees."""
    try:
        ab = np.array([a[0] - b[0], a[1] - b[1]])
        cb = np.array([c[0] - b[0], c[1] - b[1]])
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        if ab_norm == 0 or cb_norm == 0:
            return np.nan
        cos_val = np.dot(ab, cb) / (ab_norm * cb_norm)
        return degrees(np.arccos(np.clip(cos_val, -1, 1)))
    except:
        return np.nan

def compute_joint_angles(df):
    joints = {
        'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
        'left_shoulder_angle': ('neck', 'left_shoulder', 'left_elbow'),
        'right_shoulder_angle': ('neck', 'right_shoulder', 'right_elbow'),
        'left_hip_angle': ('mid_hip', 'left_hip', 'left_knee'),
        'right_hip_angle': ('mid_hip', 'right_hip', 'right_knee'),
        'left_ankle_angle': ('left_knee', 'left_ankle', 'left_foot'),  # Assuming 'left_foot' exists
        'right_ankle_angle': ('right_knee', 'right_ankle', 'right_foot'),  # Assuming 'right_foot' exists
        # Add more joints as needed
    }
    for angle, (a, b, c) in joints.items():
        a_x, a_y = f"{a}_x", f"{a}_y"
        b_x, b_y = f"{b}_x", f"{b}_y"
        c_x, c_y = f"{c}_x", f"{c}_y"
        if all(col in df.columns for col in [a_x, a_y, b_x, b_y, c_x, c_y]):
            df[angle] = df.apply(lambda row: angle_2d(
                (row[a_x], row[a_y]),
                (row[b_x], row[b_y]),
                (row[c_x], row[c_y])
            ), axis=1)
    return df

def detect_release_frame_pose(df):
    """
    Detect the release frame based on the highest y-coordinate in elbow positions.
    """
    if 'left_elbow_y' not in df.columns or 'right_elbow_y' not in df.columns:
        st.warning("No elbow y-coordinate data found for release detection.")
        logger.warning("No elbow y-coordinate data found for release detection.")
        return None
    
    # Find the frame with the highest y-coordinate in either elbow
    df['max_elbow_y'] = df[['left_elbow_y', 'right_elbow_y']].max(axis=1)
    max_y_idx = df['max_elbow_y'].idxmax()
    
    if 'frame' in df.columns:
        release_frame = df.loc[max_y_idx, 'frame']
        logger.info(f"Release frame detected at frame {release_frame} with max elbow y {df.loc[max_y_idx, 'max_elbow_y']}")
        return release_frame
    return None

def trim_shot(df, release_frame, frames_before=20, frames_after=5):
    if release_frame is None or 'frame' not in df.columns:
        return None
    start = release_frame - frames_before
    end = release_frame + frames_after
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()
    start = max(start, min_frame)
    end = min(end, max_frame)
    trimmed_df = df[(df['frame'] >= start) & (df['frame'] <= end)].copy()
    
    # Ensure 'relative_frame' is computed
    trimmed_df = add_relative_frame(trimmed_df, release_frame)
    
    # Drop duplicates for pose (assuming one pose per frame)
    trimmed_df = trimmed_df.sort_values('frame').drop_duplicates(subset=['frame'], keep='first')
    expected_frames = frames_before + frames_after + 1
    actual_frames = trimmed_df['frame'].nunique()
    if actual_frames < expected_frames:
        frame_range = np.arange(start, end + 1)
        trimmed_df = trimmed_df.set_index('frame').reindex(frame_range).reset_index()
        trimmed_df.rename(columns={'index': 'frame'}, inplace=True)
    
    return trimmed_df

def add_relative_frame(df, release_frame):
    """ 
    Add 'relative_frame' column based on the release frame.
    If release_frame is None, set relative_frame relative to the first frame.
    """
    if 'frame' not in df.columns:
        st.warning("No 'frame' column present in DataFrame.")
        return df

    if release_frame is not None:
        df['relative_frame'] = df['frame'] - release_frame
    else:
        df['relative_frame'] = df['frame'] - df['frame'].min()
        st.warning("Release frame not detected. 'relative_frame' set relative to the first frame.")
    
    return df

def separate_pose_and_ball_tracking(df_segment):
    """
    Separates pose and ball tracking data from the combined DataFrame.

    Parameters:
    - df_segment (pd.DataFrame): Combined DataFrame containing both pose and detection data.

    Returns:
    - tuple: (pose_df, ball_tracking_df)
    """
    # Pose data: all columns except detection-specific ones
    detection_columns = ['track_id', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    pose_columns = [col for col in df_segment.columns if col not in detection_columns]
    pose_df = df_segment[pose_columns].copy()
    
    # Ensure one pose per frame by selecting the first one per frame
    if 'frame' in df_segment.columns:
        pose_df['frame'] = df_segment['frame']
        pose_df = pose_df.groupby('frame').first().reset_index()
    else:
        pose_df = pd.DataFrame()
    
    # Ball tracking data: filter for class_name == 'basketball'
    if 'class_name' in df_segment.columns:
        ball_tracking_df = df_segment[df_segment['class_name'].str.lower() == 'basketball'].copy()
        # Ensure one ball per frame by selecting the highest confidence if multiple
        if not ball_tracking_df.empty:
            ball_tracking_df = ball_tracking_df.sort_values('confidence', ascending=False).drop_duplicates(subset=['frame'], keep='first')
    else:
        ball_tracking_df = pd.DataFrame()
    
    return pose_df, ball_tracking_df

def clean_pose_data(pose_df):
    """
    Cleans the pose DataFrame by renaming columns and computing joint angles.

    Parameters:
    - pose_df (pd.DataFrame): Raw pose DataFrame.

    Returns:
    - pd.DataFrame: Cleaned pose DataFrame with joint angles computed.
    """
    pose_df = map_body_columns(pose_df)
    pose_df = compute_joint_angles(pose_df)
    return pose_df

def clean_ball_tracking_data(ball_tracking_df):
    """
    Cleans the ball tracking DataFrame by handling missing values.

    Parameters:
    - ball_tracking_df (pd.DataFrame): Raw ball tracking DataFrame.

    Returns:
    - pd.DataFrame: Cleaned ball tracking DataFrame.
    """
    # Example cleaning steps: interpolate missing bounding boxes
    ball_tracking_df[['xmin', 'ymin', 'xmax', 'ymax']] = ball_tracking_df[['xmin', 'ymin', 'xmax', 'ymax']].apply(pd.to_numeric, errors='coerce')
    ball_tracking_df[['xmin', 'ymin', 'xmax', 'ymax']] = ball_tracking_df[['xmin', 'ymin', 'xmax', 'ymax']].interpolate(method='linear', limit_direction='both')
    return ball_tracking_df

# -------------------------------------------------------------------------------------
# Basic Utility for Smoothing, Velocity, Acceleration
# -------------------------------------------------------------------------------------
def smooth_series(series, window=11, frac=0.3):
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(np.nan, index=series.index)
    rolled = series.rolling(window=window, center=True, min_periods=1).mean()
    rolled = rolled.fillna(method='bfill').fillna(method='ffill')
    xvals = np.arange(len(rolled))
    try:
        smoothed = sm.nonparametric.lowess(rolled, xvals, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)
    except Exception as e:
        logger.error(f"LOWESS smoothing failed: {e}")
        return rolled
    
def compute_velocity(positions, fps, smoothing_window=7, polyorder=2):
    """
    Use Savitzky-Golay to smooth the positions, then compute finite differences
    to get velocity.
    """
    from scipy.signal import savgol_filter
    
    # Ensure we have enough data
    if len(positions) < smoothing_window:
        smoothing_window = min(len(positions), smoothing_window)

    # Savitzky-Golay smoothing of position
    smooth_positions = savgol_filter(positions, window_length=smoothing_window, polyorder=polyorder)

    # Finite difference for velocity
    velocity = np.gradient(smooth_positions, 1.0/fps)
    return velocity


def compute_acceleration(series, dt=0.04):
    shifted_forward = series.shift(-1)
    shifted_back = series.shift(1)
    accel = (shifted_forward - shifted_back) / (2.0 * dt)
    return accel

def compute_angular_vel_accel(df, dt=0.04):
    angle_vel_cols = [c for c in df.columns if c.endswith('_vel')]
    for col in angle_vel_cols:
        df[f"{col}_accel"] = compute_acceleration(df[col], dt)
    return df

# -------------------------------------------------------------------------------------
# KPI Calculation Functions
# -------------------------------------------------------------------------------------
def calculate_release_velocity(df_detection, fps=25):
    """
    Calculates the release velocity of the ball based on scaled positional data.

    Parameters:
    - df_detection (pd.DataFrame): DataFrame containing detection data with 'center_x' and 'center_y'.
    - fps (float): Frames per second rate.

    Returns:
    - tuple: (release_velocity_in_ips, rel_frame, rel_height_in_inches)
    """
    # Check if 'center_x' and 'center_y' exist
    if 'center_x' not in df_detection.columns or 'center_y' not in df_detection.columns:
        st.warning("Columns 'center_x' or 'center_y' not found in detection data. Cannot calculate release velocity.")
        logger.warning("Columns 'center_x' or 'center_y' not found in detection data.")
        return np.nan, np.nan, np.nan

    # Ensure 'center_x' and 'center_y' do not contain all NaNs
    if df_detection['center_x'].isnull().all() or df_detection['center_y'].isnull().all():
        st.warning("All values in 'center_x' or 'center_y' are NaN. Cannot calculate release velocity.")
        logger.warning("All values in 'center_x' or 'center_y' are NaN.")
        return np.nan, np.nan, np.nan

    # Calculate velocity (in inches per second)
    df_detection['vel_x'] = compute_velocity(df_detection['center_x'], fps, smoothing_window=5)  # inches/s
    df_detection['vel_y'] = compute_velocity(df_detection['center_y'], fps, smoothing_window=5)  # inches/s

    # Debugging: Display detection data with velocity
    st.write("Detection Data After Calculating Velocity:")
    st.dataframe(df_detection[['center_x', 'vel_x', 'center_y', 'vel_y']].head())

    # Identify release frame (e.g., frame where velocity changes significantly)
    # Placeholder logic; replace with your actual release detection logic
    release_frame = df_detection['vel_x'].idxmax()  # Example: frame with maximum velocity

    # Calculate release velocity and height without scaling
    try:
        release_velocity = df_detection.loc[release_frame, 'vel_x']  # inches/s
        rel_height = df_detection.loc[release_frame, 'center_y']    # inches
    except KeyError as e:
        st.warning(f"Release frame {release_frame} not found in detection data.")
        logger.warning(f"Release frame {release_frame} not found in detection data.")
        return np.nan, np.nan, np.nan

    st.write(f"Release Frame: {release_frame}")
    st.write(f"Release Velocity: {release_velocity:.2f} inches/s")
    st.write(f"Release Height: {rel_height:.2f} inches")

    return release_velocity, release_frame, rel_height



def calculate_release_angle(df_detection, release_frame, dt=0.04):
    if df_detection.empty or pd.isna(release_frame):
        return np.nan
    df_after = df_detection[df_detection['frame'] > release_frame].head(15)
    if df_after.empty:
        return np.nan
    angles = []
    for idx in range(1, len(df_after)):
        row_current = df_after.iloc[idx]
        row_prev = df_after.iloc[idx - 1]
        dy = row_current['center_y'] - row_prev['center_y']
        dx = row_current['center_x'] - row_prev['center_x']
        angle = degrees(atan2(dy, dx))
        angles.append(angle)
    if not angles:
        return np.nan
    avg_angle = np.mean(angles)
    return avg_angle

def calculate_shot_distance(df_detection, release_frame):
    if df_detection.empty or pd.isna(release_frame):
        return np.nan
    # Example: fixed rim center (Adjust as per actual setup)
    x_rim, y_rim = 250, 400
    try:
        ball_x = df_detection.loc[df_detection['frame'] == release_frame, 'center_x'].values[0]
        ball_y = df_detection.loc[df_detection['frame'] == release_frame, 'center_y'].values[0]
    except IndexError:
        return np.nan
    distance = np.sqrt((ball_x - x_rim)**2 + (ball_y - y_rim)**2)
    return distance

def calculate_apex_height(df_detection):
    if df_detection.empty:
        return np.nan
    apex = df_detection['center_y'].max()
    return apex

def calculate_release_time(df_detection, release_frame, dt=0.04):
    if df_detection.empty or pd.isna(release_frame):
        return np.nan
    return release_frame * dt

def calculate_release_quality(df_spin):
    if df_spin.empty or 'spin_x' not in df_spin.columns:
        return np.nan
    average_spin_x = df_spin['spin_x'].mean()
    release_quality = ((average_spin_x + 180) / 360) * 100  # Assuming spin_x ranges from -180 to 180
    return release_quality

def calculate_release_curvature(df_detection):
    if df_detection.empty or 'center_x' not in df_detection.columns or 'center_y' not in df_detection.columns:
        return np.nan
    angles = []
    for i in range(1, len(df_detection)-1):
        p1 = (df_detection.iloc[i-1]['center_x'], df_detection.iloc[i-1]['center_y'])
        p2 = (df_detection.iloc[i]['center_x'], df_detection.iloc[i]['center_y'])
        p3 = (df_detection.iloc[i+1]['center_x'], df_detection.iloc[i+1]['center_y'])
        angle = angle_2d(p1, p2, p3)
        angles.append(angle)
    total_curvature = np.nansum(np.abs(np.array(angles)))
    return total_curvature

def calculate_asymmetry_index(df_pose):
    """
    Measure the average difference between left and right joint angles.
    """
    joint_pairs = [
       ('left_shoulder_angle','right_shoulder_angle'),
       ('left_elbow_angle','right_elbow_angle'),
       ('left_knee_angle','right_knee_angle'),
       ('left_hip_angle','right_hip_angle'),
       ('left_ankle_angle','right_ankle_angle'),
    ]
    diffs = []
    for (left, right) in joint_pairs:
        if left in df_pose.columns and right in df_pose.columns:
            left_mean = df_pose[left].mean()
            right_mean = df_pose[right].mean()
            if not pd.isna(left_mean) and not pd.isna(right_mean):
                diffs.append(abs(left_mean - right_mean))
    if not diffs:
        return np.nan
    return np.mean(diffs)

def compute_kinematic_chain_sequence(df_pose):
    """
    Identifies the order in which joints reach peak angular velocity.
    Returns a dict: { 'joint_name': peak_vel_frame, ... }
    Then you can sort by peak_vel_frame to see the chain.
    """
    angle_vel_cols = [c for c in df_pose.columns if c.endswith('_angle_vel')]
    sequence_dict = {}
    for angle_vel_col in angle_vel_cols:
        if angle_vel_col in df_pose.columns:
            peak_vel_idx = df_pose[angle_vel_col].idxmax()
            if not pd.isna(peak_vel_idx):
                peak_frame = df_pose.loc[peak_vel_idx, 'frame']
                sequence_dict[angle_vel_col] = peak_frame
    return sequence_dict

def calculate_pivotal_score(kpi_dict):
    """
    Example calculation for Pivotal Score based on selected KPIs.
    Adjust the formula as per your requirements.
    """
    if not kpi_dict:
        return np.nan
    # Example weights and KPIs
    weights = {
        'Release Velocity': 0.3,
        'Shot Distance': 0.2,
        'Release Angle': 0.2,
        'Apex Height': 0.1,
        'Release Quality': 0.2
    }
    score = 0
    for k, w in weights.items():
        val = kpi_dict.get(k, np.nan)
        if pd.isna(val):
            continue
        # Normalize each KPI as needed
        if k == 'Release Velocity':
            norm = min(val / 10, 1)  # Assuming 10 m/s is max
        elif k == 'Shot Distance':
            norm = min(val / 50, 1)  # Assuming 50 meters is max
        elif k == 'Release Angle':
            norm = 1 - abs(val) / 90  # Assuming 0 degrees is best
        elif k == 'Apex Height':
            norm = min(val / 500, 1)  # Assuming 500 units is max
        elif k == 'Release Quality':
            norm = val / 100  # Already 0-100
        else:
            norm = 0
        score += norm * w
    return score * 100  # Scale to 0-100
def calculate_release_angle_pose(df_pose):
    """
    Calculates the release angle from Pose data.
    Implement Pose-specific logic here.
    """
    # Placeholder implementation
    # Example: Calculate the angle between the elbow and wrist at release frame
    if 'left_elbow_angle' in df_pose.columns and 'right_elbow_angle' in df_pose.columns:
        release_frame = 0  # Assuming 'relative_frame' = 0 is release
        try:
            left_angle = df_pose.loc[df_pose['relative_frame'] == 0, 'left_elbow_angle'].values[0]
            right_angle = df_pose.loc[df_pose['relative_frame'] == 0, 'right_elbow_angle'].values[0]
            release_angle = (left_angle + right_angle) / 2
            return release_angle
        except IndexError:
            return np.nan
    return np.nan

def calculate_shot_distance_pose(df_pose):
    """
    Calculates the shot distance from Pose data.
    Implement Pose-specific logic here.
    """
    # Placeholder implementation
    # Example: Distance between shoulders at release frame
    if 'left_shoulder_x' in df_pose.columns and 'right_shoulder_x' in df_pose.columns and \
       'left_shoulder_y' in df_pose.columns and 'right_shoulder_y' in df_pose.columns:
        release_frame = 0  # Assuming 'relative_frame' = 0 is release
        try:
            left_shoulder = df_pose.loc[df_pose['relative_frame'] == 0, ['left_shoulder_x', 'left_shoulder_y']].values[0]
            right_shoulder = df_pose.loc[df_pose['relative_frame'] == 0, ['right_shoulder_x', 'right_shoulder_y']].values[0]
            distance = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + (right_shoulder[1] - left_shoulder[1])**2)
            return distance
        except IndexError:
            return np.nan
    return np.nan

def calculate_apex_height_pose(df_pose):
    """
    Calculates the apex height from Pose data.
    Implement Pose-specific logic here.
    """
    # Placeholder implementation
    # Example: Maximum head y-coordinate as apex height
    if 'head_y' in df_pose.columns:
        apex = df_pose['head_y'].max()
        return apex
    return np.nan

def calculate_release_time_pose(df_pose, release_frame):
    """
    Calculates the release time from Pose data.
    Implement Pose-specific logic here.
    """
    if 'relative_frame' not in df_pose.columns:
        return np.nan
    time = release_frame * 0.04  # Assuming fps=25
    return time

def calculate_release_curvature_pose(df_pose):
    """
    Calculates the release curvature from Pose data.
    Implement Pose-specific logic here.
    """
    # Placeholder implementation
    # Example: Total curvature based on joint angles changes
    # This is a simplistic approach and should be refined based on actual biomechanical definitions
    angles = [col for col in df_pose.columns if 'angle' in col]
    if not angles:
        return np.nan
    curvature = df_pose[angles].diff().abs().sum(axis=1).sum()
    return curvature

def calculate_kpis_pose(df_pose):
    """
    Calculate KPIs specific to Pose data.
    """
    if df_pose.empty:
        return {}
    # Example calculations
    release_angle = calculate_release_angle_pose(df_pose)
    shot_distance = calculate_shot_distance_pose(df_pose)
    apex_height = calculate_apex_height_pose(df_pose)
    release_time = calculate_release_time_pose(df_pose, release_frame=0)  # Modify as needed
    release_curvature = calculate_release_curvature_pose(df_pose)
    asymmetry_index = calculate_asymmetry_index(df_pose)
    kinematic_sequence = compute_kinematic_chain_sequence(df_pose)
    
    # Kinematic Chain Sequencing Score (Example: Number of joints in correct sequence)
    kinematic_chain_sequence_score = len(kinematic_sequence)  # Placeholder
    
    pivotal_score = calculate_pivotal_score({
        'Shot Distance': shot_distance,
        'Release Angle': release_angle,
        'Apex Height': apex_height,
        'Asymmetry Index': asymmetry_index,
        # Add more Pose-specific KPIs as needed
    })
    
    return {
        'Release Angle': release_angle,
        'Shot Distance': shot_distance,
        'Apex Height': apex_height,
        'Release Time': release_time,
        'Release Curvature': release_curvature,
        'Asymmetry Index': asymmetry_index,
        'Kinematic Chain Sequencing': kinematic_sequence,
        'Kinematic Chain Sequence Score': kinematic_chain_sequence_score,
        'Pivotal Score': pivotal_score
    }


def calculate_kpis_ball_tracking(df_ball):
    """
    Calculate KPIs specific to Ball Tracking.

    Returns KPIs in real-world units (inches, inches/s, degrees).
    """
    if df_ball.empty:
        return {}
    # Example calculations
    release_velocity, rel_frame, rel_height = calculate_release_velocity(df_ball, fps=25)
    release_angle = calculate_release_angle(df_ball, rel_frame)
    shot_distance = calculate_shot_distance(df_ball, rel_frame)
    apex_height = calculate_apex_height(df_ball)
    release_time = calculate_release_time(df_ball, rel_frame)
    release_curvature = calculate_release_curvature(df_ball)

    pivotal_score = calculate_pivotal_score({
        'Release Velocity': release_velocity,
        'Shot Distance': shot_distance,
        'Release Angle': release_angle,
        'Apex Height': apex_height,
        'Release Quality': np.nan  # Not applicable for Ball Tracking
    })

    return {
        'Release Velocity': release_velocity,        # inches/s
        'Release Frame': rel_frame,
        'Release Height': rel_height,                # inches
        'Release Angle': release_angle,              # degrees
        'Shot Distance': shot_distance,              # inches
        'Apex Height': apex_height,                  # inches
        'Release Time': release_time,                # seconds
        'Release Curvature': release_curvature,      # curvature units
        'Pivotal Score': pivotal_score               # percentage
    }


def calculate_kpis_spin(df_spin):
    """
    Calculate KPIs specific to Spin data.
    """
    if df_spin.empty:
        return {}
    # Example calculations
    release_quality = calculate_release_quality(df_spin)
    # Additional Spin-specific KPIs can be added here
    pivotal_score = calculate_pivotal_score({
        'Release Velocity': np.nan,  # Not applicable
        'Shot Distance': np.nan,     # Not applicable
        'Release Angle': np.nan,     # Not applicable
        'Apex Height': np.nan,       # Not applicable
        'Release Quality': release_quality
    })
    return {
        'Release Quality': release_quality,
        'Pivotal Score': pivotal_score
    }

def calculate_consistency_percentage(kpi_list, kpi_key='Release Quality'):
    """
    Calculate consistency based on the standard deviation of a specific KPI.
    """
    if not kpi_list:
        return np.nan
    qualities = [k.get(kpi_key, np.nan) for k in kpi_list]
    qualities = [q for q in qualities if not pd.isna(q)]
    if not qualities:
        return np.nan
    std_val = np.std(qualities)
    consistency_score = 100 - (std_val / 30 * 100)  # Adjust 30 based on acceptable std
    return np.clip(consistency_score, 0, 100)

# -------------------------------------------------------------------------------------
# Utility Function to Add Time Column
# -------------------------------------------------------------------------------------
def add_time_column(df, fps=25):
    """
    Adds a 'time' column to the DataFrame based on 'relative_frame' and frames per second (fps).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing 'relative_frame'.
    - fps (float): Frames per second rate.

    Returns:
    - pd.DataFrame: DataFrame with an added 'time' column.
    """
    if 'relative_frame' not in df.columns:
        st.warning("DataFrame missing 'relative_frame' column. Cannot add 'time' column.")
        logger.warning("DataFrame missing 'relative_frame' column. Cannot add 'time' column.")
        df['time'] = np.nan  # Assign NaN if 'relative_frame' is missing
        return df

    # Calculate time in seconds
    df['time'] = df['relative_frame'] / fps
    return df

# -------------------------------------------------------------------------------------
# identify_biomech_phases Function
# -------------------------------------------------------------------------------------
def identify_biomech_phases(df, fps=25):
    """
    Identifies phases in [start_idx..(release_idx+margin)], ignoring frames outside that range.

    PHASES:
      - Unknown (frames outside the valid range)
      - Preparation
      - Release (~2 frames around release frame)
      - Inertia (after release_end)
      - Ball Elevation
    """
    # 0) If too short, return
    if len(df) < 5:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    phases = pd.Series(["Unknown"] * len(df), index=df.index)

    # 1) Find "ball in hand" start (min x_smooth or similar logic)
    if 'left_wrist_x' in df.columns and 'right_wrist_x' in df.columns:
        df['average_wrist_x'] = df[['left_wrist_x', 'right_wrist_x']].mean(axis=1)
        start_idx = df['average_wrist_x'].idxmin()
    else:
        start_idx = df['frame'].iloc[0]

    # 2) Find release idx
    release_idx = detect_release_frame_pose(df)
    if release_idx is None:
        # fallback to last frame
        release_idx = df['frame'].iloc[-1]

    # Force logic only between [start_idx..release_idx + margin]
    margin = 5
    valid_range_mask = (df['frame'] >= start_idx) & (df['frame'] <= (release_idx + margin))
    phases[~valid_range_mask] = "Unknown"

    # Subset df to that range
    valid_df = df[valid_range_mask].copy()
    valid_idx = valid_df.index  # actual row indices

    # (a) Preparation: from start_idx -> knee crouch threshold
    # We'll use right_knee_angle_vel as an example
    knee_col = "right_knee_angle_vel"  # Using angular velocity for better phase detection
    if knee_col not in valid_df.columns:
        # fallback => everything in valid range is "Preparation"
        phases[valid_range_mask] = "Preparation"
    else:
        baseline_knee = valid_df[knee_col].iloc[:10].mean()  # average of first ~10 frames in valid_df
        knee_threshold = baseline_knee + 20.0  # Adjust threshold as needed
        transitions = valid_df.index[valid_df[knee_col] > knee_threshold]

        if not transitions.empty:
            prep_end_idx = transitions.min()
        else:
            prep_end_idx = valid_idx[0]  # no real crouch => trivial

        # Label from start_loc..prep_end_idx as Preparation
        phases.loc[start_idx:prep_end_idx] = "Preparation"

        # (b) Ball Elevation Phase
        # Criteria: Wrist is moving upwards (y_smooth increasing)
        if 'left_wrist_y' in df.columns and 'right_wrist_y' in df.columns:
            df['average_wrist_y'] = df[['left_wrist_y', 'right_wrist_y']].mean(axis=1)
            ball_elev_mask = (df['frame'] > prep_end_idx) & (df['frame'] < release_idx)
            phases.loc[ball_elev_mask] = "Ball Elevation"

        # (c) Release Phase
        release_margin = 2  # frames around release
        release_mask = (df['frame'] >= release_idx) & (df['frame'] <= (release_idx + release_margin))
        phases.loc[release_mask] = "Release"

        # (d) Inertia Phase => after release_end
        inertia_mask = df['frame'] > (release_idx + release_margin)
        phases.loc[inertia_mask] = "Inertia"

    return phases

# -------------------------------------------------------------------------------------
# visualize_phases Function
# -------------------------------------------------------------------------------------

def calculate_quadratic_fit(release_phase):
    """
    Calculate curvature of the ball's path using quadratic fit and return curvature and apex height.

    Parameters:
    - release_phase (pd.DataFrame): DataFrame containing 'center_x' and 'center_y' up to the release frame.

    Returns:
    - tuple: (curvature, apex_height)
    """
    try:
        # Fit a quadratic curve: y = ax^2 + bx + c
        if len(release_phase) < 3:
            return np.nan, np.nan
        z = np.polyfit(release_phase['center_x'], release_phase['center_y'], 2)
        a, b, c = z
        curvature = 2 * a  # Simplistic curvature measure from quadratic coefficient

        # Calculate apex (vertex of the parabola)
        apex_x = -b / (2 * a) if a != 0 else np.nan
        apex_y = a * apex_x**2 + b * apex_x + c if not np.isnan(apex_x) else np.nan
        apex_height = apex_y

        return curvature, apex_height
    except Exception as e:
        logger.error(f"Error calculating quadratic fit: {str(e)}")
        return np.nan, np.nan


def calculate_lateral_curvature(spin_df):
    """
    Calculate lateral curvature based on spin axis data.
    """
    try:
        # Assuming 'spin_axis' column exists and represents lateral spin direction
        if 'ex' not in spin_df.columns:
            return np.nan
        # Calculate consistency as standard deviation of spin_axis
        lateral_spin_std = spin_df['ex'].std()
        return lateral_spin_std
    except Exception as e:
        logger.error(f"Error calculating lateral curvature: {str(e)}")
        return np.nan

def calculate_form_score(kpi_dict):
    """
    Calculate a validated form score (0-100) based on selected KPIs.
    
    Parameters:
    - kpi_dict (dict): Dictionary containing KPI names and their corresponding values.
    
    Returns:
    - float: Form score between 0 and 100.
    """
    if not kpi_dict:
        logger.warning("KPI dictionary is empty. Cannot calculate form score.")
        return np.nan
    
    # Example weights and KPIs
    weights = {
        'Release Velocity': 0.3,
        'Shot Distance': 0.2,
        'Release Angle': 0.2,
        'Apex Height': 0.1,
        'Release Quality': 0.2
    }
    
    score = 0
    for k, w in weights.items():
        val = kpi_dict.get(k, np.nan)
        if pd.isna(val):
            logger.info(f"KPI '{k}' is NaN. Skipping in form score calculation.")
            continue
        # Normalize each KPI as needed
        if k == 'Release Velocity':
            norm = min(val / 10, 1)  # Assuming 10 m/s is max
        elif k == 'Shot Distance':
            norm = min(val / 50, 1)  # Assuming 50 meters is max
        elif k == 'Release Angle':
            norm = 1 - abs(val) / 90  # Assuming 0 degrees is best
        elif k == 'Apex Height':
            norm = min(val / 500, 1)  # Assuming 500 units is max
        elif k == 'Release Quality':
            norm = val / 100  # Already 0-100
        else:
            norm = 0
        score += norm * w
        logger.debug(f"KPI '{k}': value={val}, normalized={norm}, weighted={norm * w}")
    
    form_score = score * 100  # Scale to 0-100
    form_score = min(100, max(0, form_score))  # Ensure within bounds
    logger.info(f"Calculated form score: {form_score}")
    return form_score



def is_valid_metrics(metrics, kpi_ranges):
    """
    Check if the metrics are within realistic ranges.
    """
    try:
        for kpi, (min_val, max_val) in kpi_ranges.items():
            value = metrics.get(kpi, np.nan)
            if pd.isna(value):
                return False
            if not (min_val <= value <= max_val):
                return False
        return True
    except:
        return False
    
def visualize_phases(df, phases, job_id, segment_id, fps=25):
    """
    Visualizes the phases of the shooting motion in the time series.
    """
    # Check if 'relative_frame' exists and has valid data
    if 'relative_frame' not in df.columns:
        st.error("DataFrame missing 'relative_frame' column. Cannot visualize phases.")
        logger.error("'relative_frame' column missing in DataFrame.")
        return

    if df['relative_frame'].isnull().all():
        st.error("'relative_frame' contains all NaN values. Cannot visualize phases.")
        logger.error("'relative_frame' contains all NaN values.")
        return

    # Add 'time' column
    df = add_time_column(df, fps)
    
    # Check if 'time' column has valid data
    if 'time' not in df.columns or df['time'].isnull().all():
        st.warning(f"No valid 'time' data for Job ID: {job_id}, Segment ID: {segment_id}.")
        return

    if df.empty:
        st.warning(f"No data to plot for Job ID: {job_id}, Segment ID: {segment_id}.")
        return

    fig = go.Figure()

    # Define phases and their colors
    phase_colors = {
        "Preparation": "blue",
        "Ball Elevation": "green",
        "Release": "red",
        "Inertia": "orange",
        "Unknown": "gray"
    }

    # Plot each phase with angular acceleration
    for phase, color in phase_colors.items():
        phase_mask = (phases == phase)
        if not phase_mask.any():
            continue  # Skip if no frames in this phase

        phase_df = df[phase_mask].dropna(subset=['time'])
        if phase_df.empty:
            st.warning(f"No valid 'time' data for phase '{phase}'.")
            continue

        # Plot angular acceleration if available
        angle_accel_cols = [c for c in phase_df.columns if c.endswith('_accel')]
        if angle_accel_cols:
            for angle_accel_col in angle_accel_cols:
                y_label = f"{humanize_label(angle_accel_col)} (deg/s²)"
                fig.add_trace(go.Scatter(
                    x=phase_df['time'],
                    y=phase_df[angle_accel_col],
                    mode='lines',
                    name=f"{phase} - {y_label}",
                    line=dict(color=color)
                ))
        else:
            # If no angular acceleration, plot joint angles
            angle_cols = [c for c in phase_df.columns if c.endswith('_angle')]
            for angle_col in angle_cols:
                y_label = f"{humanize_label(angle_col)} (deg)"
                fig.add_trace(go.Scatter(
                    x=phase_df['time'],
                    y=phase_df[angle_col],
                    mode='lines',
                    name=f"{phase} - {y_label}",
                    line=dict(color=color)
                ))

    fig.update_layout(
        title="Shooting Phases with Angular Metrics",
        xaxis_title="Time (s)",
        yaxis_title="Degrees / Degrees/s²",
        showlegend=True
    )

    # Assign a unique key to prevent duplicate element IDs
    st.plotly_chart(fig, use_container_width=True, key=f"visualize_phases_{job_id}_{segment_id}")

# -------------------------------------------------------------------------------------
# Utility Function to Calculate Ball Center
# -------------------------------------------------------------------------------------
def calculate_ball_center(df_detection):
    """
    Calculates the center coordinates of the ball from bounding box data
    and computes the ball's diameter in pixels.

    Parameters:
    - df_detection (pd.DataFrame): DataFrame containing bounding box columns.

    Returns:
    - pd.DataFrame: DataFrame with added 'center_x', 'center_y', and 'ball_diameter_pixels' columns.
    """
    required_columns = ['xmin', 'ymin', 'xmax', 'ymax']
    if all(col in df_detection.columns for col in required_columns):
        df_detection['center_x'] = (df_detection['xmin'] + df_detection['xmax']) / 2
        df_detection['center_y'] = (df_detection['ymin'] + df_detection['ymax']) / 2
        # Calculate diameter in pixels
        df_detection['ball_diameter_pixels'] = df_detection['xmax'] - df_detection['xmin']
    else:
        missing_cols = [col for col in required_columns if col not in df_detection.columns]
        st.warning(f"Missing bounding box columns: {missing_cols}. Cannot calculate ball center.")
        logger.warning(f"Missing bounding box columns: {missing_cols}. Cannot calculate ball center.")
        df_detection['center_x'] = np.nan
        df_detection['center_y'] = np.nan
        df_detection['ball_diameter_pixels'] = np.nan
    return df_detection


def compute_job_kpi_summary(user_email, job):
    """
    Gathers all segments for the given job, loads detection + spin data,
    calculates each segment's KPIs, returns an average of all shots (segments).
    
    Parameters:
    - user_email (str): Email of the user.
    - job (dict): Job information dictionary.
    
    Returns:
    - dict: Averaged KPIs for the job.
    """
    job_id = job['JobID']
    source = job.get('Source', '').lower()
    all_pose_kpis = []
    all_ball_kpis = []
    
    if source.startswith('pose'):
        # Handle Pose Jobs
        segments = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], user_email, job_id)
        if not segments:
            logger.warning(f"No segments found for Pose Job ID: {job_id}")
            return {}
        
        for segment_id in segments:
            df_final = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, job_id, segment_id)
            if df_final.empty:
                st.warning(f"No data found for Job ID: {job_id}, Segment ID: {segment_id}.")
                continue
            
            # Ensure unique frames by selecting the first occurrence
            df_final = df_final.sort_values(['frame', 'confidence'], ascending=[True, False])
            
            # Separate pose and ball tracking data
            pose_df, ball_tracking_df = separate_pose_and_ball_tracking(df_final)
            
            # Clean data
            if not pose_df.empty:
                pose_df = clean_pose_data(pose_df)
            if not ball_tracking_df.empty:
                ball_tracking_df = clean_ball_tracking_data(ball_tracking_df)
                ball_tracking_df = calculate_ball_center(ball_tracking_df)
            
            # Proceed if pose data is available
            if pose_df.empty:
                st.warning(f"No pose data available for Job ID: {job_id}, Segment ID: {segment_id}.")
                continue
            
            # Identify biomechanical phases
            phases = identify_biomech_phases(pose_df)
            
            # Ensure 'relative_frame' column is added
            release_frame = detect_release_frame_pose(pose_df)
            pose_df = add_relative_frame(pose_df, release_frame)
            
            # Trim and smooth data
            trimmed_pose_df = trim_shot(pose_df, release_frame)
            if trimmed_pose_df is not None and not trimmed_pose_df.empty:
                for c in trimmed_pose_df.columns:
                    if c not in ['frame', 'relative_frame']:
                        trimmed_pose_df[c] = smooth_series(trimmed_pose_df[c])
                trimmed_pose_df = compute_angular_vel_accel(trimmed_pose_df)
                # Calculate Pose KPIs
                kpi_pose = calculate_kpis_pose(trimmed_pose_df)
                if kpi_pose:
                    all_pose_kpis.append(kpi_pose)
            else:
                st.warning(f"No valid trimmed Pose data for Job ID: {job_id}, Segment ID: {segment_id}.")
            
            # Calculate Ball Tracking KPIs if Ball Tracking data exists
            if not ball_tracking_df.empty:
                kpi_ball = calculate_kpis_ball_tracking(ball_tracking_df)
                if kpi_ball:
                    all_ball_kpis.append(kpi_ball)
    elif source.startswith('spin'):
        # Handle Spin Jobs
        df_spin = load_spin_axis_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, job_id)
        if df_spin.empty:
            st.warning(f"No spin_axis.csv found for Spin Job ID: {job_id}.")
            return {}
        
        # Clean and prepare spin data
        # Assuming spin_axis.csv has columns: frame, ex, ey, ez
        if not {'frame', 'ex', 'ey', 'ez'}.issubset(df_spin.columns):
            st.warning(f"spin_axis.csv for Job ID: {job_id} is missing required columns.")
            return {}
        # Smooth spin data
        for col in ['ex', 'ey', 'ez']:
            df_spin[col] = smooth_series(df_spin[col])
        # Add 'relative_frame'
        release_frame = df_spin['frame'].min()  # Assuming first frame is release
        df_spin = add_relative_frame(df_spin, release_frame)
        # Calculate KPIs
        kpi_spin = calculate_kpis_spin(df_spin)
        if kpi_spin:
            all_ball_kpis.append(kpi_spin)  # Assuming Spin KPIs are treated similarly to Ball Tracking
    
    else:
        st.warning(f"Unknown job source type for Job ID: {job_id}.")
        logger.warning(f"Unknown job source type for Job ID: {job_id}.")
        return {}
    
    # Combine Pose and Ball Tracking KPIs
    combined_kpis = {}
    
    # Average Pose KPIs
    if all_pose_kpis:
        df_pose_kpis = pd.DataFrame(all_pose_kpis)
        mean_pose_kpi = df_pose_kpis.mean(numeric_only=True).to_dict()
        combined_kpis.update(mean_pose_kpi)
    
    # Average Ball Tracking KPIs
    if all_ball_kpis:
        df_ball_kpis = pd.DataFrame(all_ball_kpis)
        mean_ball_kpi = df_ball_kpis.mean(numeric_only=True).to_dict()
        combined_kpis.update(mean_ball_kpi)
    
    # Calculate release consistency based on a specific KPI, e.g., 'Release Velocity' or 'Release Quality'
    # Adjust 'kpi_key' as needed based on available KPIs
    kpi_key = 'Release Quality' if 'Release Quality' in combined_kpis else 'Release Velocity'
    consistency_percentage = calculate_consistency_percentage(all_ball_kpis, kpi_key=kpi_key)
    combined_kpis["Release Consistency"] = consistency_percentage
    
    return combined_kpis


# -------------------------------------------------------------------------------------
# Visualization Helpers
# -------------------------------------------------------------------------------------
def visualize_overview_job_kpis(kpi_dict):
    """
    Displays each KPI from the aggregated dictionary as an st.metric or a table.
    """
    if not kpi_dict:
        st.write("No KPI data.")
        return
    # We can just display them in a table or as metrics
    col1, col2, col3 = st.columns(3)
    items = list(kpi_dict.items())
    for i, (k, v) in enumerate(items):
        if pd.isna(v):
            display_val = "N/A"
        else:
            display_val = f"{v:.2f}"
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
        col.metric(label=k, value=display_val)

def visualize_section(mean_df, std_df, selected_metrics, metric_section, section_title, job_id, segment_id, fps=25):
    st.markdown(f"##### {section_title} Metrics")
    options = [humanize_label(metric) for metric in selected_metrics]
    selected_options = st.multiselect(
        f"Select {section_title} Metrics to Visualize",
        options=options,
        default=options
    )
    metric_mapping = {humanize_label(metric): metric for metric in selected_metrics}
    selected_actual_metrics = [metric_mapping[label] for label in selected_options]
    if selected_actual_metrics:
        fig = go.Figure()
        color_cycle = [
            "#1f77b4","#ff7f0e","#2ca02c","#d62728",
            "#9467bd","#8c564b","#e377c2","#7f7f7f",
            "#bcbd22","#17becf"
        ]
        i_color = 0
        for metric in selected_actual_metrics:
            if metric in mean_df.columns and metric in std_df.columns:
                c = color_cycle[i_color % len(color_cycle)]
                i_color += 1
                fig.add_trace(go.Scatter(
                    x=mean_df['time'],
                    y=mean_df[metric],
                    mode='lines',
                    name=f"Mean - {humanize_label(metric)}",
                    line=dict(width=2, color=c)
                ))
                upper = (mean_df[metric] + std_df[metric]).tolist()
                lower = (mean_df[metric] - std_df[metric]).tolist()
                fig.add_trace(go.Scatter(
                    x=list(mean_df['time']) + list(mean_df['time'][::-1]),
                    y=upper + lower[::-1],
                    fill='toself',
                    fillcolor=c.replace(")", ", 0.2)").replace("rgb", "rgba") if c.startswith("rgb") else 'rgba(31,119,180,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    name=f"Std Dev - {humanize_label(metric)}"
                ))
        fig.update_layout(
            title=f"{section_title} Metrics Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Degrees / Degrees/s²",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key=f"visualize_section_{metric_section}_{job_id}_{segment_id}")
    else:
        st.write(f"No {section_title} metrics selected.")

def visualize_spin_metrics(mean_spin, std_spin, spin_metrics, job_id):
    if not spin_metrics:
        st.write("No Spin metrics selected.")
        return
    fig = go.Figure()
    color_cycle = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728",
        "#9467bd","#8c564b","#e377c2","#7f7f7f",
        "#bcbd22","#17becf"
    ]
    i_color = 0
    for metric in spin_metrics:
        if metric in mean_spin.columns and metric in std_spin.columns:
            c = color_cycle[i_color % len(color_cycle)]
            i_color += 1
            fig.add_trace(go.Scatter(
                x=mean_spin['time'],
                y=mean_spin[metric],
                mode='lines',
                name=f"Mean - {humanize_label(metric)}",
                line=dict(width=2, color=c)
            ))
            upper = (mean_spin[metric] + std_spin[metric]).tolist()
            lower = (mean_spin[metric] - std_spin[metric]).tolist()
            fig.add_trace(go.Scatter(
                x=list(mean_spin['time']) + list(mean_spin['time'][::-1]),
                y=upper + lower[::-1],
                fill='toself',
                fillcolor=c.replace(")", ", 0.2)").replace("rgb", "rgba") if c.startswith("rgb") else 'rgba(31,119,180,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                name=f"Std Dev - {humanize_label(metric)}"
            ))
    fig.update_layout(
        title="Spin Metrics Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Units",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"visualize_spin_metrics_{job_id}")

def plot_single_shot(df, title="Single Shot Analysis", highlight_release=True, job_id=None, segment_id=None, fps=25):
    if df.empty:
        st.warning("No data to plot in single-shot view.")
        return
    release_frame = 0  # Since relative_frame = 0 corresponds to release
    angle_cols = [c for c in df.columns if c.endswith('_angle')]
    if not angle_cols:
        st.info("No angle columns found in single-shot data for plotting.")
        return
    selected_cols = st.multiselect(
        "Select angles to plot (single shot)",
        angle_cols,
        default=angle_cols[:3]
    )
    if not selected_cols:
        st.write("No angles selected for single-shot plot.")
        return
    # Add 'time' column
    df = add_time_column(df, fps)
    if 'time' not in df.columns or df['time'].isnull().all():
        st.warning("No valid 'time' data available for plotting.")
        return
    fig = go.Figure()
    color_cycle = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728",
        "#9467bd","#8c564b","#e377c2","#7f7f7f",
        "#bcbd22","#17becf"
    ]
    i_color = 0
    for col in selected_cols:
        c = color_cycle[i_color % len(color_cycle)]
        i_color += 1
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df[col],
            mode='lines',
            name=humanize_label(col),
            line=dict(width=2, color=c)
        ))
    if highlight_release:
        # Assuming release_frame is the time corresponding to relative_frame = 0
        release_time = 0  # Adjust if necessary
        release_x = df.loc[df['relative_frame'] == 0, 'time'].values
        if len(release_x) > 0:
            fig.add_vline(x=release_x[0], line_width=2, line_dash="dash", line_color="red", name="Release Point")
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Degrees (deg)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"single_shot_{job_id}_{segment_id}")

# -------------------------------------------------------------------------------------
# Compute KPIs for Spin Jobs
# -------------------------------------------------------------------------------------
def load_spin_axis_csv(s3_client, bucket_name, user_email, job_id):
    """Improved spin data loader with better error handling"""
    spin_key = f"processed/{user_email}/{job_id}/spin_axis.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=spin_key)
        csv_data = response['Body'].read().decode('utf-8')
        df_spin = pd.read_csv(StringIO(csv_data))
        
        # Calculate spin magnitude if possible
        if {'ex', 'ey', 'ez'}.issubset(df_spin.columns):
            df_spin['spin_magnitude'] = np.sqrt(
                df_spin['ex']**2 + 
                df_spin['ey']**2 + 
                df_spin['ez']**2
            )
        else:
            logger.warning(f"Spin axis columns missing in Job ID: {job_id}.")
        
        return df_spin
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Spin data not found for Job ID: {job_id}. Skipping spin metrics.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading spin data for Job ID {job_id}: {str(e)}")
        return pd.DataFrame()


def compute_spin_kpi(df_spin):
    """Calculate comprehensive spin metrics"""
    kpis = {}
    try:
        if not df_spin.empty:
            # Basic stats
            kpis = {
                'max_spin_rate': df_spin['spin_magnitude'].max(),
                'avg_spin_rate': df_spin['spin_magnitude'].mean(),
                'min_spin_rate': df_spin['spin_magnitude'].min(),
                'spin_consistency': df_spin['spin_magnitude'].std(),
                'dominant_axis': df_spin[['ex', 'ey', 'ez']].abs().mean().idxmax()
            }
            
            # Rotation direction analysis
            kpis['backspin_ratio'] = (df_spin['ez'] > 0).mean()
            kpis['sidespin_ratio'] = ((df_spin['ex'].abs() + df_spin['ey'].abs()) > 0.5).mean()
            
    except Exception as e:
        st.error(f"KPI calculation error: {str(e)}")
    
    return kpis

def visualize_spin_metrics(df_spin, job_id):
    """Enhanced spin visualization with 3D analysis"""
    if df_spin.empty:
        return

    with st.expander("3D Spin Axis Visualization", expanded=True):
        fig = go.Figure(data=[
            go.Scatter3d(
                x=df_spin['ex'],
                y=df_spin['ey'],
                z=df_spin['ez'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df_spin['spin_magnitude'],
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title='X-axis Spin',
                yaxis_title='Y-axis Spin',
                zaxis_title='Z-axis Spin'
            ),
            title="3D Spin Axis Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time-series visualization
    st.subheader("Spin Rate Over Time")
    fig = px.line(df_spin, x='frame', y='spin_magnitude', 
                  title="Spin Magnitude During Shot")
    st.plotly_chart(fig, use_container_width=True)



def compute_spin_kpi(df_spin):
    """
    Computes basic KPIs for Spin data.
    """
    if df_spin.empty:
        return {}
    kpi = {}
    if 'spin_x' in df_spin.columns:
        kpi['Average Spin X'] = df_spin['spin_x'].mean()
        kpi['Spin X Std Dev'] = df_spin['spin_x'].std()
    if 'spin_y' in df_spin.columns:
        kpi['Average Spin Y'] = df_spin['spin_y'].mean()
        kpi['Spin Y Std Dev'] = df_spin['spin_y'].std()
    if 'spin_z' in df_spin.columns:
        kpi['Average Spin Z'] = df_spin['spin_z'].mean()
        kpi['Spin Z Std Dev'] = df_spin['spin_z'].std()
    return kpi

def calculate_quadratic_fit(release_phase):
    """
    Calculate curvature of the ball's path using quadratic fit and return curvature and apex height.
    """
    try:
        # Fit a quadratic curve: y = ax^2 + bx + c
        if len(release_phase) < 3:
            return np.nan, np.nan
        z = np.polyfit(release_phase['center_x'], release_phase['center_y'], 2)
        a, b, c = z
        curvature = 2 * a  # Simplistic curvature measure from quadratic coefficient

        # Calculate apex (vertex of the parabola)
        apex_x = -b / (2 * a) if a != 0 else np.nan
        apex_y = a * apex_x**2 + b * apex_x + c if not np.isnan(apex_x) else np.nan
        apex_height = apex_y

        return curvature, apex_height
    except Exception as e:
        logger.error(f"Error calculating quadratic fit: {str(e)}")
        return np.nan, np.nan

def calculate_lateral_curvature(spin_df):
    """
    Calculate lateral curvature based on spin axis data.
    """
    try:
        # Assuming 'spin_axis' column exists and represents lateral spin direction
        if 'spin_axis' not in spin_df.columns:
            return np.nan
        # Calculate consistency as standard deviation of spin_axis
        lateral_spin_std = spin_df['spin_axis'].std()
        return lateral_spin_std
    except Exception as e:
        logger.error(f"Error calculating lateral curvature: {str(e)}")
        return np.nan

def is_valid_metrics(metrics, kpi_ranges):
    """
    Check if the metrics are within realistic ranges.
    """
    try:
        for kpi, (min_val, max_val) in kpi_ranges.items():
            value = metrics.get(kpi, np.nan)
            if pd.isna(value):
                return False
            if not (min_val <= value <= max_val):
                return False
        return True
    except:
        return False

def find_release_frame(pose_df, fps=25, min_release_time=1, max_release_time=5):
    """
    Finds the optimal release frame based on KPI within a specified time window.
    
    Parameters:
    - pose_df (pd.DataFrame): DataFrame containing pose data with 'left_elbow_angle' and 'right_elbow_angle'.
    - fps (float): Frames per second.
    - min_release_time (float): Minimum expected time (in seconds) for release.
    - max_release_time (float): Maximum expected time (in seconds) for release.
    
    Returns:
    - int or None: Frame index of release or None if not found within the range.
    """
    try:
        # Calculate frame indices based on time
        min_frame = int(min_release_time * fps)
        max_frame = int(max_release_time * fps)
        
        # Ensure pose_df has enough frames
        if len(pose_df) < min_frame:
            logger.warning("Pose data is too short for release frame detection.")
            return None
        
        # Limit the search to the specified time window
        pose_window = pose_df.iloc[min_frame:max_frame]
        
        if pose_window.empty:
            logger.warning("Pose window for release frame detection is empty.")
            return None
        
        # Compute average elbow angle per frame
        pose_window = pose_window.copy()
        pose_window['avg_elbow_angle'] = pose_window[['left_elbow_angle', 'right_elbow_angle']].mean(axis=1)
        
        # Identify the frame with the maximum average elbow angle within the window
        release_idx = pose_window['avg_elbow_angle'].idxmax()
        
        # Optional: Additional validation (e.g., velocity thresholds)
        # Example: Ensure the velocity at release is above a certain threshold
        
        release_time = release_idx / fps
        logger.info(f"Detected release frame at index {release_idx} (Time: {release_time:.2f} s)")
        
        return release_idx
    except Exception as e:
        logger.error(f"Error finding release frame: {str(e)}")
        return None

    
def calculate_quadratic_fit(release_phase):
    """
    Calculate curvature of the ball's path using quadratic fit and return curvature and apex height.
    """
    try:
        # Fit a quadratic curve: y = ax^2 + bx + c
        if len(release_phase) < 3:
            return np.nan, np.nan
        z = np.polyfit(release_phase['center_x'], release_phase['center_y'], 2)
        a, b, c = z
        curvature = 2 * a  # Simplistic curvature measure from quadratic coefficient

        # Calculate apex (vertex of the parabola)
        apex_x = -b / (2 * a) if a != 0 else np.nan
        apex_y = a * apex_x**2 + b * apex_x + c if not np.isnan(apex_x) else np.nan
        apex_height = apex_y

        return curvature, apex_height
    except Exception as e:
        logger.error(f"Error calculating quadratic fit: {str(e)}")
        return np.nan, np.nan

def calculate_lateral_curvature(spin_df):
    """
    Calculate lateral curvature based on spin axis data.
    """
    try:
        # Assuming 'spin_axis' column exists and represents lateral spin direction
        if 'spin_axis' not in spin_df.columns:
            return np.nan
        # Calculate consistency as standard deviation of spin_axis
        lateral_spin_std = spin_df['spin_axis'].std()
        return lateral_spin_std
    except Exception as e:
        logger.error(f"Error calculating lateral curvature: {str(e)}")
        return np.nan


def is_valid_metrics(metrics, kpi_ranges):
    """
    Check if the metrics are within realistic ranges.
    """
    try:
        for kpi, (min_val, max_val) in kpi_ranges.items():
            value = metrics.get(kpi, np.nan)
            if pd.isna(value):
                return False
            if not (min_val <= value <= max_val):
                return False
        return True
    except:
        return False
    

def calculate_quadratic_fit(release_phase):
    """
    Calculate curvature of the ball's path using quadratic fit and return curvature and apex height.
    """
    try:
        # Fit a quadratic curve: y = ax^2 + bx + c
        if len(release_phase) < 3:
            return np.nan, np.nan
        z = np.polyfit(release_phase['center_x'], release_phase['center_y'], 2)
        a, b, c = z
        curvature = 2 * a  # Simplistic curvature measure from quadratic coefficient

        # Calculate apex (vertex of the parabola)
        apex_x = -b / (2 * a) if a != 0 else np.nan
        apex_y = a * apex_x**2 + b * apex_x + c if not np.isnan(apex_x) else np.nan
        apex_height = apex_y

        return curvature, apex_height
    except Exception as e:
        logger.error(f"Error calculating quadratic fit: {str(e)}")
        return np.nan, np.nan

def calculate_lateral_curvature(spin_df):
    """
    Calculate lateral curvature based on spin axis data.
    """
    try:
        # Assuming 'spin_axis' column exists and represents lateral spin direction
        if 'spin_axis' not in spin_df.columns:
            return np.nan
        # Calculate consistency as standard deviation of spin_axis
        lateral_spin_std = spin_df['spin_axis'].std()
        return lateral_spin_std
    except Exception as e:
        logger.error(f"Error calculating lateral curvature: {str(e)}")
        return np.nan
    

def clean_metrics(df_shots):
    """
    Clean and cap unrealistic metric values within specified ranges.
    
    Parameters:
    - df_shots (pd.DataFrame): DataFrame containing shot metrics.
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame with capped values.
    """
    # Define realistic ranges
    df_shots.loc[:, 'release_time_s'] = df_shots['release_time_s'].clip(lower=1, upper=5)
    df_shots.loc[:, 'release_velocity_in_per_s'] = df_shots['release_velocity_in_per_s'].clip(lower=120, upper=240)
    df_shots.loc[:, 'release_height_in'] = df_shots['release_height_in'].clip(lower=84, upper=120)
    
    # Add more capping as needed for other metrics
    # Example:
    # df_shots.loc[:, 'some_metric'] = df_shots['some_metric'].clip(lower=min_val, upper=max_val)
    
    return df_shots

def is_valid_metrics(metrics, kpi_ranges):
    """
    Check if the metrics are within realistic ranges.
    """
    try:
        for kpi, (min_val, max_val) in kpi_ranges.items():
            value = metrics.get(kpi, np.nan)
            if pd.isna(value):
                return False
            if not (min_val <= value <= max_val):
                return False
        return True
    except:
        return False


def split_spins(spin_df):
    """
    Split the spin_df into individual spins based on frame resets.
    A new spin is detected when the frame number decreases.
    """
    try:
        # Assume spin_df has a 'frame' column
        if 'frame' not in spin_df.columns:
            logger.warning("spin_df does not contain 'frame' column.")
            return []

        spin_df = spin_df.sort_values('frame').reset_index(drop=True)
        spins = []
        current_spin = []

        previous_frame = None
        for _, row in spin_df.iterrows():
            current_frame = row['frame']
            if previous_frame is not None and current_frame < previous_frame:
                if current_spin:
                    spins.append(pd.DataFrame(current_spin))
                    current_spin = []
            current_spin.append(row)
            previous_frame = current_frame

        # Append the last spin
        if current_spin:
            spins.append(pd.DataFrame(current_spin))

        return spins
    except Exception as e:
        logger.error(f"Error splitting spins: {str(e)}")
        return []
    

    
def compute_velocity(positions, fps, smoothing_window=7, polyorder=2):
    """
    Compute velocity using a Savitzky–Golay filter to smooth positions 
    and then take the finite difference.

    :param positions: Pandas Series or numpy array of positions (e.g. center_x or center_y).
    :param fps: Frames per second (float).
    :param smoothing_window: Window length for Savitzky–Golay filter (odd int).
    :param polyorder: Polynomial order for Savitzky–Golay filter.
    :return: Numpy array of velocities (same length as positions).
    """
    data = positions.to_numpy() if hasattr(positions, 'to_numpy') else np.array(positions)
    n = len(data)
    # Ensure window length does not exceed data length and is odd
    if n < 3:
        return np.zeros_like(data)  # Not enough data to differentiate meaningfully
    if smoothing_window > n:
        smoothing_window = n if n % 2 == 1 else n - 1
    elif smoothing_window % 2 == 0:
        smoothing_window += 1  # Make sure it's odd

    # Apply Savitzky–Golay to smooth positions
    smoothed = savgol_filter(data, window_length=smoothing_window, polyorder=polyorder)

    # Finite difference => velocity
    # spacing = 1/fps
    velocity = np.gradient(smoothed, 1.0/fps)
    return velocity


def improved_find_release_frame(pose_df, ball_df=None, fps=25, max_release_time=3.0):
    """
    An improved approach to finding the release frame by combining 
    elbow-angle smoothing and (optionally) ball velocity data.

    :param pose_df: DataFrame with 'left_elbow_angle' and 'right_elbow_angle'.
    :param ball_df: (Optional) DataFrame with 'vx', 'vy' columns for ball velocity.
    :param fps: Frames per second (float).
    :param max_release_time: Maximum allowable release time in seconds.
    :return: The index of the release frame, or None if not found.
    """
    if pose_df.empty:
        return None
    
    # 1. Smooth elbow angles
    #    We'll use a simple rolling mean here; you can also use Savitzky–Golay if you like.
    pose_df['left_elbow_smooth'] = pose_df['left_elbow_angle'].rolling(window=5, min_periods=1, center=True).mean()
    pose_df['right_elbow_smooth'] = pose_df['right_elbow_angle'].rolling(window=5, min_periods=1, center=True).mean()

    # 2. Mean of left & right elbow angles
    pose_df['mean_elbow'] = (pose_df['left_elbow_smooth'] + pose_df['right_elbow_smooth']) / 2.0

    # 3. Candidate release = index of max elbow extension
    release_idx = pose_df['mean_elbow'].idxmax()

    # 4. Check if beyond max_release_time
    if release_idx is not None:
        release_time = release_idx / fps
        if release_time > max_release_time:
            # Possibly no valid release found (late in the segment)
            return None

    # 5. (Optional) You could check ball velocity sign changes or thresholds 
    #    to refine the exact release frame. For now we keep it simple.

    return release_idx

def load_and_process_shot_data(job, s3_client, bucket, user_email, fps, rim_position):
    """
    Load and process tracking data for shot KPIs with improved smoothing,
    robust scale factor, outlier filtering, and better release detection.

    Parameters:
    - job (dict): Job information dictionary.
    - s3_client: AWS S3 client.
    - bucket (str): S3 bucket name.
    - user_email (str): User's email.
    - fps (float): Frames per second.
    - rim_position (dict): Rim position in real-world units (inches).

    Returns:
    - list of dict: List of metrics dictionaries for each segment.
    """
    all_metrics = []
    try:
        # 1. List all segments for this job
        segments = list_job_segments(s3_client, bucket, user_email, job['JobID'])
        if not segments:
            logger.warning(f"No segments found for Job ID: {job['JobID']}.")
            return []

        # 2. Iterate over each segment
        for seg_id in segments:
            try:
                # -----------------------------------------------------
                # A) Load the segment CSV
                # -----------------------------------------------------
                df = load_final_output_csv(s3_client, bucket, user_email, job['JobID'], seg_id)
                if df.empty:
                    logger.warning(f"Segment {seg_id} in Job ID {job['JobID']} is empty.")
                    continue

                # Ensure required pose columns exist (fill with NaN if missing)
                pose_cols = ['left_elbow_angle', 'right_elbow_angle']
                missing_cols = [col for col in pose_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(
                        f"Missing pose columns {missing_cols} in segment {seg_id} (Job {job['JobID']})."
                    )
                    for col in missing_cols:
                        df[col] = np.nan

                # -----------------------------------------------------
                # B) Extract and Scale Ball Tracking
                # -----------------------------------------------------
                ball_df = calculate_ball_center(df)
                if ball_df.empty or 'center_x' not in ball_df.columns or 'center_y' not in ball_df.columns:
                    logger.warning(
                        f"No valid ball tracking in segment {seg_id} (Job {job['JobID']})."
                    )
                    continue

                # 1) Robust diameter => scale factor
                diam_col = 'ball_diameter_pixels'
                if diam_col not in ball_df.columns:
                    ball_df[diam_col] = np.nan

                ball_diameters = ball_df[diam_col].dropna()
                if len(ball_diameters) < 5:
                    logger.warning(
                        f"Too few ball diameter samples in seg {seg_id}. Using scale=1."
                    )
                    scale_factor = 1.0
                else:
                    median_diam = ball_diameters.median()
                    mad = 1.4826 * np.median(np.abs(ball_diameters - median_diam))
                    lower_bound = median_diam - 2.0 * mad
                    upper_bound = median_diam + 2.0 * mad
                    filtered = ball_diameters[(ball_diameters >= lower_bound) &
                                              (ball_diameters <= upper_bound)]
                    if filtered.empty:
                        logger.warning(
                            f"All diameters out of robust range in seg {seg_id}. Using scale=1."
                        )
                        scale_factor = 1.0
                    else:
                        robust_diam = filtered.median()
                        # Real-world ball diameter ~9.4 inches
                        ball_diameter_in = 29.5 / np.pi
                        raw_scale = ball_diameter_in / robust_diam
                        scale_factor = np.clip(raw_scale, 0.01, 2.0)

                logger.info(f"Seg {seg_id}: final scale_factor = {scale_factor:.4f} in/pixel")

                # Apply scale factor
                ball_df['center_x'] *= scale_factor
                ball_df['center_y'] *= scale_factor

                # 2) Compute velocities with smoothing
                ball_df.reset_index(drop=True, inplace=True)
                ball_df['vx'] = compute_velocity(ball_df['center_x'], fps, smoothing_window=7, polyorder=2)
                ball_df['vy'] = compute_velocity(ball_df['center_y'], fps, smoothing_window=7, polyorder=2)

                # -----------------------------------------------------
                # C) Release-Frame Detection
                # -----------------------------------------------------
                pose_segment = df[['left_elbow_angle', 'right_elbow_angle']].copy()
                pose_segment.dropna(subset=['left_elbow_angle','right_elbow_angle'], how='any', inplace=True)

                release_idx = improved_find_release_frame(
                    pose_df=pose_segment,
                    ball_df=ball_df,
                    fps=fps,
                    max_release_time=3.0
                )

                # -----------------------------------------------------
                # D) Compute Shot KPIs
                # -----------------------------------------------------
                if release_idx is None or release_idx >= len(ball_df):
                    # If invalid or out-of-range => metrics = NaN
                    logger.warning(
                        f"No valid release frame found in seg {seg_id} => all KPIs = NaN"
                    )
                    shot_distance = np.nan
                    release_time = np.nan
                    release_curvature = np.nan
                    apex_height = np.nan
                    arc = np.nan
                    release_angle = np.nan
                    release_height = np.nan
                    vx = vy = np.nan
                else:
                    # Valid release => compute final metrics
                    rx = ball_df.loc[release_idx, 'center_x']
                    ry = ball_df.loc[release_idx, 'center_y']
                    vx = ball_df.loc[release_idx, 'vx']
                    vy = ball_df.loc[release_idx, 'vy']

                    release_time = release_idx / fps
                    shot_distance = np.sqrt((rx - rim_position['x'])**2 + 
                                            (ry - rim_position['y'])**2)
                    release_phase = ball_df.loc[:release_idx]
                    release_curvature, apex_height = calculate_quadratic_fit(release_phase)

                    release_height = ry
                    if pd.isna(apex_height):
                        arc = np.nan
                    else:
                        arc = max(apex_height - release_height, 0.0)

                    if (vx == 0) or pd.isna(vx):
                        release_angle = 90.0
                    else:
                        release_angle = abs(np.degrees(np.arctan2(vy, vx)))

                    logger.info(
                        f"Seg {seg_id}: release_idx={release_idx}, time={release_time:.2f}s, "
                        f"dist_in={shot_distance:.1f}, apex_in={apex_height:.1f}"
                    )

                # 2) Build the metrics dictionary
                release_vel_in_s = np.hypot(vx, vy) if pd.notna(vx) and pd.notna(vy) else np.nan
                release_vel_ft_s = release_vel_in_s / 12.0 if pd.notna(release_vel_in_s) else np.nan

                metrics = {
                    'segment_id': seg_id,
                    'release_velocity_in_per_s': release_vel_in_s,
                    'release_velocity_ft_per_s': release_vel_ft_s,
                    'release_velocity': release_vel_ft_s,  # same as above
                    'release_height_in': release_height,
                    'release_height_ft': release_height / 12.0 if pd.notna(release_height) else np.nan,
                    'release_height': release_height / 12.0 if pd.notna(release_height) else np.nan,
                    'release_angle': release_angle,
                    'shot_distance_in': shot_distance,
                    'shot_distance_ft': shot_distance / 12.0 if pd.notna(shot_distance) else np.nan,
                    'apex_height_in': apex_height,
                    'apex_height_ft': apex_height / 12.0 if pd.notna(apex_height) else np.nan,
                    'arc_in': arc,
                    'arc_ft': arc / 12.0 if pd.notna(arc) else np.nan,
                    'release_time_s': release_time,
                    'release_consistency': np.nan,  # placeholder
                    'release_curvature': release_curvature,
                    'lateral_release_curvature': np.nan,  # placeholder
                    'form_score': np.nan,                 # placeholder
                    'date': pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')
                }

                # -----------------------------------------------------
                # E) Basic Sanity Checks => Invalidate out-of-bounds
                # -----------------------------------------------------
                rv_in_s = metrics['release_velocity_in_per_s']
                if pd.notna(rv_in_s) and not (120 <= rv_in_s <= 240):
                    logger.warning(f"Unrealistic velocity {rv_in_s} in/s => set to NaN")
                    metrics['release_velocity'] = np.nan

                rh_in = metrics['release_height_in']
                if pd.notna(rh_in) and not (84 <= rh_in <= 120):
                    logger.warning(f"Unrealistic height {rh_in} in => set to NaN")
                    metrics['release_height'] = np.nan

                rt_s = metrics['release_time_s']
                if pd.notna(rt_s) and not (1 <= rt_s <= 3):
                    logger.warning(f"Unrealistic release time {rt_s} s => set to NaN")
                    metrics['release_time_s'] = np.nan

                all_metrics.append(metrics)
                logger.info(f"Appended metrics for seg {seg_id}, job={job['JobID']}.")

            except Exception as seg_error:
                logger.error(f"Error processing segment {seg_id}: {seg_error}")
                continue

    except Exception as e:
        logger.error(f"Job processing error for {job['JobID']}: {e}")
        return []

    return all_metrics




spin_kpi_ranges = {
    'spin_magnitude': (0, 500),      # Example range in arbitrary units
    'spin_consistency': (0, 100),    # Percentage
    'spin_duration': (0, 10)         # Seconds
    # Add more KPIs as needed
}


def load_and_process_spin_data(job, s3_client, bucket, user_email, fps, spin_kpi_ranges):
    """Load and process spin data independently"""
    spin_metrics_list = []
    try:
        spin_df = load_spin_axis_csv(s3_client, bucket, user_email, job['JobID'])
        if spin_df.empty:
            return []

        # Split into individual spins based on frame resets
        spins = split_spins(spin_df)
        if not spins:
            return []

        for idx, spin in enumerate(spins, start=1):
            try:
                # Determine the correct column name for spin_axis
                spin_axis_col = 'spin_axis' if 'spin_axis' in spin.columns else 'ex'

                if spin_axis_col not in spin.columns:
                    continue

                # Calculate spin KPIs
                spin_magnitude = spin[spin_axis_col].mean()
                spin_std = spin[spin_axis_col].std()
                spin_duration = len(spin) / fps  # in seconds

                metrics = {
                    'spin_magnitude': spin_magnitude,  # Arbitrary units
                    'spin_consistency': spin_std,      # Standard deviation
                    'spin_duration': spin_duration,    # seconds
                    'date': pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d'),
                    'spin_id': f"{job['JobID']}_spin_{idx}"
                }

                # Validate spin metrics
                valid_spin = all([
                    spin_kpi_ranges['spin_magnitude'][0] <= spin_magnitude <= spin_kpi_ranges['spin_magnitude'][1],
                    spin_kpi_ranges['spin_consistency'][0] <= spin_std <= spin_kpi_ranges['spin_consistency'][1],
                    spin_kpi_ranges['spin_duration'][0] <= spin_duration <= spin_kpi_ranges['spin_duration'][1],
                ])

                if valid_spin:
                    spin_metrics_list.append(metrics)

            except Exception as spin_error:
                logger.error(f"Error processing spin {idx} in Job {job['JobID']}: {str(spin_error)}")
                continue

    except Exception as e:
        logger.error(f"Spin data processing error for job {job['JobID']}: {str(e)}")
        return []

    return spin_metrics_list

# -------------------------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------------------------
def load_and_process_shot_data(job, s3_client, bucket, user_email, fps, rim_position):
    """
    Load and process shot data from a job (for pose or data_file sources) and return KPI metrics.
    """
    all_metrics = []
    segments = list_job_segments(s3_client, bucket, user_email, job['JobID'])
    if not segments:
        logger.warning(f"No segments found for Job ID: {job['JobID']}.")
        return []
    for seg_id in segments:
        try:
            df = load_final_output_csv(s3_client, bucket, user_email, job['JobID'], seg_id)
            if df.empty:
                logger.warning(f"Segment {seg_id} in Job {job['JobID']} is empty.")
                continue
            # Ensure pose columns exist; if missing, fill with NaN.
            for col in ['left_elbow_angle', 'right_elbow_angle']:
                if col not in df.columns:
                    df[col] = np.nan
            # Process the shot data
            df_proc = map_body_columns(df)
            df_proc = compute_joint_angles(df_proc)
            if 'frame' not in df_proc.columns:
                df_proc['frame'] = np.arange(len(df_proc))
            release_frame = 0  # For single-shot, assume release frame is 0 (or use your logic)
            df_proc = add_relative_frame(df_proc, release_frame)
            # Use the first 20 frames as the release phase for curvature calculations
            release_phase = df_proc[df_proc['frame'] < 20]
            release_curvature, apex_height = calculate_quadratic_fit(release_phase)
            try:
                shot_metric = {
                    "Release Velocity": df_proc.get('release_velocity_in_per_s', pd.Series([np.nan])).iloc[-1] / 12.0,
                    "Release Height": df_proc.get('release_height_in', pd.Series([np.nan])).iloc[-1] / 12.0,
                    "Release Angle": df_proc.get('release_angle', pd.Series([np.nan])).iloc[-1],
                    "Shot Distance": df_proc.get('shot_distance_ft', pd.Series([np.nan])).iloc[-1],
                    "Apex Height": apex_height / 12.0 if not pd.isna(apex_height) else np.nan,
                    "Release Time": df_proc.get('release_time_s', pd.Series([np.nan])).iloc[-1],
                    "Release Quality": df_proc.get('release_quality', pd.Series([np.nan])).iloc[-1],
                    "Release Consistency": np.nan,
                    "Release Curvature": release_curvature,
                    "Lateral Release Curvature": np.nan
                }
                shot_metric['JobID'] = job['JobID']
                shot_metric['Date'] = pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')
                all_metrics.append(shot_metric)
            except Exception as e:
                logger.error(f"Error computing KPIs for Job {job['JobID']}: {e}")
        except Exception as seg_error:
            logger.error(f"Error processing segment {seg_id}: {seg_error}")
            continue
    return all_metrics

def load_and_process_spin_data(job, s3_client, bucket, user_email, fps, spin_kpi_ranges):
    """
    Load and process spin data from a job and return spin KPIs.
    """
    spin_metrics_list = []
    spin_df = load_spin_axis_csv(s3_client, bucket, user_email, job['JobID'])
    if spin_df.empty:
        return []
    spins = split_spins(spin_df)
    if not spins:
        return []
    for idx, spin in enumerate(spins, start=1):
        spin_axis_col = 'spin_axis' if 'spin_axis' in spin.columns else 'ex'
        if spin_axis_col not in spin.columns:
            continue
        spin_magnitude = spin[spin_axis_col].mean()
        spin_std = spin[spin_axis_col].std()
        spin_duration = len(spin) / fps
        metrics = {
            'spin_magnitude': spin_magnitude,
            'spin_consistency': spin_std,
            'spin_duration': spin_duration,
            'date': pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d'),
            'spin_id': f"{job['JobID']}_spin_{idx}"
        }
        valid_spin = all([
            spin_kpi_ranges['spin_magnitude'][0] <= spin_magnitude <= spin_kpi_ranges['spin_magnitude'][1],
            spin_kpi_ranges['spin_consistency'][0] <= spin_std <= spin_kpi_ranges['spin_consistency'][1],
            spin_kpi_ranges['spin_duration'][0] <= spin_duration <= spin_kpi_ranges['spin_duration'][1],
        ])
        if valid_spin:
            spin_metrics_list.append(metrics)
    return spin_metrics_list

# -------------------------------------------------------------------------
# Utility Functions for Pose/Ball
# -------------------------------------------------------------------------
def angle_2d(a, b, c):
    try:
        ab = np.array([a[0] - b[0], a[1] - b[1]])
        cb = np.array([c[0] - b[0], c[1] - b[1]])
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        if ab_norm == 0 or cb_norm == 0:
            return np.nan
        cos_val = np.dot(ab, cb) / (ab_norm * cb_norm)
        return degrees(np.arccos(np.clip(cos_val, -1, 1)))
    except:
        return np.nan

def map_body_columns(df):
    df.columns = df.columns.str.strip()
    return df  # Adjust if you have real rename logic

def compute_joint_angles(df):
    # Example partial logic; expand for all angles
    # e.g. left_elbow_angle = (left_shoulder_x, left_shoulder_y) etc.
    if 'left_shoulder_x' in df.columns and 'left_shoulder_y' in df.columns \
       and 'left_elbow_x' in df.columns and 'left_elbow_y' in df.columns \
       and 'left_wrist_x' in df.columns and 'left_wrist_y' in df.columns:
        df['left_elbow_angle'] = df.apply(
            lambda row: angle_2d(
                (row['left_shoulder_x'], row['left_shoulder_y']),
                (row['left_elbow_x'], row['left_elbow_y']),
                (row['left_wrist_x'], row['left_wrist_y'])
            ), axis=1
        )
    return df

def separate_pose_and_ball_tracking(df_segment):
    detection_cols = ['track_id', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    pose_cols = [c for c in df_segment.columns if c not in detection_cols]
    pose_df = df_segment[pose_cols].copy()
    if 'frame' in df_segment.columns:
        pose_df['frame'] = df_segment['frame']
        pose_df = pose_df.groupby('frame').first().reset_index()
    else:
        pose_df = pd.DataFrame()
    if 'class_name' in df_segment.columns:
        ball_tracking_df = df_segment[df_segment['class_name'].str.lower() == 'basketball'].copy()
        if not ball_tracking_df.empty:
            ball_tracking_df = ball_tracking_df.sort_values('confidence', ascending=False).drop_duplicates(subset=['frame'], keep='first')
    else:
        ball_tracking_df = pd.DataFrame()
    return pose_df, ball_tracking_df

def clean_pose_data(df):
    df = map_body_columns(df)
    df = compute_joint_angles(df)
    return df

def clean_ball_tracking_data(df):
    for c in ['xmin', 'ymin', 'xmax', 'ymax']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[['xmin','ymin','xmax','ymax']] = df[['xmin','ymin','xmax','ymax']].interpolate(method='linear', limit_direction='both')
    return df

def calculate_ball_center(df):
    req_cols = ['xmin','ymin','xmax','ymax']
    if all(col in df.columns for col in req_cols):
        df['center_x'] = (df['xmin'] + df['xmax']) / 2
        df['center_y'] = (df['ymin'] + df['ymax']) / 2
        df['ball_diameter_pixels'] = df['xmax'] - df['xmin']
    else:
        df['center_x'] = np.nan
        df['center_y'] = np.nan
        df['ball_diameter_pixels'] = np.nan
    return df

def detect_release_frame_pose(df):
    if 'left_elbow_y' not in df.columns or 'right_elbow_y' not in df.columns:
        return None
    df['max_elbow_y'] = df[['left_elbow_y','right_elbow_y']].max(axis=1)
    max_y_idx = df['max_elbow_y'].idxmax()
    if 'frame' in df.columns and max_y_idx in df.index:
        return df.loc[max_y_idx, 'frame']
    return None

def add_relative_frame(df, release_frame):
    if 'frame' not in df.columns:
        return df
    if release_frame is not None:
        df['relative_frame'] = df['frame'] - release_frame
    else:
        df['relative_frame'] = df['frame'] - df['frame'].min()
    return df

def trim_shot(df, release_frame, frames_before=20, frames_after=5):
    if release_frame is None or 'frame' not in df.columns:
        return None
    start = max(release_frame - frames_before, df['frame'].min())
    end = min(release_frame + frames_after, df['frame'].max())
    trimmed = df[(df['frame'] >= start) & (df['frame'] <= end)].copy()
    return add_relative_frame(trimmed, release_frame)

def smooth_series(series, window=11, frac=0.3):
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(np.nan, index=series.index)
    rolled = series.rolling(window=window, center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
    xvals = np.arange(len(rolled))
    try:
        smoothed = sm.nonparametric.lowess(rolled, xvals, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)
    except Exception as e:
        logger.error(f"LOWESS smoothing failed: {e}")
        return rolled

def compute_velocity(positions, fps, smoothing_window=7, polyorder=2):
    data = positions.to_numpy() if hasattr(positions, 'to_numpy') else np.array(positions)
    n = len(data)
    if n < 3:
        return np.zeros_like(data)
    if smoothing_window > n:
        smoothing_window = n if n%2==1 else n-1
    elif smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothed = savgol_filter(data, window_length=smoothing_window, polyorder=polyorder)
    return np.gradient(smoothed, 1.0/fps)

def compute_acceleration(series, dt=0.04):
    return (series.shift(-1) - series.shift(1)) / (2.0*dt)

def compute_angular_vel_accel(df, dt=0.04):
    angle_vel_cols = [c for c in df.columns if c.endswith('_vel')]
    for col in angle_vel_cols:
        df[f"{col}_accel"] = compute_acceleration(df[col], dt)
    return df

# -------------------------------------------------------------------------
# Shot KPI Calculation (Partial)
# -------------------------------------------------------------------------
def calculate_kpis_ball_tracking(df):
    # Placeholder for real logic
    if df.empty:
        return {}
    return {
        'Release Velocity': np.nan,
        'Release Frame': np.nan,
        'Release Height': np.nan,
        'Release Angle': np.nan,
        'Shot Distance': np.nan,
        'Apex Height': np.nan,
        'Release Time': np.nan,
        'Release Curvature': np.nan
    }

def calculate_kpis_pose(df):
    # Placeholder for real logic
    if df.empty:
        return {}
    return {
        'Release Velocity': np.nan,
        'Shot Distance': np.nan,
        'Release Angle': np.nan,
        'Apex Height': np.nan,
        'Release Time': np.nan,
        'Release Curvature': np.nan,
        'Asymmetry Index': np.nan,
        'Kinematic Chain Sequencing': {},
        'Kinematic Chain Sequence Score': 0,
        'Pivotal Score': np.nan
    }

def identify_biomech_phases(df, fps=25):
    if len(df) < 5:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    phases = pd.Series(["Unknown"] * len(df), index=df.index)
    release_idx = detect_release_frame_pose(df)
    if release_idx is None:
        return phases
    margin = 5
    start_frame = df['frame'].min()
    end_frame = release_idx + margin
    valid_mask = (df['frame'] >= start_frame) & (df['frame'] <= end_frame)
    phases[~valid_mask] = "Unknown"
    phases[valid_mask] = "Preparation"
    release_mask = (df['frame'] >= release_idx) & (df['frame'] <= (release_idx + 2))
    phases[release_mask] = "Release"
    return phases

def visualize_phases(df, phases, job_id, segment_id, fps=25):
    if 'relative_frame' not in df.columns:
        st.error("No 'relative_frame' in DataFrame for phase visualization.")
        return
    df = add_time_column(df, fps)
    if 'time' not in df.columns:
        st.warning("No 'time' column for phase visualization.")
        return
    fig = go.Figure()
    phase_colors = {"Preparation":"blue","Release":"red","Unknown":"gray"}
    for phase, color in phase_colors.items():
        mask = (phases==phase)
        sub_df = df[mask]
        if not sub_df.empty and 'time' in sub_df.columns:
            if 'left_elbow_angle' in sub_df.columns:
                fig.add_trace(go.Scatter(
                    x=sub_df['time'],
                    y=sub_df['left_elbow_angle'],
                    mode='lines',
                    name=f"{phase} - Left Elbow",
                    line=dict(color=color)
                ))
    fig.update_layout(title="Phases with Elbow Angle", xaxis_title="Time(s)", yaxis_title="Angle (deg)")
    st.plotly_chart(fig, use_container_width=True)

def add_time_column(df, fps=25):
    if 'relative_frame' not in df.columns:
        df['time'] = np.nan
    else:
        df['time'] = df['relative_frame'] / fps
    return df

def visualize_section(mean_df, std_df, selected_metrics, metric_section, section_title, job_id, segment_id, fps=25):
    st.markdown(f"##### {section_title} Metrics")
    options = [humanize_label(m) for m in selected_metrics]
    selected_options = st.multiselect(
        f"Select {section_title} Metrics to Visualize",
        options=options,
        default=options
    )
    metric_map = {humanize_label(m):m for m in selected_metrics}
    selected_actual = [metric_map[o] for o in selected_options if o in metric_map]
    if selected_actual:
        fig = go.Figure()
        color_cycle = [
            "#1f77b4","#ff7f0e","#2ca02c","#d62728",
            "#9467bd","#8c564b","#e377c2","#7f7f7f",
            "#bcbd22","#17becf"
        ]
        i_color = 0
        for m in selected_actual:
            if m in mean_df.columns and m in std_df.columns:
                c = color_cycle[i_color % len(color_cycle)]
                i_color += 1
                fig.add_trace(go.Scatter(
                    x=mean_df['time'],
                    y=mean_df[m],
                    mode='lines',
                    name=f"Mean - {humanize_label(m)}",
                    line=dict(width=2, color=c)
                ))
                upper = (mean_df[m] + std_df[m]).tolist()
                lower = (mean_df[m] - std_df[m]).tolist()
                fig.add_trace(go.Scatter(
                    x=list(mean_df['time']) + list(mean_df['time'][::-1]),
                    y=upper + lower[::-1],
                    fill='toself',
                    fillcolor=c.replace(")", ",0.2)").replace("rgb","rgba") if c.startswith("rgb") else 'rgba(31,119,180,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
        fig.update_layout(
            title=f"{section_title} Metrics Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Units",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No {section_title} metrics selected.")

def visualize_spin_metrics(df_spin, job_id):
    if df_spin.empty:
        return
    st.subheader(f"3D Spin Axis Visualization - Job {job_id}")
    if {'ex','ey','ez','spin_magnitude'}.issubset(df_spin.columns):
        fig = px.scatter_3d(
            df_spin,
            x='ex', y='ey', z='ez',
            color='spin_magnitude',
            color_continuous_scale='Viridis',
            title="3D Spin Axis Scatter",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    if 'frame' in df_spin.columns and 'spin_magnitude' in df_spin.columns:
        st.subheader(f"Spin Magnitude over Frames - Job {job_id}")
        fig2 = px.line(df_spin, x='frame', y='spin_magnitude', title="Spin Magnitude Over Time")
        st.plotly_chart(fig2, use_container_width=True)

def compute_spin_kpi(df_spin):
    if df_spin.empty:
        return {}
    kpis = {}
    if 'spin_magnitude' in df_spin.columns:
        kpis['avg_spin_rate'] = df_spin['spin_magnitude'].mean()
        kpis['max_spin_rate'] = df_spin['spin_magnitude'].max()
        kpis['spin_consistency'] = df_spin['spin_magnitude'].std()
    if 'ez' in df_spin.columns:
        kpis['backspin_ratio'] = (df_spin['ez'] > 0).mean()
    return kpis

def calculate_form_score(kpi_dict):
    if not kpi_dict:
        return np.nan
    weights = {
        'Release Velocity': 0.3,
        'Shot Distance': 0.2,
        'Release Angle': 0.2,
        'Apex Height': 0.1,
        'Release Quality': 0.2
    }
    score = 0
    for k, w in weights.items():
        val = kpi_dict.get(k, np.nan)
        if pd.isna(val):
            continue
        if k == 'Release Velocity':
            norm = min(val/10, 1)
        elif k == 'Shot Distance':
            norm = min(val/50, 1)
        elif k == 'Release Angle':
            norm = 1 - abs(val)/90
        elif k == 'Apex Height':
            norm = min(val/500, 1)
        elif k == 'Release Quality':
            norm = val/100
        else:
            norm = 0
        score += norm*w
    return min(100, max(0, score*100))

def calculate_consistency_percentage(kpi_list, kpi_key='Release Quality'):
    if not kpi_list:
        return np.nan
    values = [d.get(kpi_key, np.nan) for d in kpi_list if not pd.isna(d.get(kpi_key, np.nan))]
    if not values:
        return np.nan
    std_val = np.std(values)
    # Example formula: 100 - std_val => narrower the standard dev, higher the consistency
    return np.clip(100 - std_val, 0, 100)

def clean_metrics(df_shots):
    if 'release_time_s' in df_shots.columns:
        df_shots['release_time_s'] = df_shots['release_time_s'].clip(lower=1, upper=5)
    if 'release_velocity_in_per_s' in df_shots.columns:
        df_shots['release_velocity_in_per_s'] = df_shots['release_velocity_in_per_s'].clip(lower=120, upper=240)
    if 'release_height_in' in df_shots.columns:
        df_shots['release_height_in'] = df_shots['release_height_in'].clip(lower=84, upper=120)
    return df_shots

# -------------------------------------------------------------------------
# load_and_process_shot_data
# -------------------------------------------------------------------------
def load_and_process_shot_data(job, s3_client, bucket, user_email, fps, rim_position):
    all_metrics = []
    segments = list_job_segments(s3_client, bucket, user_email, job['JobID'])
    if not segments:
        logger.warning(f"No segments found for Job ID: {job['JobID']}.")
        return []
    for seg_id in segments:
        try:
            df = load_final_output_csv(s3_client, bucket, user_email, job['JobID'], seg_id)
            if df.empty:
                logger.warning(f"Segment {seg_id} in Job {job['JobID']} is empty.")
                continue
            for col in ['left_elbow_angle', 'right_elbow_angle']:
                if col not in df.columns:
                    df[col] = np.nan
            df_proc = map_body_columns(df)
            df_proc = compute_joint_angles(df_proc)
            if 'frame' not in df_proc.columns:
                df_proc['frame'] = np.arange(len(df_proc))
            release_frame = 0
            df_proc = add_relative_frame(df_proc, release_frame)
            # Example partial logic
            shot_metric = {
                "Release Velocity": df_proc.get('release_velocity_in_per_s', pd.Series([np.nan])).iloc[-1]/12.0 if not df_proc.empty else np.nan,
                "Release Height": df_proc.get('release_height_in', pd.Series([np.nan])).iloc[-1]/12.0 if not df_proc.empty else np.nan,
                "Release Angle": df_proc.get('release_angle', pd.Series([np.nan])).iloc[-1] if not df_proc.empty else np.nan,
                "Shot Distance": df_proc.get('shot_distance_ft', pd.Series([np.nan])).iloc[-1] if not df_proc.empty else np.nan,
                "Apex Height": np.nan,
                "Release Time": df_proc.get('release_time_s', pd.Series([np.nan])).iloc[-1] if not df_proc.empty else np.nan,
                "Release Quality": df_proc.get('release_quality', pd.Series([np.nan])).iloc[-1] if not df_proc.empty else np.nan,
                "Release Consistency": np.nan,
                "Release Curvature": np.nan,
                "Lateral Release Curvature": np.nan
            }
            shot_metric["JobID"] = job["JobID"]
            shot_metric["Date"] = pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')
            all_metrics.append(shot_metric)
        except Exception as seg_error:
            logger.error(f"Error processing segment {seg_id}: {seg_error}")
            continue
    return all_metrics

# -------------------------------------------------------------------------
# load_and_process_spin_data
# -------------------------------------------------------------------------
def load_and_process_spin_data(job, s3_client, bucket, user_email, fps, spin_kpi_ranges):
    spin_metrics_list = []
    try:
        df_spin = load_spin_axis_csv(s3_client, bucket, user_email, job['JobID'])
        if df_spin.empty:
            return []
        from math import floor
        def split_spins_local(spin_df):
            if 'frame' not in spin_df.columns:
                return []
            spin_df = spin_df.sort_values('frame').reset_index(drop=True)
            spins = []
            current_spin = []
            prev_frame = None
            for _, row in spin_df.iterrows():
                cf = row['frame']
                if prev_frame is not None and cf < prev_frame:
                    if current_spin:
                        spins.append(pd.DataFrame(current_spin))
                        current_spin = []
                current_spin.append(row)
                prev_frame = cf
            if current_spin:
                spins.append(pd.DataFrame(current_spin))
            return spins
        spins = split_spins_local(df_spin)
        if not spins:
            return []
        for idx, spin in enumerate(spins, start=1):
            spin_axis_col = 'spin_axis' if 'spin_axis' in spin.columns else 'ex'
            if spin_axis_col not in spin.columns:
                continue
            spin_magnitude = spin[spin_axis_col].mean()
            spin_std = spin[spin_axis_col].std()
            spin_duration = len(spin) / fps
            metrics = {
                'spin_magnitude': spin_magnitude,
                'spin_consistency': spin_std,
                'spin_duration': spin_duration,
                'date': pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d'),
                'spin_id': f"{job['JobID']}_spin_{idx}"
            }
            valid_spin = all([
                spin_kpi_ranges['spin_magnitude'][0] <= spin_magnitude <= spin_kpi_ranges['spin_magnitude'][1],
                spin_kpi_ranges['spin_consistency'][0] <= spin_std <= spin_kpi_ranges['spin_consistency'][1],
                spin_kpi_ranges['spin_duration'][0] <= spin_duration <= spin_kpi_ranges['spin_duration'][1],
            ])
            if valid_spin:
                spin_metrics_list.append(metrics)
    except Exception as e:
        logger.error(f"Spin data processing error for job {job['JobID']}: {e}")
        return []
    return spin_metrics_list

# -------------------------------------------------------------------------
# Main App
# -------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Pivotal Motion Visualizer", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Pivotal Motion Visualizer - Overview</h1>", unsafe_allow_html=True)

    if not st.session_state.get('authenticated', False):
        with st.form("login_form"):
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            handle_login(email, password)
        return
    else:
        with st.sidebar:
            st.header("User Information")
            st.write(f"**Username:** {st.session_state['username']}")
            st.write(f"**Email:** {st.session_state['user_email']}")
            if st.button("Logout"):
                st.session_state['authenticated'] = False
                st.experimental_rerun()

    user_email = st.session_state['user_email']
    jobs_regular = fetch_user_completed_jobs(user_email)
    jobs_data_file = fetch_user_completed_data_file_jobs(user_email)
    jobs = jobs_regular + jobs_data_file
    if not jobs:
        st.info("No completed jobs found for this user.")
        return

    teams = sorted({humanize_label(j.get('Team', 'N/A')) for j in jobs if humanize_label(j.get('Team', 'N/A')) != "N/A"})
    player_names = sorted({humanize_label(j.get('PlayerName', 'Unknown')) for j in jobs})
    shooting_types = sorted({humanize_label(j.get('ShootingType', 'N/A')) for j in jobs if humanize_label(j.get('ShootingType', 'N/A')) != "N/A"})
    dates = sorted({pd.to_datetime(int(j['UploadTimestamp']), unit='s').strftime('%Y-%m-%d') for j in jobs if j.get('UploadTimestamp')})

    with st.sidebar:
        st.header("Filters")
        team_filter = st.selectbox("Team", ["All"] + teams)
        player_filter = st.selectbox("Player Name", ["All"] + player_names)
        shooting_filter = st.selectbox("Shot Type", ["All"] + shooting_types)
        date_filter = st.selectbox("Date", ["All"] + dates)

    filtered_jobs = jobs
    if team_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get('Team', 'N/A')) == team_filter]
    if player_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get('PlayerName', 'Unknown')) == player_filter]
    if shooting_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get('ShootingType', 'N/A')) == shooting_filter]
    if date_filter != "All":
        filtered_jobs = [
            j for j in filtered_jobs
            if j.get('UploadTimestamp')
            and pd.to_datetime(int(j['UploadTimestamp']), unit='s').strftime('%Y-%m-%d') == date_filter
        ]

    if not filtered_jobs:
        st.info("No jobs match the selected filters.")
        return

    job_options = [
        f"{humanize_label(j['Source'])} | Player: {humanize_label(j['PlayerName'])} | "
        f"Team: {humanize_label(j['Team'])} | Date: {pd.to_datetime(int(j['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')} | "
        f"Job ID: {j['JobID']}"
        for j in filtered_jobs
    ]
    selected_job_options = st.multiselect("Select Jobs", options=job_options)
    selected_jobs = []
    for sel in selected_job_options:
        job_id = sel.split("Job ID: ")[-1]
        job = next((j for j in filtered_jobs if j['JobID'] == job_id), None)
        if job:
            selected_jobs.append(job)

    if not selected_jobs:
        st.info("No jobs selected for visualization.")
        return

    show_brand_header(
        [j['PlayerName'] for j in selected_jobs],
        list({j['Team'] for j in selected_jobs if j['Team'] != 'N/A'})
    )

    main_tabs = st.tabs(["Overview", "Shot Selection", "Pose Analysis", "Ball Tracking", "Spin Analysis"])
    FPS = 25
    BASKET_HEIGHT_IN = 120
    RIM_POSITION = {'x': 0, 'y': BASKET_HEIGHT_IN}
    spin_kpi_ranges = {'spin_magnitude': (0, 500), 'spin_consistency': (0, 100), 'spin_duration': (0, 10)}

    shot_metrics_list = []
    spin_metrics_list = []

    for job in selected_jobs:
        src = job['Source'].lower()
        if src in ['pose_video', 'data_file']:
            metrics = load_and_process_shot_data(job, s3_client, st.secrets["BUCKET_NAME"], user_email, FPS, RIM_POSITION)
            shot_metrics_list.extend(metrics)
        if src == 'spin_video':
            metrics = load_and_process_spin_data(job, s3_client, st.secrets["BUCKET_NAME"], user_email, FPS, spin_kpi_ranges)
            spin_metrics_list.extend(metrics)

    if not shot_metrics_list:
        st.warning("No valid shot metrics computed from the selected jobs.")
        st.stop()

    df_shots_metrics = pd.DataFrame(shot_metrics_list)
    df_shots_metrics = clean_metrics(df_shots_metrics)
    form_score = calculate_form_score(df_shots_metrics.mean(numeric_only=True).to_dict())
    averages_shots = df_shots_metrics.mean(numeric_only=True).to_dict()
    std_devs_shots = df_shots_metrics.std(numeric_only=True).to_dict()
    averages_shots["Form Score"] = form_score
    std_devs_shots["Form Score"] = 0

    if spin_metrics_list:
        df_spins = pd.DataFrame(spin_metrics_list)
        averages_spins = {k: df_spins[k].mean() for k in ['spin_magnitude','spin_consistency','spin_duration'] if k in df_spins.columns}
        std_devs_spins = {k: df_spins[k].std() for k in ['spin_magnitude','spin_consistency','spin_duration'] if k in df_spins.columns}
    else:
        df_spins = pd.DataFrame()
        averages_spins, std_devs_spins = {}, {}

    with main_tabs[0]:
        st.subheader("Overview")
        if not df_shots_metrics.empty:
            st.write("**Shot Metrics Loaded:**")
            st.dataframe(df_shots_metrics.head())
        else:
            st.write("No Shot Metrics Available.")
        if not df_spins.empty:
            st.write("**Spin Metrics Loaded:**")
            st.dataframe(df_spins.head())
        else:
            st.write("No Spin Metrics Available.")
        if not df_shots_metrics.empty:
            kpi_dict = df_shots_metrics.mean(numeric_only=True).to_dict()
            fs = calculate_form_score(kpi_dict)
            st.write(f"**Form Score (Shots):** {fs:.2f}")
        if not df_spins.empty:
            st.write("Spin data loaded; see Spin Analysis tab.")

    with main_tabs[1]:
        st.subheader("Shot Selection")
        st.write("Select a job/segment for detailed inspection.")

    with main_tabs[2]:
        st.subheader("Pose Analysis")
        st.write("Aggregated pose data and joint metrics.")

    with main_tabs[3]:
        st.subheader("Ball Tracking")
        st.write("Aggregated ball tracking trajectories.")

    with main_tabs[4]:
        st.subheader("Spin Analysis")
        st.write("Aggregated spin metrics and 3D spin data.")
        if not df_spins.empty:
            st.write("Spin Stats (Aggregated):")
            st.dataframe(df_spins.describe())

    st.subheader("Key Performance Indicators (Biomechanical Insights)")
    metric_config_shots = {
        "Release Velocity": {"unit": "ft/s", "icon": "🚀"},
        "Release Height": {"unit": "ft", "icon": "📏"},
        "Release Angle": {"unit": "°", "icon": "📐"},
        "Shot Distance": {"unit": "ft", "icon": "📏"},
        "Apex Height": {"unit": "ft", "icon": "🌌"},
        "Release Time": {"unit": "s", "icon": "⏱️"},
        "Release Quality": {"unit": "%", "icon": "💎"},
        "Release Consistency": {"unit": "%", "icon": "📉"},
        "Release Curvature": {"unit": "units", "icon": "🔄"},
        "Lateral Release Curvature": {"unit": "units", "icon": "↔️"},
        "Form Score": {"unit": "/100", "icon": "✅"}
    }
    kpi_cols = st.columns(5)
    for i, key in enumerate(metric_config_shots.keys()):
        col = kpi_cols[i % 5]
        value = averages_shots.get(key, np.nan)
        if not pd.isna(value):
            display_val = f"{value:.2f} {metric_config_shots[key]['unit']}"
        else:
            display_val = "N/A"
        col.metric(
            label=f"{metric_config_shots[key]['icon']} {key}",
            value=display_val
        )

    st.markdown("---")
    st.subheader("Ball Trajectory Visualization")
    if selected_jobs:
        sample_job = selected_jobs[0]
        segs = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], user_email, sample_job['JobID'])
        if segs:
            df_segment = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, sample_job['JobID'], segs[0])
            if not df_segment.empty:
                ball_df = calculate_ball_center(df_segment)
                if not ball_df.empty:
                    fig = px.scatter(
                        ball_df,
                        x='center_x',
                        y='center_y',
                        title="Ball Trajectory"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3D Spin Analysis")
    spin_jobs = [job for job in selected_jobs if job['Source'].lower() == 'spin_video']
    if spin_jobs:
        for job in spin_jobs:
            df_spin = load_spin_axis_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, job['JobID'])
            if not df_spin.empty:
                visualize_spin_metrics(df_spin, job['JobID'])
    else:
        st.info("No spin jobs selected for spin analysis.")

    if not df_shots_metrics.empty:
        st.markdown("---")
        st.subheader("Ball Trajectory Visualization for Shots")
        segs = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], user_email, sample_job['JobID'])
        selected_segment_shot = st.selectbox("Select a segment to visualize ball trajectory", options=segs if segs else [])
        if selected_segment_shot:
            try:
                df_segment_shot = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], user_email, sample_job['JobID'], selected_segment_shot)
                if df_segment_shot.empty:
                    st.write("Selected segment data is empty.")
                else:
                    ball_df_segment_shot = calculate_ball_center(df_segment_shot)
                    if ball_df_segment_shot.empty:
                        st.write("Ball tracking data is missing.")
                    else:
                        fig = px.scatter(
                            ball_df_segment_shot,
                            x='center_x',
                            y='center_y',
                            labels={'center_x': 'X Position (m)', 'center_y': 'Y Position (m)'},
                            title=f"Ball Trajectory for {selected_segment_shot}"
                        )
                        pose_segment_shot_full = df_segment_shot[['left_elbow_angle', 'right_elbow_angle']].dropna()
                        release_idx_shot = detect_release_frame_pose(pose_segment_shot_full)
                        if release_idx_shot is not None and release_idx_shot in ball_df_segment_shot.index:
                            release_point_shot = ball_df_segment_shot.loc[release_idx_shot]
                            fig.add_scatter(
                                x=[release_point_shot['center_x']],
                                y=[release_point_shot['center_y']],
                                mode='markers',
                                marker=dict(color='red', size=12),
                                name='Release Point'
                            )
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(f"Error visualizing trajectory for segment {selected_segment_shot}: {e}")
        else:
            st.write("No segment selected.")

    if not df_spins.empty:
        st.markdown("---")
        st.subheader("Spin Trajectory Visualization")
        spin_ids = df_spins['spin_id'].tolist()
        if spin_ids:
            selected_spin = st.selectbox("Select a spin to visualize", options=spin_ids)
            if selected_spin:
                try:
                    spin_record = df_spins[df_spins['spin_id'] == selected_spin]
                    if spin_record.empty:
                        st.write("Selected spin data is empty.")
                    else:
                        st.write(f"Spin ID: {selected_spin}")
                        st.write(f"Spin Magnitude: {spin_record['spin_magnitude'].values[0]:.2f} units")
                        st.write(f"Spin Consistency: {spin_record['spin_consistency'].values[0]:.2f} units")
                        st.write(f"Spin Duration: {spin_record['spin_duration'].values[0]:.2f} s")
                        if all(col in spin_record.columns for col in ['ex','ey','ez']):
                            fig_spin_3d = px.scatter_3d(
                                spin_record,
                                x='ex',
                                y='ey',
                                z='ez',
                                color='spin_magnitude',
                                size='spin_consistency',
                                hover_name='spin_id',
                                labels={
                                    'ex':'Ex Magnitude',
                                    'ey':'Ey Magnitude',
                                    'ez':'Ez Magnitude',
                                    'spin_magnitude':'Spin Magnitude',
                                    'spin_consistency':'Spin Consistency'
                                },
                                color_continuous_scale=px.colors.sequential.Plasma
                            )
                            st.plotly_chart(fig_spin_3d, use_container_width=True)
                        else:
                            st.write("Detailed spin axis data (ex, ey, ez) is not available for 3D visualization.")
                except Exception as e:
                    st.write(f"Error visualizing trajectory for spin {selected_spin}: {e}")
            else:
                st.write("No spin selected.")
    # -------------------------------------------------------------------------------------
    # SHOT SELECTION TAB
    # -------------------------------------------------------------------------------------
    with main_tabs[1]:
        st.subheader("Inspect Individual Shots (Segments)")
        st.write("Select a job and its segments below to explore each shot individually.")

        # Select a single job
        single_job_selection = st.selectbox(
            "Choose a Job to Inspect Shots",
            options=["None"] + [
                f"{humanize_label(job['Source'])} | "
                f"Player: {humanize_label(job['PlayerName'])} | "
                f"Team: {humanize_label(job['Team'])} | "
                f"Job ID: {job['JobID']}"
                for job in selected_jobs
            ]
        )
        if single_job_selection != "None":
            single_job_id = single_job_selection.split("Job ID: ")[-1]
            job_for_shot_view = next((j for j in selected_jobs if j['JobID'] == single_job_id), None)
            if job_for_shot_view:
                if job_for_shot_view['Source'].lower().startswith('pose'):
                    # Pose Job: List segments
                    segments = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job_for_shot_view['JobID'])
                    if not segments:
                        st.warning(f"No segments found for Pose Job ID: {single_job_id}.")
                        return
                    selected_segment = st.selectbox("Select Segment (Shot) to Inspect", options=["None"] + segments)
                    if selected_segment != "None":
                        # Load data for the selected segment
                        df_segment = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job_for_shot_view['JobID'], selected_segment)
                        if df_segment.empty:
                            st.warning(f"No data found for Job ID: {job_for_shot_view['JobID']}, Segment ID: {selected_segment}.")
                        else:
                            # Ensure unique frames by selecting the first occurrence
                            df_segment = df_segment.sort_values(['frame', 'confidence'], ascending=[True, False])
                            
                            # Separate pose and ball tracking data
                            pose_df, ball_tracking_df = separate_pose_and_ball_tracking(df_segment)
                            
                            # Clean data
                            if not pose_df.empty:
                                pose_df = clean_pose_data(pose_df)
                            if not ball_tracking_df.empty:
                                ball_tracking_df = clean_ball_tracking_data(ball_tracking_df)
                                ball_tracking_df = calculate_ball_center(ball_tracking_df)
                            
                            # Check if we have either pose or ball tracking data
                            if pose_df.empty and ball_tracking_df.empty:
                                st.warning(f"No valid pose or ball tracking data found for Segment ID: {selected_segment} in Job {single_job_id}.")
                                return
                            
                            # Pose Data Handling
                            if not pose_df.empty:
                                release_frame = detect_release_frame_pose(pose_df)
                                trimmed_pose_df = trim_shot(pose_df, release_frame)
                                if trimmed_pose_df is not None and not trimmed_pose_df.empty:
                                    # Add 'relative_frame' column
                                    trimmed_pose_df = add_relative_frame(trimmed_pose_df, release_frame)

                                    for c in trimmed_pose_df.columns:
                                        if c not in ['frame', 'relative_frame']:
                                            trimmed_pose_df[c] = smooth_series(trimmed_pose_df[c])
                                    trimmed_pose_df = compute_angular_vel_accel(trimmed_pose_df)
                                    kpi_pose = calculate_kpis_pose(trimmed_pose_df)
                                    asymmetry_index = kpi_pose.get('Asymmetry Index', np.nan)
                                    kinematic_sequence = kpi_pose.get('Kinematic Chain Sequencing', {})
                                    # Plot Pose Data
                                    st.markdown("#### Single-Shot Pose Plot")
                                    plot_single_shot(trimmed_pose_df, title=f"Single Shot: {selected_segment}", highlight_release=True, job_id=single_job_id, segment_id=selected_segment)
                                    # Display KPIs
                                    st.markdown("#### Pose Analysis Metrics")
                                    col1, col2, col3 = st.columns(3)
                                    metrics = {
                                        'Release Velocity': kpi_pose.get('Release Velocity', np.nan),
                                        'Shot Distance': kpi_pose.get('Shot Distance', np.nan),
                                        'Release Angle': kpi_pose.get('Release Angle', np.nan),
                                        'Apex Height': kpi_pose.get('Apex Height', np.nan),
                                        'Release Time': kpi_pose.get('Release Time', np.nan),
                                        'Release Curvature': kpi_pose.get('Release Curvature', np.nan),
                                        'Asymmetry Index': asymmetry_index,
                                        'Kinematic Chain Sequence Score': kpi_pose.get('Kinematic Chain Sequence Score', np.nan),
                                        'Pivotal Score': kpi_pose.get('Pivotal Score', np.nan)
                                    }
                                    for i, (k, v) in enumerate(metrics.items()):
                                        if i % 3 == 0:
                                            col = col1
                                        elif i % 3 == 1:
                                            col = col2
                                        else:
                                            col = col3
                                        if pd.isna(v):
                                            display_val = "N/A"
                                        else:
                                            display_val = f"{v:.2f}"
                                        col.metric(label=k, value=display_val)
                                    # Display Asymmetry Index and Kinematic Chain Sequencing
                                    st.markdown("##### Additional Metrics")
                                    st.write(f"**Asymmetry Index:** {asymmetry_index:.2f}" if not pd.isna(asymmetry_index) else "Asymmetry Index: N/A")
                                    if kinematic_sequence:
                                        st.write("**Kinematic Chain Sequencing Order:**")
                                        sorted_seq = sorted(kinematic_sequence.items(), key=lambda x: x[1])
                                        sequence_order = " → ".join([humanize_label(k) for k, v in sorted_seq])
                                        st.write(sequence_order)
                                    else:
                                        st.write("Kinematic Chain Sequencing: N/A")
                                else:
                                    st.warning(f"No valid trimmed Pose data for segment {selected_segment}.")
                            
                            # Ball Tracking Data Handling
                            if not ball_tracking_df.empty:
                                # Calculate KPIs for ball tracking data
                                kpi_ball = calculate_kpis_ball_tracking(ball_tracking_df)
                                
                                # Plot Ball Tracking Data
                                st.markdown("#### Single-Shot Ball Tracking Plot")
                                if 'center_x' in ball_tracking_df.columns and 'center_y' in ball_tracking_df.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=ball_tracking_df['center_x'],
                                        y=ball_tracking_df['center_y'],
                                        mode='markers+lines',
                                        name="Ball Trajectory",
                                        marker=dict(color='blue', size=8),
                                        line=dict(color='blue', width=2)
                                    ))
                                    if 'Release Frame' in kpi_ball and not pd.isna(kpi_ball['Release Frame']):
                                        release_frame = kpi_ball['Release Frame']
                                        release_time = release_frame * 0.04  # Assuming fps=25
                                        release_x = ball_tracking_df.loc[ball_tracking_df['frame'] == release_frame, 'center_x'].values
                                        release_y = ball_tracking_df.loc[ball_tracking_df['frame'] == release_frame, 'center_y'].values
                                        if len(release_x) > 0 and len(release_y) > 0:
                                            fig.add_trace(go.Scatter(
                                                x=[release_x[0]],
                                                y=[release_y[0]],
                                                mode='markers',
                                                name="Release Point",
                                                marker=dict(color='red', size=12, symbol='x')
                                            ))
                                    fig.update_layout(
                                        title=f"Ball Tracking Trajectory for Segment: {selected_segment}",
                                        xaxis_title="X Coordinate",
                                        yaxis_title="Y Coordinate",
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Ball tracking data does not contain 'center_x' and 'center_y'.")

                                # Display KPIs
                                st.markdown("#### Ball Tracking Metrics")
                                col1, col2, col3 = st.columns(3)
                                for i, (k, v) in enumerate(kpi_ball.items()):
                                    if i % 3 == 0:
                                        col = col1
                                    elif i % 3 == 1:
                                        col = col2
                                    else:
                                        col = col3
                                    if pd.isna(v):
                                        display_val = "N/A"
                                    else:
                                        display_val = f"{v:.2f}"
                                    col.metric(label=k, value=display_val)

    # -------------------------------------------------------------------------------------
    # POSE ANALYSIS TAB
    # -------------------------------------------------------------------------------------
    with main_tabs[2]:
        st.subheader("Pose Analysis")
        all_segments_trimmed = []
        asymmetry_indices = []
        kinematic_sequences = []
        for job in selected_jobs:
            if not job['Source'].lower().startswith('pose'):
                continue  # Skip non-Pose jobs
            # Handle Pose Jobs
            segments = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job['JobID'])
            if not segments:
                st.warning(f"No segments found for Pose Job ID: {job['JobID']}.")
                continue
            for segment_id in segments:
                df_final = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job['JobID'], segment_id)
                if df_final.empty:
                    st.warning(f"No data found for Job ID: {job['JobID']}, Segment ID: {segment_id}.")
                    continue
                # Ensure unique frames by selecting the first occurrence
                df_final = df_final.sort_values(['frame', 'confidence'], ascending=[True, False])
                
                # Separate pose and ball tracking data
                pose_df, ball_tracking_df = separate_pose_and_ball_tracking(df_final)
                
                # Clean data
                if not pose_df.empty:
                    pose_df = clean_pose_data(pose_df)
                if not ball_tracking_df.empty:
                    ball_tracking_df = clean_ball_tracking_data(ball_tracking_df)
                    ball_tracking_df = calculate_ball_center(ball_tracking_df)
                
                # Proceed if pose data is available
                if pose_df.empty:
                    st.warning(f"No pose data available for Job ID: {job['JobID']}, Segment ID: {segment_id}.")
                    continue
                
                # Identify biomechanical phases
                phases = identify_biomech_phases(pose_df)
                
                # Ensure 'relative_frame' column is added
                release_frame = detect_release_frame_pose(pose_df)
                pose_df = add_relative_frame(pose_df, release_frame)
                
                # Trim and smooth data
                trimmed_pose_df = trim_shot(pose_df, release_frame)
                if trimmed_pose_df is not None and not trimmed_pose_df.empty:
                    for c in trimmed_pose_df.columns:
                        if c not in ['frame', 'relative_frame']:
                            trimmed_pose_df[c] = smooth_series(trimmed_pose_df[c])
                    trimmed_pose_df = compute_angular_vel_accel(trimmed_pose_df)
                    # Calculate KPIs
                    kpi_pose = calculate_kpis_pose(trimmed_pose_df)
                    asymmetry_index = kpi_pose.get('Asymmetry Index', np.nan)
                    kinematic_sequence = kpi_pose.get('Kinematic Chain Sequencing', {})
                    # Collect KPIs
                    all_segments_trimmed.append(trimmed_pose_df)
                    asymmetry_indices.append(asymmetry_index)
                    kinematic_sequences.append(kinematic_sequence)
                    # Visualize phases
                    if not phases.empty:
                        visualize_phases(trimmed_pose_df, phases, job['JobID'], segment_id, fps=25)
                    else:
                        st.warning(f"No phases identified for Job ID: {job['JobID']}, Segment ID: {segment_id}.")
                else:
                    st.warning(f"No valid trimmed Pose data for Job ID: {job['JobID']}, Segment ID: {segment_id}.")

        if not all_segments_trimmed:
            st.warning("No pose data available for the selected jobs.")
            return
        # Summaries
        concatenated_pose = pd.concat(all_segments_trimmed, ignore_index=True)
        concatenated_pose = add_time_column(concatenated_pose, fps=25)
        grouped = concatenated_pose.groupby('relative_frame')
        mean_metrics = grouped.mean(numeric_only=True)
        std_metrics = grouped.std(numeric_only=True)
        
        if mean_metrics.empty or std_metrics.empty:
            st.warning("Insufficient numeric data to calculate overall metrics.")
            return
        sections = {
            'position': {
                'title': 'Position',
                'metrics': [
                    'left_eye_x','left_eye_y','left_eye_z',
                    'right_eye_x','right_eye_y','right_eye_z',
                    'neck_x','neck_y','neck_z',
                    'left_shoulder_x','left_shoulder_y','left_shoulder_z',
                    'right_shoulder_x','right_shoulder_y','right_shoulder_z',
                    'left_elbow_x','left_elbow_y','left_elbow_z',
                    'right_elbow_x','right_elbow_y','right_elbow_z',
                    'left_wrist_x','left_wrist_y','left_wrist_z',
                    'right_wrist_x','right_wrist_y','right_wrist_z',
                ]
            },
            'angle': {
                'title': 'Angle',
                'metrics': [
                    'left_elbow_angle','right_elbow_angle',
                    'left_knee_angle','right_knee_angle',
                    'left_shoulder_angle','right_shoulder_angle',
                    'left_hip_angle','right_hip_angle',
                    'left_ankle_angle','right_ankle_angle'
                ]
            },
            'velocity': {
                'title': 'Angular Velocity',
                'metrics': [
                    'left_elbow_angle_vel','right_elbow_angle_vel',
                    'left_knee_angle_vel','right_knee_angle_vel',
                    'left_shoulder_angle_vel','right_shoulder_angle_vel',
                    'left_hip_angle_vel','right_hip_angle_vel',
                    'left_ankle_angle_vel','right_ankle_angle_vel',
                ]
            },
            'acceleration': {
                'title': 'Angular Acceleration',
                'metrics': [
                    'left_elbow_angle_vel_accel','right_elbow_angle_vel_accel',
                    'left_knee_angle_vel_accel','right_knee_angle_vel_accel',
                    'left_shoulder_angle_vel_accel','right_shoulder_angle_vel_accel',
                    'left_hip_angle_vel_accel','right_hip_angle_vel_accel',
                    'left_ankle_angle_vel_accel','right_ankle_angle_vel_accel',
                ]
            }
        }
        for k, v in sections.items():
            valid_m = [m for m in v['metrics'] if m in mean_metrics.columns]
            sections[k]['metrics'] = valid_m

        for section_key, section_info in sections.items():
            if section_info['metrics']:
                visualize_section(mean_metrics, std_metrics, section_info['metrics'], section_key, section_info['title'], job_id="All_Jobs", segment_id="Aggregate")
            else:
                st.warning(f"No valid {section_info['title']} metrics for the combined data.")
        
        # Display Asymmetry Index and Kinematic Chain Sequencing
        st.markdown("### Aggregated Pose Metrics")
        avg_asymmetry = np.nanmean(asymmetry_indices) if asymmetry_indices else np.nan
        st.write(f"**Average Asymmetry Index:** {avg_asymmetry:.2f}" if not pd.isna(avg_asymmetry) else "Asymmetry Index: N/A")
        # Example: Display average sequencing
        if kinematic_sequences:
            # Flatten all sequences
            all_seqs = []
            for seq in kinematic_sequences:
                sorted_seq = sorted(seq.items(), key=lambda x: x[1])
                seq_order = tuple([k for k, v in sorted_seq])
                all_seqs.append(seq_order)
            # Find the most common sequence
            seq_counts = pd.Series(all_seqs).value_counts()
            most_common_seq, count = seq_counts.idxmax(), seq_counts.max()
            humanized_seq = " → ".join([humanize_label(k) for k in most_common_seq])
            st.write(f"**Most Common Kinematic Chain Sequencing Order:** {humanized_seq} ({count} occurrences)")
        else:
            st.write("Kinematic Chain Sequencing: N/A")

    # -------------------------------------------------------------------------------------
    # BALL TRACKING TAB
    # -------------------------------------------------------------------------------------
    with main_tabs[3]:
        st.subheader("Ball Tracking")
        all_segments_trimmed_ball = []
        kpi_list_ball = []
        for job in selected_jobs:
            if not job['Source'].lower().startswith('pose'):
                continue  # Only Pose jobs have ball tracking
            # Handle Pose Jobs
            segments = list_job_segments(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job['JobID'])
            if not segments:
                st.warning(f"No segments found for Pose Job ID: {job['JobID']}.")
                continue
            for segment_id in segments:
                df_final = load_final_output_csv(s3_client, st.secrets["BUCKET_NAME"], st.session_state['user_email'], job['JobID'], segment_id)
                if df_final.empty:
                    st.warning(f"No data found for Job ID: {job['JobID']}, Segment ID: {segment_id}.")
                    continue
                # Ensure unique frames by selecting the first occurrence
                df_final = df_final.sort_values(['frame', 'confidence'], ascending=[True, False])
                
                # Separate pose and ball tracking data
                pose_df, ball_tracking_df = separate_pose_and_ball_tracking(df_final)
                
                # Clean data
                if not pose_df.empty:
                    pose_df = clean_pose_data(pose_df)
                if not ball_tracking_df.empty:
                    ball_tracking_df = clean_ball_tracking_data(ball_tracking_df)
                    ball_tracking_df = calculate_ball_center(ball_tracking_df)
                
                # Inside the Ball Tracking section of the code
                if not ball_tracking_df.empty:
                    # Smooth ball tracking data
                    for col in ['center_x', 'center_y']:
                        if col in ball_tracking_df.columns:
                            ball_tracking_df[col] = smooth_series(ball_tracking_df[col])
                    
                    # Add 'relative_frame' column
                    release_frame = detect_release_frame_pose(pose_df)  # Use the release frame from pose data
                    ball_tracking_df = add_relative_frame(ball_tracking_df, release_frame)
                    
                    # Calculate KPIs for ball tracking data
                    kpi_ball = calculate_kpis_ball_tracking(ball_tracking_df)
                    if kpi_ball:
                        all_segments_trimmed_ball.append(ball_tracking_df)
                        kpi_list_ball.append(kpi_ball)
        
        if not all_segments_trimmed_ball:
            st.warning("No ball tracking data available for the selected jobs.")
            return
        # Summaries
        concatenated_ball = pd.concat(all_segments_trimmed_ball, ignore_index=True)
        concatenated_ball = add_time_column(concatenated_ball, fps=25)
        grouped_b = concatenated_ball.groupby('relative_frame')
        mean_ball = grouped_b.mean(numeric_only=True)
        std_ball = grouped_b.std(numeric_only=True)

        if mean_ball.empty or std_ball.empty:
            st.warning("Insufficient numeric data to calculate ball tracking metrics.")
            return
        ball_metrics = ['center_x','center_y']
        humanized_ball_metrics = [humanize_label(m) for m in ball_metrics]
        ball_mapping = dict(zip(humanized_ball_metrics, ball_metrics))
        st.markdown("### Ball Position Metrics")
        sel_ball_pos = st.multiselect(
            "Select Ball Position Metrics to Visualize",
            options=humanized_ball_metrics,
            default=humanized_ball_metrics
        )
        sel_ball_pos_actual = [ball_mapping[label] for label in sel_ball_pos]
        if sel_ball_pos_actual:
            fig = go.Figure()
            color_cycle = [
                "#1f77b4","#ff7f0e","#2ca02c","#d62728",
                "#9467bd","#8c564b","#e377c2","#7f7f7f",
                "#bcbd22","#17becf"
            ]
            i_color = 0
            for metric in sel_ball_pos_actual:
                if metric in mean_ball.columns and metric in std_ball.columns:
                    c = color_cycle[i_color % len(color_cycle)]
                    i_color += 1
                    fig.add_trace(go.Scatter(
                        x=mean_ball['time'],
                        y=mean_ball[metric],
                        mode='lines',
                        name=f"Mean - {humanize_label(metric)}",
                        line=dict(width=2, color=c)
                    ))
                    upper = (mean_ball[metric] + std_ball[metric]).tolist()
                    lower = (mean_ball[metric] - std_ball[metric]).tolist()
                    fig.add_trace(go.Scatter(
                        x=list(mean_ball['time']) + list(mean_ball['time'][::-1]),
                        y=upper + lower[::-1],
                        fill='toself',
                        fillcolor=c.replace(")", ", 0.2)").replace("rgb", "rgba") if c.startswith("rgb") else 'rgba(31,119,180,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False,
                        name=f"Std Dev - {humanize_label(metric)}"
                    ))
            fig.update_layout(
                title="Ball Position Metrics Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Units",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display KPIs from Ball Tracking
        st.markdown("### Ball Tracking KPIs")
        if kpi_list_ball:
            # Display average KPIs
            df_kpi_ball = pd.DataFrame(kpi_list_ball)
            mean_kpi_ball = df_kpi_ball.mean(numeric_only=True).to_dict()
            st.markdown("**Average Ball Tracking KPIs:**")
            col1, col2, col3 = st.columns(3)
            metrics = mean_kpi_ball.items()
            for i, (k, v) in enumerate(metrics):
                if i % 3 == 0:
                    col = col1
                elif i % 3 == 1:
                    col = col2
                else:
                    col = col3
                if pd.isna(v):
                    display_val = "N/A"
                else:
                    display_val = f"{v:.2f}"
                col.metric(label=k, value=display_val)
            # Release Consistency
            consistency_percentage = calculate_consistency_percentage(kpi_list_ball, kpi_key='Release Quality')
            st.markdown("**Release Consistency (Based on Release Quality):**")
            if not pd.isna(consistency_percentage):
                color_mode = "normal" if consistency_percentage >= 80 else "inverse" if consistency_percentage >= 60 else "off"
                st.metric(label="Consistency %", value=f"{consistency_percentage:.2f}%", delta_color=color_mode)
            else:
                st.metric(label="Consistency %", value="N/A")
        else:
            st.write("No KPIs available.")

    # -------------------------------------------------------------------------------------
    # SPIN ANALYSIS TAB
    # -------------------------------------------------------------------------------------
    with main_tabs[4]:
        st.subheader("🔮 Spin Analysis")
        
        spin_jobs = [j for j in selected_jobs if j['Source'].lower() == 'spin_video']
        
        if not spin_jobs:
            st.info("Select spin capture jobs in the main view to analyze ball rotation")
            return

        all_spin_data = []
        kpi_list = []
        
        for job in spin_jobs:
            with st.spinner(f"Loading spin data for {job['JobID']}..."):
                df_spin = load_spin_axis_csv(s3_client, st.secrets["BUCKET_NAME"], 
                                        st.session_state['user_email'], job['JobID'])
                
                if not df_spin.empty:
                    # Add temporal features
                    df_spin['time'] = df_spin['frame'] / 25  # Assuming 25 FPS
                    df_spin = add_relative_frame(df_spin, release_frame=df_spin['frame'].min())
                    
                    # Store data
                    all_spin_data.append(df_spin)
                    kpi_list.append(compute_spin_kpi(df_spin))
                    
                    # Show job-specific visualization
                    st.markdown(f"##### Analysis for {job['JobID']}")
                    visualize_spin_metrics(df_spin, job['JobID'])

        if not all_spin_data:
            st.warning("No valid spin data found in selected jobs")
            return

        # Aggregate analysis
        st.markdown("---")
        st.subheader("Combined Spin Analysis")
        
        # Show KPI cards
        cols = st.columns(4)
        metrics = {
            'avg_spin_rate': "Avg Spin Rate",
            'max_spin_rate': "Max Spin Rate",
            'spin_consistency': "Spin Consistency",
            'backspin_ratio': "Backspin %"
        }
        
        for i, (k, label) in enumerate(metrics.items()):
            values = [kpi.get(k, 0) for kpi in kpi_list]
            cols[i % 4].metric(
                label=label,
                value=f"{np.nanmean(values):.2f} RPM" if k == 'spin_rate' else f"{np.nanmean(values):.2f}",
                delta=f"±{np.nanstd(values):.2f} STD"
            )

        # Comparative visualization
        combined_df = pd.concat(all_spin_data)
        fig = px.box(combined_df, y='spin_magnitude', 
                    title="Spin Rate Distribution Across All Shots")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# Execute the Main Function
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
