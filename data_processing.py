# data_processing.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import plotly.graph_objects as go
import pandas as pd
from io import StringIO, BytesIO
import logging
from scipy.signal import savgol_filter
from boto3.dynamodb.conditions import Attr
from config import BUCKET_NAME, DYNAMODB_TABLE_USERS, DYNAMODB_TABLE_DATA
from aws_client import initialize_aws_clients, list_job_segments
from math import degrees, atan2
import re

from kpi import (
    angle_2d  # Add this if angle_2d is defined in kpi.py
)


logger = logging.getLogger(__name__)

# Initialize AWS clients
cognito_client, dynamodb, s3_client = initialize_aws_clients()


def load_spin_axis_csv(s3_client, bucket_name, user_email, job_id):
    spin_key = f"processed/{user_email}/{job_id}/spin_axis.csv"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=spin_key)
        csv_data = response['Body'].read().decode('utf-8')
        df_spin = pd.read_csv(StringIO(csv_data))
        logger.info(f"spin_axis.csv loaded for Job ID: {job_id}")
        return df_spin
    except s3_client.exceptions.NoSuchKey:
        st.error(f"spin_axis.csv not found for Job ID: {job_id}")
        logger.warning(f"spin_axis.csv not found for Job ID: {job_id}")
    except Exception as e:
        st.error(f"Error loading spin_axis.csv for Job ID: {job_id}: {e}")
        logger.error(f"Error in load_spin_axis_csv for Job ID: {job_id}: {e}")
    return pd.DataFrame()

def load_final_output(s3_client, bucket_name, user_email, job_id, segment_id, file_format='csv'):
    # Normalize file format to lower-case.
    file_format = file_format.lower()
    
    # Construct the S3 key based on the file format.
    if file_format == 'csv':
        final_output_key = f"processed/{user_email}/{job_id}/{segment_id}_final_output.csv"
    elif file_format in ['xlsx', 'xls']:
        final_output_key = f"processed/{user_email}/{job_id}/{segment_id}_final_output.xlsx"
    else:
        st.error(f"Unsupported file format: {file_format}. Please use 'csv' or 'xlsx'.")
        logger.error(f"Unsupported file format provided: {file_format}")
        return pd.DataFrame()

    try:
        # Retrieve the object from S3.
        response = s3_client.get_object(Bucket=bucket_name, Key=final_output_key)
        
        if file_format == 'csv':
            # For CSV files, decode to text and read using pd.read_csv.
            csv_data = response['Body'].read().decode('utf-8')
            df_final = pd.read_csv(StringIO(csv_data))
        else:
            # For Excel files, read the binary data and use pd.read_excel.
            excel_data = response['Body'].read()
            df_final = pd.read_excel(BytesIO(excel_data))
            
        logger.info(f"final_output.{file_format} loaded for Job ID: {job_id}, Segment ID: {segment_id}")
        return df_final
        
    except s3_client.exceptions.NoSuchKey:
        st.error(f"final_output.{file_format} not found for Job ID: {job_id}, Segment ID: {segment_id}.")
        logger.warning(f"final_output.{file_format} not found for Job ID: {job_id}, Segment ID: {segment_id}")
    except Exception as e:
        st.error(f"Error loading final_output.{file_format} for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
        logger.error(f"Error in load_final_output for Job ID: {job_id}, Segment ID: {segment_id}: {e}")
    
    return pd.DataFrame()



def list_job_segments(s3_client, bucket_name, user_email, job_id):
    """
    Lists segments for pose_video or spin_video jobs by searching for files
    whose names contain "segment_" and end with "_final_output.csv" under:
       processed/{user_email}/{job_id}/
    """
    prefix = f"processed/{user_email}/{job_id}/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page_number, page in enumerate(pages, start=1):
            logger.info(f"Listing segments, page {page_number}.")
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    logger.debug(f"Found key: {key}")
                    filename = os.path.basename(key)
                    # Adjust the condition if your files are named differently.
                    if "segment_" in filename and filename.endswith("_final_output.csv"):
                        seg_id = filename.replace("_final_output.csv", "")
                        segments.add(seg_id)
            else:
                logger.warning(f"No Contents found on page {page_number}.")
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing segments for job '{job_id}': {e}")
        logger.error(f"Error listing segments for job '{job_id}': {e}")
        return []


def list_data_file_job_segments(bucket_name, user_email, job_id):
    """
    Lists segments for data_file jobs by scanning the S3 folder:
      processed/{user_email}/{job_id}/data_file/
    and looking for files that match the naming convention:
      segment_XXX_final_output.csv or segment_XXX_final_output.xlsx.
    """
    prefix = f"processed/{user_email}/{job_id}/data_file/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    # Check for files starting with "segment_" and ending with CSV or XLSX
                    if filename.lower().startswith("segment_") and (
                        filename.lower().endswith("_final_output.csv") or 
                        filename.lower().endswith("_final_output.xlsx")
                    ):
                        if filename.lower().endswith("_final_output.csv"):
                            seg_id = filename.replace("_final_output.csv", "")
                        elif filename.lower().endswith("_final_output.xlsx"):
                            seg_id = filename.replace("_final_output.xlsx", "")
                        segments.add(seg_id)
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing data file segments for job '{job_id}': {e}")
        logger.error(f"Error listing data file segments for job '{job_id}': {e}")
        return []
    
def get_value_case_insensitive(row, key):
    """Return the value for the column matching key (case-insensitive)."""
    for col in row.index:
        if col.lower() == key.lower():
            return row[col]
    return None

def get_shot_type(distance):
    """
    Determine shot type based on the shot distance (in feet).
    For example:
      - Less than 15 ft: Free Throw
      - 15 to 22 ft: Mid Range
      - Over 22 ft: 3 Point
    """
    try:
        d = float(distance)
        logger.debug(f"Determining shot type for distance: {d} ft")
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid distance value: {distance}. Returning 'Unknown'. Error: {e}")
        return "Unknown"
    if d < 15:
        return "Free Throw"
    elif d < 22:
        return "Mid Range"
    else:
        return "3 Point"

def separate_pose_and_ball_tracking(df_segment, source):
    """Split data into pose and ball tracking, supporting multiple ball column name variations."""
    # Define possible ball column sets with case-insensitive and common variations
    possible_ball_columns = [
        ['Basketball_X', 'Basketball_Y', 'Basketball_Z'],
        ['Ball_X', 'Ball_Y', 'Ball_Z'],  # Matches your data
        ['ball_x', 'ball_y', 'ball_z'],
        ['BALL_X', 'BALL_Y', 'BALL_Z'],
    ]
    
    # Find the first matching set of ball columns
    active_ball_columns = None
    for ball_cols in possible_ball_columns:
        if all(col in df_segment.columns for col in ball_cols):
            active_ball_columns = ball_cols
            break
    
    if not active_ball_columns:
        logger.warning(f"No complete set of ball tracking columns found in DataFrame. Available columns: {list(df_segment.columns)}")
        pose_df = df_segment.copy()
        ball_df = pd.DataFrame()
        return pose_df, ball_df
    
    # Include 'OUTCOME' and 'IS_MADE' in ball_df if they exist
    ball_columns = active_ball_columns.copy()
    for col in ['OUTCOME', 'IS_MADE']:
        if col in df_segment.columns:
            ball_columns.append(col)
    
    # All non-ball columns (except 'OUTCOME' and 'IS_MADE') are considered pose data
    pose_columns = [col for col in df_segment.columns if col not in ball_columns]
    
    # Separate into pose and ball DataFrames
    pose_df = df_segment[pose_columns].copy()
    ball_df = df_segment[ball_columns].copy()
    
    # Rename ball columns to standard 'Basketball_X/Y/Z' if needed
    if active_ball_columns != ['Basketball_X', 'Basketball_Y', 'Basketball_Z']:
        rename_dict = {old: new for old, new in zip(active_ball_columns, ['Basketball_X', 'Basketball_Y', 'Basketball_Z'])}
        if active_ball_columns == ['Ball_X', 'Ball_Y', 'Ball_Z']:  # Matches your data
            logger.info(f"Renaming ball columns from {active_ball_columns} to {['Basketball_X', 'Basketball_Y', 'Basketball_Z']}")
        ball_df = ball_df.rename(columns=rename_dict)
    
    logger.debug(f"After separation - pose_df columns: {pose_df.columns.tolist()}")
    logger.debug(f"After separation - ball_df columns: {ball_df.columns.tolist()}")
    
    return pose_df, ball_df

def get_segment_label(s3_client, bucket_name, user_email, job_id, segment_id, source):
    """
    Loads a small portion of the final output for the given segment and extracts
    metadata such as MONTH, DAY, PERIOD, STARTGAMECLOCK, and shot distance.
    Uses the shot_distance from calculate_shot_metrics for accurate shot type classification.
    """
    try:
        # Load the final output DataFrame
        if source.lower() == "data_file":
            df = load_data_file_final_output(s3_client, bucket_name, user_email, job_id, segment_id, file_format='csv')
        else:
            df = load_final_output(s3_client, bucket_name, user_email, job_id, segment_id, file_format='csv')
        if df.empty:
            logger.warning(f"Empty DataFrame for segment {segment_id}. Returning segment_id.")
            return segment_id
    except Exception as e:
        logger.error(f"Error loading final output for segment {segment_id}: {e}")
        return segment_id

    # Separate pose and ball tracking data
    pose_df, ball_df = separate_pose_and_ball_tracking(df, source)

    # Calculate shot metrics to get the trajectory-based shot_distance
    metrics, _, _ = calculate_shot_metrics(pose_df, ball_df, fps=60)

    # Extract shot_distance from metrics (in feet, as calculated in calculate_shot_metrics)
    shot_distance = metrics.get('shot_distance', None)
    logger.debug(f"Shot distance from calculate_shot_metrics: {shot_distance} ft")

    # Extract metadata from the DataFrame
    row = df.iloc[0]

    def get_val(key):
        for col in row.index:
            if col.lower() == key.lower():
                return row[col]
        return None

    month = str(get_val("MONTH") or "")
    day = str(get_val("DAY") or "")
    period = str(get_val("PERIOD") or "")
    
    # Format the game clock from raw seconds (e.g. 215.0 becomes 03:36)
    raw_game_clock = get_val("STARTGAMECLOCK")
    def format_game_clock(seconds):
        try:
            seconds = float(seconds) if seconds is not None and not pd.isna(seconds) else 0.0
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except Exception:
            return "00:00"
    game_clock = format_game_clock(raw_game_clock)
    
    # Determine shot type using the trajectory-based shot_distance
    shot_type = get_shot_type(shot_distance) if shot_distance is not None else "Unknown"
    logger.debug(f"Determined shot type: {shot_type} for distance {shot_distance} ft")

    return f"{month}/{day} | Period: {period} | Clock: {game_clock} | {shot_type}"



def get_player_kpi_averages(player_name, shot_type):
    """
    Aggregate all previous jobs for the given player (by name) and shot type
    and compute the average value for each KPI in feet.
    """
    from data_processing import fetch_user_completed_jobs, fetch_user_completed_data_file_jobs, get_metadata_from_dynamodb
    import numpy as np

    INCHES_TO_FEET = 1 / 12

    # Get all completed jobs for this user and filter by player name and shot type
    all_jobs = fetch_user_completed_jobs(player_name) + fetch_user_completed_data_file_jobs(player_name)
    filtered_jobs = [job for job in all_jobs if job.get("PlayerName", "").lower() == player_name.lower() and job.get("ShootingType", "").lower() == shot_type.lower()]
    
    if not filtered_jobs:
        logger.warning(f"No jobs found for player '{player_name}' and shot type '{shot_type}'. Returning None for averages.")
        return None  # Explicitly return None if no data exists

    kpi_names = ["Release Height", "Release Angle", "Release Velocity", "Release Time", "Apex Height", "Release Curvature", "Lateral Deviation"]
    averages = {}
    for kpi in kpi_names:
        values = []
        for job in filtered_jobs:
            meta = get_metadata_from_dynamodb(job["JobID"])
            if meta and kpi in meta:
                try:
                    value = float(meta[kpi])
                    # Convert inches to feet if the value is in inches (assuming metadata might be in inches)
                    if kpi in ["Release Height", "Apex Height", "Lateral Deviation", "Shot Distance"]:
                        value *= INCHES_TO_FEET
                    # Release Velocity might need conversion if stored in inches/sec
                    elif kpi == "Release Velocity":
                        value *= INCHES_TO_FEET
                    # Curvature (1/inches to 1/feet)
                    elif kpi in ["Release Curvature"]:
                        value *= 12  # Convert 1/inches to 1/feet
                    values.append(value)
                except (TypeError, ValueError):
                    continue
        averages[kpi] = np.mean(values) if values else None
    return averages


def calculate_release_angle(df_ball, release_idx):
    """
    Calculate the release angle of the ball in degrees using Z for vertical motion.
    """
    if df_ball.empty or pd.isna(release_idx):
        return np.nan
    # Use the next few frames to calculate the angle
    df_after = df_ball[df_ball['frame'] > release_idx].head(5)
    if df_after.empty:
        return np.nan
    dz = df_after['center_z_m'].mean() - df_ball.loc[df_ball['frame'] == release_idx, 'center_z_m'].values[0]
    dx = df_after['center_x_m'].mean() - df_ball.loc[df_ball['frame'] == release_idx, 'center_x_m'].values[0]
    angle = degrees(atan2(dz, dx))
    return angle

def calculate_optimal_release_angle(distance, release_height, entry_angle=45):
    """Calculate optimal release angle using projectile motion equations"""
    try:
        rim_height = 10  # Standard hoop height
        g = 32.2  # ft/s²
        
        # Calculate using modified projectile equation
        term = (rim_height - release_height + distance * np.tan(np.radians(entry_angle))) / distance
        return np.degrees(np.arctan(term))
    except:
        return None

def classify_release_angle(actual_angle, distance, release_height):
    """Classify release angle into performance categories (distance and height in feet)"""
    entry_angles = np.linspace(43, 47, 5)  # Optimal entry angle range
    optimal_angles = [calculate_optimal_release_angle(distance, release_height, a) for a in entry_angles]
    opt_min = min(optimal_angles)
    opt_max = max(optimal_angles)
    
    ranges = {
        'optimal': (opt_min, opt_max, 'green'),
        'slightly_high': (opt_max, opt_max+3, 'yellow'),
        'slightly_low': (opt_min-3, opt_min, 'orange'),
        'poor_high': (opt_max+3, opt_max+8, 'red'),
        'poor_low': (opt_min-8, opt_min-3, 'red')
    }
    
    for label, (low, high, color) in ranges.items():
        if low <= actual_angle <= high:
            return {
                'classification': label.replace('_', ' ').title(),
                'color': color,
                'optimal_range': f"{opt_min:.1f}° - {opt_max:.1f}°"
            }
    return {'classification': 'Out of Range', 'color': 'darkred', 'optimal_range': f"{opt_min:.1f}° - {opt_max:.1f}°"}

def load_data_file_final_output(s3_client, bucket_name, user_email, job_id, segment_id, file_format='auto'):
    """
    Load final output for data_file jobs from a specific subfolder, supporting CSV and XLSX formats.
    
    If file_format is 'auto' (the default), the function first attempts to load a CSV file.
    If the CSV file is not found, it then attempts to load an XLSX file.
    
    Parameters:
        s3_client: The boto3 S3 client.
        bucket_name: Name of the S3 bucket.
        user_email: User email used in the file path.
        job_id: Job ID used in the file path.
        segment_id: Segment ID used in the file path.
        file_format: Format of the file ('csv', 'xlsx', or 'auto'). Defaults to 'auto'.
        
    Returns:
        A pandas DataFrame containing the file's data or an empty DataFrame if an error occurs.
    """
    file_format = file_format.lower()
    
    if file_format not in ['csv', 'xlsx', 'xls', 'auto']:
        st.error(f"Unsupported file format: {file_format}. Please use 'csv', 'xlsx', or 'auto'.")
        logger.error(f"Unsupported file format provided: {file_format}")
        return pd.DataFrame()
    
    # Define the S3 keys for both CSV and XLSX.
    csv_key = f"processed/{user_email}/{job_id}/data_file/{segment_id}_final_output.csv"
    xlsx_key = f"processed/{user_email}/{job_id}/data_file/{segment_id}_final_output.xlsx"
    
    # If file_format is set explicitly to CSV or XLSX, try that only.
    if file_format == 'csv':
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
            csv_data = response['Body'].read().decode('utf-8')
            df_final = pd.read_csv(StringIO(csv_data))
            logger.info(f"data_file CSV loaded for Job ID: {job_id}, Segment ID: {segment_id}")
            return df_final
        except s3_client.exceptions.NoSuchKey:
            st.error(f"data_file CSV not found for Job ID: {job_id}, Segment ID: {segment_id}")
            logger.warning(f"data_file CSV not found for Job ID: {job_id}, Segment ID: {segment_id}")
        except Exception as e:
            st.error(f"Error loading data_file CSV: {e}")
            logger.error(f"Error in load_data_file_final_output (CSV): {e}")
        return pd.DataFrame()
    
    elif file_format in ['xlsx', 'xls']:
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=xlsx_key)
            excel_data = response['Body'].read()
            df_final = pd.read_excel(BytesIO(excel_data))
            logger.info(f"data_file XLSX loaded for Job ID: {job_id}, Segment ID: {segment_id}")
            return df_final
        except s3_client.exceptions.NoSuchKey:
            st.error(f"data_file XLSX not found for Job ID: {job_id}, Segment ID: {segment_id}")
            logger.warning(f"data_file XLSX not found for Job ID: {job_id}, Segment ID: {segment_id}")
        except Exception as e:
            st.error(f"Error loading data_file XLSX: {e}")
            logger.error(f"Error in load_data_file_final_output (XLSX): {e}")
        return pd.DataFrame()
    
    # For 'auto', try CSV first, then XLSX if CSV is not found.
    else:  # file_format == 'auto'
        try:
            # Try CSV first.
            response = s3_client.get_object(Bucket=bucket_name, Key=csv_key)
            csv_data = response['Body'].read().decode('utf-8')
            df_final = pd.read_csv(StringIO(csv_data))
            logger.info(f"data_file CSV loaded for Job ID: {job_id}, Segment ID: {segment_id}")
            return df_final
        except s3_client.exceptions.NoSuchKey:
            logger.info(f"CSV not found for Job ID: {job_id}, Segment ID: {segment_id}; trying XLSX.")
            try:
                # Try XLSX next.
                response = s3_client.get_object(Bucket=bucket_name, Key=xlsx_key)
                excel_data = response['Body'].read()
                df_final = pd.read_excel(BytesIO(excel_data))
                logger.info(f"data_file XLSX loaded for Job ID: {job_id}, Segment ID: {segment_id}")
                return df_final
            except s3_client.exceptions.NoSuchKey:
                st.error(f"Neither CSV nor XLSX found for Job ID: {job_id}, Segment ID: {segment_id}")
                logger.warning(f"Neither CSV nor XLSX found for Job ID: {job_id}, Segment ID: {segment_id}")
            except Exception as e:
                st.error(f"Error loading data_file XLSX: {e}")
                logger.error(f"Error in load_data_file_final_output (XLSX in auto): {e}")
        except Exception as e:
            st.error(f"Error loading data_file CSV: {e}")
            logger.error(f"Error in load_data_file_final_output (CSV in auto): {e}")
        return pd.DataFrame()
    
def get_username_by_email(email):
    """
    Query DynamoDB to fetch the username for the given email.
    """
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

def get_metadata_from_dynamodb(job_id):
    """
    Retrieve job metadata from DynamoDB for the given JobID.
    """
    table = dynamodb.Table(DYNAMODB_TABLE_DATA)
    try:
        response = table.get_item(Key={'JobID': job_id})
        if 'Item' in response:
            logger.info(f"Metadata found for Job ID: {job_id}")
            return response['Item']
        else:
            logger.warning(f"No metadata found in DynamoDB for Job ID: {job_id}")
            return {}
    except Exception as e:
        st.error(f"Error retrieving metadata for Job ID {job_id}: {e}")
        logger.error(f"Error in get_metadata_from_dynamodb: {e}")
        return {}

def fetch_user_completed_jobs(user_email):
    """
    Scan the S3 bucket for job folders under "processed/{user_email}/" and then check
    in DynamoDB if the job’s status is "completed" (for non-data_file sources).
    """
    jobs = []
    prefix = f"processed/{user_email}/"
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
        for page in pages:
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    job_id = cp['Prefix'].replace(prefix, '').strip('/')
                    if not job_id:
                        continue
                    # Skip if there is a data_file folder (handled separately)
                    response = s3_client.list_objects_v2(
                        Bucket=BUCKET_NAME,
                        Prefix=f"{cp['Prefix']}data_file/",
                        MaxKeys=1
                    )
                    if 'Contents' in response:
                        continue
                    job_meta = get_metadata_from_dynamodb(job_id)
                    if not job_meta:
                        continue
                    if job_meta.get('Status', '').lower() != 'completed':
                        continue
                    jobs.append({
                        'JobID': job_id,
                        'PlayerName': job_meta.get('PlayerName', 'Unknown Player'),
                        'Team': job_meta.get('Team', 'N/A'),
                        'ShootingType': job_meta.get('ShootingType', 'N/A'),  # Ensure this is included
                        'UploadTimestamp': job_meta.get('UploadTimestamp', None),
                        'Source': job_meta.get('Source', 'Unknown')  # e.g., 'pose_video' or 'spin_video'
                    })
        logger.info(f"Fetched {len(jobs)} completed non-data_file jobs for {user_email}.")
        return jobs
    except Exception as e:
        st.error(f"Error fetching user jobs: {e}")
        logger.error(f"Error in fetch_user_completed_jobs: {e}")
        return []

def fetch_user_completed_data_file_jobs(user_email):
    """
    Scan the S3 bucket for jobs that have a 'data_file' subfolder and check in DynamoDB
    that their source is 'data_file' and status is valid.
    """
    jobs = []
    prefix = f"processed/{user_email}/"
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
        for page in pages:
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    job_id = cp['Prefix'].replace(prefix, '').strip('/')
                    if not job_id:
                        continue
                    data_file_prefix = f"{cp['Prefix']}data_file/"
                    response = s3_client.list_objects_v2(
                        Bucket=BUCKET_NAME,
                        Prefix=data_file_prefix,
                        MaxKeys=1
                    )
                    if 'Contents' not in response:
                        continue
                    job_meta = get_metadata_from_dynamodb(job_id)
                    if not job_meta:
                        continue
                    if job_meta.get('Source', '').lower() != 'data_file':
                        continue
                    if job_meta.get('Status', '').lower() not in ['processed', 'completed']:
                        continue
                    jobs.append({
                        'JobID': job_id,
                        'PlayerName': job_meta.get('PlayerName', 'Unknown Player'),
                        'Team': job_meta.get('Team', 'N/A'),
                        'ShootingType': job_meta.get('ShootingType', 'N/A'),  # Ensure this is included
                        'UploadTimestamp': job_meta.get('UploadTimestamp', None),
                        'Source': job_meta.get('Source', 'Unknown')
                    })
        logger.info(f"Fetched {len(jobs)} completed data_file jobs for {user_email}.")
        return jobs
    except Exception as e:
        st.error(f"Error fetching data_file jobs: {e}")
        logger.error(f"Error in fetch_user_completed_data_file_jobs: {e}")
        return []
# In data_processing.py, update the column names used in calculations:

def calculate_foot_angles(df):
    """Calculate foot angles using both old and new column names."""
    try:
        # Left foot (new names: LHEEL_X, LSMALLTOE_X, LBIGTOE_X)
        if all(col in df.columns for col in ['LHEEL_X', 'LHEEL_Y', 'LSMALLTOE_X', 'LSMALLTOE_Y', 'LBIGTOE_X', 'LBIGTOE_Y']):
            df['left_foot_angle'] = df.apply(lambda row: angle_2d(
                (row['LHEEL_X'], row['LHEEL_Y']),
                (row['LSMALLTOE_X'], row['LSMALLTOE_Y']),
                (row['LBIGTOE_X'], row['LBIGTOE_Y'])
            ), axis=1)
        # Fallback to old names (LSTOE_X instead of LSMALLTOE_X)
        elif all(col in df.columns for col in ['LHEEL_X', 'LHEEL_Y', 'LSTOE_X', 'LSTOE_Y', 'LBTOE_X', 'LBTOE_Y']):
            df['left_foot_angle'] = df.apply(lambda row: angle_2d(
                (row['LHEEL_X'], row['LHEEL_Y']),
                (row['LSTOE_X'], row['LSTOE_Y']),
                (row['LBTOE_X'], row['LBTOE_Y'])
            ), axis=1)
        
        # Right foot (new names: RHEEL_X, RSMALLTOE_X, RBIGTOE_X)
        if all(col in df.columns for col in ['RHEEL_X', 'RHEEL_Y', 'RSMALLTOE_X', 'RSMALLTOE_Y', 'RBIGTOE_X', 'RBIGTOE_Y']):
            df['right_foot_angle'] = df.apply(lambda row: angle_2d(
                (row['RHEEL_X'], row['RHEEL_Y']),
                (row['RSMALLTOE_X'], row['RSMALLTOE_Y']),
                (row['RBIGTOE_X'], row['RBIGTOE_Y'])
            ), axis=1)
        # Fallback to old names (RSTOE_X instead of RSMALLTOE_X)
        elif all(col in df.columns for col in ['RHEEL_X', 'RHEEL_Y', 'RSTOE_X', 'RSTOE_Y', 'RBTOE_X', 'RBTOE_Y']):
            df['right_foot_angle'] = df.apply(lambda row: angle_2d(
                (row['RHEEL_X'], row['RHEEL_Y']),
                (row['RSTOE_X'], row['RSTOE_Y']),
                (row['RBTOE_X'], row['RBTOE_Y'])
            ), axis=1)
    except Exception as e:
        logger.error(f"Error calculating foot angles: {str(e)}")
    return df

def calculate_rotation_angles(df):
    """Calculate rotations using both old and new column names."""
    try:
        # Feet Rotation (new names: LBIGTOE_X, RBIGTOE_X)
        if all(col in df.columns for col in ['LBIGTOE_X', 'LBIGTOE_Y', 'RBIGTOE_X', 'RBIGTOE_Y', 'MIDHIP_X', 'MIDHIP_Y']):
            df['feet_rotation'] = df.apply(lambda row: angle_2d(
                (row['LBIGTOE_X'], row['LBIGTOE_Y']),
                (row['MIDHIP_X'], row['MIDHIP_Y']),
                (row['RBIGTOE_X'], row['RBIGTOE_Y'])
            ), axis=1)
        # Fallback to old names (LBTOE_X, RBTOE_X)
        elif all(col in df.columns for col in ['LBTOE_X', 'LBTOE_Y', 'RBTOE_X', 'RBTOE_Y', 'MIDHIP_X', 'MIDHIP_Y']):
            df['feet_rotation'] = df.apply(lambda row: angle_2d(
                (row['LBTOE_X'], row['LBTOE_Y']),
                (row['MIDHIP_X'], row['MIDHIP_Y']),
                (row['RBTOE_X'], row['RBTOE_Y'])
            ), axis=1)
            
        # Hips Rotation (new names: LHIP_X, RHIP_X)
        if all(col in df.columns for col in ['LHIP_X', 'LHIP_Y', 'RHIP_X', 'RHIP_Y', 'MIDHIP_X', 'MIDHIP_Y']):
            df['hips_rotation'] = df.apply(lambda row: angle_2d(
                (row['LHIP_X'], row['LHIP_Y']),
                (row['MIDHIP_X'], row['MIDHIP_Y']),
                (row['RHIP_X'], row['RHIP_Y'])
            ), axis=1)
        # Fallback to old names (LHJC_X, RHJC_X)
        elif all(col in df.columns for col in ['LHJC_X', 'LHJC_Y', 'RHJC_X', 'RHJC_Y', 'MIDHIP_X', 'MIDHIP_Y']):
            df['hips_rotation'] = df.apply(lambda row: angle_2d(
                (row['LHJC_X'], row['LHJC_Y']),
                (row['MIDHIP_X'], row['MIDHIP_Y']),
                (row['RHJC_X'], row['RHJC_Y'])
            ), axis=1)
            
        # Shoulders Rotation (new names: LSHOULDER_X, RSHOULDER_X)
        if all(col in df.columns for col in ['LSHOULDER_X', 'LSHOULDER_Y', 'RSHOULDER_X', 'RSHOULDER_Y', 'NECK_X', 'NECK_Y']):
            df['shoulders_rotation'] = df.apply(lambda row: angle_2d(
                (row['LSHOULDER_X'], row['LSHOULDER_Y']),
                (row['NECK_X'], row['NECK_Y']),
                (row['RSHOULDER_X'], row['RSHOULDER_Y'])
            ), axis=1)
        # Fallback to old names (LSJC_X, RSJC_X)
        elif all(col in df.columns for col in ['LSJC_X', 'LSJC_Y', 'RSJC_X', 'RSJC_Y', 'NECK_X', 'NECK_Y']):
            df['shoulders_rotation'] = df.apply(lambda row: angle_2d(
                (row['LSJC_X'], row['LSJC_Y']),
                (row['NECK_X'], row['NECK_Y']),
                (row['RSJC_X'], row['RSJC_Y'])
            ), axis=1)
    except Exception as e:
        logger.error(f"Error calculating rotation angles: {str(e)}")
    return df

def calculate_stability(df):
    """Calculate stability ratio using both old and new column names."""
    try:
        # New names: LBIGTOE_X, RBIGTOE_X, LHIP_X, RHIP_X
        if all(col in df.columns for col in ['LBIGTOE_X', 'RBIGTOE_X', 'LHIP_X', 'RHIP_X']):
            df['feet_width'] = df['RBIGTOE_X'] - df['LBIGTOE_X']
            df['hips_width'] = df['RHIP_X'] - df['LHIP_X']
            df['stability_ratio'] = df['feet_width'] / df['hips_width']
        # Fallback to old names: LBTOE_X, RBTOE_X, LHJC_X, RHJC_X
        elif all(col in df.columns for col in ['LBTOE_X', 'RBTOE_X', 'LHJC_X', 'RHJC_X']):
            df['feet_width'] = df['RBTOE_X'] - df['LBTOE_X']
            df['hips_width'] = df['RHJC_X'] - df['LHJC_X']
            df['stability_ratio'] = df['feet_width'] / df['hips_width']
    except Exception as e:
        logger.error(f"Error calculating stability: {str(e)}")
    return df

def calculate_guide_hand_release(df):
    """Determine guide hand release using both old and new column names."""
    try:
        # New names: Basketball_X, LTHUMB_X
        if all(col in df.columns for col in ['Basketball_X', 'Basketball_Y', 'Basketball_Z', 'LTHUMB_X', 'LTHUMB_Y', 'LTHUMB_Z']):
            df['guide_hand_distance'] = np.sqrt(
                (df['Basketball_X'] - df['LTHUMB_X'])**2 +
                (df['Basketball_Y'] - df['LTHUMB_Y'])**2 +
                (df['Basketball_Z'] - df['LTHUMB_Z'])**2
            )
            df['guide_hand_smoothed'] = df['guide_hand_distance'].rolling(5, min_periods=1).mean()
            df['guide_hand_release'] = df['guide_hand_smoothed'] > 0.395
            df['guide_hand_release'] = df['guide_hand_release'].astype(int).diff().ge(1)
    except Exception as e:
        logger.error(f"Error calculating guide hand release: {str(e)}")
    return df

def calculate_centroid(df):
    """Calculate centroid using both old and new column names."""
    try:
        # New names: MIDHIP_X, NECK_X
        if all(col in df.columns for col in ['MIDHIP_X', 'MIDHIP_Y', 'MIDHIP_Z', 'NECK_X', 'NECK_Y', 'NECK_Z']):
            df['centroid_x'] = (df['MIDHIP_X'] + df['NECK_X']) / 2
            df['centroid_y'] = (df['MIDHIP_Y'] + df['NECK_Y']) / 2
            df['centroid_z'] = (df['MIDHIP_Z'] + df['NECK_Z']) / 2
    except Exception as e:
        logger.error(f"Error calculating centroid: {str(e)}")
    return df

def compute_joint_angles(df):
    """Compute joint angles using new column names with fallback to old names."""
    joints = {
        'left_elbow_angle': ('LSHOULDER', 'LELBOW', 'LWRIST'),  # New names
        'right_elbow_angle': ('RSHOULDER', 'RELBOW', 'RWRIST'),
        'left_knee_angle': ('LHIP', 'LKNEE', 'LANKLE'),
        'right_knee_angle': ('RHIP', 'RKNEE', 'RANKLE'),
        'left_shoulder_angle': ('NECK', 'LSHOULDER', 'LELBOW'),
        'right_shoulder_angle': ('NECK', 'RSHOULDER', 'RELBOW'),
        'left_hip_angle': ('MIDHIP', 'LHIP', 'LKNEE'),
        'right_hip_angle': ('MIDHIP', 'RHIP', 'RKNEE'),
        'left_ankle_angle': ('LKNEE', 'LANKLE', 'LBIGTOE'),  # Using LBIGTOE as foot
        'right_ankle_angle': ('RKNEE', 'RANKLE', 'RBIGTOE'),  # Using RBIGTOE as foot
    }
    old_joints = {
        'left_elbow_angle': ('LSJC', 'LEJC', 'LWJC'),  # Old names
        'right_elbow_angle': ('RSJC', 'REJC', 'RWJC'),
        'left_knee_angle': ('LHJC', 'LKJC', 'LAJC'),
        'right_knee_angle': ('RHJC', 'RKJC', 'RAJC'),
        'left_shoulder_angle': ('NECK', 'LSJC', 'LEJC'),
        'right_shoulder_angle': ('NECK', 'RSJC', 'REJC'),
        'left_hip_angle': ('MIDHIP', 'LHJC', 'LKJC'),
        'right_hip_angle': ('MIDHIP', 'RHJC', 'RKJC'),
        'left_ankle_angle': ('LKJC', 'LAJC', 'LBTOE'),
        'right_ankle_angle': ('RKJC', 'RAJC', 'RBTOE'),
    }
    for angle in joints:
        # Try new names first
        a, b, c = joints[angle]
        a_x, a_y = f"{a}_X", f"{a}_Y"
        b_x, b_y = f"{b}_X", f"{b}_Y"
        c_x, c_y = f"{c}_X", f"{c}_Y"
        if all(col in df.columns for col in [a_x, a_y, b_x, b_y, c_x, c_y]):
            df[angle] = df.apply(lambda row: angle_2d(
                (row[a_x], row[a_y]),
                (row[b_x], row[b_y]),
                (row[c_x], row[c_y])
            ), axis=1)
        # Fallback to old names
        else:
            a, b, c = old_joints[angle]
            a_x, a_y = f"{a}_X", f"{a}_Y"
            b_x, b_y = f"{b}_X", f"{b}_Y"
            c_x, c_y = f"{c}_X", f"{c}_Y"
            if all(col in df.columns for col in [a_x, a_y, b_x, b_y, c_x, c_y]):
                df[angle] = df.apply(lambda row: angle_2d(
                    (row[a_x], row[a_y]),
                    (row[b_x], row[b_y]),
                    (row[c_x], row[c_y])
                ), axis=1)
    return df



def calculate_shot_metrics(pose_df, ball_df, fps=60):
    """
    Calculate basketball shot metrics entirely in FEET.
    Assumptions:
      - 'Basketball_X', 'Basketball_Y', 'Basketball_Z' in ball_df are in inches.
      - Hoops are located at (±501, 0) inches from center.
      - fps is frames per second, default 60.
    """
    metrics = {}
    INCHES_TO_FEET = 1 / 12  # Conversion factor

    try:
        # 1. Validate Pose and Ball DataFrames
        required_pose = [
            'LHEEL_X', 'LHEEL_Y', 'LBIGTOE_X', 'LBIGTOE_Y',
            'RHEEL_X', 'RHEEL_Y', 'RBIGTOE_X', 'RBIGTOE_Y',
            'MIDHIP_X', 'MIDHIP_Y', 'NECK_X', 'NECK_Y',
            'LSHOULDER_X', 'LSHOULDER_Y', 'RSHOULDER_X', 'RSHOULDER_Y'
        ]
        missing_pose = [col for col in required_pose if col not in pose_df.columns]
        if missing_pose:
            logger.warning(f"Missing columns in pose_df: {missing_pose}.")

        required_ball = ['Basketball_X', 'Basketball_Y', 'Basketball_Z']
        missing_ball = [col for col in required_ball if col not in ball_df.columns]
        if missing_ball:
            logger.error(f"Missing columns in ball_df: {missing_ball}.")
            return metrics, pose_df, ball_df

        # 2. Compute additional pose angles
        pose_df = calculate_foot_angles(pose_df)
        pose_df = calculate_rotation_angles(pose_df)

        # 3. Compute velocities in inches/second
        basketball_x = ball_df['Basketball_X']
        basketball_y = ball_df['Basketball_Y']
        basketball_z = ball_df['Basketball_Z']

        ball_df['velocity_x'] = basketball_x.diff() * fps
        ball_df['velocity_y'] = basketball_y.diff() * fps
        ball_df['velocity_z'] = basketball_z.diff() * fps

        ball_df['velocity_magnitude'] = np.sqrt(
            ball_df['velocity_x']**2 +
            ball_df['velocity_y']**2 +
            ball_df['velocity_z']**2
        )

        ball_df['velocity_magnitude'] = ball_df['velocity_magnitude'].replace(0, np.nan)
        ball_df['velocity_magnitude'] = (
            ball_df['velocity_magnitude']
            .interpolate(method='linear')
            .fillna(method='bfill')
            .fillna(method='ffill')
        )

        # 4. Identify key shot indices
        metrics['apex_idx'] = basketball_z.idxmax()
        apex_window_start = max(0, metrics['apex_idx'] - 75)
        metrics['release_idx'] = ball_df['velocity_magnitude'].iloc[apex_window_start:metrics['apex_idx']].idxmax()

        release_window_start = max(0, metrics['release_idx'] - 20)
        metrics['set_idx'] = ball_df.iloc[release_window_start:metrics['release_idx']]['Basketball_X'].idxmin()

        set_window_start = max(0, metrics['set_idx'] - 15)
        metrics['lift_idx'] = ball_df.iloc[set_window_start:metrics['set_idx']]['Basketball_X'].idxmax()

        metrics['rim_impact_idx'] = (basketball_z <= 120).idxmax()  # 10 ft = 120 inches

        # 5. Compute KPIs (convert to feet where applicable)
        release_point = ball_df.loc[metrics['release_idx']]
        player_x = release_point['Basketball_X']
        hoop_x = 501.0 if player_x >= 0 else -501.0
        hoop_y = 0.0

        dx = (hoop_x - release_point['Basketball_X']) * INCHES_TO_FEET
        dy = (hoop_y - release_point['Basketball_Y']) * INCHES_TO_FEET
        original_shot_distance = np.sqrt(dx**2 + dy**2)

        if original_shot_distance > 47.0:
            shot_distance = 94.0 - original_shot_distance
            flip = True
        else:
            shot_distance = original_shot_distance
            flip = False

        metrics['shot_distance'] = shot_distance
        metrics['original_shot_distance'] = original_shot_distance
        metrics['flip'] = flip
        metrics['hoop_x'] = hoop_x
        metrics['hoop_y'] = hoop_y

        metrics['release_height'] = release_point['Basketball_Z'] * INCHES_TO_FEET
        metrics['release_time'] = (metrics['release_idx'] - metrics['lift_idx']) / fps
        metrics['apex_height'] = basketball_z.max() * INCHES_TO_FEET

        # 6. Compute Release Angle
        post_release = ball_df.loc[metrics['release_idx']:metrics['release_idx'] + 3]
        dz = post_release['Basketball_Z'].diff().iloc[1:].mean() * INCHES_TO_FEET
        dxy = np.sqrt(post_release['Basketball_X'].diff()**2 + post_release['Basketball_Y'].diff()**2).iloc[1:].mean() * INCHES_TO_FEET
        metrics['release_angle'] = np.degrees(np.arctan2(dz, dxy))

        # 7. Compute Release Velocity
        if pd.isna(metrics['release_idx']) or metrics['release_idx'] >= len(ball_df):
            logger.error(f"Invalid release_idx: {metrics['release_idx']}, ball_df length: {len(ball_df)}")
            metrics['release_velocity'] = 0.0
        else:
            release_velocity_x = basketball_x.diff().fillna(0) * fps
            release_velocity_y = basketball_y.diff().fillna(0) * fps
            release_velocity_z = basketball_z.diff().fillna(0) * fps
            rv_x = release_velocity_x.iloc[metrics['release_idx']]
            rv_y = release_velocity_y.iloc[metrics['release_idx']]
            rv_z = release_velocity_z.iloc[metrics['release_idx']]
            release_velocity = np.sqrt(rv_x**2 + rv_y**2 + rv_z**2) * INCHES_TO_FEET
            metrics['release_velocity'] = 0.0 if pd.isna(release_velocity) or release_velocity < 0 else release_velocity

        # 8. Curvature computations (convert from 1/inches to 1/feet)
        metrics['curvature_side'] = savgol_filter(np.gradient(np.gradient(basketball_x)), 11, 3) * 12
        metrics['curvature_lateral'] = savgol_filter(np.gradient(np.gradient(basketball_y)), 11, 3) * 12
        metrics['release_curvature'] = metrics['curvature_side'][metrics['release_idx']] * 12

        # 9. Classify release angle
        metrics['release_class'] = classify_release_angle(
            metrics.get('release_angle', 0),
            metrics.get('shot_distance', 0),
            metrics.get('release_height', 0)
        )

        # 10. Lateral deviation (using provided function)
        lateral_dev = calculate_lateral_deviation(
            ball_df,
            metrics['release_idx'],
            hoop_x=metrics['hoop_x'],
            hoop_y=metrics['hoop_y']
        )
        if lateral_dev:
            metrics['lateral_deviation'] = lateral_dev[0]  # Already in feet, preserve full precision
            metrics['lateral_final_x'] = lateral_dev[1]
            metrics['lateral_actual_y'] = lateral_dev[2]
            metrics['lateral_expected_y'] = lateral_dev[3]
        else:
            metrics['lateral_deviation'] = 0.0  # Only set to 0.0 if calculation fails

        # 11. Additional Pose Computations (update this section)
            pose_df = compute_joint_angles(pose_df)  # Already calculates some angles
            pose_df = calculate_centroid(pose_df)
            pose_df = calculate_stability(pose_df)
            pose_df = calculate_guide_hand_release(pose_df)

            # Add new joint angle calculations for consistency with visuals
            def calculate_angle(df, a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z):
                ab = np.sqrt((df[a_x] - df[b_x])**2 + (df[a_y] - df[b_y])**2 + (df[a_z] - df[b_z])**2)
                bc = np.sqrt((df[c_x] - df[b_x])**2 + (df[c_y] - df[b_y])**2 + (df[c_z] - df[b_z])**2)
                ac = np.sqrt((df[a_x] - df[c_x])**2 + (df[a_y] - df[c_y])**2 + (df[a_z] - df[c_z])**2)
                cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
                cos_angle = np.clip(cos_angle, -1, 1)
                return np.degrees(np.arccos(cos_angle))

            pose_df['elbow_angle'] = calculate_angle(
                pose_df, 'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z',
                'RWRIST_X', 'RWRIST_Y', 'RWRIST_Z'
            )
            pose_df['shoulder_angle'] = calculate_angle(
                pose_df, 'MIDHIP_X', 'MIDHIP_Y', 'MIDHIP_Z',
                'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z'
            )
            pose_df['wrist_angle'] = calculate_angle(
                pose_df, 'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z',
                'RWRIST_X', 'RWRIST_Y', 'RWRIST_Z',
                'RTHUMB_X', 'RTHUMB_Y', 'RTHUMB_Z'
            )
            pose_df['hip_angle'] = calculate_angle(
                pose_df, 'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                'RHIP_X', 'RHIP_Y', 'RHIP_Z',
                'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z'
            )
            pose_df['knee_angle'] = calculate_angle(
                pose_df, 'RHIP_X', 'RHIP_Y', 'RHIP_Z',
                'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z',
                'RANKLE_X', 'RANKLE_Y', 'RANKLE_Z'
            )
            pose_df['ankle_angle'] = calculate_angle(
                pose_df, 'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z',
                'RANKLE_X', 'RANKLE_Y', 'RANKLE_Z',
                'RBIGTOE_X', 'RBIGTOE_Y', 'RBIGTOE_Z'
            )

    except Exception as e:
        logger.error(f"Error calculating shot metrics: {str(e)}")
        metrics['release_velocity'] = 0.0  # Fallback

    return metrics, pose_df, ball_df

def calculate_lateral_deviation(df, release_index, hoop_x=501.0, hoop_y=0.0):
    """
    Calculate lateral deviation of the ball from the intended shot line to the hoop, in feet.
    
    Parameters:
    - df: DataFrame with 'Basketball_X', 'Basketball_Y', 'Basketball_Z' in inches.
    - release_index: Index of the release frame.
    - hoop_x: Hoop x-position in inches (default 501.0 or -501.0 based on shot side).
    - hoop_y: Hoop y-position in inches (default 0.0, center of court).
    
    Returns:
    - tuple: (deviation in feet, final_x in inches, actual_y in inches, expected_y in inches)
             deviation is positive (right) or negative (left) of the intended line.
    """
    INCHES_TO_FEET = 1 / 12
    try:
        # Get release point coordinates (in inches)
        release_x = df.loc[release_index, 'Basketball_X']
        release_y = df.loc[release_index, 'Basketball_Y']

        # Define the intended shot line from release point to hoop
        shot_dx = hoop_x - release_x  # Delta x in inches
        shot_dy = hoop_y - release_y  # Delta y in inches

        # Find the last frame before the ball descends below hoop height (10 ft = 120 inches)
        hoop_height = 120.0  # 10 feet in inches
        final_idx = df[df['Basketball_Z'] >= hoop_height].index[-1]
        final_x = df.loc[final_idx, 'Basketball_X']
        actual_y = df.loc[final_idx, 'Basketball_Y']

        # Parametric equation of the shot line: (x(t), y(t)) = (release_x + t * shot_dx, release_y + t * shot_dy)
        # Solve for t when x = final_x: t = (final_x - release_x) / shot_dx
        if abs(shot_dx) < 1e-6:  # Avoid division by zero
            logger.error("Shot direction x-component too small; cannot compute deviation.")
            return None
        t = (final_x - release_x) / shot_dx

        # Expected y-position on the shot line at final_x
        expected_y = release_y + t * shot_dy

        # Lateral deviation (perpendicular distance from the shot line)
        line_vec = np.array([shot_dx, shot_dy])
        point_vec = np.array([final_x - release_x, actual_y - release_y])
        line_len_sq = shot_dx**2 + shot_dy**2
        if line_len_sq < 1e-6:  # Avoid division by zero
            logger.error("Shot line length too small; cannot compute deviation.")
            return None

        # Cross product magnitude in 2D gives the area of parallelogram
        cross = shot_dx * (actual_y - release_y) - shot_dy * (final_x - release_x)
        deviation_inches = abs(cross) / np.sqrt(line_len_sq)  # Perpendicular distance
        deviation_feet = deviation_inches * INCHES_TO_FEET  # Preserve full precision

        # Determine sign: positive if right of line, negative if left
        sign = np.sign(cross)
        if release_x < 0:  # Flip sign for left-side shots
            sign *= -1
        deviation_feet *= sign

        return (deviation_feet, final_x, actual_y, expected_y)

    except Exception as e:
        logger.error(f"Error calculating lateral deviation: {str(e)}")
        return None



import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import comb
from numpy import trapz
from scipy.signal import savgol_filter

def fit_bezier_curve(x_data, z_data, n=6):
    m = len(x_data)
    if m < n+1:
        n = m - 1
    dx = np.diff(x_data)
    dz = np.diff(z_data)
    distances = np.sqrt(dx**2 + dz**2)
    cum_dist = np.concatenate(([0], np.cumsum(distances)))
    total_len = cum_dist[-1]
    t_data = cum_dist / total_len if total_len > 1e-12 else np.linspace(0, 1, m)
    T = np.zeros((m, n+1))
    for i in range(m):
        t = t_data[i]
        for k in range(n+1):
            T[i, k] = comb(n, k) * (1 - t)**(n - k) * (t**k)
    XZ = np.column_stack((x_data, z_data))
    P, _, _, _ = np.linalg.lstsq(T, XZ, rcond=None)
    return P, t_data

def evaluate_bezier(P, t):
    n = len(P) - 1
    t = np.array(t, ndmin=1)
    x_out = np.zeros_like(t)
    z_out = np.zeros_like(t)
    for k in range(n+1):
        b_t = comb(n, k) * (1 - t)**(n - k) * (t**k)
        x_out += P[k, 0] * b_t
        z_out += P[k, 1] * b_t
    return x_out, z_out

def bezier_signed_curvature(P, t):
    n = len(P) - 1
    t = np.array(t, ndmin=1)
    if n == 0:
        return np.zeros_like(t)
    dP = n * (P[1:] - P[:-1])
    if n >= 2:
        ddP = (n-1) * (dP[1:] - dP[:-1])
    else:
        ddP = np.zeros_like(dP)
    x1, z1 = evaluate_bezier(dP, t)
    if n >= 2:
        x2, z2 = evaluate_bezier(ddP, t)
    else:
        x2, z2 = np.zeros_like(t), np.zeros_like(t)
    numerator = x1 * z2 - z1 * x2
    denominator = (x1**2 + z1**2)**1.5
    signed_curv = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>1e-12)
    return signed_curv

def bezier_curvature(P, t, scale=1.0):
    """
    Returns the absolute curvature scaled by the provided factor.
    Adjust 'scale' until your peak curvature (κmax) values fall in the 4–8 m⁻¹ range.
    """
    return scale * np.abs(bezier_signed_curvature(P, t))




# ---------------------------
# Acceleration Computation from Raw Data (with Smoothing)
# ---------------------------
def compute_normalized_acceleration_from_raw(df_ball, lift_idx, release_idx, fps):
    # If the velocity_magnitude column doesn't exist, compute it.
    if 'velocity_magnitude' not in df_ball.columns:
        # Assuming your ball coordinates are in 'Basketball_X', 'Basketball_Y', 'Basketball_Z'
        df_ball['velocity_x'] = df_ball['Basketball_X'].diff() * fps
        df_ball['velocity_y'] = df_ball['Basketball_Y'].diff() * fps
        df_ball['velocity_z'] = df_ball['Basketball_Z'].diff() * fps
        df_ball['velocity_magnitude'] = np.sqrt(
            df_ball['velocity_x']**2 + 
            df_ball['velocity_y']**2 + 
            df_ball['velocity_z']**2
        )
        
    segment = df_ball.iloc[lift_idx:release_idx+1]
    vel = segment['velocity_magnitude'].to_numpy()
    acc_raw = np.gradient(vel) * fps
    # Apply smoothing if enough samples exist:
    if len(acc_raw) < 3:
        smooth_acc = acc_raw
    else:
        win_length = 11
        if len(acc_raw) < win_length:
            win_length = len(acc_raw) if len(acc_raw) % 2 == 1 else len(acc_raw) - 1
        smooth_acc = savgol_filter(acc_raw, window_length=win_length, polyorder=2)
    max_val = np.max(np.abs(smooth_acc))
    norm_acc = smooth_acc / max_val if max_val > 0 else smooth_acc
    t_raw = np.linspace(0, 1, len(acc_raw))
    return t_raw, norm_acc, smooth_acc


# ---------------------------
# Revised Plotting Function for Curvature Analysis
# ---------------------------
def plot_curvature_analysis(df_ball, metrics, fps=60, weighting_exponent=3, 
                            num_interp=300, bezier_order=6, curvature_scale=2.3):
    INCHES_TO_FEET = 1 / 12
    import numpy as np
    import plotly.graph_objects as go

    # Extract indices from metrics (ensure these keys exist)
    lift_idx   = int(metrics.get('lift_idx', 0))
    release_idx = int(metrics.get('release_idx', lift_idx + 1))
    
    # Use only lift_idx to release_idx (no extra frames)
    total_frames = release_idx - lift_idx
    if total_frames <= 0:
        raise ValueError("Invalid indices: release_idx must be greater than lift_idx.")
    
    # Compute time parameters relative to lift_idx:
    lift_t    = 0.0  # Start at 0%
    release_t = 1.0  # End at 100%
    
    # t_fine for interpolation over [0,1]
    t_fine = np.linspace(0, 1, num_interp)
    
    # LEFT PANEL: Normalized Acceleration (calculated from lift_idx to release_idx, smoothed)
    t_raw, norm_acc, acc_raw = compute_normalized_acceleration_from_raw(df_ball, lift_idx, release_idx, fps)
    # Apply smoothing to norm_acc if enough data
    if len(norm_acc) > 3:
        window_length = min(11, len(norm_acc) - 1)
        if window_length % 2 == 0:
            window_length += 1
        norm_acc = savgol_filter(norm_acc, window_length=window_length, polyorder=2)
    norm_acc_interp = np.interp(t_fine, t_raw, norm_acc)
    
    # Convert velocity to feet/second and smooth
    velocity_segment = df_ball['velocity_magnitude'].iloc[lift_idx:release_idx+1].to_numpy() * INCHES_TO_FEET
    if len(velocity_segment) > 3:
        window_length = min(11, len(velocity_segment) - 1)
        if window_length % 2 == 0:
            window_length += 1
        velocity_segment = savgol_filter(velocity_segment, window_length=window_length, polyorder=2)
    velocity_interp = np.interp(t_fine, t_raw, velocity_segment)
    
    # Compute Transition Marker using a Bézier fit on (Basketball_X, Basketball_Z), smoothed
    seg_x = df_ball['Basketball_X'].iloc[lift_idx:release_idx+1].to_numpy() * INCHES_TO_FEET  # Convert to feet
    seg_z = df_ball['Basketball_Z'].iloc[lift_idx:release_idx+1].to_numpy() * INCHES_TO_FEET  # Convert to feet
    if len(seg_x) > 3:
        window_length = min(11, len(seg_x) - 1)
        if window_length % 2 == 0:
            window_length += 1
        seg_x = savgol_filter(seg_x, window_length=window_length, polyorder=2)
        seg_z = savgol_filter(seg_z, window_length=window_length, polyorder=2)
    P_side, _ = fit_bezier_curve(seg_x, seg_z, n=bezier_order)
    signed_side = bezier_signed_curvature(P_side, t_fine)
    
    transition_idx = None
    for i in range(len(signed_side) - 1):
        if signed_side[i] * signed_side[i+1] < 0:
            transition_idx = i
            break
    if transition_idx is None:
        transition_idx = np.argmin(np.abs(signed_side))
    t_transition = t_fine[transition_idx] if transition_idx is not None else 0.5  # Default to midpoint if no transition
    
    l_exp = weighting_exponent
    w = (l_exp + 1) * (t_fine ** l_exp)
    weighted_acc = np.abs(norm_acc_interp) * w
    
    # RIGHT PANEL: Lateral (Side View) Curvature Analysis, smoothed
    seg_y = df_ball['Basketball_Y'].iloc[lift_idx:release_idx+1].to_numpy() * INCHES_TO_FEET  # Convert to feet
    if len(seg_y) > 3:
        window_length = min(11, len(seg_y) - 1)
        if window_length % 2 == 0:
            window_length += 1
        seg_y = savgol_filter(seg_y, window_length=window_length, polyorder=2)
    P_lat, _ = fit_bezier_curve(seg_y, seg_z, n=bezier_order)
    # Apply curvature scaling factor (curvature in 1/feet)
    lateral_curve = bezier_curvature(P_lat, t_fine, scale=curvature_scale / 12)  # Adjust scale for feet
    if len(lateral_curve) > 3:
        window_length = min(11, len(lateral_curve) - 1)
        if window_length % 2 == 0:
            window_length += 1
        lateral_curve = savgol_filter(lateral_curve, window_length=window_length, polyorder=2)
    velocity_lateral = velocity_interp  # Already in feet/second, smoothed above
    weighted_lateral = np.abs(lateral_curve) * w
    
    t_percent = t_fine * 100

    COLOR_PALETTE = {
        'lift': 'rgba(147, 112, 219, 1)',
        'release': 'rgba(255, 102, 102, 1)',
        'acceleration': 'rgba(0, 0, 0, 1)',
        'curvature': 'rgba(0, 0, 0, 1)',
        'velocity': 'rgba(107, 174, 214, 1)',
        'weighted_acc': 'rgba(255, 102, 102, 0.3)',
        'weighted_lat': 'rgba(31, 119, 180, 0.3)'
    }
    DASH_STYLES = {
        'lift': 'dash',
        'release': 'dashdot'
    }
    
    # Dummy traces for legend
    dummy_lift = go.Scatter(x=[None], y=[None], mode='lines',
                            line=dict(color=COLOR_PALETTE['lift'], dash=DASH_STYLES['lift'], width=2),
                            name=f"Lift (index: {lift_idx})")
    dummy_release = go.Scatter(x=[None], y=[None], mode='lines',
                               line=dict(color=COLOR_PALETTE['release'], dash=DASH_STYLES['release'], width=2),
                               name=f"Release (index: {release_idx})")
    dummy_transition = go.Scatter(x=[None], y=[None], mode='markers',
                                  marker=dict(color='red', size=12, symbol='asterisk'),
                                  name=f"Transition (index: {lift_idx + int(t_transition * total_frames)})")
    dummy_velocity = go.Scatter(x=[None], y=[None], mode='lines',
                                line=dict(color=COLOR_PALETTE['velocity'], width=2),
                                name="Velocity (ft/s)")
    
    # Build subplots with two panels
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Normalized Ball Acceleration", "Lift-Release Curvature Analysis"),
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        horizontal_spacing=0.15
    )
    
    # LEFT PANEL
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=weighted_acc,
            mode='none',
            fill='tozeroy',
            fillcolor=COLOR_PALETTE['weighted_acc'],
            name='Weighted Area (Acceleration)'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=norm_acc_interp,
            mode='lines',
            name='Normalized Acceleration',
            line=dict(color=COLOR_PALETTE['acceleration'], width=2)  # Reduced line width
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=velocity_interp,
            mode='lines',
            name='Velocity (ft/s)',
            line=dict(color=COLOR_PALETTE['velocity'], width=2),
            showlegend=False
        ),
        row=1, col=1, secondary_y=True
    )
    for phase, phase_t in zip(['lift', 'release'], [lift_t, release_t]):
        if phase_t is not None:
            fig.add_vline(x=phase_t*100, line=dict(color=COLOR_PALETTE[phase], width=2, dash=DASH_STYLES[phase]),
                          row=1, col=1)
    if t_transition is not None:
        fig.add_trace(
            go.Scatter(
                x=[t_transition*100],
                y=[norm_acc_interp[transition_idx]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='asterisk'),  # Reduced marker size
                name='Transition',
                showlegend=False
            ),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="% of Release", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Acceleration", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Velocity (ft/s)", row=1, col=1, secondary_y=True)
    
    # RIGHT PANEL
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=weighted_lateral,
            mode='none',
            fill='tozeroy',
            fillcolor=COLOR_PALETTE['weighted_lat'],
            name='Weighted Area (Lateral)'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=lateral_curve,
            mode='lines',
            name='Side View Curvature (1/ft)',
            line=dict(color=COLOR_PALETTE['curvature'], width=2)  # Reduced line width
        ),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=t_percent, y=velocity_lateral,
            mode='lines',
            name='Velocity (ft/s)',
            line=dict(color=COLOR_PALETTE['velocity'], width=2),
            showlegend=False
        ),
        row=1, col=2, secondary_y=True
    )
    for phase, phase_t in zip(['lift', 'release'], [lift_t, release_t]):
        if phase_t is not None:
            fig.add_vline(x=phase_t*100, line=dict(color=COLOR_PALETTE[phase], width=2, dash=DASH_STYLES[phase]),
                          row=1, col=2)
    fig.update_xaxes(title_text="% of Release", row=1, col=2)
    fig.update_yaxes(title_text="Curvature (1/ft)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Velocity (ft/s)", row=1, col=2, secondary_y=True)
    
    fig.update_layout(
        height=400,  # Reduced height
        width=800,   # Reduced width
        margin=dict(t=60, b=40, l=40, r=40),  # Reduced margins
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5
        )
    )
    fig.add_trace(dummy_lift)
    fig.add_trace(dummy_release)
    fig.add_trace(dummy_transition)
    fig.add_trace(dummy_velocity)
    
    return fig

def calculate_lateral_deviation(df, release_index, hoop_x=501.0, hoop_y=0.0):
    """
    Calculate lateral deviation of the ball from the intended shot line to the hoop, in feet.
    
    Parameters:
    - df: DataFrame with 'Basketball_X', 'Basketball_Y', 'Basketball_Z' in inches.
    - release_index: Index of the release frame.
    - hoop_x: Hoop x-position in inches (default 501.0 or -501.0 based on shot side).
    - hoop_y: Hoop y-position in inches (default 0.0, center of court).
    
    Returns:
    - deviation: Lateral deviation in feet (positive = right, negative = left of intended line).
    - final_x: Final x-position in inches where deviation is measured.
    - actual_y: Actual y-position in inches at final_x.
    - expected_y: Expected y-position in inches on the intended shot line at final_x.
    """
    INCHES_TO_FEET = 1 / 12
    try:
        # Get release point coordinates (in inches)
        release_x = df.loc[release_index, 'Basketball_X']
        release_y = df.loc[release_index, 'Basketball_Y']

        # Define the intended shot line from release point to hoop
        shot_dx = hoop_x - release_x  # Delta x in inches
        shot_dy = hoop_y - release_y  # Delta y in inches

        # Find the last frame before the ball descends below hoop height (10 ft = 120 inches)
        hoop_height = 120.0  # 10 feet in inches
        final_idx = df[df['Basketball_Z'] >= hoop_height].index[-1]
        final_x = df.loc[final_idx, 'Basketball_X']
        actual_y = df.loc[final_idx, 'Basketball_Y']

        # Parametric equation of the shot line: (x(t), y(t)) = (release_x + t * shot_dx, release_y + t * shot_dy)
        # Solve for t when x = final_x: t = (final_x - release_x) / shot_dx
        if abs(shot_dx) < 1e-6:  # Avoid division by zero
            return None
        t = (final_x - release_x) / shot_dx

        # Expected y-position on the shot line at final_x
        expected_y = release_y + t * shot_dy

        # Lateral deviation (perpendicular distance from the shot line)
        # Use point-to-line distance formula in 2D
        line_vec = np.array([shot_dx, shot_dy])
        point_vec = np.array([final_x - release_x, actual_y - release_y])
        line_len_sq = shot_dx**2 + shot_dy**2
        if line_len_sq < 1e-6:  # Avoid division by zero
            return None

        # Cross product magnitude in 2D gives the area of parallelogram
        cross = shot_dx * (actual_y - release_y) - shot_dy * (final_x - release_x)
        deviation_inches = abs(cross) / np.sqrt(line_len_sq)  # Perpendicular distance
        deviation_feet = deviation_inches * INCHES_TO_FEET

        # Determine sign: positive if right of line, negative if left
        # Use the sign of the cross product
        sign = np.sign(cross)
        # Adjust sign based on court orientation (if shooting from left side, flip)
        if release_x < 0:
            sign *= -1  # Flip sign for left-side shots
        deviation_feet *= sign

        return (deviation_feet, final_x, actual_y, expected_y)

    except Exception as e:
        logger.error(f"Error calculating lateral deviation: {str(e)}")
        return None

# -------------------------------------------------------------------------------------
# Data Cleaning, Angles, and Trimming
# -------------------------------------------------------------------------------------

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

def clean_pose_data(pose_df):
    """
    Cleans the pose DataFrame by renaming columns and computing joint angles.

    Parameters:
    - pose_df (pd.DataFrame): Raw pose DataFrame.

    Returns:
    - pd.DataFrame: Cleaned pose DataFrame with joint angles computed.
    """
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

def calculate_body_alignment(df, release_idx, hoop_x=501.0, hoop_y=0.0):
    """Calculate body alignment metrics AT SPECIFIC FRAME with dynamic hoop position."""
    metrics = {}
    try:
        frame_data = df.iloc[release_idx]
        basket_pos = np.array([hoop_x, hoop_y])  # Use dynamic hoop position
        
        # Feet alignment (new names: RBIGTOE_X, LBIGTOE_X)
        if all(col in frame_data for col in ['RBIGTOE_X', 'LBIGTOE_X', 'RBIGTOE_Y', 'LBIGTOE_Y']):
            feet_vec = np.array([
                frame_data['RBIGTOE_X'] - frame_data['LBIGTOE_X'],
                frame_data['RBIGTOE_Y'] - frame_data['LBIGTOE_Y']
            ])
            metrics['feet_basket_angle'] = vector_angle(feet_vec[0], feet_vec[1], basket_pos[0], basket_pos[1])
        # Fallback to old names (RBTOE_X, LBTOE_X)
        elif all(col in frame_data for col in ['RBTOE_X', 'LBTOE_X', 'RBTOE_Y', 'LBTOE_Y']):
            feet_vec = np.array([
                frame_data['RBTOE_X'] - frame_data['LBTOE_X'],
                frame_data['RBTOE_Y'] - frame_data['LBTOE_Y']
            ])
            metrics['feet_basket_angle'] = vector_angle(feet_vec[0], feet_vec[1], basket_pos[0], basket_pos[1])
        else:
            raise KeyError("Missing foot data (RBIGTOE_X/LBIGTOE_X or RBTOE_X/LBTOE_X)")
        
        # Hips alignment (new names: RHIP_X, LHIP_X)
        if all(col in frame_data for col in ['RHIP_X', 'LHIP_X', 'RHIP_Y', 'LHIP_Y']):
            hips_vec = np.array([
                frame_data['RHIP_X'] - frame_data['LHIP_X'],
                frame_data['RHIP_Y'] - frame_data['LHIP_Y']
            ])
            metrics['hips_basket_angle'] = vector_angle(hips_vec[0], hips_vec[1], basket_pos[0], basket_pos[1])
        # Fallback to old names (RHJC_X, LHJC_X)
        elif all(col in frame_data for col in ['RHJC_X', 'LHJC_X', 'RHJC_Y', 'LHJC_Y']):
            hips_vec = np.array([
                frame_data['RHJC_X'] - frame_data['LHJC_X'],
                frame_data['RHJC_Y'] - frame_data['LHJC_Y']
            ])
            metrics['hips_basket_angle'] = vector_angle(hips_vec[0], hips_vec[1], basket_pos[0], basket_pos[1])
        else:
            raise KeyError("Missing hip data (RHIP_X/LHIP_X or RHJC_X/LHJC_X)")
        
        # Body segment offsets
        if 'feet_basket_angle' in metrics and 'hips_basket_angle' in metrics:
            metrics['feet_hips_misalignment'] = vector_angle(feet_vec[0], feet_vec[1], hips_vec[0], hips_vec[1])
        
        # Shoulders alignment (new names: RSHOULDER_X, LSHOULDER_X)
        if all(col in frame_data for col in ['RSHOULDER_X', 'LSHOULDER_X', 'RSHOULDER_Y', 'LSHOULDER_Y']):
            shoulders_vec = np.array([
                frame_data['RSHOULDER_X'] - frame_data['LSHOULDER_X'],
                frame_data['RSHOULDER_Y'] - frame_data['LSHOULDER_Y']
            ])
            metrics['hips_shoulders_misalignment'] = vector_angle(hips_vec[0], hips_vec[1], shoulders_vec[0], shoulders_vec[1])
        # Fallback to old names (RSJC_X, LSJC_X)
        elif all(col in frame_data for col in ['RSJC_X', 'LSJC_X', 'RSJC_Y', 'LSJC_Y']):
            shoulders_vec = np.array([
                frame_data['RSJC_X'] - frame_data['LSJC_X'],
                frame_data['RSJC_Y'] - frame_data['LSJC_Y']
            ])
            metrics['hips_shoulders_misalignment'] = vector_angle(hips_vec[0], hips_vec[1], shoulders_vec[0], shoulders_vec[1])
        else:
            raise KeyError("Missing shoulder data (RSHOULDER_X/LSHOULDER_X or RSJC_X/LSJC_X)")
        
    except (KeyError, IndexError) as e:
        st.error(f"Missing data for alignment calculation: {str(e)}")
        return {
            'feet_basket_angle': 0.0,
            'hips_basket_angle': 0.0,
            'feet_hips_misalignment': 0.0,
            'hips_shoulders_misalignment': 0.0
        }
    
    return metrics

def vector_angle(v1x, v1y, v2x, v2y):
    """
    Calculate the signed angle between two vectors in degrees.
    """
    dot = v1x * v2x + v1y * v2y
    det = v1x * v2y - v1y * v2x
    return np.degrees(np.arctan2(det, dot))

def create_alignment_diagram(df, basket_position=(41.75, 0)):
    """
    Create a visualization of body alignment relative to the basket.

    Parameters:
        df (pd.DataFrame): DataFrame containing pose data.
        basket_position (tuple): Coordinates of the basket (x, y). Default is (41.75, 0).

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure showing body alignment.
    """
    fig = go.Figure()

    # Plot basket position
    fig.add_trace(go.Scatter(
        x=[basket_position[0]], y=[basket_position[1]],
        mode='markers', marker=dict(size=20, color='red'),
        name='Basket'
    ))

    # Plot body segments
    if 'LBTOE_X' in df.columns and 'RBTOE_X' in df.columns:
        # Feet
        fig.add_trace(go.Scatter(
            x=[df['LBTOE_X'].mean(), df['RBTOE_X'].mean()],
            y=[df['LBTOE_Y'].mean(), df['RBTOE_Y'].mean()],
            mode='lines+markers', name='Feet'
        ))

    if 'LHJC_X' in df.columns and 'RHJC_X' in df.columns:
        # Hips
        fig.add_trace(go.Scatter(
            x=[df['LHJC_X'].mean(), df['RHJC_X'].mean()],
            y=[df['LHJC_Y'].mean(), df['RHJC_Y'].mean()],
            mode='lines+markers', name='Hips'
        ))

    if 'LSJC_X' in df.columns and 'RSJC_X' in df.columns:
        # Shoulders
        fig.add_trace(go.Scatter(
            x=[df['LSJC_X'].mean(), df['RSJC_X'].mean()],
            y=[df['LSJC_Y'].mean(), df['RSJC_Y'].mean()],
            mode='lines+markers', name='Shoulders'
        ))

    # Update layout
    fig.update_layout(
        title="Body Alignment Visualization",
        xaxis_title="X Position (ft)",
        yaxis_title="Y Position (ft)",
        showlegend=True,
        template="plotly_white"
    )

    return fig


def plot_shot_location(ball_df, metrics, pose_df=None):
    """
    Create a 2D visualization of the shot location on a half basketball court.
    
    Parameters:
    - ball_df: DataFrame with 'Basketball_X', 'Basketball_Y', and optionally 'OUTCOME'/'IS_MADE' in inches.
    - metrics: Dictionary with 'lift_idx', 'release_idx', 'hoop_x', 'hoop_y'.
    - pose_df: Optional DataFrame with pose data (e.g., 'MIDHIP_X', 'MIDHIP_Y') for foot position.
    
    Returns:
    - fig: Plotly figure object.
    """
    import plotly.graph_objects as go
    import numpy as np

    INCHES_TO_FEET = 1 / 12

    # Get shot location at lift_idx (prefer foot position, fallback to ball position)
    lift_idx = metrics['lift_idx']
    if pose_df is not None and 'MIDHIP_X' in pose_df.columns and 'MIDHIP_Y' in pose_df.columns:
        shot_x = pose_df.loc[lift_idx, 'MIDHIP_X']
        shot_y = pose_df.loc[lift_idx, 'MIDHIP_Y']
        source_note = " (Foot Position at Lift)"
    else:
        shot_x = ball_df.loc[lift_idx, 'Basketball_X']
        shot_y = ball_df.loc[lift_idx, 'Basketball_Y']
        source_note = " (Ball at Lift)"
        logger.warning(f"Using ball position at lift_idx {lift_idx} as pose data is unavailable.")

    # Determine court side (flip court layout if shot_x is negative, keep shot coordinates unchanged)
    flip_court = shot_x < 0
    if flip_court:
        hoop_x = -501  # Flip hoop to left side
        logger.debug(f"Flipping court layout: shot_x = {shot_x} is negative, hoop_x = {hoop_x}")
    else:
        hoop_x = 501  # Default right side
        logger.debug(f"Court not flipped: shot_x = {shot_x} is positive, hoop_x = {hoop_x}")

    # Debug OUTCOME and IS_MADE presence
    logger.debug(f"ball_df columns: {ball_df.columns.tolist()}")
    logger.debug(f"ball_df index: {ball_df.index.tolist()}")
    logger.debug(f"lift_idx: {lift_idx}")
    if 'OUTCOME' in ball_df.columns:
        logger.debug(f"OUTCOME values sample: {ball_df['OUTCOME'].head().tolist()}")
    if 'IS_MADE' in ball_df.columns:
        logger.debug(f"IS_MADE values sample: {ball_df['IS_MADE'].head().tolist()}")

    # Determine marker based on shot outcome
    if 'OUTCOME' in ball_df.columns:
        try:
            outcome = ball_df.loc[lift_idx, 'OUTCOME']
            logger.debug(f"OUTCOME at lift_idx {lift_idx}: {outcome}")
            if pd.isna(outcome):
                marker_symbol = 'x'
                marker_color = 'grey'
                marker_name = f'Shot Location (Unknown Outcome)'
                logger.warning(f"OUTCOME is NaN at lift_idx {lift_idx}. Using grey 'X'.")
            elif outcome == 'Y':
                marker_symbol = 'circle'  # Filled circle for make
                marker_color = '#90EE90'  # Pastel green
                marker_name = f'Shot Location (Make)'
            elif outcome == 'N':
                marker_symbol = 'x'
                marker_color = '#FF9999'  # Pastel red
                marker_name = f'Shot Location (Miss)'
            else:
                marker_symbol = 'x'
                marker_color = 'grey'
                marker_name = f'Shot Location (Unexpected Outcome: {outcome})'
                logger.warning(f"Unexpected OUTCOME value '{outcome}' at lift_idx {lift_idx}. Using grey 'X'.")
        except KeyError:
            marker_symbol = 'x'
            marker_color = 'grey'
            marker_name = f'Shot Location (Index Error)'
            logger.error(f"lift_idx {lift_idx} not in ball_df index for OUTCOME. Using grey 'X'.")
    elif 'IS_MADE' in ball_df.columns:
        try:
            is_made = ball_df.loc[lift_idx, 'IS_MADE']
            logger.debug(f"IS_MADE at lift_idx {lift_idx}: {is_made}")
            if pd.isna(is_made):
                marker_symbol = 'x'
                marker_color = 'grey'
                marker_name = f'Shot Location (Unknown Outcome)'
                logger.warning(f"IS_MADE is NaN at lift_idx {lift_idx}. Using grey 'X'.")
            elif is_made in [True, 'TRUE', 1]:
                marker_symbol = 'circle'  # Filled circle for make
                marker_color = '#90EE90'  # Pastel green
                marker_name = f'Shot Location (Make)'
            elif is_made in [False, 'FALSE', 0]:
                marker_symbol = 'x'
                marker_color = '#FF9999'  # Pastel red
                marker_name = f'Shot Location (Miss)'
            else:
                marker_symbol = 'x'
                marker_color = 'grey'
                marker_name = f'Shot Location (Unexpected IS_MADE: {is_made})'
                logger.warning(f"Unexpected IS_MADE value '{is_made}' at lift_idx {lift_idx}. Using grey 'X'.")
        except KeyError:
            marker_symbol = 'x'
            marker_color = 'grey'
            marker_name = f'Shot Location (Index Error)'
            logger.error(f"lift_idx {lift_idx} not in ball_df index for IS_MADE. Using grey 'X'.")
    else:
        marker_symbol = 'x'
        marker_color = 'grey'
        marker_name = f'Shot Location (No Outcome Data)'
        logger.warning(f"Neither OUTCOME nor IS_MADE available at lift_idx {lift_idx}. Using grey 'X'.")

    # Log final marker settings
    logger.debug(f"Marker settings - symbol: {marker_symbol}, color: {marker_color}, name: {marker_name}")

    # Court dimensions in inches (centered at (0, 0) for base layout, flipped if needed)
    court_length = 564  # Half-court length from center to right edge
    court_width = 600   # Full width (-300 to 300)
    hoop_y = 0
    free_throw_x = 336 if not flip_court else -336  # Adjust for flip
    paint_width = 192  # 16 ft = 192 inches, -96 to 96
    paint_end = 564 if not flip_court else -564  # Extend to court end
    three_point_radius = 285  # Distance from hoop to peak (501 - 216)

    # Create figure
    fig = go.Figure()

    # Court boundary (adjusted for flip)
    court_x = [-court_length/2, court_length, court_length, -court_length/2, -court_length/2] if not flip_court else [court_length/2, -court_length, -court_length, court_length/2, court_length/2]
    fig.add_trace(
        go.Scatter(
            x=court_x,
            y=[-court_width/2, -court_width/2, court_width/2, court_width/2, -court_width/2],
            mode='lines',
            line=dict(color='black', width=2),
            name='Court Boundary'
        )
    )

    # Hoop (unfilled ring, light orange)
    fig.add_trace(
        go.Scatter(
            x=[hoop_x],
            y=[hoop_y],
            mode='markers',
            marker=dict(size=10, color='#FFA07A', symbol='circle-open'),  # Light orange ring
            name='Hoop'
        )
    )

    # Paint outline (rectangle from 336 to 564, y = ±96, flipped if needed)
    paint_x = [free_throw_x, paint_end, paint_end, free_throw_x, free_throw_x]
    fig.add_trace(
        go.Scatter(
            x=paint_x,
            y=[-paint_width/2, -paint_width/2, paint_width/2, paint_width/2, -paint_width/2],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Paint'
        )
    )

    # 3-point arc (semicircle centered at hoop, trimmed to match endpoints)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)  # Full semicircle
    three_x_full = hoop_x - three_point_radius * np.cos(theta) if not flip_court else hoop_x + three_point_radius * np.cos(theta)
    three_y_full = hoop_y + three_point_radius * np.sin(theta)

    # Trim arc to reach exactly (396, ±264) or flipped equivalent
    three_point_end_x = 396 if not flip_court else -396
    mask = (three_y_full >= -264) & (three_y_full <= 264) & (three_x_full >= 216 if not flip_court else three_x_full <= -216)
    three_x = three_x_full[mask]
    three_y = three_y_full[mask]
    
    # Ensure arc connects to (396, ±264) or (-396, ±264) by forcing endpoints
    if not flip_court:
        three_x = np.append(three_x, 396)
        three_y = np.append(three_y, 264)
        three_x = np.insert(three_x, 0, 396)
        three_y = np.insert(three_y, 0, -264)
    else:
        three_x = np.append(three_x, -396)
        three_y = np.append(three_y, 264)
        three_x = np.insert(three_x, 0, -396)
        three_y = np.insert(three_y, 0, -264)

    fig.add_trace(
        go.Scatter(
            x=three_x,
            y=three_y,
            mode='lines',
            line=dict(color='black', width=2),
            name='3-Point Line'
        )
    )

    # Extend 3-point line to court end (x = 564 or -564) at y = ±264
    three_ext_start_x = 396 if not flip_court else -396
    three_ext_end_x = court_length if not flip_court else -court_length
    fig.add_trace(
        go.Scatter(
            x=[three_ext_start_x, three_ext_end_x],
            y=[-264, -264],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[three_ext_start_x, three_ext_end_x],
            y=[264, 264],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        )
    )

    # Shot location marker (use original coordinates without flipping sign)
    fig.add_trace(
        go.Scatter(
            x=[shot_x],
            y=[shot_y],
            mode='markers',
            marker=dict(size=15, color=marker_color, symbol=marker_symbol),
            name=marker_name
        )
    )

    # Update layout with fixed size and equal aspect ratio
    x_range = [-282, 564] if not flip_court else [-564, 282]
    fig.update_layout(
        title="Shot Location on Half Court",
        xaxis_title="X Position (inches)",
        yaxis_title="Y Position (inches)",
        xaxis=dict(range=x_range, showgrid=False),
        yaxis=dict(range=[-300, 300], showgrid=False),
        width=500,
        height=500,
        autosize=False,
        showlegend=True,
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Ensure equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def plot_joint_flexion_analysis(pose_df, ball_df, metrics, fps=60):
    """
    Create a side-by-side visualization of joint flexion/extension angles with Kalman smoothing
    and compute biomechanical KPIs for basketball shooting motion.
    
    Parameters:
    - pose_df: DataFrame with pose data (e.g., 'RSHOULDER_X', 'RELBOW_X', etc.) in inches.
    - ball_df: DataFrame with 'Basketball_X', 'Basketball_Y', 'Basketball_Z' in inches (unused here).
    - metrics: Dictionary with 'lift_idx', 'set_idx', 'release_idx', etc.
    - fps: Frames per second, default 60.
    
    Returns:
    - fig: Single Plotly figure object with two subplots.
    - kpis: Dictionary of biomechanical KPIs with nested joint metrics.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    import logging
    from filterpy.kalman import KalmanFilter

    logger = logging.getLogger(__name__)

    INCHES_TO_FEET = 1 / 12

    # Define a cohesive pastel palette
    COLOR_PALETTE = {
        'lift': 'rgba(147, 112, 219, 1)',        # Pastel purple
        'set': 'rgba(255, 182, 193, 1)',         # Pastel pink
        'release': 'rgba(255, 102, 102, 1)',     # Pastel red
        'elbow': 'rgba(173, 216, 230, 1)',       # Pastel light blue
        'shoulder': 'rgba(221, 160, 221, 1)',    # Pastel plum
        'wrist': 'rgba(255, 245, 157, 1)',       # Pastel yellow
        'hip': 'rgba(144, 238, 144, 1)',         # Pastel green
        'knee': 'rgba(255, 160, 122, 1)',        # Pastel coral
        'ankle': 'rgba(176, 196, 222, 1)'        # Pastel steel blue
    }
    DASH_STYLES = {
        'lift': 'dash',
        'set': 'dot',
        'release': 'dashdot'
    }

    # Extract key indices with defaults
    lift_idx = metrics.get('lift_idx', 0)
    set_idx = metrics.get('set_idx', 0)
    release_idx = metrics.get('release_idx', 0)

    # Extend time window: 0.25s before lift_idx, 0.5s after release_idx
    frames_before = int(0.25 * fps)
    frames_after = int(0.5 * fps)
    start_idx = max(0, lift_idx - frames_before)
    end_idx = min(len(pose_df) - 1, release_idx + frames_after)

    if start_idx >= end_idx or end_idx >= len(pose_df):
        logger.error(f"Invalid indices: start_idx={start_idx}, end_idx={end_idx}, len(pose_df)={len(pose_df)}")
        return go.Figure(), {}

    pose_segment = pose_df.iloc[start_idx:end_idx + 1].copy()
    pose_segment['time'] = (pose_segment.index - start_idx) / fps

    # Calculate joint angles
    def calculate_angle(df, a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z):
        try:
            ab = np.sqrt((df[a_x] - df[b_x])**2 + (df[a_y] - df[b_y])**2 + (df[a_z] - df[b_z])**2)
            bc = np.sqrt((df[c_x] - df[b_x])**2 + (df[c_y] - df[b_y])**2 + (df[c_z] - df[b_z])**2)
            ac = np.sqrt((df[a_x] - df[c_x])**2 + (df[a_y] - df[c_y])**2 + (df[a_z] - df[c_z])**2)
            cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.degrees(np.arccos(cos_angle))
        except KeyError as e:
            logger.warning(f"Missing key for angle calculation: {e}")
            return np.nan

    # Define joint angle calculations
    joint_angles = {
        'elbow': calculate_angle(pose_segment, 'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                                'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z',
                                'RWRIST_X', 'RWRIST_Y', 'RWRIST_Z'),
        'shoulder': calculate_angle(pose_segment, 'MIDHIP_X', 'MIDHIP_Y', 'MIDHIP_Z',
                                   'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                                   'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z'),
        'wrist': calculate_angle(pose_segment, 'RELBOW_X', 'RELBOW_Y', 'RELBOW_Z',
                                'RWRIST_X', 'RWRIST_Y', 'RWRIST_Z',
                                'RTHUMB_X', 'RTHUMB_Y', 'RTHUMB_Z'),
        'hip': calculate_angle(pose_segment, 'RSHOULDER_X', 'RSHOULDER_Y', 'RSHOULDER_Z',
                              'RHIP_X', 'RHIP_Y', 'RHIP_Z',
                              'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z'),
        'knee': calculate_angle(pose_segment, 'RHIP_X', 'RHIP_Y', 'RHIP_Z',
                               'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z',
                               'RANKLE_X', 'RANKLE_Y', 'RANKLE_Z'),
        'ankle': calculate_angle(pose_segment, 'RKNEE_X', 'RKNEE_Y', 'RKNEE_Z',
                                'RANKLE_X', 'RANKLE_Y', 'RANKLE_Z',
                                'RBIGTOE_X', 'RBIGTOE_Y', 'RBIGTOE_Z')
    }

    # Apply Kalman filter to smooth angles
    def apply_kalman_filter(data, process_noise=0.01, measurement_noise=1.0):
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([data.iloc[0] if not pd.isna(data.iloc[0]) else 0.0])  # Initial state
        kf.P = np.array([[1.0]])  # Initial covariance
        kf.F = np.array([[1.0]])  # State transition model (simple 1D)
        kf.H = np.array([[1.0]])  # Measurement model
        kf.R = np.array([[measurement_noise]])  # Measurement noise
        kf.Q = np.array([[process_noise]])  # Process noise

        smoothed = []
        for measurement in data:
            if pd.isna(measurement):
                smoothed.append(np.nan)
            else:
                kf.predict()
                kf.update(np.array([measurement]))
                smoothed.append(kf.x[0])
        return np.array(smoothed)

    # Smooth joint angles
    for joint, angles in joint_angles.items():
        smoothed_angles = apply_kalman_filter(angles, process_noise=0.01, measurement_noise=1.0)
        pose_segment[f'{joint}_angle'] = smoothed_angles

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Upper Body Joint Flexion/Extension", "Lower Body Joint Flexion/Extension"),
        horizontal_spacing=0.15
    )

    # Upper body traces
    for angle in ['elbow_angle', 'shoulder_angle', 'wrist_angle']:
        joint = angle.replace('_angle', '')
        fig.add_trace(
            go.Scatter(
                x=pose_segment['time'],
                y=pose_segment[angle],
                mode='lines',
                name=joint.capitalize(),
                line=dict(color=COLOR_PALETTE[joint], width=2.5)
            ),
            row=1, col=1
        )
    for event, idx in [('lift', lift_idx), ('set', set_idx), ('release', release_idx)]:
        fig.add_vline(
            x=(idx - start_idx) / fps,
            line=dict(color=COLOR_PALETTE[event], dash=DASH_STYLES[event], width=2),
            row=1, col=1
        )

    # Lower body traces
    for angle in ['hip_angle', 'knee_angle', 'ankle_angle']:
        joint = angle.replace('_angle', '')
        fig.add_trace(
            go.Scatter(
                x=pose_segment['time'],
                y=pose_segment[angle],
                mode='lines',
                name=joint.capitalize(),
                line=dict(color=COLOR_PALETTE[joint], width=2.5)
            ),
            row=1, col=2
        )
    for event, idx in [('lift', lift_idx), ('set', set_idx), ('release', release_idx)]:
        fig.add_vline(
            x=(idx - start_idx) / fps,
            line=dict(color=COLOR_PALETTE[event], dash=DASH_STYLES[event], width=2),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title="Joint Flexion/Extension Analysis (Kalman Smoothed)",
        title_x=0.5,
        height=600,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(l=80, r=80, t=150, b=200),
        plot_bgcolor='rgba(245, 245, 245, 1)'
    )
    fig.update_xaxes(title_text="Time (s)", title_font_size=14, tickfont_size=12, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", title_font_size=14, tickfont_size=12, row=1, col=2)
    fig.update_yaxes(title_text="Angle (degrees)", range=[0, 180], title_font_size=14, tickfont_size=12, row=1, col=1)
    fig.update_yaxes(title_text="Angle (degrees)", range=[0, 180], title_font_size=14, tickfont_size=12, row=1, col=2)

    # Calculate KPIs using smoothed angles
    kpis = {}
    joints = ['elbow', 'shoulder', 'wrist', 'hip', 'knee', 'ankle']

    for joint in joints:
        angles = pose_segment[f'{joint}_angle']
        if angles.isna().all():
            kpis[joint] = {
                'max_flexion': np.nan,
                'min_flexion': np.nan,
                'at_lift': np.nan,
                'at_set': np.nan,
                'at_release': np.nan,
                'range': np.nan,
                'rate_change': np.nan
            }
        else:
            rate = angles.diff() / pose_segment['time'].diff()  # Rate of change in degrees/sec
            kpis[joint] = {
                'max_flexion': angles.max(),
                'min_flexion': angles.min(),
                'at_lift': angles.iloc[lift_idx - start_idx] if 0 <= lift_idx - start_idx < len(angles) else np.nan,
                'at_set': angles.iloc[set_idx - start_idx] if 0 <= set_idx - start_idx < len(angles) else np.nan,
                'at_release': angles.iloc[release_idx - start_idx] if 0 <= release_idx - start_idx < len(angles) else np.nan,
                'range': angles.max() - angles.min(),
                'rate_change': rate.max() if not rate.isna().all() else np.nan
            }

    # Add shoulder rotation
    if not pose_segment['shoulder_angle'].isna().all():
        shoulder_lift = pose_segment['shoulder_angle'].iloc[lift_idx - start_idx] if 0 <= lift_idx - start_idx < len(pose_segment) else np.nan
        shoulder_release = pose_segment['shoulder_angle'].iloc[release_idx - start_idx] if 0 <= release_idx - start_idx < len(pose_segment) else np.nan
        kpis['shoulder']['rotation'] = abs(shoulder_release - shoulder_lift) if not pd.isna(shoulder_lift) and not pd.isna(shoulder_release) else np.nan

    # Kinematic Chain Score
    def calculate_kinematic_chain_score(pose_segment, lift_idx, set_idx, release_idx, start_idx, end_idx, fps):
        sequence_order = ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist']
        score = 0
        max_score = 100

        # Fixed typo: Use dict() instead of Dictionary()
        peak_rate_times = dict()
        for joint in sequence_order:
            angles = pose_segment[f'{joint}_angle']
            if angles.isna().all():
                peak_rate_times[joint] = 0
                continue
            rate = angles.diff() / pose_segment['time'].diff()
            peak_rate_idx = rate.idxmax() if not rate.isna().all() else start_idx
            peak_rate_times[joint] = (peak_rate_idx - start_idx) / fps

        timing_score = 0
        for i in range(len(sequence_order) - 1):
            t1 = peak_rate_times[sequence_order[i]]
            t2 = peak_rate_times[sequence_order[i + 1]]
            if t1 < t2 and t1 >= 0 and t2 <= (end_idx - start_idx) / fps:
                timing_score += (40 / (len(sequence_order) - 1))
        score += timing_score

        range_score = 0
        optimal_ranges = {'ankle': 50, 'knee': 70, 'hip': 60, 'shoulder': 90, 'elbow': 70, 'wrist': 30}
        for joint in sequence_order:
            angles = pose_segment[f'{joint}_angle']
            if angles.isna().all():
                continue
            range_val = angles.max() - angles.min()
            range_score += min(range_val / optimal_ranges[joint], 1) * (30 / len(sequence_order))
        score += range_score

        release_angles = {
            joint: pose_segment[f'{joint}_angle'].iloc[release_idx - start_idx]
            for joint in ['shoulder', 'elbow', 'wrist'] if not pose_segment[f'{joint}_angle'].isna().all()
        }
        release_score = 0
        if len(release_angles) == 3:
            if release_angles['elbow'] > 120 and release_angles['shoulder'] > 110 and 130 <= release_angles['wrist'] <= 150:
                release_score = 30
            elif release_angles['elbow'] > 100 and release_angles['shoulder'] > 90:
                release_score = 15
        score += release_score

        return min(score, max_score)

    kpis['kinematic_chain_score'] = calculate_kinematic_chain_score(pose_segment, lift_idx, set_idx, release_idx, start_idx, end_idx, fps)

    # Center of Mass (COM) Movement
    if all(col in pose_segment for col in ['MIDHIP_X', 'MIDHIP_Y', 'MIDHIP_Z']):
        com_x = pose_segment['MIDHIP_X']
        com_y = pose_segment['MIDHIP_Y']
        com_z = pose_segment['MIDHIP_Z']
        com_speed = np.sqrt((com_x.diff()**2 + com_y.diff()**2 + com_z.diff()**2)) * fps * INCHES_TO_FEET
        com_direction = np.arctan2(com_y.diff(), com_x.diff()) * 180 / np.pi
        kpis['com'] = {
            'speed': com_speed.iloc[set_idx - start_idx:release_idx - start_idx].mean() if not com_speed.isna().all() else np.nan,
            'direction': com_direction.iloc[set_idx - start_idx:release_idx - start_idx].mean() if not com_direction.isna().all() else np.nan
        }
    else:
        kpis['com'] = {'speed': np.nan, 'direction': np.nan}

    # Stability (Feet-to-Hip Ratio)
    if all(col in pose_segment for col in ['RBIGTOE_X', 'LBIGTOE_X', 'RHIP_X', 'LHIP_X']):
        feet_width = abs(pose_segment['RBIGTOE_X'] - pose_segment['LBIGTOE_X']).mean()
        hip_width = abs(pose_segment['RHIP_X'] - pose_segment['LHIP_X']).mean()
        kpis['stability_ratio'] = feet_width / hip_width if hip_width > 0 else np.nan
    else:
        kpis['stability_ratio'] = np.nan

    logger.debug(f"KPIs computed: {kpis}")
    return fig, kpis