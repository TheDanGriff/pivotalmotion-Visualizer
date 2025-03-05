import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
import streamlit as st
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os
import logging

# ==========================================
# 0. Initialize Logger (Optional)
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Team Name Mapping Dictionary
# ==========================================
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

# ==========================================
# 2. Column Mapping
# ==========================================

def map_body_columns(df):
    """
    Remap raw columns to standardized names in a case-insensitive manner.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame with raw column names.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with standardized column names.
    """
    rename_dict = {
        # Eyes
        'leye_x': 'left_eye_x',
        'leye_y': 'left_eye_y',
        'leye_z': 'left_eye_z',
        'reye_x': 'right_eye_x',
        'reye_y': 'right_eye_y',
        'reye_z': 'right_eye_z',
        # Neck
        'neck_x': 'neck_x',
        'neck_y': 'neck_y',
        'neck_z': 'neck_z',
        # Shoulders
        'lsjc_x': 'left_shoulder_x',
        'lsjc_y': 'left_shoulder_y',
        'lsjc_z': 'left_shoulder_z',
        'rsjc_x': 'right_shoulder_x',
        'rsjc_y': 'right_shoulder_y',
        'rsjc_z': 'right_shoulder_z',
        # Elbows
        'lejc_x': 'left_elbow_x',
        'lejc_y': 'left_elbow_y',
        'lejc_z': 'left_elbow_z',
        'rejc_x': 'right_elbow_x',
        'rejc_y': 'right_elbow_y',
        'rejc_z': 'right_elbow_z',
        # Wrists
        'lwjc_x': 'left_wrist_x',
        'lwjc_y': 'left_wrist_y',
        'lwjc_z': 'left_wrist_z',
        'rwjc_x': 'right_wrist_x',
        'rwjc_y': 'right_wrist_y',
        'rwjc_z': 'right_wrist_z',
        # Pinks and Thumbs
        'lpinky_x': 'left_pinky_x',
        'lpinky_y': 'left_pinky_y',
        'lpinky_z': 'left_pinky_z',
        'lthumb_x': 'left_thumb_x',
        'lthumb_y': 'left_thumb_y',
        'lthumb_z': 'left_thumb_z',
        'rpinky_x': 'right_pinky_x',
        'rpinky_y': 'right_pinky_y',
        'rpinky_z': 'right_pinky_z',
        'rthumb_x': 'right_thumb_x',
        'rthumb_y': 'right_thumb_y',
        'rthumb_z': 'right_thumb_z',
        # Hips
        'midhip_x': 'mid_hip_x',
        'midhip_y': 'mid_hip_y',
        'midhip_z': 'mid_hip_z',
        'lhjc_x': 'left_hip_x',
        'lhjc_y': 'left_hip_y',
        'lhjc_z': 'left_hip_z',
        'rhjc_x': 'right_hip_x',
        'rhjc_y': 'right_hip_y',
        'rhjc_z': 'right_hip_z',
        # Knees
        'lkjc_x': 'left_knee_x',
        'lkjc_y': 'left_knee_y',
        'lkjc_z': 'left_knee_z',
        'rkjc_x': 'right_knee_x',
        'rkjc_y': 'right_knee_y',
        'rkjc_z': 'right_knee_z',
        # Ankles
        'lajc_x': 'left_ankle_x',
        'lajc_y': 'left_ankle_y',
        'lajc_z': 'left_ankle_z',
        'rajc_x': 'right_ankle_x',
        'rajc_y': 'right_ankle_y',
        'rajc_z': 'right_ankle_z',
        # Heels and Toes
        'lheel_x': 'left_heel_x',
        'lheel_y': 'left_heel_y',
        'lheel_z': 'left_heel_z',
        'lstoe_x': 'left_toe_x',
        'lstoe_y': 'left_toe_y',
        'lstoe_z': 'left_toe_z',
        'lbtoe_x': 'left_big_toe_x',
        'lbtoe_y': 'left_big_toe_y',
        'lbtoe_z': 'left_big_toe_z',
        'rheel_x': 'right_heel_x',
        'rheel_y': 'right_heel_y',
        'rheel_z': 'right_heel_z',
        'rstoe_x': 'right_toe_x',
        'rstoe_y': 'right_toe_y',
        'rstoe_z': 'right_toe_z',
        'rbtoe_x': 'right_big_toe_x',
        'rbtoe_y': 'right_big_toe_y',
        'rbtoe_z': 'right_big_toe_z',
        # Joint Angles
        'left_elbow_angle': 'left_elbow_angle',
        'right_elbow_angle': 'right_elbow_angle',
        'left_knee_angle': 'left_knee_angle',
        'right_knee_angle': 'right_knee_angle',
        'left_shoulder_angle': 'left_shoulder_angle',
        'right_shoulder_angle': 'right_shoulder_angle',
        'left_wrist_angle': 'left_wrist_angle',
        'right_wrist_angle': 'right_wrist_angle',
        'left_ankle_angle': 'left_ankle_angle',
        'right_ankle_angle': 'right_ankle_angle',
        'spine_angle': 'spine_angle',
        # Ball Position
        'basketball_x': 'basketball_x',
        'basketball_y': 'basketball_y',
        # Add more mappings as needed
    }

    # Convert both the DataFrame's columns and the rename_dict to lowercase for case-insensitive mapping
    lower_rename_dict = {k.lower(): v for k, v in rename_dict.items()}
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=lower_rename_dict)

    # Identify any columns that couldn't be mapped
    mapped_columns = set(df.columns)
    required_columns = set(rename_dict.values())
    missing_columns = required_columns - mapped_columns

    if missing_columns:
        logger.warning(f"Missing columns after mapping: {missing_columns}")

    return df

def load_image_from_file(filepath):
    """Loads and encodes image for display in Streamlit."""
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {filepath}")
        return None

def show_brand_header(player_name, team_name, brand_dark="#000000"):
    """
    Shows brand logo, plus the player's name and optional Team logo if available, centered.
    """
    col1, col2, col3 = st.columns([1, 3, 1])

    # Brand / Product Logo on the left
    brand_logo_path = os.path.join("images", "logo.PNG")  # Adjust the path as needed
    brand_logo = load_image_from_file(brand_logo_path)
    if brand_logo:
        with col1:
            st.image(brand_logo, use_column_width=True)
    else:
        with col1:
            st.markdown(f"<h1 style='text-align: center; color: {brand_dark};'>Pivotal Motion</h1>", unsafe_allow_html=True)

    # Player name in the middle
    with col2:
        st.markdown(
            f"<h1 style='text-align: center; color: {brand_dark};'>{player_name or 'Unknown Player'}</h1>",
            unsafe_allow_html=True
        )
        if team_name and team_name.strip().upper() != "N/A":
            cap_team = team_name.title()
            st.markdown(
                f"<h3 style='text-align: center; color: {brand_dark};'>Team: {cap_team}</h3>",
                unsafe_allow_html=True
            )

    # Team Logo on the right
    if team_name and team_name.strip().upper() != "N/A":
        # Construct the correct team logo path with capitalization
        team_key = team_name.lower().replace(" ", "")
        team_info = TEAMS.get(team_key, {})
        team_logo_filename = team_info.get('logo', '')
        team_logo_path = os.path.join("images", "teams", team_logo_filename)  # Adjust path as needed
        team_logo = load_image_from_file(team_logo_path)
        if team_logo:
            with col3:
                st.image(team_logo, use_column_width=True)
        else:
            with col3:
                st.write("")  # Leave blank if logo not found
    else:
        with col3:
            st.write("")  # Leave blank if team is N/A

###############################################################################
# Utility Functions: S3, DynamoDB, etc.
###############################################################################

def list_users(s3_client, bucket_name):
    """List unique user directories under 'Users/' prefix in S3."""
    prefix = 'Users/'
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        users = []
        for page in pages:
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    userdir = cp['Prefix'].replace(prefix, '').strip('/')
                    users.append(userdir)
        return users
    except Exception as e:
        st.error(f"Error listing users: {e}")
        return []

def list_user_jobs(s3_client, bucket_name, user_email):
    """List job IDs for a given user directory in S3."""
    prefix = f"Users/{user_email}/"
    delimiter = '/'
    jobs = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)
        for page in pages:
            if 'CommonPrefixes' in page:
                for cp in page['CommonPrefixes']:
                    job_id = cp['Prefix'].replace(prefix, '').strip('/')
                    jobs.append(job_id)
        return jobs
    except Exception as e:
        st.error(f"Error listing jobs for {user_email}: {e}")
        return []

def list_job_segments(s3_client, bucket_name, user_email, job_id):
    """Scans 'processed/{user_email}/{job_id}/' for segment CSV files."""
    prefix = f"processed/{user_email}/{job_id}/"
    segs = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                if filename.startswith("segment_") and ("_vibe_output.csv" in filename or "_final_output.csv" in filename):
                    seg_id = filename.split("_vibe_output.csv")[0].split("_final_output.csv")[0].replace("segment_", "")
                    segs.add(seg_id)
        return sorted(segs)
    except Exception as e:
        st.error(f"Error listing segments for job '{job_id}': {e}")
        return []

def load_vibe_output(s3_client, bucket_name, user_email, job_id, shot_id):
    """
    Loads either *vibe_output.csv or *final_output.csv from S3.
    Filters for class_name 'basketball' and track_id == 4.
    """
    prefix = f"processed/{user_email}/{job_id}/"
    possible_keys = [
        f"{prefix}segment_{shot_id}_vibe_output.csv",
        f"{prefix}segment_{shot_id}_final_output.csv"
    ]
    for key in possible_keys:
        try:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            body = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(body))
            df = map_body_columns(df)
            # Filter for basketball (track_id == 4) and class_name == 'basketball'
            df_basketball = df[(df['track_id'] == 4) & (df['class_name'] == 'basketball')].copy()
            # Calculate center of bbox as ball position
            df_basketball['basketball_x'] = (df_basketball['xmin'] + df_basketball['xmax']) / 2
            df_basketball['basketball_y'] = (df_basketball['ymin'] + df_basketball['ymax']) / 2
            return df_basketball
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception as e:
            st.error(f"Error loading CSV for shot '{shot_id}': {e}")
            return pd.DataFrame()
    st.warning(f"No CSV found for shot '{shot_id}' in job '{job_id}'.")
    return pd.DataFrame()

def get_metadata_from_dynamodb(dynamodb, job_id):
    """Retrieves job metadata from DynamoDB if available."""
    table_name = st.secrets.get("DYNAMODB_TABLE_NAME", "")
    if not table_name:
        st.error("DynamoDB table name missing.")
        st.stop()

    table = dynamodb.Table(table_name)
    try:
        resp = table.get_item(Key={'JobID': job_id})
        return resp.get('Item', {})
    except ClientError as e:
        st.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error retrieving metadata: {e}")
        return {}

def humanize_label(label):
    """Convert 'snake_case' into Title-case, no underscores."""
    if not label or pd.isna(label):
        return "N/A"
    return ' '.join(word.capitalize() for word in label.replace('_',' ').split())

def get_unique_frames(df):
    """Returns a DataFrame with unique frames."""
    if 'frame' not in df.columns:
        st.error("No 'frame' column found in data.")
        return df
    df_unique = df.drop_duplicates(subset=['frame']).reset_index(drop=True)
    return df_unique

def smooth_data(df, columns, window_length=11, polyorder=3):
    """
    Applies Savitzky-Golay filter to smooth specified columns.
    """
    for col in columns:
        if col in df.columns:
            try:
                # Ensure window_length is less than or equal to the size of the data and odd
                if window_length > len(df[col].dropna()):
                    window_length = len(df[col].dropna()) if len(df[col].dropna()) % 2 != 0 else len(df[col].dropna()) - 1
                if window_length < 5:
                    window_length = 5  # Minimum window length
                if window_length % 2 == 0:
                    window_length += 1  # Make it odd
                df[f"{col}_smooth"] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
            except Exception as e:
                st.warning(f"Could not smooth column '{col}': {e}")
                df[f"{col}_smooth"] = df[col]
        else:
            st.warning(f"Column '{col}' not found for smoothing.")
            df[f"{col}_smooth"] = np.nan
    return df

def remove_outliers(df, columns, z_thresh=3):
    for col in columns:
        if col in df.columns:
            # Compute z-scores with nan_policy='omit' to handle NaNs
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            threshold = z_thresh
            mask = np.abs(stats.zscore(df[col], nan_policy='omit')) > threshold
            df.loc[mask, col] = np.nan
        else:
            st.warning(f"Column '{col}' not found for outlier removal.")
    return df

def compute_velocity(df, column, fps=25.0):
    """Computes the velocity of a given column."""
    if column in df.columns:
        return df[column].diff() * fps
    else:
        return pd.Series([np.nan]*len(df))

def compute_acceleration(df, column, fps=25.0):
    """Computes the acceleration of a given column."""
    if column in df.columns:
        return df[column].diff().diff() * (fps ** 2)
    else:
        return pd.Series([np.nan]*len(df))

def compute_velocity_acceleration(df, columns, fps=25.0):
    """Computes velocity and acceleration for specified columns."""
    for col in columns:
        vel_col = f"{col}_vel"
        accel_col = f"{col}_accel"
        df[vel_col] = compute_velocity(df, col, fps)
        df[accel_col] = compute_acceleration(df, col, fps)
    return df

def compute_joint_angles(df):
    """
    Compute joint angles based on joint positions.
    For example, compute elbow angles using shoulder, elbow, and wrist positions.
    """
    def angle_between(p1, p2, p3):
        """
        Calculate the angle at p2 given three points p1, p2, p3.
        """
        a = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
        b = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return np.nan
        cosine_angle = np.dot(a, b) / (norm_a * norm_b)
        # Clamp the cosine_angle to the valid range [-1, 1] to avoid NaNs due to floating point errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    # Example: Compute left and right elbow angles
    df['left_elbow_angle'] = df.apply(lambda row: angle_between(
        (row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z']),
        (row['left_elbow_x'], row['left_elbow_y'], row['left_elbow_z']),
        (row['left_wrist_x'], row['left_wrist_y'], row['left_wrist_z'])
    ) if not any(pd.isna([row['left_shoulder_x'], row['left_shoulder_y'], row['left_shoulder_z'],
                          row['left_elbow_x'], row['left_elbow_y'], row['left_elbow_z'],
                          row['left_wrist_x'], row['left_wrist_y'], row['left_wrist_z']])) else np.nan, axis=1)

    df['right_elbow_angle'] = df.apply(lambda row: angle_between(
        (row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z']),
        (row['right_elbow_x'], row['right_elbow_y'], row['right_elbow_z']),
        (row['right_wrist_x'], row['right_wrist_y'], row['right_wrist_z'])
    ) if not any(pd.isna([row['right_shoulder_x'], row['right_shoulder_y'], row['right_shoulder_z'],
                           row['right_elbow_x'], row['right_elbow_y'], row['right_elbow_z'],
                           row['right_wrist_x'], row['right_wrist_y'], row['right_wrist_z']])) else np.nan, axis=1)

    # Compute other joint angles similarly...
    # Example: Left Knee Angle
    df['left_knee_angle'] = df.apply(lambda row: angle_between(
        (row['left_hip_x'], row['left_hip_y'], row['left_hip_z']),
        (row['left_knee_x'], row['left_knee_y'], row['left_knee_z']),
        (row['left_ankle_x'], row['left_ankle_y'], row['left_ankle_z'])
    ) if not any(pd.isna([row['left_hip_x'], row['left_hip_y'], row['left_hip_z'],
                          row['left_knee_x'], row['left_knee_y'], row['left_knee_z'],
                          row['left_ankle_x'], row['left_ankle_y'], row['left_ankle_z']])) else np.nan, axis=1)

    df['right_knee_angle'] = df.apply(lambda row: angle_between(
        (row['right_hip_x'], row['right_hip_y'], row['right_hip_z']),
        (row['right_knee_x'], row['right_knee_y'], row['right_knee_z']),
        (row['right_ankle_x'], row['right_ankle_y'], row['right_ankle_z'])
    ) if not any(pd.isna([row['right_hip_x'], row['right_hip_y'], row['right_hip_z'],
                           row['right_knee_x'], row['right_knee_y'], row['right_knee_z'],
                           row['right_ankle_x'], row['right_ankle_y'], row['right_ankle_z']])) else np.nan, axis=1)

    # Add more joint angles as needed...

    return df

def compute_kpis(df, basketball_df):
    """
    Computes Key Performance Indicators (KPIs) from the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The main DataFrame containing pose data.
    basketball_df : pandas.DataFrame
        The DataFrame containing basketball tracking data.

    Returns:
    --------
    kpis : dict
        A dictionary of computed KPIs.
    """
    kpis = {}

    # Release Velocity: Maximum velocity of the ball
    if 'basketball_x_vel' in basketball_df.columns and 'basketball_y_vel' in basketball_df.columns:
        basketball_df['ball_speed'] = np.sqrt(basketball_df['basketball_x_vel']**2 + basketball_df['basketball_y_vel']**2)
        max_velocity = basketball_df['ball_speed'].max()
        kpis['Release Velocity (m/s)'] = max_velocity
        logger.info(f"Max Velocity: {max_velocity} m/s")
    else:
        kpis['Release Velocity (m/s)'] = np.nan
        logger.warning("Velocity columns 'basketball_x_vel' or 'basketball_y_vel' not found.")

    # Release Height: Height of the ball at maximum velocity
    if 'ball_speed' in basketball_df.columns and not basketball_df['ball_speed'].isna().all():
        idx_max_vel = basketball_df['ball_speed'].idxmax()
        # Ensure 'basketball_y_smooth' exists
        if 'basketball_y_smooth' in basketball_df.columns:
            release_height = basketball_df.loc[idx_max_vel, 'basketball_y_smooth']
            kpis['Release Height (m)'] = release_height
            logger.info(f"Release Height at Max Velocity: {release_height} m")
        else:
            kpis['Release Height (m)'] = np.nan
            logger.warning("'basketball_y_smooth' column not found.")
    else:
        kpis['Release Height (m)'] = np.nan
        logger.warning("'ball_speed' column not found or all values are NaN.")

    # Release Angle: Angle of the ball relative to the floor shortly after release
    if ('basketball_x_vel' in basketball_df.columns and 
        'basketball_y_vel' in basketball_df.columns and
        'time_pose' in basketball_df.columns and
        'basketball_y_smooth' in basketball_df.columns):
        
        if 'idx_max_vel' in locals():
            post_max_vel = basketball_df.loc[idx_max_vel:idx_max_vel+4]  # Next 5 frames including max
            if len(post_max_vel) > 1:
                dx = post_max_vel['basketball_x_vel'].iloc[1] - post_max_vel['basketball_x_vel'].iloc[0]
                dy = post_max_vel['basketball_y_vel'].iloc[1] - post_max_vel['basketball_y_vel'].iloc[0]
                release_angle = math.degrees(math.atan2(dy, dx)) if dx != 0 else 90.0
                kpis['Release Angle (degrees)'] = release_angle
                logger.info(f"Release Angle: {release_angle} degrees")
            else:
                kpis['Release Angle (degrees)'] = np.nan
                logger.warning("Not enough data points after max velocity for Release Angle.")
        else:
            kpis['Release Angle (degrees)'] = np.nan
            logger.warning("'idx_max_vel' not defined. Cannot compute Release Angle.")
    else:
        kpis['Release Angle (degrees)'] = np.nan
        logger.warning("Necessary columns for Release Angle not found.")

    # Shot Distance: Distance between maximum velocity point and middle of the rim
    # Assuming rim is at (0, 3.048) meters (10 feet), adjust as necessary
    rim_x, rim_y = 0.0, 3.048  # Example coordinates
    if 'basketball_x_smooth' in basketball_df.columns and 'basketball_y_smooth' in basketball_df.columns and 'ball_speed' in basketball_df.columns:
        if not basketball_df['ball_speed'].isna().all():
            shot_x = basketball_df.loc[idx_max_vel, 'basketball_x_smooth']
            shot_y = basketball_df.loc[idx_max_vel, 'basketball_y_smooth']
            shot_distance = math.sqrt((shot_x - rim_x)**2 + (shot_y - rim_y)**2)
            kpis['Shot Distance (m)'] = shot_distance
            logger.info(f"Shot Distance: {shot_distance} m")
        else:
            kpis['Shot Distance (m)'] = np.nan
            logger.warning("'ball_speed' is all NaN. Cannot compute Shot Distance.")
    else:
        kpis['Shot Distance (m)'] = np.nan
        logger.warning("Necessary columns for Shot Distance not found.")

    # Apex Height: Height of the ball at its highest point in the shot (the arc)
    if 'basketball_y_smooth' in basketball_df.columns:
        apex_height = basketball_df['basketball_y_smooth'].max()
        kpis['Apex Height (m)'] = apex_height
        logger.info(f"Apex Height: {apex_height} m")
    else:
        kpis['Apex Height (m)'] = np.nan
        logger.warning("'basketball_y_smooth' column not found.")

    # Release Time: Time from “start of shot” up until Maximum velocity
    if 'time_pose' in basketball_df.columns and not basketball_df['time_pose'].isna().all():
        start_time = basketball_df['time_pose'].min()
        release_time = basketball_df.loc[idx_max_vel, 'time_pose'] - start_time
        kpis['Release Time (s)'] = release_time
        logger.info(f"Release Time: {release_time} s")
    else:
        kpis['Release Time (s)'] = np.nan
        logger.warning("'time_pose' column not found or all values are NaN.")

    # Release Quality: Average X of the ball spin displayed as a percentage
    # Assuming 'basketball_x_vel' represents spin; adjust based on actual data
    if 'basketball_x_vel' in basketball_df.columns:
        release_quality = basketball_df['basketball_x_vel'].mean()
        kpis['Release Quality (%)'] = release_quality * 100  # Scale as percentage
        logger.info(f"Release Quality: {release_quality * 100} %")
    else:
        kpis['Release Quality (%)'] = np.nan
        logger.warning("'basketball_x_vel' column not found for Release Quality.")

    # Release Consistency: Standard Deviation of the Release Qualities when comparing multiple shots
    # This KPI is better computed across shots, handled separately in the main app

    # Release Curvature: Amount of curvature in the path of the ball in the release phase
    # Placeholder: Implement actual curvature calculation based on trajectory
    kpis['Release Curvature'] = np.nan  # To be implemented
    logger.warning("Release Curvature calculation not implemented.")

    # Lateral Release Curvature: Similar idea but from a different angle
    # Placeholder: Implement actual lateral curvature calculation
    kpis['Lateral Release Curvature'] = np.nan  # To be implemented
    logger.warning("Lateral Release Curvature calculation not implemented.")

    return kpis

def visualize_joint_data(df, body_part, data_type, selected_metrics, key_suffix=""):
    """
    Visualizes joint data based on selected body part and data type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to visualize.
    body_part : str
        The body part category selected (Upper Body, Lower Body, Full Body).
    data_type : str
        The type of data to visualize (Angle, Velocity, Acceleration).
    selected_metrics : list
        List of specific metrics to visualize.
    key_suffix : str
        A unique suffix for Streamlit keys to avoid conflicts.
    """
    # Define joints based on body part
    body_parts = {
        "Upper Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev'],
        "Lower Body": ['left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle'],
        "Full Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev',
                      'left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle']
    }
    
    available_joints = body_parts.get(body_part, [])
    if not available_joints:
        st.info("Please select a valid body part.")
        return

    # Filter selected metrics
    metrics_to_plot = [metric for metric in selected_metrics if metric in available_joints]
    if not metrics_to_plot:
        st.warning("No valid metrics selected for the chosen body part.")
        return

    # Determine suffix based on data type
    suffix = ''
    if data_type == 'Velocity':
        suffix = '_vel'
    elif data_type == 'Acceleration':
        suffix = '_accel'
    elif data_type == 'Angle':
        suffix = ''  # Angles are already present without suffix
    else:
        suffix = '_smooth'  # Default to smoothed data

    # Adjust metric names based on data type
    if data_type != 'Angle':
        metrics_to_plot = [f"{metric}{suffix}" for metric in metrics_to_plot]
    
    # Check if smooth/velocity/acceleration columns exist
    available_metrics = [metric for metric in metrics_to_plot if metric in df.columns]
    if not available_metrics:
        st.warning(f"No data available for {data_type} in the selected body part.")
        return

    # Determine x-axis
    if 'time_pose' in df.columns:
        x_axis = 'time_pose'
    elif 'time_ball' in df.columns:
        x_axis = 'time_ball'
    elif 'frame' in df.columns:
        # Create a time column if not present
        fps = 25.0  # Assuming 25 fps
        if 'time_pose' not in df.columns:
            df['time_pose'] = df['frame'] / fps
        x_axis = 'time_pose'
    else:
        x_axis = 'frame'

    # Create Plotly figure
    fig = go.Figure()
    for metric in available_metrics:
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=df[metric],
            mode='lines',
            name=humanize_label(metric.replace('_', ' ').title()),
            line=dict(width=2)
        ))

    fig.update_layout(
        title=f"{data_type} Over Time for {body_part}",
        xaxis_title="Time (s)" if x_axis == 'time_pose' else "Frame",
        yaxis_title=f"{data_type}" + (" (degrees)" if data_type == 'Angle' else f" (units)"),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{body_part}_{data_type}_{key_suffix}")

def visualize_kpis(metrics):
    """
    Displays the computed KPIs.
    """
    st.markdown("### 🎯 **Key Performance Indicators (KPIs)**")
    for key, value in metrics.items():
        if not pd.isna(value):
            st.write(f"**{key}**: {value:.2f}")
        else:
            st.write(f"**{key}**: N/A")

def visualize_ball_trajectory_quadratic(df, key_suffix=""):
    """
    Visualizes the ball's X position with a quadratic fit and marks the point of maximum increase.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['time_pose'], df['basketball_x_smooth'], 2)
        poly_func = np.poly1d(coefficients)
        fitted_x = poly_func(df['time_pose'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_x = df['basketball_x_smooth']
        fitted_x[:] = np.nan  # Avoid plotting if fit fails

    # Calculate delta_x to find the biggest increase
    df['delta_x'] = df['basketball_x_smooth'].diff().fillna(0)
    max_increase_idx = df['delta_x'].idxmax()
    if not pd.isna(max_increase_idx):
        max_increase_time = df.loc[max_increase_idx, 'time_pose'] if 'time_pose' in df.columns else df.loc[max_increase_idx, 'frame']
        max_increase_x = df.loc[max_increase_idx, 'basketball_x_smooth']
    else:
        max_increase_time = None
        max_increase_x = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=df['basketball_x_smooth'],
        mode='markers',
        name='Original X Position',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=fitted_x,
        mode='lines',
        name='Quadratic Fit',
        line=dict(color='orange')
    ))

    # Highlight the point with the greatest increase in X
    if max_increase_time is not None and max_increase_x is not None:
        fig.add_trace(go.Scatter(
            x=[max_increase_time],
            y=[max_increase_x],
            mode='markers',
            name='Max X Increase',
            marker=dict(color='red', size=12, symbol='star')
        ))
        fig.add_annotation(
            x=max_increase_time,
            y=max_increase_x,
            text=f"Max X Increase: {max_increase_time:.2f}s" if isinstance(max_increase_time, (int, float)) else f"Max X Increase: Frame {max_increase_idx}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )

    fig.update_layout(
        title="Ball X Position with Quadratic Fit",
        xaxis_title="Time (s)" if 'time_pose' in df.columns else "Frame",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_quadratic_{key_suffix}")

def visualize_ball_trajectory(df, key_suffix=""):
    """
    Visualizes the ball's 2D trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['basketball_x_smooth'], df['basketball_y_smooth'], 2)
        poly_func = np.poly1d(coefficients)
        fitted_y = poly_func(df['basketball_x_smooth'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_y = df['basketball_y_smooth']
        fitted_y[:] = np.nan  # Avoid plotting if fit fails

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['basketball_x_smooth'],
        y=df['basketball_y_smooth'],
        mode='markers',
        name='Original Trajectory',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['basketball_x_smooth'],
        y=fitted_y,
        mode='lines',
        name='Quadratic Fit',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Ball Trajectory with Quadratic Fit",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_{key_suffix}")

def visualize_ball_heatmap(df, key_suffix=""):
    """
    Creates a heatmap of the ball's 2D positions to show frequency of locations.
    """
    if df.empty:
        st.warning("No data to plot for ball heatmap.")
        return

    if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(df.columns):
        fig = px.density_heatmap(
            df,
            x='basketball_x_smooth',
            y='basketball_y_smooth',
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale='Blues',
            title="Ball Location Heatmap"
        )
        fig.update_layout(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ball_heatmap_{key_suffix}")
    else:
        st.warning("Missing 'basketball_x_smooth' or 'basketball_y_smooth' for ball heatmap.")

def detect_release_point(df, joints=['left_wrist_y_smooth', 'right_wrist_y_smooth', 'left_elbow_y_smooth', 'right_elbow_y_smooth']):
    """
    Detects the release point as the highest position among specified joints.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing joint smooth data.
    joints : list
        List of joint Y positions to consider for release point.
    
    Returns:
    --------
    release_time : float
        The time in seconds when release occurs.
    release_point : tuple
        The (x, y) coordinates of the release point.
    """
    # Find the maximum Y position among the specified joints
    max_positions = df[joints].max(axis=1)
    idx_release = max_positions.idxmax()
    
    if 'time_pose' in df.columns:
        release_time = df.loc[idx_release, 'time_pose']
    else:
        release_time = df.loc[idx_release, 'frame'] / 25.0  # Assuming 25 fps
    
    # Get ball position at release
    if 'basketball_x_smooth' in df.columns and 'basketball_y_smooth' in df.columns:
        release_x = df.loc[idx_release, 'basketball_x_smooth']
        release_y = df.loc[idx_release, 'basketball_y_smooth']
        release_point = (release_x, release_y)
    else:
        release_point = (np.nan, np.nan)
    
    return release_time, release_point

def visualize_release_point(df, release_time, release_point, key_suffix=""):
    """
    Marks the release point on the ball trajectory plot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing ball tracking data.
    release_time : float
        The time in seconds when release occurs.
    release_point : tuple
        The (x, y) coordinates of the release point.
    key_suffix : str
        A unique suffix for Streamlit keys to avoid conflicts.
    """
    if pd.isna(release_point[0]) or pd.isna(release_point[1]):
        st.warning("Release point data is incomplete.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['basketball_x_smooth'],
        y=df['basketball_y_smooth'],
        mode='markers',
        name='Ball Trajectory',
        marker=dict(color='blue', size=6)
    ))

    # Mark the release point
    fig.add_trace(go.Scatter(
        x=[release_point[0]],
        y=[release_point[1]],
        mode='markers',
        name='Release Point',
        marker=dict(color='red', size=12, symbol='star')
    ))

    fig.update_layout(
        title="Ball Trajectory with Release Point",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"release_point_{key_suffix}")

###############################################################################
# Main Streamlit App Function
###############################################################################
def main():
    st.title("🏀 **Pivotal Motion Data Visualizer**")
    st.write("### Analyze and Visualize Shooting Motion Data")

    # Initialize AWS clients
    try:
        s3_client = boto3.client('s3')
        dynamodb = boto3.resource('dynamodb')
    except NoCredentialsError:
        st.error("AWS credentials not found.")
        return
    except PartialCredentialsError:
        st.error("Incomplete AWS credentials.")
        return
    except Exception as e:
        st.error(f"Error initializing AWS clients: {e}")
        return

    # Retrieve bucket name from secrets
    bucket_name = st.secrets.get("BUCKET_NAME", "")
    if not bucket_name:
        st.error("S3 bucket name not provided in secrets.")
        return

    st.sidebar.header("User & Job Selection")

    # 1) User Selection
    users = list_users(s3_client, bucket_name)
    if not users:
        st.sidebar.error("No users found.")
        return
    chosen_user = st.sidebar.selectbox("Select User (Email)", users)

    # 2) Job Selection
    jobs = list_user_jobs(s3_client, bucket_name, chosen_user)
    valid_jobs = []
    job_info = {}
    for j in jobs:
        segs = list_job_segments(s3_client, bucket_name, chosen_user, j)
        if segs:
            meta = get_metadata_from_dynamodb(dynamodb, j)
            pname = meta.get('PlayerName', 'Unknown')
            tname = meta.get('Team', '').title() if meta.get('Team', '').strip().upper() != "N/A" else ''
            ut = meta.get('UploadTimestamp', None)
            dt_s = "Unknown"
            if ut:
                try:
                    dt_s = pd.to_datetime(int(ut), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

            if tname:
                label = f"{pname} - {tname} - {dt_s} - {j}"
            else:
                label = f"{pname} - {dt_s} - {j}"

            job_info[j] = label
            valid_jobs.append(j)

    if not valid_jobs:
        st.sidebar.error("No jobs with shots found for this user.")
        return

    chosen_job = st.sidebar.selectbox("Select Job", valid_jobs, format_func=lambda x: job_info[x])

    # 3) Shot Selection
    shots = list_job_segments(s3_client, bucket_name, chosen_user, chosen_job)
    if not shots:
        st.sidebar.error("No shots found in processed/... path for this job.")
        return
    multi_shot = st.sidebar.checkbox("Compare Multiple Shots?")
    if multi_shot:
        chosen_shots = st.sidebar.multiselect("Select Shots", shots)
    else:
        sel_shot = st.sidebar.selectbox("Select a Shot", shots)
        chosen_shots = [sel_shot]

    if not chosen_shots:
        st.warning("No shots selected.")
        return

    # 4) Load Data
    job_meta = get_metadata_from_dynamodb(dynamodb, chosen_job)
    player_name = job_meta.get('PlayerName', 'Unknown Player')
    team_name = job_meta.get('Team', 'N/A')
    user_data = {}

    for shot in chosen_shots:
        df_basketball = load_vibe_output(s3_client, bucket_name, chosen_user, chosen_job, shot)
        if df_basketball.empty:
            st.warning(f"No basketball tracking data for shot '{shot}'.")
            continue

        # Load pose data for the shot
        df_shot = load_vibe_output(s3_client, bucket_name, chosen_user, chosen_job, shot)
        if df_shot.empty:
            st.warning(f"No pose data for shot '{shot}'.")
            continue

        # Merge job metadata
        for k, v in job_meta.items():
            df_basketball[k.lower()] = v

        # Process frame data
        if 'frame' not in df_basketball.columns:
            st.warning(f"No 'frame' column found in shot '{shot}'. Skipping.")
            continue

        # Assign time_pose based on frame number and FPS
        FPS = 25.0
        df_basketball['time_pose'] = df_basketball['frame'] / FPS

        # Smooth the ball trajectory data
        df_basketball = smooth_data(df_basketball, ['basketball_x', 'basketball_y'], window_length=11, polyorder=3)

        # Compute velocity and acceleration for the ball
        df_basketball = compute_velocity_acceleration(df_basketball, ['basketball_x', 'basketball_y'], fps=FPS)

        # Remove outliers from ball data
        df_basketball = remove_outliers(df_basketball, ['basketball_x', 'basketball_y'], z_thresh=3)

        # Detect release point
        release_time, release_point = detect_release_point(df_basketball, joints=['left_wrist_y_smooth', 'right_wrist_y_smooth', 'left_elbow_y_smooth', 'right_elbow_y_smooth'])

        # Compute KPIs
        kpis = compute_kpis(df_basketball, df_basketball)

        # Store processed data and KPIs
        user_data[shot] = {
            'df': df_basketball,
            'kpis': kpis,
            'release_time': release_time,
            'release_point': release_point
        }

    if not user_data:
        st.warning("No valid shot data loaded.")
        return

    # Combine data for overview
    combined_df = pd.DataFrame()
    for shot, data in user_data.items():
        d = data['df'].copy()
        d['shot_id'] = shot
        combined_df = pd.concat([combined_df, d], ignore_index=True)

    # Create Tabs
    t_overview, t_pose, t_adv, t_ball = st.tabs([
        "Overview", 
        "Pose Analysis", 
        "Advanced Biomechanics", 
        "Ball Tracking"
    ])

    # -------------- OVERVIEW TAB --------------
    with t_overview:
        st.markdown("## Overview")
        show_brand_header(
            player_name=player_name,
            team_name=team_name
        )

        # Display KPIs for all selected shots
        st.markdown("### **Key Performance Indicators (KPIs)**")
        kpi_list = []
        for shot, data in user_data.items():
            kpi = {'Shot': shot}
            kpi.update(data['kpis'])
            kpi_list.append(kpi)
        kpi_df = pd.DataFrame(kpi_list)
        if not kpi_df.empty:
            st.dataframe(kpi_df.set_index('Shot'))
        else:
            st.info("No KPIs to display.")

        # Dynamic Visualization with Metric Selection
        st.markdown("### **Joint Data Visualization**")
        body_part = st.selectbox("Select Body Part for Visualization", ["Upper Body", "Lower Body", "Full Body"], key="body_part_overview")
        data_type = st.selectbox("Select Data Type", ["Angle", "Velocity", "Acceleration"], key="data_type_overview")

        # Fetch available metrics based on body part and data type
        body_parts = {
            "Upper Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev'],
            "Lower Body": ['left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle'],
            "Full Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev',
                          'left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle']
        }
        available_joints = body_parts.get(body_part, [])
        if data_type == 'Angle':
            metrics = [metric for metric in available_joints if metric.endswith('_angle') or metric.endswith('_elev')]
        else:
            suffix = '_vel' if data_type == 'Velocity' else '_accel'
            metrics = [f"{metric}{suffix}" for metric in available_joints if f"{metric}{suffix}" in combined_df.columns]
        
        # Allow users to select specific metrics
        selected_metrics = st.multiselect("Select Metrics to Visualize", options=metrics, default=metrics)

        if selected_metrics:
            visualize_joint_data(combined_df, body_part, data_type, selected_metrics, key_suffix="overview")
        else:
            st.warning("No metrics selected for visualization.")

        # Visualize Ball Trajectory
        st.markdown("### **Ball Trajectory**")
        if {'basketball_x_smooth', 'basketball_y_smooth', 'time_pose'}.issubset(combined_df.columns):
            visualize_ball_trajectory_quadratic(
                df=combined_df[['time_pose','basketball_x_smooth','basketball_y_smooth']].copy(),
                key_suffix="overview"
            )
            visualize_ball_trajectory(
                df=combined_df[['basketball_x_smooth','basketball_y_smooth']].copy(),
                key_suffix="overview"
            )
            # Visualize Release Points
            for shot, data in user_data.items():
                visualize_release_point(
                    df=data['df'],
                    release_time=data['release_time'],
                    release_point=data['release_point'],
                    key_suffix=f"overview_{shot}"
                )
        else:
            st.warning("Missing 'basketball_x_smooth', 'basketball_y_smooth', or 'time_pose' for ball trajectory plots.")

        # Visualize Ball Heatmap
        st.markdown("### **Ball Location Heatmap**")
        if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(combined_df.columns):
            visualize_ball_heatmap(
                combined_df, 
                key_suffix="overview"
            )
        else:
            st.warning("Missing 'basketball_x_smooth' or 'basketball_y_smooth' for ball heatmap.")

    # -------------- POSE ANALYSIS TAB --------------
    with t_pose:
        st.markdown("## Pose Analysis")
        for shot, data in user_data.items():
            st.markdown(f"### Shot: {shot}")
            df_shot = data['df']
            kpis = data['kpis']

            # Display KPIs
            st.markdown("#### **Key Performance Indicators (KPIs)**")
            visualize_kpis(kpis)

            # Dynamic Visualization with Metric Selection
            st.markdown("#### **Joint Data Visualization**")
            body_part = st.selectbox(f"Select Body Part for Shot '{shot}'", ["Upper Body", "Lower Body", "Full Body"], key=f"body_part_pose_{shot}")
            data_type = st.selectbox(f"Select Data Type for Shot '{shot}'", ["Angle", "Velocity", "Acceleration"], key=f"data_type_pose_{shot}")

            # Fetch available metrics based on body part and data type
            body_parts = {
                "Upper Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev'],
                "Lower Body": ['left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle'],
                "Full Body": ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_elev', 'right_shoulder_elev',
                              'left_knee_angle', 'right_knee_angle', 'left_ankle_angle', 'right_ankle_angle']
            }
            available_joints = body_parts.get(body_part, [])
            if data_type == 'Angle':
                metrics = [metric for metric in available_joints if metric.endswith('_angle') or metric.endswith('_elev')]
            else:
                suffix = '_vel' if data_type == 'Velocity' else '_accel'
                metrics = [f"{metric}{suffix}" for metric in available_joints if f"{metric}{suffix}" in df_shot.columns]
            
            # Allow users to select specific metrics
            selected_metrics = st.multiselect("Select Metrics to Visualize", options=metrics, default=metrics)

            if selected_metrics:
                visualize_joint_data(df_shot, body_part, data_type, selected_metrics, key_suffix=f"pose_{shot}")
            else:
                st.warning("No metrics selected for visualization.")

    # -------------- ADVANCED BIOMECHANICS TAB --------------
    with t_adv:
        st.markdown("## Advanced Biomechanics")
        st.markdown("### **Key Performance Indicators (KPIs) for All Shots**")
        kpi_list = []
        for shot, data in user_data.items():
            kpi = {'Shot': shot}
            kpi.update(data['kpis'])
            kpi_list.append(kpi)
        kpi_df = pd.DataFrame(kpi_list)
        if not kpi_df.empty:
            st.dataframe(kpi_df.set_index('Shot'))
        else:
            st.info("No KPIs to display.")

        # Release Consistency (Standard Deviation of Release Quality across shots)
        if 'Release Quality (%)' in kpi_df.columns and len(kpi_df) > 1:
            consistency = kpi_df['Release Quality (%)'].std()
            st.markdown(f"**Release Consistency (Std Dev of Release Quality):** {consistency:.2f}%")
        else:
            st.markdown("**Release Consistency:** N/A (Requires multiple shots)")

        # Release Curvature Visualization (Placeholder)
        st.markdown("### **Release Curvature Analysis**")
        st.write("**Note:** Release Curvature metrics are placeholders and need proper implementation based on trajectory data.")
        # Add actual curvature calculations and visualizations here

        # Example: Correlation between different KPIs
        if kpi_df.shape[1] > 1:
            fig = px.scatter_matrix(kpi_df, dimensions=kpi_df.columns[1:], title="KPI Correlations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough KPI data for correlation analysis.")

    # -------------- BALL TRACKING TAB --------------
    with t_ball:
        st.markdown("## Ball Tracking & Trajectory")
        for shot, data in user_data.items():
            st.markdown(f"### Shot: {shot}")
            df_seg = data['df']

            # Visualize Ball Trajectory
            st.markdown("#### **Ball Trajectory**")
            if {'basketball_x_smooth', 'basketball_y_smooth', 'time_pose'}.issubset(df_seg.columns):
                visualize_ball_trajectory_quadratic(
                    df=df_seg[['time_pose','basketball_x_smooth','basketball_y_smooth']].copy(),
                    key_suffix=f"quadratic_{shot}"
                )
                visualize_ball_trajectory(
                    df=df_seg[['basketball_x_smooth','basketball_y_smooth']].copy(),
                    key_suffix=f"trajectory_{shot}"
                )
                # Visualize Release Point
                visualize_release_point(
                    df=df_seg,
                    release_time=data['release_time'],
                    release_point=data['release_point'],
                    key_suffix=f"ball_{shot}"
                )
            else:
                st.warning(f"Missing 'basketball_x_smooth', 'basketball_y_smooth', or 'time_pose' for ball trajectory plots in shot '{shot}'.")

            # Visualize Ball Heatmap
            st.markdown("#### **Ball Location Heatmap**")
            if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(df_seg.columns):
                visualize_ball_heatmap(
                    df_seg, 
                    key_suffix=f"heatmap_{shot}"
                )
            else:
                st.warning(f"Missing 'basketball_x_smooth' or 'basketball_y_smooth' for ball heatmap in shot '{shot}'.")

# ==========================================
# Run the App
# ==========================================
if __name__ == "__main__":
    main()
