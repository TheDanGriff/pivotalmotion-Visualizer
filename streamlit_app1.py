import streamlit as st
import boto3
import pandas as pd
import numpy as np
from io import StringIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import logging
import warnings
import statsmodels.api as sm
import os
from math import degrees, atan2
from scipy.interpolate import UnivariateSpline
import seaborn as sns  # For heatmap and advanced plots
import plotly.io as pio

# -------------------------------------------------------------------------------------
# Global Settings
# -------------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Custom Brand Colors or Theming
BRAND_PRIMARY = "#0044CC"      # Blue
BRAND_SECONDARY = "#00A0E4"    # Light Blue
BRAND_LIGHT = "#FFFFFF"        # White
BRAND_DARK = "#333333"         # Metallic Black
BRAND_LIGHTGREY = "#F0F2F6"

# -------------------------------------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------------------------------------
st.set_page_config(
    page_title="🏀 Pivotal Motion Data Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
/* Body Styling */
body {{
    background-color: {BRAND_LIGHT};
    font-family: "Helvetica Neue", Arial, sans-serif;
}}

/* Headings */
h1, h2, h3, h4 {{
    color: {BRAND_DARK};
}}

/* KPI Metrics Styling */
[data-testid="metric-container"] {{
    background-color: {BRAND_LIGHT};
    border: 1px solid {BRAND_DARK};
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}}

/* Compare Table Styling */
.compare-table {{
    width: 100%;
    border-collapse: collapse;
}}
.compare-table th, .compare-table td {{
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}}

.centered-header {{
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# AWS Setup
# -------------------------------------------------------------------------------------
@st.cache_resource
def get_s3_client():
    try:
        s3 = boto3.client(
            's3',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return s3
    except (NoCredentialsError, PartialCredentialsError):
        st.error("AWS credentials not configured properly.")
        st.stop()

@st.cache_resource
def get_dynamodb_resource():
    try:
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return dynamodb
    except (NoCredentialsError, PartialCredentialsError):
        st.error("AWS credentials not configured properly.")
        st.stop()

s3_client = get_s3_client()
dynamodb = get_dynamodb_resource()

# -------------------------------------------------------------------------------------
# Utility Functions: S3, DynamoDB, etc.
# -------------------------------------------------------------------------------------
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
        logger.error(f"Error listing users: {e}")
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
        logger.error(f"Error listing jobs for {user_email}: {e}")
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
                    seg_id = filename.split('_vibe_output.csv')[0].split('_final_output.csv')[0]
                    segs.add(seg_id)
        return sorted(segs)
    except Exception as e:
        st.error(f"Error listing segments for job '{job_id}': {e}")
        logger.error(f"Error listing segments for job '{job_id}': {e}")
        return []

def load_vibe_output(s3_client, bucket_name, user_email, job_id, segment_id):
    """
    Loads either *vibe_output.csv or *final_output.csv from S3.
    """
    prefix = f"processed/{user_email}/{job_id}/"
    possible_keys = [
        f"{prefix}{segment_id}_vibe_output.csv",
        f"{prefix}{segment_id}_final_output.csv"
    ]
    for key in possible_keys:
        try:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            body = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(body))
            df.columns = df.columns.str.strip().str.lower()
            return df
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception as e:
            st.error(f"Error loading CSV for segment '{segment_id}': {e}")
            logger.error(f"Error loading CSV for segment '{segment_id}': {e}")
            return pd.DataFrame()
    st.warning(f"No CSV found for segment '{segment_id}' in job '{job_id}'.")
    logger.warning(f"No CSV found for segment '{segment_id}' in job '{job_id}'.")
    return pd.DataFrame()

def get_metadata_from_dynamodb(dynamodb, job_id):
    """Retrieves job metadata from DynamoDB if available."""
    table_name = st.secrets.get("DYNAMODB_TABLE_NAME", "")
    if not table_name:
        st.error("DynamoDB table name missing.")
        logger.error("DynamoDB table name missing.")
        st.stop()

    table = dynamodb.Table(table_name)
    try:
        resp = table.get_item(Key={'JobID': job_id})
        return resp.get('Item', {})
    except ClientError as e:
        st.error(f"DynamoDB error: {e.response['Error']['Message']}")
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error retrieving metadata: {e}")
        logger.error(f"Unexpected error retrieving metadata: {e}")
        return {}

# -------------------------------------------------------------------------------------
# Additional Utility (Logos, etc.)
# -------------------------------------------------------------------------------------
def load_image_from_file(filepath):
    """Loads and encodes image for display in Streamlit."""
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Image file not found: {filepath}")
        return None

def show_brand_header(player_name, team_name):
    """
    Shows brand logo, plus the player's name and optional Team logo if available, centered.
    """
    col1, col2, col3 = st.columns([1, 3, 1])

    # Brand / Product Logo on the left
    brand_logo_path = os.path.join("images", "logo.PNG")  # Adjust the path as needed
    brand_logo = load_image_from_file(brand_logo_path)
    if brand_logo:
        with col1:
            st.image(brand_logo, use_container_width=True)
    else:
        with col1:
            st.markdown(f"<h1 style='text-align: center; color: {BRAND_DARK};'>Pivotal Motion</h1>", unsafe_allow_html=True)

    # Player name in the middle
    with col2:
        st.markdown(
            f"<h1 style='text-align: center; color: {BRAND_DARK};'>{player_name or 'Unknown Player'}</h1>",
            unsafe_allow_html=True
        )
        if team_name and team_name.strip().upper() != "N/A":
            cap_team = team_name.title()
            st.markdown(
                f"<h3 style='text-align: center; color: {BRAND_DARK};'>Team: {cap_team}</h3>",
                unsafe_allow_html=True
            )

    # Team Logo on the right
    if team_name and team_name.strip().upper() != "N/A":
        # Construct the correct team logo path with capitalization
        team_logo_filename = f"{team_name.title()}_logo.png"
        team_logo_path = os.path.join("images", "teams", team_logo_filename)  # Adjust the path as needed
        team_logo = load_image_from_file(team_logo_path)
        if team_logo:
            with col3:
                st.image(team_logo, use_container_width=True)
        else:
            with col3:
                st.write("")  # Leave blank if logo not found
    else:
        with col3:
            st.write("")  # Leave blank if team is N/A

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def humanize_label(label):
    """Convert 'snake_case' into Title-case, no underscores."""
    if not label or pd.isna(label):
        return "N/A"
    return ' '.join(word.capitalize() for word in label.replace('_',' ').split())

def map_body_columns(df):
    """Remap raw columns to standard left_shoulder_x, right_elbow_angle, etc."""
    rename_dict = {
        # Example partial mappings - extend as needed
        'leye_x': 'left_eye_x',
        'leye_y': 'left_eye_y',
        'reye_x': 'right_eye_x',
        'reye_y': 'right_eye_y',
        'lsjc_x': 'left_shoulder_x',
        'lsjc_y': 'left_shoulder_y',
        'lsjc_z': 'left_shoulder_z',
        'lejc_x': 'left_elbow_x',
        'lejc_y': 'left_elbow_y',
        'lejc_z': 'left_elbow_z',
        'lwjc_x': 'left_wrist_x',
        'lwjc_y': 'left_wrist_y',
        'lwjc_z': 'left_wrist_z',
        'lhjc_x': 'left_hip_x',
        'lhjc_y': 'left_hip_y',
        'lhjc_z': 'left_hip_z',
        'lkjc_x': 'left_knee_x',
        'lkjc_y': 'left_knee_y',
        'lkjc_z': 'left_knee_z',
        'lajc_x': 'left_ankle_x',
        'lajc_y': 'left_ankle_y',
        'lajc_z': 'left_ankle_z',
        'rsjc_x': 'right_shoulder_x',
        'rsjc_y': 'right_shoulder_y',
        'rsjc_z': 'right_shoulder_z',
        'rejc_x': 'right_elbow_x',
        'rejc_y': 'right_elbow_y',
        'rejc_z': 'right_elbow_z',
        'rwjc_x': 'right_wrist_x',
        'rwjc_y': 'right_wrist_y',
        'rwjc_z': 'right_wrist_z',
        'rhjc_x': 'right_hip_x',
        'rhjc_y': 'right_hip_y',
        'rhjc_z': 'right_hip_z',
        'rkjc_x': 'right_knee_x',
        'rkjc_y': 'right_knee_y',
        'rkjc_z': 'right_knee_z',
        'rajc_x': 'right_ankle_x',
        'rajc_y': 'right_ankle_y',
        'rajc_z': 'right_ankle_z',
        'right_elbow_angle': 'right_elbow_angle',
        'left_elbow_angle': 'left_elbow_angle',
        'right_knee_angle': 'right_knee_angle',
        'left_knee_angle': 'left_knee_angle',
        'right_hip_flex': 'right_hip_flex',
        'left_hip_flex': 'left_hip_flex',
        'right_ankle_flex': 'right_ankle_flex',
        'left_ankle_flex': 'left_ankle_flex',
        'right_shoulder_elev': 'right_shoulder_elev',
        'left_shoulder_elev': 'left_shoulder_elev',
        'right_shoulder_rot': 'right_shoulder_rot',
        'left_shoulder_rot': 'left_shoulder_rot',
        'torso_rot': 'torso_rot',
        'pelvis_rot': 'pelvis_rot',
        # Add more mappings as needed
    }
    df = df.rename(columns=rename_dict)
    return df

def smooth_series(series, window=11, frac=0.3):
    """Smooths a numeric series with rolling mean + LOWESS."""
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(np.nan, index=series.index)

    rolled = series.rolling(window=window, center=True, min_periods=1).mean()
    rolled = rolled.fillna(method='bfill').fillna(method='ffill')
    xvals = np.arange(len(rolled))
    try:
        smoothed = sm.nonparametric.lowess(rolled, xvals, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=series.index)
    except Exception as e:
        logger.error(f"LOWESS smoothing failed: {e}. Using rolled mean instead.")
        return rolled

def compute_velocity(series, dt=0.01):
    """Basic centered finite difference to compute velocity from a series."""
    shifted_forward = series.shift(-1)
    shifted_back = series.shift(1)
    vel = (shifted_forward - shifted_back) / (2.0 * dt)
    return vel

def compute_angular_velocity(df, angle_col, dt=0.01):
    """Compute angular velocity from an angle col, store in angle_col+'_vel'."""
    if angle_col not in df.columns:
        df[angle_col + '_vel'] = np.nan
        return df
    df[angle_col + '_vel'] = compute_velocity(df[angle_col], dt=dt)
    return df

def compute_acceleration(series, dt=0.01):
    """Compute acceleration from a velocity series."""
    shifted_forward = series.shift(-1)
    shifted_back = series.shift(1)
    accel = (shifted_forward - shifted_back) / (2.0 * dt)
    return accel

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
    """Compute elbow/knee angles from raw 2D coords."""
    # Left Elbow
    if all(x in df.columns for x in ['left_shoulder_x','left_shoulder_y','left_elbow_x','left_elbow_y','left_wrist_x','left_wrist_y']):
        df['left_elbow_angle'] = df.apply(lambda row:
            angle_2d(
                (row['left_shoulder_x'], row['left_shoulder_y']),
                (row['left_elbow_x'], row['left_elbow_y']),
                (row['left_wrist_x'], row['left_wrist_y'])
            ), axis=1)
    else:
        df['left_elbow_angle'] = np.nan

    # Right Elbow
    if all(x in df.columns for x in ['right_shoulder_x','right_shoulder_y','right_elbow_x','right_elbow_y','right_wrist_x','right_wrist_y']):
        df['right_elbow_angle'] = df.apply(lambda row:
            angle_2d(
                (row['right_shoulder_x'], row['right_shoulder_y']),
                (row['right_elbow_x'], row['right_elbow_y']),
                (row['right_wrist_x'], row['right_wrist_y'])
            ), axis=1)
    else:
        df['right_elbow_angle'] = np.nan

    # Left Knee
    if all(x in df.columns for x in ['left_hip_x','left_hip_y','left_knee_x','left_knee_y','left_ankle_x','left_ankle_y']):
        df['left_knee_angle'] = df.apply(lambda row:
            angle_2d(
                (row['left_hip_x'], row['left_hip_y']),
                (row['left_knee_x'], row['left_knee_y']),
                (row['left_ankle_x'], row['left_ankle_y'])
            ), axis=1)
    else:
        df['left_knee_angle'] = np.nan

    # Right Knee
    if all(x in df.columns for x in ['right_hip_x','right_hip_y','right_knee_x','right_knee_y','right_ankle_x','right_ankle_y']):
        df['right_knee_angle'] = df.apply(lambda row:
            angle_2d(
                (row['right_hip_x'], row['right_hip_y']),
                (row['right_knee_x'], row['right_knee_y']),
                (row['right_ankle_x'], row['right_ankle_y'])
            ), axis=1)
    else:
        df['right_knee_angle'] = np.nan

    return df

def detect_release_frame(df):
    """Find frame with max right_wrist_y_smooth (fallback to right_wrist_y)."""
    if 'right_wrist_y_smooth' in df.columns:
        col = 'right_wrist_y_smooth'
    elif 'right_wrist_y' in df.columns:
        col = 'right_wrist_y'
    else:
        logger.warning("No right wrist Y data for release detection.")
        return None

    valid_df = df.dropna(subset=[col])
    if valid_df.empty:
        logger.warning(f"No valid data in '{col}' for release detection.")
        return None
    max_idx = valid_df[col].idxmax()
    if 'frame' in df.columns:
        return df.loc[max_idx, 'frame']
    return None

def trim_shot(df, release_frame, frames_before=30, frames_after=20):
    """Return frames in [release_frame - frames_before, release_frame + frames_after]."""
    if release_frame is None or 'frame' not in df.columns:
        return df
    start = release_frame - frames_before
    end = release_frame + frames_after
    return df[(df['frame'] >= start) & (df['frame'] <= end)].copy()

def identify_phases(df, release_frame):
    """5-phase model: Preparation, Ball Elevation, Stability, Release, Inertia."""
    phases = pd.Series([np.nan] * len(df), index=df.index)
    if release_frame is None:
        return phases

    # Define phase boundaries relative to release frame
    prep_end = release_frame - 40
    elev_start = prep_end
    elev_end = release_frame - 15
    stab_start = release_frame - 15
    stab_end = release_frame
    rel_start = release_frame
    rel_end = release_frame + 15
    inert_start = rel_end
    inert_end = rel_end + 30

    phases[(df['frame'] >= prep_end) & (df['frame'] < elev_start)] = 'Preparation'
    phases[(df['frame'] >= elev_start) & (df['frame'] < elev_end)] = 'Ball Elevation'
    phases[(df['frame'] >= elev_end) & (df['frame'] < stab_end)] = 'Stability'
    phases[(df['frame'] >= rel_start) & (df['frame'] <= rel_end)] = 'Release'
    phases[(df['frame'] > inert_start) & (df['frame'] <= inert_end)] = 'Inertia'
    return phases

def compute_time_based_separations(df, angle_cols=None, extend_threshold=0.5, min_extend_frames=5):
    """
    Computes frame-differences (time-based separation) between plateau times of specified joints.
    e.g., ankle->knee, knee->hip, hip->shoulder.

    :param df: DataFrame with 'frame' plus angle columns (e.g. 'left_knee_angle_smooth').
    :param angle_cols: dict specifying which columns correspond to 'ankle','knee','hip','shoulder'.
                       e.g. {
                           'ankle': 'left_ankle_flex_smooth',
                           'knee':  'left_knee_angle_smooth',
                           'hip':   'left_hip_flex_smooth',
                           'shoulder': 'left_shoulder_elev_smooth'
                       }
    :param extend_threshold: Angular threshold (deg) for detecting a plateau 
                            (stops increasing by more than this).
    :param min_extend_frames: # of consecutive frames < threshold to confirm plateau.
    :return: dict of separation times, e.g.
            {
                'ankle_knee_sep_time': N (int frames or None),
                'knee_hip_sep_time': N,
                'hip_shoulder_sep_time': N
            }
    """
    if angle_cols is None:
        angle_cols = {
            'ankle': 'left_ankle_flex_smooth',
            'knee':  'left_knee_angle_smooth',
            'hip':   'left_hip_flex_smooth',
            'shoulder': 'left_shoulder_elev_smooth'
        }

    results = {}
    df = df.sort_values('frame').reset_index(drop=True)
    frames = df['frame'].values

    # Find plateau frames for each joint
    plateau_frames = {}
    for joint, col in angle_cols.items():
        if col not in df.columns:
            plateau_frames[joint] = None
            continue

        angle_vals = df[col].values
        dangle = np.diff(angle_vals, prepend=angle_vals[0])  # difference wrt previous
        consecutive_count = 0
        plateau_frame = None

        for i in range(1, len(dangle)):
            # If dangle[i] < threshold => angle is no longer significantly increasing
            if dangle[i] < extend_threshold:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= min_extend_frames:
                plateau_frame = frames[i]
                break

        plateau_frames[joint] = plateau_frame

    # Utility for time differences (in frames)
    def time_diff(j1, j2):
        f1 = plateau_frames.get(j1, None)
        f2 = plateau_frames.get(j2, None)
        if f1 is None or f2 is None:
            return None
        return abs(f2 - f1)

    # Calculate specific separation times
    results['ankle_knee_sep_time'] = time_diff('ankle','knee')
    results['knee_hip_sep_time']   = time_diff('knee','hip')
    results['hip_shoulder_sep_time'] = time_diff('hip','shoulder')

    return results

def compute_pivotal_score(metrics):
    """Calculate a comprehensive Pivotal Score based on key metrics."""
    # Define weights for each metric
    weights = {
        'Release Angle (deg)': 0.3,
        'Fluidity Score': 0.25,
        'Asymmetry Index': 0.2,
        'Chain Range (frames)': 0.15
    }
    # Normalize and compute Pivotal Score
    pivotal_score = 0
    total_weight = 0
    for metric, weight in weights.items():
        val = metrics.get(metric, np.nan)
        if pd.notna(val):
            # Normalize based on expected ranges (these can be adjusted)
            if metric == 'Release Angle (deg)':
                norm_val = val / 50  # Assuming max 50 degrees
            elif metric == 'Fluidity Score':
                norm_val = val  # Already between 0 and 1
            elif metric == 'Asymmetry Index':
                norm_val = 1 - val  # Lower asymmetry is better
            elif metric == 'Chain Range (frames)':
                norm_val = min(val / 50, 1)  # Assuming max 50 frames
            else:
                norm_val = 0
            pivotal_score += norm_val * weight
            total_weight += weight
    if total_weight > 0:
        metrics['Pivotal Score'] = pivotal_score / total_weight
    else:
        metrics['Pivotal Score'] = np.nan
    return metrics

def filter_for_shooter_by_phases(df):
    """
    Filters the DataFrame to include only the shooter based on phases.
    Assumes that the shooter is the player who has data during 'Preparation', 'Ball Elevation', etc.
    """
    if 'phase' not in df.columns or 'player_id' not in df.columns:
        logger.warning("Missing 'phase' or 'player_id' columns for shooter filtering.")
        return df

    # Identify the most frequent player during 'Preparation' phase
    prep_players = df[df['phase'] == 'Preparation']['player_id']
    if prep_players.empty:
        logger.warning("No 'Preparation' phase data to identify shooter.")
        return df

    shooter_id = prep_players.mode()[0]
    logger.info(f"Identified shooter with player_id: {shooter_id}")

    # Filter DataFrame for the shooter
    filtered_df = df[df['player_id'] == shooter_id].copy()
    return filtered_df

def offset_z_axis(df):
    """
    Offset Z-axis so floor=0 based on ankle Z coordinates.
    """
    # Prefer smoothed ankle Z if available
    ankle_z_cols = ['left_ankle_z_smooth', 'right_ankle_z_smooth']
    raw_ankle_z_cols = ['left_ankle_z', 'right_ankle_z']

    if all(col in df.columns for col in ankle_z_cols):
        floor_z = min(df['left_ankle_z_smooth'].min(), df['right_ankle_z_smooth'].min())
        for col in ankle_z_cols:
            df[col] = df[col] - floor_z
        # Offset other Z columns accordingly
        z_cols = [c for c in df.columns if c.endswith('_z_smooth')]
        for col in z_cols:
            df[col] = df[col] - floor_z
        logger.info("Z-axis offset using smoothed ankle Z.")
    elif all(col in df.columns for col in raw_ankle_z_cols):
        floor_z = min(df['left_ankle_z'].min(), df['right_ankle_z'].min())
        for col in raw_ankle_z_cols:
            df[col] = df[col] - floor_z
        # Offset other Z columns accordingly
        z_cols = [c for c in df.columns if c.endswith('_z')]
        for col in z_cols:
            df[col] = df[col] - floor_z
        logger.info("Z-axis offset using raw ankle Z.")
    else:
        logger.warning("Insufficient data to offset Z-axis.")

    return df

def preprocess_and_smooth(df, dt=0.01):
    """
    Smooth key position & angle columns, compute velocities & accelerations.
    """
    position_cols = [
        'right_wrist_x','right_wrist_y','right_wrist_z',
        'left_wrist_x','left_wrist_y','left_wrist_z',
        'right_ankle_x','right_ankle_y','right_ankle_z',
        'left_ankle_x','left_ankle_y','left_ankle_z',
        'right_knee_x','right_knee_y','right_knee_z',
        'left_knee_x','left_knee_y','left_knee_z',
        'right_hip_x','right_hip_y','right_hip_z',
        'left_hip_x','left_hip_y','left_hip_z',
    ]
    angle_cols = [
        'right_elbow_angle','left_elbow_angle','right_knee_angle','left_knee_angle',
        'right_hip_flex','left_hip_flex','right_ankle_flex','left_ankle_flex',
        'right_shoulder_elev','left_shoulder_elev','right_shoulder_rot','left_shoulder_rot',
        'torso_rot','pelvis_rot'
    ]
    for c in position_cols + angle_cols:
        if c in df.columns:
            df[c + '_smooth'] = smooth_series(df[c], window=11, frac=0.3)

    # Compute angular velocities
    for a in angle_cols:
        df = compute_angular_velocity(df, a, dt=dt)

    # Compute accelerations
    for a in angle_cols:
        v = a + '_vel'
        acc = a + '_accel'
        if v in df.columns and df[v].notna().any():
            df[acc] = compute_acceleration(df[v], dt=dt)
        else:
            df[acc] = np.nan
    return df

# -------------------------------------------------------------------------------------
# Ball Tracking & Visualization
# -------------------------------------------------------------------------------------
def visualize_ball_trajectory_simple(df):
    """Visualizes the ball trajectory with both time-based and court-plane views."""
    df = df.dropna(subset=['frame','x','y']).copy()
    if df.empty or len(df) < 2:
        st.warning("Not enough ball data to plot.")
        return

    # Correcting orientation: Invert Y-axis if necessary
    df['y_corrected'] = -df['y']

    # Smooth the trajectory to form an arc
    df['x_smooth'] = smooth_series(df['x'], window=5, frac=0.3)
    df['y_smooth'] = smooth_series(df['y_corrected'], window=5, frac=0.3)

    # Sort by frame to see time-based progression
    df_time = df.sort_values('frame')

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Ball Over Time (X vs Frame)", "Ball Arc (X vs Y)"),
        column_widths=[0.5, 0.5]
    )

    # Subplot 1: X vs. Frame
    fig.add_trace(go.Scatter(
        x=df_time['frame'], y=df_time['x_smooth'],
        mode='lines+markers',
        name='Ball X over Time',
        marker=dict(color='blue', size=4)
    ), row=1, col=1)

    # Subplot 2: X vs. Y (Court Plane)
    fig.add_trace(go.Scatter(
        x=df_time['x_smooth'], y=df_time['y_smooth'],
        mode='lines+markers',
        name='Ball Arc (X,Y)',
        marker=dict(color='red', size=4)
    ), row=1, col=2)

    fig.update_layout(
        title="Ball Trajectory",
        showlegend=False
    )
    fig.update_xaxes(title_text="Frame", row=1, col=1)
    fig.update_yaxes(title_text="X Position", row=1, col=1)
    fig.update_xaxes(title_text="X Position", row=1, col=2)
    fig.update_yaxes(title_text="Y Position", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# Heatmaps and Animations
# -------------------------------------------------------------------------------------
def visualize_heatmap(df):
    """Visualizes a heatmap of joint angles correlations."""
    angle_columns = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
                    'left_hip_flex', 'right_hip_flex', 'left_ankle_flex', 'right_ankle_flex']
    available_angles = [col for col in angle_columns if col in df.columns]
    if not available_angles:
        st.warning("No joint angle data available for heatmap.")
        logger.warning("No joint angle data available for heatmap.")
        return
    correlation = df[available_angles].corr()
    fig = px.imshow(correlation, text_auto=True, color_continuous_scale='Blues',
                    title="Joint Angles Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

def visualize_animation(df, segment="Segment"):
    """Creates an interactive 3D animation of the motion with selectable body parts."""
    # Check for necessary columns
    required_cols = ['frame', 'left_shoulder_x_smooth', 'left_shoulder_y_smooth', 'left_shoulder_z_smooth',
                     'right_shoulder_x_smooth', 'right_shoulder_y_smooth', 'right_shoulder_z_smooth',
                     'left_elbow_x_smooth', 'left_elbow_y_smooth', 'left_elbow_z_smooth',
                     'right_elbow_x_smooth', 'right_elbow_y_smooth', 'right_elbow_z_smooth',
                     'left_wrist_x_smooth', 'left_wrist_y_smooth', 'left_wrist_z_smooth',
                     'right_wrist_x_smooth', 'right_wrist_y_smooth', 'right_wrist_z_smooth',
                     'left_hip_x_smooth', 'left_hip_y_smooth', 'left_hip_z_smooth',
                     'right_hip_x_smooth', 'right_hip_y_smooth', 'right_hip_z_smooth',
                     'left_knee_angle_smooth', 'right_knee_angle_smooth',
                     'left_ankle_flex_smooth', 'right_ankle_flex_smooth']
    if not all(col in df.columns for col in required_cols):
        st.warning("Insufficient data for 3D animation.")
        logger.warning("Insufficient data for 3D animation.")
        return

    # Prepare data for Plotly
    df_sorted = df.sort_values('frame')
    frames = df_sorted['frame'].unique()

    # Create a dropdown to select body parts
    body_parts = {
        'Whole Body': ['left_shoulder_x_smooth', 'left_shoulder_y_smooth', 'left_shoulder_z_smooth',
                      'right_shoulder_x_smooth', 'right_shoulder_y_smooth', 'right_shoulder_z_smooth',
                      'left_elbow_x_smooth', 'left_elbow_y_smooth', 'left_elbow_z_smooth',
                      'right_elbow_x_smooth', 'right_elbow_y_smooth', 'right_elbow_z_smooth',
                      'left_wrist_x_smooth', 'left_wrist_y_smooth', 'left_wrist_z_smooth',
                      'right_wrist_x_smooth', 'right_wrist_y_smooth', 'right_wrist_z_smooth',
                      'left_hip_x_smooth', 'left_hip_y_smooth', 'left_hip_z_smooth',
                      'right_hip_x_smooth', 'right_hip_y_smooth', 'right_hip_z_smooth',
                      'left_knee_angle_smooth', 'right_knee_angle_smooth',
                      'left_ankle_flex_smooth', 'right_ankle_flex_smooth'],
        'Lower Body': ['left_hip_x_smooth', 'left_hip_y_smooth', 'left_hip_z_smooth',
                      'right_hip_x_smooth', 'right_hip_y_smooth', 'right_hip_z_smooth',
                      'left_knee_angle_smooth', 'right_knee_angle_smooth',
                      'left_ankle_flex_smooth', 'right_ankle_flex_smooth'],
        'Right Arm': ['right_shoulder_x_smooth', 'right_shoulder_y_smooth', 'right_shoulder_z_smooth',
                      'right_elbow_x_smooth', 'right_elbow_y_smooth', 'right_elbow_z_smooth',
                      'right_wrist_x_smooth', 'right_wrist_y_smooth', 'right_wrist_z_smooth'],
        'Left Arm': ['left_shoulder_x_smooth', 'left_shoulder_y_smooth', 'left_shoulder_z_smooth',
                     'left_elbow_x_smooth', 'left_elbow_y_smooth', 'left_elbow_z_smooth',
                     'left_wrist_x_smooth', 'left_wrist_y_smooth', 'left_wrist_z_smooth']
    }

    selected_body_part = st.selectbox("Select Body Part to Display in Animation", list(body_parts.keys()))

    # Initialize figure
    fig = go.Figure()

    # Add traces for each frame
    for frame in frames:
        frame_data = df_sorted[df_sorted['frame'] == frame]
        if frame_data.empty:
            continue
        row = frame_data.iloc[0]
        traces = []
        selected_cols = body_parts[selected_body_part]

        # Define connections based on selected body part
        connections = []
        if selected_body_part == 'Whole Body':
            connections = [
                ('left_shoulder_x_smooth', 'left_elbow_x_smooth'),
                ('left_elbow_x_smooth', 'left_wrist_x_smooth'),
                ('right_shoulder_x_smooth', 'right_elbow_x_smooth'),
                ('right_elbow_x_smooth', 'right_wrist_x_smooth'),
                ('left_shoulder_x_smooth', 'right_shoulder_x_smooth'),
                ('left_shoulder_x_smooth', 'left_hip_x_smooth'),
                ('right_shoulder_x_smooth', 'right_hip_x_smooth'),
                ('left_hip_x_smooth', 'right_hip_x_smooth'),
                ('left_hip_x_smooth', 'left_knee_angle_smooth'),
                ('left_knee_angle_smooth', 'left_ankle_flex_smooth'),
                ('right_hip_x_smooth', 'right_knee_angle_smooth'),
                ('right_knee_angle_smooth', 'right_ankle_flex_smooth'),
            ]
        elif selected_body_part == 'Lower Body':
            connections = [
                ('left_hip_x_smooth', 'right_hip_x_smooth'),
                ('left_hip_x_smooth', 'left_knee_angle_smooth'),
                ('left_knee_angle_smooth', 'left_ankle_flex_smooth'),
                ('right_hip_x_smooth', 'right_knee_angle_smooth'),
                ('right_knee_angle_smooth', 'right_ankle_flex_smooth'),
            ]
        elif selected_body_part == 'Right Arm':
            connections = [
                ('right_shoulder_x_smooth', 'right_elbow_x_smooth'),
                ('right_elbow_x_smooth', 'right_wrist_x_smooth'),
            ]
        elif selected_body_part == 'Left Arm':
            connections = [
                ('left_shoulder_x_smooth', 'left_elbow_x_smooth'),
                ('left_elbow_x_smooth', 'left_wrist_x_smooth'),
            ]

        # Plot each connection as a line
        for conn in connections:
            x = [row[conn[0]], row[conn[1]]]
            y = [row[conn[0].replace('_x_smooth','_y_smooth')], row[conn[1].replace('_x_smooth','_y_smooth')]]
            z = [row[conn[0].replace('_x_smooth','_z_smooth')], row[conn[1].replace('_x_smooth','_z_smooth')]]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='green', width=4),
                hoverinfo='none'
            ))

        # Scatter points for joints
        for joint in selected_cols:
            x = row[joint]
            y = row[joint.replace('_x_smooth','_y_smooth')]
            z = row[joint.replace('_x_smooth','_z_smooth')]
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=5, color='red'),
                name=joint.replace('_smooth','').replace('_',' ').title(),
                hoverinfo='text'
            ))

    # Update layout
    fig.update_layout(
        title=f"3D Animation of {selected_body_part}",
        scene=dict(
            xaxis=dict(title='X', range=[-1,1], backgroundcolor=BRAND_LIGHTGREY),
            yaxis=dict(title='Y', range=[-1,1], backgroundcolor=BRAND_LIGHTGREY),
            zaxis=dict(title='Z', range=[-1,1], backgroundcolor=BRAND_LIGHTGREY),
            aspectmode='data'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# Comparison Functionality
# -------------------------------------------------------------------------------------
def compare_two_segments(seg1_name, seg1_df, seg2_name, seg2_df, player_height=2.0):
    """
    Renders side-by-side KPI comparison and also shows side-by-side graphs.
    """
    # Compute metrics
    seg1_release = detect_release_frame(seg1_df)
    seg2_release = detect_release_frame(seg2_df)

    seg1_mets = calculate_advanced_metrics(seg1_df, seg1_release, player_height=player_height)
    seg2_mets = calculate_advanced_metrics(seg2_df, seg2_release, player_height=player_height)

    # Calculate Pivotal Score
    seg1_mets = compute_pivotal_score(seg1_mets)
    seg2_mets = compute_pivotal_score(seg2_mets)

    # For display convenience
    st.markdown(f"## Compare Segments: **{seg1_name}** vs. **{seg2_name}**")

    # 1) KPI Comparison
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"### {seg1_name}")
        display_kpis(seg1_df, seg1_mets)

    with colC:
        st.markdown(f"### {seg2_name}")
        display_kpis(seg2_df, seg2_mets)

    with colB:
        st.markdown("### Differences")
        # Calculate differences
        diffs = {}
        for key in seg1_mets.keys():
            val1 = seg1_mets.get(key, np.nan)
            val2 = seg2_mets.get(key, np.nan)
            if pd.notna(val1) and pd.notna(val2):
                diff = val2 - val1
                color = "green" if diff > 0 else "red"
                diffs[key] = f"<span style='color:{color}'>{diff:.2f}</span>"
            else:
                diffs[key] = "N/A"
        # Display differences in a table
        diffs_html = "<table class='compare-table'><tr><th>Metric</th><th>Difference</th></tr>"
        for key, val in diffs.items():
            diffs_html += f"<tr><td>{key}</td><td>{val}</td></tr>"
        diffs_html += "</table>"
        st.markdown(diffs_html, unsafe_allow_html=True)

    # 2) Graphs Side-by-Side
    st.markdown("### Joint Angles Comparison")
    col1, col2 = st.columns(2)
    with col1:
        plot_joint_angles(seg1_df, release_frame=seg1_release, selectbox_key=f"{seg1_name}_angles_compare")
    with col2:
        plot_joint_angles(seg2_df, release_frame=seg2_release, selectbox_key=f"{seg2_name}_angles_compare")

    st.markdown("### Shoulder-Hip Separation Comparison")
    col3, col4 = st.columns(2)
    with col3:
        plot_separation(seg1_df, release_frame=seg1_release)
    with col4:
        plot_separation(seg2_df, release_frame=seg2_release)

    # Plot Ball Trajectories Side-by-Side
    st.markdown("### Ball Trajectories Comparison")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"#### {seg1_name} Ball Trajectory")
        if {'frame','x','y'}.issubset(seg1_df.columns):
            visualize_ball_trajectory_simple(seg1_df[['frame','x','y']].copy())
        else:
            st.warning("No ball data found in segment 1.")
    with col6:
        st.markdown(f"#### {seg2_name} Ball Trajectory")
        if {'frame','x','y'}.issubset(seg2_df.columns):
            visualize_ball_trajectory_simple(seg2_df[['frame','x','y']].copy())
        else:
            st.warning("No ball data found in segment 2.")

    # 3) Advanced Graphs: Heatmaps and Animations
    st.markdown("### Advanced Visualizations")

    # Heatmap of Joint Angles Correlation
    st.markdown("#### Heatmap of Joint Angles Correlation")
    angle_columns = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle',
                    'left_hip_flex', 'right_hip_flex', 'left_ankle_flex', 'right_ankle_flex']
    available_angles = [col for col in angle_columns if col in seg1_df.columns and col in seg2_df.columns]
    if available_angles:
        combined_angles = pd.concat([seg1_df[available_angles], seg2_df[available_angles]], ignore_index=True)
        correlation = combined_angles.corr()
        fig_heat = px.imshow(correlation, text_auto=True, title="Joint Angles Correlation Heatmap",
                             color_continuous_scale='Blues')
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Not enough joint angle data to create a heatmap.")

    # Animation of Joint Angles Over Time
    st.markdown("#### Animation of Joint Angles Over Time")
    # Select a joint angle to animate
    angle_options_anim = [col for col in angle_columns if col in seg1_df.columns]
    if angle_options_anim:
        selected_angle_anim = st.selectbox("Select Joint Angle for Animation", angle_options_anim, key="animation_angle")
        # Melt DataFrame for both segments
        melted_seg1 = seg1_df[['frame', selected_angle_anim]].rename(columns={selected_angle_anim: 'Angle'})
        melted_seg1['Joint'] = selected_angle_anim.replace('_',' ').title()
        melted_seg1['Segment'] = seg1_name

        melted_seg2 = seg2_df[['frame', selected_angle_anim]].rename(columns={selected_angle_anim: 'Angle'})
        melted_seg2['Joint'] = selected_angle_anim.replace('_',' ').title()
        melted_seg2['Segment'] = seg2_name

        melted_combined = pd.concat([melted_seg1, melted_seg2], ignore_index=True)

        fig_anim = px.line(melted_combined, x='frame', y='Angle', color='Segment',
                           animation_frame='frame', range_y=[0, 180],
                           title=f"Animation of {selected_angle_anim.replace('_',' ').title()} Over Frames")

        fig_anim.update_layout(
            xaxis_title="Frame",
            yaxis_title="Angle (°)"
        )

        st.plotly_chart(fig_anim, use_container_width=True)
    else:
        st.warning("No joint angle data available for animation.")

# -------------------------------------------------------------------------------------
# Joint Angle Plotting Functions
# -------------------------------------------------------------------------------------
def plot_joint_angles(df, release_frame=None, selectbox_key="angles"):
    """
    Plots a selectbox-driven angle over frames with a unique key to avoid collisions.
    """
    angle_options = [
        'right_ankle_flex_smooth',
        'right_knee_angle_smooth',
        'right_hip_flex_smooth',
        'right_elbow_angle_smooth',
        'left_ankle_flex_smooth',
        'left_knee_angle_smooth',
        'left_hip_flex_smooth',
        'left_elbow_angle_smooth'
    ]
    available_angles = [c for c in angle_options if c in df.columns]
    if not available_angles:
        st.warning("No key joint angle data available for plotting.")
        logger.warning("No key joint angle data available for plotting.")
        return

    selected_angle = st.selectbox(
        "Select Joint Angle to Display",
        available_angles,
        format_func=lambda x: x.replace('_smooth','').replace('_',' ').title(),
        key=selectbox_key  # Unique key here
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df[selected_angle],
        mode='lines',
        name=selected_angle.replace('_smooth','').replace('_',' ').title()
    ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(
            x=release_frame, 
            line=dict(color='red', dash='dash'), 
            annotation_text="Release"
        )
    # Shaded phases
    phases = df['phase'].unique() if 'phase' in df.columns else []
    for phs in phases:
        if pd.isna(phs):
            continue
        color_map = {
            'Preparation':'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation':'rgba(255,165,0,0.1)',  # Orange
            'Stability':'rgba(255,255,0,0.1)',        # Yellow
            'Release':'rgba(255,0,0,0.1)',           # Red
            'Inertia':'rgba(128,0,128,0.1)'          # Purple
        }
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            color = color_map.get(phs, 'rgba(128,128,128,0.1)')        # Grey as default
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title=f"{selected_angle.replace('_smooth','').replace('_',' ').title()} Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angle (°)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_separation(df, release_frame=None):
    """
    Plots Shoulder-Hip Separation over frames.
    """
    # Ensure the correct column names
    right_sep = 'shoulder_hip_separation_right_smooth'
    left_sep = 'shoulder_hip_separation_left_smooth'
    missing_right = False
    missing_left = False
    if right_sep not in df.columns:
        if all(col in df.columns for col in ['shoulder_hip_separation_right (m)', 'shoulder_hip_separation_left (m)']):
            df[right_sep] = smooth_series(df['shoulder_hip_separation_right (m)'])
        else:
            missing_right = True
    if left_sep not in df.columns:
        if all(col in df.columns for col in ['shoulder_hip_separation_right (m)', 'shoulder_hip_separation_left (m)']):
            df[left_sep] = smooth_series(df['shoulder_hip_separation_left (m)'])
        else:
            missing_left = True

    if missing_right or missing_left:
        st.warning("Shoulder-Hip Separation data not available.")
        logger.warning("Shoulder-Hip Separation data not available.")
        return

    fig = go.Figure()
    if right_sep in df.columns:
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[right_sep],
            mode='lines',
            name='Shoulder-Hip Separation Right (m)',
            line=dict(color='blue')
        ))
    if left_sep in df.columns:
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[left_sep],
            mode='lines',
            name='Shoulder-Hip Separation Left (m)',
            line=dict(color='green')
        ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique() if 'phase' in df.columns else []
    for phs in phases:
        if pd.isna(phs):
            continue
        color_map = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            col = color_map.get(phs, 'rgba(128,128,128,0.1)')
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=col,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title="Shoulder-Hip Separation Over Frames",
        xaxis_title="Frame",
        yaxis_title="Separation (m)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_angular_velocity(df, release_frame=None):
    """
    Plots angular velocities of key joints over frames.
    """
    vel_cols = [
        'right_ankle_flex_vel',
        'right_knee_angle_vel',
        'right_hip_flex_vel',
        'right_elbow_angle_vel',
        'left_ankle_flex_vel',
        'left_knee_angle_vel',
        'left_hip_flex_vel',
        'left_elbow_angle_vel'
    ]
    vel_cols = [c for c in vel_cols if c in df.columns]

    if not vel_cols:
        st.warning("Angular velocity data not available.")
        logger.warning("Angular velocity data not available.")
        return

    # Allow user to select which velocity to view
    selected_vel = st.selectbox("Select Angular Velocity to Display", vel_cols, format_func=lambda x: x.replace('_vel','').replace('_',' ').title())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df[selected_vel],
        mode='lines',
        name=selected_vel.replace('_vel','').replace('_',' ').title()
    ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique() if 'phase' in df.columns else []
    for phs in phases:
        if pd.isna(phs):
            continue
        color_map = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            col = color_map.get(phs, 'rgba(128,128,128,0.1)')
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=col,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title=f"{selected_vel.replace('_vel','').replace('_',' ').title()} Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angular Velocity (°/frame)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_accelerations(df, release_frame=None):
    """
    Plots angular accelerations of key joints over frames.
    """
    accel_cols = [
        'right_ankle_flex_accel',
        'right_knee_angle_accel',
        'right_hip_flex_accel',
        'right_elbow_angle_accel',
        'left_ankle_flex_accel',
        'left_knee_angle_accel',
        'left_hip_flex_accel',
        'left_elbow_angle_accel'
    ]
    accel_cols = [c for c in accel_cols if c in df.columns]

    if not accel_cols:
        st.warning("Angular acceleration data not available.")
        logger.warning("Angular acceleration data not available.")
        return

    # Allow user to select which acceleration to view
    selected_accel = st.selectbox("Select Angular Acceleration to Display", accel_cols, format_func=lambda x: x.replace('_accel','').replace('_',' ').title())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df[selected_accel],
        mode='lines',
        name=selected_accel.replace('_accel','').replace('_',' ').title()
    ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique() if 'phase' in df.columns else []
    for phs in phases:
        if pd.isna(phs):
            continue
        color_map = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            col = color_map.get(phs, 'rgba(128,128,128,0.1)')
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=col,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title=f"{selected_accel.replace('_accel','').replace('_',' ').title()} Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angular Acceleration (°/frame²)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# KPI Display Function
# -------------------------------------------------------------------------------------
def display_kpis(df, metrics):
    """Displays angles, release height, angle, separation, etc."""
    st.markdown("### 🎯 Key Performance Indicators (KPIs)")

    def safe_avg(c):
        if c in df.columns and df[c].notna().any():
            return df[c].mean()
        return np.nan

    kpis = {
        "Avg Left Elbow Angle (°)": safe_avg('left_elbow_angle'),
        "Avg Right Elbow Angle (°)": safe_avg('right_elbow_angle'),
        "Avg Left Knee Angle (°)": safe_avg('left_knee_angle'),
        "Avg Right Knee Angle (°)": safe_avg('right_knee_angle'),
        "Relative Release Height": metrics.get('Relative Release Height', np.nan),
        "Release Angle (deg)": metrics.get('Release Angle (deg)', np.nan),
        "Asymmetry Index": metrics.get('Asymmetry Index', np.nan),
        "Fluidity Score": metrics.get('Fluidity Score', np.nan),
        "Chain Range (frames)": metrics.get('Chain Range (frames)', np.nan),
        "Pivotal Score": metrics.get('Pivotal Score', np.nan)
    }

    # Display KPIs in a grid
    num_cols = 4
    cols = st.columns(num_cols)
    for idx, (name, val) in enumerate(kpis.items()):
        col = cols[idx % num_cols]
        if isinstance(val, (int, float)) and not np.isnan(val):
            if "Angle" in name:
                disp = f"{val:.2f}°"
            elif "Separation" in name.lower():
                disp = f"{val:.2f} frames"
            elif "Height" in name:
                disp = f"{val:.3f} m"
            elif "Score" in name or "Index" in name:
                disp = f"{val:.2f}"
            elif "frames" in name.lower():
                disp = f"{int(val)} frames"
            else:
                disp = f"{val:.2f}"
        else:
            disp = "N/A"
        col.metric(name, disp)

    # Additional Warnings and Insights
    if 'Release Angle (deg)' in kpis and not pd.isna(kpis['Release Angle (deg)']):
        if kpis['Release Angle (deg)'] < 35 or kpis['Release Angle (deg)'] > 50:
            st.warning("Release Angle outside typical range (35–50°).")
    if 'Asymmetry Index' in kpis and not pd.isna(kpis['Asymmetry Index']):
        if kpis['Asymmetry Index'] > 0.1:
            st.warning("High asymmetry index – consider balancing training.")
    if 'Fluidity Score' in kpis and not pd.isna(kpis['Fluidity Score']):
        if kpis['Fluidity Score'] < 0.5:
            st.warning("Low fluidity score – focus on smooth movements.")
    if 'Pivotal Score' in kpis and not pd.isna(kpis['Pivotal Score']):
        if kpis['Pivotal Score'] < 0.6:
            st.warning("Pivotal Score is low – review shooting mechanics.")

# -------------------------------------------------------------------------------------
# Advanced Metrics Calculation
# -------------------------------------------------------------------------------------
def calculate_advanced_metrics(df, release_frame, player_height=2.0):
    """
    Calculates advanced metrics including Relative Release Height, Release Angle,
    Asymmetry Index, Fluidity Score, Chain Range, etc.
    """
    metrics = {}

    # (1) Relative Release Height
    # Assuming 'right_wrist_y_smooth' represents the vertical position
    if 'right_wrist_y_smooth' in df.columns:
        release_y = df[df['frame'] == release_frame]['right_wrist_y_smooth'].values
        if len(release_y) > 0:
            metrics['Relative Release Height'] = release_y[0] / player_height  # Normalize by player height
        else:
            metrics['Relative Release Height'] = np.nan
    else:
        metrics['Relative Release Height'] = np.nan

    # (2) Release Angle
    if 'right_elbow_angle_smooth' in df.columns:
        release_angle = df[df['frame'] == release_frame]['right_elbow_angle_smooth'].values
        if len(release_angle) > 0:
            metrics['Release Angle (deg)'] = release_angle[0]
        else:
            metrics['Release Angle (deg)'] = np.nan
    else:
        metrics['Release Angle (deg)'] = np.nan

    # (3) Asymmetry Index
    # Compare left and right elbow and knee angles
    if 'left_elbow_angle_smooth' in df.columns and 'right_elbow_angle_smooth' in df.columns:
        avg_left_elbow = df['left_elbow_angle_smooth'].mean()
        avg_right_elbow = df['right_elbow_angle_smooth'].mean()
        elbow_asym = abs(avg_left_elbow - avg_right_elbow) / ((avg_left_elbow + avg_right_elbow)/2)
        metrics['Asymmetry Index'] = elbow_asym

    if 'left_knee_angle_smooth' in df.columns and 'right_knee_angle_smooth' in df.columns:
        avg_left_knee = df['left_knee_angle_smooth'].mean()
        avg_right_knee = df['right_knee_angle_smooth'].mean()
        knee_asym = abs(avg_left_knee - avg_right_knee) / ((avg_left_knee + avg_right_knee)/2)
        # Combine elbow and knee asymmetry
        if 'Asymmetry Index' in metrics:
            metrics['Asymmetry Index'] = (metrics['Asymmetry Index'] + knee_asym) / 2
        else:
            metrics['Asymmetry Index'] = knee_asym

    # (4) Fluidity Score
    # Calculate the smoothness of movement by measuring variance in angular velocities
    angular_velocities = [
        'left_elbow_angle_vel',
        'right_elbow_angle_vel',
        'left_knee_angle_vel',
        'right_knee_angle_vel',
        'left_hip_flex_vel',
        'right_hip_flex_vel',
        'left_ankle_flex_vel',
        'right_ankle_flex_vel'
    ]
    vel_vars = []
    for vel in angular_velocities:
        if vel in df.columns and df[vel].notna().any():
            vel_vars.append(df[vel].var())
    if vel_vars:
        # Lower variance implies smoother movement
        # Normalize variance and invert for score
        norm_var = np.mean(vel_vars) if np.mean(vel_vars) != 0 else 1
        metrics['Fluidity Score'] = 1 - (np.mean(vel_vars) / (norm_var + 1e-6))  # Prevent division by zero
    else:
        metrics['Fluidity Score'] = np.nan

    # (5) Chain Range (frames)
    # Time between key events (e.g., from ankle to shoulder)
    angle_cols = {
        'ankle': 'left_ankle_flex_smooth',
        'knee':  'left_knee_angle_smooth',
        'hip':   'left_hip_flex_smooth',
        'shoulder': 'left_shoulder_elev_smooth'
    }
    sep_results = compute_time_based_separations(df, angle_cols=angle_cols)
    for key, val in sep_results.items():
        metrics[key] = val

    # (6) Kinematic Chain Peaks and Movement Sequence
    # Identify peaks in angular velocities as key events
    # Example: peaks in elbow and knee angular velocities
    # This can be expanded based on specific requirements
    # Here, we'll identify peaks in 'left_elbow_angle_vel_smooth' and similar
    kinematic_peaks = {}
    movement_sequence = []
    for vel in angular_velocities:
        if vel in df.columns:
            # Find the frame with maximum angular velocity
            peak_idx = df[vel].idxmax()
            peak_frame = df.loc[peak_idx, 'frame']
            kinematic_peaks[vel] = peak_frame
            movement_sequence.append(vel.replace('_vel','').replace('_',' ').title())

    metrics['Kinematic Chain Peaks'] = {
        'Peaks': kinematic_peaks,
        'Movement Sequence': movement_sequence
    }

    return metrics

# -------------------------------------------------------------------------------------
# KPI Display Function
# -------------------------------------------------------------------------------------
def display_kpis(df, metrics):
    """Displays angles, release height, angle, separation, etc."""
    st.markdown("### 🎯 Key Performance Indicators (KPIs)")

    def safe_avg(c):
        if c in df.columns and df[c].notna().any():
            return df[c].mean()
        return np.nan

    kpis = {
        "Avg Left Elbow Angle (°)": safe_avg('left_elbow_angle'),
        "Avg Right Elbow Angle (°)": safe_avg('right_elbow_angle'),
        "Avg Left Knee Angle (°)": safe_avg('left_knee_angle'),
        "Avg Right Knee Angle (°)": safe_avg('right_knee_angle'),
        "Relative Release Height": metrics.get('Relative Release Height', np.nan),
        "Release Angle (deg)": metrics.get('Release Angle (deg)', np.nan),
        "Asymmetry Index": metrics.get('Asymmetry Index', np.nan),
        "Fluidity Score": metrics.get('Fluidity Score', np.nan),
        "Chain Range (frames)": metrics.get('Chain Range (frames)', np.nan),
        "Pivotal Score": metrics.get('Pivotal Score', np.nan)
    }

    # Display KPIs in a grid
    num_cols = 4
    cols = st.columns(num_cols)
    for idx, (name, val) in enumerate(kpis.items()):
        col = cols[idx % num_cols]
        if isinstance(val, (int, float)) and not np.isnan(val):
            if "Angle" in name:
                disp = f"{val:.2f}°"
            elif "Separation" in name.lower():
                disp = f"{val:.2f} frames"
            elif "Height" in name:
                disp = f"{val:.3f} m"
            elif "Score" in name or "Index" in name:
                disp = f"{val:.2f}"
            elif "frames" in name.lower():
                disp = f"{int(val)} frames"
            else:
                disp = f"{val:.2f}"
        else:
            disp = "N/A"
        col.metric(name, disp)

    # Additional Warnings and Insights
    if 'Release Angle (deg)' in kpis and not pd.isna(kpis['Release Angle (deg)']):
        if kpis['Release Angle (deg)'] < 35 or kpis['Release Angle (deg)'] > 50:
            st.warning("Release Angle outside typical range (35–50°).")
    if 'Asymmetry Index' in kpis and not pd.isna(kpis['Asymmetry Index']):
        if kpis['Asymmetry Index'] > 0.1:
            st.warning("High asymmetry index – consider balancing training.")
    if 'Fluidity Score' in kpis and not pd.isna(kpis['Fluidity Score']):
        if kpis['Fluidity Score'] < 0.5:
            st.warning("Low fluidity score – focus on smooth movements.")
    if 'Pivotal Score' in kpis and not pd.isna(kpis['Pivotal Score']):
        if kpis['Pivotal Score'] < 0.6:
            st.warning("Pivotal Score is low – review shooting mechanics.")

# -------------------------------------------------------------------------------------
# Download Report Function
# -------------------------------------------------------------------------------------
def download_report(df, player_name, seg_name):
    """Generate a CSV download link for the DataFrame."""
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    fname = f"{player_name.replace(' ','_')}_{seg_name.replace(' ','_')}_report.csv"
    link = f'<a href="data:file/csv;base64,{b64}" download="{fname}">📥 Download CSV Report</a>'
    st.markdown(link, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------
# Main Application Function
# -------------------------------------------------------------------------------------
def main():
    # Display Brand Header Placeholder
    # The actual header will be displayed after user and job selection

    bucket_name = st.secrets["BUCKET_NAME"]
    st.sidebar.header("User & Job Selection")

    # Step 1: User Selection
    users = list_users(s3_client, bucket_name)
    if not users:
        st.sidebar.error("No users found in S3.")
        return
    selected_user = st.sidebar.selectbox("Select User (Email)", users)

    # Step 2: Select Job
    jobs = list_user_jobs(s3_client, bucket_name, selected_user)
    if not jobs:
        st.sidebar.error(f"No jobs found for {selected_user}.")
        return

    # Prepare labels
    job_labels = {}
    for j in jobs:
        meta = get_metadata_from_dynamodb(dynamodb, j)
        pname = meta.get('PlayerName', 'Unknown')
        tname = meta.get('Team', '').title() if meta.get('Team','').strip().upper() != "N/A" else ''
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
        job_labels[j] = label

    chosen_job = st.sidebar.selectbox("Select Job", jobs, format_func=lambda x: job_labels[x])

    # Step 3: Segment Selection
    segs = list_job_segments(s3_client, bucket_name, selected_user, chosen_job)
    if not segs:
        st.sidebar.error("No segments found in processed/... path for this job.")
        return
    multi_seg = st.sidebar.checkbox("Compare Multiple Segments?")
    if multi_seg:
        chosen_segs = st.sidebar.multiselect("Select Segments", segs)
    else:
        sel_seg = st.sidebar.selectbox("Select a Segment", segs)
        chosen_segs = [sel_seg]

    if not chosen_segs:
        st.warning("No segments selected.")
        return

    # Step 4: Load Data
    job_meta = get_metadata_from_dynamodb(dynamodb, chosen_job)
    player_name = job_meta.get('PlayerName', 'Unknown Player')
    team_name = job_meta.get('Team', 'N/A')

    # Show brand header now with player + team
    show_brand_header(player_name, team_name)

    user_data = {}
    for seg in chosen_segs:
        df = load_vibe_output(s3_client, bucket_name, selected_user, chosen_job, seg)
        if df.empty:
            st.warning(f"No data for segment '{seg}'.")
            continue

        # Merge job metadata
        for k, v in job_meta.items():
            df[k.lower()] = v

        # Map columns
        df = map_body_columns(df)

        # Ensure 'frame'
        if 'frame' not in df.columns:
            df['frame'] = np.arange(len(df))

        # Filter for shooter (if multiple player_ids)
        if 'player_id' in df.columns:
            df = filter_for_shooter_by_phases(df)
        else:
            st.warning(f"No 'player_id' in segment '{seg}', skipping shooter filter.")
            logger.warning(f"No 'player_id' in segment '{seg}', skipping shooter filter.")

        # Preprocessing & Offsetting
        df = offset_z_axis(df)

        # Compute joint angles
        df = compute_joint_angles(df)

        # Preprocess + smooth
        df = preprocess_and_smooth(df)

        user_data[seg] = df

    if not user_data:
        st.warning("No valid segment data loaded.")
        return

    # Combine data for aggregated analysis if needed
    combined_df = pd.DataFrame()
    for seg, d in user_data.items():
        d['segment_id'] = seg
        combined_df = pd.concat([combined_df, d], ignore_index=True)

    # Create Tabs
    t_overview, t_pose, t_adv, t_ball = st.tabs(["Overview", "Pose Analysis", "Advanced Biomechanics", "Ball Tracking"])

    # ---------------- OVERVIEW ----------------
    with t_overview:
        st.markdown("## Overview & Release Detection")

        # Detect release frame in combined data
        release_frame = detect_release_frame(combined_df)
        if release_frame is None:
            st.warning("Could not detect release frame in combined data.")
            logger.warning("Could not detect release frame in combined data.")
        else:
            st.write(f"**Detected Release Frame**: {release_frame}")

        # Trim data around release
        trimmed_agg = trim_shot(combined_df, release_frame, frames_before=30, frames_after=20)

        # Calculate advanced metrics on aggregated data
        player_ht = trimmed_agg['player_height'].iloc[0] if 'player_height' in trimmed_agg.columns else 2.0
        metrics = calculate_advanced_metrics(trimmed_agg, release_frame, player_height=player_ht)
        display_kpis(trimmed_agg, metrics)

        # Identify phases
        phases_5 = identify_phases(trimmed_agg, release_frame)
        trimmed_agg['phase'] = phases_5

        st.markdown("### Identified 5-Phase Model")
        phase_counts = trimmed_agg['phase'].value_counts(dropna=True).reset_index(name='Counts')
        phase_counts.rename(columns={'index':'phase'}, inplace=True)
        if not phase_counts.empty:
            fig_phases = px.bar(
                phase_counts, 
                x='phase', 
                y='Counts', 
                color='phase', 
                title="Distribution of Shooting Phases",
                labels={'phase': 'Phase', 'Counts': 'Number of Frames'}
            )
            st.plotly_chart(fig_phases, use_container_width=True)
        else:
            st.info("No phases recognized in the aggregated data.")

        # Plot Right Wrist Y (Smoothed) with Phase Overlays
        if 'right_wrist_y_smooth' in trimmed_agg.columns or 'right_wrist_y' in trimmed_agg.columns:
            if 'right_wrist_y_smooth' in trimmed_agg.columns:
                wrist_y_col = 'right_wrist_y_smooth'
            else:
                wrist_y_col = 'right_wrist_y'
                logger.warning("'right_wrist_y_smooth' not found. Using raw 'right_wrist_y' for plotting.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trimmed_agg['frame'],
                y=trimmed_agg[wrist_y_col],
                mode='lines+markers',
                name=f'Right Wrist Y ({wrist_y_col})',
                line=dict(color='blue')
            ))
            if release_frame is not None:
                fig.add_vline(
                    x=release_frame, 
                    line=dict(color='red', dash='dash'), 
                    annotation_text="Release"
                )
            # Visualize phases as shaded regions
            for phase in ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']:
                phase_df = trimmed_agg[trimmed_agg['phase'] == phase]
                if not phase_df.empty:
                    start = phase_df['frame'].min()
                    end = phase_df['frame'].max()
                    color = {
                        'Preparation': 'rgba(0,0,255,0.1)',        # Blue
                        'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                        'Stability': 'rgba(255,255,0,0.1)',        # Yellow
                        'Release': 'rgba(255,0,0,0.1)',           # Red
                        'Inertia': 'rgba(128,0,128,0.1)'          # Purple
                    }.get(phase, 'rgba(128,128,128,0.1)')        # Grey as default
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor=color,
                        opacity=0.5,
                        layer='below',
                        line_width=0,
                        annotation_text=phase
                    )
            fig.update_layout(
                title="Right Wrist Y Position (Smoothed) with Phases",
                xaxis_title="Frame",
                yaxis_title="Wrist Y Position"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No 'right_wrist_y_smooth' or 'right_wrist_y' column in trimmed aggregated data.")
            logger.warning("No 'right_wrist_y_smooth' or 'right_wrist_y' column in trimmed aggregated data.")

        # Display Trimmed Aggregated Data Head
        st.markdown("### Trimmed Aggregated Data (First 10 Rows)")
        st.dataframe(trimmed_agg.head(10))

    # ---------------- POSE ANALYSIS ----------------
    with t_pose:
        st.markdown("## Pose Analysis (Angles & Smoothing)")
        for seg, df in user_data.items():
            st.markdown(f"### Segment: {seg}")
            release_frame_seg = detect_release_frame(df)
            if release_frame_seg is not None:
                st.write(f"**Detected Release Frame**: {release_frame_seg}")
            else:
                st.warning("No release frame detected.")
                logger.warning(f"No release frame detected in segment '{seg}'.")

            # Trim data around release
            trimmed_df = trim_shot(df, release_frame_seg, frames_before=30, frames_after=20)

            # Calculate advanced metrics
            player_ht_seg = trimmed_df['player_height'].iloc[0] if 'player_height' in trimmed_df.columns else 2.0
            metrics_seg = calculate_advanced_metrics(trimmed_df, release_frame_seg, player_height=player_ht_seg)
            display_kpis(trimmed_df, metrics_seg)

            # Identify phases
            phases_seg = identify_phases(trimmed_df, release_frame_seg)
            trimmed_df['phase'] = phases_seg

            # Plot Key Joint Angles
            st.markdown("#### Key Joint Angles (Elbow, Knee, Hip, Ankle)")
            plot_joint_angles(trimmed_df, release_frame=release_frame_seg, selectbox_key=f"{seg}_angles")

            # Plot Shoulder-Hip Separation
            st.markdown("#### Shoulder-Hip Separation")
            plot_separation(trimmed_df, release_frame=release_frame_seg)

            # Plot Angular Velocities
            st.markdown("#### Angular Velocities of Key Joints")
            plot_angular_velocity(trimmed_df, release_frame=release_frame_seg)

            # Plot Angular Accelerations
            st.markdown("#### Angular Accelerations of Key Joints")
            plot_accelerations(trimmed_df, release_frame=release_frame_seg)

            # Plot Phases Distribution
            st.markdown("#### Shooting Phases Distribution")
            phase_counts_seg = trimmed_df['phase'].value_counts(dropna=True).reset_index(name='Counts')
            phase_counts_seg.rename(columns={'index':'phase'}, inplace=True)
            if not phase_counts_seg.empty:
                fig_phases_seg = px.bar(
                    phase_counts_seg,
                    x='phase',
                    y='Counts',
                    color='phase',
                    title=f"{seg} Phases",
                    labels={'phase': 'Phase', 'Counts': 'Number of Frames'}
                )
                st.plotly_chart(fig_phases_seg, use_container_width=True)
            else:
                st.warning("No phase data in this segment.")
                logger.warning(f"No phase data in segment '{seg}'.")

    # ---------------- ADVANCED BIOMECHANICS ----------------
    with t_adv:
        st.markdown("## Advanced Biomechanics")
        st.markdown("""
        - **Chain Range (frames)**
        - **Fluidity Score**
        - **Asymmetry Index**
        - **Pivotal Score**
        - **Heatmaps & Animations**
        - **Movement Sequence**
        """)
        for seg, df in user_data.items():
            st.markdown(f"### Segment: {seg}")
            release_frame_seg = detect_release_frame(df)
            if release_frame_seg is None:
                st.warning("No release frame detected => skipping.")
                logger.warning(f"No release frame detected in segment '{seg}'.")
                continue

            # Trim data around release
            trimmed_df = trim_shot(df, release_frame_seg, frames_before=30, frames_after=20)

            # Calculate advanced metrics
            player_ht_seg = trimmed_df['player_height'].iloc[0] if 'player_height' in trimmed_df.columns else 2.0
            metrics_seg = calculate_advanced_metrics(trimmed_df, release_frame_seg, player_height=player_ht_seg)
            display_kpis(trimmed_df, metrics_seg)

            # Display Advanced Metrics
            st.write("**Advanced Metrics at Release:**")
            if metrics_seg:
                # Convert dict to DataFrame for better display
                df_metrics = pd.DataFrame(metrics_seg, index=[0]).T
                df_metrics.columns = ["Value"]
                st.table(df_metrics)
            else:
                st.write("No advanced metrics available.")

            # Display Movement Sequence
            movement_seq = metrics_seg.get('Kinematic Chain Peaks', {}).get('Movement Sequence', [])
            if movement_seq:
                st.write("**Movement Sequence:**", ", ".join(movement_seq))
            else:
                st.write("**Movement Sequence:** N/A")

            # Heatmap of Joint Angles Correlation
            st.markdown("#### Heatmap of Joint Angles Correlation")
            visualize_heatmap(trimmed_df)

            # Animation of Joint Angles Over Time
            st.markdown("#### Animation of Joint Angles Over Time")
            visualize_animation(trimmed_df, segment=seg)

    # ---------------- BALL TRACKING ----------------
    with t_ball:
        st.markdown("## Ball Tracking & Trajectory")
        for seg, df in user_data.items():
            st.markdown(f"### Segment: {seg}")
            if not {'frame', 'x', 'y'}.issubset(df.columns):
                st.warning("No 'frame', 'x', 'y' columns => skipping ball trajectory.")
                logger.warning(f"No 'frame', 'x', 'y' columns in segment '{seg}'.")
                continue
            visualize_ball_trajectory_simple(df[['frame', 'x','y']].copy())

    # ---------------- DOWNLOAD REPORTS ----------------
    st.markdown("## Download CSV Reports")
    for seg, df in user_data.items():
        pname = df['player_name'].iloc[0] if 'player_name' in df.columns else "UnknownPlayer"
        download_report(df, pname, seg)

    # ---------------- COMPARISON FUNCTIONALITY ----------------
    if multi_seg and len(chosen_segs) == 2:
        seg1, seg2 = chosen_segs
        df1 = user_data.get(seg1)
        df2 = user_data.get(seg2)
        if df1 is not None and df2 is not None:
            compare_two_segments(seg1, df1, seg2, df2, player_height=player_ht)
        else:
            st.warning("One or both selected segments have no data for comparison.")

    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center; color:{BRAND_DARK};'>© 2024 Pivotal Motion. All rights reserved.</p>", 
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------------------------
# Advanced Visualization Functions
# -------------------------------------------------------------------------------------
# (Already defined above)

# -------------------------------------------------------------------------------------
# Main Application Entry
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
