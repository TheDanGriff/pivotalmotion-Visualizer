import streamlit as st
import boto3
import pandas as pd
import numpy as np
from io import StringIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import plotly.express as px
import plotly.graph_objects as go
import base64
import logging
import warnings
import statsmodels.api as sm
from plotly.subplots import make_subplots
import os
from math import degrees, atan2, sqrt
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.polynomial as poly
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from scipy.stats import zscore

###############################################################################
# Global Settings
###############################################################################
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

###############################################################################
# Streamlit Config
###############################################################################
st.set_page_config(
    page_title="Pivotal Motion Visualizer",
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

###############################################################################
# AWS Setup
###############################################################################
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

###############################################################################
# Additional Utility (Logos, etc.)
###############################################################################
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
        team_logo_path = os.path.join("images", "teams", team_logo_filename)  # Adjust path as needed
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
                if filename.startswith("segment_") and "_final_output.csv" in filename:
                    seg_id = filename.replace("_final_output.csv", "")
                    segs.add(seg_id)
        return sorted(segs)
    except Exception as e:
        st.error(f"Error listing segments for job '{job_id}': {e}")
        return []

def load_vibe_output(s3_client, bucket_name, user_email, job_id, shot_id):
    """
    Loads either *vibe_output.csv or *final_output.csv from S3.
    """
    prefix = f"processed/{user_email}/{job_id}/"
    possible_keys = [
        f"{prefix}{shot_id}_vibe_output.csv",
        f"{prefix}{shot_id}_final_output.csv"
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

###############################################################################
# Smoothing / Series Utility
###############################################################################
def smooth_series_custom(series, window=11, frac=0.3):
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
        st.warning(f"LOWESS smoothing failed: {e}. Using rolled mean instead.")
        return rolled

def compute_velocity(series, dt=0.01):
    """Basic centered finite difference to compute velocity from a series."""
    shifted_forward = series.shift(-1)
    shifted_back = series.shift(1)
    vel = (shifted_forward - shifted_back) / (2.0 * dt)
    return vel

def compute_acceleration(series, dt=0.05):
    """Compute acceleration from a velocity series."""
    return compute_velocity(series, dt=dt)

def compute_angular_velocity(df, angle_col, dt=0.05):
    """Compute angular velocity from an angle col, store in angle_col+'_vel_smooth'."""
    if angle_col not in df.columns:
        df[angle_col + '_vel'] = np.nan
        df[angle_col + '_vel_smooth'] = np.nan
        return df
    df[angle_col + '_vel'] = compute_velocity(df[angle_col], dt=dt)
    # Smooth the velocity
    df[angle_col + '_vel_smooth'] = smooth_series_custom(df[angle_col + '_vel'], window=11, frac=0.3)
    return df

def compute_angular_acceleration(df, angle_col, dt=0.05):
    """Compute angular acceleration from an angular velocity col, store in angle_col+'_accel_smooth'."""
    vel_col = angle_col + '_vel_smooth'
    accel_col = angle_col + '_accel_smooth'
    if vel_col not in df.columns:
        df[accel_col] = np.nan
        df[accel_col + '_smooth'] = np.nan
        return df
    df[accel_col] = compute_acceleration(df[vel_col], dt=dt)
    df[accel_col + '_smooth'] = smooth_series_custom(df[accel_col], window=11, frac=0.3)
    return df

###############################################################################
# Body Column Mapping
###############################################################################
def map_body_columns(df):
    """Remap raw columns to standard left_shoulder_x, right_elbow_angle, etc."""
    rename_dict = {
        # Example partial
        'leye_x': 'left_eye_x',
        'leye_y': 'left_eye_y',
        'leye_z': 'left_eye_z',
        'reye_x': 'right_eye_x',
        'reye_y': 'right_eye_y',
        'reye_z': 'right_eye_z',
        'neck_x': 'neck_x',
        'neck_y': 'neck_y',
        'neck_z': 'neck_z',
        'lsjc_x': 'left_shoulder_x',
        'lsjc_y': 'left_shoulder_y',
        'lsjc_z': 'left_shoulder_z',
        'lejc_x': 'left_elbow_x',
        'lejc_y': 'left_elbow_y',
        'lejc_z': 'left_elbow_z',
        'lwjc_x': 'left_wrist_x',
        'lwjc_y': 'left_wrist_y',
        'lwjc_z': 'left_wrist_z',
        'lpinky_x': 'left_pinky_x',
        'lpinky_y': 'left_pinky_y',
        'lpinky_z': 'left_pinky_z',
        'lthumb_x': 'left_thumb_x',
        'lthumb_y': 'left_thumb_y',
        'lthumb_z': 'left_thumb_z',
        'rsjc_x': 'right_shoulder_x',
        'rsjc_y': 'right_shoulder_y',
        'rsjc_z': 'right_shoulder_z',
        'rejc_x': 'right_elbow_x',
        'rejc_y': 'right_elbow_y',
        'rejc_z': 'right_elbow_z',
        'rwjc_x': 'right_wrist_x',
        'rwjc_y': 'right_wrist_y',
        'rwjc_z': 'right_wrist_z',
        'rpinky_x': 'right_pinky_x',
        'rpinky_y': 'right_pinky_y',
        'rpinky_z': 'right_pinky_z',
        'rthumb_x': 'right_thumb_x',
        'rthumb_y': 'right_thumb_y',
        'rthumb_z': 'right_thumb_z',
        'midhip_x': 'mid_hip_x',
        'midhip_y': 'mid_hip_y',
        'midhip_z': 'mid_hip_z',
        'lhjc_x': 'left_hip_x',
        'lhjc_y': 'left_hip_y',
        'lhjc_z': 'left_hip_z',
        'lkjc_x': 'left_knee_x',
        'lkjc_y': 'left_knee_y',
        'lkjc_z': 'left_knee_z',
        'lajc_x': 'left_ankle_x',
        'lajc_y': 'left_ankle_y',
        'lajc_z': 'left_ankle_z',
        'lheel_x': 'left_heel_x',
        'lheel_y': 'left_heel_y',
        'lheel_z': 'left_heel_z',
        'ltoe_x': 'left_toe_x',
        'ltoe_y': 'left_toe_y',
        'ltoe_z': 'left_toe_z',
        'rbtoe_x': 'right_big_toe_x',
        'rbtoe_y': 'right_big_toe_y',
        'rbtoe_z': 'right_big_toe_z',
        'rhjc_x': 'right_hip_x',
        'rhjc_y': 'right_hip_y',
        'rhjc_z': 'right_hip_z',
        'rkjc_x': 'right_knee_x',
        'rkjc_y': 'right_knee_y',
        'rkjc_z': 'right_knee_z',
        'rajc_x': 'right_ankle_x',
        'rajc_y': 'right_ankle_y',
        'rajc_z': 'right_ankle_z',
        'rheel_x': 'right_heel_x',
        'rheel_y': 'right_heel_y',
        'rheel_z': 'right_heel_z',
        'rtoe_x': 'right_toe_x',
        'rtoe_y': 'right_toe_y',
        'rtoe_z': 'right_toe_z',
        'left_sternoclavicular_elev': 'left_sternoclavicular_elev',
        'left_shoulder_plane': 'left_shoulder_plane',
        'left_shoulder_elev': 'left_shoulder_elev',
        'left_shoulder_rot': 'left_shoulder_rot',
        'right_sternoclavicular_elev': 'right_sternoclavicular_elev',
        'right_shoulder_plane': 'right_shoulder_plane',
        'right_shoulder_elev': 'right_shoulder_elev',
        'right_shoulder_rot': 'right_shoulder_rot',
        'left_elbow_angle': 'left_elbow_angle',
        'left_forearm_pro': 'left_forearm_pro',
        'left_wrist_dev': 'left_wrist_dev',
        'left_wrist_flex': 'left_wrist_flex',
        'right_elbow_angle': 'right_elbow_angle',
        'right_forearm_pro': 'right_forearm_pro',
        'right_wrist_dev': 'right_wrist_dev',
        'right_wrist_flex': 'right_wrist_flex',
        'torso_ext': 'torso_ext',
        'torso_side': 'torso_side',
        'torso_rot': 'torso_rot',
        'pelvis_rot': 'pelvis_rot',
        'pelvis_side': 'pelvis_side',
        'left_hip_flex': 'left_hip_flex',
        'left_hip_rot': 'left_hip_rot',
        'left_knee': 'left_knee_angle',
        'left_ankle_inv': 'left_ankle_inv',
        'left_ankle_flex': 'left_ankle_flex',
        'right_hip_flex': 'right_hip_flex',
        'right_hip_rot': 'right_hip_rot',
        'right_knee': 'right_knee_angle',
        'right_ankle_inv': 'right_ankle_inv',
        'right_ankle_flex': 'right_ankle_flex',
        # Add more mappings as needed
    }
    df = df.rename(columns=rename_dict)
    return df

###############################################################################
# Joint Angle Computation (2D)
###############################################################################
import numpy as np
import pandas as pd
from math import degrees, acos, isfinite

def find_ball_in_hand_frame(df):
    """
    Uses x_smooth of the ball to find the furthest left point.
    Returns the frame index (int) or None if not found.
    """
    if {'x_smooth','frame'}.issubset(df.columns):
        min_x_idx = df['x_smooth'].idxmin()
        return df.loc[min_x_idx, 'frame']
    return None


def angle_2d(a, b, c):
    """
    Compute the angle at point b given three points a, b, and c in 2D space.

    Parameters:
    -----------
    a, b, c : tuple
        Each tuple contains the (x, y) coordinates of the respective point.

    Returns:
    --------
    float
        The angle at point b in degrees. Returns np.nan if calculation is not possible.
    """
    try:
        ab = np.array([a[0] - b[0], a[1] - b[1]])
        cb = np.array([c[0] - b[0], c[1] - b[1]])
        ab_norm = np.linalg.norm(ab)
        cb_norm = np.linalg.norm(cb)
        if ab_norm == 0 or cb_norm == 0:
            return np.nan
        cos_theta = np.dot(ab, cb) / (ab_norm * cb_norm)
        # Clamp the cosine to the valid range to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return degrees(acos(cos_theta))
    except Exception as e:
        print(f"Error computing angle at point b: {e}")
        return np.nan

def compute_joint_angles(df):
    """
    Compute various joint angles from raw 2D coordinates.

    The function calculates angles for elbows, wrists, shoulders, hips, knees, ankles, and more.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the 2D coordinates of body markers. Expected columns should follow
        the naming convention: '<body_part>_<axis>', e.g., 'left_shoulder_x', 'right_knee_y'.

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame augmented with new columns for each computed joint angle.
    """
    # Define all joints with their corresponding markers
    joints = [
        # Elbow Angles
        {
            'name': 'left_elbow_angle',
            'marker_a': 'left_shoulder',
            'marker_b': 'left_elbow',
            'marker_c': 'left_wrist'
        },
        {
            'name': 'right_elbow_angle',
            'marker_a': 'right_shoulder',
            'marker_b': 'right_elbow',
            'marker_c': 'right_wrist'
        },
        # Shoulder Angles
        {
            'name': 'left_shoulder_angle',
            'marker_a': 'left_ear',  # Assuming 'left_ear' exists; replace with appropriate marker
            'marker_b': 'left_shoulder',
            'marker_c': 'left_elbow'
        },
        {
            'name': 'right_shoulder_angle',
            'marker_a': 'right_ear',  # Assuming 'right_ear' exists; replace with appropriate marker
            'marker_b': 'right_shoulder',
            'marker_c': 'right_elbow'
        },
        # Wrist Angles
        {
            'name': 'left_wrist_angle',
            'marker_a': 'left_elbow',
            'marker_b': 'left_wrist',
            'marker_c': 'left_pinky'  # Assuming 'left_pinky' exists; replace if necessary
        },
        {
            'name': 'right_wrist_angle',
            'marker_a': 'right_elbow',
            'marker_b': 'right_wrist',
            'marker_c': 'right_pinky'  # Assuming 'right_pinky' exists; replace if necessary
        },
        # Hip Angles
        {
            'name': 'left_hip_angle',
            'marker_a': 'left_shoulder',
            'marker_b': 'left_hip',
            'marker_c': 'left_knee'
        },
        {
            'name': 'right_hip_angle',
            'marker_a': 'right_shoulder',
            'marker_b': 'right_hip',
            'marker_c': 'right_knee'
        },
        # Knee Angles
        {
            'name': 'left_knee_angle',
            'marker_a': 'left_hip',
            'marker_b': 'left_knee',
            'marker_c': 'left_ankle'
        },
        {
            'name': 'right_knee_angle',
            'marker_a': 'right_hip',
            'marker_b': 'right_knee',
            'marker_c': 'right_ankle'
        },
        # Ankle Angles
        {
            'name': 'left_ankle_angle',
            'marker_a': 'left_knee',
            'marker_b': 'left_ankle',
            'marker_c': 'left_heel'  # Assuming 'left_heel' exists; replace if necessary
        },
        {
            'name': 'right_ankle_angle',
            'marker_a': 'right_knee',
            'marker_b': 'right_ankle',
            'marker_c': 'right_heel'  # Assuming 'right_heel' exists; replace if necessary
        },
        # Additional Joints (e.g., Neck, Spine)
        {
            'name': 'neck_angle',
            'marker_a': 'mid_hip',
            'marker_b': 'neck',
            'marker_c': 'left_shoulder'  # Or 'right_shoulder'; adjust as needed
        },
        {
            'name': 'spine_angle',
            'marker_a': 'mid_hip',
            'marker_b': 'neck',
            'marker_c': 'head'  # Assuming 'head' exists; replace if necessary
        },
        # Add more joints as needed
    ]

    for joint in joints:
        angle_col = joint['name']
        marker_a = joint['marker_a']
        marker_b = joint['marker_b']
        marker_c = joint['marker_c']

        required_cols = [f"{marker_a}_x", f"{marker_a}_y",
                         f"{marker_b}_x", f"{marker_b}_y",
                         f"{marker_c}_x", f"{marker_c}_y"]

        if all(col in df.columns for col in required_cols):
            # Compute angle using the three markers
            df[angle_col] = df.apply(
                lambda row: angle_2d(
                    (row[f"{marker_a}_x"], row[f"{marker_a}_y"]),
                    (row[f"{marker_b}_x"], row[f"{marker_b}_y"]),
                    (row[f"{marker_c}_x"], row[f"{marker_c}_y"])
                ),
                axis=1
            )
        else:
            # If any required column is missing, assign NaN and issue a warning
            df[angle_col] = np.nan
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Warning: Missing columns for {angle_col}: {missing}")

    return df

def mark_biggest_x_increase(fig, df, x_col='frame', y_col='x_smooth'):
    """
    Finds the index where x_smooth changes the most from one frame to the next,
    and draws a vertical shading or annotation.
    """
    # compute dx
    dx = df[y_col].diff().abs()  # absolute change
    if dx.isnull().all():
        return fig  # can't do anything

    max_idx = dx.idxmax()  # row index with largest change
    if pd.isna(max_idx):
        return fig

    # We can get the 'frame' at that point
    frame_val = df.loc[max_idx, x_col]

    # Optionally shade a small region from frame_val-0.5 to frame_val+0.5
    left = frame_val - 0.5
    right = frame_val + 0.5

    fig.add_vrect(
        x0=left, x1=right,
        fillcolor="red", 
        opacity=0.2,
        layer="below", 
        line_width=0
    )
    fig.add_annotation(
        x=frame_val,
        y=df[y_col].max(),
        text="Max X Speed",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        ax=0,
        ay=-40
    )
    return fig

###############################################################################
# Release Detection
###############################################################################
def detect_release_frame(df):
    """Find frame with max right_wrist_y_smooth (fallback to right_wrist_y)."""
    if 'right_wrist_y_smooth' in df.columns:
        col = 'right_wrist_y_smooth'
    elif 'right_wrist_y' in df.columns:
        col = 'right_wrist_y'
    else:
        st.warning("No right wrist Y data for release detection.")
        return None

    valid_df = df.dropna(subset=[col])
    if valid_df.empty:
        st.warning(f"No valid data in '{col}' for release detection.")
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

###############################################################################
# 5-Phase Model 
###############################################################################
def approximate_ball_speed_near_release(df, release_frame, frames_around=2, fps_ball=20):
    """
    Approximate the ball's speed using x_smooth, y_smooth around release_frame.

    :param df: DataFrame with 'frame','x_smooth','y_smooth' columns
    :param release_frame: release frame number
    :param frames_around: number of frames before/after to consider
    :param fps_ball: Frames per second of the ball tracking system
    :return: estimated speed (float) or 0.0 if insufficient data
    """
    required_cols = {'frame','x_smooth','y_smooth'}
    if not required_cols.issubset(df.columns):
        return 0.0

    # Slice frames around release_frame
    min_f = release_frame - frames_around
    max_f = release_frame + frames_around
    sub = df[(df['frame'] >= min_f) & (df['frame'] <= max_f)].copy()
    if len(sub) < 2:
        return 0.0

    # Distance traveled from the first to the last in sub
    sub = sub.sort_values('frame')
    x1, y1 = sub.iloc[0]['x_smooth'], sub.iloc[0]['y_smooth']
    x2, y2 = sub.iloc[-1]['x_smooth'], sub.iloc[-1]['y_smooth']
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Time in seconds
    frames_count = sub.iloc[-1]['frame'] - sub.iloc[0]['frame']
    if frames_count <= 0:
        return 0.0

    dt = frames_count / fps_ball  # Adjust fps_ball as per your data
    speed = dist/dt if dt > 0 else 0.0
    return speed

###################################
# identify_biomech_phases
###################################
def identify_biomech_phases(df, fps=25):
    """
    Identifies phases in [start_idx..(release_idx+margin)], ignoring frames outside that range.

    PHASES:
      - Unknown (frames outside the valid range)
      - Preparation
      - Release (~2 frames around release frame)
      - Inertia (after release_end)
      - (Optional placeholders for Ball Elevation / Stability not fully implemented here)
    """
    # 0) If too short, return
    if len(df) < 5:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    phases = pd.Series(["Unknown"] * len(df), index=df.index)

    # 1) Find "ball in hand" start (min x_smooth)
    start_idx = find_ball_in_hand_frame(df)
    if start_idx is None:
        # fallback to the first frame
        start_idx = df['frame'].iloc[0]

    # 2) Find release idx
    release_idx = detect_release_frame(df)
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
    # We'll use right_knee_angle_smooth (or left) as an example
    knee_col = "right_knee_angle_smooth"
    if knee_col not in valid_df.columns:
        # fallback => everything in valid range is "Preparation"
        phases[valid_range_mask] = "Preparation"
        return phases

    baseline_knee = valid_df[knee_col].iloc[:10].mean()  # average of first ~10 frames in valid_df
    knee_threshold = baseline_knee - 10.0
    transitions = valid_df.index[valid_df[knee_col] < knee_threshold]

    if not transitions.empty:
        prep_end_idx = transitions.min()
    else:
        prep_end_idx = valid_idx[0]  # no real crouch => trivial

    # Mark from start_loc..prep_end_idx as Preparation
    # (First find the actual row index for start_idx in valid_df)
    # This might be tricky if start_idx doesn't map exactly.
    start_row = valid_df.index[valid_df['frame'] == start_idx]
    if not start_row.empty:
        start_loc = start_row[0]
    else:
        start_loc = valid_idx[0]

    # Label them
    phases.loc[start_loc:prep_end_idx] = "Preparation"

        # 4. Ball Elevation Phase
    # Criteria: Wrist is moving upwards (y_smooth increasing)
    # Assuming higher y_smooth means higher position
    if 'y_smooth' in df.columns:
        ball_elevation_mask = (df['frame'] > prep_end_idx) & (df['frame'] < release_idx)
        phases[ball_elevation_mask] = "Ball Elevation"

    # 5. Release Phase
    if release_idx:
        release_mask = (df['frame'] >= release_idx) & (df['frame'] <= (release_idx + 2))
        phases[release_mask] = "Release"

    # Mark frames from release_row_idx..(release_row_idx+2) as "Release"
    # We'll do it by frame matching:
    release_mask = (df['frame'] >= release_idx) & (df['frame'] <= release_idx)
    phases[release_mask] = "Release"

    # (d) Inertia => after release_end
    inertia_mask = df['frame'] > release_idx
    phases.loc[inertia_mask] = "Inertia"

    return phases


def compute_elbow_range(df):
    """
    Compute the total range of the elbow angle (e.g. right elbow) as a proxy for "shooting motion".
    If you want to incorporate the left elbow angle, do so or pick the max range of either.
    """
    # Prefer the smoothed angle if available
    if 'right_elbow_angle_smooth' in df.columns:
        valid_vals = df['right_elbow_angle_smooth'].dropna()
        if not valid_vals.empty:
            return valid_vals.max() - valid_vals.min()
    elif 'right_elbow_angle' in df.columns:
        valid_vals = df['right_elbow_angle'].dropna()
        if not valid_vals.empty:
            return valid_vals.max() - valid_vals.min()
    return 0.0


def compute_dynamic_range(df):
    """(max-min) for key angles + wrist_y as a fallback measure."""
    angle_cols = [
        'right_knee_angle', 'left_knee_angle',
        'right_elbow_angle', 'left_elbow_angle',
        'right_hip_flex', 'left_hip_flex'
    ]
    score = 0.0
    for ac in angle_cols:
        if ac in df.columns and df[ac].notna().any():
            score += (df[ac].max() - df[ac].min())

    if 'right_wrist_y_smooth' in df.columns:
        score += df['right_wrist_y_smooth'].max() - df['right_wrist_y_smooth'].min()
    elif 'right_wrist_y' in df.columns:
        score += df['right_wrist_y'].max() - df['right_wrist_y'].min()
    return score

def identify_phases(df, release_frame, fps_body=25, fps_ball=20, **kwargs):
    """
    Simple wrapper to call your new function so old code doesn't break.
    release_frame is ignored in this example, because the new function
    does its own internal detection. But you could pass it in if needed.
    """
    phases_series = identify_biomech_phases(df, fps=fps_body)
    return phases_series

def filter_for_shooter_by_phases(df):
    """
    Attempts to identify which 'player_id' is the true shooter
    by checking how many of the main phases are present:
      (Preparation, Ball Elevation, Stability, Release, Inertia)
    plus checking elbow range and ball speed near release for scoring.

    Returns the DataFrame filtered to the best candidate shooter.
    """
    if 'player_id' not in df.columns:
        st.warning("No 'player_id' column; cannot filter shooter.")
        return df

    expected_phases = ['Preparation','Ball Elevation','Stability','Release','Inertia']
    candidates = []

    # Group by player
    for pid, sub_df in df.groupby('player_id'):
        # 1) Detect phases
        sub_df['phase'] = identify_biomech_phases(sub_df, fps=25)

        # 2) Count how many of the expected phases are found
        found_phases = set(sub_df['phase'].dropna().unique())
        matched_count = sum(1 for p in expected_phases if p in found_phases)

        # 3) Elbow range
        elbow_range = compute_elbow_range(sub_df)

        # 4) Velocity near release
        rel_frame = detect_release_frame(sub_df)
        if rel_frame is None:
            ball_speed_near_rel = 0.0
        else:
            ball_speed_near_rel = approximate_ball_speed_near_release(sub_df, rel_frame, frames_around=2)
            if np.isnan(ball_speed_near_rel):
                ball_speed_near_rel = 0.0

        # 5) Overall score
        overall_score = matched_count + (elbow_range / 20.0) + (ball_speed_near_rel / 2.0)

        candidates.append((pid, matched_count, elbow_range, ball_speed_near_rel, overall_score))

    if not candidates:
        return df

    # Sort by overall_score descending
    candidates.sort(key=lambda x: x[4], reverse=True)
    best_pid = candidates[0][0]
    st.write("Shooter identification debug:", candidates)

    # Return only the best candidate's data
    return df[df['player_id'] == best_pid]


def remove_outliers_and_smooth_ball(df, x_col='x', y_col='y', zscore_thresh=3.0,
                                    sg_window=21, sg_poly=3,
                                    rolling_window=7):
    """
    Removes outliers from ball (x, y) using Z-score, then applies a two-stage smoothing:
      1) Savitzky-Golay (window=sg_window, polyorder=sg_poly)
      2) Rolling mean (window=rolling_window)

    :param df: DataFrame with columns [x_col, y_col].
    :param x_col: Ball X column name, default 'x'.
    :param y_col: Ball Y column name, default 'y'.
    :param zscore_thresh: Z-score threshold for outlier removal (e.g. 3.0).
    :param sg_window: Window size for Savitzky-Golay filter (must be odd).
    :param sg_poly: Polynomial order for Savitzky-Golay filter.
    :param rolling_window: Window for secondary rolling mean.
    :return: df with new columns [x_smooth, y_smooth] (heavily smoothed).
    """
    # 1) Drop rows if x or y is missing
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        st.warning("No valid ball data to smooth after dropping NaNs.")
        df['x_smooth'] = np.nan
        df['y_smooth'] = np.nan
        return df

    # 2) Compute Z-scores for outlier removal
    df['x_z'] = zscore(df[x_col])
    df['y_z'] = zscore(df[y_col])

    # Keep rows where |zscore| < threshold
    df = df[(df['x_z'].abs() < zscore_thresh) & (df['y_z'].abs() < zscore_thresh)].copy()

    # Drop helper columns
    df.drop(columns=['x_z','y_z'], inplace=True)

    if df.empty:
        st.warning("All ball data removed by outlier filter.")
        df['x_smooth'] = np.nan
        df['y_smooth'] = np.nan
        return df

    # 3) Sort by frame if available
    if 'frame' in df.columns:
        df.sort_values('frame', inplace=True)

    # 4) First pass: Savitzky-Golay
    x_vals = df[x_col].values
    y_vals = df[y_col].values

    # If data too short for SG filter, fallback to rolling
    def sg_or_rolling(series_vals):
        if len(series_vals) < sg_window:
            return pd.Series(series_vals).rolling(sg_window, center=True, min_periods=1).mean().values
        else:
            return savgol_filter(series_vals, window_length=sg_window, polyorder=sg_poly)

    x_sg = sg_or_rolling(x_vals)
    y_sg = sg_or_rolling(y_vals)

    # 5) Second pass: Rolling mean
    x_df = pd.Series(x_sg).rolling(rolling_window, center=True, min_periods=1).mean()
    y_df = pd.Series(y_sg).rolling(rolling_window, center=True, min_periods=1).mean()

    # 6) Store final results
    df['x_smooth'] = x_df.values
    df['y_smooth'] = y_df.values

    return df



###############################################################################
# Advanced Metric Calculation
###############################################################################
def compute_cohesion_score(df, angle_cols=None, consider_chain=False, chain_cols=None):
    """
    Compute a Cohesion Score to represent consistency in a set of angle profiles.

    Basic Approach (angle standard deviation):
    - Lower standard deviation => higher cohesion.
    - We sum the std for each angle column => total_std.
    - We map that to a 0..1 range using a chosen 'max_std' threshold.

    Extended Approach (if consider_chain=True):
    - Also factor in the timing of peak velocities or angles in chain_cols.
    - If the chain sequence has consistent relative timing,
      we boost the cohesion score. More variability => lower score.

    Parameters:
    -----------
    df : pandas DataFrame
        Should contain columns for smoothed angles, e.g. 'left_elbow_angle_smooth'.
    angle_cols : list of str
        Columns used for the standard-deviation-based component.
        If None, we pick some defaults (elbows, knees, etc.).
    consider_chain : bool
        If True, also factor in a kinematic chain analysis (peak velocity times).
    chain_cols : list of str
        Columns that store velocity or angle for chain analysis,
        e.g. ['left_hip_vel', 'left_knee_vel', 'left_ankle_vel'].

    Returns:
    --------
    cohesion_score : float in [0, 1]
        A higher score indicates better "cohesion" or consistency.
    """

    # 1) Basic standard-deviation approach
    if angle_cols is None:
        angle_cols = [
            'right_elbow_angle_smooth',
            'right_knee_angle_smooth',
            'left_elbow_angle_smooth',
            'left_knee_angle_smooth',
            # Add more if needed
        ]
    valid_angle_cols = [c for c in angle_cols if c in df.columns]
    if not valid_angle_cols:
        # No valid columns to analyze
        base_score = np.nan
    else:
        # Sum of standard deviations
        total_std = df[valid_angle_cols].std().sum()
        max_std = 30.0  # domain-based guess or calibrate from your data
        base_score = 1.0 - (total_std / max_std)
        base_score = float(np.clip(base_score, 0, 1))

    if not consider_chain:
        # Just return the base score
        return base_score

    # 2) Kinematic chain approach: 
    #    - We measure how consistent the ordering of peak velocities is across frames.
    #    - For each velocity column, find the frame of max velocity.
    #    - Then measure the standard deviation of the *differences* between successive peak frames.
    if chain_cols is None:
        chain_cols = [
            'left_elbow_angle_vel_smooth',
            'right_elbow_angle_vel_smooth',
            'left_knee_angle_vel_smooth',
            'right_knee_angle_vel_smooth'
            # etc. re-order for your chain logic
        ]
    valid_chain_cols = [c for c in chain_cols if c in df.columns]
    if len(valid_chain_cols) < 2:
        # Not enough chain columns to analyze
        chain_score = 1.0  # fallback: no penalty
    else:
        # Find peak frames
        peak_frames = []
        for c in valid_chain_cols:
            # Index of max absolute velocity for each chain segment
            arr = df[c].values
            peak_idx = np.argmax(np.abs(arr))
            peak_frames.append(peak_idx)

        # Now compute differences in peak frames
        diffs = []
        for i in range(len(peak_frames) - 1):
            diffs.append(abs(peak_frames[i+1] - peak_frames[i]))

        if len(diffs) == 0:
            chain_score = 1.0
        else:
            # standard deviation of these time differences
            sd_diffs = np.std(diffs)
            # pick a max allowable std, e.g. 10 frames
            max_chain_sd = 10.0
            chain_score = 1.0 - (sd_diffs / max_chain_sd)
            chain_score = float(np.clip(chain_score, 0, 1))

    # Combine base_score (angles) and chain_score (timing) 
    # Weighted average approach: you can tweak weighting
    w1, w2 = 0.5, 0.5
    if np.isnan(base_score):
        cohesion_score = chain_score
    else:
        cohesion_score = w1 * base_score + w2 * chain_score

    # Clip final result
    return float(np.clip(cohesion_score, 0, 1))

def compute_kinetic_chain_efficiency(df, release_frame):
    """
    Example: sum of angular velocities of elbow/knee at the release frame
    or difference from a reference.
    """
    if release_frame not in df['frame'].values:
        return np.nan
    row = df.loc[df['frame'] == release_frame].iloc[0]

    angles_of_interest = [
        'left_elbow_angle_vel_smooth', 'right_elbow_angle_vel_smooth',
        'left_knee_angle_vel_smooth', 'right_knee_angle_vel_smooth'
    ]
    sum_vel = 0.0
    for col in angles_of_interest:
        if col in df.columns and not pd.isna(row[col]):
            sum_vel += abs(row[col])
    return sum_vel  # or some scaled formula

def compute_ball_trajectory_curvature(df, release_frame):
    """
    Compute the curvature of the ball's trajectory at the release frame.

    Parameters:
    - df: DataFrame containing ball x and y positions.
    - release_frame: Frame number where release occurs.

    Returns:
    - curvature: Curvature at the release frame, or np.nan if calculation isn't possible.
    """
    try:
        # Ensure required columns exist
        if not {'frame', 'x_smooth', 'y_smooth'}.issubset(df.columns):
            return np.nan

        # Find the index of the release frame
        release_idx = df.index[df['frame'] == release_frame].tolist()
        if not release_idx:
            return np.nan
        release_idx = release_idx[0]

        # Need at least one frame before and after for derivatives
        if release_idx == 0 or release_idx == len(df) -1:
            return np.nan

        # Calculate first derivatives (velocities)
        x1 = df.iloc[release_idx -1]['x_smooth']
        y1 = df.iloc[release_idx -1]['y_smooth']
        x2 = df.iloc[release_idx]['x_smooth']
        y2 = df.iloc[release_idx]['y_smooth']
        x3 = df.iloc[release_idx +1]['x_smooth']
        y3 = df.iloc[release_idx +1]['y_smooth']

        dt = 1.0 / 20  # Assuming 20 fps

        # First derivatives
        dx_dt = (x3 - x1) / (2 * dt)
        dy_dt = (y3 - y1) / (2 * dt)

        # Second derivatives
        d2x_dt2 = (x3 - 2*x2 + x1) / (dt **2)
        d2y_dt2 = (y3 - 2*y2 + y1) / (dt **2)

        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2) ** 1.5

        if denominator == 0:
            return np.nan

        curvature = numerator / denominator
        return curvature
    except Exception as e:
        logger.error(f"Error computing ball curvature: {e}")
        return np.nan

def calculate_advanced_metrics_reworked(df, release_frame, player_height=2.0, dt=0.05, fps=20):
    """
    Reworked function to compute various advanced biomechanical metrics at the release frame.
    """
    metrics = {}
    if release_frame is None or 'frame' not in df.columns or release_frame not in df['frame'].values:
        return metrics

    row = df.loc[df['frame'] == release_frame].iloc[0]

    # Release Height
    if 'right_wrist_z_smooth' in df.columns and not pd.isna(row['right_wrist_z_smooth']):
        metrics['Release Height (m)'] = abs(row['right_wrist_z_smooth'])
    elif 'right_wrist_z' in df.columns and not pd.isna(row['right_wrist_z']):
        metrics['Release Height (m)'] = abs(row['right_wrist_z'])
    else:
        metrics['Release Height (m)'] = np.nan

    # Ball Speed at release (already computed as ball_speed_smooth)
    if 'ball_speed_smooth' in df.columns and not pd.isna(row['ball_speed_smooth']):
        metrics['Release Speed (m/s)'] = float(row['ball_speed_smooth'])
    else:
        metrics['Release Speed (m/s)'] = np.nan

    # Asymmetry Index (Added Back)
    try:
        le = row.get('left_elbow_angle', np.nan)
        re = row.get('right_elbow_angle', np.nan)
        lk = row.get('left_knee_angle', np.nan)
        rk = row.get('right_knee_angle', np.nan)
        ed = abs(le - re)
        kd = abs(lk - rk)
        eN = ed / max(abs(le), abs(re), 1)
        kN = kd / max(abs(lk), abs(rk), 1)
        ai = np.nanmean([eN, kN])
        metrics['Asymmetry Index'] = ai
    except:
        metrics['Asymmetry Index'] = np.nan

    # Cohesion Score
    metrics['Cohesion Score'] = compute_cohesion_score(df)

    # Time from Ball Elevation to Release 
    # (assuming we identified phases in df['phase'])
    phases = df['phase'].fillna('').tolist()
    try:
        elev_start_idx = phases.index('Ball Elevation')
        inertia_idx = phases.index('Inertia')
        time_frames = inertia_idx - elev_start_idx
        time_seconds = time_frames * (1.0 / 25)  # Assuming 25 fps
        metrics['Time Elevation to Release (s)'] = time_seconds
    except ValueError:
        metrics['Time Elevation to Release (s)'] = np.nan

    # Kinetic Chain Efficiency
    metrics['Kinetic Chain Efficiency'] = compute_kinetic_chain_efficiency(df, release_frame)

    # Center of Mass Speed (Added Back)
    if 'com_speed' in df.columns and not pd.isna(row['com_speed']):
        metrics['CoM Speed (m/s)'] = row['com_speed']
    else:
        metrics['CoM Speed (m/s)'] = np.nan

    # Ball Trajectory Curvature (New Metric)
    metrics['Ball Trajectory Curvature'] = compute_ball_trajectory_curvature(df, release_frame)

    # Additional Metrics
    # Release Velocity (max velocity)
    metrics['Release Velocity'] = row.get('ball_speed_smooth', np.nan)

    # Release Angle (velocity vector angle)
    if 'vx' in df.columns and 'vy' in df.columns and not pd.isna(row['vx']) and not pd.isna(row['vy']):
        angle_radians = math.atan2(row['vy'], row['vx'])  # relative to +x axis
        angle_degrees = math.degrees(angle_radians)
        metrics['Release Angle (°)'] = angle_degrees
    else:
        metrics['Release Angle (°)'] = np.nan

    # Shot Distance (distance from rim at max velocity point)
    rim_x = 1000.0  # Adjust based on your coordinate system
    rim_y = 300.0
    if 'x_smooth' in df.columns and 'y_smooth' in df.columns:
        bx = row['x_smooth']
        by = row['y_smooth']
        dist = math.sqrt((rim_x - bx)**2 + (rim_y - by)**2)
        metrics['Shot Distance'] = dist
    else:
        metrics['Shot Distance'] = np.nan

    # Apex Height (highest y_smooth after release)
    df_after = df[df['frame'] >= release_frame]
    if not df_after.empty and 'y_smooth' in df.columns:
        apex_height = float(df_after['y_smooth'].max())
        metrics['Apex Height'] = apex_height
    else:
        metrics['Apex Height'] = np.nan

    # Release Time (time from start of shot until max velocity)
    first_frame = df['frame'].min()
    release_frame_num = release_frame
    dt_frames = release_frame_num - first_frame
    metrics['Release Time (s)'] = dt_frames / fps

    # Release Quality (placeholder for spin-based metric)
    if "spin_x" in df.columns:
        # Example: average spin_x near max_vel_idx +/- 2 frames
        rng = range(release_frame-2, release_frame+3)
        sub = df.loc[rng, "spin_x"] if set(rng).issubset(df.index) else df["spin_x"]
        metrics["Release Quality"] = float(sub.mean())
    else:
        metrics["Release Quality"] = np.nan

    # Release Consistency (std. dev. of Release Quality across multiple shots)
    # Placeholder: Calculate if multiple shots are present
    # This requires aggregating data across shots, which isn't shown here
    metrics["Release Consistency"] = np.nan  # To be implemented based on data

    # Release Curvature and Lateral Release Curvature
    if 'curvature' in df.columns:
        # Average curvature near release
        curvature_window = 5
        curvature_subset = df.iloc[release_frame - curvature_window : release_frame + curvature_window]['curvature']
        metrics["Release Curvature"] = float(curvature_subset.mean())
    else:
        metrics["Release Curvature"] = np.nan

    # Lateral Release Curvature (Placeholder)
    metrics["Lateral Release Curvature"] = np.nan  # To be implemented based on data

    return metrics

def compute_center_of_mass(df, frame_idx, segment_markers, segment_masses):
    """
    Computes the Center of Mass (CoM) for a given frame using segment markers and their masses.

    Parameters:
    - df: DataFrame containing marker positions.
    - frame_idx: Integer index representing the frame's position in the DataFrame.
    - segment_markers: dict of segment_name -> (markerA, markerB)
    - segment_masses: dict of segment_name -> mass_value

    Returns:
    - com_xyz: NumPy array representing the CoM coordinates (x, y, z).
    """
    total_mass = sum(segment_masses.values())

    COM_sum = np.zeros(3, dtype=float)
    for seg_name, (markerA, markerB) in segment_markers.items():
        # Check if all required columns exist
        required_cols = [markerA+'_x', markerA+'_y', markerA+'_z',
                        markerB+'_x', markerB+'_y', markerB+'_z']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing columns for segment '{seg_name}'. Skipping this segment.")
            continue

        # Use positional indexing with iloc
        try:
            A_x = df.iloc[frame_idx][markerA+'_x']
            A_y = df.iloc[frame_idx][markerA+'_y']
            A_z = df.iloc[frame_idx][markerA+'_z']
            B_x = df.iloc[frame_idx][markerB+'_x']
            B_y = df.iloc[frame_idx][markerB+'_y']
            B_z = df.iloc[frame_idx][markerB+'_z']
        except IndexError:
            logger.warning(f"Frame index {frame_idx} is out of bounds for the DataFrame.")
            return np.array([np.nan, np.nan, np.nan])

        # Handle possible NaN values
        if pd.isna(A_x) or pd.isna(A_y) or pd.isna(A_z) or pd.isna(B_x) or pd.isna(B_y) or pd.isna(B_z):
            logger.warning(f"NaN values detected in markers for segment '{seg_name}' at frame {frame_idx}. Skipping this segment.")
            continue

        # Calculate segment CoM as the midpoint between markers A and B
        seg_com = 0.5 * (np.array([A_x, A_y, A_z]) + np.array([B_x, B_y, B_z]))
        mass = segment_masses[seg_name]
        COM_sum += seg_com * mass

    if total_mass == 0:
        logger.warning("Total mass is zero. Cannot compute Center of Mass.")
        return np.array([np.nan, np.nan, np.nan])

    return COM_sum / total_mass

def compute_all_com(df, fps_pose=25.0):
    """
    Computes the Center of Mass (CoM) over time for the entire dataset.

    Parameters:
    - df: DataFrame containing marker positions.
    - fps_pose: Frames per second of the pose data.

    Returns:
    - com_x: NumPy array of CoM x-coordinates over time.
    - com_y: NumPy array of CoM y-coordinates over time.
    - com_z: NumPy array of CoM z-coordinates over time.
    - com_speed: NumPy array of CoM speeds over time.
    """
    # Define segment marker pairs
    segment_markers = {
        "left_arm": ("left_shoulder", "left_wrist"),   # approx shoulder->wrist
        "right_arm": ("right_shoulder", "right_wrist"),
        "left_leg": ("left_hip", "left_ankle"),        # approx hip->ankle
        "right_leg": ("right_hip", "right_ankle"),
        "trunk": ("mid_hip", "neck")                  # mid-hip->neck
    }
    # Rough mass distribution (example)
    segment_masses = {
        "left_arm": 0.05,
        "right_arm": 0.05,
        "left_leg": 0.15,
        "right_leg": 0.15,
        "trunk": 0.60
    }

    n_frames = len(df)
    dt = 1.0 / fps_pose

    com_x = np.zeros(n_frames)
    com_y = np.zeros(n_frames)
    com_z = np.zeros(n_frames)

    for i in range(n_frames):
        com_xyz = compute_center_of_mass(df, i, segment_markers, segment_masses)
        com_x[i] = com_xyz[0]
        com_y[i] = com_xyz[1]
        com_z[i] = com_xyz[2]

    # Compute COM velocity
    vx = compute_velocity(pd.Series(com_x), dt)
    vy = compute_velocity(pd.Series(com_y), dt)
    vz = compute_velocity(pd.Series(com_z), dt)
    com_speed = np.sqrt(vx**2 + vy**2 + vz**2)

    return com_x, com_y, com_z, com_speed

###############################################################################
# Preprocessing & Offsetting
###############################################################################
def offset_z_axis(df, reference_point=None):
    """
    Offsets the Z-axis coordinates to normalize vertical positions.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing Z-axis data for various markers.
    reference_point : dict or None
        Dictionary specifying the reference point for each marker's Z-axis.
        Example: {'right_wrist_z': 0.0, 'left_wrist_z': 0.0, ...}
        If None, defaults to subtracting the minimum Z value across all markers.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with Z-axis columns offset.
    """
    z_cols = [col for col in df.columns if col.endswith('_z')]
    if reference_point:
        for col in z_cols:
            ref = reference_point.get(col, 0.0)
            df[col] = df[col] - ref
    else:
        # Default: subtract the minimum Z value to set the lowest point to 0
        for col in z_cols:
            min_z = df[col].min()
            df[col] = df[col] - min_z
    return df


def plot_joint_angles(df, release_frame, selected_angles, phases_df, key_suffix=""):
    """
    Plots selected joint angles over frames with phase overlays and release point.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing joint angle data.
    release_frame : int or None
        Frame number indicating the release point.
    selected_angles : list of str
        List of joint angle columns to plot.
    phases_df : pandas.DataFrame
        DataFrame containing phase information for overlays.
    key_suffix : str
        Unique identifier for the plot's key.
    """
    if not selected_angles:
        st.warning("No joint angles selected for plotting.")
        return

    fig = go.Figure()

    for angle in selected_angles:
        col = angle.lower().replace(' ', '_')  # e.g., 'Right Elbow Angle' -> 'right_elbow_angle'
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in DataFrame.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[col],
            mode='lines+markers',
            name=humanize_label(col)
        ))

    if release_frame is not None:
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add phase overlays
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                    'Stability':      'rgba(255,255,0,0.1)',  # Yellow
                    'Release':        'rgba(255,0,0,0.1)',    # Red
                    'Inertia':        'rgba(128,0,128,0.1)'   # Purple
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.5,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Joint Angles Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angle (°)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"joint_angles_{key_suffix}")
def plot_accelerations(df, release_frame, selected_accelerations, phases_df, key_suffix=""):
    """
    Plots selected angular accelerations over frames with phase overlays and release point.

    :param df: DataFrame containing angular accelerations.
    :param release_frame: Frame number indicating release.
    :param selected_accelerations: List of angular acceleration columns to plot.
    :param phases_df: DataFrame containing phase information for overlays.
    :param key_suffix: Unique identifier for the plot's key.
    """
    if not selected_accelerations:
        st.warning("No angular accelerations selected for plotting.")
        return

    fig = go.Figure()

    for acc in selected_accelerations:
        acc_col = acc.lower().replace(' ', '_')  # e.g., 'Right Elbow Angle Acceleration' -> 'right_elbow_angle_acceleration'
        if acc_col not in df.columns:
            st.warning(f"Selected angular acceleration '{acc}' not found in data.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[acc_col],
            mode='lines+markers',
            name=acc,
            line=dict(width=2)
        ))

    if release_frame is not None and 'frame' in df.columns:
        # Highlight release frame
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add shaded regions for phases
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                    'Stability':      'rgba(255,255,0,0.1)',  # Yellow
                    'Release':        'rgba(255,0,0,0.1)',    # Red
                    'Inertia':        'rgba(128,0,128,0.1)'   # Purple
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Angular Accelerations Over Time",
        xaxis_title="Frame",
        yaxis_title="Angular Acceleration (°/s²)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"angular_acceleration_{key_suffix}")

def plot_angular_velocity(df, release_frame, selected_velocities, phases_df, key_suffix=""):
    """
    Plots selected angular velocities over frames with phase overlays and release point.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing angular velocity data.
    release_frame : int or None
        Frame number indicating the release point.
    selected_velocities : list of str
        List of angular velocity columns to plot.
    phases_df : pandas.DataFrame
        DataFrame containing phase information for overlays.
    key_suffix : str
        Unique identifier for the plot's key.
    """
    if not selected_velocities:
        st.warning("No angular velocities selected for plotting.")
        return

    fig = go.Figure()

    for vel in selected_velocities:
        col = vel.lower().replace(' ', '_')  # e.g., 'Right Elbow Angle Velocity' -> 'right_elbow_angle_velocity'
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in DataFrame.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[col],
            mode='lines+markers',
            name=humanize_label(col)
        ))

    if release_frame is not None:
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add phase overlays
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                    'Stability':      'rgba(255,255,0,0.1)',  # Yellow
                    'Release':        'rgba(255,0,0,0.1)',    # Red
                    'Inertia':        'rgba(128,0,128,0.1)'   # Purple
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.5,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Angular Velocities Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angular Velocity (°/s)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"angular_velocity_{key_suffix}")

ALL_MARKER_COLS = [
    # 3D Marker Positions
    "LEYE_X", "LEYE_Y", "LEYE_Z",
    "REYE_X", "REYE_Y", "REYE_Z",
    "NECK_X", "NECK_Y", "NECK_Z",
    "LSJC_X", "LSJC_Y", "LSJC_Z",
    "LEJC_X", "LEJC_Y", "LEJC_Z",
    "LWJC_X", "LWJC_Y", "LWJC_Z",
    "LPINKY_X", "LPINKY_Y", "LPINKY_Z",
    "LTHUMB_X", "LTHUMB_Y", "LTHUMB_Z",
    "RSJC_X", "RSJC_Y", "RSJC_Z",
    "REJC_X", "REJC_Y", "REJC_Z",
    "RWJC_X", "RWJC_Y", "RWJC_Z",
    "RPINKY_X", "RPINKY_Y", "RPINKY_Z",
    "RTHUMB_X", "RTHUMB_Y", "RTHUMB_Z",
    "MIDHIP_X", "MIDHIP_Y", "MIDHIP_Z",
    "LHJC_X", "LHJC_Y", "LHJC_Z",
    "LKJC_X", "LKJC_Y", "LKJC_Z",
    "LAJC_X", "LAJC_Y", "LAJC_Z",
    "LHEEL_X", "LHEEL_Y", "LHEEL_Z",
    "LSTOE_X", "LSTOE_Y", "LSTOE_Z",
    "LBTOE_X", "LBTOE_Y", "LBTOE_Z",
    "RHJC_X", "RHJC_Y", "RHJC_Z",
    "RKJC_X", "RKJC_Y", "RKJC_Z",
    "RAJC_X", "RAJC_Y", "RAJC_Z",
    "RHEEL_X", "RHEEL_Y", "RHEEL_Z",
    "RSTOE_X", "RSTOE_Y", "RSTOE_Z",
    "RBTOE_X", "RBTOE_Y", "RBTOE_Z",
    "Basketball_X", "Basketball_Y", "Basketball_Z",

    # Joint/Body Angles & Other Computed Columns
    "left_sternoclavicular_elev",
    "left_shoulder_plane",
    "left_shoulder_elev",
    "left_shoulder_rot",
    "right_sternoclavicular_elev",
    "right_shoulder_plane",
    "right_shoulder_elev",
    "right_shoulder_rot",
    "left_elbow_angle",
    "left_forearm_pro",
    "left_wrist_dev",
    "left_wrist_flex",
    "right_elbow_angle",
    "right_forearm_pro",
    "right_wrist_dev",
    "right_wrist_flex",
    "torso_ext",
    "torso_side",
    "torso_rot",
    "pelvis_rot",
    "pelvis_side",
    "left_hip_flex",
    "left_hip_add",
    "left_hip_rot",
    "left_knee",        # sometimes stored as left_knee_angle
    "left_ankle_inv",
    "left_ankle_flex",
    "right_hip_flex",
    "right_hip_add",
    "right_hip_rot",
    "right_knee",       # sometimes stored as right_knee_angle
    "right_ankle_inv",
    "right_ankle_flex",

    # Metadata or other columns (optional—only if needed)
    "shot_type",
    "upload_time",
    "player_name",
    "segment_id",
    "frame"
]

ALL_SMOOTH_COLS = [
    # 3D Marker Positions
    "LEYE_X", "LEYE_Y", "LEYE_Z",
    "REYE_X", "REYE_Y", "REYE_Z",
    "NECK_X", "NECK_Y", "NECK_Z",
    "LSJC_X", "LSJC_Y", "LSJC_Z",
    "LEJC_X", "LEJC_Y", "LEJC_Z",
    "LWJC_X", "LWJC_Y", "LWJC_Z",
    "LPINKY_X", "LPINKY_Y", "LPINKY_Z",
    "LTHUMB_X", "LTHUMB_Y", "LTHUMB_Z",
    "RSJC_X", "RSJC_Y", "RSJC_Z",
    "REJC_X", "REJC_Y", "REJC_Z",
    "RWJC_X", "RWJC_Y", "RWJC_Z",
    "RPINKY_X", "RPINKY_Y", "RPINKY_Z",
    "RTHUMB_X", "RTHUMB_Y", "RTHUMB_Z",
    "MIDHIP_X", "MIDHIP_Y", "MIDHIP_Z",
    "LHJC_X", "LHJC_Y", "LHJC_Z",
    "LKJC_X", "LKJC_Y", "LKJC_Z",
    "LAJC_X", "LAJC_Y", "LAJC_Z",
    "LHEEL_X", "LHEEL_Y", "LHEEL_Z",
    "LSTOE_X", "LSTOE_Y", "LSTOE_Z",
    "LBTOE_X", "LBTOE_Y", "LBTOE_Z",
    "RHJC_X", "RHJC_Y", "RHJC_Z",
    "RKJC_X", "RKJC_Y", "RKJC_Z",
    "RAJC_X", "RAJC_Y", "RAJC_Z",
    "RHEEL_X", "RHEEL_Y", "RHEEL_Z",
    "RSTOE_X", "RSTOE_Y", "RSTOE_Z",
    "RBTOE_X", "RBTOE_Y", "RBTOE_Z",
    "Basketball_X", "Basketball_Y", "Basketball_Z",

    # Joint/Body Angles & Other Computed Columns
    "left_sternoclavicular_elev",
    "left_shoulder_plane",
    "left_shoulder_elev",
    "left_shoulder_rot",
    "right_sternoclavicular_elev",
    "right_shoulder_plane",
    "right_shoulder_elev",
    "right_shoulder_rot",
    "left_elbow_angle",
    "left_forearm_pro",
    "left_wrist_dev",
    "left_wrist_flex",
    "right_elbow_angle",
    "right_forearm_pro",
    "right_wrist_dev",
    "right_wrist_flex",
    "torso_ext",
    "torso_side",
    "torso_rot",
    "pelvis_rot",
    "pelvis_side",
    "left_hip_flex",
    "left_hip_add",
    "left_hip_rot",
    "left_knee",        # sometimes stored as left_knee_angle
    "left_ankle_inv",
    "left_ankle_flex",
    "right_hip_flex",
    "right_hip_add",
    "right_hip_rot",
    "right_knee",       # sometimes stored as right_knee_angle
    "right_ankle_inv",
    "right_ankle_flex",

    # Metadata or other columns (optional—only if needed)
    "shot_type",
    "upload_time",
    "player_name",
    "segment_id",
    "frame"
]
def upsample_smoothed_data(df, x_col='frame', y_cols=None, factor=2):
    """
    Upsample the data by 'factor' via linear interpolation.
    If factor=2, we add 1 extra point between consecutive frames.
    If factor=4, we add 3 extra points, etc.

    :param df: DataFrame, assumed sorted by x_col.
    :param x_col: The column used as the x-axis (e.g., 'frame').
    :param y_cols: List of columns to interpolate (e.g., ['right_elbow_angle_smooth']).
    :param factor: Upsampling factor.
    :return: A new DataFrame with more rows, linearly interpolated data.
    """
    if y_cols is None:
        y_cols = []  # or pick all numeric columns

    # Ensure we sort by x_col
    df = df.sort_values(x_col).reset_index(drop=True)
    new_rows = []

    for i in range(len(df) - 1):
        x1 = df.loc[i, x_col]
        x2 = df.loc[i+1, x_col]

        # If frames are identical or out of order, skip
        if x2 <= x1:
            continue

        # Generate factor+1 points from x1 to x2 (including endpoints)
        xs = np.linspace(x1, x2, factor + 1)

        # For each y_col, linearly interpolate
        row1 = df.iloc[i]
        row2 = df.iloc[i+1]
        for j in range(factor):
            alpha = j / factor
            # Create a dict for the new row
            new_data = {x_col: xs[j]}
            for col in y_cols:
                v1 = row1[col]
                v2 = row2[col]
                # linear interpolation
                new_data[col] = (1 - alpha)*v1 + alpha*v2
            new_rows.append(new_data)

    # Append the last row from original df
    last_row = df.iloc[-1].to_dict()
    new_rows.append(last_row)

    df_new = pd.DataFrame(new_rows)
    return df_new.reset_index(drop=True)

def preprocess_and_smooth(
    df, 
    fps=25, 
    sg_window=31, 
    sg_poly=3, 
    rolling_window=17
):
    """
    Applies heavier smoothing to each column in ALL_SMOOTH_COLS:
      1) Savitzky–Golay filter (larger window, moderate polyorder)
      2) Rolling mean with rolling_window
    Then computes velocity & acceleration with a rolling pass on them too.

    :param df: DataFrame containing raw columns you want to smooth (listed in ALL_SMOOTH_COLS).
    :param fps: Frames per second (default=25).
    :param sg_window: Window length for Savitzky–Golay (must be odd). Larger => more smoothing.
    :param sg_poly: Polynomial order for Savitzky–Golay. e.g., 3 or 4 is typical.
    :param rolling_window: Window for rolling mean on top of Savitzky–Golay.
    :return: DataFrame with new columns:
        [col + "_smooth", col + "_velocity", col + "_acceleration"] 
        for each col in ALL_SMOOTH_COLS.
    """
    dt = 1.0 / fps
    
    for col in ALL_SMOOTH_COLS:
        if col not in df.columns:
            continue  # skip if that column doesn't exist in df
        
        # 1) Convert column to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].dropna().empty:
            # If the entire column is NaN, skip
            continue
        
        # 2) Fill NaNs forward/backward
        series_filled = df[col].fillna(method='ffill').fillna(method='bfill').values
        
        # 3) Apply Savitzky–Golay (with fallback to rolling if data is too short)
        try:
            if len(series_filled) < sg_window:
                # fallback: simple rolling mean if we don't have enough points
                sg_smoothed = (pd.Series(series_filled)
                                 .rolling(window=sg_window, center=True, min_periods=1)
                                 .mean()
                                 .values)
            else:
                # main approach: Savitzky–Golay
                sg_smoothed = savgol_filter(series_filled, window_length=sg_window, polyorder=sg_poly)
        except ValueError:
            # any unexpected error => fallback
            sg_smoothed = (pd.Series(series_filled)
                             .rolling(window=sg_window, center=True, min_periods=1)
                             .mean()
                             .values)
        
        # 4) Second pass: rolling mean on top of Savitzky–Golay
        rolled_smooth = (pd.Series(sg_smoothed)
                           .rolling(window=rolling_window, center=True, min_periods=1)
                           .mean())
        
        # Store final smoothed data
        df[f"{col}_smooth"] = rolled_smooth
        
        # 5) Compute velocity (gradient in time)
        vel = np.gradient(rolled_smooth, dt)
        # Additional rolling mean on velocity
        vel_smooth = (pd.Series(vel)
                        .rolling(window=rolling_window, center=True, min_periods=1)
                        .mean())
        df[f"{col}_velocity"] = vel_smooth
        
        # 6) Compute acceleration
        acc = np.gradient(vel_smooth, dt)
        acc_smooth = (pd.Series(acc)
                        .rolling(window=rolling_window, center=True, min_periods=1)
                        .mean())
        df[f"{col}_acceleration"] = acc_smooth

    return df



def preprocess_ball_data(df, x_col='x', y_col='y'):
    """
    Preprocesses the ball tracking data by interpolating missing 'x' and 'y' values,
    removing outliers, and smoothing the trajectory.

    Parameters:
    - df: DataFrame containing ball tracking data.
    - x_col: Column name for ball's x-coordinate.
    - y_col: Column name for ball's y-coordinate.

    Returns:
    - df: DataFrame with interpolated and smoothed 'x' and 'y' columns.
    """
    if not {x_col, y_col}.issubset(df.columns):
        st.warning(f"Missing columns for ball tracking: '{x_col}' and/or '{y_col}'.")
        return df

    # Interpolate missing values
    df[x_col] = df[x_col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    df[y_col] = df[y_col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Remove outliers using Z-score
    df['x_zscore'] = (df[x_col] - df[x_col].mean()) / df[x_col].std()
    df['y_zscore'] = (df[y_col] - df[y_col].mean()) / df[y_col].std()
    df = df[(df['x_zscore'].abs() < 3) & (df['y_zscore'].abs() < 3)]
    df = df.drop(columns=['x_zscore', 'y_zscore'])

    # Smooth the trajectory
    df[x_col + '_smooth'] = smooth_series_custom(df[x_col], window=11, frac=0.3)
    df[y_col + '_smooth'] = smooth_series_custom(df[y_col], window=11, frac=0.3)

    return df

def compute_ball_speed(df, fps=20.0):
    """
    Computes the ball speed using smoothed x and y positions.
    """
    df['ball_speed'] = np.sqrt(df['x_smooth'].diff()**2 + df['y_smooth'].diff()**2) * fps
    return df['ball_speed']


###############################################################################
# Summarize and Display Statistics
###############################################################################
def summarize_columns(df, columns):
    """
    Return a dict with {column_name: (min, max, mean, std)} for each col.
    """
    summary = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid_vals = df[col].dropna()
            summary[col] = {
                'min': valid_vals.min(),
                'max': valid_vals.max(),
                'mean': valid_vals.mean(),
                'std':  valid_vals.std()
            }
    return summary

def display_summary_table(summary_dict, title="Angle/Velocity/Acceleration Summary"):
    st.markdown(f"### {title}")
    if not summary_dict:
        st.write("No data to summarize.")
        return

    table_data = []
    for col, stats in summary_dict.items():
        table_data.append({
            'Column': col,
            'Min': f"{stats['min']:.2f}",
            'Max': f"{stats['max']:.2f}",
            'Mean': f"{stats['mean']:.2f}",
            'SD': f"{stats['std']:.2f}"
        })
    st.dataframe(pd.DataFrame(table_data))

###############################################################################
# Color-Coding KPI Results
###############################################################################
def color_code_kpi(value, distribution):
    """
    Takes a KPI value and a distribution (list of values) to compute percentile.
    Returns a color: 'green', 'yellow', 'orange', or 'red'.
    """
    if not distribution or pd.isna(value):
        return "grey"  # no data

    sorted_dist = sorted(distribution)
    n = len(sorted_dist)
    # find percentile rank
    rank = 0
    for i, v in enumerate(sorted_dist):
        if value > v:
            rank = i+1
    percentile = (rank / n) * 100

    if percentile >= 75:
        return "green"
    elif 50 <= percentile < 75:
        return "yellow"
    elif 25 <= percentile < 50:
        return "orange"
    else:
        return "red"

###############################################################################
# KPI Display
###############################################################################
def display_ball_metrics(metrics, multi_shot=False, selected_shots=[]):
    """
    Displays ball-specific KPIs using Streamlit's st.metric.
    Supports single and multiple shot comparisons with delta arrows.

    :param metrics: Dictionary containing ball metric names and their values.
    :param multi_shot: Boolean indicating if multiple shots are selected.
    :param selected_shots: List of shot identifiers selected for comparison.
    """
    st.markdown("### 🏀 Additional Ball Metrics")

    # Define the ball KPIs to display
    ball_kpis = {
        "Release Speed": {
            "column": "Release Speed (m/s)",
            "unit": "m/s"
        },
        "Release Height": {
            "column": "Release Height (m)",
            "unit": "m"
        },
        "Release Angle": {
            "column": "Release Angle (°)",
            "unit": "°"
        },
        "Shot Distance": {
            "column": "Shot Distance",
            "unit": "m"
        },
        "Apex Height": {
            "column": "Apex Height",
            "unit": "m"
        },
        "Release Time": {
            "column": "Release Time (s)",
            "unit": "s"
        },
        "Release Quality": {
            "column": "Release Quality",
            "unit": "%"
        },
        "Release Consistency": {
            "column": "Release Consistency",
            "unit": "std"
        },
        "Release Curvature": {
            "column": "Release Curvature",
            "unit": ""
        },
        "Lateral Release Curvature": {
            "column": "Lateral Release Curvature",
            "unit": ""
        }
    }

    # Create a dictionary to hold current shot metrics
    current_metrics = {}
    for name, info in ball_kpis.items():
        current_metrics[name] = metrics.get(info["column"], "N/A")

    if multi_shot and len(selected_shots) > 1:
        # Use the first shot as the reference
        reference_shot = selected_shots[0]
        reference_metrics = user_data[reference_shot].get('metrics', {})
        
        # Create columns for each shot
        num_cols = len(selected_shots)
        cols = st.columns(num_cols)
        
        for idx, shot in enumerate(selected_shots):
            shot_metrics = user_data[shot].get('metrics', {})
            with cols[idx]:
                st.markdown(f"#### {shot}")
                for name, info in ball_kpis.items():
                    current_value = shot_metrics.get(info["column"], "N/A")
                    if isinstance(current_value, (int, float)):
                        # Calculate delta if not the reference shot
                        if shot != reference_shot:
                            ref_value = reference_metrics.get(info["column"], None)
                            if ref_value is not None and isinstance(ref_value, (int, float)):
                                delta = current_value - ref_value
                                delta_formatted = f"{delta:+.2f}" + (info["unit"] if info["unit"] else "")
                                if delta > 0:
                                    delta_color = "green"
                                elif delta < 0:
                                    delta_color = "red"
                                else:
                                    delta_color = "gray"
                            else:
                                delta = "N/A"
                                delta_color = "gray"
                        else:
                            delta = "0"
                            delta_color = "gray"
                        
                        # Display the metric
                        st.metric(
                            label=name,
                            value=f"{current_value:.2f} {info['unit']}" if info["unit"] else f"{current_value:.2f}",
                            delta=delta_formatted if shot != reference_shot else None,
                            delta_color=delta_color if shot != reference_shot else "off"
                        )
                    else:
                        st.metric(
                            label=name,
                            value="N/A"
                        )
                idx += 1
    else:
        # Single shot display
        col1, col2, col3, col4, col5 = st.columns(5)
        all_cols = [col1, col2, col3, col4, col5]
        idx = 0
        for name, value in current_metrics.items():
            with all_cols[idx % 5]:
                if isinstance(value, (int, float)):
                    st.metric(
                        label=name,
                        value=f"{value:.2f} {ball_kpis[name]['unit']}" if ball_kpis[name]['unit'] else f"{value:.2f}",
                        delta=None
                    )
                else:
                    st.metric(
                        label=name,
                        value="N/A",
                        delta=None
                    )
                idx += 1

###############################################################################
# Key Performance Indicators (KPIs) Display
###############################################################################
def display_kpis(df, metrics, multi_shot=False, selected_shots=[]):
    """
    Displays Key Performance Indicators (KPIs) using Streamlit's st.metric.
    Supports single and multiple shot comparisons with delta arrows.

    :param df: DataFrame containing shot data.
    :param metrics: Dictionary containing metric names and their values.
    :param multi_shot: Boolean indicating if multiple shots are selected.
    :param selected_shots: List of shot identifiers selected for comparison.
    """
    st.markdown("### 🎯 Key Performance Indicators (KPIs)")

    # Define the KPIs to display
    kpis = {
        "Avg Right Elbow Angle": {
            "column": "right_elbow_angle",
            "unit": "°"
        },
        "Release Height": {
            "column": "Release Height (m)",
            "unit": "m"
        },
        "Asymmetry Index": {
            "column": "Asymmetry Index",
            "unit": ""
        },
        "Cohesion Score": {
            "column": "Cohesion Score",
            "unit": ""
        },
        "Release Speed": {
            "column": "Release Speed (m/s)",
            "unit": "m/s"
        },
        "Time Elevation to Release": {
            "column": "Time Elevation to Release (s)",
            "unit": "s"
        },
        "Kinetic Chain Efficiency": {
            "column": "Kinetic Chain Efficiency",
            "unit": ""
        },
        "CoM Speed": {
            "column": "CoM Speed (m/s)",
            "unit": "m/s"
        },
        "Ball Trajectory Curvature": {
            "column": "Ball Trajectory Curvature",
            "unit": ""
        }
    }

    # Create a dictionary to hold current shot metrics
    current_metrics = {}
    for name, info in kpis.items():
        current_metrics[name] = metrics.get(info["column"], "N/A")

    if multi_shot and len(selected_shots) > 1:
        # Use the first shot as the reference
        reference_shot = selected_shots[0]
        reference_metrics = user_data[reference_shot].get('metrics', {})
        
        # Create columns for each shot
        num_cols = len(selected_shots)
        cols = st.columns(num_cols)
        
        for idx, shot in enumerate(selected_shots):
            shot_metrics = user_data[shot].get('metrics', {})
            with cols[idx]:
                st.markdown(f"#### {shot}")
                for name, info in kpis.items():
                    current_value = shot_metrics.get(info["column"], "N/A")
                    if isinstance(current_value, (int, float)):
                        # Calculate delta if not the reference shot
                        if shot != reference_shot:
                            ref_value = reference_metrics.get(info["column"], None)
                            if ref_value is not None and isinstance(ref_value, (int, float)):
                                delta = current_value - ref_value
                                delta_formatted = f"{delta:+.2f}" + (info["unit"] if info["unit"] else "")
                                if delta > 0:
                                    delta_color = "green"
                                elif delta < 0:
                                    delta_color = "red"
                                else:
                                    delta_color = "gray"
                            else:
                                delta = "N/A"
                                delta_color = "gray"
                        else:
                            delta = "0"
                            delta_color = "gray"
                        
                        # Display the metric
                        st.metric(
                            label=name,
                            value=f"{current_value:.2f} {info['unit']}",
                            delta=delta_formatted if shot != reference_shot else None,
                            delta_color=delta_color if shot != reference_shot else "off"
                        )
                    else:
                        st.metric(
                            label=name,
                            value="N/A"
                        )
    else:
        # Single shot display
        col1, col2, col3, col4, col5 = st.columns(5)
        all_cols = [col1, col2, col3, col4, col5]
        idx = 0
        for name, value in current_metrics.items():
            with all_cols[idx % 5]:
                if isinstance(value, (int, float)):
                    st.metric(
                        label=name,
                        value=f"{value:.2f} {kpis[name]['unit']}" if kpis[name]['unit'] else f"{value:.2f}",
                        delta=None
                    )
                else:
                    st.metric(
                        label=name,
                        value="N/A",
                        delta=None
                    )
                idx += 1
###############################################################################
# Ball-Specific Release Speed Calculation (Frame-based approach)
###############################################################################
def calculate_ball_release_speed(df, release_frame, fps_ball=20.0):
    """
    Computes an approximate ball release speed using frames
    just before and after the specified release frame.

    - df must have columns ['frame','x','y'] (for 2D ball position),
      though you may rename them to match your actual columns.
    - release_frame is the frame at or near which the ball is "released".
    - fps_ball is the sampling frequency of the ball tracking (default 20 Hz).

    Steps:
      1) Identify the 'frame' values in df that are just below and just above release_frame.
      2) Compute distance traveled (2D) between these two data points.
      3) Divide by the time difference (in seconds) to get speed.
    """
    if release_frame is None or 'frame' not in df.columns:
        return np.nan

    # For safety, let's create a column for the difference from the release frame.
    df['frame_diff'] = df['frame'] - release_frame

    # We want the last frame < release_frame
    df_before = df[df['frame_diff'] < 0]
    if df_before.empty:
        df.drop(columns='frame_diff', inplace=True)
        return np.nan
    before_idx = df_before['frame_diff'].idxmax()  # largest negative => closest below release

    # We want the first frame > release_frame
    df_after = df[df['frame_diff'] > 0]
    if df_after.empty:
        df.drop(columns='frame_diff', inplace=True)
        return np.nan
    after_idx = df_after['frame_diff'].idxmin()  # smallest positive => closest above release

    # Now compute 2D distance
    bx1, by1 = df.loc[before_idx, 'x'], df.loc[before_idx, 'y']
    bx2, by2 = df.loc[after_idx, 'x'], df.loc[after_idx, 'y']
    dist = float(np.sqrt((bx2 - bx1)**2 + (by2 - by1)**2))

    # Frame difference
    frame_diff = float(df.loc[after_idx, 'frame'] - df.loc[before_idx, 'frame'])
    df.drop(columns='frame_diff', inplace=True)

    if frame_diff <= 0:
        return np.nan

    # Time in seconds
    time_sec = frame_diff / fps_ball
    speed = dist / time_sec if time_sec > 0 else np.nan
    return speed

###############################################################################
# Kinematic Chain Sequence Visualizations
###############################################################################
def visualize_kinematic_chain_sequence(df, release_frame=None, phases_df=None, key_suffix=""):
    """
    Visualize the Kinematic Chain Sequence, such as angular velocities of key joints.

    Parameters:
    - df: DataFrame containing angular velocities.
    - release_frame: Frame number indicating release.
    - phases_df: DataFrame containing phase information for overlays.
    - key_suffix: Unique identifier for the plot's key.
    """
    # Define key joints for kinetic chain
    key_joints = ['Left Elbow Angle Velocity', 'Right Elbow Angle Velocity',
                  'Left Knee Angle Velocity', 'Right Knee Angle Velocity']

    fig = go.Figure()

    for joint in key_joints:
        col = joint.replace(' ', '_').lower() + '_vel_smooth'
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['frame'],
                y=df[col],
                mode='lines',
                name=joint
            ))

    if release_frame is not None and 'frame' in df.columns:
        fig.add_vline(
            x=release_frame, 
            line=dict(color='red', dash='dash'), 
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add shaded regions for phases
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                    'Stability':      'rgba(255,255,0,0.1)',  # Yellow
                    'Release':        'rgba(255,0,0,0.1)',    # Red
                    'Inertia':        'rgba(128,0,128,0.1)'   # Purple
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.5,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Kinematic Chain Sequence: Angular Velocities of Key Joints",
        xaxis_title="Frame",
        yaxis_title="Angular Velocity (°/s)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"kinematic_chain_sequence_{key_suffix}")

def compute_temporal_consistency(df):
            """
            Computes Temporal Consistency based on the standard deviation of phase durations.
            Lower standard deviation indicates higher consistency.
            Returns a normalized score between 0 and 1.
            """
            # Assume phases are already identified
            phase_durations = df['phase'].value_counts().reindex(['Ball Elevation', 'Stability', 'Release', 'Inertia']).fillna(0)
            
            # Calculate standard deviation of durations
            sd_durations = phase_durations.std()
            mean_durations = phase_durations.mean()
            
            if mean_durations == 0:
                return 0.0
            
            # Normalize consistency score (higher is better)
            consistency_score = 1 - (sd_durations / mean_durations)
            consistency_score = max(min(consistency_score, 1.0), 0.0)
            
            return consistency_score
        
def compute_cyclical_asymmetry_index(df):
    """
    Computes Cyclical Asymmetry Index based on differences between left and right joint angles.
    Returns a normalized score between 0 and 1 (lower indicates higher symmetry).
    """
    # List of joints to compare
    joints = ['elbow_angle', 'knee_angle', 'shoulder_angle', 'hip_angle']
    
    asymmetry_scores = []
    
    for joint in joints:
        left_col = f'left_{joint}'
        right_col = f'right_{joint}'
        
        if left_col in df.columns and right_col in df.columns:
            # Compute absolute differences
            diff = (df[left_col] - df[right_col]).abs()
            # Normalize differences by the mean angle to account for scale
            mean_angle = df[[left_col, right_col]].mean().mean()
            if mean_angle == 0:
                norm_diff = 0
            else:
                norm_diff = diff.mean() / mean_angle
            asymmetry_scores.append(norm_diff)
    
    if not asymmetry_scores:
        return np.nan
    
    # Average asymmetry across joints
    avg_asymmetry = np.mean(asymmetry_scores)
    
    # Invert and normalize to get index (1 - normalized asymmetry)
    # Assume max asymmetry is 0.5 (50% difference)
    cyclical_asymmetry_index = 1 - min(avg_asymmetry / 0.5, 1.0)
    
    return cyclical_asymmetry_index

def compute_posture_analysis(df):
    """
    Computes Posture Analysis metrics such as Torso Angle and Hip Alignment.
    Returns a dictionary with metric names and their computed values.
    """
    posture_metrics = {}
    
    # Torso Angle: Angle between neck, mid_hip, and a vertical line
    # Assuming vertical line is along the y-axis
    if 'neck_x' in df.columns and 'neck_y' in df.columns and 'neck_z' in df.columns and \
    'mid_hip_x' in df.columns and 'mid_hip_y' in df.columns and 'mid_hip_z' in df.columns:
        # Calculate torso angle for each frame
        df['torso_angle'] = df.apply(lambda row: calculate_torso_angle(row), axis=1)
        # Compute average torso angle
        avg_torso_angle = df['torso_angle'].mean()
        posture_metrics['Torso Angle'] = avg_torso_angle
    
    # Hip Alignment: Difference in hip angles between left and right
    if 'left_hip_angle' in df.columns and 'right_hip_angle' in df.columns:
        hip_alignment = (df['left_hip_angle'] - df['right_hip_angle']).abs().mean()
        posture_metrics['Hip Alignment'] = hip_alignment
    
    return posture_metrics

def calculate_torso_angle(row):
    """
    Calculates the torso angle relative to vertical.
    Assumes vertical is along the y-axis.
    """
    neck = np.array([row['neck_x'], row['neck_y'], row['neck_z']])
    mid_hip = np.array([row['mid_hip_x'], row['mid_hip_y'], row['mid_hip_z']])
    
    # Vector from mid_hip to neck
    vector = neck - mid_hip
    
    # Vertical vector
    vertical = np.array([0, 1, 0])
    
    # Compute angle between vector and vertical
    norm_vector = np.linalg.norm(vector)
    norm_vertical = np.linalg.norm(vertical)
    
    if norm_vector == 0 or norm_vertical == 0:
        return np.nan
    
    cos_theta = np.dot(vector, vertical) / (norm_vector * norm_vertical)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_theta))
    
    return angle

def compute_stride(df):
    """
    Computes Stride as the total forward movement (displacement) of the mid_hip marker during the shot.
    Returns stride length in meters.
    """
    if 'mid_hip_x' not in df.columns or 'mid_hip_y' not in df.columns or 'mid_hip_z' not in df.columns:
        return np.nan
    
    # Calculate total forward movement: difference between first and last frame's mid_hip x-position
    start_x = df.iloc[0]['mid_hip_x']
    end_x = df.iloc[-1]['mid_hip_x']
    stride_length = abs(end_x - start_x)  # Assuming x-axis represents forward movement
    
    # Convert units if necessary (assuming data is in meters)
    return stride_length

def plot_temporal_consistency(consistency_score, key_suffix=""):
    """
    Plots Temporal Consistency as a gauge indicator.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = consistency_score,
        title = {'text': "Temporal Consistency"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 0.25], 'color': "red"},
                {'range': [0.25, 0.5], 'color': "orange"},
                {'range': [0.5, 0.75], 'color': "yellow"},
                {'range': [0.75, 1], 'color': "green"}
            ],
            'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': consistency_score}
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True, key=f"temporal_consistency_{key_suffix}")

def plot_cyclical_asymmetry_index(cyclical_asymmetry, key_suffix=""):
        """
        Plots Cyclical Asymmetry Index as a gauge indicator.
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cyclical_asymmetry,
            title = {'text': "Cyclical Asymmetry Index"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 0.25], 'color': "red"},
                    {'range': [0.25, 0.5], 'color': "orange"},
                    {'range': [0.5, 0.75], 'color': "yellow"},
                    {'range': [0.75, 1], 'color': "green"}
                ],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': cyclical_asymmetry}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"cyclical_asymmetry_{key_suffix}")

def plot_posture_analysis(posture_metrics, key_suffix=""):
    """
    Plots Posture Analysis metrics such as Torso Angle and Hip Alignment.
    """
    fig = go.Figure()
    
    for metric, value in posture_metrics.items():
        if metric == 'Torso Angle':
            fig.add_trace(go.Indicator(
                mode = "number+gauge",
                value = value,
                title = {'text': metric},
                gauge = {
                    'axis': {'range': [0, 180]},  # Adjust range based on expected angles
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 60], 'color': "green"},
                        {'range': [60, 120], 'color': "yellow"},
                        {'range': [120, 180], 'color': "red"}
                    ],
                    'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
                }
            ))
        elif metric == 'Hip Alignment':
            fig.add_trace(go.Indicator(
                mode = "number+gauge",
                value = value,
                title = {'text': metric},
                gauge = {
                    'axis': {'range': [0, 90]},  # Assuming maximum possible misalignment
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 22.5], 'color': "green"},
                        {'range': [22.5, 45], 'color': "yellow"},
                        {'range': [45, 90], 'color': "red"}
                    ],
                    'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
                }
            ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True, key=f"posture_analysis_{key_suffix}")

def plot_stride(stride_length, key_suffix=""):
    """
    Plots Stride Length as a gauge indicator.
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stride_length,
        title = {'text': "Stride Length (m)"},
        gauge = {
            'axis': {'range': [0, 3]},  # Adjust range based on expected stride lengths
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 1], 'color': "red"},
                {'range': [1, 2], 'color': "yellow"},
                {'range': [2, 3], 'color': "green"}
            ],
            'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': stride_length}
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True, key=f"stride_length_{key_suffix}")


###############################################################################
# Putting It All Together
###############################################################################

# Define ALL_MARKER_COLS with all relevant marker coordinates and joint/angle columns
ALL_MARKER_COLS = [
    # 3D Marker Positions
    "leye_x", "leye_y", "leye_z",
    "reye_x", "reye_y", "reye_z",
    "neck_x", "neck_y", "neck_z",
    "lsjc_x", "lsjc_y", "lsjc_z",
    "lejc_x", "lejc_y", "lejc_z",
    "lwjc_x", "lwjc_y", "lwjc_z",
    "lpinky_x", "lpinky_y", "lpinky_z",
    "lthumb_x", "lthumb_y", "lthumb_z",
    "rsjc_x", "rsjc_y", "rsjc_z",
    "rejc_x", "rejc_y", "rejc_z",
    "rwjc_x", "rwjc_y", "rwjc_z",
    "rpinky_x", "rpinky_y", "rpinky_z",
    "rthumb_x", "rthumb_y", "rthumb_z",
    "midhip_x", "midhip_y", "midhip_z",
    "lhjc_x", "lhjc_y", "lhjc_z",
    "lkjc_x", "lkjc_y", "lkjc_z",
    "lajc_x", "lajc_y", "lajc_z",
    "lheel_x", "lheel_y", "lheel_z",
    "lstoe_x", "lstoe_y", "lstoe_z",
    "lbtoe_x", "lbtoe_y", "lbtoe_z",
    "rhjc_x", "rhjc_y", "rhjc_z",
    "rkjc_x", "rkjc_y", "rkjc_z",
    "rajc_x", "rajc_y", "rajc_z",
    "rheel_x", "rheel_y", "rheel_z",
    "rstoe_x", "rstoe_y", "rstoe_z",
    "rbtoe_x", "rbtoe_y", "rbtoe_z",
    "basketball_x", "basketball_y", "basketball_z",
    
    # Joint/Body Angles & Other Computed Columns
    "left_sternoclavicular_elev",
    "left_shoulder_plane",
    "left_shoulder_elev",
    "left_shoulder_rot",
    "right_sternoclavicular_elev",
    "right_shoulder_plane",
    "right_shoulder_elev",
    "right_shoulder_rot",
    "left_elbow_angle",
    "left_forearm_pro",
    "left_wrist_dev",
    "left_wrist_flex",
    "right_elbow_angle",
    "right_forearm_pro",
    "right_wrist_dev",
    "right_wrist_flex",
    "torso_ext",
    "torso_side",
    "torso_rot",
    "pelvis_rot",
    "pelvis_side",
    "left_hip_flex",
    "left_hip_add",
    "left_hip_rot",
    "left_knee_angle",
    "left_ankle_inv",
    "left_ankle_flex",
    "right_hip_flex",
    "right_hip_add",
    "right_hip_rot",
    "right_knee_angle",
    "right_ankle_inv",
    "right_ankle_flex",
    
    # Metadata or other columns (optional—only if needed)
    "shot_type",
    "upload_time",
    "player_name",
    "segment_id",
    "frame"
]

def calculate_velocity_acceleration(df, position_col='x_smooth', fps=20.0):
    """
    Calculates velocity and acceleration from position data using numerical differentiation.
    
    :param df: DataFrame containing smoothed position data.
    :param position_col: Column name for position data.
    :param fps: Frames per second to convert per-frame changes to per-second rates.
    :return: DataFrame with added 'velocity' and 'acceleration' columns.
    """
    # Calculate velocity (first derivative)
    df['velocity'] = np.gradient(df[position_col], 1) * fps  # m/s
    
    # Calculate acceleration (second derivative)
    df['acceleration'] = np.gradient(df['velocity'], 1) * fps  # m/s²
    
    return df
def compute_all_com(df, fps_pose=25.0):
    """
    Computes center of mass and speed.
    """
    # Implement your center of mass computation here
    com_x = df['x'].mean()
    com_y = df['y'].mean()
    com_z = df['z'].mean() if 'z' in df.columns else 0
    com_speed = np.sqrt(df['x'].diff().fillna(0)**2 + df['y'].diff().fillna(0)**2)
    return com_x, com_y, com_z, com_speed
# ------------------------------
# Visualization Functions
# ------------------------------
def visualize_phases(df, phases, key_suffix=""):
    """
    Visualizes the different phases in a bar chart.
    """
    phase_counts = df['phase'].value_counts().reindex(phases).fillna(0).astype(int)
    phase_duration_seconds = (phase_counts / 25).round(2)  # Assuming 25 fps

    phase_info = pd.DataFrame({
        'Phase': phase_counts.index,
        'Frame Count': phase_counts.values,
        'Duration (s)': phase_duration_seconds.values
    })

    # Set 'Phase' as categorical with specific order
    phase_info['Phase'] = pd.Categorical(phase_info['Phase'], categories=phases, ordered=True)

    fig = px.bar(
        phase_info,
        x='Phase',
        y='Duration (s)',
        color='Phase',
        title="Duration of Shooting Phases (Seconds)",
        text='Duration (s)',
        labels={'Duration (s)': 'Duration (Seconds)'}
    )
    fig.update_traces(textposition='auto')
    st.plotly_chart(fig, use_container_width=True, key=f"overview_phases_bar_{key_suffix}")

def visualize_joint_angles(df, right_angle_col, left_angle_col, key_suffix=""):
    """
    Visualizes right vs. left elbow angles over frames with smoothing.
    """
    try:
        smoothed_right = savgol_filter(df[right_angle_col], window_length=11, polyorder=3)
        smoothed_left = savgol_filter(df[left_angle_col], window_length=11, polyorder=3)
    except Exception as e:
        st.warning(f"Error smoothing joint angles: {e}")
        smoothed_right = df[right_angle_col]
        smoothed_left = df[left_angle_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=smoothed_right,
        mode='lines',
        name='Right Elbow Angle',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=smoothed_left,
        mode='lines',
        name='Left Elbow Angle',
        line=dict(color='green')
    ))

    # Highlight phases as shaded regions if available
    if 'phase' in df.columns:
        phases = df['phase'].unique()
        for phase in phases:
            phase_df = df[df['phase'] == phase]
            if not phase_df.empty:
                start = phase_df['frame'].min()
                end = phase_df['frame'].max()
                color_map = {
                    'Preparation': 'rgba(0, 128, 255, 0.1)',
                    'Ball Elevation': 'rgba(255, 165, 0, 0.1)',
                    'Stability': 'rgba(255, 255, 0, 0.1)',
                    'Release': 'rgba(255, 0, 0, 0.1)',
                    'Inertia': 'rgba(128, 0, 128, 0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=fill_color,
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Right vs. Left Elbow Angles Over Frames",
        xaxis_title="Frame",
        yaxis_title="Elbow Angle (degrees)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_joint_angles_{key_suffix}")

def visualize_ball_trajectory_quadratic(df, key_suffix=""):
    """
    Visualizes the ball trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['frame'], df['x_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_x = poly_func(df['frame'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_x = df['x_smooth']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df['x_smooth'],
        mode='markers',
        name='Original X Position',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=fitted_x,
        mode='lines',
        name='Quadratic Fit',
        line=dict(color='orange')
    ))

    # Highlight the point with the greatest increase in velocity
    try:
        df['delta_x'] = df['x'].diff().fillna(0)
        max_increase_frame = df['delta_x'].idxmax()
        max_increase_value = df.loc[max_increase_frame, 'x']
        max_frame = df.loc[max_increase_frame, 'frame']
        fig.add_trace(go.Scatter(
            x=[max_frame],
            y=[max_increase_value],
            mode='markers',
            name='Max X Increase',
            marker=dict(color='red', size=12, symbol='star')
        ))
        fig.add_annotation(
            x=max_frame,
            y=max_increase_value,
            text=f"Max X Increase: Frame {max_frame}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    except Exception as e:
        st.warning(f"Error identifying max X increase: {e}")

    fig.update_layout(
        title="Ball X Position with Quadratic Fit",
        xaxis_title="Frame",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_quadratic_{key_suffix}")

def visualize_ball_trajectory(df, key_suffix=""):
    """
    Visualizes the ball's 2D trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['x_smooth'], df['y_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_y = poly_func(df['x_smooth'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_y = df['y_smooth']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
        y=df['y_smooth'],
        mode='markers',
        name='Original Trajectory',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
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
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_{key_suffix}")

def visualize_phases(df, phases, key_suffix=""):
    """
    Visualizes the different phases in a bar chart.
    """
    phase_counts = df['phase'].value_counts().reindex(phases).fillna(0).astype(int)
    phase_duration_seconds = (phase_counts / 25).round(2)  # Assuming 25 fps for pose tracking
    
    phase_info = pd.DataFrame({
        'Phase': phase_counts.index,
        'Frame Count': phase_counts.values,
        'Duration (s)': phase_duration_seconds.values
    })

    # Set 'Phase' as categorical with specific order
    phase_info['Phase'] = pd.Categorical(phase_info['Phase'], categories=phases, ordered=True)

    fig = px.bar(
        phase_info,
        x='Phase',
        y='Duration (s)',
        color='Phase',
        title="Duration of Shooting Phases (Seconds)",
        text='Duration (s)',
        labels={'Duration (s)': 'Duration (Seconds)'}
    )
    fig.update_traces(textposition='auto')
    st.plotly_chart(fig, use_container_width=True, key=f"overview_phases_bar_{key_suffix}")

def visualize_joint_angles(df, right_angle_col, left_angle_col, key_suffix=""):
    """
    Visualizes right vs. left elbow angles over time with smoothing.
    """
    try:
        smoothed_right = savgol_filter(df[right_angle_col], window_length=11, polyorder=3)
        smoothed_left = savgol_filter(df[left_angle_col], window_length=11, polyorder=3)
    except Exception as e:
        st.warning(f"Error smoothing joint angles: {e}")
        smoothed_right = df[right_angle_col]
        smoothed_left = df[left_angle_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=smoothed_right,
        mode='lines',
        name='Right Elbow Angle',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=smoothed_left,
        mode='lines',
        name='Left Elbow Angle',
        line=dict(color='green')
    ))

    # Highlight phases as shaded regions if available
    if 'phase' in df.columns:
        phases = df['phase'].unique()
        for phase in phases:
            phase_df = df[df['phase'] == phase]
            if not phase_df.empty:
                start = phase_df['time_pose'].min()
                end = phase_df['time_pose'].max()
                color_map = {
                    'Preparation': 'rgba(0, 128, 255, 0.1)',
                    'Ball Elevation': 'rgba(255, 165, 0, 0.1)',
                    'Stability': 'rgba(255, 255, 0, 0.1)',
                    'Release': 'rgba(255, 0, 0, 0.1)',
                    'Inertia': 'rgba(128, 0, 128, 0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=fill_color,
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Right vs. Left Elbow Angles Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Elbow Angle (degrees)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_joint_angles_{key_suffix}")

def visualize_ball_trajectory_quadratic(df, key_suffix=""):
    """
    Visualizes the ball's X position with a quadratic fit and marks the point of maximum increase.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['time_ball'], df['x_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_x = poly_func(df['time_ball'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_x = df['x_smooth']

    # Calculate delta_x to find the biggest increase
    df['delta_x'] = df['x_smooth'].diff().fillna(0)
    max_increase_idx = df['delta_x'].idxmax()
    if not pd.isna(max_increase_idx):
        max_increase_time = df.loc[max_increase_idx, 'time_ball']
        max_increase_x = df.loc[max_increase_idx, 'x_smooth']
    else:
        max_increase_time = None
        max_increase_x = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time_ball'],
        y=df['x_smooth'],
        mode='markers',
        name='Original X Position',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['time_ball'],
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
            text=f"Max X Increase: {max_increase_time:.2f}s",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )

    fig.update_layout(
        title="Ball X Position with Quadratic Fit",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_quadratic_{key_suffix}")

def visualize_ball_trajectory(df, key_suffix=""):
    """
    Visualizes the ball's 2D trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['x_smooth'], df['y_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_y = poly_func(df['x_smooth'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_y = df['y_smooth']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
        y=df['y_smooth'],
        mode='markers',
        name='Original Trajectory',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
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
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_{key_suffix}")

def visualize_ball_heatmap(df, key_suffix=""):
    """
    Creates a heatmap of the ball's 2D positions to show frequency of locations.
    """
    if df.empty:
        st.warning("No data to plot for ball heatmap.")
        return

    fig = px.density_heatmap(
        df,
        x='x_smooth',
        y='y_smooth',
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale='Blues',
        title="Ball Location Heatmap"
    )
    fig.update_layout(
        xaxis_title="X Position",
        yaxis_title="Y Position",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_heatmap_{key_suffix}")

def visualize_ball_trajectory_side_view(df, x_col='time_ball', y_col='x_smooth', spline=True, smoothing_val=1.2, enforce_data_range=True, highlight_max_speed=True, key_suffix=""):
    """
    Visualizes the side-view of the ball's X position over time.
    Highlights the region with the maximum increase in X position.
    """
    if df.empty:
        st.warning("No data to plot in side-view trajectory.")
        return

    if not {x_col, y_col}.issubset(df.columns):
        st.warning(f"Missing required columns: {x_col}, {y_col}.")
        return

    # Line shape
    line_dict = dict(color='blue')
    if spline:
        line_dict.update(shape='spline', smoothing=smoothing_val)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        line=line_dict,
        name="Side-View X"
    ))

    # Fit axis to data range if desired
    if enforce_data_range:
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])

    # Highlight biggest increase in X
    if highlight_max_speed:
        df['delta_x'] = df[y_col].diff().fillna(0)
        max_increase_idx = df['delta_x'].idxmax()
        if not pd.isna(max_increase_idx):
            max_increase_time = df.loc[max_increase_idx, x_col]
            max_increase_x = df.loc[max_increase_idx, y_col]
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
                text=f"Max X Increase: {max_increase_time:.2f}s",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

    fig.update_layout(
        title="Side-View Ball X Position Over Time",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        showlegend=True,
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_side_view_{key_suffix}")


# ------------------------------
# Preprocessing Functions
# ------------------------------
def preprocess_and_smooth(df, sg_window=51, sg_poly=3):
    """
    Applies Savitzky-Golay filter to smooth the x and y position data.
    
    :param df: DataFrame containing position data.
    :param sg_window: Window length for Savitzky-Golay filter (must be odd).
    :param sg_poly: Polynomial order for Savitzky-Golay filter.
    :return: DataFrame with smoothed 'x_smooth' and 'y_smooth' columns.
    """
    # Ensure window_length is odd and less than or equal to the size of the data
    if sg_window % 2 == 0:
        sg_window += 1
    if sg_window > len(df):
        sg_window = len(df) if len(df) % 2 != 0 else len(df) - 1
    try:
        df['x_smooth'] = savgol_filter(df['x'], window_length=sg_window, polyorder=sg_poly)
        df['y_smooth'] = savgol_filter(df['y'], window_length=sg_window, polyorder=sg_poly)
    except Exception as e:
        st.warning(f"Error applying Savitzky-Golay filter: {e}")
        df['x_smooth'] = df['x']
        df['y_smooth'] = df['y']
    return df

def calculate_velocity_acceleration(df, position_col='x_smooth', fps=20.0):
    """
    Calculates velocity and acceleration from position data using numerical differentiation.
    
    :param df: DataFrame containing smoothed position data.
    :param position_col: Column name for position data.
    :param fps: Frames per second to convert per-frame changes to per-second rates.
    :return: DataFrame with added 'velocity' and 'acceleration' columns.
    """
    # Calculate velocity (first derivative)
    df['velocity'] = np.gradient(df[position_col], 1) * fps  # m/s
    
    # Calculate acceleration (second derivative)
    df['acceleration'] = np.gradient(df['velocity'], 1) * fps  # m/s²
    
    return df

# ------------------------------
# Visualization Functions
# ------------------------------
def visualize_phases(df, phases, key_suffix=""):
    """
    Visualizes the different phases in a bar chart.
    """
    phase_counts = df['phase'].value_counts().reindex(phases).fillna(0).astype(int)
    phase_duration_seconds = (phase_counts / 25).round(2)  # Assuming 25 fps for pose tracking
    
    phase_info = pd.DataFrame({
        'Phase': phase_counts.index,
        'Frame Count': phase_counts.values,
        'Duration (s)': phase_duration_seconds.values
    })

    # Set 'Phase' as categorical with specific order
    phase_info['Phase'] = pd.Categorical(phase_info['Phase'], categories=phases, ordered=True)

    fig = px.bar(
        phase_info,
        x='Phase',
        y='Duration (s)',
        color='Phase',
        title="Duration of Shooting Phases (Seconds)",
        text='Duration (s)',
        labels={'Duration (s)': 'Duration (Seconds)'}
    )
    fig.update_traces(textposition='auto')
    st.plotly_chart(fig, use_container_width=True, key=f"overview_phases_bar_{key_suffix}")

def visualize_joint_angles(df, right_angle_col, left_angle_col, key_suffix=""):
    """
    Visualizes right vs. left elbow angles over time with smoothing.
    """
    try:
        smoothed_right = savgol_filter(df[right_angle_col], window_length=11, polyorder=3)
        smoothed_left = savgol_filter(df[left_angle_col], window_length=11, polyorder=3)
    except Exception as e:
        st.warning(f"Error smoothing joint angles: {e}")
        smoothed_right = df[right_angle_col]
        smoothed_left = df[left_angle_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=smoothed_right,
        mode='lines',
        name='Right Elbow Angle',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['time_pose'],
        y=smoothed_left,
        mode='lines',
        name='Left Elbow Angle',
        line=dict(color='green')
    ))

    # Highlight phases as shaded regions if available
    if 'phase' in df.columns:
        phases = df['phase'].unique()
        for phase in phases:
            phase_df = df[df['phase'] == phase]
            if not phase_df.empty:
                start = phase_df['time_pose'].min()
                end = phase_df['time_pose'].max()
                color_map = {
                    'Preparation': 'rgba(0, 128, 255, 0.1)',
                    'Ball Elevation': 'rgba(255, 165, 0, 0.1)',
                    'Stability': 'rgba(255, 255, 0, 0.1)',
                    'Release': 'rgba(255, 0, 0, 0.1)',
                    'Inertia': 'rgba(128, 0, 128, 0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=fill_color,
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Right vs. Left Elbow Angles Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Elbow Angle (degrees)",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_joint_angles_{key_suffix}")

def visualize_ball_trajectory_quadratic(df, key_suffix=""):
    """
    Visualizes the ball's X position with a quadratic fit and marks the point of maximum increase.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['time_ball'], df['x_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_x = poly_func(df['time_ball'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_x = df['x_smooth']

    # Calculate delta_x to find the biggest increase
    df['delta_x'] = df['x_smooth'].diff().fillna(0)
    max_increase_idx = df['delta_x'].idxmax()
    if not pd.isna(max_increase_idx):
        max_increase_time = df.loc[max_increase_idx, 'time_ball']
        max_increase_x = df.loc[max_increase_idx, 'x_smooth']
    else:
        max_increase_time = None
        max_increase_x = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time_ball'],
        y=df['x_smooth'],
        mode='markers',
        name='Original X Position',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['time_ball'],
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
            text=f"Max X Increase: {max_increase_time:.2f}s",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )

    fig.update_layout(
        title="Ball X Position with Quadratic Fit",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_quadratic_{key_suffix}")

def visualize_ball_trajectory(df, key_suffix=""):
    """
    Visualizes the ball's 2D trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['x_smooth'], df['y_smooth'], 2)
        a, b, c = coefficients
        poly_func = np.poly1d(coefficients)
        fitted_y = poly_func(df['x_smooth'])
    except Exception as e:
        st.warning(f"Error fitting quadratic curve: {e}")
        fitted_y = df['y_smooth']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
        y=df['y_smooth'],
        mode='markers',
        name='Original Trajectory',
        marker=dict(color='blue', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['x_smooth'],
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
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_trajectory_{key_suffix}")

def visualize_ball_heatmap(df, key_suffix=""):
    """
    Creates a heatmap of the ball's 2D positions to show frequency of locations.
    """
    if df.empty:
        st.warning("No data to plot for ball heatmap.")
        return

    fig = px.density_heatmap(
        df,
        x='x_smooth',
        y='y_smooth',
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
    st.plotly_chart(fig, use_container_width=True, key=f"overview_ball_heatmap_{key_suffix}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter

import boto3
# import your needed modules, e.g. from utils import ...
# from your_codebase import ...

###############################
# Utility & Preprocessing Functions
###############################
def compute_velocity(series, dt=0.01):
    """
    Basic centered finite difference to compute velocity from a series.
    """
    shifted_forward = series.shift(-1)
    shifted_back = series.shift(1)
    vel = (shifted_forward - shifted_back) / (2.0 * dt)
    return vel

def compute_acceleration(series, dt=0.05):
    """
    Compute acceleration from a velocity series, using the same finite difference approach.
    """
    return compute_velocity(series, dt=dt)

def compute_angular_velocity(df, angle_col, dt=0.05):
    """
    Compute angular velocity from an angle column, store in angle_col+'_vel' and angle_col+'_vel_smooth'.
    """
    if angle_col not in df.columns:
        df[angle_col + '_vel'] = np.nan
        df[angle_col + '_vel_smooth'] = np.nan
        return df
    
    # Raw velocity
    df[angle_col + '_vel'] = compute_velocity(df[angle_col], dt=dt)
    # Smooth the velocity
    df[angle_col + '_vel_smooth'] = smooth_series_custom(df[angle_col + '_vel'], window=11, frac=0.3)
    return df

def smooth_series_custom(series, window=11, frac=0.3):
    """
    Example smoothing function that can combine Savitzky-Golay or other local smoothing.
    You can tweak the method as needed.
    """
    # Using Savitzky-Golay here, but you can add more advanced logic if desired
    # Ensure window length is odd and not more than length of the series
    wsize = min(len(series), window)
    if wsize < 3:
        return series  # not enough data to smooth
    if wsize % 2 == 0:
        wsize -= 1  # make it odd
    try:
        return savgol_filter(series.fillna(method='ffill').fillna(0), wsize, 3)
    except:
        return series

def offset_z_axis(df, reference_point=None):
    """
    Offsets the Z-axis coordinates to normalize vertical positions.
    If `reference_point` is None, subtracts each column's minimum.
    """
    z_cols = [col for col in df.columns if col.endswith('_z')]
    if reference_point:
        for col in z_cols:
            ref = reference_point.get(col, 0.0)
            df[col] = df[col] - ref
    else:
        # Default: subtract the minimum Z value to set the lowest point to 0
        for col in z_cols:
            min_z = df[col].min()
            df[col] = df[col] - min_z
    return df

def humanize_label(label):
    """
    Convert something like 'right_elbow_angle_vel_smooth' -> 'Right Elbow Angle Vel Smooth'
    Removing underscores, capitalizing each word.
    """
    return " ".join([part.capitalize() for part in label.split("_")])

def dehumanize_label(label):
    """
    Convert 'Right Elbow Angle Vel Smooth' -> 'right_elbow_angle_vel_smooth'
    (In case you need the column name back.)
    """
    return label.lower().replace(" ", "_")

###############################
# Plotting Functions (Angles, Velocities, Accelerations)
###############################
def plot_joint_angles(df, release_frame, selected_angles, phases_df, key_suffix=""):
    """
    Plots selected joint angles over frames with phase overlays and release point.
    """
    if not selected_angles:
        st.warning("No joint angles selected for plotting.")
        return

    fig = go.Figure()

    for angle_label in selected_angles:
        # Convert the humanized label back to the actual column name
        col = dehumanize_label(angle_label)  # e.g. 'Right Elbow Angle' -> 'right_elbow_angle'
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in DataFrame.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[col],
            mode='lines+markers',
            name=angle_label
        ))

    if release_frame is not None:
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add phase overlays (if present)
    if phases_df is not None and 'phase' in phases_df.columns:
        # Modify this list if you have different or more phases
        for phase in ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation':   'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
                    'Stability':      'rgba(255,255,0,0.1)',  # Yellow
                    'Release':        'rgba(255,0,0,0.1)',    # Red
                    'Inertia':        'rgba(128,0,128,0.1)'   # Purple
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Joint Angles Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angle (°)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"joint_angles_{key_suffix}")

def plot_angular_velocity(df, release_frame, selected_velocities, phases_df, key_suffix=""):
    """
    Plots selected angular velocities over frames with phase overlays and release point.
    """
    if not selected_velocities:
        st.warning("No angular velocities selected for plotting.")
        return

    fig = go.Figure()

    for vel_label in selected_velocities:
        col = dehumanize_label(vel_label)  # e.g. 'Right Elbow Angle Vel' -> 'right_elbow_angle_vel'
        if col not in df.columns:
            st.warning(f"Column '{col}' not found in DataFrame.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[col],
            mode='lines+markers',
            name=vel_label
        ))

    if release_frame is not None:
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add phase overlays
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation':   'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',
                    'Stability':      'rgba(255,255,0,0.1)',
                    'Release':        'rgba(255,0,0,0.1)',
                    'Inertia':        'rgba(128,0,128,0.1)'
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Angular Velocities Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angular Velocity (°/s)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"angular_velocity_{key_suffix}")

def plot_accelerations(df, release_frame, selected_accelerations, phases_df, key_suffix=""):
    """
    Plots selected angular accelerations over frames with phase overlays and release point.
    """
    if not selected_accelerations:
        st.warning("No angular accelerations selected for plotting.")
        return

    fig = go.Figure()

    for acc_label in selected_accelerations:
        acc_col = dehumanize_label(acc_label)  # e.g. 'Right Elbow Angle Acc' -> 'right_elbow_angle_acc'
        if acc_col not in df.columns:
            st.warning(f"Selected angular acceleration '{acc_label}' not found in data.")
            continue
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[acc_col],
            mode='lines+markers',
            name=acc_label
        ))

    if release_frame is not None and 'frame' in df.columns:
        fig.add_vline(
            x=release_frame,
            line=dict(color='red', dash='dash'),
            annotation_text="Release",
            annotation_position="top left"
        )

    # Add phase overlays
    if phases_df is not None and 'phase' in phases_df.columns:
        for phase in ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = phases_df[phases_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation':   'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',
                    'Stability':      'rgba(255,255,0,0.1)',
                    'Release':        'rgba(255,0,0,0.1)',
                    'Inertia':        'rgba(128,0,128,0.1)'
                }
                fig.add_vrect(
                    x0=start_f, x1=end_f,
                    fillcolor=color_map.get(phase, 'rgba(128,128,128,0.1)'),
                    opacity=0.3,
                    layer='below',
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )

    fig.update_layout(
        title="Angular Accelerations Over Frames",
        xaxis_title="Frame",
        yaxis_title="Angular Acceleration (°/s²)",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, key=f"angular_acceleration_{key_suffix}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import boto3

###############################
# Helper: Trim shot end
###############################
def trim_shot_end(df, release_frame, frames_before=10):
    """
    Returns data from (release_frame - frames_before) through the end of df.
    """
    if release_frame is None:
        # No release frame => no trimming
        return df
    start_frame = max(0, release_frame - frames_before)
    return df[df['frame'] >= start_frame].copy()

###############################
# Example: Increase smoothing
###############################
def preprocess_and_smooth(df, sg_window=81, sg_poly=3):
    """
    Example of your smoothing pipeline with a larger window size (81).
    - This will apply heavier smoothing to reduce noise.
    Adjust as needed for your data.
    """
    # Safeguard: ensure the window size is not larger than data length
    # and is an odd number
    def valid_window(n):
        w = min(n, sg_window)
        if w < 3:
            return 3
        if w % 2 == 0:
            w -= 1
        return w

    # Example smoothing on x_smooth, y_smooth if present
    if 'x' in df.columns:
        w = valid_window(len(df['x']))
        try:
            df['x_smooth'] = savgol_filter(df['x'], w, sg_poly)
        except:
            pass

    if 'y' in df.columns:
        w = valid_window(len(df['y']))
        try:
            df['y_smooth'] = savgol_filter(df['y'], w, sg_poly)
        except:
            pass

    # If you already had x_smooth, y_smooth columns, similarly re-smooth them:
    if 'x_smooth' in df.columns:
        w = valid_window(len(df['x_smooth']))
        try:
            df['x_smooth'] = savgol_filter(df['x_smooth'], w, sg_poly)
        except:
            pass

    if 'y_smooth' in df.columns:
        w = valid_window(len(df['y_smooth']))
        try:
            df['y_smooth'] = savgol_filter(df['y_smooth'], w, sg_poly)
        except:
            pass

    # You may have other columns to smooth as well (e.g. joint angles, etc.)
    return df

###############################
# Increase smoothing for velocities
###############################
def compute_velocity_acceleration(df, position_col='x_smooth', fps=20.0):
    """
    Compute velocity & acceleration with heavier smoothing on velocity and acceleration themselves.
    Creates columns:
      position_col + '_vel',
      position_col + '_vel_smooth',
      position_col + '_acc',
      position_col + '_acc_smooth'
    """
    dt = 1.0 / fps

    def finite_diff(series):
        return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

    vel_col = position_col + '_vel'
    acc_col = position_col + '_acc'

    if position_col in df.columns:
        # Raw velocity
        df[vel_col] = finite_diff(df[position_col])

        # Smooth the velocity more heavily
        w = 21  # you can increase further if you want
        try:
            if w >= len(df[vel_col]):
                w = len(df[vel_col]) - 1 if (len(df[vel_col]) - 1) % 2 == 1 else len(df[vel_col]) - 2
            df[vel_col + '_smooth'] = savgol_filter(df[vel_col].fillna(method='ffill').fillna(0), w, 3)
        except:
            df[vel_col + '_smooth'] = df[vel_col]

        # Acceleration from smoothed velocity
        df[acc_col] = finite_diff(df[vel_col + '_smooth'])

        # Smooth the acceleration
        w2 = 21  # can also increase further
        try:
            if w2 >= len(df[acc_col]):
                w2 = len(df[acc_col]) - 1 if (len(df[acc_col]) - 1) % 2 == 1 else len(df[acc_col]) - 2
            df[acc_col + '_smooth'] = savgol_filter(df[acc_col].fillna(method='ffill').fillna(0), w2, 3)
        except:
            df[acc_col + '_smooth'] = df[acc_col]
    else:
        df[vel_col] = np.nan
        df[vel_col + '_smooth'] = np.nan
        df[acc_col] = np.nan
        df[acc_col + '_smooth'] = np.nan

    return df

###############################
# Main Streamlit App
###############################
def main():
    st.title("🏀 **Pivotal Motion Data Visualizer**")
    st.write("### Analyze and Visualize Shooting Motion Data")

    # Initialize AWS clients
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')

    # Retrieve bucket name from secrets
    bucket_name = st.secrets["BUCKET_NAME"]

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
        st.stop()

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
        df_shot = load_vibe_output(s3_client, bucket_name, chosen_user, chosen_job, shot)
        if df_shot.empty:
            st.warning(f"No data for shot '{shot}'.")
            continue

        # Merge job metadata
        for k, v in job_meta.items():
            df_shot[k.lower()] = v

        df_shot = map_body_columns(df_shot)

        if 'frame' not in df_shot.columns:
            df_shot['frame'] = np.arange(len(df_shot))

        # 1) Preprocess ball data (original function)
        df_shot = preprocess_ball_data(df_shot, x_col='x', y_col='y')

        # 2) Increase smoothing for body & ball
        df_shot = preprocess_and_smooth(df_shot, sg_window=81, sg_poly=3)

        # 3) Calculate linear velocity & acceleration for the ball (using heavier smoothing)
        df_shot = calculate_velocity_acceleration(df_shot, position_col='x_smooth', fps=20.0)

        # 4) Detect phases
        df_shot['phase'] = identify_biomech_phases(df_shot, fps=25)

        # 5) Filter for the main shooting player if multiple
        df_shot = filter_for_shooter_by_phases(df_shot)

        # If 'player_id' in columns, let user pick
        if 'player_id' in df_shot.columns:
            unique_pids = df_shot['player_id'].unique()
            if len(unique_pids) > 1:
                chosen_pid = st.sidebar.selectbox(f"Select Player ID for shot {shot}", unique_pids, key=f"{shot}_pid")
            else:
                chosen_pid = unique_pids[0]
            df_shot = df_shot[df_shot['player_id'] == chosen_pid]

        if df_shot.empty:
            st.warning(f"No data after filtering for player_id in shot '{shot}'.")
            continue

        # 6) Offset Z axis and compute joint angles
        df_shot = offset_z_axis(df_shot)
        df_shot = compute_joint_angles(df_shot)

        # 7) Compute angular velocity & acceleration for all angle columns
        angle_cols = [c for c in df_shot.columns if c.endswith('_angle')]
        for a_col in angle_cols:
            df_shot = compute_angular_velocity(df_shot, a_col, dt=1.0/25.0)  # or your actual FPS
            # Next, compute angular acceleration from the velocity
            vel_col = a_col + '_vel_smooth'
            acc_col = a_col + '_acc'
            if vel_col in df_shot.columns:
                df_shot[acc_col] = compute_acceleration(df_shot[vel_col], dt=1.0/25.0)

        # 8) Detect release frame & re-identify phases
        release_frame_seg = detect_release_frame(df_shot)
        phases_seg = identify_phases(df_shot, release_frame_seg, fps_body=25, fps_ball=20)
        df_shot['phase'] = phases_seg

        # 9) Compute Center of Mass
        com_x, com_y, com_z, com_speed = compute_all_com(df_shot, fps_pose=25.0)
        df_shot['com_x'] = com_x
        df_shot['com_y'] = com_y
        df_shot['com_z'] = com_z
        df_shot['com_speed'] = com_speed

        # 10) Compute Curvature (for the ball)
        if {'x_smooth', 'y_smooth'}.issubset(df_shot.columns):
            df_shot['curvature'] = df_shot.apply(
                lambda row: compute_ball_trajectory_curvature(df_shot, row['frame']), axis=1
            )
        else:
            df_shot['curvature'] = np.nan

        # 11) Calculate Advanced Metrics
        metrics = calculate_advanced_metrics_reworked(
            df_shot, 
            release_frame_seg, 
            player_height=job_meta.get('player_height', 2.0)
        )
        for metric, value in metrics.items():
            col_name = metric.lower().replace(' ', '_')
            df_shot[col_name] = value

        # 12) Convert frame to time
        df_shot['time_pose'] = df_shot['frame'] / 25.0  # Pose tracking FPS
        df_shot['time_ball'] = df_shot['frame'] / 20.0  # Ball tracking FPS

        user_data[shot] = df_shot

    if not user_data:
        st.warning("No valid shot data loaded.")
        return

    # Combine data for Overview
    combined_df = pd.DataFrame()
    for shot, d in user_data.items():
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
        show_brand_header(player_name, team_name)  # Implement as needed

        release_frame = detect_release_frame(combined_df)
        if release_frame is None:
            st.warning("Could not detect release frame in combined data.")
        else:
            st.write(f"**Detected Release Frame**: {release_frame}")

        trimmed_agg = trim_shot(combined_df, release_frame, frames_before=30, frames_after=20)
        player_ht = trimmed_agg['player_height'].iloc[0] if 'player_height' in trimmed_agg.columns else 2.0
        metrics = calculate_advanced_metrics_reworked(trimmed_agg, release_frame, player_height=player_ht)
        for metric, value in metrics.items():
            column_name = metric.lower().replace(' ', '_')
            trimmed_agg[column_name] = value

        display_kpis(trimmed_agg, metrics, multi_shot=multi_shot, selected_shots=chosen_shots)
        display_ball_metrics(metrics, multi_shot=multi_shot, selected_shots=chosen_shots)

        phases_4 = identify_phases(trimmed_agg, release_frame, fps_body=25, fps_ball=20)
        trimmed_agg['phase'] = phases_4
        visualize_phases(trimmed_agg, ['Preparation','Ball Elevation','Stability','Release','Inertia'], key_suffix="overview")

        if 'right_elbow_angle' in trimmed_agg.columns and 'left_elbow_angle' in trimmed_agg.columns:
            visualize_joint_angles(trimmed_agg, 'right_elbow_angle', 'left_elbow_angle', key_suffix="overview")
        else:
            st.warning("Required elbow angle columns are missing.")

        if {'x_smooth', 'y_smooth', 'frame', 'time_ball'}.issubset(trimmed_agg.columns):
            visualize_ball_trajectory_quadratic(trimmed_agg, key_suffix="overview")
            visualize_ball_trajectory(trimmed_agg, key_suffix="overview")
        else:
            st.warning("Missing 'x_smooth', 'y_smooth', 'frame', or 'time_ball' for ball trajectory plots.")

        if {'x_smooth', 'y_smooth'}.issubset(trimmed_agg.columns):
            visualize_ball_heatmap(trimmed_agg, key_suffix="overview")
        else:
            st.warning("Missing 'x_smooth' or 'y_smooth' for ball heatmap.")

    # -------------- POSE ANALYSIS TAB --------------
    with t_pose:
        st.markdown("## Pose Analysis")
        for shot, df_shot in user_data.items():
            st.markdown(f"### Shot: {shot}")
            release_frame_seg = detect_release_frame(df_shot)
            st.write(f"**Detected Release Frame:** {release_frame_seg}" if release_frame_seg else "No release frame found.")

            trimmed_df = trim_shot(df_shot, release_frame_seg, frames_before=30, frames_after=20)
            metrics_seg = calculate_advanced_metrics_reworked(trimmed_df, release_frame_seg)
            for metric, value in metrics_seg.items():
                column_name = metric.lower().replace(' ', '_')
                trimmed_df[column_name] = value

            display_kpis(trimmed_df, metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)
            display_ball_metrics(metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)

            phases_seg = identify_phases(trimmed_df, release_frame_seg, fps_body=25, fps_ball=20)
            trimmed_df['phase'] = phases_seg

            angle_columns = [col for col in trimmed_df.columns if col.endswith('_angle')]
            angle_options = [humanize_label(c) for c in angle_columns]
            selected_angles = st.multiselect(
                f"Select Joint Angles to Visualize (Shot: {shot})",
                angle_options,
                default=angle_options,
                key=f"{shot}_pose_angles"
            )
            plot_joint_angles(trimmed_df, release_frame_seg, selected_angles, trimmed_df, key_suffix=shot)

            vel_columns = [col for col in trimmed_df.columns if col.endswith('_vel') or col.endswith('_vel_smooth')]
            vel_options = [humanize_label(c) for c in vel_columns]
            selected_vels = st.multiselect(
                f"Select Angular Velocities to Visualize (Shot: {shot})",
                vel_options,
                default=[],
                key=f"{shot}_pose_vels"
            )
            plot_angular_velocity(trimmed_df, release_frame_seg, selected_vels, trimmed_df, key_suffix=f"vel_{shot}")

            acc_columns = [col for col in trimmed_df.columns if col.endswith('_acc')]
            acc_options = [humanize_label(c) for c in acc_columns]
            selected_accs = st.multiselect(
                f"Select Angular Accelerations to Visualize (Shot: {shot})",
                acc_options,
                default=[],
                key=f"{shot}_pose_accs"
            )
            plot_accelerations(trimmed_df, release_frame_seg, selected_accs, trimmed_df, key_suffix=f"acc_{shot}")

    # -------------- ADVANCED BIOMECHANICS TAB --------------
    with t_adv:
        st.markdown("## Advanced Biomechanics")
        for shot, df in user_data.items():
            st.markdown(f"### Shot: {shot}")
            release_frame_seg = detect_release_frame(df)
            trimmed_df = trim_shot(df, release_frame_seg, frames_before=30, frames_after=20)
            metrics_seg = calculate_advanced_metrics_reworked(trimmed_df, release_frame_seg)
            for metric, value in metrics_seg.items():
                column_name = metric.lower().replace(' ', '_')
                trimmed_df[column_name] = value

            st.write("**Advanced Metrics at Release**")
            if metrics_seg:
                show_df = pd.DataFrame(metrics_seg, index=[0]).T
                show_df.columns = ["Value"]
                st.table(show_df)
            else:
                st.write("No advanced metrics available.")

            display_kpis(trimmed_df, metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)
            display_ball_metrics(metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)

            st.markdown("### Kinematic Chain Sequence")
            visualize_kinematic_chain_sequence(trimmed_df, release_frame_seg, trimmed_df[['frame','phase']], key_suffix=shot)

            if 'cohesion_score' in trimmed_df.columns:
                fig_cohesion = px.line(
                    trimmed_df,
                    x='time_pose',
                    y='cohesion_score',
                    title="Cohesion Score Over Time",
                    labels={'time_pose': 'Time (s)', 'cohesion_score': 'Cohesion Score'},
                    markers=True
                )
                st.plotly_chart(fig_cohesion, use_container_width=True, key=f"advanced_cohesion_score_{shot}")
            else:
                st.warning("No 'cohesion_score' data available.")

    # -------------- BALL TRACKING TAB --------------
    with t_ball:
        st.markdown("## Ball Tracking & Trajectory")
        for shot, df_seg in user_data.items():
            st.markdown(f"### Shot: {shot}")
            # Make sure we have the columns needed
            if not {'frame','x','y'}.issubset(df_seg.columns):
                st.warning(f"No 'frame','x','y' columns => skipping ball trajectory for shot {shot}.")
                continue

            # Detect release for the ball data
            release_frame_seg = detect_release_frame(df_seg)

            # Trim ball data from 10 frames before release => end
            trimmed_ball_df = trim_shot_end(df_seg, release_frame_seg, frames_before=10)

            st.markdown("#### Ball Trajectory with Quadratic Fit (Trimmed)")
            if {'x_smooth','y_smooth','time_ball'}.issubset(trimmed_ball_df.columns):
                visualize_ball_trajectory_quadratic(
                    df=trimmed_ball_df[['time_ball','x_smooth','y_smooth']].copy(),
                    key_suffix=shot
                )
                visualize_ball_trajectory(
                    df=trimmed_ball_df[['x_smooth','y_smooth']].copy(),
                    key_suffix=shot
                )
            else:
                st.warning("Missing 'x_smooth','y_smooth','time_ball' for ball trajectory plots.")

            st.markdown("#### Ball-Specific Release Speed")
            speed_release = calculate_ball_release_speed(trimmed_ball_df, release_frame=release_frame_seg, fps_ball=20.0)
            if not pd.isna(speed_release):
                st.write(f"**Calculated Release Speed**: {speed_release:.3f} m/s")
            else:
                st.write("Could not calculate release speed from ball data.")

            st.markdown("#### Side-View Ball Trajectory (Trimmed)")
            if {'frame', 'x_smooth', 'time_ball'}.issubset(trimmed_ball_df.columns):
                visualize_ball_trajectory_side_view(
                    df=trimmed_ball_df[['frame','x_smooth','time_ball']].copy(),
                    x_col='time_ball',
                    y_col='x_smooth',
                    spline=True,
                    smoothing_val=1.2,
                    enforce_data_range=True,
                    highlight_max_speed=True,
                    key_suffix=shot
                )
            else:
                st.warning("Missing 'frame','x_smooth','time_ball' for side-view ball trajectory.")

        st.markdown("### Ball Tracking Heatmap (Aggregated)")
        if {'x_smooth','y_smooth'}.issubset(trimmed_agg.columns):
            visualize_ball_heatmap(trimmed_agg, key_suffix="aggregated")
        else:
            st.warning("Missing 'x_smooth' or 'y_smooth' columns for aggregated ball heatmap.")

        # Raw Data Sample
        st.markdown("### Raw Data Sample")
        st.dataframe(trimmed_agg.head(10))

# ------------------------------
# Run the App with Error Handling
# ------------------------------
if __name__ == "__main__":
    main()
