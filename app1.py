# ==========================================
# 1. Imports
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import boto3
import math
import logging
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
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

# ==========================================
# 2. Constants
# ==========================================
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
    
    # Metadata or other columns
    "shot_type",
    "upload_time",
    "player_name",
    "segment_id",
    "frame"
]
###############################################################################
# Body Column Mapping
###############################################################################
def map_body_columns(df):
    """Remap raw columns to standard left_shoulder_x, right_elbow_angle, etc."""
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
        'left_knee': 'left_knee_angle',
        'right_knee': 'right_knee_angle',
        # Add more mappings as needed
    }
    df = df.rename(columns=rename_dict)
    return df

ALL_SMOOTH_COLS = ALL_MARKER_COLS.copy()  # Assuming same columns require smoothing

# ==========================================
# 3. Utility Functions
# ==========================================
def compute_velocity(series, dt=0.01):
    """
    Compute velocity from a series using centered finite differences.
    
    Parameters:
    -----------
    series : pandas.Series
        The data series to compute velocity from.
    dt : float, optional
        Time interval between frames (default is 0.01 seconds).
    
    Returns:
    --------
    pandas.Series
        The computed velocity series.
    """
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

rolling_window=7

def smooth_series_custom(series, window=11, frac=0.3):
    """
    Custom smoothing function combining Savitzky-Golay and rolling mean.
    """
    # Ensure window length is odd and not more than the series length
    wsize = min(len(series), window)
    if wsize < 3:
        return series  # Not enough data to smooth
    if wsize % 2 == 0:
        wsize -= 1  # Make it odd
    try:
        sg_smoothed = savgol_filter(series.fillna(method='ffill').fillna(0), wsize, 3)
        rolled_smooth = pd.Series(sg_smoothed).rolling(window=rolling_window, center=True, min_periods=1).mean()
        return rolled_smooth
    except Exception as e:
        logger.warning(f"Error in smoothing series: {e}")
        return series

def humanize_label(label):
    """
    Convert something like 'right_elbow_angle_vel_smooth' -> 'Right Elbow Angle Vel Smooth'
    Removing underscores, capitalizing each word.
    """
    return " ".join([part.capitalize() for part in label.split("_")])

def dehumanize_label(label):
    """
    Convert 'Right Elbow Angle Vel Smooth' -> 'right_elbow_angle_vel_smooth'
    """
    return label.lower().replace(" ", "_")

def show_brand_header(player_name, team_name):
    """
    Placeholder for displaying a brand header.
    Implement as needed.
    """
    st.markdown(f"## Player: **{player_name}** - Team: **{team_name}**")

# ==========================================
# 4. Preprocessing Functions
# ==========================================
def remove_outliers_and_smooth_ball(df, x_col='x', y_col='y', zscore_thresh=3.0,
                                    sg_window=21, sg_poly=3,
                                    rolling_window=7):
    """
    Removes outliers from ball (x, y) using Z-score, then applies a two-stage smoothing:
      1) Savitzky-Golay
      2) Rolling mean
    """
    # 1) Drop rows if x or y is missing
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        st.warning("No valid ball data to smooth after dropping NaNs.")
        df['x_smooth'] = np.nan
        df['y_smooth'] = np.nan
        return df

    # 2) Compute Z-scores for outlier removal
    df['x_z'] = (df[x_col] - df[x_col].mean()) / df[x_col].std()
    df['y_z'] = (df[y_col] - df[y_col].mean()) / df[y_col].std()

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
    try:
        df['x_smooth'] = savgol_filter(df[x_col], window_length=sg_window, polyorder=sg_poly)
        df['y_smooth'] = savgol_filter(df[y_col], window_length=sg_window, polyorder=sg_poly)
    except Exception as e:
        st.warning(f"Error applying Savitzky-Golay filter: {e}")
        df['x_smooth'] = df[x_col]
        df['y_smooth'] = df[y_col]

    # 5) Second pass: Rolling mean
    df['x_smooth'] = df['x_smooth'].rolling(window=rolling_window, center=True, min_periods=1).mean()
    df['y_smooth'] = df['y_smooth'].rolling(window=rolling_window, center=True, min_periods=1).mean()

    return df
import numpy as np

def angle_2d(a, b, c):
    """
    Calculate the angle at point b given three points a, b, c in 2D.

    Parameters:
    -----------
    a, b, c : tuple
        Coordinates of points a, b, and c respectively.

    Returns:
    --------
    float
        Angle at point b in degrees.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def compute_head_midpoint(df):
    """
    Compute the midpoint between the left and right eyes to approximate the head position.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing eye coordinates.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'head_x', 'head_y', 'head_z' columns.
    """
    df['head_x'] = (df['left_eye_x'] + df['right_eye_x']) / 2
    df['head_y'] = (df['left_eye_y'] + df['right_eye_y']) / 2
    df['head_z'] = (df['left_eye_z'] + df['right_eye_z']) / 2
    return df

def compute_joint_angles(df):
    """
    Compute various joint angles from raw 3D coordinates.

    The function calculates angles for elbows, wrists, shoulders, hips, knees, ankles, and spine.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the 3D coordinates of body markers. Expected columns should follow
        the naming convention: '<body_part>_<axis>', e.g., 'left_shoulder_x', 'left_elbow_y', etc.

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame augmented with new columns for each computed joint angle.
    """
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
            'marker_a': 'left_eye',
            'marker_b': 'left_shoulder',
            'marker_c': 'left_elbow'
        },
        {
            'name': 'right_shoulder_angle',
            'marker_a': 'right_eye',
            'marker_b': 'right_shoulder',
            'marker_c': 'right_elbow'
        },
        # Wrist Angles
        {
            'name': 'left_wrist_angle',
            'marker_a': 'left_elbow',
            'marker_b': 'left_wrist',
            'marker_c': 'left_pinky'  # Ensure 'left_pinky_x', 'left_pinky_y', 'left_pinky_z' exist
        },
        {
            'name': 'right_wrist_angle',
            'marker_a': 'right_elbow',
            'marker_b': 'right_wrist',
            'marker_c': 'right_pinky'  # Ensure 'right_pinky_x', 'right_pinky_y', 'right_pinky_z' exist
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
            'marker_c': 'left_heel'  # Ensure 'left_heel_x', 'left_heel_y', 'left_heel_z' exist
        },
        {
            'name': 'right_ankle_angle',
            'marker_a': 'right_knee',
            'marker_b': 'right_ankle',
            'marker_c': 'right_heel'  # Ensure 'right_heel_x', 'right_heel_y', 'right_heel_z' exist
        },
        # Spine Angle using head midpoint
        {
            'name': 'spine_angle',
            'marker_a': 'mid_hip',
            'marker_b': 'neck',
            'marker_c': 'head'  # Uses 'head_x', 'head_y', 'head_z' computed as midpoint between eyes
        },
        # Add additional joints as needed
    ]

    for joint in joints:
        angle_col = joint['name']
        marker_a = joint['marker_a']
        marker_b = joint['marker_b']
        marker_c = joint['marker_c']

        required_cols = [
            f"{marker_a}_x", f"{marker_a}_y", f"{marker_a}_z",
            f"{marker_b}_x", f"{marker_b}_y", f"{marker_b}_z",
            f"{marker_c}_x", f"{marker_c}_y", f"{marker_c}_z"
        ]

        if all(col in df.columns for col in required_cols):
            # Compute angle using the three markers in 2D projection (X and Y)
            df[angle_col] = df.apply(
                lambda row: angle_2d(
                    (row[f"{marker_a}_x"], row[f"{marker_a}_y"]),
                    (row[f"{marker_b}_x"], row[f"{marker_b}_y"]),
                    (row[f"{marker_c}_x"], row[f"{marker_c}_y"])
                ),
                axis=1
            )
        else:
            # Assign NaN and issue a warning if any required column is missing
            df[angle_col] = np.nan
            missing = [col for col in required_cols if col not in df.columns]
            st.warning(f"Missing columns for {angle_col}: {missing}")

    return df

def compute_joint_angles_alt(df):
    """
    Alternative joint angle computation using available data when some markers are missing.
    For example, computing spine_angle using mid-eye midpoint.
    """
    # Example: Compute spine_angle using mid_hip, neck, and head midpoint
    if all(col in df.columns for col in ['mid_hip_x', 'mid_hip_y', 'mid_hip_z',
                                        'neck_x', 'neck_y', 'neck_z',
                                        'head_x', 'head_y', 'head_z']):
        df['spine_angle'] = df.apply(
            lambda row: angle_2d(
                (row['mid_hip_x'], row['mid_hip_y']),
                (row['neck_x'], row['neck_y']),
                (row['head_x'], row['head_y'])
            ),
            axis=1
        )
    else:
        df['spine_angle'] = np.nan
        missing = [col for col in ['mid_hip_x', 'mid_hip_y', 'mid_hip_z',
                                   'neck_x', 'neck_y', 'neck_z',
                                   'head_x', 'head_y', 'head_z'] if col not in df.columns]
        if missing:
            st.warning(f"Missing columns for spine_angle: {missing}")
    return df


from scipy.signal import savgol_filter

def compute_velocity(series, dt=1.0/25):
    """
    Compute velocity using centered finite differences.
    
    Parameters:
    -----------
    series : pandas.Series
        The data series to compute velocity from.
    dt : float
        Time interval between frames (seconds).
    
    Returns:
    --------
    pandas.Series
        The computed velocity series.
    """
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def compute_acceleration(series, dt=1.0/25):
    """
    Compute acceleration using centered finite differences.
    
    Parameters:
    -----------
    series : pandas.Series
        The velocity series to compute acceleration from.
    dt : float
        Time interval between frames (seconds).
    
    Returns:
    --------
    pandas.Series
        The computed acceleration series.
    """
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def preprocess_and_smooth(df, sg_window=81, sg_poly=3, fps=25):
    """
    Applies smoothing and computes velocity & acceleration for relevant columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing raw motion data.
    sg_window : int
        Window size for Savitzky-Golay filter.
    sg_poly : int
        Polynomial order for Savitzky-Golay filter.
    fps : float
        Frames per second of the data.
    
    Returns:
    --------
    df : pandas.DataFrame
        The processed DataFrame with smoothed, velocity, and acceleration columns.
    """
    # Define columns to smooth
    angle_smooth_cols = [col for col in df.columns if '_angle_vel' in col]
    ball_position_cols = ['basketball_x', 'basketball_y']  # Adjust if different
    position_cols = [
        'right_elbow_y',
        'left_hip_flex', 'right_hip_flex',
        'left_shoulder_elev', 'right_shoulder_elev',
        'mid_hip_x', 'mid_hip_y'  # Updated to match mapped names
        # Add other position columns as needed
    ]

    # Smoothing
    for col in angle_smooth_cols + ball_position_cols + position_cols:
        smooth_col = f"{col}_smooth"
        if col in df.columns:
            # Ensure there are enough data points for smoothing
            window = min(sg_window, len(df[col].dropna()))
            if window < sg_poly + 2:
                window = sg_poly + 2
            if window % 2 == 0:
                window += 1  # Window size must be odd
            try:
                df[smooth_col] = savgol_filter(
                    df[col].fillna(method='ffill').fillna(method='bfill'), 
                    window_length=window, 
                    polyorder=sg_poly
                )
            except Exception as e:
                st.warning(f"Error smoothing column '{col}': {e}")
                df[smooth_col] = df[col]  # Fallback to unsmoothed data
        else:
            st.warning(f"Column '{col}' not found in DataFrame.")
            df[f"{col}_smooth"] = np.nan

    # Compute Velocity and Acceleration for Angle Columns
    velocity_cols = []
    acceleration_cols = []
    
    for col in angle_smooth_cols:
        smooth_col = f"{col}_smooth"
        vel_col = f"{col}_vel"
        accel_col = f"{col}_accel"
        
        if smooth_col in df.columns:
            df[vel_col] = compute_velocity(df[smooth_col], dt=1.0/fps)
            df[accel_col] = compute_acceleration(df[vel_col], dt=1.0/fps)
            velocity_cols.append(vel_col)
            acceleration_cols.append(accel_col)
        else:
            st.warning(f"Smoothed column '{smooth_col}' not found. Cannot compute velocity and acceleration.")
            df[vel_col] = np.nan
            df[accel_col] = np.nan

    # Compute Velocity and Acceleration for Position Columns
    for col in position_cols:
        smooth_col = f"{col}_smooth"
        vel_col = f"{col}_vel"
        accel_col = f"{col}_accel"
        
        if smooth_col in df.columns:
            df[vel_col] = compute_velocity(df[smooth_col], dt=1.0/fps)
            df[accel_col] = compute_acceleration(df[vel_col], dt=1.0/fps)
            velocity_cols.append(vel_col)
            acceleration_cols.append(accel_col)
        else:
            st.warning(f"Smoothed column '{smooth_col}' not found. Cannot compute velocity and acceleration.")
            df[vel_col] = np.nan
            df[accel_col] = np.nan

    # Compute Mean Velocity Columns
    if {'left_elbow_angle_vel', 'right_elbow_angle_vel'}.issubset(df.columns):
        df['mean_elbow_angle_vel'] = df[['left_elbow_angle_vel', 'right_elbow_angle_vel']].mean(axis=1)
    else:
        st.warning("Missing 'left_elbow_angle_vel' or 'right_elbow_angle_vel' columns. 'mean_elbow_angle_vel' cannot be computed.")
        df['mean_elbow_angle_vel'] = np.nan

    if {'left_knee_angle_vel', 'right_knee_angle_vel'}.issubset(df.columns):
        df['mean_knee_angle_vel'] = df[['left_knee_angle_vel', 'right_knee_angle_vel']].mean(axis=1)
    else:
        st.warning("Missing 'left_knee_angle_vel' or 'right_knee_angle_vel' columns. 'mean_knee_angle_vel' cannot be computed.")
        df['mean_knee_angle_vel'] = np.nan

    if {'left_hip_flex_vel', 'right_hip_flex_vel'}.issubset(df.columns):
        df['mean_hip_angle_vel'] = df[['left_hip_flex_vel', 'right_hip_flex_vel']].mean(axis=1)
    else:
        st.warning("Missing 'left_hip_flex_vel' or 'right_hip_flex_vel' columns. 'mean_hip_angle_vel' cannot be computed.")
        df['mean_hip_angle_vel'] = np.nan

    if {'left_shoulder_elev_vel', 'right_shoulder_elev_vel'}.issubset(df.columns):
        df['mean_shoulder_angle_vel'] = df[['left_shoulder_elev_vel', 'right_shoulder_elev_vel']].mean(axis=1)
    else:
        st.warning("Missing 'left_shoulder_elev_vel' or 'right_shoulder_elev_vel' columns. 'mean_shoulder_angle_vel' cannot be computed.")
        df['mean_shoulder_angle_vel'] = np.nan

    # Fill NaN values
    for col in velocity_cols + acceleration_cols + [
        'mean_elbow_angle_vel', 'mean_knee_angle_vel', 
        'mean_hip_angle_vel', 'mean_shoulder_angle_vel'
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)

    return df




def validate_and_compute_velocities(df):
    """
    Validates that all necessary velocity columns are present. Computes them if missing.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate and compute velocities.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with all necessary velocity columns.
    """
    required_velocity_cols = [
        'left_knee_angle_vel', 
        'right_knee_angle_vel',
        'left_hip_flex_vel', 
        'right_hip_flex_vel',
        'left_shoulder_elev_vel', 
        'right_shoulder_elev_vel',
        'left_elbow_angle_vel', 
        'right_elbow_angle_vel'
    ]

    for col in required_velocity_cols:
        if col not in df.columns:
            # Attempt to compute the velocity column
            position_col = col.replace('_vel', '')
            if position_col in df.columns:
                df[col] = df[position_col].diff() * 25  # Assuming fps=25
                df[col] = df[col].fillna(method='ffill').fillna(0)
                st.write(f"Computed missing velocity column: {col}")
            else:
                st.warning(f"Cannot compute velocity for missing position column: {position_col}. Setting {col} to 0.")
                df[col] = 0

    return df


# ==========================================
# 5. Metric Computation Functions
# ==========================================
def compute_elbow_range(df):
    """
    Compute the total range of the elbow angle as a proxy for "shooting motion".
    """
    angle_cols = ['left_elbow_angle_smooth', 'right_elbow_angle_smooth']
    ranges = []
    for col in angle_cols:
        if col in df.columns and not df[col].dropna().empty:
            ranges.append(df[col].max() - df[col].min())
    return max(ranges) if ranges else 0.0

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

def compute_cohesion_score(df, angle_cols=None, consider_chain=False, chain_cols=None):
    """
    Compute a Cohesion Score to represent consistency in a set of angle profiles.
    """
    if angle_cols is None:
        angle_cols = [
            'right_elbow_angle_smooth',
            'right_knee_angle_smooth',
            'left_elbow_angle_smooth',
            'left_knee_angle_smooth',
        ]
    valid_angle_cols = [c for c in angle_cols if c in df.columns]
    if not valid_angle_cols:
        base_score = np.nan
    else:
        total_std = df[valid_angle_cols].std().sum()
        max_std = 30.0  # Domain-based guess or calibration
        base_score = 1.0 - (total_std / max_std)
        base_score = float(np.clip(base_score, 0, 1))

    if not consider_chain:
        return base_score

    # Kinematic chain approach
    if chain_cols is None:
        chain_cols = [
            'left_elbow_angle_vel_smooth',
            'right_elbow_angle_vel_smooth',
            'left_knee_angle_vel_smooth',
            'right_knee_angle_vel_smooth'
        ]
    valid_chain_cols = [c for c in chain_cols if c in df.columns]
    if len(valid_chain_cols) < 2:
        chain_score = 1.0  # Not enough data to penalize
    else:
        peak_frames = []
        for c in valid_chain_cols:
            arr = df[c].abs()
            peak_idx = arr.idxmax()
            peak_frames.append(df.loc[peak_idx, 'frame'])

        diffs = [abs(peak_frames[i+1] - peak_frames[i]) for i in range(len(peak_frames)-1)]
        if not diffs:
            chain_score = 1.0
        else:
            sd_diffs = np.std(diffs)
            max_chain_sd = 10.0
            chain_score = 1.0 - (sd_diffs / max_chain_sd)
            chain_score = float(np.clip(chain_score, 0, 1))

    # Combine scores
    w1, w2 = 0.5, 0.5
    if np.isnan(base_score):
        cohesion_score = chain_score
    else:
        cohesion_score = w1 * base_score + w2 * chain_score
    return float(np.clip(cohesion_score, 0, 1))

def compute_kinetic_chain_efficiency(df, release_frame):
    """
    Computes Kinetic Chain Efficiency based on angular velocities at release.
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
    return sum_vel  # Can be scaled as needed

def compute_ball_trajectory_curvature(df, release_frame):
    """
    Compute the curvature of the ball's trajectory at the release frame.
    """
    try:
        if not {'frame', 'x_smooth', 'y_smooth'}.issubset(df.columns):
            return np.nan

        release_idx = df.index[df['frame'] == release_frame].tolist()
        if not release_idx:
            return np.nan
        release_idx = release_idx[0]

        if release_idx == 0 or release_idx == len(df) -1:
            return np.nan

        x1, y1 = df.loc[release_idx -1, 'x_smooth'], df.loc[release_idx -1, 'y_smooth']
        x2, y2 = df.loc[release_idx, 'x_smooth'], df.loc[release_idx, 'y_smooth']
        x3, y3 = df.loc[release_idx +1, 'x_smooth'], df.loc[release_idx +1, 'y_smooth']

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
    Compute various advanced biomechanical metrics at the release frame.
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

    # Ball Speed at release
    if 'ball_speed_smooth' in df.columns and not pd.isna(row['ball_speed_smooth']):
        metrics['Release Speed (m/s)'] = float(row['ball_speed_smooth'])
    else:
        metrics['Release Speed (m/s)'] = np.nan

    # Asymmetry Index
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

    # Center of Mass Speed
    if 'com_speed' in df.columns and not pd.isna(row['com_speed']):
        metrics['CoM Speed (m/s)'] = row['com_speed']
    else:
        metrics['CoM Speed (m/s)'] = np.nan

    # Ball Trajectory Curvature
    metrics['Ball Trajectory Curvature'] = compute_ball_trajectory_curvature(df, release_frame)

    # Release Angle
    if 'vx' in df.columns and 'vy' in df.columns and not pd.isna(row['vx']) and not pd.isna(row['vy']):
        angle_radians = math.atan2(row['vy'], row['vx'])  # relative to +x axis
        angle_degrees = math.degrees(angle_radians)
        metrics['Release Angle (°)'] = angle_degrees
    else:
        metrics['Release Angle (°)'] = np.nan

    # Shot Distance
    rim_x = 1000.0  # Adjust based on your coordinate system
    rim_y = 300.0
    if 'x_smooth' in df.columns and 'y_smooth' in df.columns:
        bx = row['x_smooth']
        by = row['y_smooth']
        dist = math.sqrt((rim_x - bx)**2 + (rim_y - by)**2)
        metrics['Shot Distance'] = dist
    else:
        metrics['Shot Distance'] = np.nan

    # Apex Height
    df_after = df[df['frame'] >= release_frame]
    if not df_after.empty and 'y_smooth' in df.columns:
        apex_height = float(df_after['y_smooth'].max())
        metrics['Apex Height'] = apex_height
    else:
        metrics['Apex Height'] = np.nan

    # Release Time
    first_frame = df['frame'].min()
    release_frame_num = release_frame
    dt_frames = release_frame_num - first_frame
    metrics['Release Time (s)'] = dt_frames / fps

    # Release Quality
    if "spin_x" in df.columns:
        rng = range(release_frame-2, release_frame+3)
        if all(frame in df['frame'].values for frame in rng):
            sub = df[df['frame'].isin(rng)]['spin_x']
            metrics["Release Quality"] = float(sub.mean())
        else:
            metrics["Release Quality"] = np.nan
    else:
        metrics["Release Quality"] = np.nan

    # Release Curvature
    if 'curvature' in df.columns:
        curvature_subset = df.loc[df['frame'] == release_frame, 'curvature']
        if not curvature_subset.empty:
            metrics["Release Curvature"] = float(curvature_subset.mean())
        else:
            metrics["Release Curvature"] = np.nan
    else:
        metrics["Release Curvature"] = np.nan

    # Lateral Release Curvature (Placeholder)
    metrics["Lateral Release Curvature"] = np.nan  # To be implemented based on data

    return metrics

def compute_center_of_mass(df, frame_idx, segment_markers, segment_masses):
    """
    Computes the Center of Mass (CoM) for a given frame using segment markers and their masses.
    """
    try:
        COM_sum = np.zeros(3, dtype=float)
        total_mass = sum(segment_masses.values())

        for seg_name, (markerA, markerB) in segment_markers.items():
            required_cols = [f"{markerA}_x", f"{markerA}_y", f"{markerA}_z",
                            f"{markerB}_x", f"{markerB}_y", f"{markerB}_z"]

            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for segment '{seg_name}'. Skipping.")
                continue

            try:
                # Use iloc to access by integer position
                A_x = df.iloc[frame_idx][f"{markerA}_x"]
                A_y = df.iloc[frame_idx][f"{markerA}_y"]
                A_z = df.iloc[frame_idx][f"{markerA}_z"]
                B_x = df.iloc[frame_idx][f"{markerB}_x"]
                B_y = df.iloc[frame_idx][f"{markerB}_y"]
                B_z = df.iloc[frame_idx][f"{markerB}_z"]
            except IndexError:
                logger.warning(f"Frame index {frame_idx} out of bounds.")
                return np.array([np.nan, np.nan, np.nan])

            if pd.isna(A_x) or pd.isna(A_y) or pd.isna(A_z) or pd.isna(B_x) or pd.isna(B_y) or pd.isna(B_z):
                logger.warning(f"NaN values in markers for segment '{seg_name}' at frame {frame_idx}. Skipping.")
                continue

            seg_com = 0.5 * (np.array([A_x, A_y, A_z]) + np.array([B_x, B_y, B_z]))
            mass = segment_masses[seg_name]
            COM_sum += seg_com * mass

        if total_mass == 0:
            logger.warning("Total mass is zero. Cannot compute Center of Mass.")
            return np.array([np.nan, np.nan, np.nan])

        return COM_sum / total_mass
    except Exception as e:
        logger.error(f"Error computing Center of Mass: {e}")
        return np.array([np.nan, np.nan, np.nan])

def validate_columns(df, required_columns):
    """
    Validates that all required columns are present in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate.
    required_columns : list
        List of required column names.

    Returns:
    --------
    bool
        True if all required columns are present, False otherwise.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning(f"Missing required columns: {missing}")
        return False
    return True

def compute_all_com(df, fps_pose=25.0):
    """
    Computes the Center of Mass (CoM) over time for the entire dataset.
    """
    segment_markers = {
        "left_arm": ("left_shoulder", "left_wrist"),
        "right_arm": ("right_shoulder", "right_wrist"),
        "left_leg": ("left_hip", "left_ankle"),
        "right_leg": ("right_hip", "right_ankle"),
        "trunk": ("midhip", "neck")
    }
    segment_masses = {
        "left_arm": 0.05,
        "right_arm": 0.05,
        "left_leg": 0.15,
        "right_leg": 0.15,
        "trunk": 0.60
    }

    n_frames = len(df)
    com_x = np.zeros(n_frames)
    com_y = np.zeros(n_frames)
    com_z = np.zeros(n_frames)

    for i in range(n_frames):
        com_xyz = compute_center_of_mass(df, i, segment_markers, segment_masses)
        com_x[i] = com_xyz[0]
        com_y[i] = com_xyz[1]
        com_z[i] = com_xyz[2]

    # Compute COM velocity
    vx = compute_velocity(pd.Series(com_x), dt=1.0/fps_pose)
    vy = compute_velocity(pd.Series(com_y), dt=1.0/fps_pose)
    vz = compute_velocity(pd.Series(com_z), dt=1.0/fps_pose)
    com_speed = np.sqrt(vx**2 + vy**2 + vz**2)

    df['com_x'] = com_x
    df['com_y'] = com_y
    df['com_z'] = com_z
    df['com_speed'] = com_speed

    return com_x, com_y, com_z, com_speed

def compute_temporal_consistency(df):
    """
    Computes Temporal Consistency based on the standard deviation of phase durations.
    Lower standard deviation indicates higher consistency.
    Returns a normalized score between 0 and 1.
    """
    phase_durations = df['phase'].value_counts().reindex(['Ball Elevation', 'Stability', 'Release', 'Inertia']).fillna(0)
    sd_durations = phase_durations.std()
    mean_durations = phase_durations.mean()

    if mean_durations == 0:
        return 0.0

    consistency_score = 1 - (sd_durations / mean_durations)
    consistency_score = max(min(consistency_score, 1.0), 0.0)

    return consistency_score

def compute_cyclical_asymmetry_index(df):
    """
    Computes Cyclical Asymmetry Index based on differences between left and right joint angles.
    Returns a normalized score between 0 and 1 (lower indicates higher symmetry).
    """
    joints = ['elbow_angle', 'knee_angle', 'shoulder_angle', 'hip_angle']
    asymmetry_scores = []

    for joint in joints:
        left_col = f'left_{joint}'
        right_col = f'right_{joint}'
        if left_col in df.columns and right_col in df.columns:
            diff = (df[left_col] - df[right_col]).abs()
            mean_angle = df[[left_col, right_col]].mean().mean()
            norm_diff = diff.mean() / max(mean_angle, 1)
            asymmetry_scores.append(norm_diff)

    if not asymmetry_scores:
        return np.nan

    avg_asymmetry = np.mean(asymmetry_scores)
    cyclical_asymmetry_index = 1 - min(avg_asymmetry / 0.5, 1.0)

    return cyclical_asymmetry_index

def compute_posture_analysis(df):
    """
    Computes Posture Analysis metrics such as Torso Angle and Hip Alignment.
    Returns a dictionary with metric names and their computed values.
    """
    posture_metrics = {}

    # Torso Angle
    if all(col in df.columns for col in ['neck_x', 'neck_y', 'neck_z', 'midhip_x', 'midhip_y', 'midhip_z']):
        df['torso_angle'] = df.apply(lambda row: calculate_torso_angle(row), axis=1)
        avg_torso_angle = df['torso_angle'].mean()
        posture_metrics['Torso Angle'] = avg_torso_angle

    # Hip Alignment
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
    mid_hip = np.array([row['midhip_x'], row['midhip_y'], row['midhip_z']])
    vector = neck - mid_hip
    vertical = np.array([0, 1, 0])

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
    if 'midhip_x' not in df.columns:
        return np.nan
    start_x = df.iloc[0]['midhip_x']
    end_x = df.iloc[-1]['midhip_x']
    stride_length = abs(end_x - start_x)
    return stride_length

# ==========================================
# 6. Visualization Functions
# ==========================================
def visualize_angular_acceleration(df, acceleration_cols, key_suffix=""):
    """
    Visualizes angular accelerations for selected joints over time.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing angular accelerations.
    acceleration_cols : list
        List of column names for angular accelerations.
    key_suffix : str
        Suffix for Streamlit component keys to avoid duplication.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for acc_col in acceleration_cols:
        ax.plot(df['frame'], df[acc_col], label=acc_col.replace('_acc', ' ').title())
    ax.axvline(x=df['release_frame'].iloc[0], color='red', linestyle='--', label='Release Frame')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angular Acceleration (deg/s²)")
    ax.set_title("Angular Accelerations Over Time")
    ax.legend()
    st.pyplot(fig, use_container_width=True, key=f"angular_acceleration_{key_suffix}")

def visualize_joint_angles(df, right_angle_col, left_angle_col, key_suffix=""):
    """
    Visualizes the right and left joint angles over time.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the joint angles.
    right_angle_col : str
        Column name for the right joint angle.
    left_angle_col : str
        Column name for the left joint angle.
    key_suffix : str
        Suffix for Streamlit component keys to avoid duplication.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['frame'], df[right_angle_col], label=right_angle_col.replace('_', ' ').title())
    ax.plot(df['frame'], df[left_angle_col], label=left_angle_col.replace('_', ' ').title())
    ax.axvline(x=df['release_frame'].iloc[0], color='red', linestyle='--', label='Release Frame')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Joint Angles Over Time")
    ax.legend()
    st.pyplot(fig, use_container_width=True, key=f"joint_angles_{key_suffix}")

def visualize_angular_velocity(df, velocity_cols, key_suffix=""):
    """
    Visualizes angular velocities for selected joints over time.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing angular velocities.
    velocity_cols : list
        List of column names for angular velocities.
    key_suffix : str
        Suffix for Streamlit component keys to avoid duplication.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for vel_col in velocity_cols:
        ax.plot(df['frame'], df[vel_col], label=vel_col.replace('_vel', ' ').title())
    ax.axvline(x=df['release_frame'].iloc[0], color='red', linestyle='--', label='Release Frame')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angular Velocity (deg/s)")
    ax.set_title("Angular Velocities Over Time")
    ax.legend()
    st.pyplot(fig, use_container_width=True, key=f"angular_velocity_{key_suffix}")

def visualize_phases(df, phases, key_suffix=""):
    """
    Visualizes the biomechanical phases on a timeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'frame' and 'phase' columns.
    phases : list
        List of phase names to visualize.
    key_suffix : str
        Suffix for Streamlit component keys to avoid duplication.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 2))

    phase_colors = {
        "Preparation": "blue",
        "Ball Elevation": "green",
        "Stability": "orange",
        "Release": "red",
        "Inertia": "purple",
        "Undefined": "gray"
    }

    for phase in phases:
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            continue
        ax.plot(phase_df['frame'], [1]*len(phase_df), color=phase_colors.get(phase, "black"), label=phase, linewidth=10)

    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    ax.set_title("Biomechanical Phases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    st.pyplot(fig, use_container_width=True)


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
        for phase in ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']:
            ph_df = df[df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation': 'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',
                    'Stability': 'rgba(255,255,0,0.1)',
                    'Release': 'rgba(255,0,0,0.1)',
                    'Inertia': 'rgba(128,0,128,0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start_f,
                    x1=end_f,
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

def visualize_angular_velocity(df, release_frame, selected_velocities, phases_df, key_suffix=""):
    """
    Plots selected angular velocities over frames with phase overlays and release point.
    """
    if not selected_velocities:
        st.warning("No angular velocities selected for plotting.")
        return

    fig = go.Figure()

    for vel_label in selected_velocities:
        col = dehumanize_label(vel_label)
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
            ph_df = phases_df[ph_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation': 'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',
                    'Stability': 'rgba(255,255,0,0.1)',
                    'Release': 'rgba(255,0,0,0.1)',
                    'Inertia': 'rgba(128,0,128,0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start_f,
                    x1=end_f,
                    fillcolor=fill_color,
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

def visualize_accelerations(df, release_frame, selected_accelerations, phases_df, key_suffix=""):
    """
    Plots selected angular accelerations over frames with phase overlays and release point.
    """
    if not selected_accelerations:
        st.warning("No angular accelerations selected for plotting.")
        return

    fig = go.Figure()

    for acc_label in selected_accelerations:
        acc_col = dehumanize_label(acc_label)
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
            ph_df = phases_df[ph_df['phase'] == phase]
            if not ph_df.empty:
                start_f = ph_df['frame'].min()
                end_f = ph_df['frame'].max()
                color_map = {
                    'Preparation': 'rgba(0,128,255,0.1)',
                    'Ball Elevation': 'rgba(255,165,0,0.1)',
                    'Stability': 'rgba(255,255,0,0.1)',
                    'Release': 'rgba(255,0,0,0.1)',
                    'Inertia': 'rgba(128,0,128,0.1)'
                }
                fill_color = color_map.get(phase, 'rgba(128,128,128,0.1)')
                fig.add_vrect(
                    x0=start_f,
                    x1=end_f,
                    fillcolor=fill_color,
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

def visualize_kinematic_chain_sequence(df, release_frame=None, phases_df=None, key_suffix=""):
    """
    Visualize the Kinematic Chain Sequence, such as angular velocities of key joints.
    """
    key_joints = ['Left Elbow Angle Velocity', 'Right Elbow Angle Velocity',
                  'Left Knee Angle Velocity', 'Right Knee Angle Velocity']

    fig = go.Figure()

    for joint in key_joints:
        col = dehumanize_label(joint) + '_smooth'
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
            ph_df = phases_df[ph_df['phase'] == phase]
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

def visualize_ball_trajectory_quadratic(df, key_suffix=""):
    """
    Visualizes the ball's X position with a quadratic fit and marks the point of maximum increase.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['time_ball'], df['x_smooth'], 2)
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
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_quadratic_{key_suffix}")

def visualize_ball_trajectory(df, key_suffix=""):
    """
    Visualizes the ball's 2D trajectory with a quadratic fit.
    """
    if df.empty:
        st.warning("No data to plot for ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['x_smooth'], df['y_smooth'], 2)
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
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_{key_suffix}")

def visualize_ball_heatmap(df, x_col='x_smooth', y_col='y_smooth', key_suffix=""):
    """
    Creates a heatmap of the ball's 2D positions to show frequency of locations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ball position data.
    x_col : str, optional
        Column name for the x-axis. Default is 'x_smooth'.
    y_col : str, optional
        Column name for the y-axis. Default is 'y_smooth'.
    key_suffix : str, optional
        Suffix for Streamlit component keys to avoid duplication.
    """
    if df.empty:
        st.warning("No data to plot for ball heatmap.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame.")
        return

    fig = px.density_heatmap(
        df,
        x=x_col,
        y=y_col,
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


def visualize_ball_trajectory_side_view(df, key_suffix=""):
    """
    Visualizes the side-view of the ball's X position over time.
    Highlights the region with the maximum increase in X position.
    """
    if df.empty:
        st.warning("No data to plot in side-view trajectory.")
        return

    if not {'frame', 'x_smooth', 'time_ball'}.issubset(df.columns):
        st.warning("Missing required columns for side-view ball trajectory.")
        return

    try:
        coefficients = np.polyfit(df['time_ball'], df['x_smooth'], 2)
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
        mode='lines+markers',
        name='Side-View X',
        line=dict(color='blue', shape='spline', smoothing=1.2)
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
        title="Side-View Ball X Position Over Time",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ball_trajectory_side_view_{key_suffix}")

###############################################################################
# Utility Functions: S3, DynamoDB, etc.
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
    
def compute_ball_speed_and_direction(df, fps_ball=20.0):
    """
    Computes ball speed and direction based on its position over time.
    Adds 'ball_speed' and 'ball_direction' columns to the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data with 'basketball_x_smooth' and 'basketball_y_smooth' columns.
    fps_ball : float
        Frames per second for ball data.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with added 'ball_speed' and 'ball_direction' columns.
    """
    required_cols = ['basketball_x_smooth', 'basketball_y_smooth']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing 'basketball_x_smooth' or 'basketball_y_smooth' columns for ball speed and direction calculation.")
        df['ball_speed'] = np.nan
        df['ball_direction'] = np.nan
        return df

    # Calculate differences
    df['ball_dx'] = df['basketball_x_smooth'].diff()
    df['ball_dy'] = df['basketball_y_smooth'].diff()

    # Calculate speed
    df['ball_speed'] = np.sqrt(df['ball_dx']**2 + df['ball_dy']**2) * fps_ball  # Assuming units are meters/frame

    # Calculate direction (angle in degrees relative to +x axis)
    df['ball_direction'] = np.degrees(np.arctan2(df['ball_dy'], df['ball_dx']))

    # Handle NaN values
    df['ball_speed'] = df['ball_speed'].fillna(0)
    df['ball_direction'] = df['ball_direction'].fillna(0)

    return df



def trim_data_around_release(df, release_frame, before=30, after=20):
    """
    Trims the DataFrame to include only a window around the release frame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original DataFrame with 'frame' column.
    release_frame : int
        Frame number indicating the release point.
    before : int
        Number of frames before the release frame to include.
    after : int
        Number of frames after the release frame to include.

    Returns:
    --------
    trimmed_df : pandas.DataFrame
        Trimmed DataFrame.
    """
    # Exclude the first 10 frames
    df = df[df['frame'] >= 10]

    start_frame = release_frame - before
    end_frame = release_frame + after

    # Ensure start_frame is not less than 10
    start_frame = max(start_frame, 10)

    # Trim the DataFrame
    trimmed_df = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()

    st.write(f"Data trimmed to frames {start_frame} to {end_frame}.")

    return trimmed_df


def compute_velocity_acceleration(df, fps=25):
    """
    Computes velocity and acceleration for all smoothed columns in the DataFrame.
    """
    # Identify smoothed columns
    smooth_cols = [col for col in df.columns if col.endswith('_smooth')]
    
    for col in smooth_cols:
        vel_col = f"{col}_vel"
        acc_col = f"{col}_accel"
        
        # Compute velocity
        df[vel_col] = compute_velocity(df[col], dt=1.0/fps)
        # Compute acceleration
        df[acc_col] = compute_acceleration(df[vel_col], dt=1.0/fps)
    
    # Compute mean velocity columns
    if {'left_elbow_angle_vel', 'right_elbow_angle_vel'}.issubset(df.columns):
        df['mean_elbow_angle_vel'] = df[['left_elbow_angle_vel', 'right_elbow_angle_vel']].mean(axis=1)
    else:
        df['mean_elbow_angle_vel'] = np.nan
        st.warning("Missing 'left_elbow_angle_vel' or 'right_elbow_angle_vel' columns.")
    
    # Similarly for other mean velocity columns
    if {'left_knee_angle_vel', 'right_knee_angle_vel'}.issubset(df.columns):
        df['mean_knee_angle_vel'] = df[['left_knee_angle_vel', 'right_knee_angle_vel']].mean(axis=1)
    else:
        df['mean_knee_angle_vel'] = np.nan
        st.warning("Missing 'left_knee_angle_vel' or 'right_knee_angle_vel' columns.")
    
    if {'left_hip_flex_vel', 'right_hip_flex_vel'}.issubset(df.columns):
        df['mean_hip_angle_vel'] = df[['left_hip_flex_vel', 'right_hip_flex_vel']].mean(axis=1)
    else:
        df['mean_hip_angle_vel'] = np.nan
        st.warning("Missing 'left_hip_flex_vel' or 'right_hip_flex_vel' columns.")
    
    if {'left_shoulder_elev_vel', 'right_shoulder_elev_vel'}.issubset(df.columns):
        df['mean_shoulder_angle_vel'] = df[['left_shoulder_elev_vel', 'right_shoulder_elev_vel']].mean(axis=1)
    else:
        df['mean_shoulder_angle_vel'] = np.nan
        st.warning("Missing 'left_shoulder_elev_vel' or 'right_shoulder_elev_vel' columns.")
    
    # Handle NaN values by forward filling
    velocity_columns = [col for col in df.columns if col.endswith('_vel')]
    for col in velocity_columns:
        df[col] = df[col].fillna(method='ffill').fillna(0)
    
    return df


def detect_release_frame_elbow(df, elbow_y_col='right_elbow_y_smooth'):
    """
    Detects the release frame based on the peak Y-position of the elbow joint.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data.
    elbow_y_col : str
        Column name for the elbow Y-position.

    Returns:
    --------
    release_frame : int or None
        Frame number indicating the release point, or None if not detected.
    """
    if elbow_y_col not in df.columns:
        st.warning(f"Column '{elbow_y_col}' not found for release frame detection.")
        return None

    # Find the frame with the maximum elbow Y-position
    peak_idx = df[elbow_y_col].idxmax()
    release_frame = df.loc[peak_idx, 'frame']
    st.write(f"Release Frame Detected Based on Elbow Y-Position: {release_frame}")

    return release_frame


    
def detect_release_frame(df, fps_ball=20.0):
    """
    Enhanced release frame detection using ball movement direction and speed.
    
    Criteria:
    - Ball stops moving left and starts moving right.
    - Significant increase in ball speed.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data with 'frame', 'x_smooth', 'y_smooth' columns.
    fps_ball : float
        Frames per second of ball data.
    
    Returns:
    --------
    release_frame : int or None
        Frame number indicating the release point, or None if not detected.
    """
    required_cols = ['frame', 'x_smooth', 'y_smooth']
    if not all(col in df.columns for col in required_cols):
        st.warning("Required columns for release detection are missing.")
        return None

    # Compute ball velocity
    df['ball_dx'] = df['x_smooth'].diff()
    df['ball_dy'] = df['y_smooth'].diff()
    
    # Calculate speed and direction
    df['ball_speed'] = np.sqrt(df['ball_dx']**2 + df['ball_dy']**2) * fps_ball  # assuming units are meters/frame
    df['ball_direction'] = np.degrees(np.arctan2(df['ball_dy'], df['ball_dx']))
    
    # Define direction: moving right is direction > 0
    df['moving_right'] = df['ball_direction'] > 0
    
    # Identify frames where ball starts moving right
    df['direction_change'] = df['moving_right'].diff()
    
    # Define speed threshold (e.g., top 25% speed)
    speed_threshold = df['ball_speed'].quantile(0.75)
    
    # Identify release candidates: direction changes to right and speed exceeds threshold
    release_candidates = df[
        (df['direction_change'] == 1) & 
        (df['ball_speed'] > speed_threshold)
    ]
    
    if release_candidates.empty:
        st.warning("No release frame detected based on ball movement.")
        return None
    
    # Select the first candidate as release frame
    release_frame = release_candidates['frame'].iloc[0]
    st.write(f"Release Frame Detected Based on Ball Movement: {release_frame}")
    
    return release_frame


def identify_biomech_phases(df, fps=25):
    """
    Identifies biomechanical phases within the DataFrame based on joint angles and positions.
    
    Phases:
    1. Preparation
    2. Ball Elevation
    3. Stability
    4. Release
    5. Inertia
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data with at least 'frame' and 'player_id' columns.
    fps : float
        Frames per second of the data.
    
    Returns:
    --------
    df_with_phases : pandas.DataFrame
        Original DataFrame with an added 'phase' column indicating biomechanical phase per frame.
    """
    df_with_phases = df.copy()
    df_with_phases['phase'] = "Preparation"  # Default phase
    
    # Check if 'player_id' exists
    if 'player_id' in df.columns:
        players = df['player_id'].unique()
    else:
        players = [None]  # Single player case
    
    for player in players:
        if player is not None:
            player_df = df_with_phases[df_with_phases['player_id'] == player]
            st.write(f"Processing phases for Player ID: {player}")
        else:
            player_df = df_with_phases
            st.write("Processing phases for Single Player")
    
        # Detect release frame
        release_frame = detect_release_frame(player_df)
        if release_frame is None:
            st.warning(f"Release frame could not be detected for player '{player}'.")
            continue  # Skip phase assignment for this player
    
        # Define biomechanical criteria based on joint angles and positions
    
        # Preparation Phase:
        # Identify when knees, hips, and shoulders start to lower (angles increasing)
        # We'll define a frame as the end of preparation when the mean velocity of knee, hip, and shoulder angles exceeds a positive threshold
    
        knee_vel_threshold = 1.0  # degrees per second
        hip_vel_threshold = 1.0
        shoulder_vel_threshold = 1.0
    
        # Calculate mean velocities
        if {'mean_knee_angle_vel', 'mean_hip_angle_vel', 'mean_shoulder_angle_vel'}.issubset(player_df.columns):
            mask_prep_end = (
                (player_df['mean_knee_angle_vel'] > knee_vel_threshold) &
                (player_df['mean_hip_angle_vel'] > hip_vel_threshold) &
                (player_df['mean_shoulder_angle_vel'] > shoulder_vel_threshold)
            )
            if mask_prep_end.any():
                prep_end_frame = player_df.loc[mask_prep_end, 'frame'].min()
                st.write(f"Preparation phase ends at frame {prep_end_frame} for player '{player}'.")
            else:
                prep_end_frame = release_frame  # If no significant increase detected, set end to release
                st.warning(f"No clear end to Preparation phase detected for player '{player}'. Setting end to Release frame.")
        else:
            prep_end_frame = release_frame
            st.warning(f"Necessary velocity columns missing for player '{player}'. Setting end of Preparation to Release frame.")
    
        # Ball Elevation Phase:
        # Detect when the player starts to rise (angles start to decrease)
        # We'll define the start of Ball Elevation when mean velocities become negative beyond a threshold
    
        knee_neg_vel_threshold = -1.0  # degrees per second
        hip_neg_vel_threshold = -1.0
        shoulder_neg_vel_threshold = -1.0
    
        if {'mean_knee_angle_vel', 'mean_hip_angle_vel', 'mean_shoulder_angle_vel'}.issubset(player_df.columns):
            mask_ball_elev_start = (
                (player_df['mean_knee_angle_vel'] < knee_neg_vel_threshold) &
                (player_df['mean_hip_angle_vel'] < hip_neg_vel_threshold) &
                (player_df['mean_shoulder_angle_vel'] < shoulder_neg_vel_threshold)
            )
            # Ensure that the Ball Elevation starts after Preparation ends
            mask_ball_elev_start &= (player_df['frame'] >= prep_end_frame)
            if mask_ball_elev_start.any():
                ball_elev_start_frame = player_df.loc[mask_ball_elev_start, 'frame'].min()
                st.write(f"Ball Elevation phase starts at frame {ball_elev_start_frame} for player '{player}'.")
            else:
                ball_elev_start_frame = prep_end_frame
                st.warning(f"No clear start to Ball Elevation phase detected for player '{player}'. Setting start to end of Preparation phase.")
        else:
            ball_elev_start_frame = prep_end_frame
            st.warning(f"Necessary velocity columns missing for player '{player}'. Setting start of Ball Elevation to end of Preparation phase.")
    
        # Assign Ball Elevation phase up to release frame
        mask_ball_elev = (df_with_phases['player_id'] == player) & \
                         (df_with_phases['frame'] >= ball_elev_start_frame) & \
                         (df_with_phases['frame'] < release_frame)
        df_with_phases.loc[mask_ball_elev, 'phase'] = "Ball Elevation"
    
        # Stability Phase:
        # Detect when the shooting arm is extending and elbow angles are maintained or increasing
        # We'll define Stability as frames where elbow angles' velocity is >= a threshold
    
        elbow_vel_threshold = 0.5  # degrees per second
        if 'mean_elbow_angle_vel' in player_df.columns:
            mask_stability = (
                (player_df['mean_elbow_angle_vel'] >= elbow_vel_threshold) &
                (player_df['frame'] >= ball_elev_start_frame) &
                (player_df['frame'] < release_frame)
            )
            if mask_stability.any():
                stability_start_frame = player_df.loc[mask_stability, 'frame'].min()
                st.write(f"Stability phase starts at frame {stability_start_frame} for player '{player}'.")
            else:
                stability_start_frame = ball_elev_start_frame
                st.warning(f"No clear start to Stability phase detected for player '{player}'. Setting start to Ball Elevation phase.")
        else:
            stability_start_frame = ball_elev_start_frame
            st.warning(f"'mean_elbow_angle_vel' column missing for player '{player}'. Setting start of Stability to Ball Elevation phase.")
    
        # Assign Stability phase up to release frame
        mask_stability = (df_with_phases['player_id'] == player) & \
                         (df_with_phases['frame'] >= stability_start_frame) & \
                         (df_with_phases['frame'] < release_frame)
        df_with_phases.loc[mask_stability, 'phase'] = "Stability"
    
        # Release Phase:
        # The frame where release occurs
        mask_release = (df_with_phases['player_id'] == player) & (df_with_phases['frame'] == release_frame)
        df_with_phases.loc[mask_release, 'phase'] = "Release"
    
        # Inertia Phase:
        # Frames after release
        mask_inertia = (df_with_phases['player_id'] == player) & (df_with_phases['frame'] > release_frame)
        df_with_phases.loc[mask_inertia, 'phase'] = "Inertia"
        inertia_end_frame = df_with_phases.loc[mask_inertia, 'frame'].max()
        st.write(f"Inertia phase assigned from frame {release_frame + 1} to {inertia_end_frame} for player '{player}'.")
    
    return df_with_phases


import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import streamlit as st
import logging

def compute_mad(series, scale='normal'):
    """
    Compute the Median Absolute Deviation (MAD) of a pandas Series.
    
    Parameters:
    -----------
    series : pandas.Series
        The data series to compute MAD on.
    scale : str, optional
        Scaling factor. 'normal' scales MAD to make it comparable to standard deviation.
    
    Returns:
    --------
    mad : float
        The computed MAD value.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if scale == 'normal':
        return mad * 1.4826  # Scaling factor for normal distribution
    else:
        return mad

def identify_phases(df, release_frame, fps_body=25, fps_ball=20):
    """
    Enhanced phase detection using dynamic thresholds, smoothing, and improved biomechanical logic.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data. Expected columns include:
            - 'frame': Frame number
            - 'player_id': Identifier for each player (if multiple players)
            - Biomechanical metrics such as:
                - 'mean_knee_angle_vel'
                - 'mean_hip_angle_vel'
                - 'mean_shoulder_angle_vel'
                - 'Player_Velocity_Y'
                - 'Shoulder_Angular_Velocity'
                - 'Ball_Velocity_Y'
                - ... (additional relevant columns)
    release_frame : int
        Frame number indicating the release point of the shot.
    fps_body : float, optional
        Frames per second of body data. Default is 25.
    fps_ball : float, optional
        Frames per second of ball data. Default is 20.
    
    Returns:
    --------
    df_with_phases : pandas.DataFrame
        DataFrame with an added 'phase' column indicating the detected phase for each frame.
    phase_counts : dict
        Dictionary with player_id as keys and number of phases detected as values.
    """
    df_with_phases = df.copy()
    df_with_phases['phase'] = "Undefined"  # Default phase
    
    phase_counts = {}
    
    # Check if 'player_id' exists
    if 'player_id' in df.columns:
        players = df['player_id'].unique()
    else:
        players = [None]  # Single player case
    
    for player in players:
        if player is not None:
            player_df = df_with_phases[df_with_phases['player_id'] == player].copy()
            st.write(f"Processing phases for Player ID: {player}")
        else:
            player_df = df_with_phases.copy()
            st.write("Processing phases for Single Player")
    
        phase_count = 0
    
        # Data Smoothing: Apply Savitzky-Golay filter for smoothing velocity data
        velocity_columns = ['mean_knee_angle_vel', 'mean_hip_angle_vel', 'mean_shoulder_angle_vel', 
                            'Player_Velocity_Y', 'Shoulder_Angular_Velocity', 'Ball_Velocity_Y']
        for col in velocity_columns:
            smooth_col = f'{col}_smooth'
            if col in player_df.columns:
                # Ensure the data is numeric
                player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
                # Handle NaN values by interpolation or filling
                player_df[col].interpolate(method='linear', inplace=True, limit_direction='both')
                player_df[col].fillna(method='bfill', inplace=True)
                player_df[col].fillna(method='ffill', inplace=True)
                
                # Determine window_length
                window_length = min(7, len(player_df))
                if window_length < 5:
                    window_length = 5  # Minimum window length for Savitzky-Golay
                if window_length % 2 == 0:
                    window_length += 1  # Make it odd
                try:
                    player_df[smooth_col] = savgol_filter(player_df[col], window_length=window_length, polyorder=2)
                except Exception as e:
                    st.warning(f"Could not apply Savitzky-Golay filter on '{col}' for player '{player}': {e}")
                    player_df[smooth_col] = player_df[col]  # Fallback to original data
            else:
                st.warning(f"Column '{col}' not found for player '{player}'. Skipping smoothing.")
    
        # After smoothing, update the main DataFrame
        if 'player_id' in df.columns:
            df_with_phases = pd.merge(df_with_phases, player_df, on=['frame', 'player_id'], how='left', suffixes=('', '_y'))
        else:
            df_with_phases = pd.merge(df_with_phases, player_df, on='frame', how='left', suffixes=('', '_y'))
    
        # Compute dynamic thresholds based on median and MAD for robustness
        threshold_factors = {
            'mean_knee_angle_vel': 1.0,
            'mean_hip_angle_vel': 1.0,
            'mean_shoulder_angle_vel': 1.0,
            'Player_Velocity_Y': 1.0
            # Add more if needed
        }
        
        for player in players:
            if player is not None:
                player_df = df_with_phases[df_with_phases['player_id'] == player]
                st.write(f"Computing thresholds for Player ID: {player}")
            else:
                player_df = df_with_phases.copy()
                st.write("Computing thresholds for Single Player")
    
            thresholds = {}
            for col, factor in threshold_factors.items():
                smooth_col = f'{col}_smooth'
                if smooth_col in player_df.columns:
                    median = player_df[smooth_col].median()
                    # Compute Median Absolute Deviation manually
                    mad = compute_mad(player_df[smooth_col], scale='normal')
                    thresholds[col] = median + factor * mad
                else:
                    st.warning(f"Smooth column '{smooth_col}' not found for player '{player}'.")
    
            # Phase Identification
            # 1. Preparation Phase
            prep_cols = ['mean_knee_angle_vel', 'mean_hip_angle_vel', 'mean_shoulder_angle_vel']
            prep_mask = pd.Series(True, index=player_df.index)  # Initialize as Series
            
            for col in prep_cols:
                smooth_col = f'{col}_smooth'
                if smooth_col in player_df.columns and col in thresholds:
                    prep_mask &= (player_df[smooth_col] < thresholds[col])
                else:
                    prep_mask &= False  # If any column missing, cannot confirm preparation
            
            if prep_mask.any():
                # Assign Preparation phase up to the first frame where mask is False
                prep_end_index = player_df[~prep_mask].index.min()
                if pd.isna(prep_end_index):
                    # If mask never becomes False, set end to release_frame or last frame
                    if release_frame in player_df['frame'].values:
                        prep_end_frame = release_frame
                    else:
                        prep_end_frame = player_df['frame'].max()
                else:
                    prep_end_frame = player_df.loc[prep_end_index, 'frame']
                # Assign Preparation phase
                mask_prep = (df_with_phases['frame'] <= prep_end_frame)
                if player is not None:
                    mask_prep &= (df_with_phases['player_id'] == player)
                df_with_phases.loc[mask_prep, 'phase'] = "Preparation"
                phase_count += 1
                st.write(f"Preparation phase assigned up to frame {prep_end_frame} for player '{player}'.")
            else:
                prep_end_frame = release_frame
                st.warning(f"No clear end to Preparation phase detected for player '{player}'. Setting end to Release frame.")
            
            # 2. Ball Elevation Phase
            # Criteria: Positive vertical velocity indicating upward movement
            ball_elev_col = 'Player_Velocity_Y_smooth'
            if ball_elev_col in player_df.columns:
                # Identify frames where vertical velocity is significantly positive
                elevation_threshold = player_df[ball_elev_col].quantile(0.75)
                elevation_mask = player_df[ball_elev_col] > elevation_threshold
                if elevation_mask.any():
                    elevation_start_frame = player_df.loc[elevation_mask, 'frame'].min()
                    # Assign Ball Elevation phase from end of Preparation to elevation_start_frame
                    mask_elev = (df_with_phases['frame'] > prep_end_frame) & (df_with_phases['frame'] <= elevation_start_frame)
                    if player is not None:
                        mask_elev &= (df_with_phases['player_id'] == player)
                    df_with_phases.loc[mask_elev, 'phase'] = "Ball Elevation"
                    phase_count += 1
                    st.write(f"Ball Elevation phase assigned from frame {prep_end_frame + 1} to {elevation_start_frame} for player '{player}'.")
            else:
                st.warning(f"Column '{ball_elev_col}' not found for player '{player}'. Skipping Ball Elevation phase.")
            
            # 3. Stability Phase
            # Criteria: Minimal movement, consistent posture
            stability_cols = ['Shoulder_Angular_Velocity']
            stability_thresholds = {'Shoulder_Angular_Velocity': 0.5}  # Example thresholds
            stability_mask = pd.Series(True, index=player_df.index)  # Initialize as Series
            
            for col in stability_cols:
                smooth_col = f'{col}_smooth'
                if smooth_col in player_df.columns and col in stability_thresholds:
                    stability_mask &= (player_df[smooth_col].abs() < stability_thresholds[col])
                else:
                    stability_mask &= False
            
            if stability_mask.any():
                stability_start_frame = player_df.loc[stability_mask, 'frame'].min()
                # Assign Stability phase from elevation_start_frame to stability_start_frame
                if 'elevation_start_frame' in locals():
                    mask_stab = (df_with_phases['frame'] > elevation_start_frame) & (df_with_phases['frame'] <= stability_start_frame)
                else:
                    mask_stab = (df_with_phases['frame'] > prep_end_frame) & (df_with_phases['frame'] <= stability_start_frame)
                if player is not None:
                    mask_stab &= (df_with_phases['player_id'] == player)
                df_with_phases.loc[mask_stab, 'phase'] = "Stability"
                phase_count += 1
                st.write(f"Stability phase assigned from frame {prep_end_frame + 1} to {stability_start_frame} for player '{player}'.")
            else:
                st.warning(f"No Stability phase detected for player '{player}'.")
            
            # 4. Release Phase
            # Criteria: Release frame marked explicitly
            if release_frame is not None:
                mask_release = (df_with_phases['frame'] == release_frame)
                if player is not None:
                    mask_release &= (df_with_phases['player_id'] == player)
                df_with_phases.loc[mask_release, 'phase'] = "Release"
                phase_count += 1
                st.write(f"Release phase assigned at frame {release_frame} for player '{player}'.")
            else:
                st.warning(f"No Release frame provided for player '{player}'.")
            
            # 5. Inertia Phase
            # Criteria: Frames after release indicating landing and balance
            if release_frame is not None:
                mask_inertia = (df_with_phases['frame'] > release_frame)
                if player is not None:
                    mask_inertia &= (df_with_phases['player_id'] == player)
                df_with_phases.loc[mask_inertia, 'phase'] = "Inertia"
                phase_count += 1
                inertia_end_frame = df_with_phases.loc[mask_inertia, 'frame'].max()
                if not pd.isna(inertia_end_frame):
                    st.write(f"Inertia phase assigned from frame {release_frame + 1} to {int(inertia_end_frame)} for player '{player}'.")
                else:
                    st.warning(f"Inertia end frame is undefined for player '{player}'. Assigning up to the last frame.")
                    inertia_end_frame = df_with_phases['frame'].max()
                    st.write(f"Inertia phase assigned from frame {release_frame + 1} to {int(inertia_end_frame)} for player '{player}'.")
            
            phase_counts[player] = phase_count
    
    return df_with_phases, phase_counts



import math

def filter_for_shooter_by_phases_and_ball(df, phase_counts):
    """
    Selects the shooter based on the player who has gone through the most biomechanical phases
    and is most closely aligned with the ball's release point.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'phase', 'player_id', 'basketball_x_smooth', 'basketball_y_smooth', 'release_frame' columns.
    phase_counts : dict
        Dictionary with player_id as keys and number of phases as values.

    Returns:
    --------
    shooter_df : pandas.DataFrame
        Filtered DataFrame containing only the shooter's data.
    shooter_id : int or None
        Player ID of the shooter, or None if not determined.
    """
    if not phase_counts:
        st.warning("No phase counts available to determine shooter.")
        return pd.DataFrame(), None

    # Select the player(s) with the maximum number of phases
    max_phases = max(phase_counts.values())
    candidates = [pid for pid, count in phase_counts.items() if count == max_phases]

    if not candidates:
        st.warning("No candidates found based on phase counts.")
        return pd.DataFrame(), None

    if len(candidates) == 1:
        shooter_id = candidates[0]
    else:
        # If multiple candidates, use ball tracking to select the shooter
        st.write(f"Multiple players with {max_phases} phases. Using ball tracking to determine shooter.")
        # Assume the shooter is closest to the ball's release point
        # Find the release frame
        if 'release_frame' not in df.columns:
            st.warning("Missing 'release_frame' column. Cannot determine shooter based on ball tracking.")
            shooter_id = candidates[0]  # Default to first candidate
        else:
            release_frame = df['release_frame'].iloc[0]

            # Check if 'basketball_x_smooth' and 'basketball_y_smooth' exist
            if 'basketball_x_smooth' not in df.columns or 'basketball_y_smooth' not in df.columns:
                st.warning("Missing 'basketball_x_smooth' or 'basketball_y_smooth' for ball release point. Selecting the first candidate as shooter.")
                shooter_id = candidates[0]
            else:
                ball_release_x = df.loc[df['frame'] == release_frame, 'basketball_x_smooth'].values
                ball_release_y = df.loc[df['frame'] == release_frame, 'basketball_y_smooth'].values

                if len(ball_release_x) == 0 or len(ball_release_y) == 0:
                    st.warning("Ball release position not found. Selecting the first candidate as shooter.")
                    shooter_id = candidates[0]
                else:
                    ball_release_x = ball_release_x[0]
                    ball_release_y = ball_release_y[0]

                    # Compute distance of each candidate to the ball's release point
                    distances = {}
                    for pid in candidates:
                        player_df = df[df['player_id'] == pid]
                        # Get player's position at release frame
                        # Assuming 'mid_hip_x_smooth' and 'mid_hip_y_smooth' represent player position
                        player_pos = player_df[player_df['frame'] == release_frame][['mid_hip_x_smooth', 'mid_hip_y_smooth']]
                        if player_pos.empty:
                            distances[pid] = float('inf')
                        else:
                            player_x = player_pos['mid_hip_x_smooth'].values[0]
                            player_y = player_pos['mid_hip_y_smooth'].values[0]
                            distance = math.sqrt((player_x - ball_release_x)**2 + (player_y - ball_release_y)**2)
                            distances[pid] = distance

                    # Select the player with the minimum distance
                    shooter_id = min(distances, key=distances.get)

    st.write(f"Selected shooter: Player ID {shooter_id} with {phase_counts.get(shooter_id, 0)} phases.")

    # Filter the DataFrame to include only the shooter
    shooter_df = df[df['player_id'] == shooter_id].copy()

    if shooter_df.empty:
        st.warning(f"No data found for shooter Player ID {shooter_id}.")

    return shooter_df, shooter_id




def filter_for_shooter_by_phases(df, phase_counts):
    """
    Selects the shooter based on the player who has gone through the most biomechanical phases.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'phase' and 'player_id' column.
    phase_counts : dict
        Dictionary with player_id as keys and number of phases as values.

    Returns:
    --------
    shooter_df : pandas.DataFrame
        Filtered DataFrame containing only the shooter's data.
    shooter_id : int or None
        Player ID of the shooter, or None if not determined.
    """
    if not phase_counts:
        st.warning("No phase counts available to determine shooter.")
        return pd.DataFrame(), None

    # Select the player with the maximum number of phases
    shooter_id = max(phase_counts, key=phase_counts.get)
    max_phases = phase_counts[shooter_id]
    st.write(f"Selected shooter: Player ID {shooter_id} with {max_phases} phases.")

    # Filter the DataFrame to include only the shooter
    shooter_df = df[df['player_id'] == shooter_id].copy()

    if shooter_df.empty:
        st.warning(f"No data found for shooter Player ID {shooter_id}.")

    return shooter_df, shooter_id



def calculate_ball_release_speed(df, release_frame, fps_ball=20.0):
    """
    Computes an approximate ball release speed using frames just before and after the release frame.
    """
    if release_frame is None or 'frame' not in df.columns:
        return np.nan

    # Find frames just before and after release
    df_before = df[df['frame'] < release_frame]
    df_after = df[df['frame'] > release_frame]

    if df_before.empty or df_after.empty:
        return np.nan

    before_idx = df_before['frame'].idxmax()
    after_idx = df_after['frame'].idxmin()

    bx1, by1 = df.loc[before_idx, 'x_smooth'], df.loc[before_idx, 'y_smooth']
    bx2, by2 = df.loc[after_idx, 'x_smooth'], df.loc[after_idx, 'y_smooth']
    dist = math.sqrt((bx2 - bx1)**2 + (by2 - by1)**2)

    frame_diff = df.loc[after_idx, 'frame'] - df.loc[before_idx, 'frame']
    if frame_diff <= 0:
        return np.nan

    time_sec = frame_diff / fps_ball
    speed = dist / time_sec if time_sec > 0 else np.nan
    return speed

def compute_temporal_consistency(df):
    """
    Computes Temporal Consistency based on the standard deviation of phase durations.
    Lower standard deviation indicates higher consistency.
    Returns a normalized score between 0 and 1.
    """
    phase_durations = df['phase'].value_counts().reindex(['Ball Elevation', 'Stability', 'Release', 'Inertia']).fillna(0)
    sd_durations = phase_durations.std()
    mean_durations = phase_durations.mean()

    if mean_durations == 0:
        return 0.0

    consistency_score = 1 - (sd_durations / mean_durations)
    consistency_score = max(min(consistency_score, 1.0), 0.0)

    return consistency_score

def compute_cyclical_asymmetry_index(df):
    """
    Computes Cyclical Asymmetry Index based on differences between left and right joint angles.
    Returns a normalized score between 0 and 1 (lower indicates higher symmetry).
    """
    joints = ['elbow_angle', 'knee_angle', 'shoulder_angle', 'hip_angle']
    asymmetry_scores = []

    for joint in joints:
        left_col = f'left_{joint}'
        right_col = f'right_{joint}'
        if left_col in df.columns and right_col in df.columns:
            diff = (df[left_col] - df[right_col]).abs()
            mean_angle = df[[left_col, right_col]].mean().mean()
            norm_diff = diff.mean() / max(mean_angle, 1)
            asymmetry_scores.append(norm_diff)

    if not asymmetry_scores:
        return np.nan

    avg_asymmetry = np.mean(asymmetry_scores)
    cyclical_asymmetry_index = 1 - min(avg_asymmetry / 0.5, 1.0)

    return cyclical_asymmetry_index

def compute_posture_analysis(df):
    """
    Computes Posture Analysis metrics such as Torso Angle and Hip Alignment.
    Returns a dictionary with metric names and their computed values.
    """
    posture_metrics = {}

    # Torso Angle
    if all(col in df.columns for col in ['neck_x', 'neck_y', 'neck_z', 'midhip_x', 'midhip_y', 'midhip_z']):
        df['torso_angle'] = df.apply(lambda row: calculate_torso_angle(row), axis=1)
        avg_torso_angle = df['torso_angle'].mean()
        posture_metrics['Torso Angle'] = avg_torso_angle

    # Hip Alignment
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
    mid_hip = np.array([row['midhip_x'], row['midhip_y'], row['midhip_z']])
    vector = neck - mid_hip
    vertical = np.array([0, 1, 0])

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
    if 'midhip_x' not in df.columns:
        return np.nan
    start_x = df.iloc[0]['midhip_x']
    end_x = df.iloc[-1]['midhip_x']
    stride_length = abs(end_x - start_x)
    return stride_length

# ==========================================
# 8. KPI Display Functions
# ==========================================
def display_kpis(df, metrics, multi_shot=False, selected_shots=[]):
    """
    Displays Key Performance Indicators (KPIs) using Streamlit's st.metric.
    Supports single and multiple shot comparisons with delta arrows.
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
                            value=f"{current_value:.2f} {info['unit']}" if info["unit"] else f"{current_value:.2f}",
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

def display_ball_metrics(metrics, multi_shot=False, selected_shots=[]):
    """
    Displays ball-specific KPIs using Streamlit's st.metric.
    Supports single and multiple shot comparisons with delta arrows.
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

# ==========================================
# 9. Main Streamlit App Function
# ==========================================
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
    user_data = {}
    for shot in chosen_shots:
        df_shot = load_vibe_output(s3_client, bucket_name, chosen_user, chosen_job, shot)
        if df_shot.empty:
            st.warning(f"No data for shot '{shot}'.")
            continue

        # Merge job metadata
        job_meta = get_metadata_from_dynamodb(dynamodb, chosen_job)
        for k, v in job_meta.items():
            df_shot[k.lower()] = v

        # Map body columns
        df_shot = map_body_columns(df_shot)

        # Compute head midpoint
        if {'left_eye_x', 'left_eye_y', 'left_eye_z',
            'right_eye_x', 'right_eye_y', 'right_eye_z'}.issubset(df_shot.columns):
            df_shot = compute_head_midpoint(df_shot)
        else:
            st.warning(f"Missing eye coordinates for shot '{shot}'. Cannot compute head midpoint.")
            df_shot['head_x'] = np.nan
            df_shot['head_y'] = np.nan
            df_shot['head_z'] = np.nan

        # Compute joint angles
        df_shot = compute_joint_angles(df_shot)

        # Preprocess and smooth data
        df_shot = preprocess_and_smooth(df_shot, sg_window=81, sg_poly=3, fps=25)

        # Compute ball speed and direction
        df_shot = compute_ball_speed_and_direction(df_shot, fps_ball=20.0)

        # Check for release frame based on elbow
        release_frame_elbow = detect_release_frame_elbow(df_shot, elbow_y_col='right_elbow_y_smooth')
        if release_frame_elbow is None:
            st.warning(f"Release frame could not be detected for shot '{shot}' based on elbow Y-position.")
        
        # Check for release frame based on ball
        release_frame_ball = detect_release_frame(df_shot, fps_ball=20.0)
        if release_frame_ball is None:
            st.warning(f"Release frame could not be detected for shot '{shot}' based on ball movement.")
        
        # Decide on release frame (could prioritize one over the other or use a combination)
        release_frame = release_frame_elbow if release_frame_elbow is not None else release_frame_ball
        if release_frame is None:
            st.warning(f"No release frame detected for shot '{shot}'. Skipping phase identification.")
            continue

        st.write(f"Release Frame Detected: {release_frame}")

        # Trim data around release frame
        trimmed_df = trim_data_around_release(df_shot, release_frame, before=30, after=20)

        # Identify phases within trimmed data and get phase counts
        df_with_phases, phase_counts = identify_phases(trimmed_df, release_frame, fps_body=25, fps_ball=20)

        # Compute Ball Tracking Metrics
        df_with_phases = compute_ball_speed_and_direction(df_with_phases, fps_ball=20.0)

        # Assign 'release_frame' for shooter selection
        df_with_phases['release_frame'] = release_frame

        # Shooter Selection: Automatically based on phases and ball tracking
        shooter_df, shooter_id = filter_for_shooter_by_phases_and_ball(df_with_phases, phase_counts)
        if shooter_df.empty:
            st.warning(f"No data for the shooter in shot '{shot}'. Skipping.")
            continue

        # Compute Center of Mass
        compute_all_com(shooter_df, fps_pose=25.0)

        # Compute advanced metrics
        metrics = calculate_advanced_metrics_reworked(
            shooter_df, 
            release_frame, 
            player_height=job_meta.get('player_height', 2.0)
        )

        # Store data
        user_data[shot] = {'df': shooter_df, 'metrics': metrics}

    if not user_data:
        st.warning("No valid shot data loaded.")
        return

    # Combine data for Overview
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

        # Detect release frame in combined data
        release_frame_combined_elbow = detect_release_frame_elbow(combined_df, elbow_y_col='right_elbow_y_smooth')
        release_frame_combined_ball = detect_release_frame(combined_df, fps_ball=20.0)
        release_frame_combined = release_frame_combined_elbow if release_frame_combined_elbow is not None else release_frame_combined_ball

        if release_frame_combined is None:
            st.warning("Could not detect release frame in combined data.")
        else:
            st.write(f"**Detected Release Frame**: {release_frame_combined}")

        # Trim data around release
        trimmed_agg = trim_data_around_release(combined_df, release_frame_combined, before=30, after=20)

        # Identify phases and get phase counts
        df_with_phases_agg, phase_counts_agg = identify_phases(trimmed_agg, release_frame_combined, fps_body=25, fps_ball=20)

        # Compute Ball Tracking Metrics
        df_with_phases_agg = compute_ball_speed_and_direction(df_with_phases_agg, fps_ball=20.0)

        # Assign 'release_frame' for shooter selection
        df_with_phases_agg['release_frame'] = release_frame_combined

        # Shooter Selection: Automatically based on phases and ball tracking
        shooter_df_agg, shooter_id_agg = filter_for_shooter_by_phases_and_ball(df_with_phases_agg, phase_counts_agg)
        if shooter_df_agg.empty:
            st.warning("No shooter identified in the aggregated data.")
            return

        # Compute Center of Mass
        compute_all_com(shooter_df_agg, fps_pose=25.0)

        # Compute advanced metrics
        metrics_agg = calculate_advanced_metrics_reworked(
            shooter_df_agg, 
            release_frame_combined, 
            player_height=job_meta.get('player_height', 2.0)
        )

        # Display KPIs
        display_kpis(shooter_df_agg, metrics_agg, multi_shot=multi_shot, selected_shots=chosen_shots)
        display_ball_metrics(metrics_agg, multi_shot=multi_shot, selected_shots=chosen_shots)

        # Visualize Phases
        visualize_phases(
            shooter_df_agg, 
            ['Preparation','Ball Elevation','Stability','Release','Inertia'], 
            key_suffix="overview"
        )

        # Visualize Joint Angles
        if 'mean_elbow_angle_vel' in shooter_df_agg.columns and 'mean_elbow_angle_vel' in shooter_df_agg.columns:
            visualize_joint_angles(
                shooter_df_agg, 
                'right_elbow_angle_smooth', 
                'left_elbow_angle_smooth', 
                key_suffix="overview"
            )
        else:
            st.warning("Required elbow angle columns are missing.")

        # Visualize Ball Trajectory
        if {'basketball_x_smooth', 'basketball_y_smooth', 'frame', 'time_ball'}.issubset(shooter_df_agg.columns):
            visualize_ball_trajectory_quadratic(
                shooter_df_agg, 
                x_col='basketball_x_smooth', 
                y_col='basketball_y_smooth', 
                key_suffix="overview"
            )
            visualize_ball_trajectory(
                shooter_df_agg, 
                x_col='basketball_x_smooth', 
                y_col='basketball_y_smooth', 
                key_suffix="overview"
            )
        else:
            st.warning("Missing 'basketball_x_smooth', 'basketball_y_smooth', 'frame', or 'time_ball' for ball trajectory plots.")

        # Visualize Ball Heatmap
        if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(shooter_df_agg.columns):
            visualize_ball_heatmap(
                df_with_phases_agg, 
                x_col='basketball_x_smooth', 
                y_col='basketball_y_smooth', 
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
            metrics_seg = data['metrics']
            release_frame_seg = df_shot['release_frame'].iloc[0] if 'release_frame' in df_shot.columns else None
            if release_frame_seg is None:
                st.warning(f"Release frame could not be detected for shot '{shot}'. Skipping phase identification.")
                continue
            trimmed_df = trim_data_around_release(df_shot, release_frame_seg, before=30, after=20)
            df_with_phases_seg, phase_counts_seg = identify_phases(trimmed_df, release_frame_seg, fps_body=25, fps_ball=20)

            # Compute Ball Tracking Metrics
            df_with_phases_seg = compute_ball_speed_and_direction(df_with_phases_seg, fps_ball=20.0)

            # Assign 'release_frame' for shooter selection
            df_with_phases_seg['release_frame'] = release_frame_seg

            # Shooter Selection: Automatically based on phases and ball tracking
            shooter_df_seg, shooter_id_seg = filter_for_shooter_by_phases_and_ball(df_with_phases_seg, phase_counts_seg)
            if shooter_df_seg.empty:
                st.warning(f"No data for the shooter in shot '{shot}'. Skipping.")
                continue

            # Compute Center of Mass
            compute_all_com(shooter_df_seg, fps_pose=25.0)

            # Compute advanced metrics
            metrics_seg = calculate_advanced_metrics_reworked(
                shooter_df_seg, 
                release_frame_seg, 
                player_height=job_meta.get('player_height', 2.0)
            )

            # Display KPIs
            display_kpis(shooter_df_seg, metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)
            display_ball_metrics(metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)

            # Visualize Phases
            visualize_phases(
                shooter_df_seg, 
                ['Preparation','Ball Elevation','Stability','Release','Inertia'], 
                key_suffix=f"pose_{shot}"
            )

            # Visualize Joint Angles
            if 'mean_elbow_angle_vel' in shooter_df_seg.columns and 'mean_elbow_angle_vel' in shooter_df_seg.columns:
                visualize_joint_angles(
                    shooter_df_seg, 
                    'right_elbow_angle_smooth', 
                    'left_elbow_angle_smooth', 
                    key_suffix=f"pose_{shot}"
                )
            else:
                st.warning("Required elbow angle columns are missing.")

            # Visualize Angular Velocities
            velocity_cols = [col for col in shooter_df_seg.columns if col.endswith('_vel')]
            if velocity_cols:
                visualize_angular_velocity(
                    shooter_df_seg, 
                    velocity_cols, 
                    key_suffix=f"velocity_{shot}"
                )
            else:
                st.warning("No angular velocity columns found for visualization.")

            # Visualize Angular Accelerations
            acceleration_cols = [col for col in shooter_df_seg.columns if col.endswith('_accel')]
            if acceleration_cols:
                visualize_angular_acceleration(
                    shooter_df_seg, 
                    acceleration_cols, 
                    key_suffix=f"acceleration_{shot}"
                )
            else:
                st.warning("No angular acceleration columns found for visualization.")

    # -------------- ADVANCED BIOMECHANICS TAB --------------
    with t_adv:
        st.markdown("## Advanced Biomechanics")
        for shot, data in user_data.items():
            st.markdown(f"### Shot: {shot}")
            df_shot = data['df']
            metrics_seg = data['metrics']
            release_frame_seg = df_shot['release_frame'].iloc[0] if 'release_frame' in df_shot.columns else None
            if release_frame_seg is None:
                st.warning(f"Release frame could not be detected for shot '{shot}'. Skipping phase identification.")
                continue
            trimmed_df = trim_data_around_release(df_shot, release_frame_seg, before=30, after=20)
            df_with_phases_seg, phase_counts_seg = identify_phases(trimmed_df, release_frame_seg, fps_body=25, fps_ball=20)

            # Compute Ball Tracking Metrics
            df_with_phases_seg = compute_ball_speed_and_direction(df_with_phases_seg, fps_ball=20.0)

            # Assign 'release_frame' for shooter selection
            df_with_phases_seg['release_frame'] = release_frame_seg

            # Shooter Selection: Automatically based on phases and ball tracking
            shooter_df_seg, shooter_id_seg = filter_for_shooter_by_phases_and_ball(df_with_phases_seg, phase_counts_seg)
            if shooter_df_seg.empty:
                st.warning(f"No data for the shooter in shot '{shot}'. Skipping.")
                continue

            # Compute Center of Mass
            compute_all_com(shooter_df_seg, fps_pose=25.0)

            # Compute advanced metrics
            metrics_seg = calculate_advanced_metrics_reworked(
                shooter_df_seg, 
                release_frame_seg, 
                player_height=job_meta.get('player_height', 2.0)
            )

            # Display KPIs
            display_kpis(shooter_df_seg, metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)
            display_ball_metrics(metrics_seg, multi_shot=multi_shot, selected_shots=chosen_shots)

            # Visualize Phases
            visualize_phases(
                shooter_df_seg, 
                ['Preparation','Ball Elevation','Stability','Release','Inertia'], 
                key_suffix=f"adv_{shot}"
            )

            # Visualize Kinematic Chain Sequence
            visualize_kinematic_chain_sequence(
                shooter_df_seg, 
                release_frame_seg, 
                shooter_df_seg[['frame','phase']], 
                key_suffix=shot
            )

    # -------------- BALL TRACKING TAB --------------
    with t_ball:
        st.markdown("## Ball Tracking & Trajectory")
        for shot, data in user_data.items():
            st.markdown(f"### Shot: {shot}")
            df_seg = data['df']

            # Ensure required columns exist
            if not {'frame','basketball_x_smooth','basketball_y_smooth'}.issubset(df_seg.columns):
                st.warning(f"No 'frame', 'basketball_x_smooth', 'basketball_y_smooth' columns => skipping ball trajectory for shot {shot}.")
                continue

            # Detect release frame for the ball data
            release_frame_seg = df_seg['release_frame'].iloc[0] if 'release_frame' in df_seg.columns else None
            if release_frame_seg is None:
                st.warning(f"Release frame not found for shot '{shot}'. Skipping ball trajectory trimming.")
                continue

            # Trim ball data from 10 frames before release => end
            trimmed_ball_df = trim_data_around_release(df_seg, release_frame_seg, before=10, after=0)

            st.markdown("#### Ball Trajectory with Quadratic Fit (Trimmed)")
            if {'basketball_x_smooth', 'basketball_y_smooth', 'time_ball'}.issubset(trimmed_ball_df.columns):
                visualize_ball_trajectory_quadratic(
                    df=trimmed_ball_df[['time_ball','basketball_x_smooth','basketball_y_smooth']].copy(),
                    x_col='basketball_x_smooth',
                    y_col='basketball_y_smooth',
                    key_suffix=f"quadratic_{shot}"
                )
                visualize_ball_trajectory(
                    df=trimmed_ball_df[['basketball_x_smooth','basketball_y_smooth']].copy(),
                    x_col='basketball_x_smooth',
                    y_col='basketball_y_smooth',
                    key_suffix=f"trajectory_{shot}"
                )
            else:
                st.warning("Missing 'basketball_x_smooth', 'basketball_y_smooth', or 'time_ball' for ball trajectory plots.")

            st.markdown("#### Ball-Specific Release Speed")
            speed_release = calculate_ball_release_speed(trimmed_ball_df, release_frame=release_frame_seg, fps_ball=20.0)
            if not pd.isna(speed_release):
                st.write(f"**Calculated Release Speed**: {speed_release:.3f} m/s")
            else:
                st.write("Could not calculate release speed from ball data.")

            st.markdown("#### Side-View Ball Trajectory (Trimmed)")
            if {'frame', 'basketball_x_smooth', 'time_ball'}.issubset(trimmed_ball_df.columns):
                visualize_ball_trajectory_side_view(
                    df=trimmed_ball_df[['frame','basketball_x_smooth','time_ball']].copy(),
                    x_col='time_ball',
                    y_col='basketball_x_smooth',
                    spline=True,
                    smoothing_val=1.2,
                    enforce_data_range=True,
                    highlight_max_speed=True,
                    key_suffix=f"side_view_{shot}"
                )
            else:
                st.warning("Missing 'frame', 'basketball_x_smooth', or 'time_ball' for side-view ball trajectory.")

        st.markdown("### Ball Tracking Heatmap (Aggregated)")
        if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(combined_df.columns):
            visualize_ball_heatmap(
                combined_df, 
                x_col='basketball_x_smooth', 
                y_col='basketball_y_smooth', 
                key_suffix="aggregated"
            )
        else:
            st.warning("Missing 'basketball_x_smooth' or 'basketball_y_smooth' for aggregated ball heatmap.")

# ==========================================
# 10. Run the App
# ==========================================
if __name__ == "__main__":
    main()
