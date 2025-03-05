import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
# import streamlit as st  # Removed/Commented for a no-warnings version
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os
import logging
import statsmodels.api as sm

# ==========================================
# 0. Initialize Logger (Optional)
# ==========================================
logging.basicConfig(level=logging.CRITICAL)  # Only critical logs, if any
logger = logging.getLogger(__name__)



# Custom Brand Colors or Theming
BRAND_PRIMARY = "#0044CC"      # Blue
BRAND_SECONDARY = "#00A0E4"    # Light Blue
BRAND_LIGHT = "#FFFFFF"        # White
BRAND_DARK = "#333333"         # Metallic Black
BRAND_LIGHTGREY = "#F0F2F6"
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
        # Ball
        'basketball_x': 'basketball_x',
        'basketball_y': 'basketball_y',
    }
    lower_rename_dict = {k.lower(): v for k, v in rename_dict.items()}
    df.columns = df.columns.str.lower()
    df = df.rename(columns=lower_rename_dict)
    return df

# ==========================================
# 3. Compute Head Position
# ==========================================
def compute_head_position(df):
    eye_columns = [
        'left_eye_x', 'left_eye_y', 'left_eye_z',
        'right_eye_x', 'right_eye_y', 'right_eye_z'
    ]
    # If missing any, just fill with NaN
    missing = [col for col in eye_columns if col not in df.columns]
    if missing:
        df['head_x'] = np.nan
        df['head_y'] = np.nan
        df['head_z'] = np.nan
        return df

    df['head_x'] = (df['left_eye_x'] + df['right_eye_x']) / 2
    df['head_y'] = (df['left_eye_y'] + df['right_eye_y']) / 2
    df['head_z'] = (df['left_eye_z'] + df['right_eye_z']) / 2
    return df

# ==========================================
# 4. Joint Angle Computation
# ==========================================
def angle_2d(a, b, c):
    """ Calculate the angle at point b in 2D. """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.nan
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # to avoid numerical issues
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def compute_joint_angles(df):
    """ Compute various joint angles from raw 2D coordinates. """
    joints = [
        # Elbow
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
        # Shoulder
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
        # Wrist
        {
            'name': 'left_wrist_angle',
            'marker_a': 'left_elbow',
            'marker_b': 'left_wrist',
            'marker_c': 'left_pinky'
        },
        {
            'name': 'right_wrist_angle',
            'marker_a': 'right_elbow',
            'marker_b': 'right_wrist',
            'marker_c': 'right_pinky'
        },
        # Knee
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
        # Ankle
        {
            'name': 'left_ankle_angle',
            'marker_a': 'left_knee',
            'marker_b': 'left_ankle',
            'marker_c': 'left_heel'
        },
        {
            'name': 'right_ankle_angle',
            'marker_a': 'right_knee',
            'marker_b': 'right_ankle',
            'marker_c': 'right_heel'
        },
        # Spine
        {
            'name': 'spine_angle',
            'marker_a': 'mid_hip',
            'marker_b': 'neck',
            'marker_c': 'head'
        },
    ]

    for joint in joints:
        angle_col = joint['name']
        marker_a = joint['marker_a']
        marker_b = joint['marker_b']
        marker_c = joint['marker_c']
        req_cols = [
            f"{marker_a}_x", f"{marker_a}_y",
            f"{marker_b}_x", f"{marker_b}_y",
            f"{marker_c}_x", f"{marker_c}_y"
        ]
        if all(col in df.columns for col in req_cols):
            df[angle_col] = df.apply(
                lambda row: angle_2d(
                    (row[f"{marker_a}_x"], row[f"{marker_a}_y"]),
                    (row[f"{marker_b}_x"], row[f"{marker_b}_y"]),
                    (row[f"{marker_c}_x"], row[f"{marker_c}_y"])
                ) if not any(pd.isna(row[col]) for col in req_cols) else np.nan,
                axis=1
            )
        else:
            df[angle_col] = np.nan
    return df

# ==========================================
# 5. Data Smoothing
# ==========================================
def smooth_series(series, window_length=11, polyorder=3):
    if series.isnull().all():
        return series
    series_clean = series.fillna(method='ffill').fillna(method='bfill')
    window = min(window_length, len(series_clean))
    if window < polyorder + 2:
        window = polyorder + 2
    if window % 2 == 0:
        window += 1
    if window > len(series_clean):
        window = len(series_clean) if len(series_clean) % 2 != 0 else len(series_clean) - 1
    try:
        smoothed = savgol_filter(series_clean, window_length=window, polyorder=polyorder)
        return pd.Series(smoothed, index=series.index)
    except:
        return series

def apply_smoothing(df):
    # Example: smooth joint angles or any columns that end with '_angle'
    smoothed_columns = [col for col in df.columns if col.endswith('_angle')]
    for col in smoothed_columns:
        df[f"{col}_smooth"] = smooth_series(df[col], window_length=11, polyorder=3)

    # Handle ball
    if 'basketball_x' in df.columns and 'basketball_y' in df.columns:
        df['basketball_x_smooth'] = smooth_series(df['basketball_x'], window_length=11, polyorder=3)
        df['basketball_y_smooth'] = smooth_series(df['basketball_y'], window_length=11, polyorder=3)
    else:
        df['basketball_x_smooth'] = np.nan
        df['basketball_y_smooth'] = np.nan
    return df

# ==========================================
# 6. Velocity / Accel
# ==========================================
def compute_velocity(series, fps=25.0):
    dt = 1.0 / fps
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def compute_acceleration(series, fps=25.0):
    dt = 1.0 / fps
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def calculate_velocity_acceleration(df):
    smoothed_columns = [col for col in df.columns if col.endswith('_smooth')]
    for col in smoothed_columns:
        base = col.replace('_smooth', '')
        vel_col = f"{base}_vel_smooth"
        acc_col = f"{base}_accel_smooth"
        df[vel_col] = compute_velocity(df[col])
        df[acc_col] = compute_acceleration(df[vel_col])
    return df

# ==========================================
# 7. Release Frame Detection
# ==========================================
def detect_release_frame_elbow(df, elbow_y_col='right_elbow_y_smooth'):
    if elbow_y_col not in df.columns:
        return None
    peak_idx = df[elbow_y_col].idxmax()
    if pd.isna(peak_idx):
        return None
    release_frame = df.loc[peak_idx, 'frame'] if 'frame' in df.columns else peak_idx
    return release_frame

def detect_release_frame_ball_speed(df):
    # If you want to detect release by ball speed
    if 'ball_speed_smooth' not in df.columns or 'frame' not in df.columns:
        return None
    speed_threshold = df['ball_speed_smooth'].quantile(0.95)
    potential = df[df['ball_speed_smooth'] >= speed_threshold]
    if potential.empty:
        return None
    release_frame = potential['frame'].iloc[0]
    return release_frame

# ==========================================
# 8. Phases
# ==========================================
def identify_phases(df, release_frame):
    """Simple 3-phase classification: Preparation, Execution, Follow-Through."""
    if 'phase' not in df.columns:
        df['phase'] = "Undefined"
    if 'frame' not in df.columns:
        return df, {}

    preparation_end = release_frame - 30
    follow_through_end = release_frame + 15

    df.loc[df['frame'] <= preparation_end, 'phase'] = "Preparation"
    df.loc[(df['frame'] > preparation_end) & (df['frame'] <= release_frame), 'phase'] = "Execution"
    df.loc[(df['frame'] > release_frame) & (df['frame'] <= follow_through_end), 'phase'] = "Follow-Through"

    # Count unique phases for each player (or single)
    if 'player_id' in df.columns:
        phase_counts = df.groupby('player_id')['phase'].nunique().to_dict()
    else:
        phase_counts = {'Single Player': df['phase'].nunique()}
    return df, phase_counts

# ==========================================
# 9. Shooter Selection
# ==========================================
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
def compute_phase_counts(df):
    """Return a dictionary: {player_id: number_of_phases}."""
    if 'player_id' in df.columns and df['player_id'].nunique() > 1:
        return df.groupby('player_id')['phase'].nunique().to_dict()
    return {'Single Player': df['phase'].nunique()}

def filter_for_shooter_by_phases_and_ball(df, phase_counts, TEAMS):
    """
    Select the shooter based on the maximum phases.  
    If tie, you could use ball tracking.  
    This function returns the (shooter_df, shooter_id).
    """
    if not phase_counts:
        return pd.DataFrame(), None

    # Single-player check
    if 'player_id' not in df.columns or df['player_id'].nunique() <= 1:
        shooter_id = 'Single Player'
        shooter_df = df.copy()
        return shooter_df, shooter_id

    max_phases = max(phase_counts.values())
    candidates = [pid for pid, count in phase_counts.items() if count == max_phases]
    if len(candidates) == 1:
        shooter_id = candidates[0]
    else:
        # Additional logic for tie-breaking
        # e.g., pick first candidate
        shooter_id = candidates[0]

    shooter_df = df[df['player_id'] == shooter_id].copy()
    return shooter_df, shooter_id

# ==========================================
# 10. Trim Data
# ==========================================
def trim_data_around_release(df, release_frame, before=30, after=20):
    if 'frame' not in df.columns:
        return df
    start_frame = release_frame - before
    end_frame = release_frame + after
    trimmed = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].copy()
    return trimmed

# ==========================================
# 11. Ball Speed & Direction
# ==========================================
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
def compute_ball_speed_and_direction(df, fps_ball=20.0):
    if {'basketball_x_smooth', 'basketball_y_smooth'}.issubset(df.columns):
        df['ball_speed'] = np.sqrt(df['basketball_x_smooth'].diff()**2 + df['basketball_y_smooth'].diff()**2) * fps_ball
        df['ball_direction'] = np.arctan2(df['basketball_y_smooth'].diff(), df['basketball_x_smooth'].diff()) * (180 / np.pi)
        df['ball_speed_smooth'] = smooth_series(df['ball_speed'], window_length=11, polyorder=3)
        df['ball_direction_smooth'] = smooth_series(df['ball_direction'], window_length=11, polyorder=3)
    else:
        df['ball_speed'] = np.nan
        df['ball_direction'] = np.nan
        df['ball_speed_smooth'] = np.nan
        df['ball_direction_smooth'] = np.nan
    return df

# ==========================================
# 12. Advanced Metrics (Minimal Example)
# ==========================================
def calculate_advanced_metrics_reworked(df, release_frame, player_height=2.0):
    metrics = {}
    try:
        # Jump Height: difference between peak mid_hip_y_smooth and value at release
        if 'mid_hip_y_smooth' in df.columns:
            peak_hip_y = df['mid_hip_y_smooth'].max()
            base_hip_y = df.loc[df['frame'] == release_frame, 'mid_hip_y_smooth'].values[0]
            metrics['Jump Height (m)'] = peak_hip_y - base_hip_y
        else:
            metrics['Jump Height (m)'] = np.nan
    except:
        metrics['Jump Height (m)'] = np.nan

    try:
        # Shot Release Velocity using ball_speed_smooth
        if 'ball_speed_smooth' in df.columns:
            metrics['Shot Release Velocity (m/s)'] = df.loc[df['frame'] == release_frame, 'ball_speed_smooth'].values[0]
        else:
            metrics['Shot Release Velocity (m/s)'] = np.nan
    except:
        metrics['Shot Release Velocity (m/s)'] = np.nan

    try:
        # Release Angle using 'spine_angle'
        if 'spine_angle' in df.columns:
            metrics['Release Angle (degrees)'] = df.loc[df['frame'] == release_frame, 'spine_angle'].values[0]
        else:
            metrics['Release Angle (degrees)'] = np.nan
    except:
        metrics['Release Angle (degrees)'] = np.nan

    return metrics

# ==========================================
# 13. Main Process (No Streamlit popups)
# ==========================================
def process_single_shot(df, fps_body=25, fps_ball=20):
    """
    Process a single shot DataFrame:
      1) Map columns
      2) Compute head
      3) Compute angles
      4) Smooth
      5) Velocity & accel
      6) Release frame
      7) Trim
      8) Identify phases
      9) Select shooter
      10) Return shooter-only data + metrics
    """
    # 1. Map
    df = map_body_columns(df)

    # 2. Head
    df = compute_head_position(df)

    # 3. Angles
    df = compute_joint_angles(df)

    # 4. Smooth
    df = apply_smoothing(df)
    df = calculate_velocity_acceleration(df)

    # 5. Ball speed & direction
    df = compute_ball_speed_and_direction(df, fps_ball=fps_ball)

    # 6. Detect release frames
    elbow_release = detect_release_frame_elbow(df, elbow_y_col='right_elbow_y_smooth')
    ball_release = detect_release_frame_ball_speed(df)
    release_frame = elbow_release if elbow_release is not None else ball_release
    if release_frame is None:
        # No release frame => return empty
        return pd.DataFrame(), {}

    # 7. Trim
    trimmed = trim_data_around_release(df, release_frame, before=30, after=20)

    # 8. Identify phases
    phased, phase_counts = identify_phases(trimmed, release_frame)

    # 9. Shooter selection
    shooter_df, shooter_id = filter_for_shooter_by_phases_and_ball(phased, phase_counts, TEAMS)
    if shooter_df.empty:
        # No shooter => return empty
        return pd.DataFrame(), {}

    # 10. Metrics
    metrics = calculate_advanced_metrics_reworked(shooter_df, release_frame)
    return shooter_df, metrics


# Example usage (if you need a function to handle a CSV string):
def run_analysis_on_csv(csv_string):
    """
    Minimal example to run the pipeline on a CSV string (no Streamlit).
    Returns shooter_df (only the shooter's data) and metrics dict.
    """
    df = pd.read_csv(StringIO(csv_string))
    shooter_df, metrics = process_single_shot(df)
    # Return only the shooter's data + metrics
    return shooter_df, metrics

import pandas as pd
import numpy as np
import math
from scipy.signal import savgol_filter
import streamlit as st
import logging
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import make_interp_spline
import boto3

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 0. Team Name Mapping Dictionary
# ==========================================
TEAMS = {
    'hawks': 'Atlanta Hawks',
    'nets': 'Brooklyn Nets',
    'celtics': 'Boston Celtics',
    'hornets': 'Charlotte Hornets',
    'bulls': 'Chicago Bulls',
    'cavaliers': 'Cleveland Cavaliers',
    'mavericks': 'Dallas Mavericks',
    'nuggets': 'Denver Nuggets',
    'pistons': 'Detroit Pistons',
    'warriors': 'Golden State Warriors',
    'rockets': 'Houston Rockets',
    'pacers': 'Indiana Pacers',
    'clippers': 'Los Angeles Clippers',
    'lakers': 'Los Angeles Lakers',
    'grizzlies': 'Memphis Grizzlies',
    'heat': 'Miami Heat',
    'bucks': 'Milwaukee Bucks',
    'timberwolves': 'Minnesota Timberwolves',
    'pelicans': 'New Orleans Pelicans',
    'knicks': 'New York Knicks',
    'thunder': 'Oklahoma City Thunder',
    'magic': 'Orlando Magic',
    '76ers': 'Philadelphia 76ers',
    'suns': 'Phoenix Suns',
    'blazers': 'Portland Trail Blazers',
    'kings': 'Sacramento Kings',
    'spurs': 'San Antonio Spurs',
    'raptors': 'Toronto Raptors',
    'jazz': 'Utah Jazz',
    'wizards': 'Washington Wizards'
}

# ==========================================
# 1. Column Mapping
# ==========================================

def map_body_columns(df):
    """Remap raw columns to standardized names."""
    rename_dict = {
        # Eyes
        'LEYE_X': 'left_eye_x',
        'LEYE_Y': 'left_eye_y',
        'LEYE_Z': 'left_eye_z',
        'REYE_X': 'right_eye_x',
        'REYE_Y': 'right_eye_y',
        'REYE_Z': 'right_eye_z',
        # Neck
        'NECK_X': 'neck_x',
        'NECK_Y': 'neck_y',
        'NECK_Z': 'neck_z',
        # Shoulders
        'LSJC_X': 'left_shoulder_x',
        'LSJC_Y': 'left_shoulder_y',
        'LSJC_Z': 'left_shoulder_z',
        'RSJC_X': 'right_shoulder_x',
        'RSJC_Y': 'right_shoulder_y',
        'RSJC_Z': 'right_shoulder_z',
        # Elbows
        'LEJC_X': 'left_elbow_x',
        'LEJC_Y': 'left_elbow_y',
        'LEJC_Z': 'left_elbow_z',
        'REJC_X': 'right_elbow_x',
        'REJC_Y': 'right_elbow_y',
        'REJC_Z': 'right_elbow_z',
        # Wrists
        'LWJC_X': 'left_wrist_x',
        'LWJC_Y': 'left_wrist_y',
        'LWJC_Z': 'left_wrist_z',
        'RWJC_X': 'right_wrist_x',
        'RWJC_Y': 'right_wrist_y',
        'RWJC_Z': 'right_wrist_z',
        # Pinks and Thumbs
        'LPINKY_X': 'left_pinky_x',
        'LPINKY_Y': 'left_pinky_y',
        'LPINKY_Z': 'left_pinky_z',
        'LTHUMB_X': 'left_thumb_x',
        'LTHUMB_Y': 'left_thumb_y',
        'LTHUMB_Z': 'left_thumb_z',
        'RPINKY_X': 'right_pinky_x',
        'RPINKY_Y': 'right_pinky_y',
        'RPINKY_Z': 'right_pinky_z',
        'RTHUMB_X': 'right_thumb_x',
        'RTHUMB_Y': 'right_thumb_y',
        'RTHUMB_Z': 'right_thumb_z',
        # Hips
        'MIDHIP_X': 'mid_hip_x',
        'MIDHIP_Y': 'mid_hip_y',
        'MIDHIP_Z': 'mid_hip_z',
        'LHJC_X': 'left_hip_x',
        'LHJC_Y': 'left_hip_y',
        'LHJC_Z': 'left_hip_z',
        'RHJC_X': 'right_hip_x',
        'RHJC_Y': 'right_hip_y',
        'RHJC_Z': 'right_hip_z',
        # Knees
        'LKJC_X': 'left_knee_x',
        'LKJC_Y': 'left_knee_y',
        'LKJC_Z': 'left_knee_z',
        'RKJC_X': 'right_knee_x',
        'RKJC_Y': 'right_knee_y',
        'RKJC_Z': 'right_knee_z',
        # Ankles
        'LAJC_X': 'left_ankle_x',
        'LAJC_Y': 'left_ankle_y',
        'LAJC_Z': 'left_ankle_z',
        'RAJC_X': 'right_ankle_x',
        'RAJC_Y': 'right_ankle_y',
        'RAJC_Z': 'right_ankle_z',
        # Heels and Toes
        'LHEEL_X': 'left_heel_x',
        'LHEEL_Y': 'left_heel_y',
        'LHEEL_Z': 'left_heel_z',
        'LSTOE_X': 'left_toe_x',
        'LSTOE_Y': 'left_toe_y',
        'LSTOE_Z': 'left_toe_z',
        'LBTOE_X': 'left_big_toe_x',
        'LBTOE_Y': 'left_big_toe_y',
        'LBTOE_Z': 'left_big_toe_z',
        'RHEEL_X': 'right_heel_x',
        'RHEEL_Y': 'right_heel_y',
        'RHEEL_Z': 'right_heel_z',
        'RSTOE_X': 'right_toe_x',
        'RSTOE_Y': 'right_toe_y',
        'RSTOE_Z': 'right_toe_z',
        'RBTOE_X': 'right_big_toe_x',
        'RBTOE_Y': 'right_big_toe_y',
        'RBTOE_Z': 'right_big_toe_z',
        # Joint Angles
        'left_elbow_angle': 'left_elbow_angle',
        'right_elbow_angle': 'right_elbow_angle',
        'left_knee': 'left_knee_angle',
        'right_knee': 'right_knee_angle',
        # Add more mappings as needed
    }
    df = df.rename(columns=rename_dict)
    return df

# ==========================================
# 1.a Compute Head Position
# ==========================================

def compute_head_position(df):
    """
    Computes the head position as the midpoint between the left and right eyes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'left_eye_x', 'left_eye_y', 'left_eye_z',
        'right_eye_x', 'right_eye_y', 'right_eye_z' columns.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with added 'head_x', 'head_y', 'head_z' columns.
    """
    eye_columns = ['left_eye_x', 'left_eye_y', 'left_eye_z',
                   'right_eye_x', 'right_eye_y', 'right_eye_z']
    
    # Check if all required eye columns are present
    if all(col in df.columns for col in eye_columns):
        df['head_x'] = (df['left_eye_x'] + df['right_eye_x']) / 2
        df['head_y'] = (df['left_eye_y'] + df['right_eye_y']) / 2
        df['head_z'] = (df['left_eye_z'] + df['right_eye_z']) / 2
    else:
        missing = [col for col in eye_columns if col not in df.columns]
        st.warning(f"Missing columns for head position computation: {missing}")
        df['head_x'] = np.nan
        df['head_y'] = np.nan
        df['head_z'] = np.nan
    
    return df

# ==========================================
# 2. Joint Angle Computation
# ==========================================

def angle_2d(a, b, c):
    """
    Calculate the angle at point b given three points a, b, c in 2D.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return np.nan
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Handle numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

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
                ) if not any(pd.isna(row[col]) for col in required_cols) else np.nan,
                axis=1
            )
        else:
            # Assign NaN and issue a warning if any required column is missing
            df[angle_col] = np.nan
            missing = [col for col in required_cols if col not in df.columns]
            st.warning(f"Missing columns for {angle_col}: {missing}")

    return df

# ==========================================
# 2.a Extract Right Elbow Y Position
# ==========================================

def compute_right_elbow_y(df):
    """
    Extracts the Y position of the right elbow.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'right_elbow_y_smooth' or 'right_elbow_y' column.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with 'right_elbow_y' column.
    """
    if 'right_elbow_y_smooth' in df.columns:
        df['right_elbow_y'] = df['right_elbow_y_smooth']
    elif 'right_elbow_y' in df.columns:
        df['right_elbow_y'] = df['right_elbow_y']
    else:
        st.warning("Column 'right_elbow_y' not found for release frame detection.")
        df['right_elbow_y'] = np.nan
    return df

def compute_right_elbow_z(df):
    """
    Extracts the Z position of the right elbow and applies smoothing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'right_elbow_z' column.
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with 'right_elbow_z_smooth' column.
    """
    if 'right_elbow_z' in df.columns:
        df['right_elbow_z_smooth'] = smooth_series(df['right_elbow_z'], window_length=11, polyorder=3)
    else:
        st.warning("Column 'right_elbow_z' not found for Release Height calculation.")
        df['right_elbow_z_smooth'] = np.nan
    return df

# ==========================================
# 3. Data Smoothing
# ==========================================

def smooth_series(series, window_length=11, polyorder=3):
    """
    Applies Savitzky-Golay filter to smooth the data.
    Ensures window_length is odd and less than or equal to the series length.
    """
    if series.isnull().all():
        return series
    # Drop NaNs for smoothing
    series_clean = series.fillna(method='ffill').fillna(method='bfill')
    window = min(window_length, len(series_clean))
    if window < polyorder + 2:
        window = polyorder + 2
    if window % 2 == 0:
        window += 1
    if window > len(series_clean):
        window = len(series_clean) if len(series_clean) % 2 != 0 else len(series_clean) - 1
    try:
        smoothed = savgol_filter(series_clean, window_length=window, polyorder=polyorder)
        return pd.Series(smoothed, index=series.index)
    except Exception as e:
        st.warning(f"Could not apply Savitzky-Golay filter: {e}")
        return series

def apply_smoothing(df):
    """
    Applies smoothing to relevant columns.
    """
    # Columns to smooth
    columns_to_smooth = [
        'left_elbow_angle',
        'right_elbow_angle',
        'left_knee_angle',
        'right_knee_angle',
        'left_shoulder_angle',
        'right_shoulder_angle',
        'left_wrist_angle',
        'right_wrist_angle',
        'left_ankle_angle',
        'right_ankle_angle',
        'spine_angle',
        'right_elbow_y',    # Added for release frame detection
        'right_elbow_z',    # Added for Release Height calculation
        'mid_hip_x',
        'mid_hip_y',
        # Add more as needed
    ]
    
    for col in columns_to_smooth:
        smooth_col = f"{col}_smooth"
        if col in df.columns:
            df[smooth_col] = smooth_series(df[col], window_length=11, polyorder=3)
        else:
            st.warning(f"Column '{col}' not found. Cannot apply smoothing.")
            df[smooth_col] = np.nan
    
    return df

# ==========================================
# 4. Velocity and Acceleration Calculation
# ==========================================

def compute_velocity(series, fps=25.0):
    """
    Computes velocity using centered finite differences.
    """
    dt = 1.0 / fps
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def compute_acceleration(series, fps=25.0):
    """
    Computes acceleration using centered finite differences on velocity.
    """
    dt = 1.0 / fps
    return (series.shift(-1) - series.shift(1)) / (2.0 * dt)

def calculate_velocity_acceleration(df):
    """
    Calculates velocity and acceleration for smoothed joint angles.
    """
    smoothed_columns = [col for col in df.columns if col.endswith('_smooth')]
    for col in smoothed_columns:
        # Changed naming to match phase identification
        base_col = col.replace('_smooth', '')
        vel_col = f"{base_col}_vel_smooth"
        accel_col = f"{base_col}_accel_smooth"
        if col in df.columns:
            df[vel_col] = compute_velocity(df[col])
            df[accel_col] = compute_acceleration(df[vel_col])
        else:
            st.warning(f"Column '{col}' not found. Cannot compute velocity and acceleration.")
            df[vel_col] = np.nan
            df[accel_col] = np.nan
    
    return df

# ==========================================
# 5. Phase Identification
# ==========================================

def compute_dynamic_threshold(series, factor=1.5):
    """
    Computes a dynamic threshold based on median and MAD.
    """
    median = series.median()
    mad = np.median(np.abs(series - median)) * 1.4826  # Scaling for normal distribution
    return median + factor * mad

def identify_phases(df, release_frame, fps=25.0):
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
        DataFrame containing motion data with at least 'frame' and optionally 'player_id' columns.
    release_frame : int
        Frame number indicating the release point of the shot.
    fps : float
        Frames per second of the pose tracking data.
    
    Returns:
    --------
    df : pandas.DataFrame
        Original DataFrame with an added 'phase' column indicating biomechanical phase per frame.
    """
    df['phase'] = "Undefined"
    
    # Check if 'player_id' exists and has multiple unique values
    if 'player_id' in df.columns and df['player_id'].nunique() > 1:
        players = df['player_id'].unique()
    else:
        players = [None]  # Single player case
    
    for player in players:
        if player is not None:
            player_df = df[df['player_id'] == player].copy()
            st.write(f"Processing phases for Player ID: {player}")
        else:
            player_df = df.copy()
            st.write("Processing phases for Single Player")
        
        # Define thresholds based on joint angles and accelerations
        thresholds = {}
        required_cols = [
            'left_elbow_angle_vel_smooth',
            'right_elbow_angle_vel_smooth',
            'left_knee_angle_vel_smooth',
            'right_knee_angle_vel_smooth',
            'left_shoulder_angle_vel_smooth',
            'right_shoulder_angle_vel_smooth',
            'left_wrist_angle_vel_smooth',
            'right_wrist_angle_vel_smooth',
            'left_elbow_angle_accel_smooth',
            'right_elbow_angle_accel_smooth',
            'left_knee_angle_accel_smooth',
            'right_knee_angle_accel_smooth',
            # Add more if needed
        ]
        missing_cols = [col for col in required_cols if col not in player_df.columns]
        if missing_cols:
            st.warning(f"Missing columns for phase identification: {missing_cols}")
            # Assign default thresholds or skip phase detection
            for col in required_cols:
                thresholds[col] = 0
        else:
            for col in required_cols:
                # Using a higher factor to make thresholds more robust
                thresholds[col] = compute_dynamic_threshold(player_df[col], factor=1.5)
        
        # Initialize phase boundaries
        phases = ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']
        phase_start_frames = {}
        
        # 1. Preparation Phase: Low velocities and accelerations in all relevant joints
        prep_mask = (
            (player_df['left_elbow_angle_vel_smooth'] < thresholds.get('left_elbow_angle_vel_smooth', 0)) &
            (player_df['right_elbow_angle_vel_smooth'] < thresholds.get('right_elbow_angle_vel_smooth', 0)) &
            (player_df['left_knee_angle_vel_smooth'] < thresholds.get('left_knee_angle_vel_smooth', 0)) &
            (player_df['right_knee_angle_vel_smooth'] < thresholds.get('right_knee_angle_vel_smooth', 0)) &
            (player_df['left_elbow_angle_accel_smooth'] < thresholds.get('left_elbow_angle_accel_smooth', 0)) &
            (player_df['right_elbow_angle_accel_smooth'] < thresholds.get('right_elbow_angle_accel_smooth', 0)) &
            (player_df['left_knee_angle_accel_smooth'] < thresholds.get('left_knee_angle_accel_smooth', 0)) &
            (player_df['right_knee_angle_accel_smooth'] < thresholds.get('right_knee_angle_accel_smooth', 0))
        )
        
        if prep_mask.any():
            prep_end_frame = player_df.loc[~prep_mask, 'frame'].min()
            if pd.isna(prep_end_frame):
                prep_end_frame = release_frame
            phase_start_frames['Preparation'] = player_df['frame'].min()
            phase_start_frames['Ball Elevation'] = prep_end_frame + 1
            if player is not None:
                df.loc[(df['frame'] >= phase_start_frames['Preparation']) & (df['frame'] <= prep_end_frame) & (df['player_id'] == player), 'phase'] = "Preparation"
            else:
                df.loc[(df['frame'] >= phase_start_frames['Preparation']) & (df['frame'] <= prep_end_frame), 'phase'] = "Preparation"
            st.write(f"Preparation phase assigned from frame {phase_start_frames['Preparation']} to {prep_end_frame} for {'Player ID ' + str(player) if player else 'Single Player'}.")
        else:
            prep_end_frame = release_frame
            st.warning(f"No clear end to Preparation phase for {'Player ID ' + str(player) if player else 'Single Player'}.")
            phase_start_frames['Preparation'] = player_df['frame'].min()
            phase_start_frames['Ball Elevation'] = prep_end_frame + 1
        
        # 2. Ball Elevation Phase: Increasing velocities in lower body joints
        elev_mask = (
            (player_df['left_knee_angle_vel_smooth'] > thresholds.get('left_knee_angle_vel_smooth', 0)) &
            (player_df['right_knee_angle_vel_smooth'] > thresholds.get('right_knee_angle_vel_smooth', 0)) &
            (player_df['left_knee_angle_accel_smooth'] > thresholds.get('left_knee_angle_accel_smooth', 0)) &
            (player_df['right_knee_angle_accel_smooth'] > thresholds.get('right_knee_angle_accel_smooth', 0))
        )
        
        if elev_mask.any():
            elev_start_frame = player_df.loc[elev_mask, 'frame'].min()
            phase_start_frames['Ball Elevation'] = elev_start_frame
            phase_start_frames['Stability'] = elev_start_frame + 1
            if player is not None:
                df.loc[(df['frame'] >= phase_start_frames['Ball Elevation']) & (df['frame'] <= elev_start_frame) & (df['player_id'] == player), 'phase'] = "Ball Elevation"
            else:
                df.loc[(df['frame'] >= phase_start_frames['Ball Elevation']) & (df['frame'] <= elev_start_frame), 'phase'] = "Ball Elevation"
            st.write(f"Ball Elevation phase assigned from frame {phase_start_frames['Ball Elevation']} to {elev_start_frame} for {'Player ID ' + str(player) if player else 'Single Player'}.")
        else:
            elev_start_frame = prep_end_frame
            st.warning(f"No clear start to Ball Elevation phase for {'Player ID ' + str(player) if player else 'Single Player'}.")
            phase_start_frames['Ball Elevation'] = elev_start_frame + 1
            phase_start_frames['Stability'] = elev_start_frame + 1
        
        # 3. Stability Phase: Increasing velocities in upper body joints
        stability_mask = (
            (player_df['left_elbow_angle_vel_smooth'] > thresholds.get('left_elbow_angle_vel_smooth', 0)) &
            (player_df['right_elbow_angle_vel_smooth'] > thresholds.get('right_elbow_angle_vel_smooth', 0)) &
            (player_df['left_wrist_angle_vel_smooth'] > thresholds.get('left_wrist_angle_vel_smooth', 0)) &
            (player_df['right_wrist_angle_vel_smooth'] > thresholds.get('right_wrist_angle_vel_smooth', 0)) &
            (player_df['left_elbow_angle_accel_smooth'] > thresholds.get('left_elbow_angle_accel_smooth', 0)) &
            (player_df['right_elbow_angle_accel_smooth'] > thresholds.get('right_elbow_angle_accel_smooth', 0)) &
            (player_df['left_wrist_angle_accel_smooth'] > thresholds.get('left_wrist_angle_accel_smooth', 0)) &
            (player_df['right_wrist_angle_accel_smooth'] > thresholds.get('right_wrist_angle_accel_smooth', 0))
        )
        
        if stability_mask.any():
            stability_start_frame = player_df.loc[stability_mask, 'frame'].min()
            phase_start_frames['Stability'] = stability_start_frame
            phase_start_frames['Release'] = stability_start_frame + 1
            if player is not None:
                df.loc[(df['frame'] >= phase_start_frames['Stability']) & (df['frame'] <= stability_start_frame) & (df['player_id'] == player), 'phase'] = "Stability"
            else:
                df.loc[(df['frame'] >= phase_start_frames['Stability']) & (df['frame'] <= stability_start_frame), 'phase'] = "Stability"
            st.write(f"Stability phase assigned from frame {phase_start_frames['Stability']} to {stability_start_frame} for {'Player ID ' + str(player) if player else 'Single Player'}.")
        else:
            stability_start_frame = elev_start_frame
            st.warning(f"No clear start to Stability phase for {'Player ID ' + str(player) if player else 'Single Player'}.")
            phase_start_frames['Stability'] = stability_start_frame + 1
            phase_start_frames['Release'] = release_frame
        
        # 4. Release Phase: Peak in elbow angles indicating maximum extension
        if player is not None:
            df.loc[(df['frame'] == release_frame) & (df['player_id'] == player), 'phase'] = "Release"
        else:
            df.loc[df['frame'] == release_frame, 'phase'] = "Release"
        st.write(f"Release phase assigned at frame {release_frame} for {'Player ID ' + str(player) if player else 'Single Player'}.")
        
        # 5. Inertia Phase: After release
        if player is not None:
            df.loc[(df['frame'] > release_frame) & (df['player_id'] == player), 'phase'] = "Inertia"
        else:
            df.loc[df['frame'] > release_frame, 'phase'] = "Inertia"
        st.write(f"Inertia phase assigned from frame {release_frame + 1} onwards for {'Player ID ' + str(player) if player else 'Single Player'}.")
    
    return df

# ==========================================
# 6. Phase Identification (Alternate)
# ==========================================

def identify_biomech_phases(df, fps=25.0):
    """
    Placeholder for biomechanical phase identification.
    Replace this with the actual phase identification logic.
    """
    # For demonstration purposes, assign random phases
    phases = ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']
    df['phase'] = np.random.choice(phases, size=len(df))
    return df

# ==========================================
# 7. Shooter Selection Function
# ==========================================

def filter_for_shooter_by_phases(df):
    """
    Placeholder for shooter selection based on phases.
    Replace this with the actual shooter selection logic.
    """
    # For demonstration purposes, assume single player
    return df

# ==========================================
# 8. KPI Calculation
# ==========================================

def compute_kpis(df, release_frame, player_id):
    """
    Computes Key Performance Indicators (KPIs) for the shooter.
    """
    metrics = {}
    
    # Release Height
    if 'right_elbow_z_smooth' in df.columns:
        metrics['Release Height (m)'] = abs(df.loc[df['frame'] == release_frame, 'right_elbow_z_smooth'].values[0])
    else:
        metrics['Release Height (m)'] = np.nan
        st.warning("Column 'right_elbow_z_smooth' not found for Release Height calculation.")
    
    # Asymmetry Index
    try:
        left_elbow = df.loc[df['frame'] == release_frame, 'left_elbow_angle'].values[0]
        right_elbow = df.loc[df['frame'] == release_frame, 'right_elbow_angle'].values[0]
        asymmetry = abs(left_elbow - right_elbow) / max(abs(left_elbow), abs(right_elbow), 1)
        metrics['Asymmetry Index'] = asymmetry
    except Exception as e:
        metrics['Asymmetry Index'] = np.nan
        st.warning(f"Error computing Asymmetry Index: {e}")
    
    # Kinetic Chain Efficiency
    try:
        left_vel = df.loc[df['frame'] == release_frame, 'left_elbow_angle_vel_smooth'].values[0]
        right_vel = df.loc[df['frame'] == release_frame, 'right_elbow_angle_vel_smooth'].values[0]
        efficiency = abs(left_vel) + abs(right_vel)  # Simplified example
        metrics['Kinetic Chain Efficiency'] = efficiency
    except Exception as e:
        metrics['Kinetic Chain Efficiency'] = np.nan
        st.warning(f"Error computing Kinetic Chain Efficiency: {e}")
    
    # Stride
    stride_length = compute_stride(df[df['player_id'] == player_id]) if 'player_id' in df.columns and player_id != 'Single Player' else compute_stride(df)
    metrics['Stride Length (m)'] = stride_length
    
    # Temporal Consistency
    consistency = compute_temporal_consistency(df[df['player_id'] == player_id]) if 'player_id' in df.columns and player_id != 'Single Player' else compute_temporal_consistency(df)
    metrics['Temporal Consistency'] = consistency
    
    # Cyclical Asymmetry Index
    cyclical_asymmetry = compute_cyclical_asymmetry_index(df[df['player_id'] == player_id]) if 'player_id' in df.columns and player_id != 'Single Player' else compute_cyclical_asymmetry_index(df)
    metrics['Cyclical Asymmetry Index'] = cyclical_asymmetry
    
    return metrics

def compute_stride(df):
    """
    Computes Stride as the total forward movement (displacement) of the mid_hip marker during the shot.
    Returns stride length in meters.
    """
    if 'mid_hip_x_smooth' not in df.columns:
        return np.nan
    start_x = df['mid_hip_x_smooth'].iloc[0]
    end_x = df['mid_hip_x_smooth'].iloc[-1]
    stride_length = abs(end_x - start_x)
    return stride_length

def compute_temporal_consistency(df):
    """
    Computes Temporal Consistency based on the standard deviation of phase durations.
    Lower standard deviation indicates higher consistency.
    Returns a normalized score between 0 and 1.
    """
    phase_durations = df['phase'].value_counts().reindex(['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']).fillna(0)
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
    joints = ['elbow_angle', 'knee_angle']
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

# ==========================================
# 9. Visualization
# ==========================================

def visualize_joint_angles(df, player_id, key_suffix=""):
    """
    Visualizes left and right joint angles over time.
    """
    try:
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=df['left_elbow_angle_smooth'],
            mode='lines',
            name='Left Elbow Angle',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df[x_axis],
            y=df['right_elbow_angle_smooth'],
            mode='lines',
            name='Right Elbow Angle',
            line=dict(color='green')
        ))

        # Highlight phases as shaded regions if available
        if 'phase' in df.columns:
            unique_phases = df['phase'].unique()
            phase_colors = {
                'Preparation': 'rgba(0, 128, 255, 0.1)',
                'Ball Elevation': 'rgba(255, 165, 0, 0.1)',
                'Stability': 'rgba(255, 255, 0, 0.1)',
                'Release': 'rgba(255, 0, 0, 0.1)',
                'Inertia': 'rgba(128, 0, 128, 0.1)',
                'Undefined': 'rgba(128,128,128,0.1)'
            }
            for phase in unique_phases:
                phase_df = df[df['phase'] == phase]
                if not phase_df.empty:
                    start_time = phase_df[x_axis].min()
                    end_time = phase_df[x_axis].max()
                    fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor=phase_colors.get(phase, 'rgba(128,128,128,0.1)'),
                        opacity=0.3,
                        layer='below',
                        line_width=0,
                        annotation_text=phase,
                        annotation_position="top left"
                    )

        fig.update_layout(
            title=f"Joint Angles Over Time for {player_id}",
            xaxis_title="Time (s)",
            yaxis_title="Joint Angle (degrees)",
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"joint_angles_{key_suffix}")
    except KeyError as e:
        st.error(f"Missing column for joint angle visualization: {e}")

def visualize_phases(df, player_id, key_suffix=""):
    """
    Visualizes the different phases in a bar chart.
    """
    phase_counts = df['phase'].value_counts().reindex(['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']).fillna(0).astype(int)
    phase_duration_seconds = (phase_counts / 25).round(2)  # Assuming 25 fps for pose tracking

    phase_info = pd.DataFrame({
        'Phase': phase_counts.index,
        'Frame Count': phase_counts.values,
        'Duration (s)': phase_duration_seconds.values
    })

    # Set 'Phase' as categorical with specific order
    phase_info['Phase'] = pd.Categorical(phase_info['Phase'], categories=['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia'], ordered=True)

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
    st.plotly_chart(fig, use_container_width=True, key=f"phases_bar_{key_suffix}")

def visualize_kpis(metrics):
    """
    Displays the computed KPIs.
    """
    st.markdown("## 🎯 **Key Performance Indicators (KPIs)**")
    for key, value in metrics.items():
        st.write(f"**{key}**: {value if not pd.isna(value) else 'N/A'}")

def visualize_selected_columns(df):
    """
    Provides a dropdown menu to select and visualize any columns.
    """
    st.markdown("## 📊 **Data Visualization**")
    
    # Filter smoothed columns
    smoothed_columns = [col for col in df.columns if col.endswith('_smooth')]
    
    # Create a mapping for human-friendly labels
    label_mapping = {}
    for col in smoothed_columns:
        # Remove '_smooth', replace underscores with spaces, and capitalize words
        human_readable = col.replace('_smooth', '').replace('_', ' ').title()
        label_mapping[human_readable] = col
    
    # Dropdown selection
    selected_labels = st.multiselect(
        "Select Columns to Visualize",
        options=list(label_mapping.keys()),
        help="Choose from smoothed data columns."
    )
    
    if selected_labels:
        # Separate selected columns into Angles, Velocities, and Accelerations
        angles = []
        velocities = []
        accelerations = []
        
        for label in selected_labels:
            original_col = label_mapping[label]
            if 'angle' in original_col.lower() and 'vel' not in original_col.lower() and 'accel' not in original_col.lower():
                angles.append((label, original_col))
            elif 'vel' in original_col.lower():
                velocities.append((label, original_col))
            elif 'accel' in original_col.lower():
                accelerations.append((label, original_col))
        
        # Function to plot a section
        def plot_section(title, data, x_axis='time_pose'):
            if data:
                fig = go.Figure()
                for label, col in data:
                    fig.add_trace(go.Scatter(
                        x=df[x_axis],
                        y=df[col],
                        mode='lines',
                        name=label
                    ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Time (s)",
                    yaxis_title="Value",
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {title.lower()} selected.")
        
        # Plot Angles
        plot_section("Joint Angles", angles)
        
        # Plot Velocities
        plot_section("Joint Velocities", velocities)
        
        # Plot Accelerations
        plot_section("Joint Accelerations", accelerations)
    
    else:
        st.info("Please select at least one column to visualize.")

# ==========================================
# 10. Shooter Selection Function
# ==========================================

def filter_for_shooter_by_phases_and_ball(df, phase_counts, TEAMS):
    """
    Selects the shooter based on the player who has gone through the most biomechanical phases
    and is most closely aligned with the ball's release point.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'phase', 'player_id', 'basketball_x_smooth', 'basketball_y_smooth', 'release_frame' columns.
    phase_counts : dict
        Dictionary with player_id as keys and number of phases as values.
    TEAMS : dict
        Dictionary mapping team abbreviations to full names.

    Returns:
    --------
    shooter_df : pandas.DataFrame
        Filtered DataFrame containing only the shooter's data.
    shooter_id : int or str or None
        Player ID of the shooter, or a string identifier for single-player, or None if not determined.
    """
    if not phase_counts:
        st.warning("No phase counts available to determine shooter.")
        return pd.DataFrame(), None

    # Check if 'player_id' exists and has multiple unique values
    if 'player_id' not in df.columns or df['player_id'].nunique() <= 1:
        # Single-player case
        shooter_id = 'Single Player'
        shooter_df = df.copy()
        return shooter_df, shooter_id

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

    st.write(f"**Selected shooter:** {shooter_id}")

    # Filter the DataFrame to include only the shooter
    if shooter_id == 'Single Player':
        shooter_df = df.copy()
    else:
        shooter_df = df[df['player_id'] == shooter_id].copy()

    if shooter_df.empty:
        st.warning(f"No data found for shooter Player ID {shooter_id}.")

    # Map team abbreviation to full name if possible
    if shooter_id != 'Single Player' and 'team' in shooter_df.columns:
        team_abbr = shooter_df['team'].iloc[0].lower()
        full_team_name = TEAMS.get(team_abbr, shooter_df['team'].iloc[0])
        shooter_df['team_full'] = full_team_name
    else:
        shooter_df['team_full'] = 'N/A'

    return shooter_df, shooter_id

# ==========================================
# 11. Release Frame Detection Function
# ==========================================

def detect_release_frame_elbow(df, elbow_y_col='right_elbow_y_smooth', fps=25.0):
    """
    Detects the release frame based on the peak Y-position of the elbow joint.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing motion data.
    elbow_y_col : str
        Column name for the elbow Y-position.
    fps : float
        Frames per second of the pose tracking data.
    
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
    if pd.isna(peak_idx):
        st.warning("No peak detected in elbow Y-position.")
        return None
    release_frame = df.loc[peak_idx, 'frame']
    st.write(f"**Release Frame Detected Based on Elbow Y-Position:** {release_frame}")
    
    # Add 'release_frame' column for shooter selection
    df['release_frame'] = release_frame
    
    return release_frame

# ==========================================
# 12. Ball Tracking and Visualization Functions
# ==========================================

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
    st.plotly_chart(fig, use_container_width=True, key=f"ball_heatmap_{key_suffix}")

def visualize_ball_trajectory_side_view(df, x_col='time_ball', y_col='x_smooth', spline=True, smoothing_val=1.2, enforce_data_range=True, highlight_max_speed=True, key_suffix=""):
    """
    Visualizes the ball's side-view trajectory.
    """
    if df.empty:
        st.warning("No data to plot for side-view ball trajectory.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        name='Ball Position',
        marker=dict(color='blue', size=6)
    ))

    if spline:
        try:
            x_smooth = np.linspace(df[x_col].min(), df[x_col].max(), 500)
            spl = make_interp_spline(df[x_col], df[y_col], k=3)
            y_smooth = spl(x_smooth)
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name='Spline Fit',
                line=dict(color='orange')
            ))
        except Exception as e:
            st.warning(f"Error creating spline fit: {e}")

    if highlight_max_speed:
        # Assuming 'ball_speed' column exists
        if 'ball_speed' in df.columns:
            max_speed_idx = df['ball_speed'].idxmax()
            max_speed_time = df.loc[max_speed_idx, x_col]
            max_speed_pos = df.loc[max_speed_idx, y_col]
            fig.add_trace(go.Scatter(
                x=[max_speed_time],
                y=[max_speed_pos],
                mode='markers',
                name='Max Speed',
                marker=dict(color='red', size=12, symbol='star')
            ))
            fig.add_annotation(
                x=max_speed_time,
                y=max_speed_pos,
                text="Max Speed",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        else:
            st.warning("Column 'ball_speed' not found for highlighting max speed.")

    fig.update_layout(
        title="Side-View Ball Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="X Position",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ball_side_view_{key_suffix}")

# ==========================================
# 13. Main Processing Function
# ==========================================

def main():
    st.title("🏀 **Pivotal Motion Data Visualizer**")
    st.write("### Analyze and Visualize Shooting Motion Data")

    # Initialize AWS clients
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')

    # Retrieve bucket name from secrets
    try:
        bucket_name = st.secrets["BUCKET_NAME"]
    except KeyError:
        st.error("Bucket name not found in secrets.")
        st.stop()

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
            tname = meta.get('Team', '').strip().lower()
            tname_full = TEAMS.get(tname, tname.title()) if tname else 'N/A'
            ut = meta.get('UploadTimestamp', None)
            dt_s = "Unknown"
            if ut:
                try:
                    dt_s = pd.to_datetime(int(ut), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass

            if tname_full != 'N/A':
                label = f"{pname} - {tname_full} - {dt_s} - {j}"
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
    team_abbr = job_meta.get('Team', 'N/A').strip().lower()
    team_full = TEAMS.get(team_abbr, team_abbr.title()) if team_abbr != 'n/a' else 'N/A'
    user_data = {}

    # Initialize a list to store KPIs for multiple shots
    all_metrics = []

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

        # 1) Preprocess ball data
        df_shot = preprocess_ball_data(df_shot, x_col='x', y_col='y')

        # 2) Increase smoothing for body & ball
        df_shot = apply_smoothing(df_shot)

        # 3) Calculate linear velocity & acceleration for the ball (using heavier smoothing)
        df_shot = calculate_velocity_acceleration(df_shot)

        # 4) Detect phases
        release_frame = detect_release_frame_elbow(df_shot, elbow_y_col='right_elbow_y_smooth')
        if release_frame is not None:
            identify_phases(df_shot, release_frame)
        else:
            df_shot['phase'] = "Undefined"
            st.warning("Phases could not be identified without a release frame.")

        # 5) Filter for the main shooting player if multiple
        df_shot = filter_for_shooter_by_phases_and_ball(df_shot, phase_counts=compute_phase_counts(df_shot), TEAMS=TEAMS)

        if df_shot.empty:
            st.warning(f"No data after filtering for shooter in shot '{shot}'.")
            continue

        # 6) KPI Calculation
        kpis = compute_kpis(df_shot, release_frame, 'Single Player')  # Replace 'Single Player' if player_id exists
        all_metrics.append({'Shot': shot, **kpis})

        # 7) Collect Data for Visualization
        user_data[shot] = df_shot

    if not user_data:
        st.error("No valid shots to display.")
        return

    # 8) Display KPIs
    st.markdown("## 🎯 **Key Performance Indicators (KPIs)**")
    kpi_df = pd.DataFrame(all_metrics)
    st.dataframe(kpi_df.set_index('Shot'))

    # 9) Overview Visualizations
    st.markdown("## 🏀 **Shot Overview**")
    overview_tabs = st.tabs([f"Shot {shot}" for shot in chosen_shots])
    for idx, shot in enumerate(chosen_shots):
        with overview_tabs[idx]:
            df_overview = user_data[shot]
            st.subheader(f"Shot: {shot}")
            # Display main KPIs
            kpi = all_metrics[idx]
            st.markdown("### 🎯 **KPIs**")
            for key, value in kpi.items():
                if key != 'Shot':
                    st.write(f"**{key}**: {value if not pd.isna(value) else 'N/A'}")
            # Display main Pose Analysis and Ball Tracking visuals
            pose_ball_tabs = st.tabs(["Pose Analysis", "Ball Tracking"])
            with pose_ball_tabs[0]:
                visualize_joint_angles(df_overview, player_id='Single Player', key_suffix=f"{shot}_pose")
                visualize_phases(df_overview, player_id='Single Player', key_suffix=f"{shot}_phases")
            with pose_ball_tabs[1]:
                visualize_ball_trajectory_quadratic(df_overview, key_suffix=f"{shot}_ball_quad")
                visualize_ball_trajectory(df_overview, key_suffix=f"{shot}_ball_traj")
                visualize_ball_heatmap(df_overview, key_suffix=f"{shot}_ball_heatmap")
                visualize_ball_trajectory_side_view(df_overview, key_suffix=f"{shot}_ball_side")
    
    # 10) Detailed Graphs in Separate Tabs
    st.markdown("## 📊 **Detailed Graphs**")
    detailed_tabs = st.tabs(["Joint Angles", "Phases", "Ball Trajectory", "Ball Heatmap", "Side-View Trajectory"])
    for idx, shot in enumerate(chosen_shots):
        with detailed_tabs[0]:
            visualize_joint_angles(user_data[shot], player_id='Single Player', key_suffix=f"{shot}_detailed_pose")
        with detailed_tabs[1]:
            visualize_phases(user_data[shot], player_id='Single Player', key_suffix=f"{shot}_detailed_phases")
        with detailed_tabs[2]:
            visualize_ball_trajectory_quadratic(user_data[shot], key_suffix=f"{shot}_detailed_ball_quad")
            visualize_ball_trajectory(user_data[shot], key_suffix=f"{shot}_detailed_ball_traj")
        with detailed_tabs[3]:
            visualize_ball_heatmap(user_data[shot], key_suffix=f"{shot}_detailed_ball_heatmap")
        with detailed_tabs[4]:
            visualize_ball_trajectory_side_view(user_data[shot], key_suffix=f"{shot}_detailed_ball_side")

def compute_phase_counts(df):
    """
    Computes the number of unique phases for each player.
    """
    if 'player_id' in df.columns:
        return df.groupby('player_id')['phase'].nunique().to_dict()
    else:
        return {'Single Player': df['phase'].nunique()}

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
# ==========================================
# 15. Run the App
# ==========================================

if __name__ == "__main__":
    main()
