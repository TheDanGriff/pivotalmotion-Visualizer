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
import os
from math import degrees, atan2
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.polynomial as poly

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(page_title=" Pivotal Motion Data Visualizer", layout="wide")
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# =============================================================================
# AWS Setup
# =============================================================================
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

# =============================================================================
# Utility Functions: S3, DynamoDB, etc.
# =============================================================================
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
            return pd.DataFrame()
    st.warning(f"No CSV found for segment '{segment_id}' in job '{job_id}'.")
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

# =============================================================================
# Smoothing / Series Utility
# =============================================================================
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
        st.warning(f"LOWESS smoothing failed: {e}. Using rolled mean instead.")
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

# =============================================================================
# Body Column Mapping
# =============================================================================
def map_body_columns(df):
    """Remap raw columns to standard left_shoulder_x, right_elbow_angle, etc."""
    rename_dict = {
        # Example partial
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
        # Add more mappings as needed
    }
    df = df.rename(columns=rename_dict)
    return df

# =============================================================================
# Joint Angle Computation (2D)
# =============================================================================
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
    """Compute elbow/knee angles from raw 2D coords. Remove the old clamp to 35–50°."""
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

    # NOTE: We do NOT clamp to 35–50° anymore.
    return df

# =============================================================================
# Release Detection
# =============================================================================
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

# =============================================================================
# 5-Phases
# =============================================================================
def identify_phases(df, release_frame):
    """5-phase model: Preparation, Ball Elevation, Stability, Release, Inertia."""
    phases = pd.Series([np.nan] * len(df), index=df.index)
    if release_frame is None:
        return phases

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

# =============================================================================
# Shooter Identification
# =============================================================================
def compute_phase_count(df, expected_phases):
    """Return how many phases from expected_phases are found in df['phase']."""
    rel_frame = detect_release_frame(df)
    phs = identify_phases(df, rel_frame)
    df['phase'] = phs
    actual_phases = set(phs.dropna().unique())
    matched = [p for p in expected_phases if p in actual_phases]
    return len(matched), matched

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

def filter_for_shooter_by_phases(df):
    """
    Select the shooter based on the number of matched phases.
    If tied, use the higher wrist position between frame 10 and frame -10 as a tiebreaker.
    """
    if 'player_id' not in df.columns:
        st.warning("No 'player_id' column; cannot filter shooter.")
        return df

    expected = ['Preparation', 'Ball Elevation', 'Stability', 'Release', 'Inertia']
    candidates = []
    for pid, sub_df in df.groupby('player_id'):
        pc_count, matched_phases = compute_phase_count(sub_df, expected)
        candidates.append((pid, pc_count, matched_phases))

    if not candidates:
        st.warning("No player_id groups found.")
        return df

    # Primary Criterion: Number of matched phases
    max_phases = max(c[1] for c in candidates)
    best_cands = [c for c in candidates if c[1] == max_phases]

    if len(best_cands) == 1:
        chosen_pid, best_count, matched = best_cands[0]
        st.info(f"Auto-selected player_id={chosen_pid} with {best_count}/5 phases matched: {matched}.")
        return df[df['player_id'] == chosen_pid].copy()

    # Tiebreaker Criterion: Higher wrist position between frame 10 and frame -10
    tie_list = []
    for c in best_cands:
        pid, pc, mp = c
        sub_df = df[df['player_id'] == pid]
        # Define frame range
        min_frame = sub_df['frame'].min()
        max_frame = sub_df['frame'].max()
        range_start = min_frame + 10
        range_end = max_frame - 10
        wrist_y = sub_df[(sub_df['frame'] >= range_start) & (sub_df['frame'] <= range_end)]
        if 'right_wrist_y_smooth' in wrist_y.columns:
            wrist_y_val = wrist_y['right_wrist_y_smooth'].max()
        elif 'right_wrist_y' in wrist_y.columns:
            wrist_y_val = wrist_y['right_wrist_y'].max()
        else:
            wrist_y_val = 0  # Default if no data
        tie_list.append((pid, wrist_y_val))

    # Select the player with the highest wrist position
    tie_list_sorted = sorted(tie_list, key=lambda x: x[1], reverse=True)
    chosen_pid, chosen_wrist_y = tie_list_sorted[0]
    st.info(f"Tiebreaker applied: Auto-selected player_id={chosen_pid} with higher wrist Y position ({chosen_wrist_y}).")
    return df[df['player_id'] == chosen_pid].copy()

# =============================================================================
# Kinematic Chain Timings
# =============================================================================
def compute_kinematic_chain_timings(df, chain_vel_cols, dt=0.01):
    """
    Finds the peak absolute angular velocity (and frame) for each velocity column.
    Returns dict with 'Movement Sequence' and each joint's peak info.
    """
    chain_results = {}
    peak_frames = {}

    for vcol in chain_vel_cols:
        if vcol not in df.columns:
            chain_results[vcol] = {"Peak Frame": None, "Peak Angular Velocity (°/frame)": np.nan}
            continue

        valid = df.dropna(subset=[vcol])
        if valid.empty:
            chain_results[vcol] = {"Peak Frame": None, "Peak Angular Velocity (°/frame)": np.nan}
            continue

        idx = valid[vcol].abs().idxmax()
        f = df.loc[idx, 'frame']
        val = df.loc[idx, vcol]
        chain_results[vcol] = {
            "Peak Frame": f,
            "Peak Angular Velocity (°/frame)": val
        }
        peak_frames[vcol] = f

    # Corrected Line: Using list comprehension with key inside sorted()
    sorted_joints = sorted([(j, f) for j, f in peak_frames.items() if f is not None], key=lambda x: x[1])

    seq = []
    for s in sorted_joints:
        # e.g., "right_knee_angle_vel" -> "Right Knee Angle"
        nice = s[0].replace('_vel', '').replace('_', ' ').title()
        seq.append(nice)
    chain_results['Movement Sequence'] = seq

    st.write("Movement Sequence:", seq)
    return chain_results

# =============================================================================
# Advanced Metric Calculation
# =============================================================================
def calculate_advanced_metrics(df, release_frame, player_height=2.0, dt=0.01):
    """Compute release height, angle, separation, chain, asymmetry, fluidity, etc."""
    metrics = {}
    if release_frame is None or 'frame' not in df.columns or release_frame not in df['frame'].values:
        return metrics

    row = df.loc[df['frame'] == release_frame].iloc[0]

    # 1) Release Height
    if 'right_wrist_z_smooth' in df.columns:
        h = row['right_wrist_z_smooth']
    elif 'right_wrist_z' in df.columns:
        h = row['right_wrist_z']
    else:
        h = np.nan
    if pd.notna(h) and h < 0:
        h = abs(h)
    metrics['Release Height (m)'] = h
    metrics['Relative Release Height'] = (h / player_height) if pd.notna(h) else np.nan

    # 2) Release Angle
    wrx = 'right_wrist_x_smooth' if 'right_wrist_x_smooth' in df.columns else 'right_wrist_x'
    wry = 'right_wrist_y_smooth' if 'right_wrist_y_smooth' in df.columns else 'right_wrist_y'
    wrz = 'right_wrist_z_smooth' if 'right_wrist_z_smooth' in df.columns else 'right_wrist_z'
    frameset = df['frame'].unique()
    if (release_frame - 1 in frameset) and (release_frame + 1 in frameset):
        rb = df.loc[df['frame'] == (release_frame - 1)].iloc[0]
        ra = df.loc[df['frame'] == (release_frame + 1)].iloc[0]
        dx = ra.get(wrx, 0) - rb.get(wrx, 0)
        dy = ra.get(wry, 0) - rb.get(wry, 0)
        dz = ra.get(wrz, 0) - rb.get(wrz, 0)
        hs = np.sqrt(dx * dx + dy * dy)
        if hs == 0:
            angle_deg = 90.0 if dz > 0 else 0.0
        else:
            angle_deg = abs(degrees(atan2(dz, hs)))
            if angle_deg < 35:
                angle_deg = 35
            elif angle_deg > 50:
                angle_deg = 50
        metrics['Release Angle (deg)'] = angle_deg
    else:
        metrics['Release Angle (deg)'] = np.nan

    # 3) Shoulder-Hip Separation
    tro = 'torso_rot_smooth' if 'torso_rot_smooth' in df.columns else 'torso_rot'
    pro = 'pelvis_rot_smooth' if 'pelvis_rot_smooth' in df.columns else 'pelvis_rot'
    if tro in df.columns and pro in df.columns:
        metrics['Shoulder-Hip Separation (°)'] = row[tro] - row[pro]
    else:
        metrics['Shoulder-Hip Separation (°)'] = np.nan

    # 4) Kinematic Chain
    vel_cols = [
        'right_ankle_flex_vel', 'right_knee_angle_vel', 'right_hip_flex_vel',
        'torso_rot_vel', 'right_shoulder_elev_vel', 'right_elbow_angle_vel', 'right_wrist_flex_vel'
    ]
    chain_results = compute_kinematic_chain_timings(df, vel_cols, dt=dt)
    metrics['Kinematic Chain Peaks'] = chain_results
    valid_pfs = [chain_results[vc]['Peak Frame'] for vc in vel_cols if vc in chain_results and chain_results[vc]['Peak Frame'] is not None]
    if len(valid_pfs) >= 2:
        metrics['Chain Range (frames)'] = max(valid_pfs) - min(valid_pfs)
    else:
        metrics['Chain Range (frames)'] = np.nan

    # 5) Asymmetry
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

    # 6) Fluidity Score
    fluid_cols = ['right_elbow_angle_smooth', 'right_knee_angle_smooth', 'right_wrist_flex_smooth']
    if all(c in df.columns for c in fluid_cols):
        sub_d = df.dropna(subset=fluid_cols).sort_values('frame')
        if len(sub_d) > 2:
            diffs = []
            for c in fluid_cols:
                diffs.append(sub_d[c].diff().abs())
            meandiff = np.mean([d.mean() for d in diffs])
            fluid = np.clip(1.0 - (meandiff / 10.0), 0, 1)
            metrics['Fluidity Score'] = fluid
        else:
            metrics['Fluidity Score'] = np.nan
    else:
        metrics['Fluidity Score'] = np.nan

    return metrics

# =============================================================================
# KPI Display
# =============================================================================
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
        "Release Height (m)": metrics.get('Release Height (m)', np.nan),
        "Relative Release Height": metrics.get('Relative Release Height', np.nan),
        "Release Angle (°)": metrics.get('Release Angle (deg)', np.nan),
        "Shoulder-Hip Separation (°)": metrics.get('Shoulder-Hip Separation (°)', np.nan),
        "Asymmetry Index": metrics.get('Asymmetry Index', np.nan),
        "Fluidity Score": metrics.get('Fluidity Score', np.nan),
    }

    cols = st.columns(3)
    i = 0
    for name, val in kpis.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            if "Angle" in name or "Separation" in name:
                disp = f"{val:.2f}°"
            elif "Height" in name:
                disp = f"{val:.3f} m"
            else:
                disp = f"{val:.2f}"
        else:
            disp = "N/A"
        cols[i % 3].metric(name, disp)
        i += 1

    # Warnings
    if 'Release Height (m)' in kpis and kpis['Release Height (m)'] < 0:
        st.warning("Negative release height – check your coordinate system or offset.")
    if 'Shoulder-Hip Separation (°)' in kpis and not pd.isna(kpis['Shoulder-Hip Separation (°)']):
        if kpis['Shoulder-Hip Separation (°)'] < 0:
            st.warning("Negative shoulder-hip separation => opposite rotation direction.")
    if 'Release Angle (°)' in kpis and not pd.isna(kpis['Release Angle (°)']):
        if kpis['Release Angle (°)'] < 35 or kpis['Release Angle (°)'] > 50:
            st.warning("Release Angle outside typical range (35–50°).")

# =============================================================================
# Download Report
# =============================================================================
def download_report(df, player_name, seg_name):
    """Generate a CSV download link for the DataFrame."""
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    fname = f"{player_name.replace(' ','_')}_{seg_name.replace(' ','_')}_report.csv"
    link = f'<a href="data:file/csv;base64,{b64}" download="{fname}">📥 Download CSV Report</a>'
    st.markdown(link, unsafe_allow_html=True)

# =============================================================================
# Ball Trajectory
# =============================================================================
def fit_parabola_ball(x, y):
    """Fit a quadratic parabola y = a + b*x + c*x^2."""
    if len(np.unique(x)) < 3:
        raise ValueError("Not enough unique x-values to fit a parabola.")
    return poly.polyfit(x, y, 2)

def evaluate_parabola(coefs, x):
    return poly.polyval(x, coefs)

def visualize_ball_trajectory_bezier(df):
    """Visualizes the ball trajectory with a quadratic parabola fit."""
    df = df.sort_values('frame').drop_duplicates('frame')
    if df[['x', 'y']].dropna().shape[0] < 3:
        st.warning("Not enough ball points to fit a parabola.")
        return
    xvals = df['x'].values
    yvals = df['y'].values
    try:
        coefs = fit_parabola_ball(xvals, yvals)
        x_lin = np.linspace(xvals.min(), xvals.max(), 200)
        y_lin = evaluate_parabola(coefs, x_lin)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xvals, y=yvals, mode='markers', name='Raw Points', marker=dict(color='blue', size=6)
        ))
        fig.add_trace(go.Scatter(
            x=x_lin, y=y_lin, mode='lines', name='Parabola Fit', line=dict(color='red')
        ))
        fig.update_layout(
            title="Ball Trajectory (Quadratic Fit)",
            xaxis_title="X Position",
            yaxis_title="Y Position"
        )
        st.plotly_chart(fig, use_container_width=True)
    except np.linalg.LinAlgError:
        st.warning("SVD did not converge in least squares – attempting alternative plotting method.")
        # Fallback: Use spline interpolation
        try:
            spline = UnivariateSpline(xvals, yvals, s=0.5)
            x_smooth = np.linspace(xvals.min(), xvals.max(), 200)
            y_smooth = spline(x_smooth)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xvals, y=yvals, mode='markers', name='Raw Points', marker=dict(color='blue', size=6)
            ))
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth, mode='lines', name='Spline Fit', line=dict(color='green')
            ))
            fig.update_layout(
                title="Ball Trajectory (Spline Fit)",
                xaxis_title="X Position",
                yaxis_title="Y Position"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Alternative plotting failed: {e} – skipping ball trajectory.")

# =============================================================================
# Preprocessing & Offsetting
# =============================================================================
def preprocess_and_smooth(df):
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

    for a in angle_cols:
        df = compute_angular_velocity(df, a)

    for a in angle_cols:
        v = a + '_vel'
        acc = a + '_accel'
        if v in df.columns and df[v].notna().any():
            df[acc] = compute_velocity(df[v], dt=0.01)
        else:
            df[acc] = np.nan
    return df

def offset_z_axis(df):
    """
    Offset Z-axis so floor=0 if ankles exist.
    """
    ank_smooth = ('left_ankle_z_smooth' in df.columns) and ('right_ankle_z_smooth' in df.columns)
    ank_raw = ('left_ankle_z' in df.columns) and ('right_ankle_z' in df.columns)

    if ank_smooth:
        floor_z = min(df['left_ankle_z_smooth'].min(), df['right_ankle_z_smooth'].min())
        zcols = [col for col in df.columns if col.endswith('_z_smooth')]
        for zc in zcols:
            df[zc] = df[zc] - floor_z
        st.info("Z-axis offset using smoothed ankle Z.")
    elif ank_raw:
        floor_z = min(df['left_ankle_z'].min(), df['right_ankle_z'].min())
        zcols = [col for col in df.columns if col.endswith('_z')]
        for zc in zcols:
            df[zc] = df[zc] - floor_z
        st.info("Z-axis offset using raw ankle Z.")
    else:
        st.warning("Insufficient data to offset Z-axis.")
    return df

# =============================================================================
# Joint Angle Plotting Functions
# =============================================================================
def plot_joint_angles(df, release_frame=None):
    """
    Plots only the key joint angles: ankle, knee, hip, elbow.
    """
    angle_cols = [
        'right_ankle_flex_smooth',
        'right_knee_angle_smooth',
        'right_hip_flex_smooth',
        'right_elbow_angle_smooth'
    ]
    angle_cols = [c for c in angle_cols if c in df.columns]

    if not angle_cols:
        st.warning("No key joint angle data available for plotting.")
        return

    fig = go.Figure()
    for ac in angle_cols:
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[ac],
            mode='lines',
            name=ac.replace('_smooth','').replace('_',' ').title()
        ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(
            x=release_frame, 
            line=dict(color='red', dash='dash'), 
            annotation_text="Release"
        )
    # Add shaded regions for phases
    phases = df['phase'].unique()
    for phs in phases:
        if pd.isna(phs):
            continue
        color = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }.get(phs, 'rgba(128,128,128,0.1)')        # Grey as default
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title="Key Joint Angles Around Release",
        xaxis_title="Frame",
        yaxis_title="Angle (°)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_separation(df, release_frame=None):
    """
    Plots Shoulder-Hip Separation over frames.
    """
    # Ensure the correct column name
    col = 'shoulder_hip_separation_smooth'
    if col not in df.columns:
        # If 'shoulder_hip_separation_smooth' doesn't exist, check original separation
        if 'shoulder-hip separation (°)' in df.columns:
            df[col] = smooth_series(df['shoulder-hip separation (°)'])
        elif 'shoulder_hip_separation (°)' in df.columns:
            df[col] = smooth_series(df['shoulder_hip_separation (°)'])
        else:
            st.warning("Shoulder-Hip Separation data not available.")
            return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df[col],
        mode='lines',
        name='Shoulder-Hip Separation'
    ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique()
    for phs in phases:
        if pd.isna(phs):
            continue
        color = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }.get(phs, 'rgba(128,128,128,0.1)')        # Grey as default
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title="Shoulder-Hip Separation Over Frames",
        xaxis_title="Frame",
        yaxis_title="Separation (°)"
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
        'right_elbow_angle_vel'
    ]
    vel_cols = [c for c in vel_cols if c in df.columns]

    if not vel_cols:
        st.warning("Angular velocity data not available.")
        return

    fig = go.Figure()
    for vc in vel_cols:
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[vc],
            mode='lines',
            name=vc.replace('_vel','').replace('_',' ').title()
        ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique()
    for phs in phases:
        if pd.isna(phs):
            continue
        color = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }.get(phs, 'rgba(128,128,128,0.1)')        # Grey as default
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title="Angular Velocities of Key Joints",
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
        'right_elbow_angle_accel'
    ]
    accel_cols = [c for c in accel_cols if c in df.columns]

    if not accel_cols:
        st.warning("Angular acceleration data not available.")
        return

    fig = go.Figure()
    for ac in accel_cols:
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df[ac],
            mode='lines',
            name=ac.replace('_accel','').replace('_',' ').title()
        ))
    if release_frame is not None and release_frame in df['frame'].values:
        fig.add_vline(x=release_frame, line=dict(color='red', dash='dash'), annotation_text="Release")
    # Add shaded regions for phases
    phases = df['phase'].unique()
    for phs in phases:
        if pd.isna(phs):
            continue
        color = {
            'Preparation': 'rgba(0,0,255,0.1)',        # Blue
            'Ball Elevation': 'rgba(255,165,0,0.1)',  # Orange
            'Stability': 'rgba(255,255,0,0.1)',        # Yellow
            'Release': 'rgba(255,0,0,0.1)',           # Red
            'Inertia': 'rgba(128,0,128,0.1)'          # Purple
        }.get(phs, 'rgba(128,128,128,0.1)')        # Grey as default
        phase_df = df[df['phase'] == phs]
        if not phase_df.empty:
            start = phase_df['frame'].min()
            end = phase_df['frame'].max()
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                opacity=0.5,
                layer='below',
                line_width=0,
                annotation_text=phs
            )
    fig.update_layout(
        title="Angular Accelerations of Key Joints",
        xaxis_title="Frame",
        yaxis_title="Angular Acceleration (°/frame²)"
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================
def main():
    st.title("🏀 **Pivotal Motion Data Visualizer**")
    st.markdown("""
    ### A robust analysis of basketball shooting mechanics:
    1. **Identify & smooth joint data.**
    2. **Use a 5-phase model and pick the correct player by # of matched phases.**
    3. **Tie-break by wrist position between frame 10 and frame -10 if needed.**
    4. **Compute advanced metrics** (release angle, chain timings, asymmetry, fluidity).
    5. **Visualize angles & ball trajectory.**
    """)

    bucket_name = st.secrets["BUCKET_NAME"]
    st.sidebar.header("User & Job Selection")

    # Step 1: User Selection
    users = list_users(s3_client, bucket_name)
    if not users:
        st.sidebar.error("No users found.")
        return
    chosen_user = st.sidebar.selectbox("Select User (Email)", users)

    # Step 2: Job Selection
    jobs = list_user_jobs(s3_client, bucket_name, chosen_user)
    if not jobs:
        st.sidebar.error(f"No jobs found for {chosen_user}.")
        return
    job_info = {}
    for j in jobs:
        meta = get_metadata_from_dynamodb(dynamodb, j)
        pname = meta.get('PlayerName', 'Unknown')
        utime = meta.get('UploadTimestamp', None)
        dt_s = "Unknown"
        if utime:
            try:
                dt_s = pd.to_datetime(int(utime), unit='s').strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
        label = f"{pname} - {dt_s} - {j}"
        job_info[j] = label
    chosen_job = st.sidebar.selectbox("Select Job", jobs, format_func=lambda x: job_info[x])

    # Step 3: Segment Selection
    segs = list_job_segments(s3_client, bucket_name, chosen_user, chosen_job)
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
    user_data = {}
    for seg in chosen_segs:
        df = load_vibe_output(s3_client, bucket_name, chosen_user, chosen_job, seg)
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

        # Identify shooter
        if 'player_id' in df.columns:
            df = filter_for_shooter_by_phases(df)
        else:
            st.warning(f"No 'player_id' in segment '{seg}', skipping shooter filter.")

        # Offset Z
        df = offset_z_axis(df)

        # Compute joint angles
        df = compute_joint_angles(df)

        # Smooth data and compute velocities/accelerations
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
        phase_counts.rename(columns={'index':'phase'}, inplace=True)  # Ensure lowercase 'phase'
        if not phase_counts.empty:
            fig_phases = px.bar(
                phase_counts, 
                x='phase', 
                y='Counts', 
                color='phase', 
                title="Distribution of Shooting Phases"
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
                st.warning("'right_wrist_y_smooth' not found. Using raw 'right_wrist_y' for plotting.")

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
            plot_joint_angles(trimmed_df, release_frame=release_frame_seg)

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
            phase_counts_seg.rename(columns={'index':'phase'}, inplace=True)  # Ensure lowercase 'phase'
            if not phase_counts_seg.empty:
                fig_phases_seg = px.bar(
                    phase_counts_seg,
                    x='phase',
                    y='Counts',
                    color='phase',
                    title=f"{seg} Phases"
                )
                st.plotly_chart(fig_phases_seg, use_container_width=True)
            else:
                st.warning("No phase data in this segment.")

    # ---------------- ADVANCED BIOMECHANICS ----------------
    with t_adv:
        st.markdown("## Advanced Biomechanics")
        st.markdown("""
        - **3D Release Angle** from wrist velocity vector
        - **Kinematic Chain Timings** (peak angular velocity ankles→knees→hips→torso→shoulders→elbow→wrist)
        - **Shoulder–Hip Separation**
        - **Asymmetry Index**
        """)

        for seg, df in user_data.items():
            st.markdown(f"### Segment: {seg}")
            release_frame_seg = detect_release_frame(df)
            if release_frame_seg is None:
                st.warning("No release frame detected => skipping.")
                continue

            # Trim data around release
            trimmed_df = trim_shot(df, release_frame_seg, frames_before=30, frames_after=20)

            # Calculate advanced metrics
            player_ht_seg = trimmed_df['player_height'].iloc[0] if 'player_height' in trimmed_df.columns else 2.0
            metrics_seg = calculate_advanced_metrics(trimmed_df, release_frame_seg, player_height=player_ht_seg)

            st.write("**Advanced Metrics at Release:**")
            if metrics_seg:
                # Convert dict to DataFrame for better display
                df_metrics = pd.DataFrame(metrics_seg, index=[0]).T
                df_metrics.columns = ["Value"]
                st.table(df_metrics)
            else:
                st.write("No advanced metrics available.")

    # ---------------- BALL TRACKING ----------------
    with t_ball:
        st.markdown("## Ball Tracking & Trajectory")
        for seg, df in user_data.items():
            st.markdown(f"### Segment: {seg}")
            if not {'frame', 'x', 'y'}.issubset(df.columns):
                st.warning("No 'frame', 'x', 'y' columns => skipping ball trajectory.")
                continue
            visualize_ball_trajectory_bezier(df[['frame', 'x', 'y']].copy())

    # ---------------- DOWNLOAD REPORTS ----------------
    st.markdown("## Download CSV Reports")
    for seg, df in user_data.items():
        pname = df['player_name'].iloc[0] if 'player_name' in df.columns else "UnknownPlayer"
        download_report(df, pname, seg)

    st.markdown("---")
    st.markdown("© 2024 Pivotal Motion. All rights reserved.")

# =============================================================================
# MAIN APPLICATION ENTRY
# =============================================================================
if __name__ == "__main__":
    main()
