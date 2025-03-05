# kpi.py
import numpy as np
from math import atan2, degrees
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import logging
from scipy.signal import savgol_filter
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# KPI Calculation Functions
# -------------------------------------------------------------------------------------
def angle_2d(a, b, c):
    """Compute angle at b with points a, b, c in 2D => degrees."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        dot_product = np.dot(ba, bc)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        if magnitude_ba == 0 or magnitude_bc == 0:
            return np.nan
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = degrees(np.arccos(cos_angle))
        return angle
    except Exception as e:
        logger.error(f"Error in angle_2d: {str(e)}")
        return np.nan

def compute_velocity(positions, fps, smoothing_window=7, polyorder=2):
    data = positions.to_numpy() if hasattr(positions, 'to_numpy') else np.array(positions)
    n = len(data)
    if n < 3:
        return np.zeros_like(data)
    if smoothing_window > n:
        smoothing_window = n if n % 2 == 1 else n - 1
    elif smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothed = savgol_filter(data, window_length=smoothing_window, polyorder=polyorder)
    velocity = np.gradient(smoothed, 1.0)
    return velocity

def improved_find_release_frame(pose_df, ball_df=None, fps=25, max_release_time=3.0):
    if pose_df.empty:
        return None
    pose_df['left_elbow_smooth'] = pose_df['left_elbow_angle'].rolling(window=5, min_periods=1, center=True).mean()
    pose_df['right_elbow_smooth'] = pose_df['right_elbow_angle'].rolling(window=5, min_periods=1, center=True).mean()
    pose_df['mean_elbow'] = (pose_df['left_elbow_smooth'] + pose_df['right_elbow_smooth']) / 2.0
    release_idx = pose_df['mean_elbow'].idxmax()
    if release_idx is not None:
        release_time = release_idx / fps
        if release_time > max_release_time:
            return None
    return release_idx

def calculate_quadratic_fit(release_phase):
    try:
        if len(release_phase) < 3:
            return np.nan, np.nan
        z = np.polyfit(release_phase['center_x'], release_phase['center_y'], 2)
        a, b, c = z
        curvature = 2 * a
        apex_x = -b / (2 * a) if a != 0 else np.nan
        apex_y = a * apex_x**2 + b * apex_x + c if not np.isnan(apex_x) else np.nan
        apex_height = apex_y
        return curvature, apex_height
    except Exception as e:
        logger.error(f"Error calculating quadratic fit: {str(e)}")
        return np.nan, np.nan

def calculate_lateral_curvature(spin_df):
    try:
        if 'ex' not in spin_df.columns:
            return np.nan
        lateral_spin_std = spin_df['ex'].std()
        return lateral_spin_std
    except Exception as e:
        logger.error(f"Error calculating lateral curvature: {str(e)}")
        return np.nan

def calculate_form_score(kpi_dict):
    if not kpi_dict:
        logger.warning("KPI dictionary is empty. Cannot calculate form score.")
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
            logger.info(f"KPI '{k}' is NaN. Skipping in form score calculation.")
            continue
        if k == 'Release Velocity':
            norm = min(val / 10, 1)
        elif k == 'Shot Distance':
            norm = min(val / 50, 1)
        elif k == 'Release Angle':
            norm = 1 - abs(val) / 90
        elif k == 'Apex Height':
            norm = min(val / 500, 1)
        elif k == 'Release Quality':
            norm = val / 100
        else:
            norm = 0
        score += norm * w
        logger.debug(f"KPI '{k}': value={val}, normalized={norm}, weighted={norm * w}")
    form_score = score * 100
    form_score = min(100, max(0, form_score))
    logger.info(f"Calculated form score: {form_score}")
    return form_score

def get_percentile_color(value, benchmark_values):
    # Check if value is NaN or benchmark_values is empty
    if pd.isna(value) or (hasattr(benchmark_values, '__len__') and len(benchmark_values) == 0):
        return "gray"
    percentile = np.mean(np.array(benchmark_values) < value) * 100
    if percentile >= 75:
        return "green"
    elif percentile >= 50:
        return "yellow"
    elif percentile >= 25:
        return "orange"
    else:
        return "red"


def get_kpi_benchmarks():
    return {
        'Release Velocity': {'values': np.linspace(5, 15, 100)},
        'Release Height': {'values': np.linspace(60, 90, 100)},
        'Release Angle': {'values': np.linspace(30, 80, 100)},
        'Release Time': {'values': np.linspace(0.2, 0.6, 100)},
        'Apex Height': {'values': np.linspace(50, 100, 100)},
        'Release Curvature': {'values': np.linspace(0.0, 0.5, 100)},
        'Lateral Deviation': {'values': np.linspace(0.0, 0.2, 100)},
    }

def format_kpi_with_color(name, value, benchmarks):
    color = get_percentile_color(value, benchmarks.get(name, {}).get('values', []))
    return f"<span style='color: {color}; font-weight: bold;'>{value:.1f}</span>"

def calculate_release_velocity(df_ball, fps=25):
    if df_ball.empty:
        return np.nan, np.nan, np.nan
    df_ball['center_x_m'] = df_ball['center_x'] * 0.0254
    df_ball['center_z_m'] = df_ball['center_z'] * 0.0254
    df_ball['vel_x'] = compute_velocity(df_ball['center_x_m'], fps)
    df_ball['vel_z'] = compute_velocity(df_ball['center_z_m'], fps)
    release_frame = df_ball['vel_x'].idxmax()
    release_velocity = np.sqrt(df_ball.loc[release_frame, 'vel_x']**2 + df_ball.loc[release_frame, 'vel_z']**2)
    release_height = df_ball.loc[release_frame, 'center_z_m']
    return release_velocity, release_frame, release_height

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
    release_quality = ((average_spin_x + 180) / 360) * 100
    return release_quality

def calculate_release_curvature(df_ball):
    if df_ball.empty:
        return np.nan
    x = df_ball['center_x_m']
    z = df_ball['center_z_m']
    coeffs = np.polyfit(x, z, 2)
    curvature = 2 * coeffs[0]
    return curvature

def calculate_asymmetry_index(df_pose):
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
            norm = min(val / 10, 1)
        elif k == 'Shot Distance':
            norm = min(val / 50, 1)
        elif k == 'Release Angle':
            norm = 1 - abs(val) / 90
        elif k == 'Apex Height':
            norm = min(val / 500, 1)
        elif k == 'Release Quality':
            norm = val / 100
        else:
            norm = 0
        score += norm * w
    return score * 100

def calculate_shot_distance_pose(df_pose):
    if 'left_shoulder_x' in df_pose.columns and 'right_shoulder_x' in df_pose.columns and \
       'left_shoulder_y' in df_pose.columns and 'right_shoulder_y' in df_pose.columns:
        release_frame = 0
        try:
            left_shoulder = df_pose.loc[df_pose['relative_frame'] == 0, ['left_shoulder_x', 'left_shoulder_y']].values[0]
            right_shoulder = df_pose.loc[df_pose['relative_frame'] == 0, ['right_shoulder_x', 'right_shoulder_y']].values[0]
            distance = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + (right_shoulder[1] - left_shoulder[1])**2)
            return distance
        except IndexError:
            return np.nan
    return np.nan

def calculate_apex_height_pose(df_pose):
    if 'head_y' in df_pose.columns:
        apex = df_pose['head_y'].max()
        return apex
    return np.nan

def calculate_release_time_pose(df_pose, release_frame):
    if 'relative_frame' not in df_pose.columns:
        return np.nan
    time = release_frame * 0.04
    return time

def calculate_release_curvature_pose(df_pose):
    angles = [col for col in df_pose.columns if 'angle' in col]
    if not angles:
        return np.nan
    curvature = df_pose[angles].diff().abs().sum(axis=1).sum()
    return curvature

def calculate_kpis_pose(df_pose):
    if df_pose.empty:
        return {}
    shot_distance = calculate_shot_distance_pose(df_pose)
    apex_height = calculate_apex_height_pose(df_pose)
    release_time = calculate_release_time_pose(df_pose, release_frame=0)
    release_curvature = calculate_release_curvature_pose(df_pose)
    asymmetry_index = calculate_asymmetry_index(df_pose)
    kinematic_sequence = compute_kinematic_chain_sequence(df_pose)
    kinematic_chain_sequence_score = len(kinematic_sequence)
    pivotal_score = calculate_pivotal_score({
        'Shot Distance': shot_distance,
        'Apex Height': apex_height,
        'Asymmetry Index': asymmetry_index,
    })
    return {
        'Shot Distance': shot_distance,
        'Apex Height': apex_height,
        'Release Time': release_time,
        'Release Curvature': release_curvature,
        'Asymmetry Index': asymmetry_index,
        'Kinematic Chain Sequencing': kinematic_sequence,
        'Kinematic Chain Sequence Score': kinematic_chain_sequence_score,
        'Pivotal Score': pivotal_score
    }

def calculate_release_quality(df_spin):
    if 'eX' not in df_spin.columns:
        return np.nan
    release_quality = (df_spin['eX'].mean() + 1) * 50
    return release_quality

def calculate_release_consistency(df_spin):
    if 'eX' not in df_spin.columns:
        return np.nan
    release_consistency = df_spin['eX'].std()
    return release_consistency

def plot_spin_bullseye(df_spin):
    if 'eX' not in df_spin.columns or 'eY' not in df_spin.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_spin['eX'], y=df_spin['eY'],
        mode='markers',
        marker=dict(size=10, color=df_spin['eX'], colorscale='RdYlGn', showscale=True)
    ))
    fig.update_layout(
        title="Spin Bulls-eye Visualization",
        xaxis_title="eX",
        yaxis_title="eY",
        shapes=[
            dict(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line_color="black"),
            dict(type="circle", xref="x", yref="y", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5, line_color="gray"),
            dict(type="circle", xref="x", yref="y", x0=-0.25, y0=-0.25, x1=0.25, y1=0.25, line_color="lightgray")
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_shot_distance(df_ball, release_frame):
    if df_ball.empty or pd.isna(release_frame):
        return np.nan
    x_rim, z_rim = 250 * 0.0254, 400 * 0.0254
    try:
        ball_x = df_ball.loc[df_ball['frame'] == release_frame, 'center_x_m'].values[0]
        ball_z = df_ball.loc[df_ball['frame'] == release_frame, 'center_z_m'].values[0]
    except IndexError:
        return np.nan
    distance = np.sqrt((ball_x - x_rim)**2 + (ball_z - z_rim)**2)
    return distance

def calculate_release_curvature(df_ball):
    if df_ball.empty:
        return np.nan
    x = df_ball['center_x']
    y = df_ball['center_y']
    coeffs = np.polyfit(x, y, 2)
    curvature = 2 * coeffs[0]
    return curvature

def calculate_lateral_curvature(df_ball):
    if df_ball.empty:
        return np.nan
    x = df_ball['center_x_m']
    t = df_ball['time']
    coeffs = np.polyfit(t, x, 2)
    lateral_curvature = 2 * coeffs[0]
    return lateral_curvature

def calculate_kpis_spin(df_spin):
    if df_spin.empty:
        return {}
    release_quality = calculate_release_quality(df_spin)
    pivotal_score = calculate_pivotal_score({
        'Release Velocity': np.nan,
        'Shot Distance': np.nan,
        'Release Angle': np.nan,
        'Apex Height': np.nan,
        'Release Quality': release_quality
    })
    return {
        'Release Quality': release_quality,
        'Pivotal Score': pivotal_score
    }

def calculate_consistency_percentage(kpi_list, kpi_key='Release Quality'):
    if not kpi_list:
        return np.nan
    qualities = [k.get(kpi_key, np.nan) for k in kpi_list]
    qualities = [q for q in qualities if not pd.isna(q)]
    if not qualities:
        return np.nan
    std_val = np.std(qualities)
    consistency_score = 100 - (std_val / 30 * 100)
    return np.clip(consistency_score, 0, 100)

def display_clickable_kpi_card(label, value, benchmark_values, unit="", player_average=None, extra_lines=None):
    """
    Display a clickable KPI card that shows a large KPI value and,
    when clicked, reveals additional details including the player's average.
    
    Parameters:
      label (str): The display name for the KPI.
      value (float): The actual KPI value.
      benchmark_values (array-like): Array of benchmark values for computing the percentile.
      unit (str): Unit to display (e.g. "ft").
      player_average (float, optional): Average value for the player (for this KPI).
      extra_lines (list, optional): Extra lines of information to display when expanded.
    """
    # Determine background color from benchmarks
    color = get_percentile_color(value, benchmark_values)
    color_mapping = {
        "green": "#A0E6A0",
        "yellow": "#FFF3B0",
        "orange": "#FFD8A8",
        "red": "#FFB0B0",
        "gray": "#D3D3D3"
    }
    background_color = color_mapping.get(color, "#FFF3B0")
    
    # Determine arrow indicating above/below average
    arrow = ""
    if player_average is not None:
        if value > player_average:
            arrow = "↑"
        elif value < player_average:
            arrow = "↓"
        else:
            arrow = "="
    
    # Create summary card HTML with a larger KPI value
    card_summary = f"""
    <div style='padding: 15px; background: {background_color}; border-radius: 8px;
                border: 1px solid #e0e0e0; text-align: center; margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
         <h3 style='color: #333333; margin: 0 0 10px 0; font-family: Arial; font-size: 20px;'>
             {label.upper()}
         </h3>
         <div style='padding: 10px; border-radius: 6px; margin: 5px 0;
                     background: rgba(255,255,255,0.8);'>
             <p style='font-size: 32px; margin: 8px 0; color: #333333;'>
                 <strong>{value:.1f} {unit}</strong> {arrow}
             </p>
         </div>
    </div>
    """
    
    # Use an expander so that clicking reveals additional details
    with st.expander("", expanded=False):
        # Show the main card
        st.markdown(card_summary, unsafe_allow_html=True)
        # Show player average if provided
        if player_average is not None:
            st.markdown(f"**Player Average:** {player_average:.1f} {unit}")
        # Display any extra details provided
        if extra_lines:
            for line in extra_lines:
                st.markdown(line)


import streamlit.components.v1 as components

def get_metallic_color(value, player_average, min_value=None, max_value=None):
    """
    Return a metallic color (red to green gradient) based on how close the value is to the player_average,
    or use hardcoded thresholds if no player_average is available.
    Uses a metallic gradient (e.g., red #FF0000 to green #00FF00).
    """
    if pd.isna(value):
        return "#D3D3D3"  # Gray for NaN

    # If player_average is None or NaN, use hardcoded thresholds
    if pd.isna(player_average) or player_average is None:
        # Hardcoded thresholds for each KPI (adjust as needed)
        if "Velocity" in str(value):
            min_value, max_value = 5, 15  # feet/s (e.g., 5–15 ft/s for release velocity)
            target = 10  # Optimal release velocity in feet/s
        elif "Height" in str(value):
            min_value, max_value = 5, 10  # feet (e.g., 5–10 ft for release/apex height)
            target = 8  # Optimal height in feet
        elif "Distance" in str(value):
            min_value, max_value = 5, 30  # feet (e.g., 5–30 ft for shot distance)
            target = 15  # Optimal distance in feet
        elif "Time" in str(value):
            min_value, max_value = 0.2, 0.6  # seconds (e.g., 0.2–0.6 s for release time)
            target = 0.4  # Optimal release time in seconds
        elif "Curvature" in str(value):
            min_value, max_value = 0.1, 0.5  # 1/ft (e.g., 0.1–0.5 1/ft for release curvature)
            target = 0.3  # Optimal curvature in 1/ft
        elif "Deviation" in str(value):
            min_value, max_value = 0, 0.5  # feet (e.g., 0–0.5 ft for lateral deviation)
            target = 0.1  # Optimal deviation in feet
        else:
            min_value, max_value = 0, 100  # Default range
            target = 50

        # Normalize the difference between value and target to [0, 1]
        diff = abs(value - target)
        range_span = max_value - min_value
        if range_span == 0:
            normalized_diff = 0
        else:
            normalized_diff = min(1, diff / range_span)

    else:
        # Use player average for color coding
        # Define dynamic range if min_value and max_value are not provided
        if min_value is None or max_value is None:
            if "Velocity" in str(value):
                min_value, max_value = 0, 15  # feet/s
            elif "Height" in str(value):
                min_value, max_value = 0, 10  # feet
            elif "Distance" in str(value):
                min_value, max_value = 0, 50  # feet
            elif "Time" in str(value):
                min_value, max_value = 0, 1  # seconds
            elif "Curvature" in str(value):
                min_value, max_value = 0, 0.5  # 1/feet
            elif "Deviation" in str(value):
                min_value, max_value = 0, 0.5  # feet
            else:
                min_value, max_value = 0, 100  # Default range

        # Normalize the difference between value and player_average to [0, 1]
        diff = abs(value - player_average)
        range_span = max_value - min_value
        if range_span == 0:
            normalized_diff = 0
        else:
            normalized_diff = min(1, diff / range_span)

    # Create a metallic gradient from red (#FF0000) to green (#00FF00)
    r = int(255 * (1 - normalized_diff))  # Red decreases from 255 to 0
    g = int(255 * normalized_diff)        # Green increases from 0 to 255
    b = 0  # Blue stays at 0 for a metallic look
    color = f"#{r:02x}{g:02x}{b:02x}"

    return color

# Update format_kpi_with_color to use the new metallic color logic
def format_kpi_with_color(name, value, player_average, benchmarks=None, min_value=None, max_value=None):
    color = get_metallic_color(value, player_average, min_value, max_value)
    arrow = "↑" if value > player_average else "↓" if value < player_average else "="
    return f"<span style='color: {color}; font-weight: bold;'>{value:.1f} {arrow}</span>"

def display_clickable_kpi_card(label, value, player_average, unit="", benchmarks=None, min_value=None, max_value=None, extra_lines=None):
    """
    Display a clickable KPI card with a metallic color spectrum based on proximity to player average,
    and an arrow indicating above/below average.
    
    Parameters:
      label (str): The display name for the KPI.
      value (float): The actual KPI value in feet.
      player_average (float): Average value for the player (in feet).
      unit (str): Unit to display (e.g., "ft").
      benchmarks (array-like, optional): Not used here but kept for compatibility.
      min_value (float, optional): Minimum value for the KPI range.
      max_value (float, optional): Maximum value for the KPI range.
      extra_lines (list, optional): Extra lines of information to display when expanded.
    """
    color = get_metallic_color(value, player_average, min_value, max_value)
    arrow = "↑" if value > player_average else "↓" if value < player_average else "="

    # Create summary card HTML with a larger KPI value and metallic color
    card_summary = f"""
    <div style='padding: 15px; background: {color}; border-radius: 8px;
                border: 1px solid #e0e0e0; text-align: center; margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
         <h3 style='color: #333333; margin: 0 0 10px 0; font-family: Arial; font-size: 20px;'>
             {label.upper()}
         </h3>
         <div style='padding: 10px; border-radius: 6px; margin: 5px 0;
                     background: rgba(255,255,255,0.8);'>
             <p style='font-size: 32px; margin: 8px 0; color: #333333;'>
                 <strong>{value:.1f} {unit}</strong> {arrow}
             </p>
         </div>
    </div>
    """
    
    # Use an expander so that clicking reveals additional details
    with st.expander("", expanded=False):
        st.markdown(card_summary, unsafe_allow_html=True)
        if player_average is not None:
            st.markdown(f"**Player Average:** {player_average:.1f} {unit}")
        if extra_lines:
            for line in extra_lines:
                st.markdown(line)

def animated_flip_kpi_card(label, value, unit, player_average, min_value=None, max_value=None, extra_html=""):
    """
    Renders an animated flip card with a metallic color spectrum based on proximity to player average,
    and an arrow indicating above/below average, handling None values for player_average.
    
    Parameters:
      label (str): KPI name.
      value (float): KPI value in feet.
      unit (str): Unit string (e.g., "ft").
      player_average (float, optional): Player's average for this KPI in feet.
      min_value (float, optional): Minimum value for the KPI range.
      max_value (float, optional): Maximum value for the KPI range.
      extra_html (str, optional): Additional HTML content to show on the back.
    """
    # Default arrow and color if player_average is None
    if pd.isna(player_average) or player_average is None:
        arrow = ""
        color = "#D3D3D3"  # Gray for unknown average
    else:
        color = get_metallic_color(value, player_average, min_value, max_value)
        arrow = "↑" if value > player_average else "↓" if value < player_average else "="

    html_code = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400&display=swap');
      .flip-card {{
        background-color: transparent;
        width: 300px;
        height: 200px;
        perspective: 1000px;
        margin: auto;
        cursor: pointer;
      }}
      .flip-card-inner {{
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.8s;
        transform-style: preserve-3d;
      }}
      .clicked .flip-card-inner {{
        transform: rotateY(180deg);
      }}
      .flip-card-front, .flip-card-back {{
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border: 2px solid #aaa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-family: 'Montserrat', sans-serif;
        font-weight: 300;
      }}
      .flip-card-front {{
        background: {color};
        color: #333;
      }}
      .flip-card-back {{
        background: linear-gradient(135deg, #b3b3b3, #808080);
        color: #fff;
        transform: rotateY(180deg);
        padding: 15px;
        font-size: 16px;
      }}
      .kpi-value {{
        font-size: 36px;
        margin: 10px 0;
      }}
    </style>
    <script>
      function toggleFlip(id) {{
          var el = document.getElementById(id);
          if(el.classList.contains("clicked")) {{
              el.classList.remove("clicked");
          }} else {{
              el.classList.add("clicked");
          }}
      }}
    </script>
    <div id="{label.replace(' ', '_')}" class="flip-card" onclick="toggleFlip('{label.replace(' ', '_')}')">
      <div class="flip-card-inner">
        <div class="flip-card-front">
          <h3>{label.upper()}</h3>
          <div class="kpi-value">{value:.1f} {unit} {arrow}</div>
        </div>
        <div class="flip-card-back">
          <p><strong>Player Average:</strong> {player_average if player_average is not None else "N/A"} {unit}</p>
          {extra_html}
        </div>
      </div>
    </div>
    """
    components.html(html_code, height=250)
