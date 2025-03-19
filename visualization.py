#visualization.py
import os
import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd 
import logging
from scipy.signal import savgol_filter
from math import atan2, degrees
from config import TEAMS
from kpi import format_kpi_with_color
from utils import humanize_label, add_time_column

logger = logging.getLogger(__name__)

def plot_single_shot(df, title="Single Shot Analysis", fps=25):
    """
    Plots selected angle columns from the DataFrame as a time series.
    The x-axis is calculated as frame/fps (i.e. time in seconds) rather than centering around a release.
    Smoothing is applied to the selected series.
    """
    if df.empty:
        st.warning("No data to plot.")
        return

    # Use the 'frame' column to create a time column if it exists.
    if "frame" in df.columns:
        df = add_time_column(df, fps)
        x_axis = df["time"]
        x_label = "Time (s)"
    else:
        x_axis = df.index
        x_label = "Frame Index"

    # Find angle columns and humanize their names for display.
    angle_cols = [col for col in df.columns if col.endswith('_angle')]
    if not angle_cols:
        st.info("No angle columns found to plot.")
        return

    selected_angles = st.multiselect("Select angle columns", angle_cols, default=angle_cols[:1])
    if not selected_angles:
        st.write("No angle columns selected.")
        return

    # (Optional smoothing:) Here we simply use a rolling mean.
    # In your full code you might call your smooth_series() function.
    smoothed_data = {}
    window = 5  # Adjust window as needed
    for col in selected_angles:
        # Replace NaNs by forward/backward fill before smoothing.
        series = df[col].fillna(method="ffill").fillna(method="bfill")
        smoothed_data[col] = series.rolling(window=window, center=True, min_periods=1).mean()

    fig = go.Figure()
    for col in selected_angles:
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=smoothed_data[col],
            mode="lines",
            name=humanize_label(col)
        ))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Angle (deg)")
    st.plotly_chart(fig, use_container_width=True)

def show_brand_header(player_names, team_abbreviation):
    """
    Displays a header with player names and team logo.
    """
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if isinstance(player_names, list):
            players_display = ", ".join(player_names)
        else:
            players_display = player_names
        st.markdown(
            f"<h1 style='text-align: center; color: #000000;'>{players_display or 'Unknown Player'}</h1>",
            unsafe_allow_html=True
        )
        if team_abbreviation and team_abbreviation.strip().lower() != "n/a":
            team_info = TEAMS.get(team_abbreviation.lower(), {"full_name": team_abbreviation, "logo": None})
            team_full_name = team_info["full_name"]
            st.markdown(
                f"<h3 style='text-align: center; color: #000000;'>{team_full_name}</h3>",
                unsafe_allow_html=True
            )
            team_logo_filename = team_info.get("logo")
            if team_logo_filename:
                team_logo_path = os.path.join("images", "teams", team_logo_filename)
                team_logo = load_image_from_file(team_logo_path)
                if team_logo:
                    encoded_logo = base64.b64encode(team_logo).decode()
                    st.markdown(
                        f"""<div style="text-align: center;">
                            <img src="data:image/png;base64,{encoded_logo}" style="max-width: 150px;"/>
                        </div>""",
                        unsafe_allow_html=True
                    )

def load_image_from_file(filepath):
    """Loads an image from disk."""
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except Exception as e:
        st.warning(f"Could not load image from {filepath}: {e}")
        return None

def visualize_phases(df, phases, job_id, segment_id, fps=25):
    st.info("Phase visualization is not fully implemented in this example.")

# visualization.py - Add more plotting functions

def plot_velocity_profile(df_ball):
    """Plot ball velocity over time"""
    if 'vel_x' not in df_ball.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ball['time'], y=df_ball['vel_x'], name='Velocity'))
    fig.update_layout(title="Ball Velocity Profile", 
                     xaxis_title="Time (s)", yaxis_title="Velocity (in/s)")
    st.plotly_chart(fig, use_container_width=True)

def plot_trajectory_arc(df_ball):
    """Plot ball trajectory in 2D space"""
    if 'center_x' not in df_ball.columns or 'center_y' not in df_ball.columns:
        return
    fig = px.scatter(df_ball, x='center_x', y='center_y', 
                    title="Ball Trajectory", color='time')
    fig.update_layout(xaxis_title="X Position", yaxis_title="Y Position")
    st.plotly_chart(fig, use_container_width=True)

def plot_asymmetry_radar(pose_kpis):
    """Radar plot for asymmetry analysis"""
    categories = ['Shoulder', 'Elbow', 'Hip', 'Knee', 'Ankle']
    values = [
        pose_kpis.get('Asymmetry Index Shoulder', 0),
        pose_kpis.get('Asymmetry Index Elbow', 0),
        pose_kpis.get('Asymmetry Index Hip', 0),
        pose_kpis.get('Asymmetry Index Knee', 0),
        pose_kpis.get('Asymmetry Index Ankle', 0)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Joint Asymmetry Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)

    # visualization.py - Add these components

def display_kpi_grid(kpis, benchmarks):
    """Display KPIs in a grid layout with color coding"""
    cols = st.columns(4)
    for idx, (name, value) in enumerate(kpis.items()):
        with cols[idx % 4]:
            formatted_value = format_kpi_with_color(name, value, benchmarks)
            st.markdown(f"""
                <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; margin: 10px;'>
                    <h4 style='margin:0;'>{name}</h4>
                    <h2 style='margin:0;'>{formatted_value}</h2>
                </div>
            """, unsafe_allow_html=True)

def plot_kinematic_sequence(sequence):
    """Plot joint activation sequence"""
    fig = go.Figure()
    joints = sorted(sequence.items(), key=lambda x: x[1])
    fig.add_trace(go.Bar(
        x=[j[0].replace('_angle_vel', '') for j in joints],
        y=[j[1] for j in joints],
        orientation='v'
    ))
    fig.update_layout(title="Kinematic Sequencing", xaxis_title="Joint", yaxis_title="Frame")
    st.plotly_chart(fig, use_container_width=True)

def plot_spin_analysis(df_spin):
    """Plot spin axis components over time"""
    if 'ex' not in df_spin.columns or 'ey' not in df_spin.columns:
        return
    fig = px.line(df_spin, x='time', y=['ex', 'ey', 'ez'], 
                 title="Spin Axis Components Over Time")
    st.plotly_chart(fig, use_container_width=True)

def plot_distance_over_height(df_ball):
    """
    Plot distance over height for the ball trajectory using Z for vertical motion.
    """
    if 'center_x_m' not in df_ball.columns or 'center_z_m' not in df_ball.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ball['center_x_m'], y=df_ball['center_z_m'],
        mode='lines+markers',
        name="Ball Trajectory"
    ))
    fig.update_layout(
        title="Distance Over Height",
        xaxis_title="Distance (X)",
        yaxis_title="Height (Z)"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_shot_analysis(df_ball, metrics):
    """
    Create interactive basketball shot analysis visualization with two 2D trajectory plots:

    - Side View: Uses the remapped Basketball_X_ft (in feet) versus Basketball_Z (converted to feet).
      Instead of using the raw x‐values, we recenter them by subtracting the hoop’s x coordinate.
      This yields a horizontal axis from -2 to 2 (2 ft on either side of the hoop) with positive values on the left.
      A dotted red line shows the projected ball path after release.

    - Rear View: Uses the remapped Basketball_Y_ft (in feet) versus Basketball_Z (converted to feet).
      Its horizontal axis is fixed to [-2, 2] ft (with the release point at 0).

    Both plots have a fixed vertical (Z) axis from 2.5 to 11 ft.
    The visual window covers the last 32 frames before release.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter
    import logging

    logger = logging.getLogger(__name__)
    INCHES_TO_FEET = 1 / 12

    # Helper function: smooth a slice of data.
    def get_slice(data, start, end):
        if isinstance(data, (pd.Series, np.ndarray, list)) and len(data) > 0:
            seg = data[start:end].to_numpy() if hasattr(data, 'to_numpy') else np.array(data[start:end])
            if len(seg) > 3:
                win_length = min(11, len(seg) - 1)
                if win_length % 2 == 0:
                    win_length += 1
                seg = savgol_filter(seg, window_length=win_length, polyorder=2)
            return seg
        return np.array([])

    # Clamp index helper.
    def clamp_index(idx, max_idx):
        return max(0, min(int(idx), max_idx))

    max_idx = len(df_ball) - 1
    release_idx = clamp_index(metrics.get('release_idx', 0), max_idx)
    # Use the last 32 frames prior to release.
    start_idx = max(0, release_idx - 32)
    # For this visual, force the lift marker to be at the very first frame of the window.
    lift_idx = start_idx
    # Use the set point computed in metrics (if within the window).
    set_idx = clamp_index(metrics.get('set_idx', release_idx), max_idx)

    # --- SIDE VIEW ---
    # Instead of plotting the raw remapped X coordinate, we compute a new horizontal coordinate
    # that represents the distance from the hoop.
    # Retrieve hoop position (in inches) from metrics and convert to feet.
    hoop_x_ft = metrics.get('hoop_x', 501.0) * INCHES_TO_FEET
    # Compute "side_x" as the distance from the hoop:
    #   side_x = hoop_x_ft - Basketball_X_ft
    side_x_all = hoop_x_ft - df_ball['Basketball_X_ft']
    # Slice for the visual window.
    traj_side_x = get_slice(side_x_all, start_idx, release_idx + 1)
    # For the vertical axis, use Basketball_Z converted to feet.
    traj_z = get_slice(df_ball['Basketball_Z'], start_idx, release_idx + 1) * INCHES_TO_FEET
    # For the post-release (projected) path, use frames after release.
    post_release_end = min(max_idx, release_idx + 20)
    post_side_x = get_slice(hoop_x_ft - df_ball['Basketball_X_ft'], release_idx, post_release_end + 1)
    post_traj_z = get_slice(df_ball['Basketball_Z'], release_idx, post_release_end + 1) * INCHES_TO_FEET

    # We want the side view x-axis fixed to [-2, 2].
    side_range = [-2, 2]

    # --- REAR VIEW ---
    # For the rear view, we use the remapped Y coordinate (in feet) as-is.
    traj_y = get_slice(df_ball['Basketball_Y_ft'], start_idx, release_idx + 1)
    # Use the same vertical trajectory.
    rear_range = [-2, 2]

    # Create subplots.
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Side View (Distance from Hoop vs. Height)", "Rear View (Y vs. Height)"),
        horizontal_spacing=0.1
    )

    # --- Side View Plot ---
    if len(traj_side_x) > 0 and len(traj_z) > 0:
        # Plot the main trajectory (solid line).
        fig.add_trace(
            go.Scatter(
                x=traj_side_x,
                y=traj_z,
                mode='lines',
                name='Trajectory',
                line=dict(color='rgba(31, 119, 180, 1)', width=4)
            ),
            row=1, col=1
        )
        # Mark the lift, set, and release points.
        phase_info = {
            'lift': {'idx': lift_idx, 'color': 'rgba(147, 112, 219, 1)', 'symbol': 'circle'},
            'set': {'idx': set_idx, 'color': 'rgba(255, 182, 193, 1)', 'symbol': 'diamond'},
            'release': {'idx': release_idx, 'color': 'rgba(255, 102, 102, 1)', 'symbol': 'x'}
        }
        for phase, info in phase_info.items():
            idx = info['idx']
            if start_idx <= idx <= release_idx:
                marker_side_x = hoop_x_ft - df_ball.at[idx, 'Basketball_X_ft']
                marker_z = df_ball.at[idx, 'Basketball_Z'] * INCHES_TO_FEET
                fig.add_trace(
                    go.Scatter(
                        x=[marker_side_x],
                        y=[marker_z],
                        mode='markers',
                        marker=dict(color=info['color'],
                                    symbol=info['symbol'],
                                    size=12),
                        name=f'{phase.capitalize()}'
                    ),
                    row=1, col=1
                )
        # Add a dotted (red) line for the post-release trajectory.
        if release_idx < post_release_end:
            fig.add_trace(
                go.Scatter(
                    x=post_side_x,
                    y=post_traj_z,
                    mode='lines',
                    name='Projected Path',
                    line=dict(color='red', width=2, dash='dot')
                ),
                row=1, col=1
            )
    else:
        fig.add_trace(go.Scatter(x=[0], y=[2.5], mode='text', text=["No Data"]), row=1, col=1)

    # --- Rear View Plot ---
    if len(traj_y) > 0 and len(traj_z) > 0:
        fig.add_trace(
            go.Scatter(
                x=traj_y,
                y=traj_z,
                mode='lines',
                name='Trajectory',
                line=dict(color='rgba(31, 119, 180, 1)', width=4)
            ),
            row=1, col=2
        )
        for phase, info in phase_info.items():
            idx = info['idx']
            if start_idx <= idx <= release_idx:
                marker_y = df_ball.at[idx, 'Basketball_Y_ft']
                marker_z = df_ball.at[idx, 'Basketball_Z'] * INCHES_TO_FEET
                fig.add_trace(
                    go.Scatter(
                        x=[marker_y],
                        y=[marker_z],
                        mode='markers',
                        marker=dict(color=info['color'],
                                    symbol=info['symbol'],
                                    size=12),
                        name=f'{phase.capitalize()}'
                    ),
                    row=1, col=2
                )
        # Add a dotted (red) line for post-release trajectory.
        if release_idx < post_release_end:
            post_traj_y = get_slice(df_ball['Basketball_Y_ft'], release_idx, post_release_end + 1)
            fig.add_trace(
                go.Scatter(
                    x=post_traj_y,
                    y=post_traj_z,
                    mode='lines',
                    name='Projected Path',
                    line=dict(color='red', width=2, dash='dot')
                ),
                row=1, col=2
            )
    else:
        fig.add_trace(go.Scatter(x=[0], y=[2.5], mode='text', text=["No Data"]), row=1, col=2)

    # --- Axes Configuration ---
    # For the side view, set the x-axis (distance from hoop) to [-2, 2].
    fig.update_xaxes(title_text="Distance from Hoop (ft)", row=1, col=1,
                     range=side_range, title_font=dict(size=14), tickfont=dict(size=12))
    # For both views, set the y-axis (height) to [2, 11] ft.
    fig.update_yaxes(title_text="Height (ft)", row=1, col=1,
                     range=[2, 11], title_font=dict(size=14), tickfont=dict(size=12), title_standoff=20)
    fig.update_xaxes(title_text="Lateral Position (ft)", row=1, col=2,
                     range=rear_range, title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(title_text="Height (ft)", row=1, col=2,
                     range=[2, 11], title_font=dict(size=14), tickfont=dict(size=12), title_standoff=20)

    # --- Overall Layout ---
    fig.update_layout(
        height=800,
        width=1400,
        title_text="Ball Path Analysis (32 Frames Before Release)",
        title_x=0.38,
        title_font=dict(size=20),
        margin=dict(t=120, b=100, l=80, r=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=12)),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        showlegend=True
    )
    # Ensure equal aspect ratio for both subplots.
    for col in [1, 2]:
        fig.update_xaxes(row=1, col=col, scaleanchor=f"y{col}", scaleratio=1)
    for annotation in fig.layout.annotations:
        annotation.y = 1.05

    return fig






def plot_foot_alignment(df):
    """Professional foot alignment visualization with error handling"""
    fig = go.Figure()
    basket_pos = (41.75, 0)
    
    # Style configuration
    FOOT_STYLE = {
        'left': {'color': '#89CFF0', 'width': 6},
        'right': {'color': '#B0E0E6', 'width': 6}
    }
    
    
    # Plot feet with error handling
    for side in ['left', 'right']:
        try:
            heel_x = f"{side.upper()}HEEL_X"
            heel_y = f"{side.upper()}HEEL_Y"
            toe_x = f"{side.upper()}BTOE_X"
            toe_y = f"{side.upper()}BTOE_Y"
            
            if all(col in df.columns for col in [heel_x, heel_y, toe_x, toe_y]):
                heel = (df[heel_x].mean(), df[heel_y].mean())
                toe = (df[toe_x].mean(), df[toe_y].mean())
                
                # Foot line plot
                fig.add_trace(go.Scatter(
                    x=[heel[0], toe[0]],
                    y=[heel[1], toe[1]],
                    mode='lines',
                    line=FOOT_STYLE[side],
                    name=f"{side.capitalize()} Foot"
                ))
                
                # Angle calculation
                midpoint = ((heel[0]+toe[0])/2, (heel[1]+toe[1])/2)
                foot_vec = np.array(toe) - np.array(heel)
                basket_vec = np.array(basket_pos) - np.array(midpoint)
                
                # Safe angle calculation
                cosine = np.dot(foot_vec, basket_vec) / (np.linalg.norm(foot_vec) * np.linalg.norm(basket_vec))
                cosine = np.clip(cosine, -1.0, 1.0)  # Prevent arccos errors
                angle = np.degrees(np.arccos(cosine))
                
                # Annotation positioning
                fig.add_annotation(
                    x=midpoint[0],
                    y=midpoint[1] + (0.2 if side == 'left' else -0.2),
                    text=f"{angle:.1f}°",
                    showarrow=False,
                    font=dict(size=12, color=FOOT_STYLE[side]['color']),
                    bgcolor="rgba(255,255,255,0.9)"
                )
        except Exception as e:
            st.error(f"Error plotting {side} foot: {str(e)}")
            continue

    # Dynamic axis ranges
    x_vals = []
    y_vals = []
    for trace in fig.data:
        x_vals.extend(trace.x)
        y_vals.extend(trace.y)
    
    x_range = [min(x_vals + [40]) - 1, max(x_vals + [42]) + 1]
    y_range = [min(y_vals + [-2]), max(y_vals + [2])]

    fig.update_layout(
        title="Foot Alignment Analysis",
        xaxis_title="Court Position (ft)",
        yaxis_title="Lateral Position (ft)",
        xaxis_range=x_range,
        yaxis_range=y_range,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=20, r=20, b=20, t=40),
        height=400,
        plot_bgcolor='rgba(248,248,248,1)'
    )
    
    return fig

def plot_release_angle_analysis(metrics):
    """Display release angle classification as compact text card"""
    classification = metrics.get('release_class', {})
    actual_angle = metrics.get('release_angle', 0)
    optimal_range = classification.get('optimal_range', 'N/A')
    classification_name = classification.get('classification', 'Unknown')
    
    # Pastel yellow color
    card_color = '#FFF3B0'  # Soft pastel yellow
    
    # Create compact text display
    st.markdown(f"""
        <div style='padding: 15px; 
                    background: {card_color};
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                    text-align: center;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #333333; 
                       margin: 0 0 10px 0;
                       font-family: Arial;
                       font-size: 20px;'>
                RELEASE ANGLE ANALYSIS
            </h3>
            <div style='padding: 10px;
                        border-radius: 6px;
                        margin: 5px 0;
                        background: rgba(255,255,255,0.8);'>
                <p style='font-size: 18px; margin: 8px 0; color: #333333;'>
                    <strong>Actual:</strong> {actual_angle:.1f}°
                </p>
                <p style='font-size: 16px; margin: 8px 0; color: #333333;'>
                    <strong>Class:</strong> {classification_name}
                </p>
                <p style='font-size: 16px; margin: 8px 0; color: #333333;'>
                    <strong>Optimal:</strong> {optimal_range}
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def plot_3d_ball_path(df_ball):
    """
    Plot a 3D animation of the ball path using XYZ data.
    Supports both 'center_x' and 'Basketball_X' column naming conventions.
    """
    # Define possible column naming conventions
    x_col = 'Basketball_X' if 'Basketball_X' in df_ball.columns else 'center_x'
    y_col = 'Basketball_Y' if 'Basketball_Y' in df_ball.columns else 'center_y'
    z_col = 'Basketball_Z' if 'Basketball_Z' in df_ball.columns else 'center_z'

    # Check if required columns exist
    required_columns = [x_col, y_col, z_col]
    if not all(col in df_ball.columns for col in required_columns):
        st.error(f"Missing required columns: {required_columns}")
        return None  # Return None if columns are missing

    # Debugging: Check for NaN values
    if df_ball[required_columns].isnull().any().any():
        st.error("NaN values found in ball trajectory data.")
        return None  # Return None if NaN values are present

    # Create 3D plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df_ball[x_col],  # Use the correct column name
        y=df_ball[y_col],  # Use the correct column name
        z=df_ball[z_col],  # Use the correct column name
        mode='lines+markers',
        marker=dict(
            size=4,
            color=df_ball.index,  # Use frame index for color
            colorscale='Blues',   # Use blue color scale
            line=dict(width=1, color='white')  # White border for markers
        ),
        line=dict(
            color='#1F78B4',  # Medium blue for the line
            width=4
        )
    )])
    
    # Update layout
    fig.update_layout(
        title="3D Ball Path",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)  # Adjust camera view
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig  # Return the Plotly figure object


def plot_velocity_vs_angle(df_ball, release_frame):
    """
    Plot release velocity vs. release angle at the moment of release.
    """
    if 'vel_x' not in df_ball.columns or 'vel_y' not in df_ball.columns:
        return
    # Calculate release angle at the release frame
    df_after = df_ball[df_ball['frame'] > release_frame].head(5)
    if df_after.empty:
        return
    dy = df_after['center_y'].mean() - df_ball.loc[df_ball['frame'] == release_frame, 'center_y'].values[0]
    dx = df_after['center_x'].mean() - df_ball.loc[df_ball['frame'] == release_frame, 'center_x'].values[0]
    release_angle = degrees(atan2(dy, dx))
    release_velocity = df_ball.loc[release_frame, 'vel_x']

    # Plot single point
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[release_angle], y=[release_velocity],
        mode='markers',
        name="Release Point"
    ))
    fig.update_layout(
        title="Release Velocity vs. Release Angle",
        xaxis_title="Release Angle (degrees)",
        yaxis_title="Release Velocity (m/s)"
    )
    st.plotly_chart(fig, use_container_width=True)

def get_kpi_color(value, thresholds):
    """
    Return a color based on the value and thresholds.
    """
    if value < thresholds[0]:
        return "red"
    elif value < thresholds[1]:
        return "orange"
    elif value < thresholds[2]:
        return "yellow"
    else:
        return "green"
def plot_spin_bullseye(df_spin):
    """
    Plot a bulls-eye visualization for spin data.
    """
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

def plot_distance_over_height(df_ball):
    """
    Plot distance over height for the ball trajectory.
    """
    if 'center_x' not in df_ball.columns or 'center_y' not in df_ball.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ball['center_x'], y=df_ball['center_y'],
        mode='lines+markers',
        name="Ball Trajectory"
    ))
    fig.update_layout(
        title="Distance Over Height",
        xaxis_title="Distance (X)",
        yaxis_title="Height (Y)"
    )
    st.plotly_chart(fig, use_container_width=True)

def vector_angle(v1x, v1y, v2x, v2y):
    """
    Calculate the signed angle between two vectors in degrees.
    """
    dot = v1x * v2x + v1y * v2y
    det = v1x * v2y - v1y * v2x
    return np.degrees(np.arctan2(det, dot))

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

def create_body_alignment_visual(frame_data, hoop_x=501.0, hoop_y=0.0):
    """
    Create a 2D visualization comparing the orientations of Feet, Hips, and Shoulders,
    with each segment's offset computed relative to the Hips, using dynamic hoop position.
    
    For each segment:
      - Compute the midpoint from its left/right keypoints.
      - Compute the segment’s unit vector (from left to right).
      - Compute the rightward perpendicular vector as (seg_unit_y, -seg_unit_x) [90° clockwise].
      - Draw an arrow (fixed length) from the midpoint in the direction of the perpendicular.
    
    Then use the Hips' perpendicular angle as the baseline and compute:
      - Feet offset = (Feet perpendicular angle - Hips perpendicular angle)
      - Shoulders offset = (Shoulders perpendicular angle - Hips perpendicular angle)
      - Hips offset = 0° (baseline)
    
    These offsets are shown in a metrics box in the bottom left.
    
    Color palette (pastel shades):
      - Feet: navy ("#000080")
      - Hips: pastel blue ("#EF553B")
      - Shoulders: lighter pastel blue ("#64B5F6")
    
    Parameters:
        frame_data (dict or Series): Contains keypoint values for the frame.
        hoop_x: X-coordinate of the hoop in inches (default 501.0).
        hoop_y: Y-coordinate of the hoop in inches (default 0.0).
        
    Returns:
        fig (plotly.graph_objects.Figure): The alignment offset comparison visualization.
    """
    import numpy as np
    import plotly.graph_objects as go
    from math import atan2, degrees, radians, cos, sin, sqrt

    # Define colors
    colors = {
        "Feet": "#000080",       # navy
        "Hips": "#EF553B",       # pastel blue
        "Shoulders": "#64B5F6"   # lighter pastel blue
    }
    
    fig = go.Figure()
    
    # Dictionary to store computed perpendicular angles for each segment
    perp_angles = {}
    
    # List to store all x and y values (for axis adjustment)
    all_x = []
    all_y = []
    
    # Define segments with required keys
    segments = []
    
    # Feet: use average of heel and big toe for left/right
    # New names
    if all(key in frame_data for key in ['LHEEL_X', 'LBIGTOE_X', 'LHEEL_Y', 'LBIGTOE_Y',
                                         'RHEEL_X', 'RBIGTOE_X', 'RHEEL_Y', 'RBIGTOE_Y']):
        left_foot = ((frame_data['LHEEL_X'] + frame_data['LBIGTOE_X']) / 2,
                     (frame_data['LHEEL_Y'] + frame_data['LBIGTOE_Y']) / 2)
        right_foot = ((frame_data['RHEEL_X'] + frame_data['RBIGTOE_X']) / 2,
                      (frame_data['RHEEL_Y'] + frame_data['RBIGTOE_Y']) / 2)
        segments.append(("Feet", left_foot, right_foot))
    # Fallback to old names
    elif all(key in frame_data for key in ['LHEEL_X', 'LBTOE_X', 'LHEEL_Y', 'LBTOE_Y',
                                           'RHEEL_X', 'RBTOE_X', 'RHEEL_Y', 'RBTOE_Y']):
        left_foot = ((frame_data['LHEEL_X'] + frame_data['LBTOE_X']) / 2,
                     (frame_data['LHEEL_Y'] + frame_data['LBTOE_Y']) / 2)
        right_foot = ((frame_data['RHEEL_X'] + frame_data['RBTOE_X']) / 2,
                      (frame_data['RHEEL_Y'] + frame_data['RBTOE_Y']) / 2)
        segments.append(("Feet", left_foot, right_foot))
    
    # Hips: using left/right hip joint centers
    # New names
    if all(key in frame_data for key in ['LHIP_X', 'RHIP_X', 'LHIP_Y', 'RHIP_Y']):
        left_hip = (frame_data['LHIP_X'], frame_data['LHIP_Y'])
        right_hip = (frame_data['RHIP_X'], frame_data['RHIP_Y'])
        segments.append(("Hips", left_hip, right_hip))
    # Fallback to old names
    elif all(key in frame_data for key in ['LHJC_X', 'RHJC_X', 'LHJC_Y', 'RHJC_Y']):
        left_hip = (frame_data['LHJC_X'], frame_data['LHJC_Y'])
        right_hip = (frame_data['RHJC_X'], frame_data['RHJC_Y'])
        segments.append(("Hips", left_hip, right_hip))
    
    # Shoulders: using left/right shoulder joint centers
    # New names
    if all(key in frame_data for key in ['LSHOULDER_X', 'RSHOULDER_X', 'LSHOULDER_Y', 'RSHOULDER_Y']):
        left_shoulder = (frame_data['LSHOULDER_X'], frame_data['LSHOULDER_Y'])
        right_shoulder = (frame_data['RSHOULDER_X'], frame_data['RSHOULDER_Y'])
        segments.append(("Shoulders", left_shoulder, right_shoulder))
    # Fallback to old names
    elif all(key in frame_data for key in ['LSJC_X', 'RSJC_X', 'LSJC_Y', 'RSJC_Y']):
        left_shoulder = (frame_data['LSJC_X'], frame_data['LSJC_Y'])
        right_shoulder = (frame_data['RSJC_X'], frame_data['RSJC_Y'])
        segments.append(("Shoulders", left_shoulder, right_shoulder))
    
    # Fixed arrow length for drawing perpendicular arrows (in inches)
    arrow_length = 5 * 12  # 5 feet in inches
    
    # Process each segment
    for seg_name, left_pt, right_pt in segments:
        color = colors.get(seg_name, "#808080")
        
        # Draw the segment line
        fig.add_trace(go.Scatter(
            x=[left_pt[0], right_pt[0]],
            y=[left_pt[1], right_pt[1]],
            mode="lines+markers",
            line=dict(width=6, color=color),
            marker=dict(size=12),
            name=seg_name
        ))
        all_x.extend([left_pt[0], right_pt[0]])
        all_y.extend([left_pt[1], right_pt[1]])
        
        # Compute the midpoint
        mid_pt = ((left_pt[0] + right_pt[0]) / 2, (left_pt[1] + right_pt[1]) / 2)
        
        # Compute the segment vector and its norm
        seg_vec = (right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
        seg_norm = sqrt(seg_vec[0]**2 + seg_vec[1]**2)
        if seg_norm == 0:
            continue
        seg_unit = (seg_vec[0] / seg_norm, seg_vec[1] / seg_norm)
        
        # Compute the rightward perpendicular vector
        # (Rotate 90° clockwise: (dx,dy) -> (dy, -dx))
        perp = (seg_unit[1], -seg_unit[0])
        # Compute its angle in degrees
        perp_angle = degrees(atan2(perp[1], perp[0]))
        perp_angles[seg_name] = perp_angle
        
        # Draw an arrow from the midpoint along the perpendicular vector
        arrow_end = (mid_pt[0] + arrow_length * perp[0], mid_pt[1] + arrow_length * perp[1])
        fig.add_trace(go.Scatter(
            x=[mid_pt[0], arrow_end[0]],
            y=[mid_pt[1], arrow_end[1]],
            mode="lines+markers",
            line=dict(width=3, color=color),
            marker=dict(size=8),
            showlegend=False
        ))
        
        # Mark the midpoint
        fig.add_trace(go.Scatter(
            x=[mid_pt[0]],
            y=[mid_pt[1]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            showlegend=False
        ))
        all_x.append(mid_pt[0])
        all_y.append(mid_pt[1])
    
    # Use Hips' perpendicular angle as baseline
    baseline = perp_angles.get("Hips", None)
    offsets = {}
    if baseline is not None:
        for seg in perp_angles:
            # Compute difference relative to baseline
            diff = perp_angles[seg] - baseline
            # Normalize to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            offsets[seg] = diff
    else:
        # If no hips, set all offsets to 0
        for seg in perp_angles:
            offsets[seg] = 0.0

    # Prepare offset text (baseline relative to hips)
    offset_text = "Alignment Offsets (relative to Hips):<br>"
    for seg in ["Feet", "Shoulders"]:
        if seg in offsets:
            offset_text += f"{seg}: {offsets[seg]:.1f}°<br>"
    
    # Add offset metrics annotation in the bottom left (using paper coordinates)
    fig.add_annotation(
        xref="paper", x=0.01,
        yref="paper", y=0.01,
        text=offset_text,
        showarrow=False,
        font=dict(size=16, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    # Adjust axis ranges based on collected points (in inches)
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_margin = (x_max - x_min) * 0.5 if (x_max - x_min) != 0 else 5 * 12
        y_margin = (y_max - y_min) * 0.5 if (y_max - y_min) != 0 else 5 * 12
        fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin])
        fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])
    else:
        fig.update_xaxes(range=[-600, 600])  # Full court range in inches
        fig.update_yaxes(range=[-300, 300])  # Full court range in inches
    
    # Update overall layout (in inches)
    fig.update_layout(
        title="Body Alignment Comparison (Feet, Hips, Shoulders)",
        xaxis_title="Court Position (in)",
        yaxis_title="Lateral Position (in)",
        template="plotly_white",
        height=500,
        width=900,
        margin=dict(l=20, r=20, b=20, t=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
        plot_bgcolor='rgba(248,248,248,1)'
    )
    
    return fig

def create_foot_alignment_visual(frame_data, shot_distance, flip=False, hoop_x=41.75, hoop_y=0.0):
    """
    Create a foot alignment visualization at a specific frame with new column names and dynamic hoop position in feet.
    
    Parameters:
        frame_data: dict or Series containing the keypoint values in inches.
        shot_distance: The flipped shot distance (in feet).
        flip: Boolean flag indicating whether the shot was flipped.
        hoop_x: X-coordinate of the hoop in feet (default 41.75, or -41.75 if flipped).
        hoop_y: Y-coordinate of the hoop in feet (default 0.0).
        
    Returns:
        A Plotly figure.
    """
    INCHES_TO_FEET = 1 / 12
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    # Convert hoop position to feet if passed in inches
    hoop_pos = (hoop_x * 12, hoop_y * 12) if not flip else (-hoop_x * 12, hoop_y * 12)  # Flip to target opposite hoop

    foot_midpoints = {}
    x_vals = []
    y_vals = []

    # Plot each foot (Left and Right)
    for side, color in [('L', '#636EFA'), ('R', '#EF553B')]:
        heel_x = f'{side}HEEL_X'
        toe_x = f'{side}BIGTOE_X'  # New name
        heel_y = f'{side}HEEL_Y'
        toe_y = f'{side}BIGTOE_Y'  # New name
        
        # Try new names first
        if all(col in frame_data for col in [heel_x, toe_x, heel_y, toe_y]):
            heel = (frame_data[heel_x] * INCHES_TO_FEET, frame_data[heel_y] * INCHES_TO_FEET)
            toe = (frame_data[toe_x] * INCHES_TO_FEET, frame_data[toe_y] * INCHES_TO_FEET)
        # Fallback to old names
        elif all(col in frame_data for col in [heel_x, f'{side}BTOE_X', heel_y, f'{side}BTOE_Y']):
            heel = (frame_data[heel_x] * INCHES_TO_FEET, frame_data[heel_y] * INCHES_TO_FEET)
            toe = (frame_data[f'{side}BTOE_X'] * INCHES_TO_FEET, frame_data[f'{side}BTOE_Y'] * INCHES_TO_FEET)
        else:
            continue  # Skip if neither set is complete

        # Draw a thick line with markers for the foot vector
        fig.add_trace(go.Scatter(
            x=[heel[0], toe[0]],
            y=[heel[1], toe[1]],
            mode='lines+markers',
            line=dict(width=10, color=color),
            marker=dict(size=15, symbol='circle'),
            name=f"{'Left' if side=='L' else 'Right'} Foot"
        ))
        
        midpoint = ((heel[0] + toe[0]) / 2, (heel[1] + toe[1]) / 2)
        foot_midpoints[side] = midpoint
        x_vals.extend([heel[0], toe[0]])
        y_vals.extend([heel[1], toe[1]])
        
        # Compute the foot vector (from heel to toe) and the vector from midpoint to hoop
        foot_vec = np.array(toe) - np.array(heel)
        direction_vec = np.array(hoop_pos) - np.array(midpoint)
        norm_dir = np.linalg.norm(direction_vec) if np.linalg.norm(direction_vec) != 0 else 1
        cosine = np.dot(foot_vec, direction_vec) / (np.linalg.norm(foot_vec) * norm_dir)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        
        # Adjust angle if flipped
        if flip:
            angle = angle - 180
            
        fig.add_annotation(
            x=midpoint[0] + 0.5,  # Adjust for feet scale
            y=midpoint[1] - 0.2,
            text=f"{angle:.1f}°",
            showarrow=False,
            font=dict(size=16, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            xanchor="left"
        )
    
    # Compute overall foot center from available midpoints
    if 'L' in foot_midpoints and 'R' in foot_midpoints:
        foot_center = ((foot_midpoints['L'][0] + foot_midpoints['R'][0]) / 2,
                       (foot_midpoints['L'][1] + foot_midpoints['R'][1]) / 2)
    elif 'L' in foot_midpoints:
        foot_center = foot_midpoints['L']
    elif 'R' in foot_midpoints:
        foot_center = foot_midpoints['R']
    else:
        foot_center = (0, 0)
    
    # Dummy trace for the legend
    dummy_arrow = go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color="black", width=3),
        name="Direction of Hoop"
    )
    
    # Zoom in based on collected x and y values (in feet)
    if x_vals and y_vals:
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_margin = (x_max - x_min) * 0.5 if (x_max - x_min) != 0 else 5
        y_margin = (y_max - y_min) * 0.5 if (y_max - y_min) != 0 else 5
        fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin])
        fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])
    else:
        fig.update_xaxes(range=[-47, 47])  # Full half-court range in feet
        fig.update_yaxes(range=[-25, 25])  # Full court range in feet (lateral)
    
    fig.update_layout(
        title="Foot Alignment",
        xaxis_title="Court Position (ft)",
        yaxis_title="Lateral Position (ft)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, b=20, t=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
        plot_bgcolor='rgba(248,248,248,1)'
    )
    
    fig.add_trace(dummy_arrow)
    
    return fig

def plot_shot_location_in_inches(shot_x, shot_y):
    """
    Plot an overhead view of a basketball court (in inches) with
    a dot at the given shot_x, shot_y location.
    
    The standard coordinate system here:
      - The origin (0,0) is center court.
      - X from -564 (left sideline) to +564 (right sideline).
      - Y from -294 (near baseline) to +294 (far baseline).
    """
    fig = go.Figure()

    # Draw the outer boundary of the court
    # We'll add it as a shape with corners at ~ -564, -294 and +564, +294
    fig.add_shape(
        type="rect",
        x0=-564, y0=-294, x1=564, y1=294,
        line=dict(color="black", width=3),
        fillcolor="rgba(0,0,0,0)"  # transparent fill
    )

    # Example: half-court line at y=0
    fig.add_shape(
        type="line",
        x0=-564, y0=0, x1=564, y1=0,
        line=dict(color="black", width=2, dash="dot")
    )

    # Plot the shot location as a red dot
    fig.add_trace(go.Scatter(
        x=[shot_x], 
        y=[shot_y],
        mode="markers",
        marker=dict(size=12, color="red", symbol="circle"),
        name="Shot Location"
    ))

    # Layout settings
    fig.update_xaxes(range=[-600, 600], zeroline=False, showgrid=False)
    fig.update_yaxes(range=[-320, 320], zeroline=False, showgrid=False)
    fig.update_layout(
        title="Shot Location (Inches)",
        width=800, height=500,
        xaxis=dict(title="X (inches)"),
        yaxis=dict(title="Y (inches)", scaleanchor="x", scaleratio=1),
        showlegend=True
    )

    fig.show()