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
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter
    import logging

    logger = logging.getLogger(__name__)
    INCHES_TO_FEET = 1 / 12

    def get_slice(df, column, start, end, clip_min=None, clip_max=None):
        if column in df.columns and len(df) > 0:
            seg = df[column].iloc[start:end].to_numpy()
            if len(seg) == 0:
                return np.array([])  # Return empty array if no data
            window_length = min(11, len(seg) - 1) if len(seg) > 1 else 1
            if len(seg) > 3:  # Only smooth if enough points
                if window_length % 2 == 0:  # Ensure odd window length
                    window_length += 1
                seg = savgol_filter(seg, window_length=window_length, polyorder=2)
            if clip_min is not None or clip_max is not None:
                seg = np.clip(seg, clip_min, clip_max)
            return seg
        return np.array([])

    def clamp_index(idx, max_idx):
        return max(0, min(int(idx), max_idx))

    max_idx = len(df_ball) - 1
    release_idx = clamp_index(metrics.get('release_idx', 0), max_idx)
    start_idx = max(0, release_idx - 32)
    lift_idx = clamp_index(metrics.get('lift_idx', start_idx), max_idx)

    # Redefine set_idx: minimum X position, at least 10 frames after lift_idx
    set_window_start = max(lift_idx + 10, start_idx)
    set_window_end = release_idx + 1
    candidate_set_window = df_ball.iloc[set_window_start:set_window_end]
    if not candidate_set_window.empty and len(candidate_set_window) >= 1:
        set_idx = candidate_set_window['Basketball_X_ft'].idxmin()
    else:
        set_idx = release_idx
    logger.debug(f"Set_idx redefined as minimum X position between {set_window_start} and {set_window_end}: {set_idx}")

    # Store all indices in metrics with explicit keys
    metrics['lift_frame'] = lift_idx
    metrics['set_frame'] = set_idx
    metrics['release_frame'] = release_idx
    logger.debug(f"Stored indices in metrics - lift_frame: {lift_idx}, set_frame: {set_idx}, release_frame: {release_idx}")

    # --- SIDE VIEW ---
    traj_x = get_slice(df_ball, 'Basketball_X_ft', lift_idx, release_idx + 1)
    traj_z = get_slice(df_ball, 'Basketball_Z_ft', lift_idx, release_idx + 1, clip_min=2, clip_max=11)
    traj_v = get_slice(df_ball, 'velocity_magnitude', lift_idx, release_idx + 1)
    release_x = df_ball.at[release_idx, 'Basketball_X_ft']

    if np.any(traj_x < 0):
        logger.debug(f"Negative X values detected in traj_x (e.g., {traj_x[:5]}), flipping to positive")
        traj_x = -traj_x
        release_x = -release_x if release_x < 0 else release_x

    side_range = [release_x - 1, release_x + 3]
    if len(traj_x) > 0 and len(traj_z) > 0 and len(traj_v) > 0:
        min_len = min(len(traj_x), len(traj_z), len(traj_v))
        traj_x = traj_x[:min_len]
        traj_z = traj_z[:min_len]
        traj_v = traj_v[:min_len]
        mask = (traj_x >= side_range[0]) & (traj_x <= side_range[1]) & (traj_z >= 2) & (traj_z <= 11)
        traj_x = traj_x[mask]
        traj_z = traj_z[mask]
        traj_v = traj_v[mask]
    else:
        logger.error("Trajectory data empty after slicing")
        traj_x, traj_z, traj_v = [], [], []

    post_release_end = min(max_idx, release_idx + 20)
    proj_x = get_slice(df_ball, 'Basketball_X_ft', release_idx, post_release_end + 1)
    proj_z = get_slice(df_ball, 'Basketball_Z_ft', release_idx, post_release_end + 1, clip_min=2, clip_max=11)
    if np.any(proj_x < 0):
        logger.debug(f"Negative X values detected in proj_x (e.g., {proj_x[:5]}), flipping to positive")
        proj_x = -proj_x

    if len(proj_x) > 0 and len(proj_z) > 0:
        min_len = min(len(proj_x), len(proj_z))
        proj_x = proj_x[:min_len]
        proj_z = proj_z[:min_len]
        proj_mask = (proj_x >= side_range[0]) & (proj_x <= side_range[1]) & (proj_z >= 2) & (proj_z <= 11)
        proj_x = proj_x[proj_mask]
        proj_z = proj_z[proj_mask]
    else:
        logger.debug("Projection data empty after slicing")
        proj_x, proj_z = [], []

    tickvals_x = np.linspace(side_range[0], side_range[1], 5)
    ticktext_x = [f"{val:.1f}" for val in tickvals_x]
    tickvals_y = np.arange(2, 12, 1)

    # --- REAR VIEW (Reversed) ---
    traj_y = get_slice(df_ball, 'Basketball_Y_ft', lift_idx, release_idx + 1)
    traj_z_rear = traj_z
    traj_v_rear = traj_v
    rear_range = [-2, 2]
    if len(traj_y) > 0 and len(traj_z_rear) > 0 and len(traj_v_rear) > 0:
        min_len = min(len(traj_y), len(traj_z_rear), len(traj_v_rear))
        traj_y = -traj_y[:min_len]
        traj_z_rear = traj_z_rear[:min_len]
        traj_v_rear = traj_v_rear[:min_len]
        rear_mask = (traj_y >= -2) & (traj_y <= 2) & (traj_z_rear >= 2) & (traj_z_rear <= 11)
        traj_y = traj_y[rear_mask]
        traj_z_rear = traj_z_rear[rear_mask]
        traj_v_rear = traj_v_rear[rear_mask]
    else:
        logger.debug("Rear trajectory data empty after slicing")
        traj_y, traj_z_rear, traj_v_rear = [], [], []

    proj_y = get_slice(df_ball, 'Basketball_Y_ft', release_idx, post_release_end + 1)
    proj_z_rear = proj_z
    if len(proj_y) > 0 and len(proj_z_rear) > 0:
        min_len = min(len(proj_y), len(proj_z_rear))
        proj_y = -proj_y[:min_len]
        proj_z_rear = proj_z_rear[:min_len]
        proj_mask = (proj_y >= -2) & (proj_y <= 2) & (proj_z_rear >= 2) & (proj_z_rear <= 11)
        proj_y = proj_y[proj_mask]
        proj_z_rear = proj_z_rear[proj_mask]
    else:
        logger.debug("Rear projection data empty after slicing")
        proj_y, proj_z_rear = [], []

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Side View (X vs. Z)", "Rear View (Y vs. Z)"),
        horizontal_spacing=0.15
    )

    # --- SIDE VIEW PLOT ---
    if len(traj_x) > 0 and len(traj_z) > 0 and len(traj_v) > 0:
        fig.add_trace(
            go.Scatter(
                x=traj_x,
                y=traj_z,
                mode='lines+markers',
                name='Trajectory',
                line=dict(width=2, color='grey'),
                marker=dict(
                    size=8,
                    color=traj_v,
                    colorscale='Blues',
                    cmin=0,
                    cmax=max(traj_v.max(), 40),
                    colorbar=dict(
                        title=dict(text="Velocity (in/s)", side="right"),
                        thickness=15,
                        len=0.5,
                        x=1.05
                    )
                )
            ),
            row=1, col=1
        )
        phase_info = {
            'lift': {'idx': lift_idx, 'color': 'rgba(147, 112, 219, 1)', 'symbol': 'circle'},
            'set': {'idx': set_idx, 'color': 'rgba(255, 182, 193, 1)', 'symbol': 'diamond'},
            'release': {'idx': release_idx, 'color': 'rgba(255, 102, 102, 1)', 'symbol': 'x'}
        }
        for phase, info in phase_info.items():
            idx = info['idx']
            if lift_idx <= idx <= release_idx:
                marker_x = df_ball.at[idx, 'Basketball_X_ft']
                if marker_x < 0:
                    marker_x = -marker_x
                marker_z = df_ball.at[idx, 'Basketball_Z_ft']
                if side_range[0] <= marker_x <= side_range[1] and 2 <= marker_z <= 11:
                    fig.add_trace(
                        go.Scatter(
                            x=[marker_x],
                            y=[marker_z],
                            mode='markers',
                            marker=dict(color=info['color'], symbol=info['symbol'], size=12),
                            name=f'{phase.capitalize()}'
                        ),
                        row=1, col=1
                    )
        if release_idx < post_release_end and len(proj_x) > 0 and len(proj_z) > 0:
            fig.add_trace(
                go.Scatter(
                    x=proj_x,
                    y=proj_z,
                    mode='lines',
                    name='Projected Path',
                    line=dict(color='red', width=2, dash='dot')
                ),
                row=1, col=1
            )
    else:
        fig.add_trace(go.Scatter(x=[release_x], y=[2.0], mode='text', text=["No Data"]), row=1, col=1)

    # --- REAR VIEW PLOT (Reversed) ---
    if len(traj_y) > 0 and len(traj_z_rear) > 0 and len(traj_v_rear) > 0:
        fig.add_trace(
            go.Scatter(
                x=traj_y,
                y=traj_z_rear,
                mode='lines+markers',
                name='Trajectory',
                line=dict(width=2, color='grey'),
                marker=dict(
                    size=8,
                    color=traj_v_rear,
                    colorscale='Blues',
                    cmin=0,
                    cmax=max(traj_v_rear.max(), 40),
                    showscale=False
                )
            ),
            row=1, col=2
        )
        for phase, info in phase_info.items():
            idx = info['idx']
            if lift_idx <= idx <= release_idx:
                marker_y = -df_ball.at[idx, 'Basketball_Y_ft']
                marker_z = df_ball.at[idx, 'Basketball_Z_ft']
                if -2 <= marker_y <= 2 and 2 <= marker_z <= 11:
                    fig.add_trace(
                        go.Scatter(
                            x=[marker_y],
                            y=[marker_z],
                            mode='markers',
                            marker=dict(color=info['color'], symbol=info['symbol'], size=12),
                            name=f'{phase.capitalize()}'
                        ),
                        row=1, col=2
                    )
        if release_idx < post_release_end and len(proj_y) > 0 and len(proj_z_rear) > 0:
            fig.add_trace(
                go.Scatter(
                    x=proj_y,
                    y=proj_z_rear,
                    mode='lines',
                    name='Projected Path',
                    line=dict(color='red', width=2, dash='dot')
                ),
                row=1, col=2
            )
    else:
        fig.add_trace(go.Scatter(x=[0], y=[2.0], mode='text', text=["No Data"]), row=1, col=2)

    # --- AXES CONFIGURATION ---
    fig.update_xaxes(
        title_text="Horizontal Position (ft)", row=1, col=1,
        range=side_range, tickmode='array', tickvals=tickvals_x, ticktext=ticktext_x,
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        title_font=dict(size=14), tickfont=dict(size=12)
    )
    fig.update_yaxes(
        title_text="Height (ft)", row=1, col=1,
        range=[2, 11], tickmode='array', tickvals=tickvals_y, ticktext=[f"{val:.0f}" for val in tickvals_y],
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        title_font=dict(size=14), tickfont=dict(size=12), title_standoff=20
    )
    fig.update_xaxes(
        title_text="Lateral Position (ft)", row=1, col=2,
        range=rear_range,
        tickmode='array', tickvals=[-2, -1, 0, 1, 2], ticktext=["-2", "-1", "0", "1", "2"],
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        title_font=dict(size=14), tickfont=dict(size=12)
    )
    fig.update_yaxes(
        title_text="Height (ft)", row=1, col=2,
        range=[2, 11], tickmode='array', tickvals=tickvals_y, ticktext=[f"{val:.0f}" for val in tickvals_y],
        showgrid=True, gridwidth=1, gridcolor='lightgrey',
        title_font=dict(size=14), tickfont=dict(size=12), title_standoff=20
    )

    # --- OVERALL LAYOUT ---
    pixels_per_foot = 60
    subplot_width = 4 * pixels_per_foot
    subplot_height = 9 * pixels_per_foot
    total_width = subplot_width * 2 + 100

    fig.update_layout(
        height=subplot_height + 200, width=total_width,
        title_text="Ball Path Analysis",
        title_x=0.38, title_font=dict(size=20),
        margin=dict(t=120, b=100, l=80, r=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=12)),
        plot_bgcolor='rgba(255, 255, 255, 1)', paper_bgcolor='rgba(255, 255, 255, 1)',
        showlegend=True, autosize=False
    )

    for col in [1, 2]:
        fig.update_xaxes(row=1, col=col, scaleanchor=f"y{col}", scaleratio=1, constrain='domain')
        fig.update_yaxes(row=1, col=col, scaleanchor=f"x{col}", scaleratio=1, constrain='domain')

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

def create_body_alignment_visual(frame_data):
    """
    Create a 2D visualization of body alignment with the basket to the right.
    
    Parameters:
        frame_data (Series): Pose keypoints at release frame in feet (remapped coordinates).
    
    Returns:
        fig (plotly.graph_objects.Figure): Body alignment visualization.
    """
    import numpy as np
    import plotly.graph_objects as go
    from math import atan2, degrees, sqrt

    # Define colors for each segment
    colors = {"Feet": "#000080", "Hips": "#EF553B", "Shoulders": "#64B5F6"}
    fig = go.Figure()
    segments = []
    all_x, all_y = [], []
    midpoints = []

    # Define segments: Feet, Hips, Shoulders
    # Feet: Midpoint of heel and big toe for each foot
    if all(key in frame_data for key in ['LHEEL_X', 'LBIGTOE_X', 'LHEEL_Y', 'LBIGTOE_Y',
                                         'RHEEL_X', 'RBIGTOE_X', 'RHEEL_Y', 'RBIGTOE_Y']):
        left_foot = ((frame_data['LHEEL_X'] + frame_data['LBIGTOE_X']) / 2,
                     (frame_data['LHEEL_Y'] + frame_data['LBIGTOE_Y']) / 2)
        right_foot = ((frame_data['RHEEL_X'] + frame_data['RBIGTOE_X']) / 2,
                      (frame_data['RHEEL_Y'] + frame_data['RBIGTOE_Y']) / 2)
        segments.append(("Feet", left_foot, right_foot))

    # Hips
    if all(key in frame_data for key in ['LHIP_X', 'RHIP_X', 'LHIP_Y', 'RHIP_Y']):
        left_hip = (frame_data['LHIP_X'], frame_data['LHIP_Y'])
        right_hip = (frame_data['RHIP_X'], frame_data['RHIP_Y'])
        segments.append(("Hips", left_hip, right_hip))

    # Shoulders
    if all(key in frame_data for key in ['LSHOULDER_X', 'RSHOULDER_X', 'LSHOULDER_Y', 'RSHOULDER_Y']):
        left_shoulder = (frame_data['LSHOULDER_X'], frame_data['LSHOULDER_Y'])
        right_shoulder = (frame_data['RSHOULDER_X'], frame_data['RSHOULDER_Y'])
        segments.append(("Shoulders", left_shoulder, right_shoulder))

    # Plot segments and collect coordinates
    for seg_name, left_pt, right_pt in segments:
        color = colors.get(seg_name, "#808080")
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
        mid_x = (left_pt[0] + right_pt[0]) / 2
        mid_y = (left_pt[1] + right_pt[1]) / 2
        midpoints.append((mid_x, mid_y))

    # Handle case with no data
    if not all_x or not all_y:
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
    else:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

    # Add annotations with arrows
    annotation_x = x_max + 1  # Closer to body for tighter zoom
    annotation_y = {
        "Feet": y_min - 0.5,
        "Hips": (y_min + y_max) / 2,
        "Shoulders": y_max + 0.5
    }

    for seg_name, left_pt, right_pt in segments:
        color = colors.get(seg_name, "#808080")
        mid_x = (left_pt[0] + right_pt[0]) / 2
        mid_y = (left_pt[1] + right_pt[1]) / 2

        # Calculate width and angle
        width = sqrt((right_pt[0] - left_pt[0])**2 + (right_pt[1] - left_pt[1])**2)
        seg_vec = (right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
        basket_dir = (1, 0)  # Positive X-axis
        angle_rad = atan2(seg_vec[1], seg_vec[0]) - atan2(basket_dir[1], basket_dir[0])
        angle = degrees(angle_rad)
        angle = ((angle + 90) % 180) - 90  # Normalize to [-90, 90]

        text = f"{seg_name}<br>Width: {width:.1f} ft<br>Angle: {angle:.1f}°"
        fig.add_annotation(
            x=annotation_x,
            y=annotation_y[seg_name],
            ax=mid_x,
            ay=mid_y,
            text=text,
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            font=dict(size=10, color=color),  # Smaller font to reduce clutter
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    # Add small basket direction arrow
    if midpoints:
        avg_mid_x = np.mean([mp[0] for mp in midpoints])
        avg_mid_y = np.mean([mp[1] for mp in midpoints])
        arrow_length = 1  # Small arrow
        fig.add_trace(go.Scatter(
            x=[avg_mid_x, avg_mid_x + arrow_length],
            y=[avg_mid_y, avg_mid_y],
            mode="lines+markers",
            line=dict(width=2, color="red", dash="dash"),  # Dashed to reduce emphasis
            marker=dict(size=6, symbol="arrow-bar-right"),
            name="Basket Direction"
        ))

    # Tight zoom on body with space for annotations
    if all_x and all_y:
        padding = 0.5  # Minimal padding
        x_range = [x_min - padding, annotation_x + padding]
        y_range = [y_min - 1, y_max + 1]  # Enough to include annotations
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range, scaleanchor="x", scaleratio=1)  # Equal aspect ratio

    # Update layout
    fig.update_layout(
        title="Body Alignment",
        xaxis_title="X (ft, toward basket)",
        yaxis_title="Y (ft, lateral)",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5)
    )
    return fig

def create_foot_alignment_visual(frame_data, hoop_x=0, hoop_y=0):
    """
    Create a foot alignment visualization with each foot’s direction towards the basket at (0, 0).
    
    Parameters:
        frame_data (Series): Pose keypoints at release frame in feet (remapped coordinates).
        hoop_x, hoop_y (float): Basket position (default 0, 0 in remapped system).
    
    Returns:
        fig (plotly.graph_objects.Figure): Foot alignment visualization.
    """
    import numpy as np
    import plotly.graph_objects as go
    from math import atan2, degrees, sqrt

    fig = go.Figure()
    hoop_pos = (hoop_x, hoop_y)
    x_vals, y_vals = [], []

    for side, color in [('L', '#636EFA'), ('R', '#EF553B')]:
        heel_x = f'{side}HEEL_X'
        toe_x = f'{side}BIGTOE_X'
        heel_y = f'{side}HEEL_Y'
        toe_y = f'{side}BIGTOE_Y'
        
        if all(col in frame_data for col in [heel_x, toe_x, heel_y, toe_y]):
            heel = (frame_data[heel_x], frame_data[heel_y])
            toe = (frame_data[toe_x], frame_data[toe_y])
            
            # Plot foot
            fig.add_trace(go.Scatter(
                x=[heel[0], toe[0]],
                y=[heel[1], toe[1]],
                mode='lines+markers',
                line=dict(width=10, color=color),
                marker=dict(size=15),
                name=f"{'Left' if side=='L' else 'Right'} Foot"
            ))
            x_vals.extend([heel[0], toe[0]])
            y_vals.extend([heel[1], toe[1]])

            # Midpoint and direction
            midpoint = ((heel[0] + toe[0]) / 2, (heel[1] + toe[1]) / 2)
            foot_vec = (toe[0] - heel[0], toe[1] - heel[1])
            foot_norm = sqrt(foot_vec[0]**2 + foot_vec[1]**2)
            if foot_norm == 0:
                continue
            foot_unit = (foot_vec[0] / foot_norm, foot_vec[1] / foot_norm)

            # Perpendicular towards basket
            perp1 = (foot_unit[1], -foot_unit[0])
            perp2 = (-foot_unit[1], foot_unit[0])
            basket_vec = (hoop_pos[0] - midpoint[0], hoop_pos[1] - midpoint[1])
            basket_norm = sqrt(basket_vec[0]**2 + basket_vec[1]**2)
            basket_unit = (1, 0) if basket_norm == 0 else (basket_vec[0] / basket_norm, basket_vec[1] / basket_norm)
            
            dot1 = perp1[0] * basket_unit[0] + perp1[1] * basket_unit[1]
            dot2 = perp2[0] * basket_unit[0] + perp2[1] * basket_unit[1]
            perp = perp1 if dot1 > dot2 else perp2

            # Angle to basket
            angle = degrees(np.arccos(np.clip(np.dot(foot_unit, basket_unit), -1.0, 1.0)))
            fig.add_annotation(
                x=midpoint[0], y=midpoint[1] - 0.2,
                text=f"{angle:.1f}°",
                showarrow=False,
                font=dict(size=16, color=color)
            )

    # Add basket marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=10, color="red", symbol="x"),
        name="Basket"
    ))

    # Adjust axes
    if x_vals and y_vals:
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_margin = (x_max - x_min) * 0.5 or 5
        y_margin = (y_max - y_min) * 0.5 or 5
        fig.update_xaxes(range=[x_min - x_margin, x_max + x_margin])
        fig.update_yaxes(range=[y_min - y_margin, y_max + y_margin])

    fig.update_layout(
        title="Foot Alignment (Towards Basket)",
        xaxis_title="Court Position (ft)",
        yaxis_title="Lateral Position (ft)",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5)
    )
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