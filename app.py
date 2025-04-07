import streamlit as st
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64
from datetime import datetime
from scipy import stats

# Import your project modules
from auth import handle_login
from aws_client import initialize_aws_clients
from config import BUCKET_NAME, FPS
from data_processing import (
    load_final_output,
    load_spin_axis_csv,
    separate_pose_and_ball_tracking,
    get_username_by_email,
    fetch_user_completed_jobs,
    fetch_user_completed_data_file_jobs,
    list_data_file_job_segments,
    load_data_file_final_output,
    calculate_shot_metrics,
    calculate_body_alignment,
    vector_angle,
    create_alignment_diagram,
    plot_curvature_analysis,
    get_segment_label,
    get_player_kpi_averages,
    get_shot_type,
    plot_shot_location,
    plot_joint_flexion_analysis
)
from visualization import (
    plot_single_shot,
    show_brand_header,
    display_kpi_grid,
    plot_kinematic_sequence,
    plot_spin_analysis,
    plot_velocity_profile,
    plot_trajectory_arc,
    plot_asymmetry_radar,
    plot_distance_over_height,
    plot_3d_ball_path,
    plot_velocity_vs_angle,
    plot_spin_bullseye,
    plot_shot_analysis,
    plot_release_angle_analysis,
    plot_foot_alignment,
    create_foot_alignment_visual,
    create_body_alignment_visual,
    plot_shot_location_in_inches
)
from kpi import (
    calculate_kpis_pose,
    calculate_kpis_spin,
    get_kpi_benchmarks,
    calculate_release_velocity,
    display_clickable_kpi_card,
    animated_flip_kpi_card
)
from utils import add_time_column, humanize_label, humanize_segment_label

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize AWS clients
cognito_client, dynamodb, s3_client = initialize_aws_clients()

# Teams Dictionary
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

# ---------------- Helper Functions ---------------- #
import streamlit as st
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64
from datetime import datetime
from scipy import stats

# Import your project modules
from auth import handle_login
from aws_client import initialize_aws_clients
from config import BUCKET_NAME, FPS
from data_processing import (
    load_final_output,
    load_spin_axis_csv,
    separate_pose_and_ball_tracking,
    get_username_by_email,
    fetch_user_completed_jobs,
    fetch_user_completed_data_file_jobs,
    list_data_file_job_segments,
    load_data_file_final_output,
    calculate_shot_metrics,
    calculate_body_alignment,
    vector_angle,
    create_alignment_diagram,
    plot_curvature_analysis,
    get_segment_label,
    get_player_kpi_averages,
    get_shot_type,
    plot_shot_location,
    plot_joint_flexion_analysis
)
from visualization import (
    plot_single_shot,
    show_brand_header,
    display_kpi_grid,
    plot_kinematic_sequence,
    plot_spin_analysis,
    plot_velocity_profile,
    plot_trajectory_arc,
    plot_asymmetry_radar,
    plot_distance_over_height,
    plot_3d_ball_path,
    plot_velocity_vs_angle,
    plot_spin_bullseye,
    plot_shot_analysis,
    plot_release_angle_analysis,
    plot_foot_alignment,
    create_foot_alignment_visual,
    create_body_alignment_visual,
    plot_shot_location_in_inches
)
from kpi import (
    calculate_kpis_pose,
    calculate_kpis_spin,
    get_kpi_benchmarks,
    calculate_release_velocity,
    display_clickable_kpi_card,
    animated_flip_kpi_card
)
from utils import add_time_column, humanize_label, humanize_segment_label

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize AWS clients
cognito_client, dynamodb, s3_client = initialize_aws_clients()

# ---------------- Helper Functions ---------------- #

def list_segments(s3_client, bucket_name, user_email, job_id):
    prefix = f"processed/{user_email}/{job_id}/"
    segments = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    if "segment_" in filename and ("_final_output.csv" in filename or "_final_output.xlsx" in filename):
                        if filename.endswith("_final_output.csv"):
                            seg_id = filename.replace("_final_output.csv", "")
                        elif filename.endswith("_final_output.xlsx"):
                            seg_id = filename.replace("_final_output.xlsx", "")
                        segments.add(seg_id)
        return sorted(segments)
    except Exception as e:
        st.error(f"Error listing segments: {e}")
        logger.error(f"Error in list_segments: {e}")
        return []

def format_source_type(source):
    if not source or not isinstance(source, str):
        return "Unknown Source"
    return source.replace('_', ' ').title()

def parse_segment_label(segment_label):
    parts = [part.strip() for part in segment_label.split("|")]
    date = parts[0] if parts else "Unknown"
    period = parts[1].replace("Period:", "").strip() if len(parts) > 1 and "Period:" in parts[1] else "N/A"
    clock = parts[2].replace("Clock:", "").strip() if len(parts) > 2 and "Clock:" in parts[2] else "N/A"
    shot_type = parts[3] if len(parts) > 3 else "Unknown"
    return date, period, clock, shot_type

def process_segment_for_table(job, segment, s3_client):
    try:
        if job['Source'].lower() in ['pose_video', 'data_file']:
            if job['Source'].lower() == 'data_file':
                df_segment = load_data_file_final_output(s3_client, BUCKET_NAME, job['user_email'], job['JobID'], segment)
            else:
                df_segment = load_final_output(s3_client, BUCKET_NAME, job['user_email'], job['JobID'], segment)
            if df_segment is not None and not df_segment.empty:
                df_pose, df_ball = separate_pose_and_ball_tracking(df_segment, job['Source'])
                metrics, _, _ = calculate_shot_metrics(df_pose, df_ball)
            else:
                metrics = {}
        else:
            metrics = {}

        segment_label = get_segment_label(s3_client, BUCKET_NAME, job['user_email'], job['JobID'], segment, job['Source'])
        date, period, clock, parsed_label_shot_type = parse_segment_label(segment_label)
        outcome = "Miss"
        if 'OUTCOME' in df_segment.columns:
            outcome = "Make" if df_segment['OUTCOME'].iloc[0] == 'Y' else "Miss"
        elif 'IS_MADE' in df_segment.columns:
            outcome = "Make" if df_segment['IS_MADE'].iloc[0] in [True, 'TRUE', 1] else "Miss"
        return {
            "Game Date": date,
            "Period": period,
            "Clock": clock,
            "Player": humanize_label(job.get("PlayerName", "Unknown")),
            "Team": humanize_label(job.get("Team", "N/A")),
            "Shot Type": get_shot_type(metrics.get('shot_distance', 0)),
            "Distance (ft)": metrics.get('shot_distance', 0),
            "Release Height": metrics.get('release_height', 0),
            "Release Angle": metrics.get('release_angle', 0),
            "Release Velocity": metrics.get('release_velocity', 0),
            "Apex Height": metrics.get('apex_height', 0),
            "Release Time": metrics.get('release_time', 0),
            "Side Curvature": metrics.get('release_curvature_side', 0),
            "Rear Curvature": metrics.get('release_curvature_rear', 0),
            "Lateral Deviation": metrics.get('lateral_deviation', 0),
            "Result": outcome,
            "JobID": job['JobID'],
            "Segment": segment
        }
    except Exception as e:
        logger.error(f"Error processing segment {segment} for job {job['JobID']}: {str(e)}")
        return None

def create_comparison_table(group_a_df, group_b_df):
    """Create a Plotly table comparing two groups of shots."""
    numeric_cols = [
        "Release Height", "Release Angle", "Release Velocity",
        "Apex Height", "Release Time", "Side Curvature", "Rear Curvature",
        "Lateral Deviation"
    ]
    metrics = [col for col in numeric_cols]

    def compute_stats(df):
        if df.empty:
            return ["N/A"] * len(metrics), ["N/A"] * len(metrics)
        means = [f"{df[col].mean():.2f}" if len(df) > 0 else "N/A" for col in numeric_cols]
        stds = [f"{df[col].std():.2f}" if len(df) > 1 else "N/A" for col in numeric_cols]
        return means, stds

    group_a_means, group_a_stds = compute_stats(group_a_df)
    group_b_means, group_b_stds = compute_stats(group_b_df)
    differences = []
    for a, b in zip(group_a_means, group_b_means):
        if a != "N/A" and b != "N/A":
            diff = float(a) - float(b)
            differences.append(f"{diff:.2f}")
        else:
            differences.append("N/A")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Group A Mean', 'Group A Std', 'Group B Mean', 'Group B Std', 'Difference'],
            fill_color='#2E3E4F',
            align='center',
            font=dict(color='white', size=16, family='Oswald'),
            height=50
        ),
        cells=dict(
            values=[metrics, group_a_means, group_a_stds, group_b_means, group_b_stds, differences],
            fill_color='#FFFFFF',
            align='center',
            font=dict(color='#333333', size=14, family='Oswald'),
            height=40
        )
    )])
    fig.update_layout(
        title={
            'text': f"Comparison of Group A ({len(group_a_df)} shots) and Group B ({len(group_b_df)} shots)",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#2E3E4F', family='Oswald')
        },
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ---------------- Detailed View Helper Functions ---------------- #

def show_overview_page(df_pose, df_ball, df_spin, metrics, player_name, shot_type):
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)
    benchmarks = get_kpi_benchmarks()
    player_averages = get_player_kpi_averages(player_name, shot_type)
    kpis = {
        'Release Height': {'value': metrics.get('release_height', 0), 'min': 0, 'max': 12},
        'Shot Distance': {'value': metrics.get('shot_distance', 0), 'min': 0, 'max': 50},
        'Release Angle': {'value': metrics.get('release_angle', 0), 'min': 0, 'max': 90},
        'Release Velocity': {'value': metrics.get('release_velocity', 0), 'min': 0, 'max': 30},
        'Release Time': {'value': metrics.get('release_time', 0), 'min': 0, 'max': 1},
        'Apex Height': {'value': metrics.get('apex_height', 0), 'min': 0, 'max': 20},
        'Side Curvature': {'value': metrics.get('release_curvature_side', 0), 'min': 0, 'max': 5.0},
        'Rear Curvature': {'value': metrics.get('release_curvature_rear', 0), 'min': 0, 'max': 5.0},
        'Lateral Deviation': {'value': metrics.get('lateral_deviation', 0), 'min': -0.5, 'max': 0.5}
    }
    try:
        fig_shot = plot_shot_analysis(df_ball, metrics)
    except Exception as e:
        logger.error(f"Precompute shot analysis error: {str(e)}")
        fig_shot = None

    st.subheader("Shot Location")
    if not df_ball.empty:
        shot_location_fig = plot_shot_location(df_ball, metrics)
        st.plotly_chart(shot_location_fig, use_container_width=True)
    else:
        st.error("No ball data available for shot location visualization.")
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        animated_flip_kpi_card(
            "Release Height",
            kpis['Release Height']['value'],
            "ft",
            player_average=player_averages.get('Release Height') if player_averages else None,
            min_value=kpis['Release Height']['min'],
            max_value=kpis['Release Height']['max'],
            description="Good range: 8-10 ft",
            calculation_info="Height of the ball at release point."
        )
    with col2:
        animated_flip_kpi_card(
            "Shot Distance",
            kpis['Shot Distance']['value'],
            "ft",
            player_average=player_averages.get('Shot Distance') if player_averages else None,
            min_value=kpis['Shot Distance']['min'],
            max_value=kpis['Shot Distance']['max'],
            description="Distance from release point to hoop.",
            calculation_info="Measured shot distance."
        )
    with col3:
        animated_flip_kpi_card(
            "Release Angle",
            kpis['Release Angle']['value'],
            "°",
            player_average=player_averages.get('Release Angle') if player_averages else None,
            min_value=kpis['Release Angle']['min'],
            max_value=kpis['Release Angle']['max'],
            description="Good range: 45-55°",
            calculation_info="Angle of ball trajectory at release."
        )
    with col4:
        animated_flip_kpi_card(
            "Release Velocity",
            kpis['Release Velocity']['value'],
            "ft/s",
            player_average=player_averages.get('Release Velocity') if player_averages else None,
            min_value=kpis['Release Velocity']['min'],
            max_value=kpis['Release Velocity']['max'],
            description="Good range: 20-25 ft/s",
            calculation_info="Speed of the ball at release."
        )
    col5, col6, col7 = st.columns([1, 1, 1])
    with col5:
        animated_flip_kpi_card(
            "Release Time",
            kpis['Release Time']['value'],
            "s",
            player_average=player_averages.get('Release Time') if player_averages else None,
            min_value=kpis['Release Time']['min'],
            max_value=kpis['Release Time']['max'],
            description="Good range: 0.4-0.6 s",
            calculation_info="Time from lift to release."
        )
    with col6:
        animated_flip_kpi_card(
            "Apex Height",
            kpis['Apex Height']['value'],
            "ft",
            player_average=player_averages.get('Apex Height') if player_averages else None,
            min_value=kpis['Apex Height']['min'],
            max_value=kpis['Apex Height']['max'],
            description="Maximum height reached by the ball.",
            calculation_info="Apex height metric."
        )
    with col7:
        animated_flip_kpi_card(
            "Lateral Deviation",
            kpis['Lateral Deviation']['value'],
            "ft",
            player_average=player_averages.get('Lateral Deviation') if player_averages else None,
            min_value=kpis['Lateral Deviation']['min'],
            max_value=kpis['Lateral Deviation']['max'],
            description="Deviation from the ideal shot line.",
            calculation_info="Measured lateral deviation."
        )
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.subheader("Curvature Analysis")
    col_curv_left, col_curv_right = st.columns(2)
    with col_curv_left:
        st.markdown("### Side View (XZ Plane)")
        animated_flip_kpi_card(
            "Side Release Curvature",
            metrics.get('release_curvature_side', 0.0),
            "1/ft",
            player_average=player_averages.get('Side Curvature') if player_averages else None,
            min_value=0,
            max_value=5.0,
            description="Curvature at release in the XZ plane.",
            calculation_info="Computed using Bezier curve at τ=1."
        )
    with col_curv_right:
        st.markdown("### Rear View (YZ Plane)")
        animated_flip_kpi_card(
            "Rear Release Curvature",
            metrics.get('release_curvature_rear', 0.0),
            "1/ft",
            player_average=player_averages.get('Rear Curvature') if player_averages else None,
            min_value=0,
            max_value=5.0,
            description="Curvature at release in the YZ plane.",
            calculation_info="Computed using Bezier curve at τ=1."
        )

    # Add the curvature analysis plots below the KPI cards
    try:
        fig_curvature = plot_curvature_analysis(
            df_ball,
            metrics,
            fps=60,              # Adjust based on your data’s frame rate
            weighting_exponent=3,
            num_interp=300,
            bezier_order=6,
            curvature_scale=2.3
        )
        st.plotly_chart(fig_curvature, use_container_width=True)
    except Exception as e:
        logger.error(f"Error generating curvature analysis plot: {str(e)}")
        st.error("Failed to generate curvature analysis plot.")
    
    # Add Shot Path Analysis section here
    st.subheader("Shot Path Analysis")
    if fig_shot is not None:
        st.plotly_chart(fig_shot, use_container_width=True)
    else:
        st.error("Failed to generate shot path analysis plot.")
    
    st.subheader("3D Ball Path")
    if not df_ball.empty:
        try:
            fig_3d = plot_3d_ball_path(df_ball)
            st.plotly_chart(fig_3d, use_container_width=True, key="overview_3d_ball_path")
        except Exception as e:
            st.error(f"3D Ball Path Visualization error: {str(e)}")
    else:
        st.error("No ball data available for 3D ball path visualization.")

def show_biomechanics_page(df_pose, df_ball, df_spin, metrics):
    st.header("Biomechanics Analysis")
    release_idx = metrics.get('release_idx', 0)
    player_name = st.session_state.get('username', 'Unknown')
    shot_type = st.session_state.get('shot_type', 'Unknown')
    player_averages = get_player_kpi_averages(player_name, shot_type) or {}
    fig, kpis = plot_joint_flexion_analysis(df_pose, df_ball, metrics)
    kpi_ranges = {
        'Kinematic Chain Score': {'min': 0, 'max': 100},
        'Stability Ratio': {'min': 0, 'max': 2},
        'Max Knee Flexion': {'min': 0, 'max': 180},
        'Max Elbow Flexion': {'min': 0, 'max': 180},
        'Asymmetry Score': {'min': 0, 'max': 180},
        'Shoulder Rotation': {'min': 0, 'max': 180},
        'COM Acceleration': {'min': -50, 'max': 50}
    }
    st.subheader("Biomechanical KPIs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        animated_flip_kpi_card(
            "Kinematic Chain Score",
            kpis.get('kinematic_chain_score', 0),
            "/100",
            player_average=player_averages.get('Kinematic Chain Score'),
            min_value=kpi_ranges['Kinematic Chain Score']['min'],
            max_value=kpi_ranges['Kinematic Chain Score']['max'],
            description="Good range: 70-100. Higher indicates better energy transfer.",
            calculation_info="Sum of timing (40%), range (30%), and release angle (30%) scores."
        )
    with col2:
        animated_flip_kpi_card(
            "Stability Ratio",
            kpis.get('stability_ratio', 0),
            "",
            player_average=player_averages.get('Stability Ratio'),
            min_value=kpi_ranges['Stability Ratio']['min'],
            max_value=kpi_ranges['Stability Ratio']['max'],
            description="Good range: 1.5-2. Wider feet relative to hips is better.",
            calculation_info="Average feet width divided by average hip width."
        )
    with col3:
        animated_flip_kpi_card(
            "Max Knee Flexion",
            kpis.get('right_knee', {}).get('max_flexion', 0),
            "°",
            player_average=player_averages.get('Max Knee Flexion'),
            min_value=kpi_ranges['Max Knee Flexion']['min'],
            max_value=kpi_ranges['Max Knee Flexion']['max'],
            description="Good range: 60-90°. Higher aids power generation.",
            calculation_info="Maximum right knee angle during the shot."
        )
    with col4:
        animated_flip_kpi_card(
            "Max Elbow Flexion",
            kpis.get('right_elbow', {}).get('max_flexion', 0),
            "°",
            player_average=player_averages.get('Max Elbow Flexion'),
            min_value=kpi_ranges['Max Elbow Flexion']['min'],
            max_value=kpi_ranges['Max Elbow Flexion']['max'],
            description="Good range: 90-120°. Higher aids shot control.",
            calculation_info="Maximum right elbow angle during the shot."
        )
    col5, col6, col7, _ = st.columns(4)
    with col5:
        animated_flip_kpi_card(
            "Asymmetry Score",
            kpis.get('asymmetry_score', 0),
            "°",
            player_average=player_averages.get('Asymmetry Score'),
            min_value=kpi_ranges['Asymmetry Score']['min'],
            max_value=kpi_ranges['Asymmetry Score']['max'],
            description="Good range: 0-20°. Lower indicates better symmetry.",
            calculation_info="Average difference between left and right knee/elbow angles at release."
        )
    with col6:
        animated_flip_kpi_card(
            "Shoulder Rotation",
            kpis.get('shoulder_rotation', 0),
            "°",
            player_average=player_averages.get('Shoulder Rotation'),
            min_value=kpi_ranges['Shoulder Rotation']['min'],
            max_value=kpi_ranges['Shoulder Rotation']['max'],
            description="Good range: 30-60°. Higher aids shot power.",
            calculation_info="Absolute difference in right shoulder angle from lift to release."
        )
    with col7:
        animated_flip_kpi_card(
            "COM Acceleration",
            kpis.get('com_acceleration', 0),
            "ft/s²",
            player_average=player_averages.get('COM Acceleration'),
            min_value=kpi_ranges['COM Acceleration']['min'],
            max_value=kpi_ranges['COM Acceleration']['max'],
            description="Positive values indicate upward/forward motion. Good range: 10-30 ft/s².",
            calculation_info="Mean acceleration of centroid from lift to release."
        )
    st.subheader("Joint Flexion/Extension")
    st.plotly_chart(fig, use_container_width=True)
    if not df_pose.empty and release_idx < len(df_pose):
        frame_data = df_pose.iloc[release_idx]
        st.subheader("Body Alignment Visuals")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Body Alignment (Feet, Hips, Shoulders)")
            body_fig = create_body_alignment_visual(frame_data, hoop_x=501.0, hoop_y=0.0)
            st.plotly_chart(body_fig, use_container_width=True)
        with col2:
            st.write("Foot Alignment")
            foot_fig = create_foot_alignment_visual(
                frame_data,
                shot_distance=metrics.get('shot_distance', 0) / 12,
                flip=False,
                hoop_x=41.75,
                hoop_y=0.0
            )
            st.plotly_chart(foot_fig, use_container_width=True)
    else:
        st.warning("Insufficient pose data or invalid release index for alignment visuals.")

def show_spin_analysis_page(df_spin):
    if not df_spin.empty:
        st.header("Spin Analysis")
        plot_spin_analysis(df_spin)
        plot_spin_bullseye(df_spin)

# ---------------- Main App Function ---------------- #

def main():
    st.set_page_config(page_title="ShotMetrics", layout="wide")
    
    # Load and encode logo image
    logo_path = os.path.join("images", "whiteoutline.png")
    try:
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode("utf-8")
        logo_src = f"data:image/png;base64,{logo_data}"
    except FileNotFoundError:
        logo_src = None
        st.warning(f"Logo not found at {logo_path}")

    # Insert combined CSS styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

        /* Main app background */
        .stApp {
            padding: 10px;
            background: #FFFFFF;
            color: #333333 !important;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2E3E4F 0%, #3A506B 100%) !important;
            padding: 20px !important;
            border-right: 2px solid #2E3E4F !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1) !important;
            border-radius: 0 10px 10px 0 !important;
        }
        /* Sidebar box styling */
        .sidebar-box {
            background: linear-gradient(135deg, #2E3E4F 0%, #3A506B 100%) !important;
            border: 2px solid #FFFFFF !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }
        .sidebar-box h2 {
            color: #FFFFFF !important;
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 22px !important;
            margin: 0 !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
            text-align: center !important;
        }
        .sidebar-box p {
            color: #FFFFFF !important;
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            margin: 5px 0 !important;
            text-align: center !important;
        }
        /* Remove conflicting user info styles */
        [data-testid="stSidebar"] .css-17eq0hr {
            display: none !important;
        }
        /* Creative dropdown styling */
        [data-testid="stSidebar"] .stSelectbox {
            margin: 10px 0 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div > div {
            background-color: #FFFFFF !important;
            color: #2E3E4F !important;
            border: 2px solid #3A506B !important;
            border-radius: 8px !important;
            padding: 8px !important;
            min-height: 40px !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div > div:hover {
            background-color: #F0F0F0 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div > div:focus {
            border-color: #FF4500 !important;
            box-shadow: 0 0 5px rgba(255,69,0,0.4) !important;
        }
        /* Dropdown titles */
        [data-testid="stSidebar"] .stSelectbox > label {
            color: #FFFFFF !important;
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
            margin-bottom: 5px !important;
        }
        /* Sidebar button styling */
        [data-testid="stSidebar"] .stButton > button {
            color: #2E3E4F !important;
            background: linear-gradient(135deg, #FFFFFF 0%, #E0E0E0 100%) !important;
            border: 1px solid #FFFFFF !important;
            border-radius: 5px !important;
            padding: 8px 16px !important;
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            transition: background 0.3s ease, transform 0.3s ease !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #E0E0E0 0%, #FFFFFF 100%) !important;
            transform: scale(1.05) !important;
        }
        /* ShotMetrics header styling */
        .shotmetrics-header {
            background: linear-gradient(135deg, #2E3E4F 0%, #3A506B 100%) !important;
            padding: 50px 20px !important;
            text-align: center !important;
            border-bottom: 6px solid #FFFFFF !important;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
            position: relative !important;
            width: 100% !important;
            margin: 0 !important;
            top: 0 !important;
            left: 0 !important;
            z-index: 1 !important;
            overflow: hidden !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .shotmetrics-header::before {
            content: '';
            position: absolute !important;
            top: -50% !important;
            left: -50% !important;
            width: 200% !important;
            height: 200% !important;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 10%, transparent 60%) !important;
            animation: rotateGlow 15s linear infinite !important;
            z-index: -1 !important;
        }
        .shotmetrics-title {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 85px !important;
            font-weight: 400 !important;
            color: #FFFFFF !important;
            text-shadow: 0 0 5px #FFFFFF, 0 0 10px #2E3E4F, 2px 2px 4px rgba(0,0,0,0.4) !important;
            margin: 0 !important;
            animation: metallicShine 3s infinite alternate !important;
        }
        @keyframes rotateGlow {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes metallicShine {
            0% { 
                text-shadow: 0 0 5px #FFFFFF, 2px 2px 4px rgba(46,62,79,0.4), -2px -2px 4px rgba(46,62,79,0.4);
            }
            100% { 
                text-shadow: 0 0 10px #FFFFFF, 3px 3px 6px rgba(46,62,79,0.6), -3px -3px 6px rgba(46,62,79,0.6);
            }
        }
        .shotmetrics-header::after {
            content: '';
            position: absolute !important;
            bottom: 10px !important;
            left: 0 !important;
            width: 100% !important;
            height: 2px !important;
            background: linear-gradient(to right, transparent, #FFFFFF, transparent) !important;
        }
        /* Logo and title container */
        .header-container {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 30px !important;
        }
        /* Divider styling */
        .divider-space {
            margin: 60px 0 !important;
        }
        .subtle-divider {
            border: none !important;
            height: 4px !important;
            background: linear-gradient(to right, transparent, #2E3E4F, transparent) !important;
            margin: 20px 0 !important;
        }
        /* Content styling */
        .team-name {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 36px !important;
            font-weight: bold !important;
            color: transparent !important;
            background-clip: text !important;
            -webkit-background-clip: text !important;
            background-image: linear-gradient(135deg, #2E3E4F, #3A506B) !important;
            text-transform: uppercase !important;
            -webkit-text-stroke: 0.5px #2E3E4F !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3), -1px -1px 2px rgba(255,255,255,0.5) !important;
            margin: 0 !important;
            line-height: 1.2 !important;
            word-wrap: break-word !important;
            text-align: center !important;
            transition: transform 0.3s ease !important;
        }
        .team-name:hover {
            transform: scale(1.05) !important;
        }
        .player-name {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 36px !important;
            color: transparent !important;
            background-clip: text !important;
            -webkit-background-clip: text !important;
            background-image: linear-gradient(135deg, #2E3E4F, #3A506B) !important;
            text-transform: uppercase !important;
            -webkit-text-stroke: 0.5px #2E3E4F !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3), -1px -1px 2px rgba(255,255,255,0.5) !important;
            margin: 20px 0 !important;
            line-height: 1.2 !important;
            word-wrap: break-word !important;
            text-align: center !important;
            transition: transform 0.3s ease !important;
        }
        .player-name:hover {
            transform: scale(1.05) !important;
        }
        .job-details {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            color: #333333 !important;
            text-transform: uppercase !important;
            margin: 0 !important;
            line-height: 1.2 !important;
            word-wrap: break-word !important;
            text-align: center !important;
            transition: transform 0.3s ease !important;
        }
        .job-details span.numeric {
            font-family: 'Oswald', sans-serif !important;
            color: #2E3E4F !important;
        }
        .job-details:hover {
            transform: scale(1.05) !important;
        }
        .logo-img {
            width: 150px !important;
            height: auto !important;
            margin-bottom: 30px !important;
            display: block !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 50% !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
            transition: transform 0.3s ease !important;
        }
        .logo-img:hover {
            transform: scale(1.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display header with logo and title
    if logo_src:
        st.markdown(f"""
            <div class='shotmetrics-header'>
                <div class='header-container'>
                    <img src="{logo_src}" style='width: 150px; height: auto;'>
                    <h1 class='shotmetrics-title'>ShotMetrics</h1>
                </div>
            </div>
            <style>
                .shotmetrics-header {{ margin-bottom: 40px; }}
                .header-container {{ display: flex; align-items: center; gap: 20px; justify-content: center; }}
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='shotmetrics-header'>
                <h1 class='shotmetrics-title'>ShotMetrics</h1>
            </div>
            <style>
                .shotmetrics-header { margin-bottom: 60px; }
            </style>
        """, unsafe_allow_html=True)

    # --- Authentication ---
    if not st.session_state.get('authenticated', False):
        with st.form("login_form"):
            if logo_src:
                st.markdown(f"""
                    <div style='text-align: center; margin-bottom: 30px; margin-top: 20px;'>
                        <img src="{logo_src}" style='width: 100px; height: auto;'>
                    </div>
                """, unsafe_allow_html=True)
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            handle_login(cognito_client, get_username_by_email, email, password)
        return
    else:
        if logo_src:
            st.sidebar.markdown(f"""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <img src="{logo_src}" style='width: 120px; height: auto;'>
                </div>
            """, unsafe_allow_html=True)
        st.sidebar.markdown("""
            <div class='sidebar-box'>
                <h2>User Information</h2>
                <p><b>Username:</b> {}</p>
                <p><b>Email:</b> {}</p>
            </div>
        """.format(st.session_state['username'], st.session_state['user_email']), unsafe_allow_html=True)
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.experimental_rerun()

    user_email = st.session_state['user_email']
    username = st.session_state['username']

    # --- Fetch Jobs & Apply Filters ---
    pose_spin_jobs = fetch_user_completed_jobs(user_email)
    data_file_jobs = fetch_user_completed_data_file_jobs(user_email)
    jobs = pose_spin_jobs + data_file_jobs
    for job in jobs:
        job['user_email'] = user_email

    if not jobs:
        st.info("No completed jobs found for this user.")
        return

    all_table_data = []
    with st.spinner("Loading initial shot data..."):
        for job in jobs:
            segments = list_segments(s3_client, BUCKET_NAME, user_email, job['JobID'])
            for segment in segments:
                segment_data = process_segment_for_table(job, segment, s3_client)
                if segment_data:
                    all_table_data.append(segment_data)

    all_df = pd.DataFrame(all_table_data)

    with st.sidebar:
        st.markdown("<div class='sidebar-box'><h2>Filters</h2></div>", unsafe_allow_html=True)
        
        teams = ["All"] + sorted({humanize_label(j.get("Team", "N/A")) for j in jobs if humanize_label(j.get("Team", "N/A")) != "N/A"})
        team_filter = st.selectbox("Select Team", teams, key="team_filter")
        team_filtered_jobs = [j for j in jobs if team_filter == "All" or humanize_label(j.get("Team")) == team_filter]
        
        player_names = ["All"] + sorted({humanize_label(j.get("PlayerName", "Unknown")) for j in team_filtered_jobs})
        player_filter = st.selectbox("Select Player", player_names, key="player_filter")
        player_filtered_jobs = [j for j in team_filtered_jobs if player_filter == "All" or humanize_label(j.get("PlayerName", "Unknown")) == player_filter]
        
        game_dates = set()
        for job in player_filtered_jobs:
            if job['Source'].lower() in ['pose_video', 'data_file']:
                segments = list_segments(s3_client, BUCKET_NAME, user_email, job['JobID'])
                for segment in segments:
                    segment_label = get_segment_label(s3_client, BUCKET_NAME, user_email, job['JobID'], segment, job['Source'])
                    date, _, _, _ = parse_segment_label(segment_label)
                    if date != "Unknown":
                        game_dates.add(date)
        date_filter = st.selectbox("Select Game Date", ["All"] + sorted(game_dates), key="date_filter")
        
        sources = ["All"] + sorted({j.get("Source", "Unknown").title() for j in player_filtered_jobs})
        source_filter = st.selectbox("Select Source", sources, key="source_filter")
        
        shot_types = ["All", "3 Point", "Free Throw", "Mid Range"]
        shot_type_filter = st.selectbox("Select Shot Type", shot_types, key="shot_type_filter")

    filtered_table_data = all_table_data
    if team_filter != "All":
        filtered_table_data = [d for d in filtered_table_data if d["Team"] == team_filter]
    if player_filter != "All":
        filtered_table_data = [d for d in filtered_table_data if d["Player"] == player_filter]
    if date_filter != "All":
        filtered_table_data = [d for d in filtered_table_data if d["Game Date"] == date_filter]
    if source_filter != "All":
        filtered_table_data = [d for d in filtered_table_data if format_source_type(d["Source"]) == source_filter]
    if shot_type_filter != "All":
        shot_type_filter = shot_type_filter.replace("Mid Range", "Mid-Range")
        filtered_table_data = [d for d in filtered_table_data if d["Shot Type"] == shot_type_filter]

    if not filtered_table_data:
        st.info("No jobs match the selected filters.")
        return

    df_table = pd.DataFrame(filtered_table_data)

    # --- Introduce Tabs ---
    tab1, tab2 = st.tabs(["Shot Analysis", "Compare Shots"])

    # --- Shot Analysis Tab ---
    with tab1:
        st.markdown("""
            <div style='margin: 20px 0;'>
                <h2 style='font-family: Oswald; color: #2E3E4F; text-align: center;'>
                    Select a Shot for Detailed Analysis
                </h2>
            </div>
        """, unsafe_allow_html=True)

        # Add Shot ID for dropdown
        df_table['Shot ID'] = df_table.apply(
            lambda row: f"{row['Player']} - {row['Game Date']} - {row['Period']} - {row['Clock']} - {row['Shot Type']}",
            axis=1
        )
        shot_labels = df_table['Shot ID'].tolist()

        if shot_labels:
            selected_shot_label = st.selectbox("Select a shot", shot_labels, key="shot_analysis_select")
            selected_row = df_table[df_table['Shot ID'] == selected_shot_label].iloc[0]

            # Load data for selected shot
            selected_job = next(j for j in jobs if j['JobID'] == selected_row['JobID'])
            selected_segment = selected_row['Segment']
            selected_job_id = selected_job['JobID']
            shot_type = selected_row['Shot Type']

            if selected_job['Source'].lower() in ['pose_video', 'data_file']:
                if selected_job['Source'].lower() == 'data_file':
                    df_segment = load_data_file_final_output(s3_client, BUCKET_NAME, user_email, selected_job_id, selected_segment)
                else:
                    df_segment = load_final_output(s3_client, BUCKET_NAME, user_email, selected_job_id, selected_segment)
            elif selected_job['Source'].lower() == 'spin_video':
                df_segment = load_spin_axis_csv(s3_client, BUCKET_NAME, user_email, selected_job_id)
            else:
                st.error(f"Unsupported source type: {selected_job['Source']}")
                return

            if df_segment is not None and not df_segment.empty:
                if selected_job['Source'].lower() in ['pose_video', 'data_file']:
                    df_pose, df_ball = separate_pose_and_ball_tracking(df_segment, selected_job['Source'])
                    df_spin = pd.DataFrame()
                elif selected_job['Source'].lower() == 'spin_video':
                    df_spin = df_segment
                    df_pose, df_ball = pd.DataFrame(), pd.DataFrame()
            else:
                st.error("No data loaded. Please check the file format and contents.")
                return

            # Recalculate full shot metrics
            metrics, df_pose, df_ball = calculate_shot_metrics(df_pose, df_ball)

            # Sub-tabs for detailed analysis
            subtab1, subtab2, subtab3 = st.tabs(["Overview", "Biomechanics Analysis", "Spin Analysis"])
            with subtab1:
                show_overview_page(df_pose, df_ball, df_spin, metrics, selected_job['PlayerName'], shot_type)
            with subtab2:
                show_biomechanics_page(df_pose, df_ball, df_spin, metrics)
            with subtab3:
                show_spin_analysis_page(df_spin)
        else:
            st.info("No shots available for analysis.")

    # --- Compare Shots Tab ---
    with tab2:
        st.markdown("""
            <div style='margin: 20px 0;'>
                <h2 style='font-family: Oswald; color: #2E3E4F; text-align: center;'>
                    Select Shots for Comparison
                </h2>
                <p style='font-family: Oswald; color: #333333; text-align: center;'>
                    Select shots for Group A and Group B to compare their aggregated statistics.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Add checkbox columns
        df_table['Group A'] = False
        df_table['Group B'] = False

        edited_df = st.data_editor(
            df_table,
            column_config={
                "Group A": st.column_config.CheckboxColumn("Group A"),
                "Group B": st.column_config.CheckboxColumn("Group B"),
                "Game Date": st.column_config.TextColumn("Game Date"),
                "Period": st.column_config.TextColumn("Period"),
                "Clock": st.column_config.TextColumn("Clock"),
                "Player": st.column_config.TextColumn("Player"),
                "Team": st.column_config.TextColumn("Team"),
                "Shot Type": st.column_config.TextColumn("Shot Type"),
                "Distance (ft)": st.column_config.NumberColumn("Distance (ft)", format="%.1f"),
                "Release Height": st.column_config.NumberColumn("Release Height", format="%.1f"),
                "Release Angle": st.column_config.NumberColumn("Release Angle", format="%.1f°"),
                "Release Velocity": st.column_config.NumberColumn("Release Velocity", format="%.1f ft/s"),
                "Apex Height": st.column_config.NumberColumn("Apex Height", format="%.1f ft"),
                "Release Time": st.column_config.NumberColumn("Release Time", format="%.2f s"),
                "Side Curvature": st.column_config.NumberColumn("Side Curvature", format="%.3f 1/ft"),
                "Rear Curvature": st.column_config.NumberColumn("Rear Curvature", format="%.3f 1/ft"),
                "Lateral Deviation": st.column_config.NumberColumn("Lateral Deviation", format="%.2f ft"),
                "Result": st.column_config.TextColumn("Result"),
                "JobID": None,
                "Segment": None,
                "Shot ID": None
            },
            hide_index=True,
            disabled=["Game Date", "Period", "Clock", "Player", "Team", "Shot Type",
                      "Distance (ft)", "Release Height", "Release Angle", "Release Velocity",
                      "Apex Height", "Release Time", "Side Curvature", "Rear Curvature",
                      "Lateral Deviation", "Result"],
            use_container_width=True,
            key="compare_table"
        )

        group_a_rows = edited_df[edited_df['Group A'] == True]
        group_b_rows = edited_df[edited_df['Group B'] == True]

        if not group_a_rows.empty and not group_b_rows.empty:
            st.markdown("""
                <div style='margin: 40px 0;'>
                    <h2 style='font-family: Oswald; color: #2E3E4F; text-align: center;'>
                        Comparison Results
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            fig = create_comparison_table(group_a_rows, group_b_rows)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one shot for each group to compare.")

if __name__ == "__main__":
    main()