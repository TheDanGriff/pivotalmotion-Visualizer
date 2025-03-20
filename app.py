import streamlit as st
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

def main():
    st.set_page_config(page_title="Pivotal Motion Visualizer", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Pivotal Motion Visualizer</h1>", unsafe_allow_html=True)

    if not st.session_state.get('authenticated', False):
        with st.form("login_form"):
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            handle_login(cognito_client, get_username_by_email, email, password)
        return
    else:
        st.sidebar.header("User Information")
        st.sidebar.write(f"**Username:** {st.session_state['username']}")
        st.sidebar.write(f"**Email:** {st.session_state['user_email']}")
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.experimental_rerun()

    user_email = st.session_state['user_email']

    pose_spin_jobs = fetch_user_completed_jobs(user_email)
    data_file_jobs = fetch_user_completed_data_file_jobs(user_email)
    jobs = pose_spin_jobs + data_file_jobs
    if not jobs:
        st.info("No completed jobs found for this user.")
        return

    teams = sorted({humanize_label(job.get("Team", "N/A")) for job in jobs if humanize_label(job.get("Team", "N/A")) != "N/A"})
    player_names = sorted({humanize_label(job.get("PlayerName", "Unknown")) for job in jobs})
    shooting_types = sorted({humanize_label(job.get("ShootingType", "N/A")) for job in jobs if humanize_label(job.get("ShootingType", "N/A")) != "N/A"})
    dates = sorted({pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')
                    for job in jobs if job.get("UploadTimestamp")})

    with st.sidebar:
        st.header("Filters")
        team_filter = st.selectbox("Team", ["All"] + teams)
        player_filter = st.selectbox("Player Name", ["All"] + player_names)
        shooting_filter = st.selectbox("Shot Type", ["All"] + shooting_types)
        date_filter = st.selectbox("Upload Date", ["All"] + dates)

    filtered_jobs = jobs
    if team_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get("Team", "N/A")) == team_filter]
    if player_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get("PlayerName", "Unknown")) == player_filter]
    if shooting_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get("ShootingType", "N/A")) == shooting_filter]
    if date_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if pd.to_datetime(int(j["UploadTimestamp"]), unit='s').strftime('%Y-%m-%d') == date_filter]

    job_options_dict = {
        (f"{job['PlayerName']} - {format_source_type(job['Team'])} - {format_source_type(job['Source'])} - "
         f"{pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d %H:%M')}"): job
        for job in filtered_jobs
    }

    selected_job_str = st.selectbox("Select a job to view details", options=list(job_options_dict.keys()))
    selected_job = job_options_dict.get(selected_job_str)
    if not selected_job:
        st.error("Selected job not found. Please try again.")
        return

    selected_job_id = selected_job['JobID']
    shot_type = selected_job.get("ShootingType", "Unknown")

    # Get player averages based on shot type
    player_averages = get_player_kpi_averages(selected_job['PlayerName'], shot_type)

    # After retrieving your segments
    if selected_job['Source'].lower() == 'pose_video':
        segments = list_segments(s3_client, BUCKET_NAME, user_email, selected_job_id)
    elif selected_job['Source'].lower() == 'data_file':
        segments = list_data_file_job_segments(BUCKET_NAME, user_email, selected_job_id)
    elif selected_job['Source'].lower() == 'spin_video':
        segments = ["spin_axis"]
    else:
        st.error(f"Unsupported source type: {selected_job['Source']}")
        return

    if not segments:
        st.error("No segments found for this job. Please ensure the data files are properly uploaded.")
        return

    segment_labels = {
        seg_id: get_segment_label(s3_client, BUCKET_NAME, user_email, selected_job_id, seg_id, selected_job['Source'])
        for seg_id in segments
    }

    selected_segment = st.selectbox(
        "Select Segment",
        options=list(segment_labels.keys()),
        format_func=lambda seg_id: segment_labels[seg_id]
    )

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

    metrics = {
        'shot_distance': 0.0,
        'release_height': 0.0,
        'release_angle': 0.0,
        'release_velocity': 0.0,
        'release_time': 0.0,
        'apex_height': 0.0,
        'release_curvature': 0.0,
        'lateral_deviation': 0.0
    }
    if not df_ball.empty:
        try:
            logger.debug(f"Columns in df_ball: {df_ball.columns.tolist()}")
            logger.debug(f"Columns in df_pose: {df_pose.columns.tolist()}")
            metrics, df_pose, df_ball = calculate_shot_metrics(df_pose, df_ball)
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")

    show_brand_header(selected_job.get('PlayerName', ''), selected_job.get('Team', ''))
    tab1, tab2, tab3 = st.tabs(["Overview", "Biomechanics", "Spin Analysis"])

    with tab1:
        show_overview_page(df_pose, df_ball, df_spin, metrics, selected_job['PlayerName'], shot_type)
    with tab2:
        show_biomechanics_page(df_pose, df_ball, df_spin, metrics)
    with tab3:
        show_spin_analysis_page(df_spin)

def show_overview_page(df_pose, df_ball, df_spin, metrics, player_name, shot_type):
    st.header("Basketball Shot Analysis")
    benchmarks = get_kpi_benchmarks()

    # Get player averages or use None if no data exists
    player_averages = get_player_kpi_averages(player_name, shot_type)

    # Define KPIs with ranges for metallic color
    kpis = {
        'Release Height': {'value': metrics.get('release_height', 0), 'min': 0, 'max': 12},  # feet
        'Shot Distance': {'value': metrics.get('shot_distance', 0), 'min': 0, 'max': 50},    # feet
        'Release Angle': {'value': metrics.get('release_angle', 0), 'min': 0, 'max': 90},    # degrees
        'Release Velocity': {'value': metrics.get('release_velocity', 0), 'min': 0, 'max': 30},  # feet/s
        'Release Time': {'value': metrics.get('release_time', 0), 'min': 0, 'max': 1},      # seconds
        'Apex Height': {'value': metrics.get('apex_height', 0), 'min': 0, 'max': 20},       # feet (adjusted max for realism)
        'Release Curvature (Side)': {'value': metrics.get('release_curvature_side', 0), 'min': 0, 'max': 0.5},  # 1/ft
        'Release Curvature (Rear)': {'value': metrics.get('release_curvature_rear', 0), 'min': 0, 'max': 0.5},  # 1/ft
        'Lateral Deviation': {'value': metrics.get('lateral_deviation', 0), 'min': -0.5, 'max': 0.5}  # feet
    }

    # KPI Grid (two rows of four columns, plus a third row for the extra KPI)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        animated_flip_kpi_card(
            "Release Height",
            kpis['Release Height']['value'],
            "ft",
            player_average=player_averages.get('Release Height') if player_averages else None,
            min_value=kpis['Release Height']['min'],
            max_value=kpis['Release Height']['max'],
            description="Good range: 7-10 ft. Height at release impacts arc and entry angle.",
            calculation_info="Ball's Z-coordinate at release frame, converted from inches to feet."
        )
    with col2:
        animated_flip_kpi_card(
            "Shot Distance",
            kpis['Shot Distance']['value'],
            "ft",
            player_average=player_averages.get('Shot Distance') if player_averages else None,
            min_value=kpis['Shot Distance']['min'],
            max_value=kpis['Shot Distance']['max'],
            description="Ranges: <15 ft (Free Throw), 15-22 ft (Mid Range), >22 ft (3-Point).",
            calculation_info="Horizontal distance from release point to hoop, adjusted for court side."
        )
    with col3:
        classification = metrics.get("release_class", {})
        class_info = f"Class: {classification.get('classification', 'Unknown')}, Optimal: {classification.get('optimal_range', 'N/A')}"
        animated_flip_kpi_card(
            "Release Angle",
            kpis['Release Angle']['value'],
            "°",
            player_average=player_averages.get('Release Angle') if player_averages else None,
            min_value=kpis['Release Angle']['min'],
            max_value=kpis['Release Angle']['max'],
            description=f"Good range: 43-47°. {class_info}. Optimal varies with distance.",
            calculation_info="Angle of ball trajectory post-release using arctan(dz/dxy) over 3 frames."
        )
    with col4:
        animated_flip_kpi_card(
            "Release Velocity",
            kpis['Release Velocity']['value'],
            "ft/s",
            player_average=player_averages.get('Release Velocity') if player_averages else None,
            min_value=kpis['Release Velocity']['min'],
            max_value=kpis['Release Velocity']['max'],
            description="Good range: 20-25 ft/s. Higher speeds suit longer shots.",
            calculation_info="Magnitude of velocity vector (vx, vy, vz) at release, converted to ft/s."
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        animated_flip_kpi_card(
            "Release Time",
            kpis['Release Time']['value'],
            "s",
            player_average=player_averages.get('Release Time') if player_averages else None,
            min_value=kpis['Release Time']['min'],
            max_value=kpis['Release Time']['max'],
            description="Good range: 0.3-0.6 s. Quicker releases reduce defender impact.",
            calculation_info="Time from lift index to release index, divided by FPS (default 60)."
        )
    with col6:
        animated_flip_kpi_card(
            "Apex Height",
            kpis['Apex Height']['value'],
            "ft",
            player_average=player_averages.get('Apex Height') if player_averages else None,
            min_value=kpis['Apex Height']['min'],
            max_value=kpis['Apex Height']['max'],
            description="Good range: 12-16 ft. Higher apex aids shot clearance.",
            calculation_info="Maximum Z-coordinate of ball trajectory, converted to feet."
        )
    with col7:
        animated_flip_kpi_card(
            "Release Curvature (Side)",
            kpis['Release Curvature (Side)']['value'],
            "1/ft",
            player_average=player_averages.get('Release Curvature (Side)') if player_averages else None,
            min_value=kpis['Release Curvature (Side)']['min'],
            max_value=kpis['Release Curvature (Side)']['max'],
            description="Good range: 0.05-0.15 1/ft. Moderate curvature in the side view optimizes the shot arc.",
            calculation_info="Terminal curvature from Bezier fit in the XZ plane (side view)."
        )
    with col8:
        animated_flip_kpi_card(
            "Lateral Deviation",
            kpis['Lateral Deviation']['value'],
            "ft",
            player_average=player_averages.get('Lateral Deviation') if player_averages else None,
            min_value=kpis['Lateral Deviation']['min'],
            max_value=kpis['Lateral Deviation']['max'],
            description="Good range: -0.1 to 0.1 ft. Closer to 0 indicates better aim.",
            calculation_info="Perpendicular distance from shot line to ball at hoop height, in feet."
        )

    # Third row for the additional KPI
    col9 = st.columns(1)[0]  # Single column
    with col9:
        animated_flip_kpi_card(
            "Release Curvature (Rear)",
            kpis['Release Curvature (Rear)']['value'],
            "1/ft",
            player_average=player_averages.get('Release Curvature (Rear)') if player_averages else None,
            min_value=kpis['Release Curvature (Rear)']['min'],
            max_value=kpis['Release Curvature (Rear)']['max'],
            description="Good range: 0.05-0.15 1/ft. Moderate curvature in the rear view ensures straight trajectory.",
            calculation_info="Terminal curvature from Bezier fit in the YZ plane (rear view)."
        )

    # Shot Location Visualization (Fixed Size, Proper Scale)
    if not df_ball.empty:
        st.subheader("Shot Location")
        shot_location_fig = plot_shot_location(df_ball, metrics)
        st.plotly_chart(shot_location_fig)  # No use_container_width=True to respect fixed size

    # Existing Visualizations
    fig = plot_curvature_analysis(df_ball, metrics, weighting_exponent=3, num_interp=300, curvature_scale=2.3)
    st.plotly_chart(fig, use_container_width=True)

    if not df_ball.empty:
        try:
            fig2 = plot_shot_analysis(df_ball, metrics)
            st.plotly_chart(fig2, use_container_width=True, key="overview_shot_analysis")
            st.subheader("3D Ball Path")
            fig3 = plot_3d_ball_path(df_ball)
            st.plotly_chart(fig3, use_container_width=True, key="overview_3d_ball_path")
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

def show_biomechanics_page(df_pose, df_ball, df_spin, metrics):
    import streamlit as st
    import pandas as pd
    from streamlit.components.v1 import components

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
        'COM Acceleration': {'min': -50, 'max': 50}  # Adjusted range for acceleration
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

    col5, col6, col7, _ = st.columns(4)  # Reduced to 3 columns since we have 7 KPIs
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
        plot_spin_bullseye(df_spin)  # Add bulls-eye visualization

if __name__ == "__main__":
    main()