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
        'Apex Height': {'value': metrics.get('apex_height', 0), 'min': 0, 'max': 10},       # feet
        'Release Curvature': {'value': metrics.get('release_curvature', 0), 'min': 0, 'max': 0.5},  # 1/ft
        'Lateral Deviation': {'value': metrics.get('lateral_deviation', 0), 'min': -0.5, 'max': 0.5}  # feet
    }

    # KPI Grid (two rows of four columns)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        animated_flip_kpi_card("Release Height",
                               kpis['Release Height']['value'],
                               "ft",
                               player_average=player_averages['Release Height'] if player_averages is not None else None,
                               min_value=kpis['Release Height']['min'],
                               max_value=kpis['Release Height']['max'])
    with col2:
        animated_flip_kpi_card("Shot Distance",
                               kpis['Shot Distance']['value'],
                               "ft",
                               player_average=player_averages['Shot Distance'] if player_averages is not None else None,
                               min_value=kpis['Shot Distance']['min'],
                               max_value=kpis['Shot Distance']['max'])
    with col3:
        classification = metrics.get("release_class", {})
        extra_info = f"<p>Class: {classification.get('classification', 'Unknown')}<br>Optimal: {classification.get('optimal_range', 'N/A')}</p>"
        animated_flip_kpi_card("Release Angle",
                               kpis['Release Angle']['value'],
                               "°",
                               player_average=player_averages['Release Angle'] if player_averages is not None else None,
                               min_value=kpis['Release Angle']['min'],
                               max_value=kpis['Release Angle']['max'],
                               extra_html=extra_info)
    with col4:
        animated_flip_kpi_card("Release Velocity",
                               kpis['Release Velocity']['value'],
                               "ft/s",
                               player_average=player_averages['Release Velocity'] if player_averages is not None else None,
                               min_value=kpis['Release Velocity']['min'],
                               max_value=kpis['Release Velocity']['max'])

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        animated_flip_kpi_card("Release Time",
                               kpis['Release Time']['value'],
                               "s",
                               player_average=player_averages['Release Time'] if player_averages is not None else None,
                               min_value=kpis['Release Time']['min'],
                               max_value=kpis['Release Time']['max'])
    with col6:
        animated_flip_kpi_card("Apex Height",
                               kpis['Apex Height']['value'],
                               "ft",
                               player_average=player_averages['Apex Height'] if player_averages is not None else None,
                               min_value=kpis['Apex Height']['min'],
                               max_value=kpis['Apex Height']['max'])
    with col7:
        animated_flip_kpi_card("Release Curvature",
                               kpis['Release Curvature']['value'],
                               "1/ft",
                               player_average=player_averages['Release Curvature'] if player_averages is not None else None,
                               min_value=kpis['Release Curvature']['min'],
                               max_value=kpis['Release Curvature']['max'])
    with col8:
        animated_flip_kpi_card("Lateral Deviation",
                               kpis['Lateral Deviation']['value'],
                               "ft",
                               player_average=player_averages['Lateral Deviation'] if player_averages is not None else None,
                               min_value=kpis['Lateral Deviation']['min'],
                               max_value=kpis['Lateral Deviation']['max'])

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
    st.header("Biomechanics Analysis")
    release_idx = metrics.get('release_idx', 0)

    # New joint flexion/extension analysis
    st.subheader("Joint Flexion/Extension")
    fig, kpis = plot_joint_flexion_analysis(df_pose, df_ball, metrics)
    st.plotly_chart(fig, use_container_width=True)  # Use the single figure directly

    # Display KPIs
    st.subheader("Joint Flexion/Extension KPIs")
    col1, col2, col3 = st.columns(3)
    for i, joint in enumerate(kpis.keys()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.metric(f"{joint.capitalize()} Max Flexion", f"{kpis[joint]['max_flexion']:.1f}°")
            st.metric(f"{joint.capitalize()} Min Flexion", f"{kpis[joint]['min_flexion']:.1f}°")
            st.metric(f"{joint.capitalize()} at Lift", f"{kpis[joint]['at_lift']:.1f}°")
            st.metric(f"{joint.capitalize()} at Set", f"{kpis[joint]['at_set']:.1f}°")
            st.metric(f"{joint.capitalize()} at Release", f"{kpis[joint]['at_release']:.1f}°")
            st.metric(f"{joint.capitalize()} Range", f"{kpis[joint]['range']:.1f}°")
            st.metric(f"{joint.capitalize()} Max Rate", f"{kpis[joint]['rate_change']:.1f}°/s")
    
    # Get hoop position and flip from metrics
    hoop_x = metrics.get('hoop_x', 501.0)  # Default to 501 if not in metrics
    hoop_y = metrics.get('hoop_y', 0.0)
    flip = metrics.get('flip', False)

    # Get alignment metrics with dynamic hoop position
    alignment_metrics = calculate_body_alignment(df_pose, release_idx, hoop_x, hoop_y)
    
    # Create two columns for visuals
    col1, col2 = st.columns(2)
    
    with col1:
        pose_row = df_pose.iloc[release_idx]
        fig = create_body_alignment_visual(pose_row, hoop_x, hoop_y)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        try:
            frame_data = df_pose.iloc[release_idx]
            fig_feet = create_foot_alignment_visual(frame_data, metrics['shot_distance'], flip=flip, hoop_x=hoop_x, hoop_y=hoop_y)
            st.plotly_chart(fig_feet, use_container_width=True)
        except KeyError as e:
            st.error(f"Missing foot data: {e}")

def show_spin_analysis_page(df_spin):
    if not df_spin.empty:
        st.header("Spin Analysis")
        plot_spin_analysis(df_spin)
        plot_spin_bullseye(df_spin)  # Add bulls-eye visualization

if __name__ == "__main__":
    main()