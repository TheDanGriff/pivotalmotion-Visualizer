import streamlit as st
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

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
    
    # Load and encode logo
    logo_path = os.path.join("images", "whiteoutline.jpeg")
    try:
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode("utf-8")
        logo_src = f"data:image/jpeg;base64,{logo_data}"
    except FileNotFoundError:
        logo_src = None
        st.warning(f"Logo not found at {logo_path}")

    # Enhanced CSS with creative dropdown styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

        /* Main app background remains white */
        .stApp {
            padding: 10px;
            background: #FFFFFF;
            color: #333333 !important; /* Dark grey text on white background */
        }

        /* Sidebar styling with dark grey-to-blue gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2E3E4F 0%, #3A506B 100%);
            padding: 20px;
            border-right: 2px solid #2E3E4F;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
            border-radius: 0 10px 10px 0;
        }

        /* Sidebar box for Filters header */
        .sidebar-box {
            background: linear-gradient(135deg, #2E3E4F 0%, #3A506B 100%);
            border: 2px solid #FFFFFF;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .sidebar-box h2 {
            color: #FFFFFF !important; /* White for Filters header */
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 22px !important;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            text-align: center;
        }

        /* User info text */
        [data-testid="stSidebar"] .css-17eq0hr:not(:nth-child(3)) {
            color: #FFFFFF !important; /* White for username */
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 14px;
            border-radius: 5px;
            margin: 5px 0;
            transition: background 0.3s ease;
        }
        [data-testid="stSidebar"] .css-17eq0hr:nth-child(3) {
            color: #ff4500 !important; /* Orange for email */
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 14px;
            border-radius: 5px;
            margin: 5px 0;
            transition: background 0.3s ease;
        }
        [data-testid="stSidebar"] .css-17eq0hr:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        /* Creative dropdown styling */
        [data-testid="stSidebar"] .css-1v3fvcr {
            background-color: #FFFFFF; /* White background */
            color: #2E3E4F !important; /* Grey text for options */
            border: 1px solid #3A506B; /* Dark grey border */
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            margin: 5px 0;
        }
        [data-testid="stSidebar"] .css-1v3fvcr:hover {
            background-color: rgba(255, 255, 255, 0.9); /* Slight hover effect */
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }
        [data-testid="stSidebar"] .css-1v3fvcr:focus {
            border-color: #FF4500; /* Orange focus border */
            box-shadow: 0 0 5px rgba(255, 69, 0, 0.4);
        }

        /* Dropdown titles */
        [data-testid="stSidebar"] .css-145kmo2 {
            color: #FFFFFF !important; /* White for dropdown titles */
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 18px !important;
            margin-bottom: 5px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar button styling */
        [data-testid="stSidebar"] .stButton > button {
            color: #2E3E4F !important; /* Grey text for button */
            background: linear-gradient(135deg, #FFFFFF 0%, #E0E0E0 100%);
            border: 1px solid #FFFFFF;
            border-radius: 5px;
            padding: 8px 16px;
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            transition: background 0.3s ease, transform 0.3s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #E0E0E0 0%, #FFFFFF 100%);
            transform: scale(1.05);
        }

        /* ShotMetrics header with animations and spacing */
        .shotmetrics-header {
            background: linear-gradient(135deg, #2E3E4F 0%, #3A506B 100%);
            padding: 50px 20px;
            text-align: center;
            border-bottom: 6px solid #FFFFFF;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            position: relative;
            width: 100%;
            margin: 0;
            top: 0;
            left: 0;
            z-index: 1;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .shotmetrics-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 10%, transparent 60%);
            animation: rotateGlow 15s linear infinite;
            z-index: -1;
        }
        .shotmetrics-title {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif !important;
            font-size: 85px !important;
            font-weight: 400;
            color: #FFFFFF;
            text-shadow: 
                0 0 15px #FFFFFF,
                0 0 30px #2E3E4F,
                2px 2px 8px rgba(0, 0, 0, 0.6);
            margin: 0;
            animation: metallicShine 3s infinite alternate;
        }
        @keyframes rotateGlow {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes metallicShine {
            0% { text-shadow: 0 0 10px #FFFFFF, 2px 2px 8px rgba(46, 62, 79, 0.6), -2px -2px 8px rgba(46, 62, 79, 0.6); }
            100% { text-shadow: 0 0 20px #FFFFFF, 3px 3px 12px rgba(46, 62, 79, 0.8), -3px -3px 12px rgba(46, 62, 79, 0.8); }
        }
        .shotmetrics-header::after {
            content: '';
            position: absolute;
            bottom: 10px;
            left: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(to right, transparent, #FFFFFF, transparent);
        }

        /* Logo and title container */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px; /* Space between logo and title */
        }

        /* Divider Styling with increased spacing */
        .divider-space {
            margin: 60px 0; /* Increased spacing between header and team logo */
        }
        .subtle-divider {
            border: none;
            height: 4px;
            background: linear-gradient(to right, transparent, #2E3E4F, transparent);
            margin: 20px 0;
        }

        /* Content Styling */
        .team-name {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif;
            font-size: 36px !important;
            font-weight: bold;
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            background-image: linear-gradient(135deg, #2E3E4F, #3A506B);
            text-transform: uppercase;
            -webkit-text-stroke: 0.5px #2E3E4F;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3), -1px -1px 2px rgba(255, 255, 255, 0.5);
            margin: 0;
            line-height: 1.2;
            word-wrap: break-word;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .team-name:hover {
            transform: scale(1.05);
        }
        .player-name {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif;
            font-size: 36px !important;
            color: transparent;
            background-clip: text;
            -webkit-background-clip: text;
            background-image: linear-gradient(135deg, #2E3E4F, #3A506B);
            text-transform: uppercase;
            -webkit-text-stroke: 0.5px #2E3E4F;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3), -1px -1px 2px rgba(255, 255, 255, 0.5);
            margin: 20px 0;
            line-height: 1.2;
            word-wrap: break-word;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .player-name:hover {
            transform: scale(1.05);
        }
        .job-details {
            font-family: 'Oswald', 'Roboto', 'Arial', sans-serif;
            font-size: 18px !important;
            color: #333333 !important; /* Dark grey on white background */
            text-transform: uppercase;
            margin: 0;
            line-height: 1.2;
            word-wrap: break-word;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .job-details span.numeric {
            font-family: 'Arial', sans-serif;
            color: #2E3E4F;
        }
        .job-details:hover {
            transform: scale(1.05);
        }
        .logo-img {
            width: 150px;
            height: auto;
            margin-bottom: 30px;
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 50%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        .logo-img:hover {
            transform: scale(1.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Display the ShotMetrics header with logo and title spaced apart
    if logo_src:
        st.markdown(
            f"""
            <div class='shotmetrics-header'>
                <div class='header-container'>
                    <img src="{logo_src}" style='width: 150px; height: auto;'>
                    <h1 class='shotmetrics-title'>ShotMetrics</h1>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class='shotmetrics-header'>
                <h1 class='shotmetrics-title'>ShotMetrics</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not st.session_state.get('authenticated', False):
        with st.form("login_form"):
            if logo_src:
                st.markdown(
                    f"""
                    <div style='text-align: center; margin-bottom: 20px;'>
                        <img src="{logo_src}" style='width: 100px; height: auto;'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            handle_login(cognito_client, get_username_by_email, email, password)
        return
    else:
        # Add logo to the top of the sidebar
        if logo_src:
            st.sidebar.markdown(
                f"""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <img src="{logo_src}" style='width: 120px; height: auto;'>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Sidebar content
        st.sidebar.markdown(
            """
            <div class='sidebar-box'>
                <h2>User Information</h2>
                <p><b>Username:</b> {}</p>
                <p><b>Email:</b> {}</p>
            </div>
            """.format(st.session_state['username'], st.session_state['user_email']),
            unsafe_allow_html=True
        )
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
    sources = sorted({job.get("Source", "Unknown").title() for job in jobs})
    shot_types = ["3 Point", "Free Throw", "Mid-Range"]
    dates = sorted({pd.to_datetime(int(job['UploadTimestamp']), unit='s').strftime('%Y-%m-%d')
                    for job in jobs if job.get("UploadTimestamp")})

    with st.sidebar:
        st.markdown("<div class='sidebar-box'><h2>Filters</h2></div>", unsafe_allow_html=True)
        # Creative dropdowns
        team_filter = st.selectbox("Select Team", ["All"] + teams, key="team_filter")
        player_filter = st.selectbox("Select Player", ["All"] + player_names, key="player_filter")
        source_filter = st.selectbox("Select Source", ["All"] + sources, key="source_filter")
        shot_type_filter = st.selectbox("Select Shot Type", ["All"] + shot_types, key="shot_type_filter")
        date_filter = st.selectbox("Select Upload Date", ["All"] + dates, key="date_filter")

    filtered_jobs = jobs
    if team_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get("Team", "N/A")) == team_filter]
    if player_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if humanize_label(j.get("PlayerName", "Unknown")) == player_filter]
    if source_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if j.get("Source", "Unknown").title() == source_filter]
    if shot_type_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if get_shot_type(j.get("ShootingType", "Unknown")) == shot_type_filter]
    if date_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if pd.to_datetime(int(j["UploadTimestamp"]), unit='s').strftime('%Y-%m-%d') == date_filter]

    if not filtered_jobs:
        st.info("No jobs match the selected filters.")
        return

    selected_job = filtered_jobs[0]
    selected_job_id = selected_job['JobID']
    shot_type = get_shot_type(selected_job.get("ShootingType", "Unknown"))

    # Get segment details
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
        st.error("No segments found for this job.")
        return

    selected_segment = segments[0]
    segment_label = humanize_segment_label(get_segment_label(s3_client, BUCKET_NAME, user_email, selected_job_id, selected_segment, selected_job['Source']))

    # Parse segment label
    parts = segment_label.split(" | ")
    segment_number = parts[0] if len(parts) > 0 else "1/1"
    period = parts[1].replace("Period: ", "") if len(parts) > 1 and "Period: " in parts[1] else "N/A"
    clock = parts[2].replace("Clock: ", "") if len(parts) > 2 and "Clock: " in parts[2] else "N/A"
    shot_display = parts[3] if len(parts) > 3 else shot_type if shot_type in ["3 Point", "Free Throw", "Mid-Range"] else "Unknown"

    # Display player, team, logo, and job details
    player_name = humanize_label(selected_job.get('PlayerName', 'Unknown'))
    team_name_shorthand = humanize_label(selected_job.get('Team', 'N/A'))
    team_name = next((value for key, value in TEAMS.items() if key.lower() == team_name_shorthand.lower() or value.lower() == team_name_shorthand.lower()), team_name_shorthand)
    team_shorthand = next((key for key, value in TEAMS.items() if value.lower() == team_name.lower()), team_name.lower().replace(' ', '-'))
    logo_path = os.path.join("images", "teams", f"{team_shorthand}_logo.png")
    default_logo_path = os.path.join("images", "teams", "default.png")

    # Load and encode team logo as base64
    try:
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode("utf-8")
        team_logo_src = f"data:image/png;base64,{logo_data}"
    except FileNotFoundError:
        try:
            with open(default_logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode("utf-8")
            team_logo_src = f"data:image/png;base64,{logo_data}"
        except FileNotFoundError:
            team_logo_src = None
            st.warning(f"Team logo not found in {default_logo_path}")

    # Format job details with numbers in Arial
    job_details_html = ""
    for char in f"{segment_number} | Period: {period} | Clock: {clock} | {shot_display}":
        if char.isdigit() or char in ":/":
            job_details_html += f"<span class='numeric'>{char}</span>"
        else:
            job_details_html += char

    # Display content with the logo and increased spacing
    st.markdown("<div class='divider-space'></div>", unsafe_allow_html=True)  # Increased spacing before team logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if team_logo_src:
            st.markdown(
                f"""
                <img src="{team_logo_src}" class='logo-img'>
                <p class='team-name'>{team_name}</p>
                <p class='player-name'>{player_name}</p>
                <p class='job-details'>{job_details_html}</p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <p>No logo available</p>
                <p class='team-name'>{team_name}</p>
                <p class='player-name'>{player_name}</p>
                <p class='job-details'>{job_details_html}</p>
                """,
                unsafe_allow_html=True
            )
    st.markdown("<hr class='subtle-divider'>", unsafe_allow_html=True)

    # Load segment data
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

    tab1, tab2, tab3 = st.tabs(["Overview", "Biomechanics", "Spin Analysis"])

    with tab1:
        show_overview_page(df_pose, df_ball, df_spin, metrics, selected_job['PlayerName'], shot_type)
    with tab2:
        show_biomechanics_page(df_pose, df_ball, df_spin, metrics)
    with tab3:
        show_spin_analysis_page(df_spin)

def show_overview_page(df_pose, df_ball, df_spin, metrics, player_name, shot_type):
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)

    benchmarks = get_kpi_benchmarks()
    player_averages = get_player_kpi_averages(player_name, shot_type)
    logger.debug(f"Metrics before KPI rendering: {metrics}")

    kpis = {
        'Release Height': {'value': metrics.get('release_height', 0), 'min': 0, 'max': 12},
        'Shot Distance': {'value': metrics.get('shot_distance', 0), 'min': 0, 'max': 50},
        'Release Angle': {'value': metrics.get('release_angle', 0), 'min': 0, 'max': 90},
        'Release Velocity': {'value': metrics.get('release_velocity', 0), 'min': 0, 'max': 30},
        'Release Time': {'value': metrics.get('release_time', 0), 'min': 0, 'max': 1},
        'Apex Height': {'value': metrics.get('apex_height', 0), 'min': 0, 'max': 20},
        'Side Curvature': {'value': metrics.get('weighted_curvature_area_side', 0), 'min': 0, 'max': 0.5},
        'Rear Curvature': {'value': metrics.get('weighted_curvature_area_rear', 0), 'min': 0, 'max': 0.5},
        'Lateral Deviation': {'value': metrics.get('lateral_deviation', 0), 'min': -0.5, 'max': 0.5}
    }

    if not df_ball.empty:
        try:
            fig_shot = plot_shot_analysis(df_ball, metrics)
        except Exception as e:
            logger.error(f"Precompute shot analysis error: {str(e)}")
            fig_shot = None
    else:
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
            description="Good range: Varies by shot type",
            calculation_info="Distance from release point to hoop."
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
            description="Good range: 12-16 ft",
            calculation_info="Maximum height reached by the ball."
        )
    with col7:
        animated_flip_kpi_card(
            "Lateral Deviation",
            kpis['Lateral Deviation']['value'],
            "ft",
            player_average=player_averages.get('Lateral Deviation') if player_averages else None,
            min_value=kpis['Lateral Deviation']['min'],
            max_value=kpis['Lateral Deviation']['max'],
            description="Good range: -0.1 to 0.1 ft",
            calculation_info="Perpendicular distance from shot line to ball at hoop height."
        )
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)

    st.subheader("Curvature Analysis")
    col_curv_left, col_curv_right = st.columns(2)
    with col_curv_left:
        st.markdown("### Side View (XZ Plane)")
        animated_flip_kpi_card(
            "Side Curvature",
            kpis['Side Curvature']['value'],
            "1/ft",
            player_average=player_averages.get('Side Curvature') if player_averages else None,
            min_value=kpis['Side Curvature']['min'],
            max_value=kpis['Side Curvature']['max'],
            description="Good range: 0.05-0.15 1/ft",
            calculation_info="Cubic-weighted curvature area in XZ plane."
        )
    with col_curv_right:
        st.markdown("### Rear View (YZ Plane)")
        animated_flip_kpi_card(
            "Rear Curvature",
            kpis['Rear Curvature']['value'],
            "1/ft",
            player_average=player_averages.get('Rear Curvature') if player_averages else None,
            min_value=kpis['Rear Curvature']['min'],
            max_value=kpis['Rear Curvature']['max'],
            description="Good range: 0.05-0.15 1/ft",
            calculation_info="Cubic-weighted curvature area in YZ plane."
        )
    fig_curvature = plot_curvature_analysis(df_ball, metrics, weighting_exponent=3, num_interp=300, curvature_scale=2.3)
    st.plotly_chart(fig_curvature, use_container_width=True)
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)

    st.subheader("Ball Path Analysis")
    if not df_ball.empty:
        if fig_shot is not None:
            st.plotly_chart(fig_shot, use_container_width=True, key="overview_shot_analysis")
        else:
            st.error("Failed to precompute shot analysis visualization.")
    else:
        st.error("No ball data available for shot path visualization.")
    st.markdown("<hr style='border: 1px solid #e0e0e0; margin: 20px 0;'>", unsafe_allow_html=True)

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

if __name__ == "__main__":
    main()