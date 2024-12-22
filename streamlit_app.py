# visualizer_app.py

import streamlit as st
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# -------------------------------
# Configuration and AWS Setup
# -------------------------------

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Pivotal Motion Data Visualizer", layout="wide")

@st.cache_resource
def get_s3_client():
    """Initialize and return an S3 client using credentials from Streamlit secrets."""
    try:
        s3 = boto3.client(
            's3',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return s3
    except (NoCredentialsError, PartialCredentialsError):
        st.error("AWS credentials are not properly configured in Streamlit secrets.")
        st.stop()

@st.cache_resource
def get_dynamodb_resource():
    """Initialize and return a DynamoDB resource using credentials from Streamlit secrets."""
    try:
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return dynamodb
    except (NoCredentialsError, PartialCredentialsError):
        st.error("AWS credentials are not properly configured in Streamlit secrets.")
        st.stop()

# Initialize the S3 client
s3_client = get_s3_client()

# Initialize DynamoDB resource
dynamodb = get_dynamodb_resource()

# -------------------------------
# Helper Functions
# -------------------------------

def list_users(s3_client, bucket_name):
    """List unique users based on folder structure in S3."""
    prefix = 'Users/'  # Ensure correct case sensitivity
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        users = []
        for page in pages:
            prefixes = page.get('CommonPrefixes')
            if prefixes:
                users.extend([p['Prefix'].replace(prefix, '').strip('/') for p in prefixes])
        return users
    except Exception as e:
        st.error(f"Error listing users from S3: {e}")
        return []

def list_user_segments(s3_client, bucket_name, user):
    """List all segment directories for a given user."""
    prefix = f"Users/{user}/output/"
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
        segments = []
        for page in pages:
            prefixes = page.get('CommonPrefixes')
            if prefixes:
                segments.extend([p['Prefix'].replace(prefix, '').strip('/') for p in prefixes])
        return segments
    except Exception as e:
        st.error(f"Error listing segments for user {user}: {e}")
        return []

def load_vibe_output(s3_client, bucket_name, user, segment):
    """Load vibe_output.csv from a specific segment."""
    key = f"Users/{user}/output/{segment}/vibe_output.csv"
    try:
        csv_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        body = csv_obj['Body'].read().decode('utf-8')
        data = pd.read_csv(StringIO(body))
        
        # Normalize column names
        data = data.rename(columns=lambda x: x.strip().lower())
        
        # Ensure essential columns exist
        essential_cols = ['upload_time', 'player_name', 'shot_type', 'segment_id', 'frame', 'tracker_id', 'x', 'y']
        missing_cols = [col for col in essential_cols if col not in data.columns]
        if missing_cols:
            st.warning(f"Missing essential columns in {segment}/vibe_output.csv: {missing_cols}")
        
        return data
    except s3_client.exceptions.NoSuchKey:
        st.warning(f"'vibe_output.csv' not found for segment '{segment}' under user '{user}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {key}: {e}")
        return pd.DataFrame()

def get_metadata_from_dynamodb(dynamodb, job_id):
    """Retrieve metadata from DynamoDB using the job ID."""
    table_name = st.secrets.get("DYNAMODB_TABLE_NAME")
    if not table_name:
        st.error("DynamoDB table name not specified in Streamlit secrets.")
        st.stop()
    
    table = dynamodb.Table(table_name)
    try:
        response = table.get_item(Key={'job_id': job_id})
        if 'Item' in response:
            return response['Item']
        else:
            st.warning(f"No metadata found in DynamoDB for job_id: {job_id}")
            return {}
    except Exception as e:
        st.error(f"Error retrieving metadata from DynamoDB for job_id {job_id}: {e}")
        return {}

def get_readable_label(column_name):
    """Convert column names to human-readable labels."""
    mapping = {
        'left_elbow': 'Left Elbow Angle',
        'right_elbow': 'Right Elbow Angle',
        'left_shoulder_elev': 'Left Shoulder Elevation',
        'right_shoulder_elev': 'Right Shoulder Elevation',
        'left_shoulder_rot': 'Left Shoulder Rotation',
        'right_shoulder_rot': 'Right Shoulder Rotation',
        'left_forearm_pro': 'Left Forearm Pro',
        'right_forearm_pro': 'Right Forearm Pro',
        'left_wrist_dev': 'Left Wrist Deviation',
        'right_wrist_dev': 'Right Wrist Deviation',
        'left_wrist_flex': 'Left Wrist Flexion',
        'right_wrist_flex': 'Right Wrist Flexion',
        'left_hip_flex': 'Left Hip Flexion',
        'right_hip_flex': 'Right Hip Flexion',
        'left_knee': 'Left Knee Angle',
        'right_knee': 'Right Knee Angle',
        'left_ankle_inv': 'Left Ankle Inversion',
        'right_ankle_inv': 'Right Ankle Inversion',
        'left_ankle_flex': 'Left Ankle Flexion',
        'right_ankle_flex': 'Right Ankle Flexion',
        # Add more mappings as needed
    }
    return mapping.get(column_name, column_name.replace('_', ' ').title())

def display_metadata(metadata):
    """Display metadata in a clean and professional format."""
    # Load and display team logo
    team = metadata.get('team', 'na').lower()
    team_logo_path = f"static/images/team/{team}_logo.png"
    if not os.path.exists(team_logo_path):
        team_logo_path = "static/images/team/na_logo.png"  # Default logo
    
    # Create two columns: logo and metadata
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(team_logo_path, width=100)
    with col2:
        st.markdown("### 📝 Metadata")
        metadata_display = {
            "Player Name": metadata.get("player_name", "N/A"),
            "Team": metadata.get("team", "N/A"),
            "Shot Type": metadata.get("shot_type", "N/A"),
            "Shooting Motion": metadata.get("shooting_motion", "N/A"),
            "Upload Time": metadata.get("upload_time", "N/A"),
            "Segment ID": metadata.get("segment_id", "N/A")
        }
        for key, value in metadata_display.items():
            st.markdown(f"**{key}:** {value}")

def calculate_velocities(data, frame_rate=30):
    """Calculate velocities for joints based on positional data."""
    # Assuming data has columns like 'left_elbow_x', 'left_elbow_y', etc.
    # Create a list of joints to calculate velocities
    joints = ['left_elbow', 'right_elbow', 'left_hip', 'right_hip']
    velocity_df = pd.DataFrame()
    
    for joint in joints:
        x_col = f"{joint}_x"
        y_col = f"{joint}_y"
        if x_col in data.columns and y_col in data.columns:
            # Calculate velocity components
            vx = data[x_col].diff() * frame_rate  # Change per second
            vy = data[y_col].diff() * frame_rate
            # Calculate magnitude
            velocity = np.sqrt(vx**2 + vy**2)
            velocity_df[f"{joint}_velocity"] = velocity.fillna(0)
        else:
            st.warning(f"Position columns for {joint} not found.")
    
    return velocity_df

def detect_motion_phases(data, vibe_data, frame_rate=30):
    """Detect shooting motion phases and calculate related metrics."""
    # Example implementation:
    # Detect release point based on ball trajectory
    # Calculate velocities during different phases
    
    if vibe_data.empty or data.empty:
        st.warning("Insufficient data to define shooting phases.")
        return pd.DataFrame()
    
    # Assuming 'x' and 'y' in vibe_data represent ball coordinates
    # Calculate distance between ball and shooter (assuming shooter is at mean x, y)
    shooter_x = data['x'].mean()
    shooter_y = data['y'].mean()
    vibe_data['distance'] = np.sqrt((vibe_data['x'] - shooter_x)**2 + (vibe_data['y'] - shooter_y)**2)
    
    # Identify release point as the frame where distance starts increasing significantly
    vibe_data['distance_diff'] = vibe_data['distance'].diff().fillna(0)
    threshold = 0.1  # Define a threshold for significant increase
    release_point = vibe_data[vibe_data['distance_diff'] > threshold].index.min()
    
    if pd.isna(release_point):
        st.warning("Unable to determine release point.")
        return pd.DataFrame()
    
    # Define phases
    vibe_data['Phase'] = 'Loading Phase'
    vibe_data.loc[release_point:, 'Phase'] = 'Release Phase'
    
    # Calculate velocities for joints
    velocity_df = calculate_velocities(data)
    vibe_data = vibe_data.reset_index(drop=True)
    combined_df = pd.concat([vibe_data, velocity_df], axis=1)
    
    # Calculate metrics
    metrics = {}
    # Release Speed: Velocity of the ball at release point
    if release_point < len(combined_df):
        release_speed = combined_df.loc[release_point, 'distance_diff'] * frame_rate
        metrics['Release Speed'] = release_speed
    else:
        metrics['Release Speed'] = np.nan
    
    # Elbow Velocity During Release Phase
    if 'left_elbow_velocity' in combined_df.columns:
        avg_left_elbow_vel = combined_df.loc[release_point:, 'left_elbow_velocity'].mean()
        metrics['Avg Left Elbow Velocity (units/s)'] = avg_left_elbow_vel
    if 'right_elbow_velocity' in combined_df.columns:
        avg_right_elbow_vel = combined_df.loc[release_point:, 'right_elbow_velocity'].mean()
        metrics['Avg Right Elbow Velocity (units/s)'] = avg_right_elbow_vel
    
    # Hip Velocity During Release Phase
    if 'left_hip_velocity' in combined_df.columns:
        avg_left_hip_vel = combined_df.loc[release_point:, 'left_hip_velocity'].mean()
        metrics['Avg Left Hip Velocity (units/s)'] = avg_left_hip_vel
    if 'right_hip_velocity' in combined_df.columns:
        avg_right_hip_vel = combined_df.loc[release_point:, 'right_hip_velocity'].mean()
        metrics['Avg Right Hip Velocity (units/s)'] = avg_right_hip_vel
    
    metrics_df = pd.DataFrame([metrics])
    
    return metrics_df

def plot_motion_phases(vibe_data):
    """Visualize shooting motion phases on ball trajectory."""
    st.subheader("🎯 Shooting Motion Phases")
    if vibe_data.empty:
        st.warning("No data available to plot shooting phases.")
        return
    
    fig = px.scatter(
        vibe_data, x='x', y='y', color='Phase',
        title="Shooting Phases on Ball Trajectory",
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
        hover_data={'x': True, 'y': True, 'Phase': True}
    )
    fig.add_trace(
        go.Scatter(
            x=vibe_data['x'],
            y=vibe_data['y'],
            mode='lines',
            name='Ball Path'
        )
    )
    # Highlight release point
    release_data = vibe_data[vibe_data['Phase'] == 'Release Phase'].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[release_data['x']],
            y=[release_data['y']],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Release Point'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_biomechanical_angles(data, angle_type='Elbow'):
    """Plot biomechanical angles over time."""
    st.subheader(f"📊 {angle_type} Angles Over Time")
    angle_columns = [col for col in data.columns if angle_type.lower() in col.lower()]
    
    if not angle_columns:
        st.warning(f"No {angle_type} angle data available.")
        return
    
    # Create interactive Plotly line charts for each angle
    for angle in angle_columns:
        if angle in data.columns and pd.api.types.is_numeric_dtype(data[angle]):
            fig = px.line(
                data, y=angle, title=f"{get_readable_label(angle)} Over Time",
                labels={'index': 'Frame', angle: get_readable_label(angle)},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Angle column {angle} not found or non-numeric.")

def plot_biomechanical_angles_overall(data):
    """Plot key biomechanical angles in a consolidated manner."""
    st.markdown("### ⚙️ Key Biomechanical Angles")
    
    angle_types = ['Elbow', 'Shoulder', 'Knee', 'Hip', 'Ankle']
    for angle_type in angle_types:
        plot_biomechanical_angles(data, angle_type=angle_type)

def plot_ball_path(vibe_data):
    """Plot the ball's trajectory based on x and y coordinates."""
    st.subheader("🏀 Ball Path Trajectory")
    if vibe_data.empty:
        st.warning("No ball path data available.")
        return
    
    fig = px.scatter(
        vibe_data, x='x', y='y', title="Ball Trajectory",
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
        hover_data={'x': True, 'y': True},
        trendline="lowess"
    )
    fig.add_trace(
        go.Scatter(
            x=vibe_data['x'],
            y=vibe_data['y'],
            mode='lines',
            name='Ball Path'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_comparative_analysis(metadata_df, user_data, selected_uploads, metrics=['Left Elbow Angle', 'Right Elbow Angle']):
    """Compare biomechanical metrics across multiple uploads."""
    st.subheader("🔄 Comparative Analysis Across Uploads")
    if not selected_uploads:
        st.warning("No uploads selected for comparison.")
        return
    
    comparison_df = pd.DataFrame()
    for upload_id in selected_uploads:
        data = user_data.get(upload_id)
        if data is not None:
            # Convert readable metrics back to column names
            metrics_columns = [col for col in user_data[upload_id].columns if get_readable_label(col) in metrics]
            if metrics_columns:
                summary = data[metrics_columns].mean().to_frame().T
                summary['Upload'] = upload_id
                comparison_df = pd.concat([comparison_df, summary], ignore_index=True)
    
    if comparison_df.empty:
        st.warning("No valid metrics found for the selected uploads.")
        return
    
    # Rename columns for readability
    readable_columns = {col: get_readable_label(col) for col in comparison_df.columns if col not in ['Upload']}
    comparison_df = comparison_df.rename(columns=readable_columns)
    
    # Map upload IDs to human-readable labels
    comparison_df['Upload'] = comparison_df['Upload'].apply(lambda x: map_upload_label(x, metadata_df))
    
    # Melt the DataFrame for Plotly
    melted_df = comparison_df.melt(id_vars=['Upload'], var_name='Metric', value_name='Average Value')
    
    fig = px.bar(
        melted_df, x='Upload', y='Average Value', color='Metric',
        barmode='group', title="Average Biomechanical Metrics per Upload",
        labels={'Upload': 'Upload', 'Average Value': 'Average Value'},
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the comparison table
    st.markdown("### Detailed Metrics")
    st.dataframe(comparison_df.set_index('Upload'))
    
    # Allow downloading comparative report
    download_comparative_report(comparison_df)

def compute_trends(metadata_df, user_data, metric='left_elbow'):
    """Compute and plot trends for a specific metric across uploads."""
    st.subheader(f"📈 Trend Analysis for {get_readable_label(metric)}")
    trends = {}
    for upload_id, data in user_data.items():
        if metric in data.columns:
            # Get human-readable label for the upload
            upload_label = map_upload_label(upload_id, metadata_df)
            trends[upload_label] = data[metric].mean()
    
    if not trends:
        st.warning(f"No data available for metric: {metric}")
        return
    
    trend_df = pd.DataFrame(list(trends.items()), columns=['Upload', 'Average Value'])
    fig = px.line(
        trend_df, x='Upload', y='Average Value',
        title=f"Trend of {get_readable_label(metric)} Across Uploads",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_insights(data, metrics_df):
    """Provide insights based on the motion data."""
    st.subheader("🔍 Insights and Analytics")
    st.markdown("### 📊 Statistical Summary")
    st.write(data.describe())
    
    # Example Insight: Average Angles
    elbow_columns = [col for col in data.columns if 'elbow' in col.lower()]
    if elbow_columns:
        for elbow in elbow_columns:
            avg_angle = data[elbow].mean()
            st.markdown(f"**Average {get_readable_label(elbow)}:** {avg_angle:.2f}°")
    
    # Movement Consistency
    rotation_columns = [col for col in data.columns if 'rotation' in col.lower()]
    for rot in rotation_columns:
        std_dev = data[rot].std()
        st.markdown(f"**{get_readable_label(rot)} Standard Deviation:** {std_dev:.2f}°")
        if std_dev > 10:
            st.warning(f"High variability in {get_readable_label(rot)} detected.")
        else:
            st.success(f"Consistent {get_readable_label(rot)} observed.")
    
    # Correlation Between Joints
    st.markdown("### 🔗 Correlation Between Biomechanical Metrics")
    biomech_metrics = [col for col in data.columns if any(keyword in col.lower() for keyword in ['elbow', 'shoulder', 'knee', 'hip', 'ankle'])]
    if len(biomech_metrics) >= 2:
        corr = data[biomech_metrics].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Biomechanical Metrics Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Not enough biomechanical metrics available for correlation analysis.")
    
    # Additional Insights from Time-Domain Metrics
    st.markdown("### 📈 Time-Domain Metrics Insights")
    if not metrics_df.empty:
        for column in metrics_df.columns:
            value = metrics_df[column].iloc[0]
            st.markdown(f"**{column}:** {value:.2f}")
    else:
        st.warning("No time-domain metrics available for insights.")

def display_kpis(data):
    """Display key performance indicators for the shooting motion."""
    st.markdown("### 🎯 Key Performance Indicators (KPIs)")
    
    # Define KPIs based on available metrics
    kpi_metrics = {
        'Average Left Elbow Angle': data['left_elbow'].mean(),
        'Average Right Elbow Angle': data['right_elbow'].mean(),
        'Average Left Shoulder Elevation': data['left_shoulder_elev'].mean(),
        'Average Right Shoulder Elevation': data['right_shoulder_elev'].mean(),
        'Average Left Knee Angle': data['left_knee'].mean(),
        'Average Right Knee Angle': data['right_knee'].mean(),
        'Average Shot Duration (frames)': data['frame'].max() - data['frame'].min(),
        # Add more KPIs as needed
    }
    
    # Display KPIs using Streamlit's metric component
    cols = st.columns(4)  # Adjust number of columns based on KPIs
    for idx, (kpi, value) in enumerate(kpi_metrics.items()):
        col = cols[idx % 4]
        if 'Duration' in kpi:
            col.metric(kpi, f"{int(value)} frames")
        else:
            col.metric(kpi, f"{value:.2f}°")

def download_report(df, player_name, upload_label):
    """Allow users to download the report as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    filename = f"{player_name.replace(' ', '_')}_{upload_label.replace(' ', '_')}_report.csv"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV Report</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_comparative_report(comparison_df):
    """Allow users to download the comparative analysis as a CSV."""
    csv = comparison_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="comparative_analysis_report.csv">📥 Download Comparative Report</a>'
    st.markdown(href, unsafe_allow_html=True)

def define_shooting_phases(data, vibe_data, metadata_df, frame_rate=30):
    """Define and visualize shooting phases based on pose and ball data."""
    st.markdown("### 🎯 Shooting Phases")
    
    if vibe_data.empty or data.empty:
        st.warning("Insufficient data to define shooting phases.")
        return pd.DataFrame()
    
    # Detect motion phases and calculate time-domain metrics
    metrics_df = detect_motion_phases(data, vibe_data, frame_rate=frame_rate)
    
    if metrics_df.empty:
        st.warning("No time-domain metrics available.")
        return metrics_df
    
    # Plot shooting motion phases
    plot_motion_phases(vibe_data)
    
    return metrics_df

# -------------------------------
# Streamlit Application Layout
# -------------------------------

def main():
    st.title("🏀 **Pivotal Motion Data Visualizer**")
    st.markdown("""
    Welcome to the **Pivotal Motion Data Visualizer**! This application allows you to explore, analyze, and gain insights from your motion capture data. Designed for professional use, it provides comprehensive biomechanical analyses, interactive visualizations, and performance metrics to enhance your training and performance evaluation.
    """)
    
    # Sidebar for navigation
    st.sidebar.header("🔍 Explore Your Data")
    
    # List all users
    bucket_name = st.secrets["BUCKET_NAME"]
    users = list_users(s3_client, bucket_name)
    if not users:
        st.sidebar.error("No users found in the S3 bucket.")
        st.stop()
    
    # Debugging: Display found users
    st.sidebar.markdown(f"**Found {len(users)} users.**")
    st.sidebar.write(users)
    
    # Select user
    selected_user = st.sidebar.selectbox("👤 Select User (Email)", users)
    
    # List segments for the selected user
    segments = list_user_segments(s3_client, bucket_name, selected_user)
    if not segments:
        st.sidebar.error(f"No segments found for user: {selected_user}")
        st.stop()
    
    # Debugging: Display found segments
    st.sidebar.markdown(f"**Found {len(segments)} segments for {selected_user}.**")
    st.sidebar.write(segments)
    
    # Option to select multiple segments for comparison
    compare_segments = st.sidebar.checkbox("🔄 Compare Multiple Segments")
    if compare_segments:
        selected_segments = st.sidebar.multiselect(
            "📂 Select Segments to Compare", 
            options=segments, 
            default=segments[:1]  # Select the first segment by default
        )
    else:
        selected_segments = [st.sidebar.selectbox("📁 Select Segment", segments)]
    
    # Load data for selected segments
    user_data = {}
    for segment in selected_segments:
        data = load_vibe_output(s3_client, bucket_name, selected_user, segment)
        if not data.empty:
            # Assume that 'job_id' or similar identifier is present to fetch metadata
            # Here, we'll extract 'job_id' from the segment name or another column if available
            # Modify this part based on your actual data structure
            job_id = f"{selected_user}/{segment}"  # Example job_id; adjust as needed
            metadata = get_metadata_from_dynamodb(dynamodb, job_id)
            if metadata:
                # Merge metadata with data if necessary
                for key, value in metadata.items():
                    data[key] = value
            user_data[segment] = data
        else:
            st.warning(f"No data loaded for segment '{segment}'.")
    
    if not user_data:
        st.warning("No valid data available for the selected segments.")
        st.stop()
    
    # Assuming all selected segments belong to the same player and team
    # You may need to adjust this if multiple players/teams are involved
    first_segment = selected_segments[0]
    current_metadata = {}
    if first_segment in user_data:
        current_metadata = {
            'player_name': user_data[first_segment].get('player_name', ['Unknown Player'])[0],
            'team': user_data[first_segment].get('team', ['NA'])[0],
            'shot_type': user_data[first_segment].get('shot_type', ['Unknown Shot Type'])[0],
            'shooting_motion': user_data[first_segment].get('shooting_motion', ['N/A'])[0],
            'upload_time': user_data[first_segment].get('upload_time', [pd.NaT])[0],
            'segment_id': user_data[first_segment].get('segment_id', [first_segment])[0]
        }
    
    player_name = current_metadata.get('player_name', 'Unknown Player')
    team = current_metadata.get('team', 'NA')
    formatted_label = f"{player_name} ({team}) - {current_metadata.get('shot_type', 'N/A')} - {current_metadata.get('segment_id', 'N/A')}"
    
    # Header with logo and player/team info
    st.markdown("## 📄 **Report**")
    col1, col2 = st.columns([1, 3])
    with col1:
        team_logo_path = f"static/images/team/{team.lower()}_logo.png"
        if not os.path.exists(team_logo_path):
            team_logo_path = "static/images/team/na_logo.png"  # Default logo
        st.image(team_logo_path, width=100)
    with col2:
        st.markdown(f"### **Player:** {player_name}")
        st.markdown(f"### **Team:** {team}")
        st.markdown(f"### **Upload:** {formatted_label}")
    
    # Display metadata
    display_metadata(current_metadata)
    
    # Display KPIs for each segment
    for segment, data in user_data.items():
        st.markdown(f"### 📊 KPIs for Segment: {segment}")
        display_kpis(data)
    
    # **Data Preview Section**
    st.markdown("### 📊 Data Preview")
    for segment, data in user_data.items():
        st.write(f"**Segment:** {segment}")
        st.write("**Columns in CSV:**", data.columns.tolist())
        st.write("**First 5 Rows:**")
        st.dataframe(data.head())
    
    # Data Visualization
    st.markdown("## 🖼️ Visualizations")
    
    for segment, data in user_data.items():
        st.markdown(f"### 🗂️ Segment: {segment}")
        
        # Plot key biomechanical angles
        plot_biomechanical_angles_overall(data)
        
        # Plot Ball Path
        vibe_upload = data  # Since 'vibe_output.csv' data is already loaded
        if not vibe_upload.empty:
            plot_ball_path(vibe_upload)
        else:
            st.warning(f"No ball path data available for segment '{segment}'.")
        
        # Define and plot shooting phases and calculate metrics
        metrics_df = define_shooting_phases(data, vibe_upload, metadata_df=None)  # Adjust if metadata_df is needed
        
        # Insights and Analytics
        generate_insights(data, metrics_df)
    
    # Comparative Analysis
    if compare_segments and len(selected_segments) > 1:
        st.markdown("## 🔄 Comparative Analysis")
        # Select metrics to compare
        metric_options = [
            'Left Elbow Angle', 'Right Elbow Angle',
            'Left Shoulder Elevation', 'Right Shoulder Elevation',
            'Left Knee Angle', 'Right Knee Angle',
            'Left Ankle Flexion', 'Right Ankle Flexion'
        ]
        selected_metrics = st.multiselect("📈 Select Metrics to Compare", metric_options, default=['Left Elbow Angle', 'Right Elbow Angle'])
        
        if selected_metrics:
            plot_comparative_analysis(None, user_data, selected_segments, metrics=selected_metrics)
            # Download link is already handled within plot_comparative_analysis
        else:
            st.warning("Please select at least one metric for comparison.")
    
    # Trend Analysis
    st.markdown("## 📈 Trend Analysis")
    trend_metric = st.selectbox(
        "📌 Select a Metric for Trend Analysis", 
        [
            'Left Elbow Angle', 'Right Elbow Angle',
            'Left Shoulder Elevation', 'Right Shoulder Elevation',
            'Left Knee Angle', 'Right Knee Angle',
            'Left Ankle Flexion', 'Right Ankle Flexion'
        ],
        index=0
    )
    compute_trends(None, user_data, metric=trend_metric.lower().replace(' ', '_'))
    
    # Download Report
    st.markdown("## 📥 Download Reports")
    for segment, data in user_data.items():
        download_report(data, player_name, segment)
    
    st.markdown("---")
    st.markdown("© 2024 Pivotal Motion. All rights reserved.")

if __name__ == "__main__":
    main()
