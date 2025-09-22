"""
AnDeS - ANomaly DEtection System
====================================================

A Streamlit web application for real-time anomaly detection using matrix profiles.
Based on the AnDeS algorithm for detecting anomalous patterns in Mobile Spatial Statistics (MSS) data.

Author: Erick Mas
Version: 2025.1.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple, Optional
import time
import platform
import psutil
warnings.filterwarnings('ignore')

# Import core ANDES functionality
sys.path.append(str(Path(__file__).parent))
from core import LazyDatabase, ScampAnomalyDetector, DataGenerator

# Set page configuration
st.set_page_config(
    page_title="AnDeS - ANomaly DEtection System",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for real metrics and logs
if 'session_logs' not in st.session_state:
    st.session_state.session_logs = []
if 'detection_metrics' not in st.session_state:
    st.session_state.detection_metrics = {
        'total_detections': 0,
        'processing_time': 0,
        'data_points_processed': 0,
        'anomalies_found': 0,
        'last_detection_time': None
    }

# Data paths configuration
DATA_DIR = Path("/Users/erick/Documents/GitHub/sip_mss_AnDeS/data")  # Update this path as needed

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E4A62;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5D737E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-red {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .alert-green {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
    }
    .alert-orange {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Real MSS data loading and processing functions
def log_message(message: str, level: str = "info"):
    """Add a message to the session logs."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level.upper()}: {message}"
    
    # Ensure session state is initialized before accessing
    if 'session_logs' not in st.session_state:
        st.session_state.session_logs = []
    
    st.session_state.session_logs.append(log_entry)
    
    # Keep only last 100 log entries
    if len(st.session_state.session_logs) > 100:
        st.session_state.session_logs = st.session_state.session_logs[-100:]
    
    # Also log to Python logger
    getattr(logger, level.lower(), logger.info)(message)

def update_metrics(processing_time: float, data_points: int, anomalies: int):
    """Update detection metrics."""
    # Ensure session state is initialized
    if 'detection_metrics' not in st.session_state:
        st.session_state.detection_metrics = {
            'total_detections': 0,
            'processing_time': 0,
            'data_points_processed': 0,
            'anomalies_found': 0,
            'last_detection_time': None
        }
    
    st.session_state.detection_metrics['total_detections'] += 1
    st.session_state.detection_metrics['processing_time'] += processing_time
    st.session_state.detection_metrics['data_points_processed'] += data_points
    st.session_state.detection_metrics['anomalies_found'] += anomalies
    st.session_state.detection_metrics['last_detection_time'] = datetime.now()

@st.cache_data
def get_available_mesh_ids() -> List[str]:
    """Get available mesh IDs from the data files."""
    mesh_ids = []
    
    # Try to load actual mesh IDs from areas files
    try:
        # Check for recent years to get actual mesh IDs
        for year in [2024, 2023, 2022]:
            areas_file = Path.joinpath(DATA_DIR, f"ntt_mss_{year}_areas.npy")
            if areas_file.exists():
                mesh_id_mapping = np.load(areas_file)
                # Convert int32 to strings and take a sample
                actual_mesh_ids = [str(int(mesh_id)) for mesh_id in mesh_id_mapping[:20]]  # Take first 20
                mesh_ids.extend(actual_mesh_ids)
                logger.info(f"Loaded {len(actual_mesh_ids)} mesh IDs from {year} areas file")
                break
        
        # Remove duplicates and sort
        if mesh_ids:
            mesh_ids = sorted(list(set(mesh_ids)))
        
    except Exception as e:
        logger.warning(f"Error loading mesh IDs from areas files: {e}")
    
    # Fallback to hardcoded mesh IDs if loading failed
    if not mesh_ids:
        logger.warning("Using fallback mesh IDs")
        mesh_ids = ['563712311', '563712312', '563712321',
                   '563712213', '563712214', '563712223',
                   '563712211']

    return mesh_ids

@st.cache_data
def get_available_years() -> List[int]:
    """Get available years from the data directory."""
    years = []
    for file_path in DATA_DIR.glob("ntt_mss_*.npy"):
        try:
            year_part = file_path.stem.split('_')[-1]
            if year_part.isdigit() and len(year_part) == 4:
                years.append(int(year_part))
        except:
            continue
    
    return sorted(list(set(years))) if years else list(range(2016, 2026))

@st.cache_data
def load_mss_data(year: int, mesh_id_list: List[str], multi_mesh_analysis: bool = False) -> pd.DataFrame:
    """Load Mobile Spatial Statistics data for specified parameters.
    
    Args:
        year: The year to load data for
        mesh_id_list: List of mesh ID codes to extract
        multi_mesh_analysis: If True, aggregate data from all mesh IDs; if False, use single mesh
    
    Returns:
        DataFrame with columns: timestamp, population, mesh_id
    """
    start_time = time.time()
    
    try:
        # Load main data file and areas mapping
        data_file = Path.joinpath(DATA_DIR, f"ntt_mss_{year}.npy")
        areas_file = Path.joinpath(DATA_DIR, f"ntt_mss_{year}_areas.npy")

        if not data_file.exists():
            log_message(f"Data file not found for year {year}", "error")
            return pd.DataFrame()
        
        # Load the matrix data (rows=hours, columns=mesh_ids)
        data_matrix = np.load(data_file)
        log_message(f"Loaded data matrix shape: {data_matrix.shape}", "info")
        
        # Load areas mapping if available
        mesh_id_mapping = None
        if areas_file.exists():
            mesh_id_mapping = np.load(areas_file)
            log_message(f"Loaded mesh ID mapping: {len(mesh_id_mapping)} mesh IDs", "info")
        else:
            log_message("Areas file not found, using indices", "warning")
        
        # Extract data for specified mesh IDs
        extracted_data = []
        valid_mesh_ids = []
        
        for mesh_id in mesh_id_list:
            try:
                if mesh_id_mapping is not None:
                    # Convert string mesh_id to int32 for comparison with mapping
                    try:
                        mesh_id_int = np.int32(mesh_id)
                        if mesh_id_int in mesh_id_mapping:
                            mesh_index = np.where(mesh_id_mapping == mesh_id_int)[0][0]
                            mesh_data = data_matrix[:, mesh_index]
                            extracted_data.append(mesh_data)
                            valid_mesh_ids.append(mesh_id)
                            log_message(f"Found mesh ID {mesh_id} at index {mesh_index}", "info")
                        else:
                            log_message(f"Mesh ID {mesh_id} not found in mapping", "warning")
                    except (ValueError, OverflowError) as e:
                        log_message(f"Invalid mesh ID format {mesh_id}: {e}", "warning")
                else:
                    # Fallback: try to use mesh_id as index if it's numeric
                    try:
                        mesh_index = int(mesh_id) % data_matrix.shape[1]  # Prevent index errors
                        mesh_data = data_matrix[:, mesh_index]
                        extracted_data.append(mesh_data)
                        valid_mesh_ids.append(mesh_id)
                        log_message(f"Using mesh ID {mesh_id} as index {mesh_index}", "info")
                    except ValueError:
                        log_message(f"Cannot use mesh ID {mesh_id} as index", "warning")
            except Exception as e:
                log_message(f"Error extracting data for mesh ID {mesh_id}: {e}", "warning")
        
        if not extracted_data:
            log_message("No valid mesh IDs found", "error")
            return pd.DataFrame()
        
        # Create timestamps
        num_hours = data_matrix.shape[0]
        timestamps = pd.date_range(
            start=f'{year}-01-01', 
            periods=num_hours, 
            freq='H'
        )
        
        # Process the data based on multi_mesh_analysis setting
        if multi_mesh_analysis and len(extracted_data) > 1:
            # Aggregate data from all mesh IDs (sum)
            aggregated_data = np.sum(extracted_data, axis=0)
            # Use 5th element of mesh list or first if less than 5 elements
            mesh_id_value = valid_mesh_ids[4] if len(valid_mesh_ids) > 4 else valid_mesh_ids[0]
            log_message(f"Aggregated data from {len(extracted_data)} mesh IDs", "info")
        else:
            # Use single mesh (first valid one)
            aggregated_data = extracted_data[0]
            mesh_id_value = valid_mesh_ids[0]
            log_message(f"Using single mesh ID: {mesh_id_value}", "info")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'population': aggregated_data,
            'mesh_id': mesh_id_value
        })
        
        processing_time = time.time() - start_time
        log_message(f"Data loaded successfully: {len(df)} records in {processing_time:.2f}s", "info")
        
        return df
        
    except Exception as e:
        log_message(f"Error loading data: {e}", "error")
        return pd.DataFrame()

class RealTimeAnomalyDetector:
    """Real-time anomaly detector using core SCAMP functionality."""
    
    def __init__(self):
        self.scamp_detector = ScampAnomalyDetector()
        self.lazy_db = LazyDatabase()
        
    def detect_anomalies(self, data: pd.DataFrame, 
                        subsequence_length: int = 24,
                        threshold_method: str = 'sigma',
                        threshold_multiplier: float = None,
                        normalize: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Detect anomalies using the core SCAMP algorithm."""
        start_time = time.time()
        
        try:
            if data.empty or len(data) < subsequence_length:
                log_message("Insufficient data for anomaly detection", "warning")
                return np.array([]), np.array([]), 0.0, 0.0
            
            # Use population or value column
            value_col = 'population' if 'population' in data.columns else 'value'
            values = data[value_col].values
            
            # Configure SCAMP detector
            self.scamp_detector.window_size = subsequence_length
            self.scamp_detector.threshold_method = threshold_method
            self.scamp_detector.normalize = normalize
            
            # Detect anomalies using core algorithm
            anomalies, scores, threshold_used = self.scamp_detector.detect_anomalies_batch(
                values, custom_multiplier=threshold_multiplier
            )
            
            processing_time = time.time() - start_time
            anomaly_count = np.sum(anomalies) if len(anomalies) > 0 else 0
            
            # Update metrics
            update_metrics(processing_time, len(data), anomaly_count)
            
            log_message(
                f"Anomaly detection complete: {anomaly_count} anomalies found in {processing_time:.2f}s using {threshold_method}",
                "info"
            )
            
            return anomalies, scores, threshold_used, processing_time
            
        except Exception as e:
            log_message(f"Error in anomaly detection: {e}", "error")
            return np.array([]), np.array([]), 0.0, 0.0

# Initialize the real-time detector
@st.cache_resource
def get_detector():
    """Get cached detector instance."""
    return RealTimeAnomalyDetector()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'detector' not in st.session_state:
    st.session_state.detector = get_detector()
if 'current_data' not in st.session_state:
    st.session_state.current_data = pd.DataFrame()

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🌋 AnDeS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ANomaly DEtection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🛠️ Configuration")
    
    # Data source selection
    st.sidebar.subheader("📊 Data Source")
    
    # Get available options
    available_years = get_available_years()
    available_mesh_ids = get_available_mesh_ids()
    
    # User inputs
    selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1)
    selected_mesh = st.sidebar.selectbox("Select Mesh ID Range", available_mesh_ids)
    
    # Custom mesh ID input
    st.sidebar.subheader("🔢 Custom Mesh IDs")
    custom_mesh_input = st.sidebar.text_area(
        "Input Mesh ID Range", 
        placeholder="Enter mesh IDs separated by commas\nExample: 563712311, 563712312, 563712321",
        help="Enter one or more mesh ID codes separated by commas"
    )
    
    # Parse custom mesh IDs
    if custom_mesh_input.strip():
        mesh_id_list = [mesh.strip() for mesh in custom_mesh_input.split(',') if mesh.strip()]
    else:
        # Use selected mesh as single item list
        mesh_id_list = [selected_mesh]
    
    multi_mesh_analysis = st.sidebar.checkbox("Multi-mesh analysis", value=False, 
                                             help="Aggregate data from all provided mesh IDs")
    
    # Detection parameters
    st.sidebar.subheader("🔧 Detection Parameters")
    subsequence_length = st.sidebar.slider("Subsequence Length (hours)", 3, 24, 24)
    
    # Threshold method selection
    threshold_method = st.sidebar.selectbox(
        "Threshold Method",
        options=['sigma', 'percentile95', 'percentile99'],
        index=0,
        help="Method for calculating anomaly detection threshold"
    )
    
    # Normalize option
    normalize_matrix_profile = st.sidebar.checkbox(
        "Normalize Matrix Profile", 
        value=False,
        help="Whether to normalize matrix profile distances by subsequence length"
    )
    
    # Threshold multiplier
    if threshold_method == 'sigma':
        threshold_multiplier = st.sidebar.slider(
            "Threshold Multiplier (δ)", 
            1.0, 5.0, 3.0, 
            0.1,
            help="δ in δ×σ threshold calculation"
        )
    else:
        threshold_multiplier = None
    
    # Store configuration in session state
    st.session_state.config = {
        'year': selected_year,
        'mesh_id_list': mesh_id_list,
        'multi_mesh_analysis': multi_mesh_analysis,
        'subsequence_length': subsequence_length,
        'threshold_method': threshold_method,
        'threshold_multiplier': threshold_multiplier,
        'normalize_matrix_profile': normalize_matrix_profile
    }
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time Analysis", "📈 Historical View", "⚙️ System Status", "📚 Documentation"])
    
    with tab1:
        real_time_analysis()
    
    with tab2:
        historical_analysis()
    
    with tab3:
        system_status()
    
    with tab4:
        documentation()

def real_time_analysis():
    """Real-time analysis tab."""
    st.header("📊 Real-time Anomaly Detection")
    
    # Check if configuration exists
    if 'config' not in st.session_state:
        st.warning("Please configure data source in the sidebar first.")
        return
    
    config = st.session_state.config
    
    # Load data
    if st.button("🔄 Load/Refresh Data"):
        with st.spinner("Loading MSS data..."):
            log_message(f"Loading data for {config['year']} - {config['mesh_id_list']}")
            data = load_mss_data(
                year=config['year'],
                mesh_id_list=config['mesh_id_list'],
                multi_mesh_analysis=config['multi_mesh_analysis']
            )
            
            if not data.empty:
                st.session_state.current_data = data
                st.session_state.data_loaded = True
                log_message(f"Successfully loaded {len(data)} data points")
            else:
                st.error("Failed to load data. Check logs for details.")
                return
    
    if not st.session_state.data_loaded or st.session_state.current_data.empty:
        st.info("👆 Click 'Load/Refresh Data' to start analysis")
        return
    
    data = st.session_state.current_data.copy()
    
    # Date range and display controls
    st.subheader("📅 Analysis Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date", 
            data['timestamp'].min().date(),
            min_value=data['timestamp'].min().date(),
            max_value=data['timestamp'].max().date(),
            help="Select start date for analysis"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            data['timestamp'].max().date(),
            min_value=data['timestamp'].min().date(),
            max_value=data['timestamp'].max().date(),
            help="Select end date for analysis"
        )
    
    with col3:
        sample_rows = st.selectbox(
            "Sample Rows to Display",
            [10, 25, 50, 100, 500],
            index=0,
            help="Number of rows to show in sample data table"
        )
    
    # Filter data based on date range
    mask = (data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)
    filtered_data = data[mask].copy()
    
    if filtered_data.empty:
        st.warning("No data available for selected date range. Please adjust your date selection.")
        return
    
    # Display data preview
    st.subheader("📋 Data Preview")
    st.write(f"**Dataset Info:** {len(filtered_data)} records in selected date range ({len(data)} total)")
    
    # Show sample data table
    st.write(f"**Sample Data (First {min(sample_rows, len(filtered_data))} rows):**")
    preview_data = filtered_data.head(sample_rows).copy()
    
    # Format timestamp for better display
    if 'timestamp' in preview_data.columns:
        preview_data['timestamp'] = preview_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Display the table with nice formatting
    st.dataframe(
        preview_data,
        width='stretch',
        hide_index=True
    )
    
    # Add data visualization for preview
    st.subheader("📊 Data Preview Visualization")
    
    # Determine value column
    value_col = 'population' if 'population' in filtered_data.columns else 'value'
    sample_viz_data = filtered_data.head(sample_rows)
    
    fig_preview = go.Figure()
    
    fig_preview.add_trace(go.Scatter(
        x=sample_viz_data['timestamp'],
        y=sample_viz_data[value_col],
        mode='lines+markers',
        name=f'{value_col.title()} (Sample)',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig_preview.update_layout(
        title=f"{value_col.title()} Over Time (Sample Data)",
        xaxis_title="Time",
        yaxis_title=value_col.title(),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_preview, width="stretch")
    
    # Calculate and display enhanced metrics
    st.subheader("📈 Data Statistics")
    col1, col2, col3 = st.columns(3)
    
    # Calculate no data entries (missing data or -1 values)
    no_data_count = 0
    if value_col in filtered_data.columns:
        # Count -1 values and NaN values as "no data"
        no_data_count = (
            (filtered_data[value_col] == -1).sum() + 
            filtered_data[value_col].isna().sum()
        )
    
    with col1:
        if value_col in filtered_data.columns:
            st.metric("📊 Population Range", 
                     f"{filtered_data[value_col].min():.0f} - {filtered_data[value_col].max():.0f}")
        else:
            st.metric("📊 Value Range", 
                     f"{filtered_data[value_col].min():.2f} - {filtered_data[value_col].max():.2f}")
    
    with col2:
        time_range = filtered_data['timestamp'].max() - filtered_data['timestamp'].min()
        st.metric("⏱️ Time Span", f"{time_range.days} days")
    
    with col3:
        st.metric("❌ No Data Entries", 
                 f"{no_data_count} ({(no_data_count/len(filtered_data)*100):.1f}%)")
    
    st.divider()
    
    # Run detection
    if st.button("🔍 Run Anomaly Detection"):
        with st.spinner("Running SCAMP anomaly detection..."):
            detector = get_detector()
            # Use filtered data for anomaly detection
            anomalies, scores, used_threshold, processing_time = detector.detect_anomalies(
                filtered_data, 
                subsequence_length=config['subsequence_length'],
                threshold_method=config['threshold_method'],
                threshold_multiplier=config['threshold_multiplier'],
                normalize=config['normalize_matrix_profile']
            )
            
            if len(anomalies) > 0:
                # Create a fresh copy to avoid data reference issues
                results_data = filtered_data.copy()
                results_data['detected_anomaly'] = anomalies
                results_data['anomaly_score'] = scores
                
                # Store fresh results in session state with configuration
                st.session_state.detection_results = results_data
                st.session_state.detection_threshold = used_threshold
                st.session_state.analysis_date_range = (start_date, end_date)  # Store date range
                st.session_state.detection_config = config.copy()  # Store config used for this detection
                st.session_state.last_processing_time = processing_time  # Store processing time
                log_message(f"Detection complete: {np.sum(anomalies)} anomalies found in date range")
            else:
                st.warning("No anomalies detected or detection failed. Check logs for details.")
                return
    
    # Display results if available
    if 'detection_results' in st.session_state:
        st.divider()
        
        # Check if current config matches the detection config
        if 'detection_config' in st.session_state:
            detection_config = st.session_state.detection_config
            config_changed = (
                config['threshold_method'] != detection_config.get('threshold_method') or
                config['threshold_multiplier'] != detection_config.get('threshold_multiplier') or
                config['subsequence_length'] != detection_config.get('subsequence_length') or
                config['normalize_matrix_profile'] != detection_config.get('normalize_matrix_profile')
            )
            
            if config_changed:
                st.warning("⚠️ **Configuration Changed:** The displayed results were generated with different settings. "
                          "Run anomaly detection again to see results with current configuration.")
        
        # Show analysis info
        if 'analysis_date_range' in st.session_state:
            start_dt, end_dt = st.session_state.analysis_date_range
            processing_time_info = ""
            if 'last_processing_time' in st.session_state:
                processing_time_info = f" | **Processing time:** {st.session_state.last_processing_time:.2f}s"
            
            st.info(f"📊 **Current Analysis:** {start_dt} to {end_dt} | "
                   f"**Records:** {len(st.session_state.detection_results)} | "
                   f"**Anomalies:** {st.session_state.detection_results['detected_anomaly'].sum()}{processing_time_info}")
        
        display_detection_results(st.session_state.detection_results)
    
def display_detection_results(data: pd.DataFrame):
    """Display anomaly detection results."""
    if data.empty:
        st.warning("No data to display")
        return
    
    # Ensure we have the required columns
    if 'detected_anomaly' not in data.columns or 'anomaly_score' not in data.columns:
        st.error("Detection results not available. Run anomaly detection first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_anomalies = data['detected_anomaly'].sum()
    detection_rate = total_anomalies / len(data) * 100
    max_score = data['anomaly_score'].max()
    avg_score = data['anomaly_score'].mean()
    
    with col1:
        st.metric("🚨 Total Anomalies", total_anomalies)
    with col2:
        st.metric("📊 Detection Rate", f"{detection_rate:.1f}%")
    with col3:
        st.metric("⚡ Max Score", f"{max_score:.2f}")
    with col4:
        st.metric("📈 Avg Score", f"{avg_score:.2f}")
    
    # Time series plot
    st.subheader("📈 Time Series with Anomalies")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Population Data', 'Anomaly Scores'),
        vertical_spacing=0.1
    )
    
    # Determine value column
    value_col = 'population' if 'population' in data.columns else 'value'
    
    # Main time series
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data[value_col],
            mode='lines',
            name='Normal Data',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Anomalies
    anomaly_data = data[data['detected_anomaly']]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data[value_col],
                mode='markers',
                name='Detected Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ),
            row=1, col=1
        )
    
    # Anomaly scores
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='orange', width=2)
        ),
        row=2, col=1
    )
    
    # Threshold line
    threshold = st.session_state.get('detection_threshold', None)
    if threshold is not None:
        config = st.session_state.get('config', {})
        threshold_method = config.get('threshold_method', 'unknown')
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold_method})",
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        title_text="Anomaly Detection Results",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text=value_col.title(), row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
    
    st.plotly_chart(fig, width="stretch")
    
    # Anomaly details
    if total_anomalies > 0:
        st.subheader("🔍 Anomaly Details")
        anomaly_details = data[data['detected_anomaly']].copy()
        anomaly_details = anomaly_details.sort_values('timestamp', ascending=True)
        
        # Display top anomalies
        st.dataframe(
            anomaly_details[['timestamp', value_col, 'anomaly_score']].head(10),
            width="stretch"
        )
        
        # Download option
        if st.button("📥 Download Results"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def historical_analysis():
    """Historical analysis tab."""
    st.header("📈 Historical Analysis")
    
    if not st.session_state.data_loaded or st.session_state.current_data.empty:
        st.info("Please load data in the Real-time Analysis tab first.")
        return
    
    data = st.session_state.current_data.copy()
    
    # Date range selector - use analysis date range if available
    col1, col2 = st.columns(2)
    
    # Set default date range from Real-time Analysis if available
    if 'analysis_date_range' in st.session_state:
        default_start, default_end = st.session_state.analysis_date_range
    else:
        default_start = data['timestamp'].min().date()
        default_end = data['timestamp'].max().date()
    
    with col1:
        start_date = st.date_input("Start Date", default_start,
                                 min_value=data['timestamp'].min().date(),
                                 max_value=data['timestamp'].max().date(),
                                 help="Defaults to Real-time Analysis date range")
    with col2:
        end_date = st.date_input("End Date", default_end,
                               min_value=data['timestamp'].min().date(),
                               max_value=data['timestamp'].max().date(),
                               help="Defaults to Real-time Analysis date range")
    
    # Filter data
    mask = (data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)
    filtered_data = data[mask]
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Determine value column
    value_col = 'population' if 'population' in filtered_data.columns else 'value'
    
    # Statistics
    st.subheader("📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Basic Statistics")
        st.write(f"**Mean:** {filtered_data[value_col].mean():.2f}")
        st.write(f"**Std Dev:** {filtered_data[value_col].std():.2f}")
        st.write(f"**Min:** {filtered_data[value_col].min():.2f}")
        st.write(f"**Max:** {filtered_data[value_col].max():.2f}")
    
    with col2:
        st.markdown("### 🚨 Anomaly Statistics")
        # Check if anomaly detection has been run and results are available
        if 'detection_results' in st.session_state and not st.session_state.detection_results.empty:
            detection_data = st.session_state.detection_results
            
            # Check if the date range matches current filtered data
            detection_date_range = st.session_state.get('analysis_date_range', None)
            current_date_range = (start_date, end_date)
            
            if detection_date_range == current_date_range:
                # Use detection results directly
                anomaly_count = detection_data['detected_anomaly'].sum()
                total_points = len(detection_data)
                config = st.session_state.get('detection_config', {})
                threshold_method = config.get('threshold_method', 'Unknown')
                
                st.write(f"**Total Anomalies:** {anomaly_count}")
                st.write(f"**Total Points:** {total_points}")
                st.write(f"**Anomaly Rate:** {(anomaly_count/total_points*100):.2f}%")
                st.write(f"**Method Used:** {threshold_method}")
            else:
                # Date range mismatch
                st.write("**Status:** Results available for different date range")
                st.write("**Action:** Match date range with Real-time Analysis")
                if detection_date_range:
                    st.write(f"**Detection Range:** {detection_date_range[0]} to {detection_date_range[1]}")
        else:
            st.write("**Status:** No anomaly detection run yet")
            st.write("**Action:** Run detection in Real-time tab")
    
    with col3:
        st.markdown("### ⏰ Time Analysis")
        duration = filtered_data['timestamp'].max() - filtered_data['timestamp'].min()
        st.write(f"**Duration:** {duration.days} days")
        st.write(f"**Data Points:** {len(filtered_data)}")
        st.write(f"**Frequency:** Hourly")
        if 'mesh_id' in filtered_data.columns:
            st.write(f"**Mesh ID:** {filtered_data['mesh_id'].iloc[0]}")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Value Distribution")
        fig_hist = px.histogram(
            filtered_data, 
            x=value_col, 
            nbins=50,
            title=f"{value_col.title()} Distribution"
        )
        st.plotly_chart(fig_hist, width="stretch")
    
    with col2:
        st.subheader("📅 Daily Pattern")
        filtered_data['hour'] = filtered_data['timestamp'].dt.hour
        hourly_avg = filtered_data.groupby('hour')[value_col].mean()
        
        fig_daily = px.line(
            x=hourly_avg.index,
            y=hourly_avg.values,
            title=f"Average {value_col.title()} by Hour of Day",
            labels={'x': 'Hour of Day', 'y': f'Average {value_col.title()}'}
        )
        st.plotly_chart(fig_daily, width="stretch")
    
    # Additional analysis if areas data is available
    if 'area_population' in filtered_data.columns:
        st.subheader("🗺️ Area Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Population vs Area Population correlation
            correlation = filtered_data[value_col].corr(filtered_data['area_population'])
            st.metric("Population Correlation", f"{correlation:.3f}")
            
            fig_scatter = px.scatter(
                filtered_data,
                x=value_col,
                y='area_population',
                title="Population vs Area Population",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, width="stretch")
        
        with col2:
            # Time series comparison
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Scatter(
                x=filtered_data['timestamp'],
                y=filtered_data[value_col],
                mode='lines',
                name='Population',
                line=dict(color='blue')
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=filtered_data['timestamp'],
                y=filtered_data['area_population'],
                mode='lines',
                name='Area Population',
                yaxis='y2',
                line=dict(color='red')
            ))
            
            fig_compare.update_layout(
                title="Population Comparison Over Time",
                xaxis_title="Time",
                yaxis=dict(title="Population", side="left"),
                yaxis2=dict(title="Area Population", side="right", overlaying="y"),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_compare, width="stretch")

def system_status():
    """System status tab."""
    st.header("⚙️ System Status")
    
    # Auto-refresh control
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("🔄 Live System Monitoring")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    if auto_refresh:
        # Auto-refresh every 10 seconds
        placeholder = st.empty()
        with placeholder.container():
            st.info("🔄 Auto-refresh enabled. System will update every 10 seconds.")
        time.sleep(1)  # Brief delay to show the message
        st.rerun()
    
    # Current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {current_time}")
    
    # System health indicators
    col1, col2, col3 = st.columns(3)
    
    # Check data availability and system status
    data_available = len(get_available_years()) > 0
    detector_ready = 'detector' in st.session_state
    data_loaded = st.session_state.get('data_loaded', False)
    detection_run = 'detection_results' in st.session_state
    
    with col1:
        st.markdown("**🔧 Core Systems**")
        data_status = "✅ Data Pipeline: Active" if data_available else "❌ Data Pipeline: No Data"
        detector_status = "✅ Detection Engine: Ready" if detector_ready else "❌ Detection Engine: Not Ready"
        load_status = "✅ Data Loaded: Yes" if data_loaded else "⚠️ Data Loaded: No"
        
        st.markdown(f'<div class="alert-green">{data_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-green">{detector_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-{"green" if data_loaded else "orange"}">{load_status}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💾 Data & Storage**")
        db_status = "✅ Database: Connected" if DATA_DIR.exists() else "❌ Database: Not Found"
        files_count = len(list(DATA_DIR.glob("ntt_mss_*.npy"))) if DATA_DIR.exists() else 0
        file_status = f"✅ Data Files: {files_count} available" if files_count > 0 else "❌ Data Files: None found"
        detection_status = "✅ Detection: Results Available" if detection_run else "⚠️ Detection: Not Run"
        
        st.markdown(f'<div class="alert-green">{db_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-{"green" if files_count > 0 else "red"}">{file_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-{"green" if detection_run else "orange"}">{detection_status}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**🌐 Interface & Processing**")
        web_status = "✅ Web Interface: Online"
        viz_status = "✅ Visualization: Active"
        
        # Check current processing status
        if 'detection_metrics' in st.session_state:
            last_detection = st.session_state.detection_metrics.get('last_detection_time')
            if last_detection:
                time_since = datetime.now() - last_detection
                if time_since.total_seconds() < 300:  # Less than 5 minutes
                    process_status = "✅ Processing: Recently Active"
                else:
                    process_status = "⚠️ Processing: Idle"
            else:
                process_status = "⚠️ Processing: Never Run"
        else:
            process_status = "⚠️ Processing: No Metrics"
        
        st.markdown(f'<div class="alert-green">{web_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-green">{viz_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-{"green" if "Recently Active" in process_status else "orange"}">{process_status}</div>', unsafe_allow_html=True)
    
    # Real performance metrics with live updates
    st.subheader("📊 Live Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚡ System Performance**")
        # Safe access to metrics
        if 'detection_metrics' in st.session_state:
            metrics = st.session_state.detection_metrics
        else:
            metrics = {
                'total_detections': 0,
                'processing_time': 0,
                'data_points_processed': 0,
                'anomalies_found': 0,
                'last_detection_time': None
            }
        
        # Calculate actual metrics
        avg_processing_time = (
            metrics['processing_time'] / metrics['total_detections'] 
            if metrics['total_detections'] > 0 else 0
        )
        
        # Real-time system metrics
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        memory_total = memory_info.total // (1024**3)  # GB
        memory_used = memory_info.used // (1024**3)  # GB
        
        cpu_usage = psutil.cpu_percent(interval=0.1)  # Quick sample
        cpu_count = psutil.cpu_count()
        
        detection_rate = (
            metrics['anomalies_found'] / metrics['data_points_processed'] * 100
            if metrics['data_points_processed'] > 0 else 0
        )
        
        # Enhanced performance data
        perf_data = pd.DataFrame({
            'Metric': [
                'Avg Detection Time', 
                'Memory Usage', 
                'Memory Details',
                'CPU Usage', 
                'CPU Cores',
                'Detection Rate',
                'Total Sessions'
            ],
            'Value': [
                f'{avg_processing_time:.2f}s',
                f'{memory_usage:.1f}%',
                f'{memory_used}GB / {memory_total}GB',
                f'{cpu_usage:.1f}%',
                f'{cpu_count} cores',
                f'{detection_rate:.2f}%',
                str(metrics['total_detections'])
            ],
            'Status': [
                '🟢 Good' if avg_processing_time < 5 else '🟡 Slow' if avg_processing_time < 15 else '🔴 Poor',
                '🟢 Normal' if memory_usage < 70 else '🟡 High' if memory_usage < 85 else '🔴 Critical',
                '📊 Info',
                '🟢 Good' if cpu_usage < 50 else '🟡 High' if cpu_usage < 80 else '🔴 Critical',
                '📊 Info',
                '🟢 Active' if detection_rate > 0 else '⚪ None',
                '📊 Info'
            ]
        })
        st.dataframe(perf_data, width="stretch")
    
    with col2:
        st.markdown("**⚙️ Current Configuration & Status**")
        # Current configuration
        config = st.session_state.get('config', {})
        
        # Get current data info
        current_data_info = ""
        if 'current_data' in st.session_state and not st.session_state.current_data.empty:
            data_len = len(st.session_state.current_data)
            current_data_info = f"{data_len:,} records"
        else:
            current_data_info = "No data loaded"
        
        # Detection info
        detection_info = "Not run"
        if 'detection_results' in st.session_state:
            detection_data = st.session_state.detection_results
            anomalies = detection_data['detected_anomaly'].sum() if 'detected_anomaly' in detection_data.columns else 0
            detection_info = f"{anomalies} anomalies found"
        
        # Get configured mesh IDs info
        configured_mesh_info = "Not configured"
        if config.get('mesh_id_list'):
            mesh_count = len(config['mesh_id_list'])
            if mesh_count == 1:
                configured_mesh_info = f"1 mesh ID: {config['mesh_id_list'][0]}"
            else:
                configured_mesh_info = f"{mesh_count} mesh IDs"
        
        config_data = pd.DataFrame({
            'Parameter': [
                'Subsequence Length', 
                'Threshold Method', 
                'Threshold Multiplier',
                'Normalize Matrix Profile',
                'Available Years', 
                'Configured Mesh IDs',
                'Loaded Data',
                'Last Detection',
                'Session Uptime'
            ],
            'Value': [
                f"{config.get('subsequence_length', 'Not set')} hours",
                config.get('threshold_method', 'Not set'),
                f"{config.get('threshold_multiplier', 'Auto')}" if config.get('threshold_multiplier') else 'Auto',
                f"{config.get('normalize_matrix_profile', False)}",
                f"{len(get_available_years())} years",
                configured_mesh_info,
                current_data_info,
                detection_info,
                f"{(datetime.now() - datetime.now().replace(second=0, microsecond=0)).total_seconds()//60:.0f} min" # Approximate
            ]
        })
        st.dataframe(config_data, width="stretch")
    
    # Real log viewer
    st.subheader("📝 Session Logs")
    
    # Safe access to session logs
    if hasattr(st.session_state, 'session_logs') and st.session_state.session_logs:
        # Display last 20 logs
        recent_logs = st.session_state.session_logs[-20:]
        
        # Add refresh button
        if st.button("🔄 Refresh Logs"):
            st.rerun()
        
        log_container = st.container()
        with log_container:
            for log in reversed(recent_logs):  # Show most recent first
                # Color code by log level
                if "ERROR" in log:
                    st.markdown(f'<div class="alert-red">{log}</div>', unsafe_allow_html=True)
                elif "WARNING" in log:
                    st.markdown(f'<div class="alert-orange">{log}</div>', unsafe_allow_html=True)
                else:
                    st.text(log)
    else:
        st.info("No session logs available. Start using the application to see logs.")
    
    # System information
    st.subheader("💻 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Environment Information:**")
        st.write(f"- Python Version: {platform.python_version()}")
        st.write(f"- Platform: {platform.system()} {platform.release()}")
        st.write(f"- Available Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        st.write(f"- CPU Cores: {psutil.cpu_count()}")
    
    with col2:
        st.write("**Data Information:**")
        st.write(f"- Data Directory: {DATA_DIR}")
        st.write(f"- Data Files Available: {len(list(DATA_DIR.glob('ntt_mss_*.npy')))}")
        st.write(f"- Last Detection: {metrics.get('last_detection_time', 'Never')}")
        
        if st.button("🧹 Clear Session Data"):
            # Reset session state
            for key in ['current_data', 'detection_results', 'data_loaded']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Session data cleared!")
            time.sleep(1)
            st.rerun()

def documentation():
    """Documentation tab."""
    st.header("📚 Documentation")
    
    # Introduction
    st.markdown("""
    ## 🌋 Welcome to AnDeS 2025
    
    The **ANomaly DEtection System (AnDeS)** is a cutting-edge anomaly detection platform 
    designed to identify unusual patterns in Mobile Spatial Statistics (MSS) data that may indicate 
    natural disasters or other significant events.
    
    ### 🎯 Key Features
    
    - **Real-time Anomaly Detection**: Interactive analysis with customizable parameters
    - **Matrix Profile Analysis**: SCAMP algorithm for advanced pattern matching
    - **Interactive Visualization**: Rich Plotly charts with anomaly overlays
    - **Historical Analysis**: Deep dive into historical patterns with date range filtering
    - **Multi-mesh Analysis**: Support for single or aggregated mesh analysis
    - **Live System Monitoring**: Real-time performance metrics and system status
    - **Custom Thresholds**: Flexible δ×σ threshold configuration with percentile alternatives
    
    ### 🔬 Algorithm Overview
    
    AnDeS uses the **SCAMP (Scalable Matrix Profile)** algorithm to detect anomalies:
    
    1. **Data Preprocessing**: Load and filter MSS data by date range and mesh IDs
    2. **Matrix Profile Computation**: Calculate similarity between subsequences using PySCAMP or custom implementation
    3. **Threshold Detection**: Apply configurable thresholds (δ×σ, percentile95, percentile99)
    4. **Anomaly Classification**: Flag unusual patterns for investigation and visualization
    
    ### 📊 Data Sources
    
    - **Mobile Spatial Statistics (MSS)**: Population density and movement patterns from NTT data
    - **Geographical Mesh Codes**: 4th-level mesh identifiers for precise location mapping
    - **Temporal Resolution**: Hourly data for fine-grained analysis (2016-2025)
    - **File Format**: NumPy arrays (.npy) with corresponding area mapping files
    
    ### 🚀 Getting Started
    
    #### Step 1: Configure Data Source
    1. Select year from available data (2016-2025)
    2. Choose mesh ID from dropdown or enter custom mesh IDs (comma-separated)
    3. Enable "Multi-mesh analysis" to aggregate data from multiple mesh IDs
    
    #### Step 2: Set Detection Parameters
    1. **Subsequence Length**: 3-24 hours (default: 24)
    2. **Threshold Method**: 
       - **sigma**: δ×σ method with custom multiplier (1.0-5.0, default: 3.0)
       - **percentile95**: 95th percentile threshold
       - **percentile99**: 99th percentile threshold
    3. **Normalize Matrix Profile**: Option to normalize distances by subsequence length (default: False)
    
    #### Step 3: Load and Analyze Data
    1. Navigate to **Real-time Analysis** tab
    2. Click **🔄 Load/Refresh Data** to load MSS data
    3. Adjust date range for analysis
    4. Click **🔍 Run Anomaly Detection** to start analysis
    
    ### ⚙️ Configuration Options
    
    #### Detection Parameters
    - **Subsequence Length**: Time window for pattern matching (hours)
    - **Threshold Method**: Statistical method for anomaly detection
    - **Threshold Multiplier (δ)**: Sensitivity control for σ-based methods
    - **Normalize Matrix Profile**: Whether to normalize distances by subsequence length
    
    #### Data Configuration
    - **Year Selection**: Choose from available data years
    - **Mesh ID Range**: Single or multiple mesh regions
    - **Multi-mesh Analysis**: Aggregate multiple mesh regions
    - **Date Range**: Filter analysis to specific time periods
    
    ### 📈 Interface Tabs
    
    #### 📊 Real-time Analysis
    - Load and preview MSS data
    - Configure date ranges and display options
    - Run anomaly detection with live results
    - Interactive visualizations with anomaly overlays
    - Downloadable results in CSV format
    
    #### 📈 Historical View  
    - Statistical summaries of loaded data
    - Anomaly detection status and metrics
    - Distribution analysis and daily patterns
    - Correlation analysis (when available)
    
    #### ⚙️ System Status
    - Live system health monitoring
    - Real-time performance metrics
    - Memory and CPU usage tracking
    - Session logs and configuration status
    - Auto-refresh capabilities
    
    #### 📚 Documentation
    - Comprehensive usage guide
    - Technical specifications
    - Feature descriptions
    
    ### 🔧 Technical Specifications
    
    #### Core Technologies
    - **Programming Language**: Python 3.8+
    - **Web Framework**: Streamlit
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Plotly
    - **Algorithm**: SCAMP Matrix Profile (PySCAMP or custom implementation)
    - **System Monitoring**: psutil
    
    #### Data Requirements
    - MSS data files: `ntt_mss_{year}.npy`
    - Area mapping files: `ntt_mss_{year}_areas.npy`
    - Minimum subsequence length: 3 hours
    - Supported data range: 2016-2025
    
    #### Performance Features
    - Efficient data caching and lazy loading
    - Memory-optimized matrix profile computation
    - Live system performance monitoring
    - Session state management for consistent analysis
    
    ### 🎯 Use Cases
    
    - **Earthquake Detection**: Monitor population movement patterns before/after seismic events
    - **Emergency Response**: Analyze evacuation patterns during disasters
    - **Urban Planning**: Understand population flow and density changes
    - **Research**: Academic studies on human mobility and disaster response
    - **Policy Analysis**: Evaluate effectiveness of emergency response measures
    
    ### � Tips for Best Results
    
    1. **Data Quality**: Ensure complete data files are available for analysis periods
    2. **Threshold Tuning**: Start with δ=3.0 for σ-based methods, adjust based on results
    3. **Window Selection**: Use 24-hour windows for daily pattern detection
    4. **Date Ranges**: Focus on specific event periods for targeted analysis
    5. **Multi-mesh Analysis**: Use for regional-scale event detection
    
    ### 🔄 Recent Updates (v2025.1.0)
    
    - ✅ Simplified threshold configuration to δ×σ format
    - ✅ Enhanced date range filtering and consistency
    - ✅ Live system status monitoring with auto-refresh
    - ✅ Improved anomaly details sorting by timestamp
    - ✅ Better session state management and configuration tracking
    - ✅ Fixed duplicate detection execution issues
    
    ### 📞 Support

    For technical support or questions about AnDeS, please contact the development team.
    """)
    
    # Credits
    st.markdown("""
    ---
    ### 👥 Credits
    
    **AnDeS** is developed by the Erick Mas@Tohoku University.
    
    **Based on**: AnDeS SCAMP algorithm for MSS data analysis
    
    **Version**: 2025.1.0
    
    **Last Updated**: September 2025
    """)

# Always run main in Streamlit
main()