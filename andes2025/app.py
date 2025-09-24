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
import geopandas as gpd
import json
import tempfile
warnings.filterwarnings('ignore')

# Import core ANDES functionality
sys.path.append(str(Path(__file__).parent))
from core import LazyDatabase, ScampAnomalyDetector, DataGenerator

# Define timezone for event dates
from datetime import timezone

events = [
    {
        'event_dt': datetime(2024,1,1,16,0,0,0,timezone.utc),
        'meshcode': 563712214,
        'meshcodes': [563712311, 563712312, 563712321,
                      563712213, 563712214, 563712223,
                      563712211],
        'event': 'Noto Peninsula Earthquake (Mw7.5)'
    },

    {
        'event_dt': datetime(2024,1,2,17,0,0,0,timezone.utc),
        'meshcode': 533926621,
        'meshcodes': [533926614, 533926623, 533926624,
                      533926612, 533926621, 533926622,
                      533926514, 533926523, 533926524],
        'event': 'Haneda Airport runway collision'
    },
    
        {
        'event_dt': datetime(2019,10,12,17,0,0,0,timezone.utc),
        'meshcode': 564003222,
        'meshcodes': [564003114, 564003103, 564003113,
                      564003623, 564003222, 564003231,
                      564003523, 564003504, 564003624],
        'event': 'Typhoon Hagibis (Koriyama)'
    },
        
        {
        'event_dt': datetime(2021,7,3,10,30,0,0,timezone.utc),
        'meshcode': 523950251,
        'meshcodes': [523950244, 523950253, 523950254,
                      523950242, 523950251, 523950252,
                      523950144, 523950153, 523950154],
        'event': 'Atami (Isuzan) Landslide'
    },

]


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
def get_available_events() -> List[dict]:
    """Get available events with their mesh codes."""
    return events

@st.cache_data
def get_available_mesh_ids() -> List[str]:
    """Get available mesh IDs from the predefined events."""
    mesh_ids = []
    
    # Extract all unique mesh codes from events
    for event in events:
        # Add main meshcode
        if 'meshcode' in event:
            mesh_ids.append(str(event['meshcode']))
        
        # Add all meshcodes from the list
        if 'meshcodes' in event:
            mesh_ids.extend([str(code) for code in event['meshcodes']])
    
    # Remove duplicates and sort
    mesh_ids = sorted(list(set(mesh_ids)))
    
    logger.info(f"Loaded {len(mesh_ids)} unique mesh IDs from {len(events)} events")
    return mesh_ids

def get_event_by_name(event_name: str) -> dict:
    """Get event details by event name."""
    for event in events:
        if event['event'] == event_name:
            return event
    return None

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
def get_available_date_range() -> Tuple[datetime, datetime]:
    """Get the available date range from all data files."""
    available_years = get_available_years()
    if not available_years:
        return datetime(2016, 1, 1), datetime(2025, 12, 31)
    
    # Return full range from first year to last year
    start_date = datetime(min(available_years), 1, 1)
    end_date = datetime(max(available_years), 12, 31)
    return start_date, end_date

@st.cache_data
def load_mss_data_monthly_timeseries(event_datetime: datetime, 
                                    mesh_id_list: List[str], 
                                    multi_mesh_analysis: bool = False,
                                    years_back: int = None,
                                    is_actual_event: bool = False,
                                    is_predefined_event: bool = False) -> pd.DataFrame:
    """Load Mobile Spatial Statistics data using the new monthly time series logic.
    
    This function:
    1. Takes the event datetime from the user
    2. For custom analysis: Searches in each year from (event_year - years_back) to the event year
       and extracts exactly one month of data from one month before until the event date/time in each year
    3. For predefined events: Extracts one month before + one week after the event date/time for better analysis
    4. For actual events: Extracts one month of data from one month before the actual event date
    5. Concatenates all monthly periods into one time series
    
    Args:
        event_datetime: The datetime of the event of interest
        mesh_id_list: List of mesh ID codes to extract
        multi_mesh_analysis: If True, aggregate data from all mesh IDs; if False, use single mesh
        years_back: Number of years back from event year to include (if None, defaults to all available from 2016)
        is_actual_event: If True, treats this as a specific event and extracts data around the actual event date only
        is_predefined_event: If True, extends the extraction period to include one week after the event date
    
    Returns:
        DataFrame with columns: timestamp, population, mesh_id, year_source
    """
    start_time = time.time()
    
    try:
        # event_datetime should already be timezone-naive from the UI processing
        # Calculate which years to analyze
        end_year = event_datetime.year
        
        if years_back is None:
            # Default behavior: from 2016 to event year
            start_year = 2016
        else:
            # User-specified years back
            start_year = max(2016, end_year - years_back + 1)  # Ensure we don't go before 2016
        
        # Validate that we have data available
        if start_year > end_year:
            log_message(f"Invalid year range: start_year {start_year} > end_year {end_year}", "error")
            return pd.DataFrame()
        
        years_to_load = list(range(start_year, end_year + 1))
        
        log_message(f"Loading monthly time series for {len(years_to_load)} years: {years_to_load}", "info")
        log_message(f"Event reference: {event_datetime.strftime('%Y-%m-%d %H:%M')}", "info")
        if years_back is not None:
            log_message(f"User specified {years_back} years back from {end_year}", "info")
        if is_actual_event:
            log_message("Processing as actual event - extracting data around the specific event date only", "info")
        
        all_dataframes = []
        
        if is_actual_event:
            # For actual events, extract data only around the actual event date
            # Calculate one month before the actual event date
            event_year = event_datetime.year
            
            if event_datetime.month == 1:
                # Handle January case - go to December of previous year
                # Check if previous year data is available
                if event_year - 1 < 2016:
                    log_message(f"Skipping actual event extraction - January {event_year} event requires {event_year-1} data which is not available", "warning")
                    return pd.DataFrame()
                month_before = event_datetime.replace(year=event_year-1, month=12)
            else:
                # Normal case - subtract one month
                try:
                    month_before = event_datetime.replace(month=event_datetime.month - 1)
                except ValueError:
                    # Handle case where day doesn't exist in target month (e.g., Jan 31 -> Feb 31)
                    if event_datetime.month == 3:  # March to February
                        # Use last day of February
                        import calendar
                        last_day = calendar.monthrange(event_year, 2)[1]
                        month_before = event_datetime.replace(month=2, day=min(event_datetime.day, last_day))
                    else:
                        # For other months, use day 1 as fallback
                        month_before = event_datetime.replace(month=event_datetime.month - 1, day=1)
            
            # Define the one-month extraction period (from one month before TO the event date)
            period_start = month_before
            period_end = event_datetime
            
            log_message(f"Actual event extraction: {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}", "info")
            
            # Load data for the event year only
            event_df = load_mss_data_single_year(event_year, mesh_id_list, multi_mesh_analysis)
            
            # If January event and we need December data from previous year, also load that
            if event_datetime.month == 1 and event_year - 1 >= 2016:
                prev_year_df = load_mss_data_single_year(event_year - 1, mesh_id_list, multi_mesh_analysis)
                if not prev_year_df.empty:
                    event_df = pd.concat([prev_year_df, event_df], ignore_index=True)
            
            if not event_df.empty:
                if 'timestamp' in event_df.columns:
                    # Convert period_start and period_end to pandas Timestamp objects for proper comparison
                    period_start_ts = pd.Timestamp(period_start)
                    period_end_ts = pd.Timestamp(period_end)
                    
                    # Filter to the monthly period
                    mask = (event_df['timestamp'] >= period_start_ts) & (event_df['timestamp'] <= period_end_ts)
                    filtered_df = event_df[mask].copy()
                    
                    if not filtered_df.empty:
                        # Add year source information for tracking
                        filtered_df['year_source'] = event_year
                        all_dataframes.append(filtered_df)
                        log_message(f"Added {len(filtered_df)} records for actual event period", "info")
                    else:
                        log_message(f"No data found for actual event period", "warning")
                else:
                    log_message(f"No timestamp column found in event year data", "warning")
            else:
                log_message(f"No data available for event year {event_year}", "warning")
                
        else:
            # Original logic for custom analysis - multiple years with same month/day pattern
            for year in years_to_load:
                # For each year, calculate the one-month period from one month before to the event date
                # Create the reference date for this year (the event date in this year)
                try:
                    year_event_date = event_datetime.replace(year=year)
                except ValueError:
                    # Handle leap year case (Feb 29)
                    if event_datetime.month == 2 and event_datetime.day == 29:
                        year_event_date = event_datetime.replace(year=year, day=28)
                    else:
                        log_message(f"Error creating reference date for year {year}, skipping", "warning")
                        continue
                
                # Calculate one month before the event date
                if year_event_date.month == 1:
                    # Handle January case - go to December of previous year
                    # Special case: for January 2016, skip because 2015 data is not available
                    if year == 2016:
                        log_message(f"Skipping year {year} - January 2016 event requires 2015 data which is not available", "warning")
                        continue
                    month_before = year_event_date.replace(year=year-1, month=12)
                else:
                    # Normal case - subtract one month
                    try:
                        month_before = year_event_date.replace(month=year_event_date.month - 1)
                    except ValueError:
                        # Handle case where day doesn't exist in target month (e.g., Jan 31 -> Feb 31)
                        if year_event_date.month == 3:  # March to February
                            # Use last day of February
                            import calendar
                            last_day = calendar.monthrange(year, 2)[1]
                            month_before = year_event_date.replace(month=2, day=min(year_event_date.day, last_day))
                        else:
                            # For other months, use day 1 as fallback
                            month_before = year_event_date.replace(month=year_event_date.month - 1, day=1)
                
                # Define the extraction period (from one month before TO the event date, plus one week for predefined events)
                period_start = month_before
                period_end = year_event_date
                
                # For predefined events, extend the period by one week after the event date
                if is_predefined_event:
                    from datetime import timedelta
                    period_end = year_event_date + timedelta(weeks=1)
                    log_message(f"Predefined event: extending extraction period by 1 week to {period_end.strftime('%Y-%m-%d %H:%M')}", "info")
                
                log_message(f"Year {year}: Extracting {period_start.strftime('%Y-%m-%d %H:%M')} to {period_end.strftime('%Y-%m-%d %H:%M')}", "info")
                
                # Handle cross-year periods (e.g., January events need December data from previous year, or extended periods into next year)
                years_to_load_for_period = [year]
                if period_start.year != period_end.year:
                    # Cross-year period: need to load all years between start and end
                    all_years_needed = list(range(period_start.year, period_end.year + 1))
                    years_to_load_for_period = all_years_needed
                    log_message(f"Cross-year period detected: loading data from years {years_to_load_for_period}", "info")
                
                # Load data for all required years for this period
                period_dataframes = []
                for load_year in years_to_load_for_period:
                    year_df = load_mss_data_single_year(load_year, mesh_id_list, multi_mesh_analysis)
                    if not year_df.empty:
                        period_dataframes.append(year_df)
                
                # Combine data from all required years
                if period_dataframes:
                    combined_year_df = pd.concat(period_dataframes, ignore_index=True)
                    
                    if 'timestamp' in combined_year_df.columns:
                        # Convert period_start and period_end to pandas Timestamp objects for proper comparison
                        period_start_ts = pd.Timestamp(period_start)
                        period_end_ts = pd.Timestamp(period_end)
                        
                        # Filter to the monthly period
                        mask = (combined_year_df['timestamp'] >= period_start_ts) & (combined_year_df['timestamp'] <= period_end_ts)
                        filtered_df = combined_year_df[mask].copy()
                        
                        if not filtered_df.empty:
                            # Add year source information for tracking
                            filtered_df['year_source'] = year
                            all_dataframes.append(filtered_df)
                            log_message(f"Added {len(filtered_df)} records from year {year} monthly period (cross-year: {len(period_dataframes)} data files)", "info")
                        else:
                            log_message(f"No data found for year {year} in the specified monthly period", "warning")
                    else:
                        log_message(f"No timestamp column found in combined year data", "warning")
                else:
                    log_message(f"No data available for required years {years_to_load_for_period}", "warning")
        
        if not all_dataframes:
            log_message("No data found for any of the specified monthly periods", "error")
            return pd.DataFrame()
        
        # Concatenate all monthly dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        processing_time = time.time() - start_time
        total_records = len(combined_df)
        years_with_data = combined_df['year_source'].nunique()
        
        log_message(f"Monthly time series loaded successfully: {total_records} records from {years_with_data} years in {processing_time:.2f}s", "info")
        log_message(f"Time series spans: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}", "info")
        
        return combined_df
        
    except Exception as e:
        log_message(f"Error loading monthly time series data: {e}", "error")
        return pd.DataFrame()

@st.cache_data
def load_mss_data_by_date_range(start_date: datetime.date, end_date: datetime.date, 
                               mesh_id_list: List[str], multi_mesh_analysis: bool = False) -> pd.DataFrame:
    """Load Mobile Spatial Statistics data for specified date range (can span multiple years).
    
    Args:
        start_date: Start date for data loading
        end_date: End date for data loading
        mesh_id_list: List of mesh ID codes to extract
        multi_mesh_analysis: If True, aggregate data from all mesh IDs; if False, use single mesh
    
    Returns:
        DataFrame with columns: timestamp, population, mesh_id
    """
    start_time = time.time()
    
    try:
        # Determine which years we need to load
        years_to_load = list(range(start_date.year, end_date.year + 1))
        log_message(f"Loading data for years: {years_to_load}", "info")
        
        all_dataframes = []
        
        for year in years_to_load:
            # Load data for this year using the original function
            year_df = load_mss_data_single_year(year, mesh_id_list, multi_mesh_analysis)
            
            if not year_df.empty:
                # Filter to date range
                year_start = max(datetime(year, 1, 1), datetime.combine(start_date, datetime.min.time()))
                year_end = min(datetime(year, 12, 31, 23, 59, 59), datetime.combine(end_date, datetime.max.time()))
                
                mask = (year_df['timestamp'] >= year_start) & (year_df['timestamp'] <= year_end)
                filtered_df = year_df[mask].copy()
                
                if not filtered_df.empty:
                    all_dataframes.append(filtered_df)
                    log_message(f"Added {len(filtered_df)} records from year {year}", "info")
        
        if not all_dataframes:
            log_message("No data found for the specified date range", "error")
            return pd.DataFrame()
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        processing_time = time.time() - start_time
        log_message(f"Multi-year data loaded successfully: {len(combined_df)} records across {len(years_to_load)} years in {processing_time:.2f}s", "info")
        
        return combined_df
        
    except Exception as e:
        log_message(f"Error loading multi-year data: {e}", "error")
        return pd.DataFrame()

@st.cache_data
def load_mss_data_single_year(year: int, mesh_id_list: List[str], multi_mesh_analysis: bool = False) -> pd.DataFrame:
    """Load Mobile Spatial Statistics data for a single year.
    
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
            # Aggregate data from all mesh IDs (sum), handling -1 values properly
            # Convert to numpy array for easier manipulation
            data_array = np.array(extracted_data)
            
            # Replace -1 with np.nan for proper aggregation
            data_array = np.where(data_array == -1, np.nan, data_array)
            
            # Sum ignoring NaN values (use nansum)
            aggregated_data = np.nansum(data_array, axis=0)
            
            # If all values for a time point are NaN, set result to -1 (no data)
            all_nan_mask = np.all(np.isnan(data_array), axis=0)
            aggregated_data = np.where(all_nan_mask, -1, aggregated_data)
            
            # Use 5th element of mesh list or first if less than 5 elements
            mesh_id_value = valid_mesh_ids[4] if len(valid_mesh_ids) > 4 else valid_mesh_ids[0]
            log_message(f"Aggregated data from {len(extracted_data)} mesh IDs (handling -1 values properly)", "info")
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

# GeoJSON Processing Functions
@st.cache_data
def load_japan_mesh_source(mesh_path="/Volumes/Pegasus32/japan/mesh/japan_mesh4_CRS84.geojson"):
    """Load the Japan mesh source file."""
    try:
        # Check alternative paths if the default doesn't exist
        alternative_paths = [
            mesh_path,
            "/Users/erick/Documents/GitHub/sip_mss_AnDeS/data/japan_mesh4_CRS84.geojson",
            "./data/japan_mesh4_CRS84.geojson",
            "../data/japan_mesh4_CRS84.geojson"
        ]
        
        working_path = None
        for path in alternative_paths:
            if os.path.exists(path):
                working_path = path
                break
        
        if working_path:
            st.info(f"Loading Japan mesh data from: {working_path}... This may take a moment.")
            
            # Try to load a sample first to check structure
            sample_mesh = gpd.read_file(working_path, rows=10)
            log_message(f"Sample loaded with columns: {list(sample_mesh.columns)}", "info")
            
            # Load the full dataset
            japan_mesh = gpd.read_file(working_path)
            log_message(f"Japan mesh loaded: {len(japan_mesh)} mesh regions with columns: {list(japan_mesh.columns)}", "info")
            
            # Ensure there's a mesh ID column
            mesh_id_cols = [col for col in japan_mesh.columns if 'MESH' in col.upper() and '4' in col]
            if not mesh_id_cols:
                # Look for any ID-like column
                mesh_id_cols = [col for col in japan_mesh.columns if 'ID' in col.upper() or 'CODE' in col.upper()]
            
            if mesh_id_cols:
                log_message(f"Found potential mesh ID columns: {mesh_id_cols}", "info")
            else:
                st.warning("No obvious mesh ID column found in Japan mesh data")
                
            return japan_mesh
        else:
            st.error(f"""
            Japan mesh file not found at any of these locations:
            - {mesh_path}
            - /Users/erick/Documents/GitHub/sip_mss_AnDeS/data/japan_mesh4_CRS84.geojson
            - ./data/japan_mesh4_CRS84.geojson
            - ../data/japan_mesh4_CRS84.geojson
            
            Please make sure the Pegasus32 volume is mounted or copy the mesh file to one of the alternative locations.
            """)
            log_message("Japan mesh file not found at any alternative paths", "error")
            return None
    except Exception as e:
        st.error(f"Error loading Japan mesh: {e}")
        log_message(f"Error loading Japan mesh: {e}", "error")
        return None

def process_uploaded_geojson(uploaded_file):
    """Process uploaded GeoJSON file and return as GeoDataFrame."""
    try:
        if uploaded_file is not None:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()
                
                # Read GeoJSON
                gdf = gpd.read_file(tmp_file.name)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
                # Ensure CRS is set to WGS84 if not specified
                if gdf.crs is None:
                    gdf.set_crs("EPSG:4326", inplace=True)
                
                log_message(f"GeoJSON loaded: {len(gdf)} polygons, CRS: {gdf.crs}", "info")
                return gdf
        return None
    except Exception as e:
        st.error(f"Error processing GeoJSON file: {e}")
        return None

def extract_mesh_ids_from_polygon(polygon_gdf, japan_mesh_gdf):
    """Extract MeshID4 codes from polygon using spatial intersection."""
    try:
        if polygon_gdf is None or japan_mesh_gdf is None:
            return []
        
        # Debug: Log the available columns
        log_message(f"Japan mesh columns: {list(japan_mesh_gdf.columns)}", "info")
        log_message(f"Polygon columns: {list(polygon_gdf.columns)}", "info")
        
        # Ensure both GeoDataFrames have the same CRS
        if polygon_gdf.crs != japan_mesh_gdf.crs:
            polygon_gdf = polygon_gdf.to_crs(japan_mesh_gdf.crs)
        
        # Perform spatial intersection
        intersected = gpd.overlay(japan_mesh_gdf, polygon_gdf, how='intersection')
        
        # Try to find mesh ID column with different possible names
        mesh_id_column = None
        possible_columns = ['MESH4_ID', 'MESH_ID', 'meshcode', 'mesh_id', 'id', 'ID']
        
        for col in possible_columns:
            if col in intersected.columns:
                mesh_id_column = col
                break
        
        if mesh_id_column:
            mesh_ids = intersected[mesh_id_column].astype(str).tolist()
            mesh_ids = [mid for mid in mesh_ids if mid and mid != 'nan']
            log_message(f"Extracted {len(mesh_ids)} mesh IDs from column '{mesh_id_column}'", "info")
            return mesh_ids
        else:
            available_cols = list(intersected.columns)
            st.error(f"No mesh ID column found. Available columns: {available_cols}")
            log_message(f"No mesh ID column found in intersected data. Available columns: {available_cols}", "error")
            return []
            
    except Exception as e:
        st.error(f"Error extracting mesh IDs: {e}")
        log_message(f"Error in extract_mesh_ids_from_polygon: {e}", "error")
        return []

def create_mesh_visualization_plot(mesh_gdf, polygon_gdf=None, population_data=None):
    """Create a matplotlib plot similar to plot_gdf from ntt_dev.py but adapted for Streamlit."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot mesh data
        if population_data is not None and 'population' in mesh_gdf.columns:
            # Plot with population data
            im = mesh_gdf.plot(
                ax=ax,
                column="population",
                cmap="viridis",
                alpha=0.7,
                legend=True,
                legend_kwds={"label": "Population", "shrink": 0.8}
            )
        else:
            # Plot without population data
            mesh_gdf.plot(ax=ax, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        
        # Plot polygon overlay if provided
        if polygon_gdf is not None:
            polygon_gdf.plot(ax=ax, alpha=0.3, color='red', edgecolor='red', linewidth=2)
        
        # Try to add basemap (optional, may fail if contextily not available)
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=mesh_gdf.crs.to_string(),
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution=False,
                alpha=0.8
            )
        except ImportError:
            pass  # Skip basemap if contextily not available
        except Exception:
            pass  # Skip basemap if any other error occurs
        
        ax.set_title("Mesh Visualization with Selected Area")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Set aspect ratio and tight layout
        ax.set_aspect('equal')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

class RealTimeAnomalyDetector:
    """Real-time anomaly detector using core SCAMP functionality."""
    
    def __init__(self):
        self.scamp_detector = None
        self.lazy_db = LazyDatabase()
        
    def _get_detector(self, config):
        """Get or create detector with current configuration."""
        if (self.scamp_detector is None or 
            getattr(self.scamp_detector, 'implementation', None) != config.get('implementation') or
            getattr(self.scamp_detector, 'use_left_mp', None) != config.get('use_left_mp')):
            
            self.scamp_detector = ScampAnomalyDetector(
                window_size=config.get('subsequence_length', 24),
                normalize=config.get('normalize_matrix_profile', False),
                threshold_method=config.get('threshold_method', 'sigma'),
                implementation=config.get('implementation', 'auto'),
                use_left_mp=config.get('use_left_mp', False)
            )
        else:
            # Update parameters that can change without recreating detector
            self.scamp_detector.window_size = config.get('subsequence_length', 24)
            self.scamp_detector.normalize = config.get('normalize_matrix_profile', False)
            self.scamp_detector.threshold_method = config.get('threshold_method', 'sigma')
        
        return self.scamp_detector
    
    def get_implementation_info(self):
        """Get implementation information from the underlying detector."""
        if self.scamp_detector is None:
            # Return default info if no detector is initialized yet
            available_libs = []
            try:
                import pyscamp
                available_libs.append("PySCAMP")
            except ImportError:
                pass
            try:
                import stumpy
                available_libs.append("STUMPY")
            except ImportError:
                pass
            available_libs.append("Custom")
            
            return {
                'selected_implementation': 'auto',
                'available_implementations': available_libs,
                'use_left_mp': False,
                'window_size': 24,
                'normalize': False,
                'threshold_method': 'sigma'
            }
        else:
            return self.scamp_detector.get_implementation_info()
        
    def detect_anomalies(self, data: pd.DataFrame, 
                        subsequence_length: int = 24,
                        threshold_method: str = 'sigma',
                        threshold_multiplier: float = None,
                        normalize: bool = False,
                        implementation: str = 'auto',
                        use_left_mp: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Detect anomalies using the core SCAMP algorithm."""
        start_time = time.time()
        
        try:
            if data.empty or len(data) < subsequence_length:
                log_message("Insufficient data for anomaly detection", "warning")
                return np.array([]), np.array([]), 0.0, 0.0
            
            # Use population or value column
            value_col = 'population' if 'population' in data.columns else 'value'
            values = data[value_col].values
            
            # Note: -1 values are already converted to np.nan during data loading
            # Check for any remaining -1 values (should be none)
            remaining_minus_ones = np.sum(values == -1) if len(values) > 0 else 0
            if remaining_minus_ones > 0:
                log_message(f"Warning: Found {remaining_minus_ones} unexpected -1 values in processed data", "warning")
                values = np.where(values == -1, np.nan, values)
            
            # Create configuration dict
            config = {
                'subsequence_length': subsequence_length,
                'threshold_method': threshold_method,
                'normalize_matrix_profile': normalize,
                'implementation': implementation,
                'use_left_mp': use_left_mp
            }
            
            # Get detector with current configuration
            detector = self._get_detector(config)
            
            # Detect anomalies using core algorithm
            anomalies, scores, threshold_used = detector.detect_anomalies_batch(
                values, custom_multiplier=threshold_multiplier
            )
            
            processing_time = time.time() - start_time
            anomaly_count = np.sum(anomalies) if len(anomalies) > 0 else 0
            
            # Update metrics
            update_metrics(processing_time, len(data), anomaly_count)
            
            # Log implementation info
            impl_info = detector.get_implementation_info()
            log_message(
                f"Anomaly detection complete: {anomaly_count} anomalies found in {processing_time:.2f}s using "
                f"{impl_info['selected_implementation'].upper()} with "
                f"{'left' if impl_info['use_left_mp'] else 'standard'} matrix profile",
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
    min_date, max_date = get_available_date_range()
    
    # Event Date and Time Selection (New Logic)
    st.sidebar.write("**Event Date & Time**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        event_date = st.date_input(
            "Event Date",
            value=datetime(2024, 1, 1).date(),
            min_value=datetime(2016, 1, 1).date(),
            max_value=datetime.now().date(),
            help="Select the date of the event of interest"
        )
    
    with col2:
        event_time = st.time_input(
            "Event Time",
            value=datetime.now().time().replace(second=0, microsecond=0),
            help="Select the time of the event of interest"
        )
    
    # Combine date and time (always create timezone-naive datetime for consistency)
    event_datetime = datetime.combine(event_date, event_time)
    
    # User control for years back
    st.sidebar.subheader("📅 Data Period Selection")
    
    # Calculate available years range
    max_available_years = event_datetime.year - 2016 + 1
    
    years_back_option = st.sidebar.radio(
        "Data extraction period:",
        options=["All available years (2016 onwards)", "Custom years back"],
        help="Choose how many years of data to include in the analysis"
    )
    
    if years_back_option == "Custom years back":
        years_back = st.sidebar.slider(
            "Years back from event year:",
            min_value=1,
            max_value=max_available_years,
            value=min(5, max_available_years),  # Default to 5 years or max available
            help=f"Number of years back from {event_datetime.year} to include (limited by available data since 2016)"
        )
        start_year_display = max(2016, event_datetime.year - years_back + 1)
        analysis_years = list(range(start_year_display, event_datetime.year + 1))
    else:
        years_back = None  # Use all available years
        analysis_years = list(range(2016, event_datetime.year + 1))
    
    # Show analysis info
    st.sidebar.info(f"📅 **Event DateTime:** {event_datetime.strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.info(f"📊 **Analysis:** 1-month periods from {len(analysis_years)} years ({min(analysis_years)}-{max(analysis_years)})")
    
    # Event-based mesh ID selection
    st.sidebar.subheader("🎯 Event Selection")
    
    # Get available events
    available_events = get_available_events()
    event_options = ["Custom Selection"] + [event['event'] for event in available_events]
    
    selected_event_name = st.sidebar.selectbox(
        "Select Event",
        options=event_options,
        index=0,
        help="Choose a predefined event or use custom selection"
    )
    
    # Show different period info based on selection type
    if selected_event_name != "Custom Selection":
        st.sidebar.info(f"🔍 **Monthly Period:** 1 month before + 1 week after event date/time each year")
    else:
        st.sidebar.info(f"🔍 **Monthly Period:** 1 month before until event date/time each year")
    
    # Adjust dates for predefined events
    if selected_event_name != "Custom Selection":
        selected_event = get_event_by_name(selected_event_name)
        if selected_event:
            # Use predefined event date and time (convert to timezone-naive for consistency)
            event_datetime_raw = selected_event['event_dt']
            if hasattr(event_datetime_raw, 'tzinfo') and event_datetime_raw.tzinfo is not None:
                event_datetime = event_datetime_raw.replace(tzinfo=None)
            else:
                event_datetime = event_datetime_raw
            event_date = event_datetime.date()
            event_time = event_datetime.time()
            
            # Note: analysis_years calculation moved below to respect user's years_back selection
            
            st.sidebar.info(f"📅 **Auto-set Event:** {event_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    # Handle event selection
    if selected_event_name != "Custom Selection":
        selected_event = get_event_by_name(selected_event_name)
        if selected_event:
            # Display event information
            event_date = selected_event['event_dt'].strftime("%Y-%m-%d %H:%M")
            st.sidebar.info(f"📅 **Event Date:** {event_date}")
            st.sidebar.info(f"🏢 **Event:** {selected_event['event']}")
            st.sidebar.info(f"📍 **Main Mesh:** {selected_event['meshcode']}")
            st.sidebar.info(f"🗺️ **Total Meshes:** {len(selected_event['meshcodes'])}")
            
            # Use event mesh codes
            mesh_id_list = [str(code) for code in selected_event['meshcodes']]
            
            # Option to use only main mesh or all meshes
            use_all_meshes = st.sidebar.checkbox(
                "Use all event meshes", 
                value=True,
                help="Use all mesh codes for this event or just the main one"
            )
            
            if not use_all_meshes:
                mesh_id_list = [str(selected_event['meshcode'])]
                
            # Show selected meshes
            st.sidebar.write(f"**Selected Mesh IDs:** {', '.join(mesh_id_list[:3])}{'...' if len(mesh_id_list) > 3 else ''}")
            
            # For events, handle multi-mesh analysis logic
            if use_all_meshes and len(mesh_id_list) > 1:
                # Automatically enable multi-mesh analysis for multiple meshes
                multi_mesh_analysis = True
                st.sidebar.info("🔗 **Multi-mesh analysis automatically enabled** for event analysis")
            else:
                # Single mesh selected, no need for multi-mesh analysis
                multi_mesh_analysis = False
            
    else:
        # GeoJSON Polygon Upload Feature
        st.sidebar.subheader("📐 Area Selection")
        
        # Option to select area method
        area_method = st.sidebar.radio(
            "Select Area Method:",
            options=["Upload GeoJSON Polygon", "Manual Mesh IDs"],
            help="Choose how to define the analysis area"
        )
        
        if area_method == "Upload GeoJSON Polygon":
            # GeoJSON file uploader
            uploaded_geojson = st.sidebar.file_uploader(
                "Upload GeoJSON Polygon File", 
                type=["geojson", "json"],
                help="Upload a GeoJSON file containing polygon(s) to define the analysis area"
            )
            
            if uploaded_geojson is not None:
                # Process uploaded GeoJSON
                polygon_gdf = process_uploaded_geojson(uploaded_geojson)
                
                if polygon_gdf is not None:
                    # Check if the uploaded file is already a mesh file (has MESH4_ID column)
                    if 'MESH4_ID' in polygon_gdf.columns:
                        st.sidebar.info("🎯 Detected that uploaded file contains mesh data!")
                        
                        # Use the uploaded file directly as mesh data
                        mesh_ids = polygon_gdf['MESH4_ID'].astype(str).tolist()
                        mesh_ids = [mid for mid in mesh_ids if mid and mid != 'nan']
                        
                        if mesh_ids:
                            mesh_id_list = mesh_ids
                            multi_mesh_analysis = len(mesh_id_list) > 1
                            
                            st.sidebar.success(f"✅ Found {len(mesh_id_list)} mesh regions")
                            st.sidebar.write(f"**Mesh IDs:** {', '.join(mesh_id_list[:5])}{'...' if len(mesh_id_list) > 5 else ''}")
                            
                            # Store data for visualization
                            st.session_state.uploaded_polygon = polygon_gdf
                            st.session_state.extracted_mesh_gdf = polygon_gdf
                        else:
                            st.sidebar.error("❌ No valid mesh IDs found in file")
                            mesh_id_list = []
                            multi_mesh_analysis = False
                    else:
                        # Load Japan mesh data for intersection
                        japan_mesh = load_japan_mesh_source()
                        
                        if japan_mesh is not None:
                            # Extract mesh IDs
                            extracted_mesh_ids = extract_mesh_ids_from_polygon(polygon_gdf, japan_mesh)
                            
                            if extracted_mesh_ids:
                                mesh_id_list = extracted_mesh_ids
                                multi_mesh_analysis = len(mesh_id_list) > 1
                                
                                st.sidebar.success(f"✅ Found {len(mesh_id_list)} mesh regions")
                                st.sidebar.write(f"**Mesh IDs:** {', '.join(mesh_id_list[:5])}{'...' if len(mesh_id_list) > 5 else ''}")
                                
                                # Store polygon data for visualization
                                st.session_state.uploaded_polygon = polygon_gdf
                                
                                # Find mesh ID column for filtering
                                mesh_id_col = None
                                possible_cols = ['MESH4_ID', 'MESH_ID', 'meshcode', 'mesh_id', 'id', 'ID']
                                for col in possible_cols:
                                    if col in japan_mesh.columns:
                                        mesh_id_col = col
                                        break
                                
                                if mesh_id_col:
                                    st.session_state.extracted_mesh_gdf = japan_mesh[japan_mesh[mesh_id_col].astype(str).isin(mesh_id_list)]
                                else:
                                    st.session_state.extracted_mesh_gdf = japan_mesh
                            else:
                                st.sidebar.error("❌ No mesh regions found in the polygon area")
                                mesh_id_list = []
                                multi_mesh_analysis = False
                        else:
                            st.sidebar.error("❌ Could not load Japan mesh data")
                            mesh_id_list = []
                            multi_mesh_analysis = False
                else:
                    st.sidebar.error("❌ Could not process GeoJSON file")
                    mesh_id_list = []
                    multi_mesh_analysis = False
            else:
                st.sidebar.info("📁 Please upload a GeoJSON file to define the analysis area")
                mesh_id_list = []
                multi_mesh_analysis = False
        else:
            # Manual mesh ID selection (original behavior)
            available_mesh_ids = get_available_mesh_ids()
            selected_mesh = st.sidebar.selectbox("Select Mesh ID Range", available_mesh_ids)
            
            # Custom mesh ID input
            st.sidebar.subheader("🔢 Custom Mesh IDs")
            custom_mesh_input = st.sidebar.text_area(
                "Input Mesh ID Range", 
                placeholder="Enter mesh IDs separated by commas\nExample: 533937621, 533946403, 533947534",
                help="Enter one or more mesh ID codes separated by commas"
            )
            
            # Parse custom mesh IDs
            if custom_mesh_input.strip():
                mesh_id_list = [mesh.strip() for mesh in custom_mesh_input.split(',') if mesh.strip()]
            else:
                # Use selected mesh as single item list
                mesh_id_list = [selected_mesh]
            
            # Multi-mesh analysis option for custom selection
            multi_mesh_analysis = st.sidebar.checkbox("Multi-mesh analysis", value=False, 
                                                     help="Aggregate data from all provided mesh IDs")
    
    # Detection parameters
    st.sidebar.subheader("🔧 Detection Parameters")
    subsequence_length = st.sidebar.slider("Subsequence Length (hours)", 3, 24, 3)
    
    # Warm-up period slider - between subsequence_length and 8 * subsequence_length
    warm_up_period = st.sidebar.slider(
        "Warm-up Period (hours)", 
        subsequence_length, 
        8 * subsequence_length, 
        subsequence_length,
        help="Initial data points to exclude from anomaly flagging due to matrix profile warm-up"
    )
    
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
    
    # Matrix Profile Implementation Selection
    st.sidebar.subheader("⚙️ Matrix Profile Implementation")
    
    # Get available implementations
    detector = get_detector()
    impl_info = detector.get_implementation_info()
    available_impls = impl_info['available_implementations']
    
    # Create options with descriptions
    impl_options = []
    impl_descriptions = {
        'auto': '🤖 Auto (Best Available)',
        'pyscamp': '🚀 PySCAMP (GPU Accelerated)',
        'stumpy': '🌟 STUMPY (Streaming Support)',
        'custom': '🛠️ Custom (Pure Python)'
    }
    
    for impl in ['auto', 'pyscamp', 'stumpy', 'custom']:
        if impl == 'auto':
            impl_options.append(impl_descriptions[impl])
        elif impl.upper() in [lib.upper() for lib in available_impls]:
            impl_options.append(impl_descriptions[impl])
    
    selected_impl_display = st.sidebar.selectbox(
        "Matrix Profile Library",
        options=impl_options,
        index=0,
        help="Choose which matrix profile implementation to use"
    )
    
    # Extract actual implementation name
    selected_implementation = selected_impl_display.split(' ')[1].lower()
    
    # Left Matrix Profile option
    use_left_mp = st.sidebar.checkbox(
        "Use Left Matrix Profile",
        value=True,
        help="Only consider past data for nearest neighbor search (prevents future data from updating past anomaly scores)"
    )
    
    # Show info about left matrix profile
    if use_left_mp:
        st.sidebar.info("🔒 **Left Matrix Profile Mode**: Only uses historical data for anomaly detection. This prevents future data from changing past anomaly classifications.")
    
    # Show selected configuration summary
    st.sidebar.subheader("📋 Configuration Summary")
    
    # Create mesh analysis description
    if selected_event_name != "Custom Selection":
        mesh_description = f"Event: {len(mesh_id_list)} meshes"
        if multi_mesh_analysis:
            mesh_description += " (aggregated)"
        else:
            mesh_description += " (single)"
    else:
        mesh_description = f"Custom: {len(mesh_id_list)} mesh{'es' if len(mesh_id_list) > 1 else ''}"
        if multi_mesh_analysis and len(mesh_id_list) > 1:
            mesh_description += " (aggregated)"
    
    config_summary = f"""
    **Event DateTime:** {event_datetime.strftime('%Y-%m-%d %H:%M')}
    **Analysis Years:** {len(analysis_years)} years ({min(analysis_years)}-{max(analysis_years)})
    **Library:** {selected_impl_display.split(' ')[1]}
    **Matrix Profile:** {'Left' if use_left_mp else 'Standard'}
    **Window Size:** {subsequence_length} hours
    **Warm-up Period:** {warm_up_period} hours
    **Threshold:** {threshold_method}
    **Mesh Analysis:** {mesh_description}
    """
    st.sidebar.text(config_summary)
    
    # Store configuration in session state
    st.session_state.config = {
        'event_datetime': event_datetime,
        'analysis_years': analysis_years,
        'years_back': years_back,  # Add the years_back parameter
        # Note: Always use custom analysis logic (is_actual_event=False) to respect years_back setting
        # Both Event Selection and Custom Selection should look back multiple years if requested
        'is_actual_event': False,  # Always use custom analysis logic to respect years_back setting
        'is_predefined_event': selected_event_name != "Custom Selection",  # True for predefined events
        'mesh_id_list': mesh_id_list,
        'multi_mesh_analysis': multi_mesh_analysis,
        'subsequence_length': subsequence_length,
        'warm_up_period': warm_up_period,
        'threshold_method': threshold_method,
        'threshold_multiplier': threshold_multiplier,
        'normalize_matrix_profile': normalize_matrix_profile,
        'implementation': selected_implementation,
        'use_left_mp': use_left_mp
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
        with st.spinner("Loading MSS monthly time series data..."):
            log_message(f"Loading monthly time series data for event: {config['event_datetime']} - {config['mesh_id_list']}")
            data = load_mss_data_monthly_timeseries(
                event_datetime=config['event_datetime'],
                mesh_id_list=config['mesh_id_list'],
                multi_mesh_analysis=config['multi_mesh_analysis'],
                years_back=config.get('years_back', None),  # Use years_back if available
                is_actual_event=config.get('is_actual_event', False),  # Pass event type
                is_predefined_event=config.get('is_predefined_event', False)  # Pass predefined event flag
            )
            
            if not data.empty:
                # Convert -1 values to np.nan for proper handling throughout the app
                value_col = 'population' if 'population' in data.columns else 'value'
                original_minus_ones = (data[value_col] == -1).sum()
                data[value_col] = data[value_col].replace(-1, np.nan)
                log_message(f"Data preprocessing: converted {original_minus_ones} instances of -1 to np.nan for proper visualization and analysis", "info")
                
                st.session_state.current_data = data
                st.session_state.data_loaded = True
                
                # Enhanced success message with mesh and time series information
                mesh_info = ""
                if config['multi_mesh_analysis'] and len(config['mesh_id_list']) > 1:
                    mesh_info = f" from {len(config['mesh_id_list'])} aggregated meshes"
                elif len(config['mesh_id_list']) > 1:
                    mesh_info = f" from first of {len(config['mesh_id_list'])} meshes"
                else:
                    mesh_info = f" from mesh {config['mesh_id_list'][0]}"
                
                # Add time series info
                years_with_data = data['year_source'].nunique() if 'year_source' in data.columns else len(config['analysis_years'])
                time_span = f"{data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}"
                
                success_msg = f"Successfully loaded {len(data)} data points{mesh_info} from {years_with_data} yearly periods"
                log_message(success_msg)
                log_message(f"Time series spans: {time_span}")
                st.success(success_msg)
                st.info(f"📊 **Time Series:** {time_span} ({years_with_data} yearly monthly periods)")
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
        sample_rows_option = st.selectbox(
            "Sample Rows to Display",
            ["10", "25", "50", "100", "500", "All Data"],
            index=0,
            help="Number of rows to show in sample data table"
        )
        
        # Convert to integer or set to all data
        if sample_rows_option == "All Data":
            sample_rows = len(data)
        else:
            sample_rows = int(sample_rows_option)
    
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
    
    # Add enhanced data visualization with year-by-year plots
    st.subheader("📊 Data Preview Visualization")
    
    # Check if we have year_source data for separate year plotting
    if 'year_source' in filtered_data.columns:
        st.write("**Individual Year Analysis:**")
        
        # Create year-by-year visualization
        available_years = sorted(filtered_data['year_source'].unique())
        
        # Option to select which years to display
        selected_years_for_viz = st.multiselect(
            "Select years to visualize",
            options=available_years,
            default=available_years[:3] if len(available_years) > 3 else available_years,
            help="Select which years to display in the visualization"
        )
        
        if selected_years_for_viz:
            # Create single plot with all years on same axis
            fig_years = go.Figure()
            
            # Determine value column
            value_col = 'population' if 'population' in filtered_data.columns else 'value'
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, year in enumerate(selected_years_for_viz):
                year_data = filtered_data[filtered_data['year_source'] == year].copy()
                
                # Create a relative time axis for better comparison (hours from start of period)
                if not year_data.empty:
                    year_data = year_data.sort_values('timestamp')
                    start_time = year_data['timestamp'].min()
                    year_data['relative_time'] = (year_data['timestamp'] - start_time).dt.total_seconds() / 3600  # Hours from start
                    
                    fig_years.add_trace(
                        go.Scatter(
                            x=year_data['relative_time'],
                            y=year_data[value_col],
                            mode='lines+markers',
                            name=f'Year {year}',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=3),
                            showlegend=True
                        )
                    )
            
            fig_years.update_layout(
                height=400,
                title=f"{value_col.title()} Over Time - Year by Year Comparison (Same Axis)",
                xaxis_title="Hours from Period Start",
                yaxis_title=value_col.title(),
                showlegend=True
            )
            
            st.plotly_chart(fig_years, width='stretch')
        
        # Combined view toggle
        show_combined = st.checkbox("Show Combined View", value=True)
        
        if show_combined:
            st.write("**Combined Time Series View:**")
    else:
        # Fallback for data without year_source
        show_combined = True
    
    if show_combined or 'year_source' not in filtered_data.columns:
        # Determine value column
        value_col = 'population' if 'population' in filtered_data.columns else 'value'
        sample_viz_data = filtered_data.head(sample_rows)
        
        fig_preview = go.Figure()
        
        # Create sequential plotting to avoid gaps between years
        if 'year_source' in filtered_data.columns:
            # Sort data and create sequential index
            sorted_data = sample_viz_data.sort_values(['year_source', 'timestamp']).reset_index(drop=True)
            colors = px.colors.qualitative.Set1
            year_colors = {year: colors[i % len(colors)] for i, year in enumerate(sorted_data['year_source'].unique())}
            
            # Create sequential x-axis (0, 1, 2, 3, ...)
            sequential_x = list(range(len(sorted_data)))
            
            # Create datetime labels for hover/display
            datetime_labels = sorted_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Plot by year with sequential x but datetime hover
            for year in sorted_data['year_source'].unique():
                year_mask = sorted_data['year_source'] == year
                year_indices = [i for i, mask in enumerate(year_mask) if mask]
                year_values = sorted_data[year_mask][value_col]
                year_datetime_labels = sorted_data[year_mask]['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                fig_preview.add_trace(go.Scatter(
                    x=year_indices,
                    y=year_values,
                    mode='lines+markers',
                    name=f'Year {year}',
                    line=dict(color=year_colors[year], width=2),
                    marker=dict(size=4),
                    customdata=year_datetime_labels,
                    hovertemplate='<b>Year %{fullData.name}</b><br>Sequential Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
                ))
            
            # Add vertical lines between years
            year_boundaries = []
            current_index = 0
            for year in sorted(sorted_data['year_source'].unique()):
                year_count = (sorted_data['year_source'] == year).sum()
                if current_index > 0:  # Don't add line at the beginning
                    year_boundaries.append(current_index)
                current_index += year_count
            
            # Add vertical lines at year boundaries
            for boundary in year_boundaries:
                fig_preview.add_vline(
                    x=boundary,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.7,
                    annotation_text=f"Year boundary"
                )
            
            # Create custom x-axis labels (every few points show the datetime)
            tick_interval = max(1, len(sorted_data) // 10)  # Show ~10 labels
            tick_positions = list(range(0, len(sorted_data), tick_interval))
            tick_labels = [datetime_labels.iloc[i] if i < len(datetime_labels) else "" for i in tick_positions]
            
            fig_preview.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=tick_positions,
                    ticktext=tick_labels,
                    tickangle=45
                )
            )
            
        else:
            # Fallback for data without year_source
            sequential_x = list(range(len(sample_viz_data)))
            datetime_labels = sample_viz_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            fig_preview.add_trace(go.Scatter(
                x=sequential_x,
                y=sample_viz_data[value_col],
                mode='lines+markers',
                name=f'{value_col.title()} (Sample)',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                customdata=datetime_labels,
                hovertemplate='<b>Sequential Data</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
            ))
        
        fig_preview.update_layout(
            title=f"{value_col.title()} Over Time (Combined Sequential View)",
            xaxis_title="Sequential Time Index",
            yaxis_title=value_col.title(),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_preview, width='stretch')
    
    # Calculate and display enhanced metrics
    st.subheader("📈 Data Statistics")
    col1, col2, col3 = st.columns(3)
    
    # Calculate no data entries (missing data or NaN values)
    no_data_count = 0
    if value_col in filtered_data.columns:
        # Count NaN values as "no data" (previously -1 values, now converted)
        no_data_count = filtered_data[value_col].isna().sum()
    
    with col1:
        if value_col in filtered_data.columns:
            # Use nanmin and nanmax to ignore NaN values (previously -1)
            min_val = np.nanmin(filtered_data[value_col])
            max_val = np.nanmax(filtered_data[value_col])
            st.metric("📊 Population Range", 
                     f"{min_val:.0f} - {max_val:.0f}")
        else:
            min_val = np.nanmin(filtered_data[value_col])
            max_val = np.nanmax(filtered_data[value_col])
            st.metric("📊 Value Range", 
                     f"{min_val:.2f} - {max_val:.2f}")
    
    with col2:
        time_range = filtered_data['timestamp'].max() - filtered_data['timestamp'].min()
        st.metric("⏱️ Time Span", f"{time_range.days} days")
    
    with col3:
        st.metric("❌ No Data Entries", 
                 f"{no_data_count} ({(no_data_count/len(filtered_data)*100):.1f}%)")
    
    st.divider()
    
    # Warm-up period validation
    warm_up_period = config.get('warm_up_period', config['subsequence_length'])
    data_length = len(filtered_data)
    
    if warm_up_period >= data_length:
        st.error(f"⚠️ Warm-up period ({warm_up_period} hours) is larger than available data ({data_length} points). "
                f"Please reduce the warm-up period to less than {data_length} hours.")
        return
    elif warm_up_period > data_length * 0.5:
        st.warning(f"⚠️ Warm-up period ({warm_up_period} hours) is quite large compared to available data ({data_length} points). "
                  f"Consider reducing it for more effective anomaly detection.")
    
    # Display warm-up period info and analysis approach
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🔧 **Warm-up Period:** {warm_up_period} hours")
    with col2:
        st.info(f"📊 **Analysis Period:** {data_length - warm_up_period} hours")
    with col3:
        effective_rate = ((data_length - warm_up_period) / data_length) * 100
        st.info(f"📈 **Effective Coverage:** {effective_rate:.1f}%")
    
    # Analysis approach information
    st.info("""
    **🔬 Analysis Approach:** The matrix profile algorithm (STUMPY/SCAMP) analyzes the time series as 
    **sequential numerical data** without using datetime information directly. The datetime stamps are only 
    used for visualization and filtering - the core anomaly detection works on the ordered sequence of values, 
    making it robust to irregular time intervals and focused on pattern detection.
    """)
    
    # Run detection
    if st.button("🔍 Run Anomaly Detection"):
        with st.spinner("Running matrix profile anomaly detection..."):
            detector = get_detector()
            # Use filtered data for anomaly detection
            anomalies, scores, used_threshold, processing_time = detector.detect_anomalies(
                filtered_data, 
                subsequence_length=config['subsequence_length'],
                threshold_method=config['threshold_method'],
                threshold_multiplier=config['threshold_multiplier'],
                normalize=config['normalize_matrix_profile'],
                implementation=config.get('implementation', 'auto'),
                use_left_mp=config.get('use_left_mp', False)
            )
            
            if len(anomalies) > 0:
                # Create a fresh copy to avoid data reference issues
                results_data = filtered_data.copy()
                results_data['detected_anomaly'] = anomalies
                results_data['anomaly_score'] = scores
                
                # Apply warm-up period logic - mark warm-up anomalies separately
                results_data['anomaly_in_warmup'] = False
                results_data['anomaly_after_warmup'] = False
                
                # Separate anomalies by warm-up period
                warm_up_mask = results_data.index < warm_up_period
                
                # Mark anomalies in warm-up period
                warmup_anomalies = results_data['detected_anomaly'] & warm_up_mask
                results_data.loc[warmup_anomalies, 'anomaly_in_warmup'] = True
                
                # Mark anomalies after warm-up period (these are the "real" anomalies)
                effective_anomalies = results_data['detected_anomaly'] & ~warm_up_mask
                results_data.loc[effective_anomalies, 'anomaly_after_warmup'] = True
                
                # Count anomalies
                warmup_anomaly_count = warmup_anomalies.sum()
                effective_anomaly_count = effective_anomalies.sum()
                total_anomaly_count = results_data['detected_anomaly'].sum()
                
                # Store fresh results in session state with configuration
                st.session_state.detection_results = results_data
                st.session_state.detection_threshold = used_threshold
                st.session_state.analysis_date_range = (start_date, end_date)  # Store date range
                st.session_state.detection_config = config.copy()  # Store config used for this detection
                st.session_state.last_processing_time = processing_time  # Store processing time
                
                log_message(f"Detection complete: {total_anomaly_count} total anomalies found "
                           f"({warmup_anomaly_count} in warm-up, {effective_anomaly_count} after warm-up) in date range")
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
                config['warm_up_period'] != detection_config.get('warm_up_period') or
                config['normalize_matrix_profile'] != detection_config.get('normalize_matrix_profile') or
                config.get('implementation') != detection_config.get('implementation') or
                config.get('use_left_mp') != detection_config.get('use_left_mp')
            )
            
            # Special handling for warm-up period changes
            warmup_changed = config['warm_up_period'] != detection_config.get('warm_up_period')
            
            if config_changed:
                if warmup_changed:
                    st.warning("⚠️ **Warm-up Period Changed:** The displayed results were generated with a different warm-up period. "
                              "Run anomaly detection again to apply the new warm-up period to the analysis.")
                    
                    # Offer dynamic warm-up period adjustment
                    if st.button("🔄 Quick Update: Re-apply Warm-up Period"):
                        with st.spinner("Re-applying warm-up period to existing results..."):
                            results_data = st.session_state.detection_results.copy()
                            new_warm_up_period = config['warm_up_period']
                            
                            # Reset warm-up columns
                            results_data['anomaly_in_warmup'] = False
                            results_data['anomaly_after_warmup'] = False
                            
                            # Re-apply warm-up logic with new period
                            warm_up_mask = results_data.index < new_warm_up_period
                            
                            # Mark anomalies in warm-up period
                            warmup_anomalies = results_data['detected_anomaly'] & warm_up_mask
                            results_data.loc[warmup_anomalies, 'anomaly_in_warmup'] = True
                            
                            # Mark anomalies after warm-up period (these are the "real" anomalies)
                            effective_anomalies = results_data['detected_anomaly'] & ~warm_up_mask
                            results_data.loc[effective_anomalies, 'anomaly_after_warmup'] = True
                            
                            # Update session state with new analysis
                            st.session_state.detection_results = results_data
                            
                            # Update config to match current settings for warm-up period
                            updated_config = st.session_state.detection_config.copy()
                            updated_config['warm_up_period'] = new_warm_up_period
                            st.session_state.detection_config = updated_config
                            
                            # Count for log message
                            warmup_count = warmup_anomalies.sum()
                            effective_count = effective_anomalies.sum()
                            
                            log_message(f"Warm-up period updated to {new_warm_up_period} hours: "
                                       f"{warmup_count} warm-up anomalies, {effective_count} effective anomalies", "info")
                            
                            st.success(f"✅ Warm-up period updated to {new_warm_up_period} hours!")
                            st.rerun()
                else:
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
    
def create_anomaly_heatmap(data: pd.DataFrame):
    """Create a heatmap visualization with improved color scheme - white for normal data."""
    try:
        if data.empty or 'detected_anomaly' not in data.columns:
            st.warning("No anomaly data available for heatmap")
            return
        
        # Get anomaly data
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        # Create array for anomaly visualization
        unique_dates = sorted(data_copy['date'].unique())
        
        if len(unique_dates) == 0:
            st.warning("No date data available for heatmap")
            return
        
        # Check if we have year_source for year-separated heatmaps
        if 'year_source' in data_copy.columns:
            st.write("**Anomaly Heatmaps by Year:**")
            
            years = sorted(data_copy['year_source'].unique())
            
            # Option to select years for heatmap display
            selected_years_heatmap = st.multiselect(
                "Select years for heatmap display",
                options=years,
                default=years[:2] if len(years) > 2 else years,
                help="Select which years to display in heatmaps"
            )
            
            for year in selected_years_heatmap:
                st.write(f"**Year {year} Anomaly Heatmap:**")
                
                year_data = data_copy[data_copy['year_source'] == year]
                year_dates = sorted(year_data['date'].unique())
                
                if len(year_dates) == 0:
                    st.write(f"No data available for year {year}")
                    continue
                
                # Create matrix for this year: rows = days, columns = hours (0-23)
                anomaly_matrix = np.zeros((len(year_dates), 24))
                
                # Fill the matrix: 0 = no data, 1 = normal, 2 = anomaly
                for i, date in enumerate(year_dates):
                    day_data = year_data[year_data['date'] == date]
                    for _, row in day_data.iterrows():
                        hour = row['hour']
                        if 0 <= hour <= 23:  # Valid hour range
                            if row['detected_anomaly']:
                                anomaly_matrix[i, hour] = 2  # Anomaly detected
                            else:
                                anomaly_matrix[i, hour] = 1  # Normal data
                
                # Create the matplotlib figure for this year
                fig, ax = plt.subplots(figsize=(12, max(4, len(year_dates) * 0.2)))
                
                # Custom colormap: 0=gray (no data), 1=white (normal), 2=red (anomaly)
                from matplotlib.colors import ListedColormap
                colors = ['lightgray', 'white', 'red']  # 0, 1, 2 respectively
                custom_cmap = ListedColormap(colors)
                
                im = ax.imshow(anomaly_matrix, cmap=custom_cmap, aspect='auto', 
                              interpolation='nearest', vmin=0, vmax=2)
                
                # Set up y-axis (dates)
                if len(year_dates) <= 20:
                    ax.set_yticks(np.arange(len(year_dates)))
                    ax.set_yticklabels([f"{date}" for date in year_dates], fontsize=8)
                else:
                    # For many dates, show only every nth date
                    step = max(1, len(year_dates) // 20)
                    ticks = np.arange(0, len(year_dates), step)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([f"{year_dates[i]}" for i in ticks], fontsize=8)
                
                # Set up x-axis (hours)
                ax.set_xticks(np.arange(24))
                ax.set_xticklabels([f"{i:02d}" for i in range(24)], rotation=45, fontsize=8)
                
                # Add grid lines
                for i in range(1, 24):
                    ax.axvline(i-0.5, color='black', linewidth=0.2, alpha=0.3)
                for i in range(1, len(year_dates)):
                    ax.axhline(i-0.5, color='black', linewidth=0.2, alpha=0.3)
                
                # Labels and title
                ax.set_xlabel('Hour of Day', fontsize=10)
                ax.set_ylabel('Date', fontsize=10)
                ax.set_title(f'Anomaly Detection Heatmap - Year {year}\n(White=Normal, Red=Anomaly, Gray=No Data)', 
                           fontsize=12, pad=20)
                
                # Create custom legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='white', edgecolor='black', label='Normal'),
                    Patch(facecolor='red', edgecolor='black', label='Anomaly'),
                    Patch(facecolor='lightgray', edgecolor='black', label='No Data')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        else:
            # Original combined heatmap for data without year separation
            st.write("**Combined Anomaly Heatmap:**")
            
            # Create a matrix: rows = days, columns = hours (0-23)
            anomaly_matrix = np.zeros((len(unique_dates), 24))
            
            # Fill the matrix with anomaly data
            for i, date in enumerate(unique_dates):
                day_data = data_copy[data_copy['date'] == date]
                for _, row in day_data.iterrows():
                    hour = row['hour']
                    if 0 <= hour <= 23:  # Valid hour range
                        if row['detected_anomaly']:
                            anomaly_matrix[i, hour] = 2  # Anomaly detected
                        else:
                            anomaly_matrix[i, hour] = 1  # Normal data
            
            # Create the matplotlib figure
            fig, ax = plt.subplots(figsize=(12, max(6, len(unique_dates) * 0.25)))
            
            # Custom colormap: 0=gray (no data), 1=white (normal), 2=red (anomaly)
            from matplotlib.colors import ListedColormap
            colors = ['lightgray', 'white', 'red']
            custom_cmap = ListedColormap(colors)
            
            im = ax.imshow(anomaly_matrix, cmap=custom_cmap, aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=2)
            
            # Set up y-axis (dates) - limit to reasonable number
            if len(unique_dates) <= 30:
                ax.set_yticks(np.arange(len(unique_dates)))
                ax.set_yticklabels([f"{date}" for date in unique_dates], fontsize=8)
            else:
                # For many dates, show only every nth date
                step = max(1, len(unique_dates) // 30)
                ticks = np.arange(0, len(unique_dates), step)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{unique_dates[i]}" for i in ticks], fontsize=8)
            
            # Set up x-axis (hours)
            ax.set_xticks(np.arange(24))
            ax.set_xticklabels([f"{i:02d}" for i in range(24)], rotation=45, fontsize=8)
            
            # Add grid lines
            for i in range(1, 24):
                ax.axvline(i-0.5, color='black', linewidth=0.2, alpha=0.3)
            for i in range(1, len(unique_dates)):
                ax.axhline(i-0.5, color='black', linewidth=0.2, alpha=0.3)
            
            # Labels and title
            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Date', fontsize=10)
            ax.set_title('Anomaly Detection Heatmap\n(White=Normal, Red=Anomaly, Gray=No Data)', 
                       fontsize=12, pad=20)
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Normal'),
                Patch(facecolor='red', edgecolor='black', label='Anomaly'),
                Patch(facecolor='lightgray', edgecolor='black', label='No Data')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Horizontal lines at 0.5 of each day
        for i in range(1, len(unique_dates)):
            ax.axhline(i-0.5, color='black', linewidth=0.3)
        
        # Labels and title
        ax.set_xlabel("Hour Frame of the day (e.g. 08 means 08:00 to 08:59)")
        ax.set_ylabel("Date")
        
        # Create title with mesh info if available
        mesh_info = ""
        if 'mesh_id' in data.columns and not data['mesh_id'].empty:
            mesh_id = data['mesh_id'].iloc[0]
            mesh_info = f" - Mesh ID: {mesh_id}"
        
        date_range = f"{unique_dates[0]} to {unique_dates[-1]}"
        ax.set_title(f"Anomaly Detection Heatmap{mesh_info}\n{date_range}")
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['No Data', 'Normal', 'Anomaly'])
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)  # Clean up to avoid memory leaks
        
        # Display summary statistics
        total_hours = np.sum(anomaly_matrix > 0)
        anomaly_hours = np.sum(anomaly_matrix == 2)
        if total_hours > 0:
            anomaly_rate = (anomaly_hours / total_hours) * 100
            st.info(f"📊 **Heatmap Summary:** {anomaly_hours} anomaly hours out of {total_hours} total hours ({anomaly_rate:.1f}% anomaly rate)")
        
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        st.write("Debug info:", str(e))

def create_anomaly_score_heatmap(data: pd.DataFrame):
    """Create a heatmap visualization showing anomaly scores as continuous values."""
    try:
        if data.empty or 'anomaly_score' not in data.columns:
            st.warning("No anomaly score data available for heatmap")
            return
        
        # Get anomaly data
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        # Create array for anomaly score visualization
        # Get unique dates and sort them
        unique_dates = sorted(data_copy['date'].unique())
        
        if len(unique_dates) == 0:
            st.warning("No date data available for score heatmap")
            return
        
        # Create a matrix: rows = days, columns = hours (0-23)
        # Initialize with NaN to represent no data
        score_matrix = np.full((len(unique_dates), 24), np.nan)
        
        # Fill the matrix with anomaly scores
        for i, date in enumerate(unique_dates):
            day_data = data_copy[data_copy['date'] == date]
            for _, row in day_data.iterrows():
                hour = row['hour']
                if 0 <= hour <= 23:  # Valid hour range
                    score_matrix[i, hour] = row['anomaly_score']
        
        # Create the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(unique_dates) * 0.3)))
        
        # Use viridis colormap for continuous score values
        # Mask NaN values to show them as white/transparent
        masked_matrix = np.ma.masked_where(np.isnan(score_matrix), score_matrix)
        
        im = ax.imshow(masked_matrix, cmap='viridis', aspect=0.7, interpolation='nearest')
        
        # Set up y-axis (dates)
        ax.set_yticks(np.arange(len(unique_dates)))
        ax.set_yticklabels([f"{date}" for date in unique_dates])
        
        # Set up x-axis (hours)
        ax.set_xticks(np.arange(24))
        ax.set_xticklabels([f"{i:02d}" for i in range(24)], rotation=45)
        
        # Add grid lines
        # Vertical lines at 0.5 of each hour
        for i in range(1, 24):
            ax.axvline(i-0.5, color='black', linewidth=0.3)
        
        # Horizontal lines at 0.5 of each day
        for i in range(1, len(unique_dates)):
            ax.axhline(i-0.5, color='black', linewidth=0.3)
        
        # Labels and title
        ax.set_xlabel("Hour Frame of the day (e.g. 08 means 08:00 to 08:59)")
        ax.set_ylabel("Date")
        
        # Create title with mesh info if available
        mesh_info = ""
        if 'mesh_id' in data.columns and not data['mesh_id'].empty:
            mesh_id = data['mesh_id'].iloc[0]
            mesh_info = f" - Mesh ID: {mesh_id}"
        
        date_range = f"{unique_dates[0]} to {unique_dates[-1]}"
        ax.set_title(f"Anomaly Score Heatmap{mesh_info}\n{date_range}")
        
        # Add colorbar with continuous scale
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
        
        # Add threshold line if available
        if 'detection_threshold' in st.session_state and st.session_state.detection_threshold is not None:
            threshold = st.session_state.detection_threshold
            # Add threshold information to the plot
            ax.text(0.02, 0.98, f'Threshold: {threshold:.2f}', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)  # Clean up to avoid memory leaks
        
        # Display summary statistics for scores
        valid_scores = score_matrix[~np.isnan(score_matrix)]
        if len(valid_scores) > 0:
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            mean_score = np.mean(valid_scores)
            
            # Count scores above threshold if threshold is available
            threshold_info = ""
            if 'detection_threshold' in st.session_state and st.session_state.detection_threshold is not None:
                threshold = st.session_state.detection_threshold
                above_threshold = np.sum(valid_scores > threshold)
                threshold_info = f" | {above_threshold} scores above threshold ({threshold:.2f})"
            
            st.info(f"📊 **Score Heatmap Summary:** Min: {min_score:.2f}, Max: {max_score:.2f}, Mean: {mean_score:.2f}{threshold_info}")
        
    except Exception as e:
        st.error(f"Error creating score heatmap: {e}")
        st.write("Debug info:", str(e))

def display_detection_results(data: pd.DataFrame):
    """Display anomaly detection results with warm-up period separation and year-by-year analysis."""
    if data.empty:
        st.warning("No data to display")
        return
    
    # Ensure we have the required columns
    if 'detected_anomaly' not in data.columns or 'anomaly_score' not in data.columns:
        st.error("Detection results not available. Run anomaly detection first.")
        return
    
    # Check if warm-up period columns exist
    has_warmup_cols = 'anomaly_in_warmup' in data.columns and 'anomaly_after_warmup' in data.columns
    
    # Get warm-up period from config
    config = st.session_state.get('detection_config', {})
    warm_up_period = config.get('warm_up_period', config.get('subsequence_length', 3))
    
    # Year-by-year analysis if year_source is available
    if 'year_source' in data.columns:
        st.subheader("📊 Year-by-Year Anomaly Analysis")
        
        years = sorted(data['year_source'].unique())
        year_stats = []
        
        for year in years:
            year_data = data[data['year_source'] == year]
            total_points = len(year_data)
            total_anomalies = year_data['detected_anomaly'].sum()
            if has_warmup_cols:
                effective_anomalies = year_data['anomaly_after_warmup'].sum()
            else:
                effective_anomalies = total_anomalies
            
            max_score = year_data['anomaly_score'].max()
            avg_score = year_data['anomaly_score'].mean()
            
            year_stats.append({
                'Year': year,
                'Total Points': total_points,
                'Total Anomalies': total_anomalies,
                'Effective Anomalies': effective_anomalies,
                'Detection Rate (%)': (total_anomalies / total_points * 100) if total_points > 0 else 0,
                'Max Score': max_score,
                'Avg Score': avg_score
            })
        
        # Display year-by-year table
        year_df = pd.DataFrame(year_stats)
        st.dataframe(year_df, width='stretch')
        
        # Year-by-year visualization
        st.write("**Anomaly Distribution by Year:**")
        
        fig_years = go.Figure()
        
        fig_years.add_trace(go.Bar(
            x=year_df['Year'],
            y=year_df['Total Anomalies'],
            name='Total Anomalies',
            marker_color='red',
            opacity=0.7
        ))
        
        if has_warmup_cols:
            fig_years.add_trace(go.Bar(
                x=year_df['Year'],
                y=year_df['Effective Anomalies'],
                name='Effective Anomalies',
                marker_color='darkred'
            ))
        
        fig_years.update_layout(
            title="Anomaly Count by Year",
            xaxis_title="Year",
            yaxis_title="Number of Anomalies",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig_years, width='stretch')
        
        st.divider()
    
    # Overall metrics with warm-ßup separation
    st.subheader("📈 Overall Detection Metrics")
    
    if has_warmup_cols:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_anomalies = data['detected_anomaly'].sum()
        warmup_anomalies = data['anomaly_in_warmup'].sum()
        effective_anomalies = data['anomaly_after_warmup'].sum()
        detection_rate = total_anomalies / len(data) * 100
        effective_rate = effective_anomalies / (len(data) - warm_up_period) * 100 if len(data) > warm_up_period else 0
        max_score = data['anomaly_score'].max()
        avg_score = data['anomaly_score'].mean()
        
        with col1:
            st.metric("🚨 Total Anomalies", total_anomalies)
        with col2:
            st.metric("🔧 Warm-up Anomalies", warmup_anomalies, help="Anomalies in warm-up period (may be false positives)")
        with col3:
            st.metric("✅ Effective Anomalies", effective_anomalies, help="Anomalies after warm-up period")
        with col4:
            st.metric("📊 Effective Rate", f"{effective_rate:.1f}%", help="Detection rate excluding warm-up period")
        with col5:
            st.metric("⚡ Max Score", f"{max_score:.2f}")
    else:
        # Fallback to original metrics if warm-up columns don't exist
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
    
    # Time series plot - split by year to avoid gaps
    st.subheader("📈 Time Series with Anomalies")
    
    # Check if we have year data to split the plots
    if 'year_source' in data.columns:
        # Create separate plots for each year
        available_years = sorted(data['year_source'].unique())
        
        # Option to select which years to display
        selected_years_anomaly = st.multiselect(
            "Select years to display in anomaly plots:",
            options=available_years,
            default=available_years,
            help="Choose which years to show in the anomaly detection plots"
        )
        
        if selected_years_anomaly:
            for year in selected_years_anomaly:
                st.write(f"**Year {year} Anomaly Detection Results:**")
                year_data = data[data['year_source'] == year].copy()
                
                if year_data.empty:
                    st.warning(f"No data available for year {year}")
                    continue
                
                # Create sequential x-axis for this year to avoid timestamp gaps
                year_data = year_data.sort_values('timestamp').reset_index(drop=True)
                sequential_x = list(range(len(year_data)))
                datetime_labels = year_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                
                fig_year = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f'Population Data - Year {year}', f'Anomaly Scores - Year {year}'),
                    vertical_spacing=0.1
                )
                
                # Determine value column
                value_col = 'population' if 'population' in year_data.columns else 'value'
                
                # Main time series with sequential x-axis
                fig_year.add_trace(
                    go.Scatter(
                        x=sequential_x,
                        y=year_data[value_col],
                        mode='lines',
                        name='Normal Data',
                        line=dict(color='blue', width=1),
                        customdata=datetime_labels,
                        hovertemplate='<b>Normal Data</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add warm-up period shading only for the first year in the merged dataset
                if has_warmup_cols and warm_up_period > 0:
                    # Check if this year contains the beginning of the merged dataset
                    first_year = sorted(data['year_source'].unique())[0]
                    if year == first_year:
                        # Only show warm-up period for the first year since detection runs on merged data
                        warmup_end_idx = min(warm_up_period-1, len(year_data)-1)
                        fig_year.add_vrect(
                            x0=0,
                            x1=warmup_end_idx,
                            fillcolor="gray",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                            annotation_text="Warm-up Period (Global)",
                            annotation_position="top left",
                            row=1, col=1
                        )
                        fig_year.add_vrect(
                            x0=0,
                            x1=warmup_end_idx,
                            fillcolor="gray",
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                            row=2, col=1
                        )
                
                # Anomalies with different markers for warm-up vs effective
                if has_warmup_cols:
                    # Warm-up anomalies (orange triangles)
                    warmup_anomaly_mask = year_data['anomaly_in_warmup']
                    warmup_anomaly_indices = [i for i, mask in enumerate(warmup_anomaly_mask) if mask]
                    if warmup_anomaly_indices:
                        warmup_anomaly_values = year_data[warmup_anomaly_mask][value_col]
                        warmup_anomaly_times = year_data[warmup_anomaly_mask]['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        fig_year.add_trace(
                            go.Scatter(
                                x=warmup_anomaly_indices,
                                y=warmup_anomaly_values,
                                mode='markers',
                                name='Warm-up Anomalies',
                                marker=dict(color='orange', size=8, symbol='triangle-up'),
                                customdata=warmup_anomaly_times,
                                hovertemplate='<b>Warm-up Anomaly</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    
                    # Effective anomalies (red X marks)
                    effective_anomaly_mask = year_data['anomaly_after_warmup']
                    effective_anomaly_indices = [i for i, mask in enumerate(effective_anomaly_mask) if mask]
                    if effective_anomaly_indices:
                        effective_anomaly_values = year_data[effective_anomaly_mask][value_col]
                        effective_anomaly_times = year_data[effective_anomaly_mask]['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        fig_year.add_trace(
                            go.Scatter(
                                x=effective_anomaly_indices,
                                y=effective_anomaly_values,
                                mode='markers',
                                name='Effective Anomalies',
                                marker=dict(color='red', size=8, symbol='x'),
                                customdata=effective_anomaly_times,
                                hovertemplate='<b>Effective Anomaly</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                else:
                    # Fallback: show all anomalies in red
                    anomaly_mask = year_data['detected_anomaly']
                    anomaly_indices = [i for i, mask in enumerate(anomaly_mask) if mask]
                    if anomaly_indices:
                        anomaly_values = year_data[anomaly_mask][value_col]
                        anomaly_times = year_data[anomaly_mask]['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        fig_year.add_trace(
                            go.Scatter(
                                x=anomaly_indices,
                                y=anomaly_values,
                                mode='markers',
                                name='Detected Anomalies',
                                marker=dict(color='red', size=8, symbol='x'),
                                customdata=anomaly_times,
                                hovertemplate='<b>Detected Anomaly</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Value: %{y}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                
                # Anomaly scores with sequential x-axis
                fig_year.add_trace(
                    go.Scatter(
                        x=sequential_x,
                        y=year_data['anomaly_score'],
                        mode='lines',
                        name='Anomaly Score',
                        line=dict(color='orange', width=2),
                        customdata=datetime_labels,
                        hovertemplate='<b>Anomaly Score</b><br>Index: %{x}<br>DateTime: %{customdata}<br>Score: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Threshold line
                threshold = st.session_state.get('detection_threshold', None)
                if threshold is not None:
                    config = st.session_state.get('config', {})
                    threshold_method = config.get('threshold_method', 'unknown')
                    fig_year.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Threshold ({threshold_method})",
                        row=2, col=1
                    )
                
                # Custom x-axis labels with datetime
                tick_interval = max(1, len(year_data) // 10)  # Show ~10 labels
                tick_positions = list(range(0, len(year_data), tick_interval))
                tick_labels = [datetime_labels[i] if i < len(datetime_labels) else "" for i in tick_positions]
                
                fig_year.update_layout(
                    height=500,
                    title_text=f"Anomaly Detection Results - Year {year}",
                    showlegend=True,
                    xaxis=dict(
                        tickmode='array',
                        tickvals=tick_positions,
                        ticktext=tick_labels,
                        tickangle=45
                    ),
                    xaxis2=dict(
                        tickmode='array',
                        tickvals=tick_positions,
                        ticktext=tick_labels,
                        tickangle=45
                    )
                )
                
                fig_year.update_xaxes(title_text="Sequential Time Index", row=2, col=1)
                fig_year.update_yaxes(title_text=value_col.title(), row=1, col=1)
                fig_year.update_yaxes(title_text="Anomaly Score", row=2, col=1)
                
                st.plotly_chart(fig_year, width="stretch")
                st.divider()
    else:
        # Fallback: original single plot for data without year_source
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
        
        # Add warm-up period shading if we have the data
        if has_warmup_cols and warm_up_period > 0:
            warmup_end_time = data['timestamp'].iloc[min(warm_up_period-1, len(data)-1)]
            fig.add_vrect(
                x0=data['timestamp'].iloc[0],
                x1=warmup_end_time,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Warm-up Period",
                annotation_position="top left",
                row=1, col=1
            )
            fig.add_vrect(
                x0=data['timestamp'].iloc[0],
                x1=warmup_end_time,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=2, col=1
            )
        
        # Anomalies with different markers for warm-up vs effective
        if has_warmup_cols:
            # Warm-up anomalies (orange triangles)
            warmup_anomaly_data = data[data['anomaly_in_warmup']]
            if not warmup_anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=warmup_anomaly_data['timestamp'],
                        y=warmup_anomaly_data[value_col],
                        mode='markers',
                        name='Warm-up Anomalies',
                        marker=dict(color='orange', size=8, symbol='triangle-up'),
                        hovertemplate='<b>Warm-up Anomaly</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Effective anomalies (red X marks)
            effective_anomaly_data = data[data['anomaly_after_warmup']]
            if not effective_anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=effective_anomaly_data['timestamp'],
                        y=effective_anomaly_data[value_col],
                        mode='markers',
                        name='Effective Anomalies',
                        marker=dict(color='red', size=8, symbol='x'),
                        hovertemplate='<b>Effective Anomaly</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
        else:
            # Fallback: show all anomalies in red
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
        # Add anomaly score heatmap (year-by-year plots are already available above)
        st.subheader("🌡️ Anomaly Score Heatmap")
        create_anomaly_score_heatmap(data)
        
        st.subheader("🔍 Anomaly Details")
        anomaly_details = data[data['detected_anomaly']].copy()
        anomaly_details = anomaly_details.sort_values('timestamp', ascending=True)
        
        # Add options for table display
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Total anomalies found:** {len(anomaly_details)}")
        with col2:
            table_height = st.selectbox(
                "Table Height",
                [200, 300, 400, 500, 600],
                index=1,
                help="Select table height for scrolling"
            )
        
        # Display all anomalies in a scrollable table
        st.dataframe(
            anomaly_details[['timestamp', value_col, 'anomaly_score']],
            width="stretch",
            height=table_height
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
    
    # Display information about the loaded time series
    if 'year_source' in data.columns:
        years_available = sorted(data['year_source'].unique())
        st.info(f"📊 **Monthly Time Series Data:** Loaded data from {len(years_available)} years: {', '.join(map(str, years_available))}")
        st.info(f"🕐 **Time Range:** {data['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {data['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    
    # Date range selector for filtering within the loaded time series
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Filter Start Date", 
                                 data['timestamp'].min().date(),
                                 min_value=data['timestamp'].min().date(),
                                 max_value=data['timestamp'].max().date(),
                                 help="Filter the loaded time series data from this date")
    with col2:
        end_date = st.date_input("Filter End Date", 
                               data['timestamp'].max().date(),
                               min_value=data['timestamp'].min().date(),
                               max_value=data['timestamp'].max().date(),
                               help="Filter the loaded time series data to this date")
    
    # Filter data
    mask = (data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)
    filtered_data = data[mask]
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Determine value column
    value_col = 'population' if 'population' in filtered_data.columns else 'value'
    
    # Area Visualization Section
    if 'uploaded_polygon' in st.session_state and 'extracted_mesh_gdf' in st.session_state:
        st.subheader("🗺️ Area Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.write("**Visualization Options:**")
            show_population = st.checkbox("Show Population Data", value=True, 
                                        help="Color mesh regions based on population data")
            
            if st.button("🗺️ Generate Area Map", type="primary"):
                with st.spinner("Creating area visualization..."):
                    try:
                        # Get mesh data with population if available
                        mesh_gdf = st.session_state.extracted_mesh_gdf.copy()
                        polygon_gdf = st.session_state.uploaded_polygon.copy()
                        
                        # Add population data if requested and data is available
                        if show_population and not filtered_data.empty:
                            # For simplicity, use mean population for each mesh
                            pop_data = filtered_data.groupby('mesh_id')[value_col].mean().reset_index()
                            pop_data['mesh_id'] = pop_data['mesh_id'].astype(str)
                            
                            # Find the mesh ID column in the mesh_gdf
                            mesh_id_col = None
                            possible_cols = ['MESH4_ID', 'MESH_ID', 'meshcode', 'mesh_id', 'id', 'ID']
                            for col in possible_cols:
                                if col in mesh_gdf.columns:
                                    mesh_id_col = col
                                    break
                            
                            if mesh_id_col:
                                mesh_gdf[mesh_id_col] = mesh_gdf[mesh_id_col].astype(str)
                                mesh_gdf = mesh_gdf.merge(pop_data, left_on=mesh_id_col, right_on='mesh_id', how='left')
                                mesh_gdf['population'] = mesh_gdf[value_col].fillna(0)
                                log_message(f"Merged population data using column: {mesh_id_col}", "info")
                            else:
                                log_message("Could not find mesh ID column for population data merge", "warning")
                                show_population = False  # Disable population visualization
                        
                        # Create the plot
                        fig = create_mesh_visualization_plot(
                            mesh_gdf, 
                            polygon_gdf, 
                            population_data=filtered_data if show_population else None
                        )
                        
                        if fig is not None:
                            st.pyplot(fig)
                            log_message("Area visualization created successfully", "info")
                        else:
                            st.error("Failed to create area visualization")
                            
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
                        log_message(f"Error in area visualization: {e}", "error")
        
        with col1:
            st.write("**Area Information:**")
            mesh_count = len(st.session_state.extracted_mesh_gdf)
            polygon_count = len(st.session_state.uploaded_polygon)
            st.write(f"📍 **Polygons uploaded:** {polygon_count}")
            st.write(f"🗺️ **Mesh regions found:** {mesh_count}")
            if not filtered_data.empty:
                st.write(f"📊 **Data points:** {len(filtered_data)}")
                st.write(f"📅 **Time range:** {filtered_data['timestamp'].min().strftime('%Y-%m-%d')} to {filtered_data['timestamp'].max().strftime('%Y-%m-%d')}")
    
    # Statistics
    st.subheader("📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Basic Statistics")
        # Use nan-aware functions to handle NaN values (previously -1)
        st.write(f"**Mean:** {np.nanmean(filtered_data[value_col]):.2f}")
        st.write(f"**Std Dev:** {np.nanstd(filtered_data[value_col]):.2f}")
        st.write(f"**Min:** {np.nanmin(filtered_data[value_col]):.2f}")
        st.write(f"**Max:** {np.nanmax(filtered_data[value_col]):.2f}")
    
    with col2:
        st.markdown("### 🚨 Anomaly Statistics")
        # Check if anomaly detection has been run and results are available
        if 'detection_results' in st.session_state and not st.session_state.detection_results.empty:
            detection_data = st.session_state.detection_results
            
            # Check if the date range matches current filtered data
            detection_date_range = st.session_state.get('analysis_date_range', None)
            current_date_range = (start_date, end_date)
            
            if detection_date_range == current_date_range:
                # Use detection results directly with warm-up period awareness
                config = st.session_state.get('detection_config', {})
                threshold_method = config.get('threshold_method', 'Unknown')
                warm_up_period = config.get('warm_up_period', config.get('subsequence_length', 3))
                
                # Check if warm-up columns exist
                has_warmup_cols = 'anomaly_in_warmup' in detection_data.columns and 'anomaly_after_warmup' in detection_data.columns
                
                if has_warmup_cols:
                    total_anomalies = detection_data['detected_anomaly'].sum()
                    warmup_anomalies = detection_data['anomaly_in_warmup'].sum()
                    effective_anomalies = detection_data['anomaly_after_warmup'].sum()
                    total_points = len(detection_data)
                    effective_points = max(1, total_points - warm_up_period)  # Avoid division by zero
                    
                    st.write(f"**Total Anomalies:** {total_anomalies}")
                    st.write(f"**Warm-up Anomalies:** {warmup_anomalies} (period: {warm_up_period}h)")
                    st.write(f"**Effective Anomalies:** {effective_anomalies}")
                    st.write(f"**Overall Rate:** {(total_anomalies/total_points*100):.2f}%")
                    st.write(f"**Effective Rate:** {(effective_anomalies/effective_points*100):.2f}%")
                    st.write(f"**Method Used:** {threshold_method}")
                else:
                    # Fallback to original display if warm-up columns don't exist
                    anomaly_count = detection_data['detected_anomaly'].sum()
                    total_points = len(detection_data)
                    
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
    
    # Library Compatibility Status
    st.subheader("🔧 Library Compatibility Status")
    
    # Get implementation info
    detector = get_detector()
    impl_info = detector.get_implementation_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📦 NumPy Status**")
        numpy_version = impl_info.get('numpy_version', 'unknown')
        numpy_major = int(numpy_version.split('.')[0]) if numpy_version != 'unknown' else 0
        
        version_status = f"NumPy {numpy_version}"
        if numpy_major >= 2:
            numpy_2_compat = impl_info.get('numpy_2_compat', False)
            if numpy_2_compat:
                compat_status = "✅ NumPy 2.0 Compatibility: Applied"
                st.markdown(f'<div class="alert-green">{version_status}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="alert-green">{compat_status}</div>', unsafe_allow_html=True)
            else:
                compat_status = "⚠️ NumPy 2.0 Compatibility: Check Required"
                st.markdown(f'<div class="alert-orange">{version_status}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="alert-orange">{compat_status}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">{version_status}</div>', unsafe_allow_html=True)
            st.markdown('<div class="alert-green">✅ NumPy 1.x: Fully Compatible</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🌟 STUMPY Status**")
        available_impls = impl_info.get('available_implementations', [])
        if 'STUMPY' in available_impls:
            stumpy_version = impl_info.get('stumpy_version', 'unknown')
            stumpy_status = f"✅ STUMPY {stumpy_version}: Available"
            st.markdown(f'<div class="alert-green">{stumpy_status}</div>', unsafe_allow_html=True)
            
            # Check for compatibility warnings
            warnings = impl_info.get('compatibility_warnings', [])
            if warnings:
                st.markdown('<div class="alert-orange">⚠️ Compatibility Issues Detected</div>', unsafe_allow_html=True)
                for warning in warnings:
                    st.caption(f"• {warning}")
            else:
                st.markdown('<div class="alert-green">✅ No Compatibility Issues</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-red">❌ STUMPY: Not Available</div>', unsafe_allow_html=True)
            st.markdown('<div class="alert-orange">⚠️ Left Matrix Profile: Limited</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**🚀 Active Implementation**")
        selected_impl = impl_info.get('selected_implementation', 'unknown')
        use_left_mp = impl_info.get('use_left_mp', False)
        window_size = impl_info.get('window_size', 0)
        
        impl_display = {
            'auto': '🤖 Auto Selection',
            'pyscamp': '🚀 PySCAMP',
            'stumpy': '🌟 STUMPY', 
            'custom': '🛠️ Custom'
        }.get(selected_impl, f'❓ {selected_impl}')
        
        st.markdown(f'<div class="alert-green">{impl_display}</div>', unsafe_allow_html=True)
        
        mp_type = "🔒 Left Matrix Profile" if use_left_mp else "📊 Standard Matrix Profile"
        st.markdown(f'<div class="alert-green">{mp_type}</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="alert-green">⚙️ Window Size: {window_size}</div>', unsafe_allow_html=True)
    
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
        
        # Get date range info
        date_range_info = "Not configured"
        if config.get('start_date') and config.get('end_date'):
            date_span = (config['end_date'] - config['start_date']).days + 1
            years_span = config.get('years_span', [])
            date_range_info = f"{config['start_date']} to {config['end_date']} ({date_span} days, {len(years_span)} years)"
        
        config_data = pd.DataFrame({
            'Parameter': [
                'Subsequence Length', 
                'Threshold Method', 
                'Threshold Multiplier',
                'Normalize Matrix Profile',
                'Available Data Range', 
                'Selected Date Range',
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
                f"{min(get_available_years())}-{max(get_available_years())} ({len(get_available_years())} years)",
                date_range_info,
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
    ## 🌋 Welcome to AnDeS
    
    The **ANomaly DEtection System (AnDeS)** is a cutting-edge anomaly detection platform 
    designed to identify unusual patterns in Mobile Spatial Statistics (MSS) data that may indicate 
    natural disasters or other significant events.
    
    ### 🎯 Key Features
    
    - **Real-time Anomaly Detection**: Interactive analysis with customizable parameters
    - **Multiple Matrix Profile Implementations**: Choose between PySCAMP, STUMPY, or custom implementation
    - **Left Matrix Profile Mode**: Option to only consider past data for anomaly detection
    - **Interactive Visualization**: Rich Plotly charts with anomaly overlays and heatmaps
    - **Historical Analysis**: Deep dive into historical patterns with date range filtering
    - **Multi-mesh Analysis**: Support for single or aggregated mesh analysis
    - **Live System Monitoring**: Real-time performance metrics and system status
    - **Custom Thresholds**: Flexible δ×σ threshold configuration with percentile alternatives
    
    ### 🔬 Algorithm Overview
    
    AnDeS uses **Matrix Profile** algorithms to detect anomalies with multiple implementation options:
    
    **Available Implementations:**
    - **🚀 PySCAMP**: GPU-accelerated matrix profile computation for large datasets
    - **🌟 STUMPY**: Streaming support with left matrix profile capabilities
    - **🛠️ Custom**: Pure Python implementation for maximum compatibility
    
    **Matrix Profile Types:**
    - **Standard Matrix Profile**: Considers all data for finding nearest neighbors
    - **Left Matrix Profile**: Only uses historical data (prevents future hindsight)
    
    **Detection Process:**
    1. **Data Preprocessing**: Load and filter MSS data by date range and mesh IDs
    2. **Matrix Profile Computation**: Calculate similarity between subsequences
    3. **Threshold Detection**: Apply configurable thresholds (δ×σ, percentile95, percentile99)
    4. **Anomaly Classification**: Flag unusual patterns for investigation and visualization
    
    ### 💡 Left Matrix Profile Explained
    
    The **Left Matrix Profile** option addresses a key consideration in anomaly detection:
    
    - **Standard Mode**: Uses all available data to find nearest neighbors, which means future data can update past anomaly scores
    - **Left Matrix Profile Mode**: Only considers past data when finding nearest neighbors, ensuring that once an anomaly is detected, future data won't change that classification
    
    This is particularly useful for real-time scenarios where you want consistent historical anomaly detection that doesn't change as new data arrives.
    
    ### 📊 Data Sources
    
    - **Mobile Spatial Statistics (MSS)**: Population density and movement patterns from NTT data
    - **Geographical Mesh Codes**: 4th-level mesh identifiers for precise location mapping
    - **Temporal Resolution**: Hourly data for fine-grained analysis (2016-2025)
    - **File Format**: NumPy arrays (.npy) with corresponding area mapping files
    
    ### 🚀 Getting Started
    
    #### Step 1: Configure Data Source
    1. **Select Date Range**: Choose start and end dates (can span multiple years from 2016-2025)
    2. **Choose Mesh ID**: Select from dropdown or enter custom mesh IDs (comma-separated)
    3. **Multi-mesh Analysis**: Enable to aggregate data from multiple mesh IDs
    
    **New Feature**: Multi-year analysis support! You can now select date ranges that span across multiple years for comprehensive long-term anomaly detection.
    
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
    - **Date Range Selection**: Choose start and end dates (supports multi-year ranges)
    - **Event Selection**: Choose from predefined major events with pre-configured mesh codes (1 month before + 1 week after event date)
    - **Custom Selection**: Manual date/mesh selection (1 month before until event date)
    - **Event-based Mesh IDs**: Automatic mesh code selection based on event location
    - **Custom Mesh Selection**: Manual mesh ID input for custom analysis areas
    - **Multi-mesh Analysis**: Aggregate multiple mesh regions or use main event mesh only
    - **Years Back Control**: Both Event Selection and Custom Selection respect the years back setting for multi-year analysis
    
    #### Available Events
    - **Haneda Airport runway collision** (Jan 2, 2024) - 9 mesh codes around Tokyo Haneda
    - **Taylor Swift – The Eras Tour (Tokyo Dome)** (Feb 7, 2024) - 9 mesh codes around Tokyo Dome
    - **Bruno Mars - (Tokyo Dome)** (Jan 1, 2024) - 9 mesh codes around Tokyo Dome
    - **Comic Market 104 (Tokyo Big Sight)** (Aug 11, 2024) - 9 mesh codes around Tokyo Big Sight
    - **Tokyo Game Show 2024 (Makuhari Messe)** (Sep 28, 2024) - 9 mesh codes around Makuhari Messe
    
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
    
    - ✅ **Event-based Mesh Selection**: Added predefined major events with pre-configured mesh codes
    - ✅ **NumPy 2.0 Compatibility**: Automatic fixes for STUMPY library compatibility issues
    - ✅ Enhanced library compatibility monitoring in System Status
    - ✅ Simplified threshold configuration to δ×σ format
    - ✅ Enhanced date range filtering and consistency
    - ✅ Live system status monitoring with auto-refresh
    - ✅ Improved anomaly details sorting by timestamp
    - ✅ Better session state management and configuration tracking
    - ✅ Fixed duplicate detection execution issues
    
    ### 📞 Support

    For technical support or questions about AnDeS, please contact the development team.
    
    ### 🔧 Troubleshooting
    
    #### NumPy 2.0 Compatibility Issues
    
    **Problem**: Error message about `np.NINF` being removed in NumPy 2.0
    
    **Solution**: The system automatically handles this compatibility issue by:
    - ✅ Applying compatibility patches for STUMPY library
    - ✅ Detecting NumPy 2.0 and missing deprecated constants
    - ✅ Falling back to custom implementation if needed
    - ✅ Showing compatibility status in System Status tab
    
    **Manual Fix** (if automatic fix fails):
    ```bash
    # Option 1: Update STUMPY to NumPy 2.0 compatible version
    pip install --upgrade stumpy
    
    # Option 2: Use NumPy 1.x
    pip install "numpy<2.0"
    ```
    
    #### Common Issues
    
    **Data Loading Failures**:
    - Check data file paths in DATA_DIR
    - Verify .npy files exist for selected years
    - Ensure mesh IDs match those in areas files
    
    **Detection Not Running**:
    - Load data first using "Load/Refresh Data" button
    - Check that subsequence length < data length
    - Verify date range contains valid data
    
    **Performance Issues**:
    - Monitor memory usage in System Status tab
    - Reduce date range for large datasets
    - Use PySCAMP for GPU acceleration if available
    
    **STUMPY Import Errors**:
    - Check System Status tab for compatibility info
    - Install STUMPY: `pip install stumpy`
    - System will fallback to custom implementation
    """)
    
    # Credits
    st.markdown("""
    ---
    ### 👥 Credits
    
    **AnDeS** is developed by Erick Mas@Tohoku University.
    
    **Based on**: SCAMP algorithm
    
    **Version**: 2025.1.0
    
    **Last Updated**: September 2025
    """)

# Always run main in Streamlit
main()