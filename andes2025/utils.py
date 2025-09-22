"""
Utility functions for ANDES 2025 application.
Includes date/time utilities, data processing helpers, and visualization functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import arrow
from dateutil import tz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up Japanese timezone for our data
timezone = tz.gettz("Asia/Tokyo")

def get_date_one_month_ago(target_year, target_month, target_day, target_hour):
    """
    Calculate the date exactly one month before the target date.
    
    Args:
        target_year, target_month, target_day, target_hour: Target datetime components
        
    Returns:
        datetime: Date one month before the target
    """
    to_date = arrow.get(target_year, target_month, target_day, target_hour).replace(tzinfo=timezone)
    last_month = to_date.shift(months=-1)
    return last_month.datetime

def get_hours_between_dates(start_year, start_month, start_day, start_hour, 
                           stop_year, stop_month, stop_day, stop_hour):
    """
    Calculate the number of hours between two datetime points.
    
    Returns:
        int: Number of hours between start and stop dates
    """
    start = datetime(start_year, start_month, start_day, start_hour, 0, 0, 0, tzinfo=timezone)
    stop = datetime(stop_year, stop_month, stop_day, stop_hour, 0, 0, 0, tzinfo=timezone)
    delta = stop - start
    return delta.days * 24 + delta.seconds // 3600

def add_hours_to_date(base_date, hours):
    """
    Add a specified number of hours to a base date.
    
    Args:
        base_date: Base datetime object
        hours: Number of hours to add
        
    Returns:
        datetime: New datetime with hours added
    """
    return base_date + timedelta(hours=hours)

def parse_datetime(datetime_str):
    """
    Parse a datetime string in format 'YYYY-MM-DD HH:MM:SS'.
    
    Args:
        datetime_str: String representation of datetime
        
    Returns:
        datetime: Parsed datetime object with timezone
    """
    # Handle different formats
    if len(datetime_str.split(' ')) == 2:
        # Format: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM'
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M' if ':' in datetime_str.split(' ')[1] and datetime_str.split(' ')[1].count(':') == 1 else '%Y-%m-%d %H:%M:%S')
    else:
        # Assume ISO format
        dt = datetime.fromisoformat(datetime_str)
    
    # Add timezone if not present
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone)
    
    return dt

def format_datetime_for_filename(dt):
    """Format datetime for use in filenames (YYYYMMDDHHMM)."""
    return dt.strftime('%Y%m%d%H%M')

def format_datetime_for_display(dt):
    """Format datetime for display (YYYY-MM-DD HH:MM)."""
    return dt.strftime('%Y-%m-%d %H:%M')

class PlotGenerator:
    """Generate various plots for the ANDES application."""
    
    @staticmethod
    def create_time_series_plot(data, anomalies=None, scores=None, threshold=None):
        """
        Create a comprehensive time series plot with anomalies.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
            anomalies: Boolean array indicating anomalies
            scores: Anomaly scores array
            threshold: Threshold value for anomaly detection
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        # Determine number of subplots
        n_subplots = 2 if scores is not None else 1
        
        subplot_titles = ['Time Series Data']
        if scores is not None:
            subplot_titles.append('Anomaly Scores')
        
        fig = make_subplots(
            rows=n_subplots, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Main time series
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['value'],
                mode='lines',
                name='Normal Data',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add anomalies if provided
        if anomalies is not None:
            anomaly_data = data[anomalies]
            if not anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['value'],
                        mode='markers',
                        name='Detected Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ),
                    row=1, col=1
                )
        
        # Add scores subplot if provided
        if scores is not None:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=scores,
                    mode='lines',
                    name='Anomaly Score',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Add threshold line
            if threshold is not None:
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold:.2f}",
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=600 if n_subplots == 2 else 400,
            title_text="Anomaly Detection Results",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=n_subplots, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        if scores is not None:
            fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
        
        return fig
    
    @staticmethod
    def create_distribution_plot(data, anomalies=None):
        """
        Create distribution plots for data analysis.
        
        Args:
            data: DataFrame with 'value' column
            anomalies: Boolean array indicating anomalies
            
        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Value Distribution', 'Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data['value'],
                nbinsx=50,
                name='Data Distribution',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data['value'],
                name='Value Distribution',
                boxpoints='outliers'
            ),
            row=1, col=2
        )
        
        # Add anomaly distributions if provided
        if anomalies is not None:
            normal_data = data[~anomalies]['value']
            anomaly_data = data[anomalies]['value']
            
            if len(anomaly_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=anomaly_data,
                        nbinsx=20,
                        name='Anomalies',
                        opacity=0.7,
                        marker_color='red'
                    ),
                    row=1, col=1
                )
        
        fig.update_layout(
            height=400,
            title_text="Data Distribution Analysis",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_daily_pattern_plot(data):
        """
        Create a plot showing daily patterns.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
            
        Returns:
            plotly.graph_objects.Figure: Daily pattern plot
        """
        # Extract hour of day
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['day_of_week'] = data_copy['timestamp'].dt.day_name()
        
        # Calculate hourly averages
        hourly_avg = data_copy.groupby('hour')['value'].mean()
        hourly_std = data_copy.groupby('hour')['value'].std()
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg.values,
                mode='lines+markers',
                name='Average',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=list(hourly_avg.index) + list(hourly_avg.index[::-1]),
                y=list(hourly_avg.values + hourly_std.values) + list((hourly_avg.values - hourly_std.values)[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            )
        )
        
        fig.update_layout(
            title="Daily Pattern Analysis",
            xaxis_title="Hour of Day",
            yaxis_title="Average Value",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_summary_plot(anomaly_counts, time_periods):
        """
        Create a summary plot of anomaly counts over time periods.
        
        Args:
            anomaly_counts: List of anomaly counts
            time_periods: List of time period labels
            
        Returns:
            plotly.graph_objects.Figure: Summary plot
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=time_periods,
                y=anomaly_counts,
                name='Anomaly Count',
                marker_color='red',
                opacity=0.7
            )
        )
        
        fig.update_layout(
            title="Anomaly Summary by Time Period",
            xaxis_title="Time Period",
            yaxis_title="Number of Anomalies",
            height=400
        )
        
        return fig

class DataProcessor:
    """Data processing utilities for ANDES."""
    
    @staticmethod
    def clean_data(data, remove_outliers=True, outlier_threshold=3):
        """
        Clean data by handling missing values and optionally removing outliers.
        
        Args:
            data: DataFrame to clean
            remove_outliers: Whether to remove statistical outliers
            outlier_threshold: Z-score threshold for outlier removal
            
        Returns:
            DataFrame: Cleaned data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        if 'value' in cleaned_data.columns:
            # Forward fill then backward fill
            cleaned_data['value'] = cleaned_data['value'].fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers if requested
            if remove_outliers:
                z_scores = np.abs((cleaned_data['value'] - cleaned_data['value'].mean()) / cleaned_data['value'].std())
                cleaned_data = cleaned_data[z_scores < outlier_threshold]
        
        return cleaned_data
    
    @staticmethod
    def resample_data(data, frequency='H', agg_method='mean'):
        """
        Resample data to a different frequency.
        
        Args:
            data: DataFrame with timestamp index
            frequency: Target frequency ('H', 'D', '15T', etc.)
            agg_method: Aggregation method ('mean', 'sum', 'max', 'min')
            
        Returns:
            DataFrame: Resampled data
        """
        if 'timestamp' in data.columns:
            data_copy = data.set_index('timestamp')
        else:
            data_copy = data.copy()
        
        if agg_method == 'mean':
            resampled = data_copy.resample(frequency).mean()
        elif agg_method == 'sum':
            resampled = data_copy.resample(frequency).sum()
        elif agg_method == 'max':
            resampled = data_copy.resample(frequency).max()
        elif agg_method == 'min':
            resampled = data_copy.resample(frequency).min()
        else:
            resampled = data_copy.resample(frequency).mean()
        
        return resampled.reset_index()
    
    @staticmethod
    def calculate_statistics(data, value_column='value'):
        """
        Calculate comprehensive statistics for the data.
        
        Args:
            data: DataFrame containing the data
            value_column: Name of the column to analyze
            
        Returns:
            dict: Dictionary of statistics
        """
        values = data[value_column].values
        
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'skewness': pd.Series(values).skew(),
            'kurtosis': pd.Series(values).kurtosis()
        }
        
        return stats

# Caching utilities for performance
class CacheManager:
    """Simple cache manager for data and computation results."""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        """Get item from cache."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Put item in cache."""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Add/update item
        if key in self.cache:
            self.access_order.remove(key)
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()

# Global cache instance
cache_manager = CacheManager(max_size=50)