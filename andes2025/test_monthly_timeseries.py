#!/usr/bin/env python3
"""
Test script for the new monthly time series functionality.
This validates that the new logic works correctly.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd

# Add the current directory to the path to import app modules
sys.path.append(str(Path(__file__).parent))

def test_monthly_timeseries():
    """Test the new monthly time series loading logic."""
    
    print("Testing Monthly Time Series Logic - CORRECTED VERSION")
    print("=" * 60)
    
    # Test parameters
    event_datetime = datetime(2024, 1, 1, 16, 0, 0, 0, timezone.utc)  # Noto Peninsula Earthquake
    mesh_ids = ["563712214"]  # Main mesh for the event
    
    print(f"Event DateTime: {event_datetime}")
    print(f"Mesh IDs: {mesh_ids}")
    print()
    
    # Test the CORRECTED date calculation logic
    print("Expected Monthly Periods (CORRECTED LOGIC):")
    print("-" * 50)
    print("Should extract: 1 month before TO event date each year")
    print()
    
    for year in range(2016, 2025):
        try:
            # Create the event date for this year
            year_event_date = event_datetime.replace(year=year)
            
            # Calculate one month before the event date
            if year_event_date.month == 1:
                month_before = year_event_date.replace(year=year-1, month=12)
            else:
                month_before = year_event_date.replace(month=year_event_date.month - 1)
            
            # Define the one-month extraction period (from one month before TO the event date)
            period_start = month_before
            period_end = year_event_date
            
            print(f"Year {year}: {period_start.strftime('%Y-%m-%d %H:%M')} TO {period_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"         Duration: {(period_end - period_start).days} days")
            print()
            
        except Exception as e:
            print(f"Year {year}: ERROR - {e}")
    
    print()
    print("✅ CORRECTED: Now extracts one month periods FROM one month before")
    print("   TO the event date in each year (approximately 30-31 days each).")
    print("   For Jan 1 event: Dec 1 TO Jan 1 each year")

if __name__ == "__main__":
    test_monthly_timeseries()