#!/usr/bin/env python3
"""
Debug script to test datetime handling in the monthly time series function.
"""

import pandas as pd
from datetime import datetime, timezone

def test_datetime_comparison():
    """Test datetime comparison to identify the issue."""
    
    print("Testing datetime comparison issue")
    print("=" * 50)
    
    # Simulate what happens in the app
    print("1. Creating event datetime (timezone-aware):")
    event_datetime = datetime(2024, 1, 1, 16, 0, 0, 0, timezone.utc)
    print(f"   event_datetime: {event_datetime}")
    print(f"   type: {type(event_datetime)}")
    print(f"   tzinfo: {event_datetime.tzinfo}")
    
    print("\n2. Converting to timezone-naive:")
    event_datetime_naive = event_datetime.replace(tzinfo=None)
    print(f"   event_datetime_naive: {event_datetime_naive}")
    print(f"   type: {type(event_datetime_naive)}")
    print(f"   tzinfo: {event_datetime_naive.tzinfo}")
    
    print("\n3. Creating period dates:")
    year_event_date = event_datetime_naive.replace(year=2017)
    month_before = year_event_date.replace(month=12, year=2016)
    print(f"   year_event_date: {year_event_date}")
    print(f"   month_before: {month_before}")
    
    print("\n4. Creating pandas timestamp column (like in load_mss_data_single_year):")
    timestamps = pd.date_range(start='2017-01-01', periods=24, freq='H')
    print(f"   timestamps sample: {timestamps[:5]}")
    print(f"   timestamps type: {type(timestamps[0])}")
    print(f"   timestamps dtype: {timestamps.dtype}")
    
    print("\n5. Creating comparison timestamps:")
    period_start_ts = pd.Timestamp(month_before)
    period_end_ts = pd.Timestamp(year_event_date)
    print(f"   period_start_ts: {period_start_ts}")
    print(f"   period_end_ts: {period_end_ts}")
    print(f"   period_start_ts type: {type(period_start_ts)}")
    print(f"   period_end_ts type: {type(period_end_ts)}")
    
    print("\n6. Testing comparison:")
    try:
        test_df = pd.DataFrame({'timestamp': timestamps})
        mask = (test_df['timestamp'] >= period_start_ts) & (test_df['timestamp'] <= period_end_ts)
        filtered = test_df[mask]
        print(f"   ✅ Comparison successful!")
        print(f"   Filtered rows: {len(filtered)}")
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        print(f"   Error type: {type(e)}")

if __name__ == "__main__":
    test_datetime_comparison()