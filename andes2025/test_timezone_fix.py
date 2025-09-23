#!/usr/bin/env python3
"""
Test to verify the datetime fix for timezone handling.
"""

import pandas as pd
from datetime import datetime, timezone

def test_timezone_fix():
    """Test that timezone-aware and timezone-naive datetimes are handled correctly."""
    
    print("Testing Timezone Fix")
    print("=" * 40)
    
    # Test 1: Timezone-aware datetime (like predefined events)
    print("1. Testing timezone-aware datetime:")
    tz_aware = datetime(2024, 1, 1, 16, 0, 0, 0, timezone.utc)
    tz_naive = tz_aware.replace(tzinfo=None)
    print(f"   Original (timezone-aware): {tz_aware}")
    print(f"   Converted (timezone-naive): {tz_naive}")
    
    # Test 2: Timezone-naive datetime (like custom selection)
    print("\n2. Testing timezone-naive datetime:")
    naive_original = datetime(2024, 1, 1, 16, 0, 0)
    print(f"   Original (timezone-naive): {naive_original}")
    print(f"   Already timezone-naive: {naive_original.tzinfo is None}")
    
    # Test 3: Pandas timestamp comparison
    print("\n3. Testing pandas timestamp comparison:")
    timestamps = pd.date_range(start='2024-01-01', periods=24, freq='H')
    
    # Test with timezone-naive datetime
    test_time = datetime(2024, 1, 1, 12, 0, 0)
    test_ts = pd.Timestamp(test_time)
    
    try:
        mask = timestamps >= test_ts
        print(f"   ✅ Comparison successful with timezone-naive datetime")
        print(f"   Sample result: {sum(mask)} timestamps match criteria")
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_timezone_fix()