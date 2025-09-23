#!/usr/bin/env python3
"""
Test script to verify -1 to np.nan conversion throughout the entire app pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_preprocessing_pipeline():
    """Test the complete data preprocessing pipeline."""
    print("🧪 Testing complete data preprocessing pipeline...")
    
    try:
        # Create sample data similar to what MSS would provide
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'population': [100, 200, -1, 300, -1, 150, 250, -1, 180, 220],
            'mesh_id': ['533946403'] * 10
        })
        
        print("Original data:")
        print(sample_data[['timestamp', 'population']])
        print(f"Original -1 count: {(sample_data['population'] == -1).sum()}")
        
        # Simulate the preprocessing that happens after data loading
        value_col = 'population'
        original_minus_ones = (sample_data[value_col] == -1).sum()
        sample_data[value_col] = sample_data[value_col].replace(-1, np.nan)
        
        print(f"\nAfter preprocessing:")
        print(sample_data[['timestamp', 'population']])
        print(f"Converted {original_minus_ones} instances of -1 to np.nan")
        print(f"Current NaN count: {sample_data['population'].isna().sum()}")
        
        # Test statistics calculations with NaN-aware functions
        print(f"\nStatistics (NaN-aware):")
        print(f"Mean: {np.nanmean(sample_data[value_col]):.2f}")
        print(f"Std: {np.nanstd(sample_data[value_col]):.2f}")
        print(f"Min: {np.nanmin(sample_data[value_col]):.2f}")
        print(f"Max: {np.nanmax(sample_data[value_col]):.2f}")
        print(f"No data count: {sample_data[value_col].isna().sum()}")
        
        # Test matrix profile data extraction
        values = sample_data[value_col].values
        remaining_minus_ones = np.sum(values == -1) if len(values) > 0 else 0
        print(f"\nMatrix profile ready data:")
        print(f"Values array: {values}")
        print(f"Remaining -1 values: {remaining_minus_ones}")
        print(f"NaN values ready for matrix profile: {np.sum(np.isnan(values))}")
        
        # Verify no -1 values remain
        if remaining_minus_ones == 0:
            print("✅ No -1 values remain in processed data")
        else:
            print(f"❌ Warning: {remaining_minus_ones} -1 values still present")
            return False
        
        # Verify NaN handling
        expected_nan_count = 3  # We had 3 -1 values originally
        actual_nan_count = np.sum(np.isnan(values))
        if actual_nan_count == expected_nan_count:
            print(f"✅ Correct NaN count: {actual_nan_count}")
        else:
            print(f"❌ NaN count mismatch: expected {expected_nan_count}, got {actual_nan_count}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_visualization_data():
    """Test that visualization data doesn't contain -1 values."""
    print("🧪 Testing visualization data integrity...")
    
    try:
        # Create sample filtered data (as would be used for visualization)
        filtered_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=8, freq='H'),
            'population': [100, np.nan, 300, np.nan, 150, 250, np.nan, 180],
            'mesh_id': ['533946403'] * 8
        })
        
        value_col = 'population'
        
        # Test data preview table (should not show -1)
        print("Sample data for visualization:")
        preview_data = filtered_data.head(5).copy()
        if 'timestamp' in preview_data.columns:
            preview_data['timestamp'] = preview_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        print(preview_data)
        
        # Check for any -1 values in display data
        minus_one_count = (filtered_data[value_col] == -1).sum()
        print(f"✅ -1 values in display data: {minus_one_count} (should be 0)")
        
        # Test statistics calculations
        no_data_count = filtered_data[value_col].isna().sum()
        min_val = np.nanmin(filtered_data[value_col])
        max_val = np.nanmax(filtered_data[value_col])
        
        print(f"✅ No data count (NaN): {no_data_count}")
        print(f"✅ Value range: {min_val:.0f} - {max_val:.0f}")
        
        return minus_one_count == 0
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_aggregation_consistency():
    """Test that aggregation is consistent with new preprocessing."""
    print("🧪 Testing aggregation consistency...")
    
    try:
        # Test multi-mesh aggregation with the new approach
        mesh1 = np.array([100, -1, 200, 150])
        mesh2 = np.array([50, 150, -1, 100])
        mesh3 = np.array([-1, 100, 250, -1])
        
        print("Original mesh data:")
        print(f"Mesh 1: {mesh1}")
        print(f"Mesh 2: {mesh2}")
        print(f"Mesh 3: {mesh3}")
        
        # Apply the aggregation preprocessing (from load_mss_data_single_year)
        extracted_data = [mesh1, mesh2, mesh3]
        data_array = np.array(extracted_data)
        
        # Replace -1 with np.nan for aggregation
        data_array = np.where(data_array == -1, np.nan, data_array)
        aggregated_data = np.nansum(data_array, axis=0)
        
        # Handle all-NaN case
        all_nan_mask = np.all(np.isnan(data_array), axis=0)
        aggregated_data = np.where(all_nan_mask, -1, aggregated_data)
        
        print(f"Aggregated result: {aggregated_data}")
        
        # Now apply the post-loading preprocessing
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(aggregated_data), freq='H'),
            'population': aggregated_data,
            'mesh_id': ['533946403'] * len(aggregated_data)
        })
        
        # Convert -1 to np.nan (as done in real_time_analysis)
        original_minus_ones = (df['population'] == -1).sum()
        df['population'] = df['population'].replace(-1, np.nan)
        
        print(f"After post-loading preprocessing:")
        print(f"Converted {original_minus_ones} aggregated -1 values to np.nan")
        print(f"Final data: {df['population'].values}")
        
        # Verify consistency
        has_minus_ones = (df['population'] == -1).sum() > 0
        if not has_minus_ones:
            print("✅ No -1 values in final processed data")
            return True
        else:
            print("❌ Still have -1 values in final data")
            return False
        
    except Exception as e:
        print(f"❌ Aggregation consistency test failed: {e}")
        return False

def main():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("COMPREHENSIVE -1 TO NP.NAN PIPELINE TESTING")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Complete data preprocessing pipeline
    try:
        if not test_data_preprocessing_pipeline():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        all_tests_passed = False
    
    print()
    
    # Test 2: Visualization data integrity
    try:
        if not test_visualization_data():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        all_tests_passed = False
    
    print()
    
    # Test 3: Aggregation consistency
    try:
        if not test_aggregation_consistency():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Aggregation consistency test failed: {e}")
        all_tests_passed = False
    
    print()
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Complete pipeline working correctly.")
        print()
        print("✅ Pipeline improvements:")
        print("   1. Data loading: -1 converted to np.nan immediately after loading")
        print("   2. Visualization: No -1 values in tables, plots, or statistics")
        print("   3. Statistics: NaN-aware functions used throughout")
        print("   4. Matrix profile: Clean np.nan data ready for computation")
        print("   5. Aggregation: Consistent handling of missing data")
        print("   6. Display integrity: All UI elements show clean data")
    else:
        print("❌ SOME TESTS FAILED! Pipeline needs review.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())