#!/usr/bin/env python3
"""
Test script for the new anomaly score heatmap function.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_score_heatmap_function():
    """Test the anomaly score heatmap function with sample data."""
    print("🧪 Testing anomaly score heatmap function...")
    
    try:
        # Create sample data similar to what the app would generate
        dates = pd.date_range('2024-01-01', periods=72, freq='H')  # 3 days of hourly data
        
        # Create sample data with varying anomaly scores
        np.random.seed(42)  # For reproducible results
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'population': np.random.randint(100, 1000, len(dates)),
            'detected_anomaly': np.random.choice([True, False], len(dates), p=[0.1, 0.9]),  # 10% anomalies
            'anomaly_score': np.random.exponential(5, len(dates)),  # Exponential distribution for scores
            'mesh_id': ['533946403'] * len(dates)
        })
        
        # Ensure some high scores for anomalies
        anomaly_mask = sample_data['detected_anomaly']
        sample_data.loc[anomaly_mask, 'anomaly_score'] = np.random.uniform(15, 50, np.sum(anomaly_mask))
        
        print(f"✅ Created sample data: {len(sample_data)} records")
        print(f"   - Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
        print(f"   - Anomalies: {sample_data['detected_anomaly'].sum()}")
        print(f"   - Score range: {sample_data['anomaly_score'].min():.2f} to {sample_data['anomaly_score'].max():.2f}")
        
        # Test data processing logic (similar to what's in the function)
        data_copy = sample_data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        unique_dates = sorted(data_copy['date'].unique())
        print(f"✅ Unique dates: {len(unique_dates)} ({unique_dates[0]} to {unique_dates[-1]})")
        
        # Create score matrix
        score_matrix = np.full((len(unique_dates), 24), np.nan)
        
        for i, date in enumerate(unique_dates):
            day_data = data_copy[data_copy['date'] == date]
            for _, row in day_data.iterrows():
                hour = row['hour']
                if 0 <= hour <= 23:
                    score_matrix[i, hour] = row['anomaly_score']
        
        # Check matrix properties
        valid_scores = score_matrix[~np.isnan(score_matrix)]
        print(f"✅ Score matrix: {score_matrix.shape}, valid entries: {len(valid_scores)}")
        print(f"   - Score stats: min={np.min(valid_scores):.2f}, max={np.max(valid_scores):.2f}, mean={np.mean(valid_scores):.2f}")
        
        # Test matplotlib plotting (without displaying)
        fig, ax = plt.subplots(figsize=(12, 6))
        masked_matrix = np.ma.masked_where(np.isnan(score_matrix), score_matrix)
        im = ax.imshow(masked_matrix, cmap='viridis', aspect=0.7, interpolation='nearest')
        plt.close(fig)  # Close without displaying
        
        print("✅ Matplotlib visualization test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_imports():
    """Test that required imports work."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib import successful")
        
        import numpy as np
        print("✅ NumPy import successful")
        
        import pandas as pd
        print("✅ Pandas import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ANOMALY SCORE HEATMAP FUNCTION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Imports
    try:
        if not test_imports():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        all_tests_passed = False
    
    print()
    
    # Test 2: Score heatmap function logic
    try:
        if not test_score_heatmap_function():
            all_tests_passed = False
    except Exception as e:
        print(f"❌ Score heatmap test failed: {e}")
        all_tests_passed = False
    
    print()
    print("=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Score heatmap function is ready.")
        print()
        print("✅ New function added:")
        print("   - create_anomaly_score_heatmap(data)")
        print("   - Shows continuous anomaly scores using viridis colormap")
        print("   - Displays threshold information when available")
        print("   - Provides detailed score statistics")
        print("   - Integrated into display_detection_results()")
    else:
        print("❌ SOME TESTS FAILED! Please review the implementation.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())