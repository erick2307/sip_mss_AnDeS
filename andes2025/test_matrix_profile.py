#!/usr/bin/env python3
"""
Test script for the enhanced matrix profile implementation.
"""

import numpy as np
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import ScampAnomalyDetector, custom_matrix_profile

def test_implementations():
    """Test all available matrix profile implementations."""
    print("🧪 Testing Enhanced Matrix Profile Implementations")
    print("=" * 50)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_points = 100
    time_series = np.random.randn(n_points) * 10 + 100
    
    # Add some artificial anomalies
    time_series[30] += 50  # Anomaly at position 30
    time_series[70] -= 40  # Anomaly at position 70
    
    print(f"📊 Generated test data: {n_points} points with 2 artificial anomalies")
    
    # Test parameters
    window_size = 10
    
    # Test each implementation
    implementations = ['auto', 'custom', 'stumpy', 'pyscamp']
    
    for impl in implementations:
        print(f"\n🔧 Testing {impl.upper()} implementation:")
        
        try:
            # Test standard matrix profile
            detector = ScampAnomalyDetector(
                window_size=window_size,
                implementation=impl,
                use_left_mp=False,
                threshold_method='sigma'
            )
            
            anomalies, scores, threshold = detector.detect_anomalies_batch(time_series, custom_multiplier=2.0)
            anomaly_count = np.sum(anomalies)
            
            print(f"   ✅ Standard MP: {anomaly_count} anomalies detected (threshold: {threshold:.2f})")
            
            # Test left matrix profile if supported
            try:
                detector_left = ScampAnomalyDetector(
                    window_size=window_size,
                    implementation=impl,
                    use_left_mp=True,
                    threshold_method='sigma'
                )
                
                anomalies_left, scores_left, threshold_left = detector_left.detect_anomalies_batch(
                    time_series, custom_multiplier=2.0
                )
                anomaly_count_left = np.sum(anomalies_left)
                
                print(f"   ✅ Left MP: {anomaly_count_left} anomalies detected (threshold: {threshold_left:.2f})")
                
            except Exception as e:
                print(f"   ⚠️  Left MP not supported: {str(e)}")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    print("\n🎯 Testing custom matrix profile function directly:")
    
    # Test custom function directly
    try:
        mp_standard = custom_matrix_profile(time_series, window_size, normalize=False, use_left_mp=False)
        mp_left = custom_matrix_profile(time_series, window_size, normalize=False, use_left_mp=True)
        
        print(f"   ✅ Custom standard MP: {len(mp_standard)} profile points")
        print(f"   ✅ Custom left MP: {len(mp_left)} profile points")
        
        # Compare results
        if len(mp_standard) == len(mp_left):
            diff_count = np.sum(mp_standard != mp_left)
            print(f"   📊 Difference in {diff_count}/{len(mp_standard)} points between standard and left MP")
        
    except Exception as e:
        print(f"   ❌ Custom function failed: {str(e)}")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_implementations()