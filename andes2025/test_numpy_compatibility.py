#!/usr/bin/env python3
"""
Test script to verify NumPy 2.0 compatibility fixes for STUMPY.
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def test_numpy_compatibility():
    """Test NumPy version and STUMPY compatibility."""
    print("=" * 60)
    print("NUMPY 2.0 COMPATIBILITY TEST")
    print("=" * 60)
    
    # Check NumPy version
    print(f"📊 NumPy version: {np.__version__}")
    major_version = int(np.__version__.split('.')[0])
    
    if major_version >= 2:
        print("⚠️  NumPy 2.0+ detected - checking for compatibility issues")
        
        # Check for deprecated constants
        deprecated_attrs = ['NINF', 'PINF', 'NAN']
        for attr in deprecated_attrs:
            if hasattr(np, attr):
                print(f"✅ np.{attr} is available")
            else:
                print(f"❌ np.{attr} is missing (expected in NumPy 2.0+)")
    else:
        print("✅ NumPy 1.x detected - should be compatible")
    
    print()
    
    # Test STUMPY import and basic functionality
    print("🧪 Testing STUMPY import and basic functionality...")
    try:
        # Import our core module (which includes the compatibility fix)
        sys.path.append('.')
        from core import ScampAnomalyDetector
        
        print("✅ Core module imported successfully")
        
        # Create test data
        np.random.seed(42)
        test_data = np.random.randn(100)
        
        # Test STUMPY specifically
        detector = ScampAnomalyDetector(
            window_size=10, 
            implementation='stumpy',
            use_left_mp=False
        )
        
        print("✅ STUMPY detector created successfully")
        
        # Test computation
        anomalies, scores, threshold = detector.detect_anomalies_batch(test_data)
        
        print(f"✅ STUMPY computation successful")
        print(f"   - Matrix profile length: {len(scores)}")
        print(f"   - Anomalies detected: {np.sum(anomalies)}")
        print(f"   - Threshold used: {threshold:.4f}")
        
        # Test left matrix profile
        detector_left = ScampAnomalyDetector(
            window_size=10, 
            implementation='stumpy',
            use_left_mp=True
        )
        
        anomalies_left, scores_left, threshold_left = detector_left.detect_anomalies_batch(test_data)
        
        print(f"✅ STUMPY left matrix profile computation successful")
        print(f"   - Left MP length: {len(scores_left)}")
        print(f"   - Left MP anomalies: {np.sum(anomalies_left)}")
        
    except Exception as e:
        print(f"❌ STUMPY test failed: {str(e)}")
        if 'NINF' in str(e):
            print("💡 This is the NumPy 2.0 compatibility issue we're trying to fix")
        return False
    
    print()
    print("=" * 60)
    print("✅ ALL TESTS PASSED - NumPy 2.0 compatibility working!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_numpy_compatibility()
    sys.exit(0 if success else 1)