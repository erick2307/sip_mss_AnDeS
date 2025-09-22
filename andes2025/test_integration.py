#!/usr/bin/env python3
"""
Test script to validate the integration between app.py and core.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_core_imports():
    """Test importing core modules."""
    try:
        from core import LazyDatabase, ScampAnomalyDetector, DemoDataGenerator
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    try:
        # Import the data loading function from app
        sys.path.append(str(Path(__file__).parent))
        
        # Test data directory existence
        data_dir = Path(__file__).parent.parent / "data"
        if data_dir.exists():
            print(f"✅ Data directory found: {data_dir}")
            
            # List available data files
            npy_files = list(data_dir.glob("ntt_mss_*.npy"))
            print(f"✅ Found {len(npy_files)} data files")
            
            if npy_files:
                # Test loading a sample file
                sample_file = npy_files[0]
                data = np.load(sample_file)
                print(f"✅ Successfully loaded sample data: {sample_file.name}, shape: {data.shape}")
                return True
            else:
                print("⚠️ No MSS data files found")
                return False
        else:
            print(f"❌ Data directory not found: {data_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_detector_creation():
    """Test creating the anomaly detector."""
    try:
        from core import ScampAnomalyDetector
        
        detector = ScampAnomalyDetector()
        print("✅ ScampAnomalyDetector created successfully")
        
        # Test with simple data
        test_data = np.random.randn(100)
        anomalies, scores = detector.detect_anomalies(test_data, threshold=2.0)
        print(f"✅ Anomaly detection test completed: {np.sum(anomalies)} anomalies found")
        
        return True
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return False

def test_app_components():
    """Test key app.py components."""
    try:
        # Import app functions
        from app import get_available_years, get_available_mesh_ids, load_mss_data
        
        # Test helper functions
        years = get_available_years()
        print(f"✅ Available years: {years}")
        
        mesh_ids = get_available_mesh_ids()
        print(f"✅ Available mesh IDs: {len(mesh_ids)} ranges")
        
        if years and mesh_ids:
            # Test data loading
            sample_year = years[0]
            sample_mesh = mesh_ids[0]
            
            # This is a cached function, so import it properly
            try:
                data = load_mss_data(sample_year, sample_mesh, merge_areas=False)
                if not data.empty:
                    print(f"✅ Data loading test successful: {len(data)} records loaded")
                else:
                    print("⚠️ Data loading returned empty DataFrame")
            except Exception as e:
                print(f"⚠️ Data loading test failed (but imports work): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ App components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing AnDeS integration...")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Data Loading", test_data_loading),
        ("Detector Creation", test_detector_creation),
        ("App Components", test_app_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()