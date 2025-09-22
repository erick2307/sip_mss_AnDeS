#!/usr/bin/env python3
"""
Test script to verify the get_implementation_info method fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_implementation_info():
    """Test if the get_implementation_info method works correctly."""
    try:
        from app import RealTimeAnomalyDetector
        
        print("🔧 Testing RealTimeAnomalyDetector.get_implementation_info()...")
        
        # Create detector instance
        detector = RealTimeAnomalyDetector()
        
        # Test get_implementation_info before any detector is created
        info = detector.get_implementation_info()
        print("✅ Method exists and returns info!")
        print(f"   Available implementations: {info['available_implementations']}")
        print(f"   Selected implementation: {info['selected_implementation']}")
        print(f"   Use left matrix profile: {info['use_left_mp']}")
        print(f"   Window size: {info['window_size']}")
        print(f"   Normalize: {info['normalize']}")
        print(f"   Threshold method: {info['threshold_method']}")
        
        return True
        
    except AttributeError as e:
        print(f"❌ AttributeError still exists: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_implementation_info()
    if success:
        print("\n🎉 Fix successful! The get_implementation_info method is working correctly.")
    else:
        print("\n💥 Fix failed! The error still exists.")