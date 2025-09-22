"""
Simple validation script to test if the app will run
"""

try:
    print("Testing imports...")
    
    # Test basic imports
    import streamlit as st
    print("✅ Streamlit imported")
    
    import pandas as pd
    import numpy as np
    print("✅ Core data libraries imported")
    
    import plotly.graph_objects as go
    import plotly.express as px
    print("✅ Plotly imported")
    
    # Test core module import
    from core import LazyDatabase, ScampAnomalyDetector, DemoDataGenerator
    print("✅ Core modules imported")
    
    # Test creating instances
    lazy_db = LazyDatabase()
    print("✅ LazyDatabase created")
    
    detector = ScampAnomalyDetector()
    print("✅ ScampAnomalyDetector created")
    
    # Test app functions
    import app
    print("✅ App module imported")
    
    print("\n🎉 All imports successful! App should run correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()