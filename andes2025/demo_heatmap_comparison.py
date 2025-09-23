#!/usr/bin/env python3
"""
Demonstration script showing the difference between the two heatmap visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_demo_comparison():
    """Create a side-by-side comparison of both heatmap types."""
    print("🎨 Creating heatmap comparison demo...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=72, freq='h')  # 3 days
    
    # Create sample data with varying patterns
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'population': np.random.randint(100, 1000, len(dates)),
        'mesh_id': ['533946403'] * len(dates)
    })
    
    # Generate realistic anomaly scores (exponential distribution)
    sample_data['anomaly_score'] = np.random.exponential(5, len(dates))
    
    # Create some high-score anomalies
    anomaly_indices = np.random.choice(len(dates), size=10, replace=False)
    sample_data.loc[anomaly_indices, 'anomaly_score'] = np.random.uniform(15, 50, len(anomaly_indices))
    
    # Binary anomaly detection (scores above threshold)
    threshold = 12.0
    sample_data['detected_anomaly'] = sample_data['anomaly_score'] > threshold
    
    # Process data for visualization
    data_copy = sample_data.copy()
    data_copy['hour'] = data_copy['timestamp'].dt.hour
    data_copy['date'] = data_copy['timestamp'].dt.date
    unique_dates = sorted(data_copy['date'].unique())
    
    # Create both matrices
    binary_matrix = np.zeros((len(unique_dates), 24))
    score_matrix = np.full((len(unique_dates), 24), np.nan)
    
    for i, date in enumerate(unique_dates):
        day_data = data_copy[data_copy['date'] == date]
        for _, row in day_data.iterrows():
            hour = row['hour']
            if 0 <= hour <= 23:
                # Binary matrix
                if row['detected_anomaly']:
                    binary_matrix[i, hour] = 2  # Anomaly
                else:
                    binary_matrix[i, hour] = 1  # Normal
                
                # Score matrix
                score_matrix[i, hour] = row['anomaly_score']
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Binary heatmap (left)
    im1 = ax1.imshow(binary_matrix, cmap='CMRmap_r', aspect=0.7, interpolation='nearest')
    ax1.set_title('Original: Binary Anomaly Detection Heatmap\n(0=No Data, 1=Normal, 2=Anomaly)', fontsize=14)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Date')
    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f"{i:02d}" for i in range(24)])
    ax1.set_yticks(range(len(unique_dates)))
    ax1.set_yticklabels([str(date) for date in unique_dates])
    
    # Add grid for binary
    for i in range(1, 24):
        ax1.axvline(i-0.5, color='black', linewidth=0.3)
    for i in range(1, len(unique_dates)):
        ax1.axhline(i-0.5, color='black', linewidth=0.3)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_ticks([0, 1, 2])
    cbar1.set_ticklabels(['No Data', 'Normal', 'Anomaly'])
    
    # Score heatmap (right)
    masked_scores = np.ma.masked_where(np.isnan(score_matrix), score_matrix)
    im2 = ax2.imshow(masked_scores, cmap='viridis', aspect=0.7, interpolation='nearest')
    ax2.set_title('New: Continuous Anomaly Score Heatmap\n(Continuous values with viridis colormap)', fontsize=14)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Date')
    ax2.set_xticks(range(24))
    ax2.set_xticklabels([f"{i:02d}" for i in range(24)])
    ax2.set_yticks(range(len(unique_dates)))
    ax2.set_yticklabels([str(date) for date in unique_dates])
    
    # Add grid for scores
    for i in range(1, 24):
        ax2.axvline(i-0.5, color='black', linewidth=0.3)
    for i in range(1, len(unique_dates)):
        ax2.axhline(i-0.5, color='black', linewidth=0.3)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Anomaly Score', rotation=270, labelpad=15)
    
    # Add threshold line to score heatmap
    ax2.text(0.02, 0.98, f'Threshold: {threshold:.1f}', 
             transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('heatmap_comparison_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"📊 Demo Data Statistics:")
    print(f"   - Total data points: {len(sample_data)}")
    print(f"   - Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"   - Anomaly threshold: {threshold}")
    print(f"   - Binary anomalies detected: {sample_data['detected_anomaly'].sum()}")
    print(f"   - Score range: {sample_data['anomaly_score'].min():.2f} to {sample_data['anomaly_score'].max():.2f}")
    print(f"   - Score mean: {sample_data['anomaly_score'].mean():.2f}")
    print(f"✅ Comparison saved as 'heatmap_comparison_demo.png'")
    
    return True

def main():
    """Create the demo comparison."""
    print("=" * 60)
    print("HEATMAP COMPARISON DEMO")
    print("=" * 60)
    
    try:
        create_demo_comparison()
        print("\n🎉 Demo created successfully!")
        print("\nKey Differences:")
        print("📍 Binary Heatmap (Left):")
        print("   - Shows only detected/not detected")
        print("   - Uses CMRmap_r colormap") 
        print("   - 3 discrete values: No Data, Normal, Anomaly")
        print("\n🌡️ Score Heatmap (Right):")
        print("   - Shows continuous anomaly score values")
        print("   - Uses viridis colormap")
        print("   - Full spectrum of anomaly intensity")
        print("   - Threshold reference displayed")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())