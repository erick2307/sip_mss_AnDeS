# Anomaly Score Heatmap Feature

## Overview
Added a new visualization function `create_anomaly_score_heatmap()` that displays anomaly scores as continuous values in a heatmap format, complementing the existing binary anomaly detection heatmap.

## New Function: `create_anomaly_score_heatmap()`

### Purpose
- Displays anomaly scores as continuous values instead of binary detection
- Provides more detailed visualization of anomaly intensity
- Shows the full spectrum of anomaly scores across time and date

### Key Features

#### Visual Design
- **Colormap**: Uses `viridis` colormap for better continuous value representation
- **Matrix Structure**: Same as original heatmap (rows = days, columns = hours 0-23)
- **Missing Data**: NaN values are masked and appear transparent/white
- **Grid Lines**: Same grid structure as original heatmap for consistency

#### Enhanced Information Display
- **Threshold Overlay**: Shows detection threshold value when available
- **Score Statistics**: Displays min, max, mean scores and threshold comparison
- **Mesh Information**: Includes mesh ID in title when available
- **Date Range**: Shows analysis date range in title

#### Integration
- **Placement**: Appears right after the binary anomaly heatmap
- **Section Header**: "🌡️ Anomaly Score Heatmap"
- **Automatic Display**: Triggered when anomalies are detected (same condition as binary heatmap)

## Implementation Details

### Function Signature
```python
def create_anomaly_score_heatmap(data: pd.DataFrame):
    """Create a heatmap visualization showing anomaly scores as continuous values."""
```

### Required Data Columns
- `timestamp`: DateTime column for temporal grouping
- `anomaly_score`: Continuous anomaly score values
- `mesh_id` (optional): For title display

### Matrix Creation Logic
```python
# Initialize matrix with NaN (no data)
score_matrix = np.full((len(unique_dates), 24), np.nan)

# Fill with actual anomaly scores
for i, date in enumerate(unique_dates):
    day_data = data_copy[data_copy['date'] == date]
    for _, row in day_data.iterrows():
        hour = row['hour']
        if 0 <= hour <= 23:
            score_matrix[i, hour] = row['anomaly_score']
```

### Visualization Features
- **Continuous Scale**: Shows full range of anomaly score values
- **Colorbar**: Labeled with "Anomaly Score" and continuous scale
- **Threshold Display**: Text overlay showing threshold value when available
- **Grid Lines**: Consistent with original heatmap design

## User Experience

### Display Order
1. **Time Series Plot**: Standard anomaly detection visualization
2. **Binary Anomaly Heatmap**: Original "🗓️ Anomaly Heatmap" 
3. **Score Heatmap**: New "🌡️ Anomaly Score Heatmap" ← **NEW**
4. **Anomaly Details Table**: Detailed anomaly information

### Information Provided
```
📊 Score Heatmap Summary: Min: X.XX, Max: Y.YY, Mean: Z.ZZ | N scores above threshold (T.TT)
```

### Benefits for Users
- **Intensity Visualization**: See relative strength of anomalies
- **Pattern Recognition**: Identify gradual vs. sharp anomaly patterns
- **Threshold Analysis**: Understand how scores relate to detection threshold
- **Continuous Insight**: Beyond binary detection to score magnitude

## Technical Implementation

### Error Handling
- Checks for required columns (`anomaly_score`)
- Handles empty data gracefully
- Provides informative error messages
- Manages matplotlib figure cleanup

### Performance Considerations
- Efficient matrix creation using vectorized operations
- Proper memory management with `plt.close(fig)`
- NaN masking for sparse data handling

### Integration Points
- **Called from**: `display_detection_results()` function
- **Condition**: When `total_anomalies > 0` (same as binary heatmap)
- **Session State**: Uses `st.session_state.detection_threshold` when available

## Code Changes

### Files Modified
- `app.py`: Added `create_anomaly_score_heatmap()` function
- `app.py`: Updated `display_detection_results()` to call new function

### New Dependencies
- Uses existing imports (matplotlib, numpy, pandas, streamlit)
- No additional dependencies required

## Testing
- Comprehensive test suite in `test_score_heatmap.py`
- Tests data processing, visualization, and integration
- Validates matrix creation and matplotlib rendering

## Example Output
The score heatmap shows:
- **Dark colors**: Lower anomaly scores (normal behavior)
- **Bright colors**: Higher anomaly scores (anomalous behavior)  
- **White/transparent**: No data available
- **Threshold line**: Reference point for anomaly detection
- **Continuous scale**: Full spectrum of score values

This provides users with a more nuanced view of their data's anomaly patterns beyond simple binary detection.