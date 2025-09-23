# Complete -1 to np.nan Pipeline Implementation

## Overview
Implemented comprehensive handling of -1 values (no data indicators) throughout the entire application pipeline, ensuring consistent data representation from loading through visualization and analysis.

## Root Issue Identified
The original implementation only converted -1 to np.nan right before matrix profile computation, but left -1 values in:
- Data preview tables
- Visualization plots
- Statistical calculations
- Anomaly detection result displays

## Complete Solution Implemented

### 1. Early Data Preprocessing (After Data Loading)

**Location**: `real_time_analysis()` function, right after `load_mss_data_by_date_range()`

**Implementation**:
```python
if not data.empty:
    # Convert -1 values to np.nan for proper handling throughout the app
    value_col = 'population' if 'population' in data.columns else 'value'
    original_minus_ones = (data[value_col] == -1).sum()
    data[value_col] = data[value_col].replace(-1, np.nan)
    log_message(f"Data preprocessing: converted {original_minus_ones} instances of -1 to np.nan for proper visualization and analysis", "info")
    
    st.session_state.current_data = data
    st.session_state.data_loaded = True
```

**Benefits**:
- All downstream processing works with clean np.nan data
- Single point of conversion ensures consistency
- Proper logging for transparency

### 2. Updated Statistics Calculations

**Real-time Analysis Statistics**:
```python
# Calculate no data entries (missing data or NaN values)
no_data_count = filtered_data[value_col].isna().sum()

# Use nanmin and nanmax to ignore NaN values
min_val = np.nanmin(filtered_data[value_col])
max_val = np.nanmax(filtered_data[value_col])
```

**Historical Analysis Statistics**:
```python
# Use nan-aware functions to handle NaN values
st.write(f"**Mean:** {np.nanmean(filtered_data[value_col]):.2f}")
st.write(f"**Std Dev:** {np.nanstd(filtered_data[value_col]):.2f}")
st.write(f"**Min:** {np.nanmin(filtered_data[value_col]):.2f}")
st.write(f"**Max:** {np.nanmax(filtered_data[value_col]):.2f}")
```

### 3. Matrix Profile Preprocessing Update

**Updated `detect_anomalies()` method**:
```python
# Note: -1 values are already converted to np.nan during data loading
# Check for any remaining -1 values (should be none)
remaining_minus_ones = np.sum(values == -1) if len(values) > 0 else 0
if remaining_minus_ones > 0:
    log_message(f"Warning: Found {remaining_minus_ones} unexpected -1 values in processed data", "warning")
    values = np.where(values == -1, np.nan, values)
```

**Benefits**:
- Validation that preprocessing worked correctly
- Fallback handling for edge cases
- Clear logging of any issues

### 4. Visualization Improvements

**Data Preview Table**:
- Now shows NaN instead of -1 values
- Streamlit dataframe automatically formats NaN appropriately

**Plotly Charts**:
- NaN values create natural gaps in line plots
- No artificial -1 spikes in visualizations
- Cleaner, more accurate representations

**Anomaly Detection Results**:
- Normal data points no longer show -1 values
- Anomaly heatmaps handle NaN properly
- Statistical summaries exclude missing data correctly

## Data Flow (Updated)

```
Raw MSS Data (with -1 for no data)
           ↓
Multi-mesh Aggregation (if enabled)
  ├─ Replace -1 with np.nan during aggregation
  ├─ Use np.nansum for proper totals
  └─ Restore -1 if all values are NaN
           ↓
Data Loading Completion
           ↓
Early Preprocessing (NEW)
  ├─ Convert any remaining -1 to np.nan
  ├─ Store clean data in session state
  └─ Log conversion count
           ↓
All Downstream Processing
  ├─ Data Preview: Clean NaN display
  ├─ Visualizations: Natural gap handling
  ├─ Statistics: NaN-aware calculations
  └─ Matrix Profile: Already clean data
           ↓
Anomaly Detection Results
  └─ Consistent NaN handling throughout
```

## UI/UX Improvements

### Before the Complete Fix
- ❌ Data preview showed -1 values in tables
- ❌ Plots had artificial spikes at -1
- ❌ Statistics included -1 in min/max calculations
- ❌ Anomaly results displayed -1 as normal data
- ❌ Inconsistent representation across different views

### After the Complete Fix
- ✅ Data preview shows clean NaN values
- ✅ Plots have natural gaps where data is missing
- ✅ Statistics properly exclude missing data
- ✅ Anomaly results show only valid data points
- ✅ Consistent representation throughout the application

## Technical Benefits

### Data Integrity
- Single source of truth for missing data representation
- Consistent handling across all application components
- No contamination of statistical calculations

### Algorithm Compatibility
- Matrix profile algorithms receive properly formatted data
- NaN handling follows scientific computing best practices
- Aggregation functions work correctly with missing data

### User Experience
- Clear visual representation of missing data
- No confusing -1 values in displays
- Accurate statistical summaries
- Professional data visualization standards

## Validation

### Key Test Cases
1. **Data Loading**: -1 values converted immediately after loading
2. **Statistics**: NaN-aware functions produce correct results
3. **Visualization**: Clean plots without artificial -1 spikes
4. **Matrix Profile**: Receives pre-processed clean data
5. **Aggregation**: Consistent missing data handling

### Expected Behavior
- Data preview tables show NaN instead of -1
- Plots have gaps where data is missing (not -1 spikes)
- Statistics exclude missing data from calculations
- Anomaly detection works with clean input data
- All UI components show consistent data representation

## Logging Enhancement

The application now provides clear feedback about data preprocessing:
```
Data preprocessing: converted X instances of -1 to np.nan for proper visualization and analysis
```

This ensures users understand how missing data is being handled and can verify the conversion process.

## Backward Compatibility

- No breaking changes to existing functionality
- API remains unchanged
- Data files are not modified
- Session state structure preserved
- All existing features continue to work

## Impact Summary

This comprehensive implementation ensures that -1 values (MSS no-data indicators) are properly converted to np.nan at the earliest stage of data processing, providing:

1. **Clean Data Visualization**: No more -1 artifacts in plots and tables
2. **Accurate Statistics**: Proper exclusion of missing data from calculations
3. **Consistent UI**: Uniform representation across all application views
4. **Algorithm Compatibility**: Matrix profile functions receive properly formatted data
5. **Professional Standards**: Follows scientific computing best practices for missing data

The solution provides a complete, systematic approach to handling missing data throughout the entire application lifecycle.