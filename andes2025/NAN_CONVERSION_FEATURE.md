# -1 to np.nan Conversion Implementation

## Overview
Implemented proper handling of -1 values (no data indicators) in MSS time series data by converting them to np.nan before matrix profile computation and during multi-mesh aggregation.

## Changes Made

### 1. Matrix Profile Preprocessing (`detect_anomalies` method)

**Location**: Line ~575 in `app.py`

**Implementation**:
```python
# Use population or value column
value_col = 'population' if 'population' in data.columns else 'value'
values = data[value_col].values

# Replace -1 values (no data indicators) with np.nan before matrix profile computation
values = np.where(values == -1, np.nan, values)
log_message(f"Preprocessed data: replaced {np.sum(data[value_col] == -1)} instances of -1 with np.nan", "info")
```

**Benefits**:
- Matrix profile algorithms handle np.nan appropriately 
- Prevents -1 values from being treated as valid low population counts
- Maintains data integrity by preserving no-data indicators
- Provides logging of conversion for transparency

### 2. Multi-mesh Aggregation Improvement (`load_mss_data_single_year` method)

**Location**: Line ~462 in `app.py`

**Implementation**:
```python
if multi_mesh_analysis and len(extracted_data) > 1:
    # Aggregate data from all mesh IDs (sum), handling -1 values properly
    # Convert to numpy array for easier manipulation
    data_array = np.array(extracted_data)
    
    # Replace -1 with np.nan for proper aggregation
    data_array = np.where(data_array == -1, np.nan, data_array)
    
    # Sum ignoring NaN values (use nansum)
    aggregated_data = np.nansum(data_array, axis=0)
    
    # If all values for a time point are NaN, set result to -1 (no data)
    all_nan_mask = np.all(np.isnan(data_array), axis=0)
    aggregated_data = np.where(all_nan_mask, -1, aggregated_data)
    
    mesh_id_value = valid_mesh_ids[4] if len(valid_mesh_ids) > 4 else valid_mesh_ids[0]
    log_message(f"Aggregated data from {len(extracted_data)} mesh IDs (handling -1 values properly)", "info")
```

**Benefits**:
- Proper aggregation when some meshes have no data (-1) at certain time points
- Uses `np.nansum` to ignore missing data rather than including -1 in calculations
- Maintains -1 output when ALL meshes have no data for a time point
- Prevents incorrect population sums due to -1 contamination

## Problem Solved

### Before the Changes
- **Matrix Profile**: -1 values treated as valid low population counts, causing false anomaly detections
- **Multi-mesh Aggregation**: -1 values included in sums, creating artificially low totals
  - Example: Mesh A=100, Mesh B=-1, Mesh C=50 → Sum=149 (incorrect)

### After the Changes  
- **Matrix Profile**: -1 values converted to np.nan, properly handled by algorithms
- **Multi-mesh Aggregation**: -1 values ignored during summation
  - Example: Mesh A=100, Mesh B=-1, Mesh C=50 → Sum=150 (correct)
  - Example: All meshes=-1 → Result=-1 (preserves no-data indicator)

## Data Flow

```
Raw MSS Data (with -1 for no data)
           ↓
Multi-mesh Aggregation (if enabled)
  ├─ Replace -1 with np.nan
  ├─ Use np.nansum for aggregation  
  └─ Restore -1 if all values are NaN
           ↓
Matrix Profile Preprocessing
  ├─ Replace -1 with np.nan
  └─ Log conversion count
           ↓
Matrix Profile Computation
  └─ Algorithms handle np.nan properly
           ↓
Anomaly Detection Results
```

## Logging Enhancement

Both preprocessing steps now log their actions:
- Matrix profile: `"Preprocessed data: replaced X instances of -1 with np.nan"`
- Aggregation: `"Aggregated data from X mesh IDs (handling -1 values properly)"`

## Backward Compatibility

- No breaking changes to existing API
- Statistics calculations still count -1 as "no data" 
- Visualization functions unchanged
- Original data files remain unmodified

## Testing

Comprehensive test suite in `test_nan_conversion.py` validates:
- Basic -1 to np.nan conversion logic
- Multi-mesh aggregation with mixed -1/valid values
- Matrix profile preprocessing accuracy
- Edge cases (all -1 values, no -1 values)

## Impact

This change ensures that:
1. **Accurate anomaly detection**: No false positives from -1 values
2. **Correct aggregation**: Multi-mesh sums reflect actual population totals
3. **Data integrity**: No-data indicators preserved appropriately
4. **Algorithm compatibility**: np.nan handling follows scientific computing best practices