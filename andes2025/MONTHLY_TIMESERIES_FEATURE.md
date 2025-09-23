# Monthly Time Series Feature Implementation

## Overview

This document describes the implementation of the new monthly time series logic for the AnDeS anomaly detection app. The changes transform the app from using continuous date ranges to using targeted monthly periods from multiple years.

## Implementation Details

### New Logic Flow

1. **User Input**: User selects a single event date and time (instead of start/end date range)
2. **Year Analysis**: System analyzes years from 2016 to the event year
3. **Monthly Extraction**: For each year, extract exactly one month of data FROM one month before TO the event date/time
4. **Time Series Concatenation**: All monthly periods are concatenated into one continuous time series
5. **Analysis**: Standard anomaly detection proceeds on this concatenated time series

### Key Functions

#### `load_mss_data_monthly_timeseries(event_datetime, mesh_id_list, multi_mesh_analysis)`

This is the core function implementing the new logic:

- **Input**: Event datetime, mesh IDs, and analysis options
- **Process**: 
  - Calculates event dates for each year (same month/day/hour as original event)
  - Calculates one month before each event date
  - Extracts data FROM one month before TO the event date (approximately 30-31 days)
  - Handles edge cases (leap years, month boundaries)
  - Concatenates all periods with year tracking
- **Output**: DataFrame with timestamp, population, mesh_id, and year_source columns

#### Edge Case Handling

- **Leap Year**: February 29 events use February 28 for non-leap years
- **Month Boundaries**: January events look at December of the previous year
- **Day Mismatches**: March 31 → February uses last day of February
- **Data Availability**: Gracefully handles missing data for specific years

### UI Changes

#### Before
```
Date Range Selection:
- Start Date: [date picker]
- End Date: [date picker]
```

#### After
```
Event Date & Time:
- Event Date: [date picker]
- Event Time: [time picker]
Analysis Info: 1-month periods from X years (2016-YYYY)
```

### Data Structure Changes

#### Before
```
DataFrame columns: timestamp, population, mesh_id
Time series: Continuous from start_date to end_date
```

#### After
```
DataFrame columns: timestamp, population, mesh_id, year_source
Time series: Concatenated monthly periods from multiple years
```

### Configuration Changes

The configuration now stores:
- `event_datetime`: Single datetime instead of start/end dates
- `analysis_years`: List of years to analyze instead of date ranges
- All other parameters remain the same

## Example Usage

### Event: January 1, 2024 at 16:00

The system will extract:
- **2016**: Dec 1, 2015 16:00 - Jan 1, 2016 16:00 (1 month period)
- **2017**: Dec 1, 2016 16:00 - Jan 1, 2017 16:00 (1 month period)
- **2018**: Dec 1, 2017 16:00 - Jan 1, 2018 16:00 (1 month period)
- ...
- **2024**: Dec 1, 2023 16:00 - Jan 1, 2024 16:00 (1 month period)

All these monthly periods are concatenated into one time series for analysis.
Each period is approximately 30-31 days depending on the month.

## Preserved Features

✅ **Data Preprocessing**: -1 to np.nan conversion remains<br>
✅ **Event Selection**: Predefined events still work<br>
✅ **Multi-mesh Analysis**: Aggregation functionality preserved<br>
✅ **Detection Parameters**: All thresholds and settings unchanged<br>
✅ **Visualizations**: All charts and analysis views remain<br>
✅ **Matrix Profile**: SCAMP/STUMPY implementations unchanged<br>

## Benefits

1. **Targeted Analysis**: Focuses on relevant time patterns around the event
2. **Historical Context**: Uses multiple years of data for better baseline
3. **Seasonal Relevance**: Maintains seasonal patterns by using same month/day/hour
4. **Improved Detection**: More relevant historical data should improve anomaly detection accuracy

## Migration Notes

- Existing event definitions work automatically
- Custom mesh selection still supported
- Historical analysis tab adapted to work with new time series structure
- All other functionality remains backward compatible

## Testing

The implementation includes:
- Comprehensive error handling
- Edge case management (leap years, month boundaries)
- Logging for debugging and monitoring
- Validation of time series construction

## Future Enhancements

Potential improvements could include:
- Configurable time window size (currently ±15 days)
- Multiple event analysis
- Advanced seasonal adjustment options
- Performance optimizations for large datasets