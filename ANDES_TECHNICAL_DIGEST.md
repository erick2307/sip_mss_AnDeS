# AnDeS - ANomaly DEtection System: Technical Digest

## Executive Summary

AnDeS (ANomaly DEtection System) is a sophisticated web-based application designed for real-time anomaly detection in Mobile Spatial Statistics (MSS) data using matrix profile algorithms. The system analyzes population movement patterns to detect unusual events such as earthquakes, emergencies, and large-scale gatherings.

## System Architecture

### Core Components

1. **Frontend Interface** (`app.py`)
   - Streamlit-based web application
   - Interactive dashboard with multiple analysis tabs
   - Real-time visualization and configuration

2. **Anomaly Detection Engine** (`core.py`)
   - Matrix profile-based anomaly detection
   - Multiple algorithm implementations (SCAMP, STUMPY, custom)
   - Intelligent caching and memory management

3. **Data Management**
   - Lazy loading database system
   - Efficient memory management with configurable cache
   - Support for multi-year time series analysis

## Input Data Structure

### Primary Data Sources
- **MSS Data Files**: `ntt_mss_{year}.npy` - Population density time series
- **Area Mapping Files**: `ntt_mss_{year}_areas.npy` - Geographic mesh code mapping
- **Data Coverage**: 2016-2025 (hourly resolution)

### Data Characteristics
- **Temporal Resolution**: Hourly measurements
- **Spatial Resolution**: Japanese mesh code system (geographic grid)
- **Data Format**: Numpy arrays with population counts
- **Missing Data Handling**: -1 values converted to np.nan for proper processing

## Core Algorithm Logic

### Matrix Profile Anomaly Detection

1. **Time Series Preparation**
   ```
   Input: MSS population data → Preprocessing → Normalized time series
   ```

2. **Matrix Profile Computation**
   ```
   For each subsequence in time series:
   - Calculate distance to all other subsequences
   - Find nearest neighbor distance
   - Build matrix profile of distances
   ```

3. **Anomaly Threshold Calculation**
   ```
   Threshold = μ + (δ × σ)
   Where: μ = mean, σ = standard deviation, δ = sensitivity multiplier
   ```

4. **Anomaly Detection**
   ```
   Anomaly = Matrix Profile[i] > Threshold
   ```

### Implementation Variants

- **PySCAMP**: GPU-accelerated matrix profile computation
- **STUMPY**: NumPy-based implementation with streaming capabilities
- **Custom**: Fallback implementation for compatibility

## Critical Parameters

### Detection Parameters
| Parameter | Description | Range | Default | Impact |
|-----------|-------------|-------|---------|---------|
| **Subsequence Length** | Pattern matching window (hours) | 6-168 | 24 | Shorter = local anomalies, Longer = trend anomalies |
| **Threshold Multiplier (δ)** | Sensitivity control | 1.0-5.0 | 3.0 | Lower = more sensitive, Higher = fewer false positives |
| **Normalize Matrix Profile** | Distance normalization | Boolean | False | Improves comparison across different scales |
| **Use Left Matrix Profile** | Only consider past data | Boolean | False | Real-time compatible mode |

### Data Configuration Parameters
| Parameter | Description | Options | Impact |
|-----------|-------------|---------|---------|
| **Year Range** | Analysis period | 2016-2025 | More years = better baseline |
| **Mesh ID Selection** | Geographic area | Custom/Predefined | Defines analysis region |
| **Multi-mesh Analysis** | Aggregate multiple regions | Boolean | Regional vs. local detection |
| **Date Range Type** | Analysis mode | Monthly/Custom | Monthly for events, Custom for trends |

## Monthly Time Series Logic

### Event-Based Analysis
```
For each year in range:
  Extract period: [1 month before event] → [event datetime]
  Concatenate all yearly periods
  Run anomaly detection on combined series
```

### Benefits
- **Seasonal Consistency**: Same calendar periods across years
- **Historical Context**: Multi-year baseline for comparison
- **Event Focus**: Targeted analysis around specific incidents

## User Interface Architecture

### Tab Structure

1. **📊 Real-time Analysis**
   - Data loading and preview
   - Parameter configuration
   - Anomaly detection execution
   - Interactive visualizations

2. **📈 Historical View**
   - Statistical summaries
   - Distribution analysis
   - Correlation studies
   - Pattern identification

3. **⚙️ System Status**
   - Performance monitoring
   - Memory/CPU usage
   - Library compatibility
   - Session management

4. **📚 Documentation**
   - Usage guidelines
   - Technical specifications
   - Troubleshooting guides

### Visualization Components

- **Time Series Plots**: Population vs. time with anomaly overlays
- **Anomaly Heatmaps**: Binary detection grid (day × hour)
- **Score Heatmaps**: Continuous anomaly score visualization
- **Statistical Charts**: Distribution plots and correlation matrices

## Data Flow Architecture

```
Raw MSS Data (.npy files)
    ↓
Lazy Database (cached loading)
    ↓
Data Preprocessing (-1 → np.nan)
    ↓
Time Series Construction (monthly/custom)
    ↓
Matrix Profile Computation (SCAMP/STUMPY/Custom)
    ↓
Threshold Calculation (statistical methods)
    ↓
Anomaly Detection (threshold comparison)
    ↓
Visualization & Results Export
```

## Memory Management Strategy

### Lazy Loading System
- **Cache Size**: Configurable (default: 2 years)
- **LRU Eviction**: Automatic removal of oldest data
- **On-demand Loading**: Data loaded only when needed
- **Progress Tracking**: Real-time loading status

### Memory Optimization
- **Vectorized Operations**: NumPy-based efficient computation
- **Sparse Data Handling**: NaN masking for missing data
- **Session State Management**: Persistent configuration across interactions

## Event Detection Capabilities

### Predefined Events
- **Natural Disasters**: Earthquakes, floods, typhoons
- **Transportation Incidents**: Airport collisions, evacuations
- **Large Gatherings**: Concerts, conventions, game shows
- **Geographic Coverage**: Tokyo, Osaka, Hiroshima, Hokkaido regions

### Event Configuration
- **Main Mesh Code**: Primary geographic focus
- **Surrounding Mesh Codes**: 3×3 grid around main location
- **Time Context**: Automatic date/time association
- **Analysis Period**: 1 month before + 1 week after event

## Technical Innovations

### NumPy 2.0 Compatibility
- **Automatic Detection**: Version checking and compatibility patches
- **Deprecated Constants**: Automatic fallback for np.NINF, np.PINF, np.NAN
- **Library Monitoring**: Real-time compatibility status reporting

### Adaptive Algorithm Selection
- **Auto Implementation**: Intelligent selection based on availability
- **Graceful Degradation**: Fallback from PySCAMP → STUMPY → Custom
- **Performance Monitoring**: Real-time processing metrics

### Real-time System Monitoring
- **Resource Tracking**: CPU, memory, disk usage
- **Performance Metrics**: Processing time, data points, anomaly counts
- **Session Logging**: Comprehensive activity tracking
- **Auto-refresh**: Live status updates

## Quality Assurance

### Data Validation
- **Range Checking**: Valid mesh IDs and date ranges
- **Completeness Verification**: Missing data identification
- **Consistency Checks**: Cross-year data alignment

### Error Handling
- **Graceful Degradation**: Fallback options for failed operations
- **User Feedback**: Clear error messages and suggestions
- **Recovery Mechanisms**: Automatic retry and alternative methods

### Testing Framework
- **Unit Tests**: Core functionality validation
- **Integration Tests**: End-to-end workflow verification
- **Performance Tests**: Memory and speed benchmarking

## Performance Characteristics

### Computational Complexity
- **Matrix Profile**: O(n²) for n time points (optimized implementations available)
- **Memory Usage**: ~2-4 GB for typical yearly datasets
- **Processing Time**: Minutes for monthly analysis, hours for full yearly analysis

### Scalability Considerations
- **Data Size**: Tested up to 10 years of hourly data
- **Geographic Coverage**: Supports nationwide mesh code analysis
- **Concurrent Users**: Single-user desktop application design

## Future Enhancement Opportunities

### Algorithm Improvements
- **Streaming Detection**: Real-time analysis for live data feeds
- **Multi-scale Analysis**: Simultaneous analysis at different temporal resolutions
- **Ensemble Methods**: Combining multiple detection algorithms

### Interface Enhancements
- **Mobile Optimization**: Touch-friendly controls and responsive design
- **Advanced Visualizations**: 3D mapping and temporal animations
- **Collaborative Features**: Multi-user analysis and sharing capabilities

### Data Integration
- **External Data Sources**: Weather, social media, traffic data integration
- **Real-time Feeds**: Live MSS data connection
- **Multi-modal Analysis**: Combining different data types for enhanced detection

## Deployment Considerations

### System Requirements
- **Python**: 3.8+ with scientific computing libraries
- **Memory**: 8GB+ RAM recommended for large datasets
- **Storage**: 50GB+ for full historical data
- **Network**: Moderate bandwidth for data loading

### Security Considerations
- **Data Privacy**: Local processing, no external data transmission
- **Access Control**: Single-user application model
- **Data Integrity**: Checksum verification for data files

This technical digest provides a comprehensive overview of the AnDeS system architecture, algorithms, and implementation details necessary for understanding the application's technical foundation and creating effective presentations about its capabilities.