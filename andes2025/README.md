# ANDES 2025 - ANomaly DEtection System

A sophisticated web application for real-time anomaly detection in Mobile Spatial Statistics (MSS) data using the SCAMP (Scalable Matrix Profile) algorithm.

## 🌟 Overview

ANDES 2025 is a Streamlit-based web application that provides real-time anomaly detection capabilities for time-series data, specifically designed for Mobile Spatial Statistics analysis. The system uses matrix profile algorithms to identify unusual patterns and potential anomalies in population movement data.

### Key Features

- 🔴 **Real-Time Anomaly Detection**: Live monitoring with adjustable sensitivity
- 📊 **Interactive Visualizations**: Plotly-powered charts with zoom, pan, and hover capabilities
- 📈 **Historical Analysis**: Comprehensive analysis of past data with trend identification
- 💾 **Memory-Efficient Processing**: Lazy loading system for handling large datasets
- 🎛️ **Customizable Parameters**: Adjustable detection sensitivity and time windows
- 📱 **Responsive Design**: Mobile-friendly interface with intuitive navigation
- 🔧 **System Monitoring**: Real-time system status and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/erick2307/sip_mss_AnDeS.git
   cd sip_mss_AnDeS/andes2025
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
andes2025/
├── app.py                  # Main Streamlit application
├── core.py                 # Core anomaly detection classes
├── utils.py                # Utility functions and helpers
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── andes_scamp.ipynb      # Reference Jupyter notebook
├── config/                # Configuration files (optional)
│   └── settings.yaml
├── data/                  # Data directory (auto-created)
│   └── cache/            # Cache directory
└── logs/                  # Application logs (auto-created)
```

## 🎯 Application Features

### 1. Real-Time Analysis Tab

- **Live Data Processing**: Continuous monitoring with configurable update intervals
- **Anomaly Detection**: Real-time identification of unusual patterns
- **Interactive Controls**:
  - Detection sensitivity slider (0.1 - 5.0)
  - Batch size adjustment (100 - 10,000 points)
  - Subsequence length configuration (50 - 500)
- **Visualizations**:
  - Time series plot with anomaly markers
  - Anomaly score visualization
  - Real-time statistics

### 2. Historical Analysis Tab

- **Data Upload**: Support for CSV and NPY files
- **Batch Processing**: Efficient analysis of large historical datasets
- **Analysis Options**:
  - Custom date range selection
  - Multiple aggregation methods
  - Trend analysis
- **Advanced Visualizations**:
  - Multi-panel time series plots
  - Distribution analysis
  - Daily pattern recognition
  - Anomaly summary reports

### 3. System Status Tab

- **Performance Monitoring**: Real-time system metrics
- **Data Statistics**: Current dataset information
- **Cache Management**: Memory usage and cache statistics
- **Configuration Display**: Current parameter settings
- **Health Checks**: System component status

### 4. Documentation Tab

- **User Guide**: Comprehensive usage instructions
- **Algorithm Information**: Technical details about SCAMP
- **Parameter Explanations**: Detailed parameter descriptions
- **Troubleshooting**: Common issues and solutions
- **API Reference**: Technical documentation

## ⚙️ Configuration

### Basic Configuration

The application can be configured through the Streamlit interface or by modifying the configuration files:

#### Detection Parameters

- **Sensitivity**: Controls the threshold for anomaly detection (lower = more sensitive)
- **Subsequence Length**: Length of the pattern window for comparison
- **Batch Size**: Number of data points processed in each batch

#### Data Parameters

- **Cache Size**: Maximum number of data batches to keep in memory
- **Update Interval**: Frequency of real-time updates (seconds)
- **File Format**: Supported formats (CSV, NPY)

### Advanced Configuration

Create a `config/settings.yaml` file for advanced settings:

```yaml
# Detection Settings
detection:
  default_sensitivity: 2.0
  min_subsequence_length: 50
  max_subsequence_length: 500
  default_batch_size: 1000

# Data Settings
data:
  cache_size: 100
  supported_formats: ['.csv', '.npy']
  default_data_path: './data'

# UI Settings
ui:
  update_interval: 1
  max_plot_points: 10000
  theme: 'plotly_white'

# Performance Settings
performance:
  use_parallel_processing: true
  max_workers: 4
  memory_limit_mb: 1024
```

## 📊 Data Format

### Input Data Requirements

The application expects time-series data in one of the following formats:

#### CSV Format
```csv
timestamp,value
2024-01-01 00:00:00,123.45
2024-01-01 01:00:00,124.67
2024-01-01 02:00:00,122.89
...
```

#### NPY Format
- NumPy arrays with shape `(n_samples,)` for values
- Corresponding timestamp arrays (optional)
- Files should be named with pattern: `data_YYYYMMDD.npy`

### Supported Data Types

- **Mobile Spatial Statistics (MSS)**: Population movement data
- **Time Series Data**: Any sequential numerical data
- **Event Data**: Timestamped measurements
- **Sensor Data**: IoT device readings

## 🔍 Algorithm Details

### SCAMP (Scalable Matrix Profile)

The application uses the SCAMP algorithm for anomaly detection:

1. **Matrix Profile Computation**: Calculates the similarity between all subsequences
2. **Distance Calculation**: Computes normalized Euclidean distances
3. **Anomaly Scoring**: Identifies subsequences with high distances
4. **Threshold Application**: Flags anomalies based on configurable sensitivity

### Performance Optimizations

- **Lazy Loading**: Data is loaded on-demand to minimize memory usage
- **Batch Processing**: Large datasets are processed in configurable chunks
- **Caching**: Computed results are cached for faster subsequent access
- **Parallel Processing**: Multi-core processing for improved performance

## 🛠️ Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem**: "MemoryError" when processing large datasets
**Solution**: 
- Reduce batch size in the settings
- Enable lazy loading
- Increase system memory or use smaller datasets

#### 2. Slow Performance
**Problem**: Application runs slowly with large datasets
**Solution**:
- Reduce the number of data points displayed
- Enable caching
- Use smaller subsequence lengths
- Enable parallel processing

#### 3. No Anomalies Detected
**Problem**: No anomalies are found in the data
**Solution**:
- Decrease sensitivity (lower values = more sensitive)
- Adjust subsequence length
- Check data quality and format

#### 4. Import Errors
**Problem**: Missing dependencies or import failures
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Performance Tips

1. **Optimize Batch Size**: Start with 1000 points and adjust based on performance
2. **Use Appropriate Sensitivity**: Start with 2.0 and fine-tune
3. **Enable Caching**: Helps with repeated analysis
4. **Monitor Memory**: Keep an eye on system resources

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

2. **Code formatting**:
   ```bash
   black *.py
   ```

3. **Linting**:
   ```bash
   flake8 *.py
   ```

### Adding New Features

1. **Core Logic**: Add new detection algorithms in `core.py`
2. **UI Components**: Extend the Streamlit interface in `app.py`
3. **Utilities**: Add helper functions in `utils.py`
4. **Configuration**: Update settings in `config/settings.yaml`

## 📚 Technical Documentation

### Core Classes

#### `LazyDatabase`
- **Purpose**: Memory-efficient data loading and caching
- **Key Methods**: `load_data()`, `get_batch()`, `clear_cache()`

#### `ScampAnomalyDetector`
- **Purpose**: SCAMP-based anomaly detection
- **Key Methods**: `detect_anomalies()`, `compute_matrix_profile()`

#### `StreamlitAnomalyDetector`
- **Purpose**: Streamlit interface wrapper
- **Key Methods**: `run_real_time()`, `analyze_historical()`

### API Reference

Detailed API documentation is available in the application's Documentation tab.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

For support and questions:

1. **Documentation**: Check the in-app documentation tab
2. **Issues**: Open an issue on GitHub
3. **Discussions**: Use GitHub Discussions for general questions

## 🔮 Future Enhancements

### Planned Features

- **Multi-variate Analysis**: Support for multiple data streams
- **Machine Learning Integration**: Advanced ML-based anomaly detection
- **Real-time Alerts**: Email/SMS notifications for critical anomalies
- **Data Export**: Enhanced export capabilities with various formats
- **Dashboard Customization**: User-configurable dashboard layouts
- **API Endpoints**: REST API for programmatic access

### Performance Improvements

- **GPU Acceleration**: CUDA support for large-scale processing
- **Distributed Computing**: Support for cluster-based processing
- **Advanced Caching**: Redis/Memcached integration
- **Streaming Data**: Real-time data ingestion from external sources

---

**ANDES 2025** - Empowering data-driven decisions through advanced anomaly detection.

*Built with ❤️ using Python, Streamlit, and SCAMP*