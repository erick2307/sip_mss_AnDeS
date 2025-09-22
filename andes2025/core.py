"""
Core anomaly detection classes extracted from the ANDES notebook.
Provides the main functionality for matrix profile-based anomaly detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import pyscamp
    USE_PYSCAMP = True
    print("✅ PySCAMP imported successfully!")
except ImportError:
    USE_PYSCAMP = False
    print("⚠️  PySCAMP not available. Using custom implementation...")

try:
    import stumpy
    
    # Handle NumPy 2.0 compatibility issues for STUMPY
    # STUMPY may use deprecated constants that were removed in NumPy 2.0
    numpy_2_compat_applied = False
    
    if not hasattr(np, 'NINF'):
        np.NINF = -np.inf
        numpy_2_compat_applied = True
    
    if not hasattr(np, 'PINF'):
        np.PINF = np.inf
        numpy_2_compat_applied = True
    
    if not hasattr(np, 'NAN'):
        np.NAN = np.nan
        numpy_2_compat_applied = True
    
    if numpy_2_compat_applied:
        print("🔧 Applied NumPy 2.0 compatibility patches for STUMPY")
    
    USE_STUMPY = True
    print("✅ STUMPY imported successfully!")
except ImportError as e:
    USE_STUMPY = False
    print(f"⚠️  STUMPY not available: {e}")
except Exception as e:
    USE_STUMPY = False
    print(f"⚠️  STUMPY import failed: {e}")

def check_numpy_stumpy_compatibility():
    """
    Check NumPy and STUMPY compatibility and provide guidance.
    
    Returns:
        dict: Compatibility information and recommendations
    """
    info = {
        'numpy_version': np.__version__,
        'numpy_major': int(np.__version__.split('.')[0]),
        'stumpy_available': USE_STUMPY,
        'compatibility_patches_applied': False,
        'recommendations': []
    }
    
    # Check for NumPy 2.0+ compatibility issues
    if info['numpy_major'] >= 2:
        deprecated_attrs = ['NINF', 'PINF', 'NAN']
        missing_attrs = [attr for attr in deprecated_attrs if not hasattr(np, attr)]
        
        if missing_attrs:
            info['recommendations'].append(
                f"NumPy 2.0+ detected. Missing deprecated attributes: {missing_attrs}"
            )
            info['recommendations'].append(
                "Consider updating STUMPY to a NumPy 2.0 compatible version"
            )
        else:
            info['compatibility_patches_applied'] = True
    
    if USE_STUMPY:
        try:
            import stumpy
            info['stumpy_version'] = stumpy.__version__
        except:
            info['stumpy_version'] = 'unknown'
    
    return info

class LazyDatabase:
    """
    Memory-efficient database class that loads data on-demand.
    
    Features:
    - Intelligent caching with configurable cache size
    - Automatic memory management
    - Lazy loading to avoid memory exhaustion
    - Progress tracking for data loading operations
    """
    
    def __init__(self, start=2016, stop=2025, max_cache_size=2):
        self.start = start
        self.stop = stop
        self.years = list(range(start, stop + 1))
        self._cache = {}
        self._max_cache_size = max_cache_size
        self._load_count = 0
        
    def keys(self):
        """Return available years."""
        return self.years
    
    def _manage_cache(self, year_to_add):
        """Manage cache size by removing oldest items if necessary."""
        if len(self._cache) >= self._max_cache_size and year_to_add not in self._cache:
            # Remove oldest item
            oldest_year = min(self._cache.keys())
            print(f"🗑️  Removing year {oldest_year} from cache to free memory")
            del self._cache[oldest_year]
    
    def __getitem__(self, year):
        """Load data for specified year (with caching)."""
        if year not in self._cache:
            # Manage cache before loading new data
            self._manage_cache(year)
            
            # Load the data with progress indication
            print(f"📥 Loading year {year} data...")
            
            try:
                # Load data and areas files
                y = np.load(f"data/ntt_mss_{year}.npy")
                mids = np.load(f"data/ntt_mss_{year}_areas.npy")
                
                # Store in cache
                self._cache[year] = (y, mids)
                self._load_count += 1
                
                # Show statistics
                data_size = y.nbytes / (1024**3)  # Size in GB
                print(f"✅ Year {year}: {y.shape} data points, {data_size:.1f} GB loaded")
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Data files for year {year} not found in 'data/' directory")
            except Exception as e:
                raise RuntimeError(f"Error loading data for year {year}: {str(e)}")
        
        return self._cache[year]
    
    def __contains__(self, year):
        """Check if year is available."""
        return year in self.years
    
    def get_cache_info(self):
        """Get information about current cache state."""
        cached_years = list(self._cache.keys())
        total_memory = sum(
            data[0].nbytes + data[1].nbytes 
            for data in self._cache.values()
        ) / (1024**3)  # Convert to GB
        
        return {
            'cached_years': cached_years,
            'cache_size': len(cached_years),
            'total_memory_gb': round(total_memory, 2),
            'loads_performed': self._load_count
        }

def custom_matrix_profile(T, m, normalize=False, use_left_mp=False):
    """
    Custom implementation of matrix profile calculation.
    
    Args:
        T: Time series data
        m: Subsequence length (window size)
        normalize: Whether to normalize the data
        use_left_mp: If True, only consider past data (j < i) for nearest neighbor search
        
    Returns:
        numpy.ndarray: Matrix profile distances
    """
    n = len(T)
    if n < m:
        raise ValueError(f"Time series length ({n}) must be >= window size ({m})")
    
    # Initialize matrix profile
    mp = np.full(n - m + 1, np.inf)
    
    # Calculate sliding window distances
    for i in tqdm(range(n - m + 1), desc="Computing distances"):
        query = T[i:i + m]
        
        # Skip if query contains NaN
        if np.any(np.isnan(query)):
            continue
        
        # Calculate distances to all other subsequences
        search_range = range(i) if use_left_mp else range(n - m + 1)
        
        for j in search_range:
            if abs(i - j) < m:  # Skip trivial matches
                continue
                
            candidate = T[j:j + m]
            
            # Skip if candidate contains NaN
            if np.any(np.isnan(candidate)):
                continue
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((query - candidate) ** 2))
            
            if normalize and len(candidate) > 0:
                distance = distance / m
            
            mp[i] = min(mp[i], distance)
    
    return mp

class ScampAnomalyDetector:
    """
    Anomaly detection using SCAMP, STUMPY, or custom matrix profile implementation.
    
    This class provides a unified interface for anomaly detection with multiple
    matrix profile backends and supports left matrix profile mode.
    """
    
    def __init__(self, window_size=3, normalize=False, threshold_method='sigma', 
                 implementation='auto', use_left_mp=False):
        """
        Initialize the anomaly detector.
        
        Args:
            window_size: Size of sliding window for pattern matching
            normalize: Whether to normalize matrix profile distances
            threshold_method: Method for determining anomaly threshold
            implementation: Which implementation to use ('auto', 'pyscamp', 'stumpy', 'custom')
            use_left_mp: Whether to use left matrix profile (only consider past data)
        """
        self.window_size = window_size
        self.normalize = normalize
        self.threshold_method = threshold_method
        self.implementation = implementation
        self.use_left_mp = use_left_mp
        self.mp_history = []
        self._stumpy_stream = None
        
        # Validate implementation choice
        if implementation == 'pyscamp' and not USE_PYSCAMP:
            raise ValueError("PySCAMP not available. Choose 'stumpy', 'custom', or 'auto'")
        if implementation == 'stumpy' and not USE_STUMPY:
            raise ValueError("STUMPY not available. Choose 'pyscamp', 'custom', or 'auto'")
    
    def _select_implementation(self):
        """Select the best available implementation."""
        if self.implementation == 'auto':
            if self.use_left_mp and USE_STUMPY:
                return 'stumpy'
            elif USE_PYSCAMP:
                return 'pyscamp'
            elif USE_STUMPY:
                return 'stumpy'
            else:
                return 'custom'
        return self.implementation
    
    def compute_matrix_profile(self, time_series):
        """
        Compute matrix profile using selected implementation.
        
        Args:
            time_series: Input time series data
            
        Returns:
            numpy.ndarray: Matrix profile distances
        """
        selected_impl = self._select_implementation()
        
        if selected_impl == 'pyscamp':
            return self._compute_with_pyscamp(time_series)
        elif selected_impl == 'stumpy':
            return self._compute_with_stumpy(time_series)
        else:
            return self._compute_with_custom(time_series)
    
    def _compute_with_stumpy(self, time_series):
        """Compute matrix profile using STUMPY."""
        try:
            print("🌟 Using STUMPY for matrix profile calculation...")
            
            # Prepare data for STUMPY
            T = np.array(time_series, dtype=np.float64)
            
            # Handle NaN values by interpolating
            if np.any(np.isnan(T)):
                mask = np.isnan(T)
                if np.all(mask):
                    raise ValueError("All values are NaN")
                T[mask] = np.interp(np.flatnonzero(mask), 
                                  np.flatnonzero(~mask), T[~mask])
            
            if self.use_left_mp:
                # Use streaming interface for left matrix profile
                print("🔄 Computing left matrix profile with STUMPY streaming...")
                
                # Initialize stream with first portion of data
                init_size = max(self.window_size * 2, min(100, len(T) // 2))
                T_init = T[:init_size]
                
                self._stumpy_stream = stumpy.stumpi(T_init, self.window_size, 
                                                  normalize=self.normalize, egress=False)
                
                # Stream remaining data
                for i in range(init_size, len(T)):
                    self._stumpy_stream.update(T[i])
                
                # Get left matrix profile
                mp = self._stumpy_stream.left_P_[self.window_size:]
                
                print(f"✅ STUMPY left matrix profile computation complete: {len(mp)} profile points")
            else:
                # Use standard matrix profile
                mp = stumpy.stump(T, self.window_size, normalize=self.normalize)[:, 0]
                print(f"✅ STUMPY standard matrix profile computation complete: {len(mp)} profile points")
            
            return mp
            
        except AttributeError as e:
            if 'NINF' in str(e):
                print(f"⚠️  STUMPY NumPy 2.0 compatibility issue detected: {str(e)}")
                print("💡 This is likely due to STUMPY using deprecated np.NINF. Consider updating STUMPY.")
                print("🔄 Falling back to custom implementation...")
            else:
                print(f"⚠️  STUMPY attribute error: {str(e)}")
                print("🔄 Falling back to custom implementation...")
            return self._compute_with_custom(time_series)
        except Exception as e:
            error_msg = str(e)
            if 'NINF' in error_msg:
                print(f"⚠️  STUMPY NumPy 2.0 compatibility issue: {error_msg}")
                print("💡 Solution: Update STUMPY to a NumPy 2.0 compatible version or downgrade NumPy to 1.x")
            else:
                print(f"⚠️  STUMPY failed: {error_msg}")
            print("🔄 Falling back to custom implementation...")
            return self._compute_with_custom(time_series)
    
    def _compute_with_pyscamp(self, time_series):
        """Compute matrix profile using PySCAMP."""
        try:
            print("🚀 Using PySCAMP for matrix profile calculation...")
            
            if self.use_left_mp:
                print("⚠️  PySCAMP doesn't support left matrix profile directly. Using custom implementation...")
                return self._compute_with_custom(time_series)
            
            # Prepare data for PySCAMP
            T = np.array(time_series, dtype=np.float64)
            
            # Handle NaN values by interpolating
            if np.any(np.isnan(T)):
                mask = np.isnan(T)
                T[mask] = np.interp(np.flatnonzero(mask), 
                                  np.flatnonzero(~mask), T[~mask])
            
            # Compute matrix profile with PySCAMP
            mp, mpi = pyscamp.selfjoin(T, self.window_size)
            
            print(f"✅ PySCAMP computation complete: {len(mp)} profile points")
            return mp
            
        except Exception as e:
            print(f"⚠️  PySCAMP failed: {str(e)}")
            print("🔄 Falling back to custom implementation...")
            return self._compute_with_custom(time_series)
    
    def _compute_with_custom(self, time_series):
        """Compute matrix profile using custom implementation."""
        impl_type = "left matrix profile" if self.use_left_mp else "standard matrix profile"
        print(f"🛠️  Using custom {impl_type} implementation...")
        return custom_matrix_profile(time_series, self.window_size, self.normalize, self.use_left_mp)
    
    def calculate_threshold(self, matrix_profile, custom_multiplier=None):
        """
        Calculate anomaly detection threshold.
        
        Args:
            matrix_profile: Computed matrix profile
            custom_multiplier: Optional custom multiplier for sigma methods
            
        Returns:
            float: Threshold value for anomaly detection
        """
        # Filter out infinite values
        valid_mp = matrix_profile[np.isfinite(matrix_profile)]
        
        if len(valid_mp) == 0:
            print("⚠️  No valid matrix profile values found!")
            return np.inf
        
        if self.threshold_method == 'sigma':
            # Use delta*sigma where delta is the custom multiplier (default 3.0)
            multiplier = custom_multiplier if custom_multiplier is not None else 3.0
            threshold = valid_mp.mean() + multiplier * valid_mp.std()
        elif self.threshold_method == 'percentile95':
            threshold = np.percentile(valid_mp, 95)
        elif self.threshold_method == 'percentile99':
            threshold = np.percentile(valid_mp, 99)
        else:
            # Default to 3-sigma
            multiplier = custom_multiplier if custom_multiplier is not None else 3.0
            threshold = valid_mp.mean() + multiplier * valid_mp.std()
        
        print(f"🎯 Threshold calculated ({self.threshold_method}): {threshold:.4f}")
        if custom_multiplier and self.threshold_method == 'sigma':
            print(f"   🔧 δ×σ multiplier used: {custom_multiplier}")
        print(f"   📊 MP stats: mean={valid_mp.mean():.4f}, std={valid_mp.std():.4f}")
        
        return threshold
    
    def detect_anomalies_batch(self, data_series, custom_multiplier=None):
        """
        Detect anomalies in a batch of data.
        
        Args:
            data_series: pandas Series or numpy array of time series data
            custom_multiplier: Optional custom multiplier for sigma-based threshold methods
            
        Returns:
            tuple: (anomaly_flags, anomaly_scores, threshold)
        """
        if isinstance(data_series, pd.Series):
            values = data_series.values
        else:
            values = np.array(data_series)
        
        # Show configuration info
        selected_impl = self._select_implementation()
        mp_type = "left matrix profile" if self.use_left_mp else "standard matrix profile"
        print(f"🔧 Configuration: {selected_impl.upper()} with {mp_type}")
        
        # Compute matrix profile for entire series
        matrix_profile = self.compute_matrix_profile(values)
        threshold = self.calculate_threshold(matrix_profile, custom_multiplier)
        
        # Extend matrix profile to match data length
        anomaly_scores = np.zeros(len(values))
        anomaly_flags = np.zeros(len(values), dtype=bool)
        
        # Fill in available scores
        for i in range(len(matrix_profile)):
            if i + self.window_size < len(values):
                anomaly_scores[i + self.window_size] = matrix_profile[i]
                anomaly_flags[i + self.window_size] = matrix_profile[i] > threshold
        
        return anomaly_flags, anomaly_scores, threshold
    
    def get_implementation_info(self):
        """Get information about the current configuration."""
        selected_impl = self._select_implementation()
        available_libs = []
        if USE_PYSCAMP:
            available_libs.append("PySCAMP")
        if USE_STUMPY:
            available_libs.append("STUMPY")
        available_libs.append("Custom")
        
        # Get compatibility information
        compat_info = check_numpy_stumpy_compatibility()
        
        info = {
            'selected_implementation': selected_impl,
            'available_implementations': available_libs,
            'use_left_mp': self.use_left_mp,
            'window_size': self.window_size,
            'normalize': self.normalize,
            'threshold_method': self.threshold_method,
            'numpy_version': compat_info['numpy_version'],
            'numpy_2_compat': compat_info['compatibility_patches_applied']
        }
        
        # Add STUMPY version if available
        if USE_STUMPY and 'stumpy_version' in compat_info:
            info['stumpy_version'] = compat_info['stumpy_version']
        
        # Add warnings if there are compatibility issues
        if compat_info['recommendations']:
            info['compatibility_warnings'] = compat_info['recommendations']
        
        return info

class DataGenerator:
    """Generate synthetic MSS-like data for demonstration purposes."""
    
    @staticmethod
    def generate_demo_data(start_date='2024-01-01', days=30, frequency='H'):
        """
        Generate synthetic Mobile Spatial Statistics data.
        
        Args:
            start_date: Start date for data generation
            days: Number of days to generate
            frequency: Data frequency ('H' for hourly)
            
        Returns:
            pandas.DataFrame: Generated data with timestamps and values
        """
        # Create time index
        end_date = pd.to_datetime(start_date) + pd.Timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate base patterns
        hours = len(timestamps)
        time_idx = np.arange(hours)
        
        # Daily pattern (24-hour cycle)
        daily_pattern = np.sin(time_idx * 2 * np.pi / 24) * 100
        
        # Weekly pattern (7-day cycle)
        weekly_pattern = np.sin(time_idx * 2 * np.pi / (24 * 7)) * 50
        
        # Monthly trend
        monthly_trend = np.sin(time_idx * 2 * np.pi / (24 * 30)) * 30
        
        # Base level
        base_level = 1000
        
        # Noise
        np.random.seed(42)
        noise = np.random.normal(0, 20, hours)
        
        # Combine patterns
        values = base_level + daily_pattern + weekly_pattern + monthly_trend + noise
        
        # Add some anomalies
        anomaly_indices = []
        num_anomalies = max(1, hours // 200)  # ~1 anomaly per 200 hours
        
        for _ in range(num_anomalies):
            idx = np.random.randint(100, hours - 100)  # Avoid edges
            anomaly_magnitude = np.random.choice([-1, 1]) * np.random.uniform(200, 400)
            values[idx] += anomaly_magnitude
            anomaly_indices.append(idx)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps[:-1],  # Remove last timestamp to match values length
            'value': values,
            'is_true_anomaly': [i in anomaly_indices for i in range(len(values))]
        })
        
        return data

# Event configurations for different types of analysis
EVENT_CONFIGS = {
    'noto_earthquake': {
        'name': 'Noto Peninsula Earthquake (Mw7.5)',
        'datetime': '2024-01-01 16:10:00',
        'meshcode': 564203,
        'description': 'Major earthquake in Ishikawa Prefecture',
        'magnitude': 7.5,
        'location': 'Noto Peninsula, Japan'
    },
    'fukushima_earthquake': {
        'name': 'Fukushima Earthquake (Mw7.4)',
        'datetime': '2022-03-16 23:36:00',
        'meshcode': 574007414,
        'description': 'Strong earthquake off Fukushima coast',
        'magnitude': 7.4,
        'location': 'Off Fukushima, Japan'
    },
    'demo_event': {
        'name': 'Demonstration Event',
        'datetime': '2024-01-15 10:00:00',
        'meshcode': 123456,
        'description': 'Synthetic event for demonstration',
        'magnitude': 6.0,
        'location': 'Demo Location'
    }
}