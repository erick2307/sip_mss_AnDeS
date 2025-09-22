# NumPy 2.0 Compatibility Fix for STUMPY

## Problem Description

When using STUMPY with NumPy 2.0+, you may encounter this error:
```
STUMPY failed: `np.NINF` was removed in the NumPy 2.0 release. Use `-np.inf` instead.
```

This happens because STUMPY was developed for earlier NumPy versions and uses deprecated constants that were removed in NumPy 2.0.

## Solution Implemented

The AnDeS system now includes automatic compatibility fixes:

### 1. Automatic Compatibility Patches
- When importing STUMPY, the system checks for missing NumPy constants
- Automatically adds the missing constants (`np.NINF`, `np.PINF`, `np.NAN`) if needed
- Applies patches transparently without affecting functionality

### 2. Enhanced Error Handling
- Detects NumPy 2.0 compatibility issues in STUMPY calls
- Provides informative error messages with solutions
- Automatically falls back to custom implementation when STUMPY fails

### 3. System Status Monitoring
- Added compatibility status section in the System Status tab
- Shows NumPy version and compatibility patch status
- Displays warnings if compatibility issues are detected

## Code Changes Made

### core.py
1. **Enhanced STUMPY import** with compatibility patches:
   ```python
   # Handle NumPy 2.0 compatibility issues for STUMPY
   if not hasattr(np, 'NINF'):
       np.NINF = -np.inf
   # Similar patches for PINF and NAN
   ```

2. **Improved error handling** in `_compute_with_stumpy()`:
   - Specific detection of `NINF` errors
   - Informative error messages
   - Automatic fallback to custom implementation

3. **Added compatibility checking** function:
   - `check_numpy_stumpy_compatibility()` provides detailed status
   - Enhanced `get_implementation_info()` with compatibility information

### app.py
1. **System Status enhancements**:
   - New "Library Compatibility Status" section
   - Real-time compatibility monitoring
   - Visual indicators for NumPy 2.0 compatibility

## User Actions

### For End Users
1. **No action required** - the system handles compatibility automatically
2. **Check System Status tab** to verify compatibility status
3. **If issues persist**:
   - Update STUMPY: `pip install --upgrade stumpy`
   - Or downgrade NumPy: `pip install "numpy<2.0"`

### For Developers
1. **Test the fix** using `test_numpy_compatibility.py`
2. **Monitor logs** for compatibility warnings
3. **Consider updating dependencies** for optimal performance

## Testing

Run the compatibility test:
```bash
python test_numpy_compatibility.py
```

This will verify that:
- NumPy version is detected correctly
- Compatibility patches are applied if needed
- STUMPY functions work with both standard and left matrix profiles
- Fallback mechanisms are working

## Long-term Solution

The proper long-term solution is to update STUMPY to a version compatible with NumPy 2.0. This fix is a temporary workaround that maintains functionality while the ecosystem catches up.

## Verification

After implementing these changes, the error should no longer occur, and the system will:
1. ✅ Work with both NumPy 1.x and 2.0+
2. ✅ Provide clear error messages if issues occur
3. ✅ Automatically fall back to working implementations
4. ✅ Show compatibility status in the UI