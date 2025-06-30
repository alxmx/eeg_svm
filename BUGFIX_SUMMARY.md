# Bug Fixes for v4_realtime_mi_lsl_dual_calibration.py

## Issue 1: EDA Channel Indexing Error
**Problem**: `IndexError: index 1 is out of bounds for axis 1 with size 1`
- The code assumed EDA data always has 2 channels (using `EDA_CHANNEL_INDEX = 1`)
- When EDA stream only has 1 channel, accessing index 1 causes an error

**Fix Applied**:
1. Changed `EDA_CHANNEL_INDEX = 1` to `EDA_CHANNEL_INDEX = 0` as default
2. Added dynamic channel detection in both `extract_features()` and `extract_mindfulness_features()`:
   ```python
   # Handle variable EDA channels (1 or 2 channels)
   if eda_window.shape[1] == 1:
       # Single channel EDA
       raw_eda = np.mean(eda_window[:, 0])
   else:
       # Multi-channel EDA, use the second channel (index 1) if available, otherwise first
       eda_channel = min(1, eda_window.shape[1] - 1)
       raw_eda = np.mean(eda_window[:, eda_channel])
   ```
3. Added safety checks to prevent similar issues:
   ```python
   # Safety checks
   if len(eeg_window) == 0 or len(eda_window) == 0:
       return np.zeros(9)  # Return zeros if no data
   
   if eda_window.shape[1] == 0:
       print(f"[WARNING] EDA window has no channels")
       return np.zeros(9)
   ```

## Issue 2: Calibration Duration Problem
**Problem**: Calibration phases were based on sample count (30 samples) instead of actual time (30 seconds)
- This could result in calibration lasting only milliseconds instead of 30 seconds
- Duration was dependent on data availability rather than real time

**Fix Applied**:
1. Replaced sample-based collection with time-based collection in `collect_calibration_data()`:
   ```python
   # Time-based collection for accurate duration
   while elapsed_time < duration_sec:
       current_time = time.time()
       elapsed_time = current_time - start_time
       
       # Collect samples...
       
       # Small delay to prevent excessive CPU usage
       time.sleep(0.001)
   ```
2. Added real-time progress monitoring:
   ```python
   # Progress indicator every 1.5 seconds
   if current_time - last_progress_time >= 1.5:
       progress_pct = min(100, int((elapsed_time / duration_sec) * 100))
       print(f"█", end="", flush=True)
       last_progress_time = current_time
   ```
3. Added sample rate reporting for verification:
   ```python
   print(f"[CALIBRATION] Sample rate: {sample_count/actual_duration:.1f} Hz")
   ```

## Additional Improvements
1. **EDA Data Handling**: Updated EDA sample collection to handle variable channel counts
2. **Error Prevention**: Added comprehensive safety checks in feature extraction functions
3. **Better Fallback**: Changed EDA simulation to use zeros instead of random values for consistency
4. **Missing Imports**: Added `warnings`, `threading`, and `msvcrt` imports

## Testing Status
- ✅ File compiles without syntax errors
- ✅ EDA channel indexing is now dynamic and safe
- ✅ Calibration timing is now based on actual seconds (30 seconds)
- ✅ Added comprehensive error handling and safety checks

## Next Steps
1. Test the calibration duration to confirm it runs for exactly 30 seconds
2. Test with both single-channel and dual-channel EDA streams
3. Verify that real-time processing no longer crashes on EDA channel indexing
4. Confirm that calibration provides adequate dynamic range for MI calculation
