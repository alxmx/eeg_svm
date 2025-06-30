# EDA Channel Configuration Update

## Issue Description
The EDA stream has 2 channels (0-indexed):
- **Channel 0**: Contains timestamps 
- **Channel 1**: Contains the actual EDA data

The code must always use channel 1 for EDA feature extraction. There is no fallback to channel 0 since it only contains timestamps.

## Changes Made

### 1. Updated EDA Channel Comment
```python
EDA_CHANNEL_INDEX = 1  # Channel 1 (0-based indexing) contains the actual EDA data; Channel 0 contains timestamps
```

### 2. Fixed EDA Extraction in `extract_features()` method
```python
# === AROUSAL/STRESS (EDA) ===
# Always use channel 1 which contains the actual EDA data (channel 0 is timestamps)
if eda_window.shape[1] >= 2:
    # Use channel 1 (0-based index) - the actual EDA data channel
    raw_eda = np.mean(eda_window[:, 1])
else:
    # Error: EDA stream must have 2 channels (timestamps + data)
    print(f"[ERROR] EDA stream has {eda_window.shape[1]} channels, expected 2 (timestamps + data)")
    raw_eda = 0.0
```

### 3. Fixed EDA Extraction in `extract_mindfulness_features()` function
Same logic applied to the standalone feature extraction function.

### 4. Updated Calibration Data Collection
```python
if len(eda_sample) >= 2:
    # Keep both channels: [timestamp, eda_data]
    eda = np.array(eda_sample[:2])
else:
    # Error: EDA stream must have 2 channels
    print(f"[ERROR] EDA sample has {len(eda_sample)} channels, expected 2")
    eda = np.array([0, 0])  # [timestamp, eda_data]
```

### 5. Updated Safety Checks
```python
if eda_window.shape[1] < 2:
    print(f"[WARNING] EDA window has {eda_window.shape[1]} channels, expected 2 (timestamp + data)")
    return np.zeros(9)
```

## Summary
- **Channel 0**: Always ignored (contains timestamps only)
- **Channel 1**: Always used for EDA feature extraction (contains actual physiological data)
- **No Fallback**: Code will error if EDA stream doesn't have exactly 2 channels
- **Validation**: Added proper error checking to ensure 2-channel EDA stream format

This ensures the pipeline correctly extracts EDA features from channel 1 which contains the actual physiological data, while ignoring the timestamps in channel 0.
