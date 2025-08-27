# EEG Signal Processing Workflow

This snippet contains the core signal processing workflow from the EEG emotion classification pipeline. It provides a clean, focused implementation of the essential signal processing steps without the visualization and analysis components.

## Overview

The workflow processes EEG data through the following key steps:

1. **Data Loading**: Load EEG data from CSV files
2. **Bandpass Filtering**: Remove frequencies outside 1-40 Hz range
3. **Notch Filtering**: Remove 50 Hz powerline noise
4. **Windowing/Epoching**: Create overlapping time windows
5. **Feature Extraction**: Extract statistical features from frequency bands
6. **Normalization**: Apply z-score normalization

## Features

- **200 features per epoch**: 8 channels × 5 frequency bands × 5 statistical measures
- **Overlapping windows**: 2-second windows with 1-second overlap
- **Robust filtering**: Butterworth bandpass + notch filtering
- **Statistical features**: Mean, variance, std, kurtosis, skewness for each frequency band
- **Flexible normalization**: Can use training statistics or compute from current data

## Quick Start

```python
import signal_processing_workflow as spw

# Process a single EEG file
results = spw.process_eeg_signal("path/to/eeg_file.csv")

# Access the processed data
filtered_data = results['filtered_data']
epochs = results['epochs']
features = results['features_normalized']
```

## File Structure

### Core Functions

- `load_eeg_csv()`: Load EEG data from CSV files
- `bandpass_filter()`: Apply Butterworth bandpass filter (1-40 Hz)
- `notch_filter()`: Remove powerline noise (50 Hz)
- `windowed_epochs()`: Create overlapping time windows
- `extract_stat_features()`: Extract 200 statistical features per epoch
- `zscore_features()`: Apply z-score normalization

### Main Workflow

- `process_eeg_signal()`: Complete end-to-end processing pipeline

## Parameters

```python
# Configurable parameters
FS = 250                    # Sampling frequency (Hz)
BANDPASS = (1, 40)         # Bandpass filter range (Hz)
NOTCH_FREQ = 50            # Notch filter frequency (Hz)
EPOCH_SEC = 2              # Window duration (seconds)
EPOCH_OVERLAP = 1          # Window overlap (seconds)

# EEG channels (8 channels)
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# Frequency bands for feature extraction
BANDS = {
    "delta": (1, 4),       # Delta waves (1-4 Hz)
    "theta": (4, 8),       # Theta waves (4-8 Hz)
    "alpha": (8, 13),      # Alpha waves (8-13 Hz)
    "beta": (13, 30),      # Beta waves (13-30 Hz)
    "gamma": (30, 40)      # Gamma waves (30-40 Hz)
}
```

## Data Format

The workflow expects CSV files with:
- Column 0: Timestamps (not used)
- Columns 1-8: EEG data from 8 channels
- Data should be numeric (conversion errors are handled automatically)

## Output

The `process_eeg_signal()` function returns a dictionary with:

```python
{
    'raw_data': numpy.ndarray,           # Original EEG data
    'filtered_data': numpy.ndarray,      # Filtered EEG data
    'epochs': numpy.ndarray,             # Windowed epochs
    'features': numpy.ndarray,           # Raw features (epochs × 200)
    'features_normalized': numpy.ndarray, # Normalized features
    'normalization_stats': tuple,        # (mean, std) for normalization
    'n_epochs': int,                     # Number of epochs
    'n_features': int                    # Number of features (200)
}
```

## Dependencies

```bash
pip install numpy pandas scipy
```

## Example Usage

```python
# Basic usage
results = spw.process_eeg_signal("eeg_data.csv")
print(f"Processed {results['n_epochs']} epochs with {results['n_features']} features each")

# Using training statistics for normalization (typical for classification)
mu, sigma = training_stats  # From training data
results = spw.process_eeg_signal("test_file.csv", training_stats=(mu, sigma))

# Access specific components
epochs = results['epochs']          # Shape: (n_epochs, 500, 8) for 2s epochs at 250 Hz
features = results['features_normalized']  # Shape: (n_epochs, 200)
```

## Technical Details

### Feature Extraction

For each epoch (2-second window):
- Each of 8 channels is bandpass filtered into 5 frequency bands
- 5 statistical measures are computed from each band-filtered signal:
  - Mean amplitude
  - Variance  
  - Standard deviation
  - Kurtosis (tail heaviness)
  - Skewness (asymmetry)
- Total: 8 channels × 5 bands × 5 stats = **200 features per epoch**

### Filtering Pipeline

1. **Bandpass Filter**: 4th-order Butterworth filter (1-40 Hz)
   - Removes DC drift and high-frequency noise
   - Preserves physiologically relevant EEG frequencies

2. **Notch Filter**: IIR notch filter at 50 Hz (Q=30)
   - Removes powerline interference
   - Narrow rejection band to preserve nearby frequencies

### Windowing

- **Window size**: 2 seconds (500 samples at 250 Hz)
- **Overlap**: 1 second (50% overlap)
- **Step size**: 1 second (250 samples)
- Provides temporal resolution while maintaining sufficient data per epoch

## Integration

This workflow integrates seamlessly with machine learning pipelines:

```python
# Train a classifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Process training data
train_results = spw.process_eeg_signal("train_data.csv")
X_train = train_results['features_normalized']
# ... get labels y_train ...

# Train classifier
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
clf = SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_scaled, y_train)

# Process and classify new data
test_results = spw.process_eeg_signal("test_data.csv", 
                                     training_stats=train_results['normalization_stats'])
X_test = scaler.transform(test_results['features_normalized'])
predictions = clf.predict(X_test)
```

## Testing

The workflow has been tested with real EEG data and successfully processes:
- 52,661 samples → 209 epochs → 200 features per epoch
- Handles data conversion errors gracefully
- Provides detailed progress output

## License

This code is part of the EEG-SVM emotion classification pipeline.