"""
EEG Mindfulness Index Pipeline

This project provides a comprehensive pipeline for analyzing EEG (and optional EDA) data to quantify mindfulness using the Mindfulness Index (MI). It supports both discrete state classification and continuous regression, and includes robust preprocessing, feature extraction, visualization, and reporting.

Main Functionalities:
- Data loading and preprocessing for EEG and EDA (CSV/OpenSignals)
- Signal filtering (bandpass)
- Feature extraction: Frontal Theta, Posterior Alpha, Frontal Alpha Asymmetry, Frontal Beta, EDA
- Mindfulness Index (MI) calculation (continuous 0-1 value)
- Behavioral state classification (Focused, Neutral, Unfocused) using MI thresholds
- SVM-based classification (discrete states) and regression (continuous MI)
- Visualization: MI time series, feature contributions, state distributions, and model performance
- Batch processing and comprehensive reporting (CSV, JSON, PDF)

Electrode Key:
    Frontal: Fz (ch1)
    Left Central: C3 (ch2)
    Central Midline: Cz (ch3)
    Right Central: C4 (ch4)
    Parietal Midline: Pz (ch5)
    Left Parietal-Occipital: PO7 (ch6)
    Occipital: Oz (ch7)
    Right Parietal-Occipital: PO8 (ch8)
    Accelerometers (ch9–11): Not relevant for EEG analysis.

Mindfulness Index Formula:
    MI_raw = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)
    MI = 1 / (1 + exp(-MI_raw + 1))  # Normalized to 0-1 range
    
    Default weights:
    w1 = 0.25 (Frontal Theta)
    w2 = 0.25 (Posterior Alpha)
    w3 = 0.20 (Frontal Alpha Asymmetry)
    w4 = 0.15 (Frontal Beta)
    w5 = 0.10 (EDA)
    
    The formula includes a normalization step to ensure MI values stay within 0-1 range.

Classification Methods Supported:
- Threshold-based: MI is binned into Focused, Neutral, Unfocused using fixed thresholds.
- SVM Classification: Predicts discrete states from features using synthetic MI-based labels.
- SVM Regression: Predicts continuous MI value from features for nuanced feedback.

Outputs:
- Per-window MI and state
- Model predictions and metrics
- Plots and comprehensive PDF/text/CSV reports
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime
from fpdf import FPDF

# --- Parameters ---
FS = 250  # Sampling frequency (Hz)
BANDPASS = (4, 30)  # Hz, for general filtering

# =============================================================================
# EDA DATA FOLDER CONFIGURATION - EASY TO MODIFY
# =============================================================================
# Define the folder where EDA data files are stored
# Change this path to modify where EDA files are searched
EDA_DATA_FOLDER = "data/eda_data"  # <-- CHANGE THIS PATH AS NEEDED
THETA_BAND = (4, 7.99)  # Hz
ALPHA_BAND = (8, 12.99)  # Hz
BETA_BAND = (13, 30)  # Hz
WINDOW_SEC = 3  # Window size for features (seconds)
OVERLAP = 0.5  # 50% overlap between windows
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CH_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# MI calculation weights
MI_WEIGHTS = {
    'theta_fz': 0.25,
    'alpha_po': 0.25,
    'faa': 0.20,
    'beta_frontal': -0.15,  # Beta activity is correlated with mental concentration and active thinking
    'eda_norm': -0.25  # EDA is negatively correlated with mindfulness
}

# Behavioral state thresholds
THRESHOLDS = {
    'focused': 0.5,  # MI >= 0.5 (was 0.6)
    'neutral': 0.37,  # 0.37 <= MI < 0.5 (was 0.4)
    # 'unfocused': MI < 0.37
}

# --- Helper Functions ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def compute_band_power(data, fs, band):
    """Calculate power in a frequency band using Welch's method"""
    fmin, fmax = band
    f, Pxx = welch(data, fs=fs, nperseg=min(len(data), fs*2))
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    return np.mean(Pxx[idx_band]) if np.any(idx_band) else 0

def load_eeg_csv(filepath):
    """Load EEG data from CSV file"""
    print(f"Loading EEG data from: {filepath}")
    df = pd.read_csv(filepath, dtype=str)  # Read as string to avoid dtype warning
    # Convert all EEG channel columns to float, coerce errors to NaN
    eeg_data = df.iloc[:, 1:9].apply(pd.to_numeric, errors='coerce').values
    # Remove rows with any NaN (from conversion errors)
    eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
    return eeg_data

def load_eda_csv(filepath):
    """Load EDA data from CSV file if available"""
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            print(f"EDA file does not exist: {filepath}")
            return None
            
        # Check if it's an OpenSignals txt file or a regular CSV
        if filepath.endswith('.txt'):
            print(f"Detected OpenSignals format: {filepath}")
            # Extract the header to verify it contains EDA data
            eda_channel_idx = None
            with open(filepath, 'r') as f:
                header_lines = []
                for line in f:
                    if line.startswith('#') or line.startswith('//'):
                        header_lines.append(line)
                        # Try to identify which column contains EDA data from the header
                        if '"sensor":' in line and '"EDA"' in line:
                            print("Found EDA sensor in header")
                            try:
                                header_json = json.loads(line.strip('#').strip())
                                for device_id, device_info in header_json.items():
                                    if 'sensor' in device_info and 'EDA' in device_info['sensor']:
                                        eda_idx = device_info['sensor'].index('EDA')
                                        # Adding 2 because first columns are typically nSeq and DI
                                        eda_channel_idx = eda_idx + 2
                                        print(f"EDA data found in column index {eda_channel_idx}")
                                        break
                            except Exception as e:
                                print(f"Could not parse header JSON: {e}")
                
            # Skip header lines that start with # or //
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            data_lines = [line for line in lines if not (line.startswith('#') or line.startswith('/'))]
            data_text = '\n'.join(data_lines)
            
            # Parse the data using pandas
            import io
            eda_df = pd.read_csv(io.StringIO(data_text), delimiter='\t', header=None)
            
            # If we identified the EDA column from header, use it, otherwise default to column 2
            if eda_channel_idx is not None and eda_channel_idx < eda_df.shape[1]:
                eda_data = eda_df.iloc[:, eda_channel_idx].apply(pd.to_numeric, errors='coerce').values
            else:
                # Default: EDA value is typically in the third column (index 2)
                eda_data = eda_df.iloc[:, 2].apply(pd.to_numeric, errors='coerce').values
                
            print(f"Successfully loaded {len(eda_data)} EDA samples from OpenSignals file")
            
            # Basic validation of EDA data
            if len(eda_data) > 0:
                print(f"EDA data range: {np.min(eda_data)} to {np.max(eda_data)}")
                if np.max(eda_data) <= 0:
                    print("WARNING: All EDA values are non-positive, data may be invalid")
            
            return eda_data
        else:
            # Regular CSV format
            eda_df = pd.read_csv(filepath, dtype=str)
            eda_data = eda_df.iloc[:, 1].apply(pd.to_numeric, errors='coerce').values
            print(f"Successfully loaded {len(eda_data)} EDA samples from CSV file")
            return eda_data
    except Exception as e:
        print(f"Error loading EDA data from {filepath}: {e}")
        return None

def create_windows(data, fs, window_sec, overlap):
    """Create overlapping windows from continuous data"""
    n_samples = data.shape[0]
    window_samples = int(fs * window_sec)
    step_samples = int(window_samples * (1 - overlap))
    
    windows = []
    timestamps = []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(data[start:end])
        timestamps.append(start / fs)  # Convert to seconds
    
    return np.array(windows), np.array(timestamps)

def normalize_eda(eda_data, method='zscore'):
    """Normalize EDA data using z-score or min-max scaling"""
    if eda_data is None:
        return None
    
    # Filter out any NaN or invalid values
    valid_data = eda_data[~np.isnan(eda_data)]
    if len(valid_data) == 0:
        print("WARNING: No valid EDA data points found")
        return None
        
    print(f"Normalizing {len(valid_data)} EDA data points using {method} method")
    
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        if std_val == 0:
            print("WARNING: EDA data has zero standard deviation, using raw values")
            return valid_data
        normalized = (valid_data - mean_val) / (std_val + 1e-8)
        print(f"EDA Z-score normalization: mean={mean_val:.2f}, std={std_val:.2f}")
        
    elif method == 'minmax':
        # Min-max scaling: (x - min) / (max - min)
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val == min_val:
            print("WARNING: EDA data has identical min and max values, using zeros")
            return np.zeros_like(valid_data)
        normalized = (valid_data - min_val) / (max_val - min_val + 1e-8)
        print(f"EDA Min-max normalization: min={min_val:.2f}, max={max_val:.2f}")
        
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")
        
    # Verify the normalization worked as expected
    print(f"Normalized EDA range: {np.min(normalized):.4f} to {np.max(normalized):.4f}")
    return normalized

def preprocess_eeg(eeg_data, fs):
    """Apply preprocessing to EEG data"""
    # Apply bandpass filter
    eeg_filtered = bandpass_filter(eeg_data, *BANDPASS, fs)
    
    # Artifact rejection (remove windows with amplitude > ±100 µV)
    # This is simplified - in practice, you might want to keep track of rejected segments
    amplitude_threshold = 100  # µV
    valid_indices = np.all(np.abs(eeg_filtered) < amplitude_threshold, axis=1)
    eeg_cleaned = eeg_filtered[valid_indices]
    
    return eeg_cleaned

def extract_features(window, fs):
    """Extract relevant features from an EEG window"""
    features = {}
    
    # 1. Frontal Theta (Fz)
    theta_fz = compute_band_power(window[:, CH_IDX["Fz"]], fs, THETA_BAND)
    features['theta_fz'] = theta_fz
    
    # 2. Posterior Alpha (PO7/PO8)
    alpha_po7 = compute_band_power(window[:, CH_IDX["PO7"]], fs, ALPHA_BAND)
    alpha_po8 = compute_band_power(window[:, CH_IDX["PO8"]], fs, ALPHA_BAND)
    alpha_po = (alpha_po7 + alpha_po8) / 2
    features['alpha_po'] = alpha_po
    
    # 3. Frontal Alpha Asymmetry (FAA)
    alpha_c3 = compute_band_power(window[:, CH_IDX["C3"]], fs, ALPHA_BAND)
    alpha_c4 = compute_band_power(window[:, CH_IDX["C4"]], fs, ALPHA_BAND)
    # Use log transformation (avoid log(0))
    alpha_c3_log = np.log(alpha_c3 + 1e-8)
    alpha_c4_log = np.log(alpha_c4 + 1e-8)
    faa = alpha_c4_log - alpha_c3_log
    features['faa'] = faa
    
    # 4. Frontal Beta (Fz/C3/C4)
    beta_fz = compute_band_power(window[:, CH_IDX["Fz"]], fs, BETA_BAND)
    beta_c3 = compute_band_power(window[:, CH_IDX["C3"]], fs, BETA_BAND)
    beta_c4 = compute_band_power(window[:, CH_IDX["C4"]], fs, BETA_BAND)
    beta_frontal = (beta_fz + beta_c3 + beta_c4) / 3
    features['beta_frontal'] = beta_frontal
    
    return features

def calculate_mi(features, eda_value=None, weights=None):
    """Calculate Mindfulness Index (MI) from features
    
    Args:
        features: Dictionary of features
        eda_value: Optional EDA value
        weights: Optional custom weights (default: MI_WEIGHTS)
    
    Returns:
        Normalized MI score between 0 and 1
    """
    # Handle the EDA value - set to 0 if not available
    if eda_value is None:
        print("No EDA data available for this window, using 0")
        eda_norm = 0
    else:
        # Ensure the EDA value is valid
        if np.isnan(eda_value) or np.isinf(eda_value):
            print(f"Invalid EDA value ({eda_value}), using 0")
            eda_norm = 0
        else:
            eda_norm = eda_value
            
    features['eda_norm'] = eda_norm
    
    # Use provided weights or default weights
    use_weights = weights if weights is not None else MI_WEIGHTS
    
    # Calculate raw MI using weights
    mi_raw = 0
    for feature, weight in use_weights.items():
        if feature in features:
            feature_value = features[feature]
            # Ensure the feature value is valid
            if np.isnan(feature_value) or np.isinf(feature_value):
                print(f"WARNING: Invalid {feature} value: {feature_value}, using 0")
                feature_value = 0
            mi_raw += weight * feature_value
    
    # Normalize MI to 0-1 range
    # Empirically observed values can be very high, so we use a sigmoid-like normalization
    # This ensures MI stays within the 0-1 range while preserving rank ordering
    mi_normalized = 1 / (1 + np.exp(-mi_raw + 1))
    
    # Verify the MI value is within expected range
    if not (0 <= mi_normalized <= 1):
        print(f"WARNING: MI normalization failed, got value: {mi_normalized}, clamping to [0,1]")
        mi_normalized = max(0, min(1, mi_normalized))
    
    return mi_normalized

def normalize_mi_value(mi_raw):
    """Normalize a raw MI value to 0-1 range using sigmoid-like normalization.
    This function can be used to normalize historical MI values for backward compatibility.
    """
    return 1 / (1 + np.exp(-mi_raw + 1))

def classify_behavioral_state(mi, thresholds=None):
    """Classify behavioral state based on MI value
    
    Args:
        mi: Mindfulness Index value
        thresholds: Optional custom thresholds (default: THRESHOLDS)
        
    Returns:
        Behavioral state as string ("Focused", "Neutral", or "Unfocused")
    """
    use_thresholds = thresholds if thresholds is not None else THRESHOLDS
    
    if mi >= use_thresholds['focused']:
        return "Focused"
    elif mi >= use_thresholds['neutral']:
        return "Neutral"
    else:
        return "Unfocused"

def process_eeg_file(eeg_filepath, eda_filepath=None):
    """Process a single EEG file and return results"""
    # Load data
    print(f"Loading EEG file: {eeg_filepath}")
    eeg_data = load_eeg_csv(eeg_filepath)
    
    # Check if EDA data is available
    if eda_filepath:
        print(f"Loading EDA file: {eda_filepath}")
        eda_data = load_eda_csv(eda_filepath)
        if eda_data is None or len(eda_data) == 0:
            print("WARNING: Failed to load EDA data or file is empty")
    else:
        print("No EDA filepath provided. Calculating MI without EDA data.")
        eda_data = None
    
    # Calculate duration of the EEG data
    eeg_duration_sec = len(eeg_data) / FS if len(eeg_data) > 0 else 0
    print(f"EEG data duration: {eeg_duration_sec:.2f} seconds ({eeg_duration_sec/60:.2f} minutes)")
    
    if eeg_data.shape[0] < FS * WINDOW_SEC:
        print(f"WARNING: EEG file duration ({eeg_duration_sec:.2f}s) is shorter than the minimum required window size ({WINDOW_SEC}s)")
        print(f"Analysis may not be reliable. Consider using a longer recording.")
    
    # Preprocess EEG
    eeg_cleaned = preprocess_eeg(eeg_data, FS)
    
    # Create windows
    windows, timestamps = create_windows(eeg_cleaned, FS, WINDOW_SEC, OVERLAP)
    
    # Check if we have enough data to analyze
    if len(windows) == 0:
        print(f"ERROR: No complete windows could be created from the data. File may be too short.")
        return {'results': [], 'metadata': {'duration_sec': eeg_duration_sec, 'windows': 0, 'valid': False}}
    
    # Calculate EDA duration and sampling rate if available
    eda_fs = FS  # Default: assume same rate as EEG
    eda_duration_sec = 0
    
    if eda_data is not None and len(eda_data) > 0:
        # For OpenSignals files, get the sampling rate from the filename if available
        if eda_filepath and 'opensignals_' in eda_filepath:
            # OpenSignals often includes sampling rate in the filename e.g., opensignals_lsl_500hz_gain1_...
            filename = os.path.basename(eda_filepath)
            if '_hz_' in filename.lower():
                try:
                    # Extract the part between '_' and 'hz_'
                    rate_part = filename.lower().split('_hz_')[0].split('_')[-1]
                    if rate_part.isdigit():
                        eda_fs = int(rate_part)
                        print(f"Detected EDA sampling rate from filename: {eda_fs} Hz")
                except:
                    pass
                    
        # Calculate duration based on the sampling rate
        eda_duration_sec = len(eda_data) / eda_fs
        print(f"EDA data duration: {eda_duration_sec:.2f} seconds ({eda_duration_sec/60:.2f} minutes)")
        print(f"EDA sampling rate: {eda_fs} Hz")
        
        # Check for significant duration mismatch
        if abs(eda_duration_sec - eeg_duration_sec) > 10:  # More than 10 seconds difference
            print(f"WARNING: Significant duration mismatch between EEG ({eeg_duration_sec:.2f}s) and EDA ({eda_duration_sec:.2f}s)")
            print("This may indicate the files are not from the same recording session.")
            print("EDA data will still be used, but results may be less accurate.")
    
    # Normalize EDA (if available)
    eda_normalized = normalize_eda(eda_data) if eda_data is not None else None
        
    # Process each window
    results = []
    
    # Keep track of how many times we use EDA values
    eda_usage_count = 0
    
    for i, (window, timestamp) in enumerate(zip(windows, timestamps)):
        # Extract features
        features = extract_features(window, FS)
        
        # Get EDA value for this window (if available)
        if eda_normalized is not None:
            # Find corresponding EDA value for this timestamp
            # Calculate the proper index based on timestamp and sampling rate
            eda_idx = int(timestamp * eda_fs)
            
            # Make sure we don't exceed EDA array bounds
            if eda_idx < len(eda_normalized):
                eda_value = eda_normalized[eda_idx]
                eda_usage_count += 1
            else:
                # If timestamp is beyond EDA data length, use the last available value
                if len(eda_normalized) > 0:
                    eda_value = eda_normalized[-1]  # Use last value if we've run out of EDA data
                    eda_usage_count += 1
                    if i == 0 or i % 10 == 0:  # Only print warning occasionally
                        print(f"Note: EDA data shorter than EEG at timestamp {timestamp:.1f}s, using last available value")
                else:
                    eda_value = None
                    print("Warning: No valid EDA values available")
        else:
            eda_value = None
        
        # Calculate MI
        mi = calculate_mi(features, eda_value)
        
        # Classify state
        behavioral_state = classify_behavioral_state(mi)
        
        # Create result entry
        result = {
            'timestamp': timestamp,
            'features': features,
            'eda_value': float(eda_value) if eda_value is not None else None,
            'mi_score': round(float(mi), 2),
            'behavioral_state': behavioral_state
        }
        results.append(result)
    
    # Calculate the actual analyzed duration based on the last timestamp plus window size
    analyzed_duration = timestamps[-1] + WINDOW_SEC if timestamps.size > 0 else 0
    
    # Report EDA usage statistics
    if eda_normalized is not None:
        print(f"EDA data used in {eda_usage_count} out of {len(windows)} windows ({eda_usage_count/len(windows)*100:.1f}%)")
    
    # Create metadata dictionary with duration information
    metadata = {
        'file_path': eeg_filepath,
        'eda_path': eda_filepath,
        'total_duration_sec': eeg_duration_sec,
        'analyzed_duration_sec': analyzed_duration,
        'windows_count': len(windows),
        'valid': True,
        'sampling_rate': FS,
        'window_size_sec': WINDOW_SEC,
        'overlap_pct': OVERLAP * 100,
        'has_eda': eda_data is not None,
        'eda_sampling_rate': eda_fs if eda_data is not None else None,
        'eda_duration_sec': eda_duration_sec,
        'eda_usage_count': eda_usage_count
    }
    
    return {'results': results, 'metadata': metadata}

def save_results_to_json(results, output_filepath):
    """Save processing results to JSON file"""
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_filepath}")

def plot_mi_timeseries(results, output_filepath, file_name=None):
    """Plot MI values over time"""
    timestamps = [r['timestamp'] for r in results]
    mi_values = [r['mi_score'] for r in results]
    states = [r['behavioral_state'] for r in results]
    
    # Create color map for states
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    
    colors = [state_colors[state] for state in states]
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot MI values
    scatter = plt.scatter(timestamps, mi_values, c=colors, alpha=0.7)
    plt.plot(timestamps, mi_values, 'k-', alpha=0.3)
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7, label='Focused Threshold')
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7, label='Neutral Threshold')
    
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI)')
    
    # Add file name to the title if provided
    if file_name:
        plt.title(f'Mindfulness Index Over Time - {file_name}')
    else:
        plt.title('Mindfulness Index Over Time')
    
    # Create legend for states
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=state, markersize=10)
        for state, color in state_colors.items()
    ]
    legend_elements.append(Line2D([0], [0], linestyle='--', color='g', label='Focused Threshold'))
    legend_elements.append(Line2D([0], [0], linestyle='--', color='b', label='Neutral Threshold'))
    
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    
    # Add data source information
    plt.figtext(0.02, 0.02, f"Source: {os.path.basename(output_filepath).split('_')[0]}", 
                fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"MI plot saved to {output_filepath}")

def plot_feature_contributions(results, output_filepath, file_name=None):
    """Plot contributions of each feature to the MI score"""
    # Extract data
    timestamps = [r['timestamp'] for r in results]
    feature_names = list(MI_WEIGHTS.keys())
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate weighted feature values
    weighted_features = []
    for result in results:
        weighted_vals = {}
        for feature in feature_names:
            if feature == 'eda_norm' and result['eda_value'] is not None:
                val = MI_WEIGHTS[feature] * result['eda_value']
            elif feature != 'eda_norm':
                val = MI_WEIGHTS[feature] * result['features'].get(feature, 0)
            else:
                val = 0
            weighted_vals[feature] = val
        weighted_features.append(weighted_vals)
    
    # Create subplots
    plt.subplot(2, 1, 1)
    
    # Plot MI values
    mi_values = [r['mi_score'] for r in results]
    plt.plot(timestamps, mi_values, 'k-', linewidth=2, label='MI Score')
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7, label='Focused Threshold')
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7, label='Neutral Threshold')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI)')
    
    # Add file name to the title if provided
    if file_name:
        plt.title(f'Mindfulness Index Over Time - {file_name}')
    else:
        plt.title('Mindfulness Index Over Time')
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature contributions
    plt.subplot(2, 1, 2)
    
    # Get weighted contribution of each feature
    for feature in feature_names:
        values = [wf[feature] for wf in weighted_features]
        plt.plot(timestamps, values, label=f"{feature} (w={MI_WEIGHTS[feature]})")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Weighted Contribution')
    
    # Add file name to the title if provided
    if file_name:
        plt.title(f'Feature Contributions to MI - {file_name}')
    else:
        plt.title('Feature Contributions to MI')
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add data source information
    plt.figtext(0.02, 0.02, f"Source: {os.path.basename(output_filepath).split('_')[0]}", 
                fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"Feature contributions plot saved to {output_filepath}")
    
    # Return the weighted features for use in other functions
    return weighted_features

def plot_behavioral_state_summary(results, output_filepath, file_name=None):
    """Plot summary of behavioral states"""
    states = [r['behavioral_state'] for r in results]
    unique_states = ['Focused', 'Neutral', 'Unfocused']
    state_counts = {state: states.count(state) for state in unique_states}
    
    # Calculate percentages
    total = len(states)
    state_pcts = {state: (count / total) * 100 for state, count in state_counts.items()}
    
    # Create colors
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    colors = [state_colors[state] for state in unique_states]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.subplot(1, 2, 1)
    plt.pie([state_counts[state] for state in unique_states], 
            labels=unique_states, 
            autopct='%1.1f%%', 
            colors=colors,
            explode=[0.05] * len(unique_states))
    
    # Add file name to the title if provided
    if file_name:
        plt.title(f'Distribution of Behavioral States - {file_name}')
    else:
        plt.title('Distribution of Behavioral States')
    
    # Create bar chart
    plt.subplot(1, 2, 2)
    plt.bar(unique_states, [state_pcts[state] for state in unique_states], color=colors)
    plt.xlabel('Behavioral State')
    plt.ylabel('Percentage (%)')
    
    # Add file name to the title if provided
    if file_name:
        plt.title(f'Percentage of Time in Each State - {file_name}')
    else:
        plt.title('Percentage of Time in Each State')
    
    # Add exact percentages on top of bars
    for i, state in enumerate(unique_states):
        plt.text(i, state_pcts[state] + 1, f"{state_pcts[state]:.1f}%", 
                ha='center', va='bottom')
    
    # Add data source information
    plt.figtext(0.02, 0.02, f"Source: {os.path.basename(output_filepath).split('_')[0]}", 
                fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"Behavioral state summary plot saved to {output_filepath}")

def generate_report(results_data, file_label, output_dir='results'):
    """Generate comprehensive report with visualization and CSV output"""
    # Extract results and metadata
    if isinstance(results_data, dict) and 'results' in results_data:
        results = results_data['results']
        metadata = results_data.get('metadata', {})
    else:
        # For backward compatibility
        results = results_data
        metadata = {}
    
    # If no valid results, create a minimal error report
    if len(results) == 0:
        print(f"WARNING: No valid results to generate report for {file_label}")
        error_report_path = os.path.join(output_dir, f"{file_label}_error_report.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(error_report_path, 'w') as f:
            f.write(f"ERROR: Unable to analyze file {file_label}\n")
            f.write(f"Reason: File too short or no valid data windows\n")
            if metadata:
                f.write(f"File duration: {metadata.get('total_duration_sec', 0):.2f} seconds\n")
                f.write(f"Required minimum: {metadata.get('window_size_sec', WINDOW_SEC)} seconds\n")
        print(f"Error report saved to {error_report_path}")
        return {"error_report": error_report_path}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{file_label}_{timestamp}"
      # Save JSON results
    json_filepath = os.path.join(output_dir, f"{base_filename}_mindfulness_results.json")
    save_results_to_json(results_data, json_filepath)
    
    # Save CSV summary
    csv_data = []
    for r in results:
        row = {
            'timestamp': r['timestamp'],
            'mi_score': round(r['mi_score'], 2),
            'behavioral_state': r['behavioral_state']
        }
        for feature, value in r['features'].items():
            row[feature] = round(value, 2)  
        if r['eda_value'] is not None:
            row['eda_norm'] = round(r['eda_value'], 2)  
        csv_data.append(row)
    
    csv_filepath = os.path.join(output_dir, f"{base_filename}_mindfulness_data.csv")
    pd.DataFrame(csv_data).to_csv(csv_filepath, index=False)
    print(f"CSV data saved to {csv_filepath}")
      # Save simplified CSV with just MI index
    simplified_csv_data = []
    for r in results:
        simplified_csv_data.append({
            'timestamp': r['timestamp'],
            'mi_score': r['mi_score'],
            'behavioral_state': r['behavioral_state']
        })
    simplified_csv_filepath = os.path.join(output_dir, f"{base_filename}_mi_only.csv")
    pd.DataFrame(simplified_csv_data).to_csv(simplified_csv_filepath, index=False)
    print(f"Simplified MI data saved to {simplified_csv_filepath}")
    
    # Create plots
    plot_mi_filepath = os.path.join(output_dir, f"{base_filename}_mi_timeseries.png")
    plot_mi_timeseries(results, plot_mi_filepath, file_label)
    
    feature_plot_filepath = os.path.join(output_dir, f"{base_filename}_feature_contributions.png")
    plot_feature_contributions(results, feature_plot_filepath, file_label)
    
    state_plot_filepath = os.path.join(output_dir, f"{base_filename}_behavioral_states.png")
    plot_behavioral_state_summary(results, state_plot_filepath, file_label)
    
    # Generate summary text file
    summary_filepath = os.path.join(output_dir, f"{base_filename}_summary.txt")
    
    with open(summary_filepath, 'w') as f:
        f.write("=== Mindfulness Index (MI) Analysis Summary ===\n\n")
        f.write(f"File: {file_label}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # Add EDA file used
        eda_file = metadata.get('eda_path', None)
        f.write(f"EDA file used: {eda_file if eda_file else 'None'}\n\n")
        # Parameters
        f.write(f"- Window Size: {WINDOW_SEC} seconds\n")
        f.write(f"- Window Overlap: {OVERLAP * 100}%\n")
        f.write(f"- Frequency Bands: Theta {THETA_BAND}, Alpha {ALPHA_BAND}, Beta {BETA_BAND}\n")
        f.write(f"- MI Weights: {MI_WEIGHTS}\n")
        f.write(f"- State Thresholds: Focused >= {THRESHOLDS['focused']}, " \
                f"Neutral >= {THRESHOLDS['neutral']}, Unfocused < {THRESHOLDS['neutral']}\n\n")

        # --- Rationale Section ---
        f.write("Rationale for Data and Parameters\n")
        f.write("-------------------------------\n")
        f.write("Mindfulness Index (MI) is computed as a weighted sum of neurophysiological features, followed by normalization:\n")
        f.write("\n")
        f.write("    MI_raw = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)\n")
        f.write("    MI = 1 / (1 + exp(-MI_raw + 1))  # Normalized to 0-1 range\n\n")
        f.write("- Theta_Fz (frontal theta): Higher values reflect focused attention and meditation.\n")
        f.write("- Alpha_PO (posterior alpha): Higher values reflect relaxed alertness.\n")
        f.write("- FAA (frontal alpha asymmetry): Reflects emotional valence and approach/withdrawal.\n")
        f.write("- Beta_Frontal: Higher values reflect active thinking and less mindfulness (negative weight).\n")
        f.write("- EDA_norm: Higher EDA (arousal) is negatively correlated with mindfulness (negative weight).\n\n")
        f.write("EDA data is normalized (z-score or min-max) and aligned to EEG windows. If EDA is missing or invalid, it is set to zero for that window.\n\n")
        f.write("Behavioral state thresholds:\n")
        f.write(f"- Focused: MI >= {THRESHOLDS['focused']}\n")
        f.write(f"- Neutral: {THRESHOLDS['neutral']} <= MI < {THRESHOLDS['focused']}\n")
        f.write(f"- Unfocused: MI < {THRESHOLDS['neutral']}\n\n")
        f.write("These thresholds are empirically chosen to reflect meaningful distinctions in mindfulness levels.\n\n")
        
    print(f"Summary report saved to {summary_filepath}")
    
    # After summary report is saved, generate PDF report
    summary_text = open(summary_filepath).read()
    plot_paths = [plot_mi_filepath, feature_plot_filepath, state_plot_filepath]
    table_csv_paths = [csv_filepath, simplified_csv_filepath]
    output_pdf_path = os.path.join(output_dir, f"{base_filename}_report.pdf")
    generate_pdf_report(summary_text, plot_paths, table_csv_paths, output_pdf_path)
    
    # Add PDF path to return dict
    return {
        'json': json_filepath,
        'csv': csv_filepath,
        'simplified_csv': simplified_csv_filepath,
        'mi_plot': plot_mi_filepath,
        'feature_plot': feature_plot_filepath,
        'state_plot': state_plot_filepath,
        'summary': summary_filepath,
        'report_pdf': output_pdf_path
    }

def generate_pdf_report(summary_text, plot_paths, table_csv_paths, output_pdf_path, title="EEG Mindfulness Index Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)

    # Add summary text
    pdf.set_font("Arial", '', 12)
    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 8, line)
    pdf.ln(5)

    # Add plots
    for plot_path in plot_paths:
        if os.path.exists(plot_path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, os.path.basename(plot_path), ln=True, align='C')
            pdf.image(plot_path, x=10, w=pdf.w - 20)
            pdf.ln(5)

    # Add tables (as CSVs)
    for csv_path in table_csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, os.path.basename(csv_path), ln=True, align='C')
            pdf.set_font("Arial", '', 8)
            col_width = pdf.w / (len(df.columns) + 1)
            for col in df.columns:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()
            for _, row in df.iterrows():
                for item in row:
                    pdf.cell(col_width, 8, str(item), border=1)
                pdf.ln()
            pdf.ln(5)

    pdf.output(output_pdf_path)
    print(f"PDF report saved to {output_pdf_path}")

# --- Comprehensive Report Generation ---
def generate_comprehensive_report(all_results, output_dir):
    """Generate a comprehensive report combining all per-file results, summaries, plots, and tables."""
    import pandas as pd
    from fpdf import FPDF
    import os
    from datetime import datetime

    # Prepare output paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"comprehensive_mindfulness_report_{timestamp}"
    markdown_report_path = os.path.join(output_dir, f"{base_filename}.md")
    pdf_report_path = os.path.join(output_dir, f"{base_filename}.pdf")
    merged_csv_path = os.path.join(output_dir, f"{base_filename}_all_mi_data.csv")

    # Aggregate summaries and plots
    summary_texts = []
    all_csvs = []
    all_plots = []
    all_tables = []
    for entry in all_results:
        report = entry.get('file_report', {})
        eda_file = entry.get('eda_file', None)
        # Add summary text
        summary_path = report.get('summary')
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                # Prepend EDA file info to each file's summary
                summary_texts.append(f"# {entry['file_label']}\nEDA file used: {eda_file if eda_file else 'None'}\n" + f.read() + "\n\n")
        # Add CSVs for merging
        csv_path = report.get('csv')
        if csv_path and os.path.exists(csv_path):
            all_csvs.append(csv_path)
            all_tables.append(csv_path)
        # Add simplified CSVs as tables
        simp_csv = report.get('simplified_csv')
        if simp_csv and os.path.exists(simp_csv):
            all_tables.append(simp_csv)
        # Add plots
        for plot_key in ['mi_plot', 'feature_plot', 'state_plot']:
            plot_path = report.get(plot_key)
            if plot_path and os.path.exists(plot_path):
                all_plots.append(plot_path)

    # Merge all MI CSVs
    if all_csvs:
        df_list = [pd.read_csv(csv) for csv in all_csvs]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(merged_csv_path, index=False)
        all_tables.insert(0, merged_csv_path)

    # Combine all summary text
    combined_summary = "\n---\n".join(summary_texts)
    with open(markdown_report_path, 'w') as f:
        f.write(f"# Comprehensive EEG Mindfulness Index Report\n\n")
        f.write(combined_summary)
        f.write(f"\n---\n")
        f.write(f"## Merged MI Data Table: {os.path.basename(merged_csv_path)}\n")

    # Generate comprehensive PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Comprehensive EEG Mindfulness Index Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    for line in combined_summary.split('\n'):
        pdf.multi_cell(0, 8, line)
    pdf.ln(5)
    # Add merged table
    if os.path.exists(merged_csv_path):
        df = pd.read_csv(merged_csv_path)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, os.path.basename(merged_csv_path), ln=True, align='C')
        pdf.set_font("Arial", '', 8)
        col_width = pdf.w / (len(df.columns) + 1)
        for col in df.columns:
            pdf.cell(col_width, 8, str(col), border=1)
        pdf.ln()
        for _, row in df.iterrows():
            for item in row:
                pdf.cell(col_width, 8, str(item), border=1)
            pdf.ln()
        pdf.ln(5)
    # Add all plots
    for plot_path in all_plots:
        if os.path.exists(plot_path):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, os.path.basename(plot_path), ln=True, align='C')
            pdf.image(plot_path, x=10, w=pdf.w - 20)
            pdf.ln(5)
    # Add all tables (as CSVs)
    for csv_path in all_tables:
        if os.path.exists(csv_path) and csv_path != merged_csv_path:
            df = pd.read_csv(csv_path)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, os.path.basename(csv_path), ln=True, align='C')
            pdf.set_font("Arial", '', 8)
            col_width = pdf.w / (len(df.columns) + 1)
            for col in df.columns:
                pdf.cell(col_width, 8, str(col), border=1)
            pdf.ln()
            for _, row in df.iterrows():
                for item in row:
                    pdf.cell(col_width, 8, str(item), border=1)
                pdf.ln()
            pdf.ln(5)
    pdf.output(pdf_report_path)
    print(f"Comprehensive PDF report saved to {pdf_report_path}")
    return {
        'markdown_report': markdown_report_path,
        'pdf_report': pdf_report_path,
        'merged_csv': merged_csv_path
    }

# --- Comprehensive Report Generation ---
def generate_comprehensive_report(all_results, output_dir):
    """Generate a comprehensive report across all processed files."""
    import pandas as pd
    from datetime import datetime
    import os

    # Prepare summary text
    summary_lines = ["=== Comprehensive Mindfulness Index (MI) Analysis Summary ===\n"]
    summary_lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_lines.append(f"Number of files analyzed: {len(all_results)}\n\n")

    # Aggregate MI and state data
    all_mi = []
    all_states = []
    per_file_stats = []
    for entry in all_results:
        file_label = entry['file_label']
        results_data = entry['results_data']
        if isinstance(results_data, dict) and 'results' in results_data:
            results = results_data['results']
        else:
            results = results_data
        mi_values = [r['mi_score'] for r in results] if results else []
        states = [r['behavioral_state'] for r in results] if results else []
        all_mi.extend(mi_values)
        all_states.extend(states)
        per_file_stats.append({
            'file': file_label,
            'windows': len(results),
            'avg_mi': np.mean(mi_values) if mi_values else None,
            'min_mi': np.min(mi_values) if mi_values else None,
            'max_mi': np.max(mi_values) if mi_values else None,
            'focused_pct': (states.count('Focused') / len(states) * 100) if states else 0,
            'neutral_pct': (states.count('Neutral') / len(states) * 100) if states else 0,
            'unfocused_pct': (states.count('Unfocused') / len(states) * 100) if states else 0
        })

    # Overall stats
    summary_lines.append(f"Total windows analyzed: {len(all_mi)}\n")
    if all_mi:
        summary_lines.append(f"Average MI score (all files): {np.mean(all_mi):.4f}\n")
        summary_lines.append(f"MI score range: {np.min(all_mi):.4f} to {np.max(all_mi):.4f}\n")
    for state in ['Focused', 'Neutral', 'Unfocused']:
        count = all_states.count(state)
        pct = (count / len(all_states) * 100) if all_states else 0
        summary_lines.append(f"- {state}: {count} windows ({pct:.1f}%)\n")
    summary_lines.append("\nPer-file summary:\n")
    for stats in per_file_stats:
        summary_lines.append(
            f"{stats['file']}: windows={stats['windows']}, avg_mi={stats['avg_mi']:.4f} "
            f"(min={stats['min_mi']:.4f}, max={stats['max_mi']:.4f}), "
            f"Focused={stats['focused_pct']:.1f}%, Neutral={stats['neutral_pct']:.1f}%, Unfocused={stats['unfocused_pct']:.1f}%\n"
        )

    # Save summary text
    summary_text = ''.join(summary_lines)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"comprehensive_report_{timestamp}"
    summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Comprehensive summary saved to {summary_path}")

    # Combine all simplified MI CSVs
    combined_csv_path = os.path.join(output_dir, f"{base_filename}_mi_only_combined.csv")
    combined_rows = []
    for entry in all_results:
        file_report = entry.get('file_report', {})
        csv_path = file_report.get('simplified_csv')
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['file'] = entry['file_label']
            combined_rows.append(df)
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined MI CSV saved to {combined_csv_path}")
    else:
        combined_csv_path = None

    # Collect all plots for PDF
    plot_paths = []
    for entry in all_results:
        file_report = entry.get('file_report', {})
        for key in ['mi_plot', 'feature_plot', 'state_plot']:
            plot_path = file_report.get(key)
            if plot_path and os.path.exists(plot_path):
                plot_paths.append(plot_path)

    # Add all per-file summary text files as tables (optional)
    table_csv_paths = [combined_csv_path] if combined_csv_path else []

    # Generate PDF report
    output_pdf_path = os.path.join(output_dir, f"{base_filename}_report.pdf")
    generate_pdf_report(summary_text, plot_paths, table_csv_paths, output_pdf_path, title="Comprehensive EEG Mindfulness Index Report")

    return {
        'summary': summary_path,
        'combined_csv': combined_csv_path,
        'report_pdf': output_pdf_path
    }

def process_directory(input_dir, output_dir='results'):
    """Process all EEG files in a directory"""
    # Get all CSV files in the directory
    eeg_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not eeg_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(eeg_files)} EEG files to process")
    
    # Check EDA data folder existence
    if not os.path.exists(EDA_DATA_FOLDER):
        print(f"WARNING: EDA data folder {EDA_DATA_FOLDER} does not exist")
        print("Creating directory...")
        try:
            os.makedirs(EDA_DATA_FOLDER, exist_ok=True)
        except Exception as e:
            print(f"Error creating EDA folder: {e}")
    
    # Pre-scan for available OpenSignals files
    available_opensignals = []
    if os.path.exists(EDA_DATA_FOLDER):
        available_opensignals = glob.glob(os.path.join(EDA_DATA_FOLDER, "opensignals_*.txt"))
        print(f"Found {len(available_opensignals)} OpenSignals files in {EDA_DATA_FOLDER}")
        
        # Print details of found OpenSignals files
        if available_opensignals:
            print("Available OpenSignals files:")
            for file in available_opensignals:
                size_kb = os.path.getsize(file) / 1024
                mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {os.path.basename(file)} ({size_kb:.1f} KB, modified: {mod_time})")
    
    # Process each file
    for eeg_file in eeg_files:
        file_label = os.path.splitext(os.path.basename(eeg_file))[0]
        print(f"\nProcessing {file_label}...")
        
        # Extract EEG file date components if available (for potential date-based matching)
        eeg_date_parts = None
        # Check if filename follows UnicornRecorder_DD_MM_YYYY pattern
        if file_label.startswith("UnicornRecorder_"):
            parts = file_label.split("_")
            if len(parts) >= 4:
                try:
                    # Extract day, month, year from filename
                    day, month, year = int(parts[1]), int(parts[2]), int(parts[3])
                    eeg_date_parts = {"day": day, "month": month, "year": year}
                    print(f"Extracted date from EEG filename: {day}/{month}/{year}")
                except (ValueError, IndexError):
                    pass
            
        # Look for corresponding EDA file in the dedicated EDA folder
        # Try different possible file formats (CSV and TXT)
        possible_eda_files = [
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.csv"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.txt"),
            os.path.join(EDA_DATA_FOLDER, f"opensignals_{file_label}.txt"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}.txt")  # Same name but .txt extension
        ]
        
        eda_file = None
        for potential_file in possible_eda_files:
            print(f"Looking for EDA data at: {potential_file}")
            if os.path.exists(potential_file):
                eda_file = potential_file
                print(f"Found EDA data: {eda_file}")
                break
        
        # If no exact match found, try date-based matching with OpenSignals files
        if eda_file is None and eeg_date_parts and available_opensignals:
            print("Attempting date-based matching with OpenSignals files...")
            # First attempt: Look for OpenSignals files from the same date
            for opensignals_file in available_opensignals:
                filename = os.path.basename(opensignals_file)
                # OpenSignals files typically have format: opensignals_*_YYYY-MM-DD_HH-MM-SS.txt
                if f"{eeg_date_parts['year']}-{eeg_date_parts['month']:02d}-{eeg_date_parts['day']:02d}" in filename:
                    eda_file = opensignals_file
                    print(f"Found date-matching OpenSignals file: {os.path.basename(eda_file)}")
                    break
        
        # If still no match, use the most recent OpenSignals file
        if eda_file is None and available_opensignals:
            # Use the most recent OpenSignals file if multiple exist
            available_opensignals.sort(key=os.path.getmtime, reverse=True)
            eda_file = available_opensignals[0]
            print(f"No exact match found. Using most recent OpenSignals file: {os.path.basename(eda_file)}")
        
        if eda_file is None:
            print(f"No EDA data found for {file_label} in {EDA_DATA_FOLDER}")
            print("WARNING: Mindfulness Index will be calculated without EDA data")
        
        # Process EEG file
        results = process_eeg_file(eeg_file, eda_file)
        
        # Generate report
        generate_report(results, file_label, output_dir)
        
        print(f"Completed processing {file_label}")
    
    print(f"\nAll {len(eeg_files)} files processed")

if __name__ == "__main__":
    # Process a single file example
    # Replace with your actual file path
    data_dir = "data/toClasify"
    
    # Create results directory if it doesn't exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/mindfulness_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create EDA data directory if it doesn't exist
    os.makedirs(EDA_DATA_FOLDER, exist_ok=True)
    
    print("\n============================================================")
    print("EEG MINDFULNESS INDEX CALCULATOR")
    print("============================================================")
    print("This script calculates the Mindfulness Index (MI) from EEG and EDA data.")
    print("The MI value ranges from 0 to 1, where higher values indicate higher mindfulness.")
    print("\nMI Formula:")
    print("  MI_raw = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)")
    print("  MI = 1 / (1 + exp(-MI_raw + 1))  # Normalized to 0-1 range")
    print("\n============================================================")
    print("EDA DATA CONFIGURATION")
    print("============================================================")
    print(f"EDA data will be searched in: {EDA_DATA_FOLDER}")
    print("To change this location, modify EDA_DATA_FOLDER in the script.")
    print("\nSupported EDA file formats and detection methods:")
    print("1. Exact name matching (in order of preference):")
    print("  - [filename]_eda.csv")
    print("  - [filename]_eda.txt")
    print("  - opensignals_[filename].txt")
    print("  - [filename].txt")
    print("2. Date-based matching:")
    print("  - For UnicornRecorder_DD_MM_YYYY_* EEG files, will search for")
    print("    opensignals_*_YYYY-MM-DD_*.txt files")
    print("3. Fallback method:")
    print("  - If no match is found, will use the most recent opensignals_*.txt file")
    print("\nSupported EDA file formats:")
    print("  - CSV files with EDA values in the second column")
    print("  - OpenSignals text files (.txt) with tab-separated values")
    print("    (EDA channel is auto-detected from the header)")
    print("============================================================\n")
    
    # Process all files in directory and collect results for comprehensive report
    print("Processing EEG files...")
    
    # Get all CSV files in the directory
    eeg_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # Keep track of all results
    all_results = []
    
    # Process each file
    for eeg_file in eeg_files:
        file_label = os.path.splitext(os.path.basename(eeg_file))[0]
        print(f"\nProcessing {file_label}...")
        
        # Find EDA file (using the logic from process_directory)
        # Extract EEG file date components if available (for potential date-based matching)
        eeg_date_parts = None
        # Check if filename follows UnicornRecorder_DD_MM_YYYY pattern
        if file_label.startswith("UnicornRecorder_"):
            parts = file_label.split("_")
            if len(parts) >= 4:
                try:
                    # Extract day, month, year from filename
                    day, month, year = int(parts[1]), int(parts[2]), int(parts[3])
                    eeg_date_parts = {"day": day, "month": month, "year": year}
                    print(f"Extracted date from EEG filename: {day}/{month}/{year}")
                except (ValueError, IndexError):
                    pass
                    
        # Look for corresponding EDA file in the dedicated EDA folder
        # Try different possible file formats (CSV and TXT)
        possible_eda_files = [
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.csv"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.txt"),
            os.path.join(EDA_DATA_FOLDER, f"opensignals_{file_label}.txt"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}.txt")  # Same name but .txt extension
        ]
        
        eda_file = None
        for potential_file in possible_eda_files:
            print(f"Looking for EDA data at: {potential_file}")
            if os.path.exists(potential_file):
                eda_file = potential_file
                print(f"Found EDA data: {eda_file}")
                break
        
        # If no exact match found, try date-based matching with OpenSignals files
        available_opensignals = []
        if os.path.exists(EDA_DATA_FOLDER):
            available_opensignals = glob.glob(os.path.join(EDA_DATA_FOLDER, "opensignals_*.txt"))
            
        if eda_file is None and eeg_date_parts and available_opensignals:
            print("Attempting date-based matching with OpenSignals files...")
            # First attempt: Look for OpenSignals files from the same date
            for opensignals_file in available_opensignals:
                filename = os.path.basename(opensignals_file)
                # OpenSignals files typically have format: opensignals_*_YYYY-MM-DD_HH-MM-SS.txt
                if f"{eeg_date_parts['year']}-{eeg_date_parts['month']:02d}-{eeg_date_parts['day']:02d}" in filename:
                    eda_file = opensignals_file
                    print(f"Found date-matching OpenSignals file: {os.path.basename(eda_file)}")
                    break
        
        # If still no match, use the most recent OpenSignals file
        if eda_file is None and available_opensignals:
            # Use the most recent OpenSignals file if multiple exist
            available_opensignals.sort(key=os.path.getmtime, reverse=True)
            eda_file = available_opensignals[0]
            print(f"No exact match found. Using most recent OpenSignals file: {os.path.basename(eda_file)}")
        
        if eda_file is None:
            print(f"No EDA data found for {file_label} in {EDA_DATA_FOLDER}")
            print("WARNING: Mindfulness Index will be calculated without EDA data")
        
        # Process EEG file
        results = process_eeg_file(eeg_file, eda_file)
        
        # Generate individual report
        file_report = generate_report(results, file_label, results_dir)
        
        # Add to list for comprehensive report
        all_results.append({
            'file_label': file_label,
            'results_data': results,
            'file_report': file_report,
            'eeg_file': eeg_file,
            'eda_file': eda_file
        })
        
        print(f"Completed processing {file_label}")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    comprehensive_report = generate_comprehensive_report(all_results, results_dir)
    
    print("\nProcessing complete!")
    print(f"Results are saved in: {os.path.abspath(results_dir)}")
    if comprehensive_report and 'markdown_report' in comprehensive_report:
        print(f"Comprehensive report: {os.path.abspath(comprehensive_report['markdown_report'])}")
    
    print("\nMindfulness index analysis complete.")
