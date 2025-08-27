"""
EEG Signal Processing Workflow - Core Components

This snippet demonstrates the essential signal processing workflow used in the EEG emotion classification pipeline.
It includes the key steps: data loading, filtering, windowing, feature extraction, and normalization.

Key Components:
1. Data loading from CSV files
2. Bandpass filtering (1-40 Hz) and notch filtering (50 Hz powerline noise removal)
3. Windowing/epoching with overlap
4. Feature extraction (statistical features from frequency bands)
5. Z-score normalization

Author: EEG-SVM Pipeline
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch, filtfilt
from scipy.stats import kurtosis, skew

# =============================================================================
# SIGNAL PROCESSING PARAMETERS
# =============================================================================

# Sampling parameters
FS = 250  # Sampling frequency (Hz)
BANDPASS = (1, 40)  # Bandpass filter range (Hz)
NOTCH_FREQ = 50  # Notch filter frequency (Hz) - powerline noise

# Windowing parameters
EPOCH_SEC = 2  # Window/epoch duration in seconds
EPOCH_OVERLAP = 1  # Overlap between windows in seconds

# EEG channel names (8 channels)
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# Frequency bands for feature extraction
BANDS = {
    "delta": (1, 4),    # Delta waves
    "theta": (4, 8),    # Theta waves  
    "alpha": (8, 13),   # Alpha waves
    "beta": (13, 30),   # Beta waves
    "gamma": (30, 40)   # Gamma waves
}

# =============================================================================
# CORE SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def load_eeg_csv(filepath):
    """
    Load EEG data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file containing EEG data
    
    Returns:
    --------
    numpy.ndarray
        EEG data array (samples x channels)
    """
    df = pd.read_csv(filepath, dtype=str)
    # Convert EEG channel columns (1-8) to float, handle conversion errors
    eeg_data = df.iloc[:, 1:9].apply(pd.to_numeric, errors='coerce').values
    # Remove rows with any NaN values
    eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
    return eeg_data


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter to EEG data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data (samples x channels)
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : int
        Sampling frequency (Hz)
    order : int
        Filter order
    
    Returns:
    --------
    numpy.ndarray
        Filtered EEG data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)


def notch_filter(data, freq, fs, Q=30):
    """
    Apply notch filter to remove powerline noise.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data (samples x channels)
    freq : float
        Notch frequency (Hz) - typically 50 Hz or 60 Hz
    fs : int
        Sampling frequency (Hz)
    Q : int
        Quality factor for notch filter
    
    Returns:
    --------
    numpy.ndarray
        Notch-filtered EEG data
    """
    b, a = iirnotch(freq, Q, fs)
    return lfilter(b, a, data, axis=0)


def windowed_epochs(data, fs, epoch_sec, overlap_sec):
    """
    Create overlapping windows/epochs from continuous EEG data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data (samples x channels)
    fs : int
        Sampling frequency (Hz)
    epoch_sec : float
        Duration of each epoch in seconds
    overlap_sec : float
        Overlap between epochs in seconds
    
    Returns:
    --------
    numpy.ndarray
        Array of epochs (epochs x samples x channels)
    """
    step = int(fs * (epoch_sec - overlap_sec))  # Step size between windows
    win_size = int(fs * epoch_sec)  # Window size in samples
    epochs = []
    
    for start in range(0, data.shape[0] - win_size + 1, step):
        epochs.append(data[start:start+win_size, :])
    
    return np.array(epochs)


def compute_band_power(data, fs, band):
    """
    Compute average power in a specific frequency band using Welch's method.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Single-channel EEG data
    fs : int
        Sampling frequency (Hz)
    band : tuple
        Frequency band as (low_freq, high_freq)
    
    Returns:
    --------
    float
        Average power in the specified band
    """
    fmin, fmax = band
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    return np.mean(Pxx[idx_band])


def extract_stat_features(epoch, fs):
    """
    Extract statistical features from an EEG epoch.
    
    For each channel and each frequency band, computes 5 statistical features:
    - Mean, Variance, Standard Deviation, Kurtosis, Skewness
    
    Total features: 8 channels × 5 bands × 5 stats = 200 features per epoch
    
    Parameters:
    -----------
    epoch : numpy.ndarray
        Single epoch of EEG data (samples x channels)
    fs : int
        Sampling frequency (Hz)
    
    Returns:
    --------
    list
        Feature vector (200 features)
    """
    feats = []
    
    # For each channel
    for ch in range(epoch.shape[1]):
        # For each frequency band
        for band in BANDS.values():
            # Apply bandpass filter for this specific band
            low, high = band
            nyq = 0.5 * fs
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            band_sig = filtfilt(b, a, epoch[:, ch])
            
            # Extract 5 statistical features from the band-filtered signal
            feats.extend([
                np.mean(band_sig),      # Mean amplitude
                np.var(band_sig),       # Variance
                np.std(band_sig),       # Standard deviation
                kurtosis(band_sig),     # Kurtosis (tail heaviness)
                skew(band_sig)          # Skewness (asymmetry)
            ])
    
    return feats


def zscore_features(X):
    """
    Apply z-score normalization to feature matrix.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix (samples x features)
    
    Returns:
    --------
    tuple
        (normalized_features, mean, std) for later use
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    zX = (X - mu) / (sigma + 1e-8)  # Small epsilon to avoid division by zero
    return zX, mu, sigma


# =============================================================================
# COMPLETE SIGNAL PROCESSING WORKFLOW
# =============================================================================

def process_eeg_signal(filepath, training_stats=None):
    """
    Complete signal processing workflow for EEG data.
    
    Parameters:
    -----------
    filepath : str
        Path to EEG CSV file
    training_stats : tuple, optional
        (mu, sigma) from training data for normalization. If None, computes from current data.
    
    Returns:
    --------
    dict
        Dictionary containing processed data and intermediate results
    """
    print(f"Processing EEG file: {filepath}")
    
    # Step 1: Load EEG data
    print("1. Loading EEG data...")
    eeg_raw = load_eeg_csv(filepath)
    print(f"   Loaded {eeg_raw.shape[0]} samples × {eeg_raw.shape[1]} channels")
    
    # Step 2: Apply bandpass filter
    print("2. Applying bandpass filter (1-40 Hz)...")
    eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, FS)
    
    # Step 3: Apply notch filter for powerline noise
    print("3. Applying notch filter (50 Hz powerline noise removal)...")
    eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, FS)
    
    # Step 4: Create windowed epochs
    print(f"4. Creating overlapping epochs ({EPOCH_SEC}s windows, {EPOCH_OVERLAP}s overlap)...")
    epochs = windowed_epochs(eeg_filt, FS, EPOCH_SEC, EPOCH_OVERLAP)
    print(f"   Created {epochs.shape[0]} epochs")
    
    # Step 5: Extract features from each epoch
    print("5. Extracting statistical features from each epoch...")
    feats = []
    for ep in epochs:
        feats.append(extract_stat_features(ep, FS))
    feats = np.array(feats)
    print(f"   Extracted {feats.shape[0]} epochs × {feats.shape[1]} features")
    
    # Step 6: Normalize features
    print("6. Normalizing features...")
    if training_stats is not None:
        # Use provided training statistics
        mu, sigma = training_stats
        feats_normalized = (feats - mu) / (sigma + 1e-8)
        print("   Applied training statistics for normalization")
    else:
        # Compute statistics from current data
        feats_normalized, mu, sigma = zscore_features(feats)
        print("   Computed normalization statistics from current data")
    
    print("Signal processing complete!\n")
    
    # Return all processed data
    return {
        'raw_data': eeg_raw,
        'filtered_data': eeg_filt,
        'epochs': epochs,
        'features': feats,
        'features_normalized': feats_normalized,
        'normalization_stats': (mu, sigma),
        'n_epochs': epochs.shape[0],
        'n_features': feats.shape[1]
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example demonstrating the complete signal processing workflow.
    """
    
    # Example: Process a single EEG file
    try:
        # You would replace this with your actual file path
        eeg_file = "data/toClasify/example_eeg.csv"  # Replace with actual path
        
        # Run the complete workflow
        results = process_eeg_signal(eeg_file)
        
        # Display summary
        print("=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Original data shape: {results['raw_data'].shape}")
        print(f"Number of epochs: {results['n_epochs']}")
        print(f"Features per epoch: {results['n_features']}")
        print(f"Normalized features shape: {results['features_normalized'].shape}")
        print(f"Normalization stats shapes: μ={results['normalization_stats'][0].shape}, σ={results['normalization_stats'][1].shape}")
        
        # Example: Show first few features of first epoch
        print(f"\nFirst epoch features (first 10): {results['features'][0][:10]}")
        print(f"First epoch normalized (first 10): {results['features_normalized'][0][:10]}")
        
    except FileNotFoundError:
        print("Example file not found. Please update the file path in the example.")
        print("\nTo use this workflow:")
        print("1. Update 'eeg_file' variable with your actual CSV file path")
        print("2. Ensure CSV has EEG data in columns 1-8")
        print("3. Run: results = process_eeg_signal('your_file.csv')")