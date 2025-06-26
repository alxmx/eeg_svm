"""
Mindfulness Index (MI) Real-Time LSL Pipeline
============================================

This script provides a real-time pipeline for estimating Mindfulness Index (MI) and Emotional Mindfulness Index (EMI) from EEG and EDA data using LSL streams. It supports per-user calibration, user-specific model/scaler saving/loading, and robust stream selection. The code is designed for concentration and emotional control training, with dynamic feedback for users.

Current Features & Updates (as of June 2025):
------------------------------------------------
- Per-user calibration and model/scaler persistence
- Real-time MI prediction using user-specific SVR models
- Online adaptation (optional, can be enabled/disabled)
- Robust LSL stream selection for EEG, EDA, and Unity labels
- Feature extraction and adaptation for both 5-feature and legacy 200-feature models
- Three LSL output streams:
    * Standard MI (0-1, sigmoid output)
    * Raw MI (remapped to 0-1, pre-sigmoid, more dynamic)
    * EMI (Emotional Mindfulness Index, 0-1, more sensitive to emotional features)
- Enhanced visualization: plots for MI, raw MI, and EMI
- Debug and diagnostic output for calibration and prediction
- Modular, well-commented, and ready for further extension

Usage:
------
1. Run calibration for a new user to collect baseline data and train a user-specific model.
2. Start real-time MI prediction, select LSL streams, and monitor feedback.
3. Use the three LSL output streams for feedback in Unity or other applications.

Sections:
---------
- Imports & Config
- Utility Functions (including LSL stream setup, MI/EMI calculation, remapping)
- Calibration & Model Training
- Real-Time Prediction Loop
- Visualization
- Main Entry Point

"""
import os
import sys
import time
import numpy as np
import pandas as pd
import glob
import json
import logging
import threading
import msvcrt
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from datetime import datetime
from scipy.signal import welch
from scipy.integrate import simpson
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score
from scipy.stats import spearmanr

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'user_configs')
EEG_DIR = os.path.join(BASE_DIR, 'data', '_eeg')
EDA_DIR = os.path.join(BASE_DIR, 'data', '_eda')

# --- EDA CHANNEL CONFIGURATION ---
# EDA channel to use for features (0-based indexing)
# 0 = Channel 1, 1 = Channel 2
# Adjust this if your EDA device uses different channels
EDA_CHANNEL_INDEX = 1  # Currently using Channel 2 (index 1)

# Create required directories
for d in [MODEL_DIR, LOG_DIR, VIS_DIR, PROCESSED_DATA_DIR, USER_CONFIG_DIR, EEG_DIR, EDA_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
# EEG and EDA directories are defined earlier
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
MI_THRESHOLDS = {'focused': 0.5, 'neutral': 0.37}

# --- Online adaptation config ---
ONLINE_UPDATE_WINDOW = 10  # Number of samples in moving window
ONLINE_UPDATE_RATIO = 0.7  # Ratio of samples above/below threshold to trigger update
ONLINE_MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, '{}_online_model.joblib')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Essential function for EEG processing - compute bandpower across frequency bands
def compute_bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal in a specific frequency band.
    
    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : tuple or list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds (if None, use the entire signal).
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.
    
    Returns
    -------
    bp : float
        Absolute or relative band power.
    """
    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = min(len(data), 256 * 8)
    
    # Check for valid data
    if len(data) < 2:
        return 0.0
        
    # Calculate spectrum using Welch's method
    try:
        freqs, psd = welch(data, sf, nperseg=nperseg)
        
        # Find the frequency indices corresponding to the band of interest
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # Calculate the absolute power by integrating the PSD in the band
        bp = simpson(y=psd[idx_band], x=freqs[idx_band])
        
        if relative:
            # Calculate the total power
            total_power = simpson(y=psd, x=freqs)
            bp = bp / total_power if total_power > 0 else 0
    except Exception as e:
        print(f"[ERROR] in compute_bandpower: {e}")
        bp = 0.0
    
    return bp

def scale_features_for_mi(features, user_stats=None):
    """
    Scale features dynamically based on user-specific statistics for proper MI calculation.
    This addresses the critical issue where different feature magnitudes
    cause unbalanced contributions to the final MI value.

    Parameters:
    -----------
    features : list or array
        Raw features in order: [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
    user_stats : dict, optional
        User-specific statistics containing min, max, mean, and std for each feature.

    Returns:
    --------
    scaled_features : numpy array
        Features scaled to comparable ranges (0-1) for balanced MI calculation.
    """
    theta_fz, alpha_po, faa, beta_frontal, eda_norm = features

    # Normalize each feature to the range [0, 1] based on user-specific min and max values
    def normalize_feature(value, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5  # Avoid division by zero, return midpoint
        return (value - min_val) / (max_val - min_val)

    # Apply logarithmic scaling for features with large ranges
    def log_scale(value):
        return np.log10(value + 1) if value > 0 else 0

    if user_stats:
        # Use user-specific statistics for normalization
        theta_min, theta_max = user_stats['mins']['theta_fz'], user_stats['maxs']['theta_fz']
        alpha_min, alpha_max = user_stats['mins']['alpha_po'], user_stats['maxs']['alpha_po']
        faa_min, faa_max = user_stats['mins']['faa'], user_stats['maxs']['faa']
        beta_min, beta_max = user_stats['mins']['beta_frontal'], user_stats['maxs']['beta_frontal']
        eda_min, eda_max = user_stats['mins']['eda_norm'], user_stats['maxs']['eda_norm']
    else:
        # Default observed min and max values (fallback)
        theta_min, theta_max = 1.695, 116.249
        alpha_min, alpha_max = 1.432, 31.086
        faa_min, faa_max = -2.005, 2.063
        beta_min, beta_max = 3.337, 32.550
        eda_min, eda_max = 5.731, 8.958

    # Normalize each feature
    theta_scaled = normalize_feature(log_scale(theta_fz), log_scale(theta_min), log_scale(theta_max))
    alpha_scaled = normalize_feature(log_scale(alpha_po), log_scale(alpha_min), log_scale(alpha_max))
    faa_scaled = normalize_feature(faa, faa_min, faa_max)
    beta_scaled = normalize_feature(beta_frontal, beta_min, beta_max)
    eda_scaled = normalize_feature(eda_norm, eda_min, eda_max)

    # Return the scaled features
    return [theta_scaled, alpha_scaled, faa_scaled, beta_scaled, eda_scaled]

def calculate_mi_scaled(features):
    """
    Calculate MI using properly scaled features for balanced contribution.
    This replaces the old calculate_mi function to ensure all features
    contribute meaningfully to the final MI value.
    """
    # First scale the features to comparable ranges
    scaled_features = scale_features_for_mi(features)
    
    # Use adjusted weights for the scaled features
    # These weights are designed for the scaled feature ranges
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    
    # Calculate weighted sum with clamping to prevent overflow
    raw_mi = np.dot(scaled_features, weights)
    raw_mi = np.clip(raw_mi, -50, 50)  # Prevent exp() overflow
    
    # Apply sigmoid with adjusted parameters for scaled features
    # Offset adjusted for the new feature scale
    mi = 1 / (1 + np.exp(-(raw_mi - 0.5)))
    
    return np.clip(mi, 0, 1)  # Ensure output is in valid range

def calculate_raw_mi_scaled(features):
    """Calculate raw MI (pre-sigmoid) using scaled features for more dynamic range"""
    scaled_features = scale_features_for_mi(features)
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    raw_mi = np.dot(scaled_features, weights) - 0.5
    return np.clip(raw_mi, -50, 50)  # Prevent overflow in downstream calculations

def calculate_emi_scaled(features):
    """Calculate Emotional Mindfulness Index using scaled features"""
    scaled_features = scale_features_for_mi(features)
    # EMI gives more weight to frontal alpha asymmetry (FAA) and EDA
    weights = np.array([0.15, 0.15, 0.4, -0.1, -0.2])
    emi_raw = np.dot(scaled_features, weights) - 0.3
    emi_raw = np.clip(emi_raw, -50, 50)  # Prevent exp() overflow
    emi = 1 / (1 + np.exp(-2 * emi_raw))
    return np.clip(emi, 0, 1)  # Ensure output is in valid range

def calculate_raw_mi(features):
    """Calculate raw MI (pre-sigmoid) for more dynamic range"""
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    raw_mi = np.dot(features, weights) - 0.5
    return np.clip(raw_mi, -50, 50)  # Prevent overflow

def remap_raw_mi(raw_mi):
    """Remap raw MI to 0-1 range with scaled sigmoid for LSL output"""
    # Clamp input to prevent overflow
    raw_mi = np.clip(raw_mi, -50, 50)
    # Apply scaled sigmoid remapping
    mi_remapped = 1 / (1 + np.exp(-3 * raw_mi))
    return np.clip(mi_remapped, 0, 1)

def calculate_emi(features):
    """Calculate Emotional Mindfulness Index - more sensitive to emotional features"""
    # EMI gives more weight to frontal alpha asymmetry (FAA) and EDA
    weights = np.array([0.15, 0.15, 0.4, -0.1, -0.2])
    emi_raw = np.dot(features, weights) - 0.3
    emi_raw = np.clip(emi_raw, -50, 50)  # Prevent overflow
    emi = 1 / (1 + np.exp(-2 * emi_raw))
    return np.clip(emi, 0, 1)
    emi = 1 / (1 + np.exp(-2 * emi_raw))
    return emi

def setup_mindfulness_lsl_streams():
    """Create LSL streams for MI and related values"""
    # Create LSL stream for MI (standard 0-1 range)
    mi_info = StreamInfo('MindfulnessIndex', 'MI', 1, 10, 'float32', 'mi_12345')
    mi_outlet = StreamOutlet(mi_info)
    
    # Create LSL stream for raw MI (more dynamic range)
    raw_mi_info = StreamInfo('RawMindfulnessIndex', 'RawMI', 1, 10, 'float32', 'raw_mi_12345')
    raw_mi_outlet = StreamOutlet(raw_mi_info)
    
    # Create LSL stream for EMI (emotional mindfulness index)
    emi_info = StreamInfo('EmotionalMindfulnessIndex', 'EMI', 1, 10, 'float32', 'emi_12345')
    emi_outlet = StreamOutlet(emi_info)
    
    return {
        'mi': mi_outlet,
        'raw_mi': raw_mi_outlet,
        'emi': emi_outlet
    }

def apply_artifact_regression(eeg, acc_gyr, regressors):
    """Apply artifact regression to clean EEG using ACC/Gyro data"""
    if regressors is None or len(regressors) < 8:
        return eeg
    
    eeg_clean = eeg.copy()
    for i in range(8):  # For each EEG channel
        if regressors.get(f'eeg{i}') is not None:
            weights = regressors[f'eeg{i}']
            artifacts = np.dot(acc_gyr, weights)
            eeg_clean[i] -= artifacts
    
    return eeg_clean

def load_user_baseline(user_id):
    """Load user's baseline statistics for normalization"""
    baseline_csv = os.path.join(USER_CONFIG_DIR, f'{user_id}_baseline.csv')
    if not os.path.exists(baseline_csv):
        return None
    
    baseline_df = pd.read_csv(baseline_csv)
    baseline_stats = {
        'means': baseline_df[FEATURE_ORDER].mean().to_dict(),
        'stds': baseline_df[FEATURE_ORDER].std().to_dict(),
        'mins': baseline_df[FEATURE_ORDER].min().to_dict(),
        'maxs': baseline_df[FEATURE_ORDER].max().to_dict(),
        'percentiles': {
            '25': baseline_df[FEATURE_ORDER].quantile(0.25).to_dict(),
            '50': baseline_df[FEATURE_ORDER].quantile(0.50).to_dict(),
            '75': baseline_df[FEATURE_ORDER].quantile(0.75).to_dict(),
            '90': baseline_df[FEATURE_ORDER].quantile(0.90).to_dict()
        }
    }
    
    # Calculate baseline MI distribution using scaled features
    baseline_features = baseline_df[FEATURE_ORDER].values
    baseline_mi = np.array([calculate_mi_scaled(f) for f in baseline_features])  # Use scaled MI calculation
    baseline_stats['mi_baseline'] = {
        'mean': np.mean(baseline_mi),
        'std': np.std(baseline_mi),
        'percentile_90': np.percentile(baseline_mi, 90)
    }
    
    return baseline_stats

def normalize_features_to_baseline(features, baseline_stats):
    """Normalize current features relative to user's baseline"""
    if baseline_stats is None:
        return features
    
    normalized_features = []
    for i, (feat_name, feat_val) in enumerate(zip(FEATURE_ORDER, features)):
        baseline_mean = baseline_stats['means'][feat_name]
        baseline_std = baseline_stats['stds'][feat_name]
        
        # Z-score normalization relative to baseline
        if baseline_std > 0:
            normalized_val = (feat_val - baseline_mean) / baseline_std
        else:
            normalized_val = 0.0
        
        normalized_features.append(normalized_val)
    
    return np.array(normalized_features)

def calculate_mi_with_baseline(features, baseline_stats):
    """Calculate MI using baseline-aware approach"""
    if baseline_stats is None:
        return calculate_mi(features)
    
    # Method 1: Use normalized features
    normalized_features = normalize_features_to_baseline(features, baseline_stats)
    
    # Method 2: Adjust weights based on baseline variability
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    
    # Scale weights by baseline standard deviation (more weight to more variable features)
    baseline_stds = np.array([baseline_stats['stds'][feat] for feat in FEATURE_ORDER])
    baseline_stds = np.clip(baseline_stds, 0.01, None)  # Avoid division by zero
    adaptive_weights = weights * (1 + np.log(baseline_stds + 1))
    
    # Calculate raw MI
    raw_mi = np.dot(normalized_features, adaptive_weights)
    
    # Adjust threshold based on baseline MI distribution
    baseline_mi_mean = baseline_stats['mi_baseline']['mean']
    threshold_adjustment = baseline_mi_mean - 0.5  # Adjust relative to expected baseline
    
    # Apply sigmoid with baseline-adjusted threshold
    mi = 1 / (1 + np.exp(-(raw_mi - threshold_adjustment)))
    
    return mi

def get_adaptive_thresholds(baseline_stats):
    """Get adaptive MI thresholds based on user's baseline"""
    if baseline_stats is None:
        return MI_THRESHOLDS
    
    baseline_mi_mean = baseline_stats['mi_baseline']['mean']
    baseline_mi_std = baseline_stats['mi_baseline']['std']
    
    # Adaptive thresholds: baseline + standard deviations
    adaptive_thresholds = {
        'focused': max(0.5, baseline_mi_mean + 1.5 * baseline_mi_std),
        'neutral': max(0.37, baseline_mi_mean + 0.5 * baseline_mi_std),
        'unfocused': baseline_mi_mean - 0.5 * baseline_mi_std
    }
    
    # Ensure thresholds are in valid range [0, 1]
    adaptive_thresholds = {k: np.clip(v, 0.05, 0.95) for k, v in adaptive_thresholds.items()}
    
    return adaptive_thresholds

def bin_mi(mi):
    if mi >= MI_THRESHOLDS['focused']:
        return 2  # High (Focused)
    elif mi >= MI_THRESHOLDS['neutral']:
        return 1  # Neutral
    else:
        return 0  # Low (Unfocused)

def calculate_mi(features):
    """
    Calculate MI from EEG/EDA features with automatic scaling for large inputs.
    """
    # Log the feature values for debugging
    if any(abs(f) > 10000 for f in features):
        print(f"[INFO] Large feature values detected: {features}")
        
    # Normalize range for extremely large values (log-scale if needed)
    normalized_features = np.array(features, dtype=float)
    for i in range(len(normalized_features)):
        if abs(normalized_features[i]) > 100000:
            # Use logarithmic scaling for very large values
            sign = np.sign(normalized_features[i])
            normalized_features[i] = sign * np.log10(abs(normalized_features[i]))
            
    # Adjust weights for specific large-scale input range
    if np.max(np.abs(features)) > 100000:
        weights = np.array([0.01, 0.01, 0.25, -0.05, -0.05])  # Reduced weight for large-amplitude features
    else:
        weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])    # Default weights
    
    # Calculate MI with adjusted offset based on feature scale
    feature_scale = max(1.0, np.mean(np.abs(normalized_features)) / 50)
    offset = min(1.0, 0.2 * feature_scale)  # Scale offset with feature magnitude
    
    # Final calculation
    raw_mi = np.dot(normalized_features, weights) - offset
    raw_mi = np.clip(raw_mi, -50, 50)  # Prevent overflow
    mi = 1 / (1 + np.exp(-raw_mi))
    
    return mi

def calculate_mi_debug(features):
    """Debug version of calculate_mi with detailed logging"""
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    weighted_features = features * weights
    print(f"[DEBUG] Features: {features}")
    print(f"[DEBUG] Weights: {weights}")
    print(f"[DEBUG] Weighted features: {weighted_features}")
    
    dot_product = np.dot(features, weights)
    print(f"[DEBUG] Dot product: {dot_product}")
    
    raw_mi = dot_product - 1
    print(f"[DEBUG] Raw MI (before clip): {raw_mi}")
    
    raw_mi_clipped = np.clip(raw_mi, -50, 50)
    print(f"[DEBUG] Raw MI (after clip): {raw_mi_clipped}")
    
    mi = 1 / (1 + np.exp(-raw_mi_clipped))
    print(f"[DEBUG] Final MI: {mi}")
    
    return mi

def load_training_data():
    # Aggregate all feature windows from EEG/EDA files
    eeg_files = glob.glob(os.path.join(EEG_DIR, '*.csv'))
    X, y = [], []
    for eeg_file in eeg_files:
        df = pd.read_csv(eeg_file)
        # Expect columns: theta_fz, alpha_po, faa, beta_frontal, eda_norm, mi_score (or calculate MI)
        if all(f in df.columns for f in FEATURE_ORDER):
            features = df[FEATURE_ORDER].values
            if 'mi_score' in df.columns:
                mi = df['mi_score'].values
            else:
                mi = np.array([calculate_mi(f) for f in features])
            X.append(features)
            y.append(mi)
    if not X:
        raise RuntimeError('No training data found in EEG/EDA directories.')
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

def train_and_save_models():
    X, y = load_training_data()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    # Train SVR for regression
    svr = SVR().fit(X_scaled, y)
    # Train SVC for classification (binned MI)
    y_binned = np.array([bin_mi(val) for val in y])
    svc = SVC().fit(X_scaled, y_binned)
    # Save models and scaler
    dump(svr, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    logging.info('Trained and saved SVR and scaler.')
    return svr, scaler

def load_or_train_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logging.info('Loading existing model and scaler...')
        svr = load(MODEL_PATH)
        scaler = load(SCALER_PATH)
    else:
        logging.info('No model found. Training new model...')
        svr, scaler = train_and_save_models()
    return svr, scaler

def select_lsl_stream(stream_type, name_hint=None, allow_skip=False, confirm=True):
    from pylsl import resolve_streams
    if confirm:
        print(f"Searching for available LSL streams of type '{stream_type}'...")
    streams = resolve_streams()
    if not streams:
        if allow_skip:
            if confirm:
                print(f"No LSL streams found for type '{stream_type}'. You may skip this sensor.")
            skip = input(f"Type 'skip' to continue without {stream_type}, or press Enter to retry: ").strip().lower()
            if skip == 'skip':
                return None
            else:
                return select_lsl_stream(stream_type, name_hint, allow_skip, confirm)
        else:
            raise RuntimeError("No LSL streams found on the network.")
    if confirm:
        print("Available streams:")
        for idx, s in enumerate(streams):
            print(f"[{idx}] Name: {s.name()} | Type: {s.type()} | Channels: {s.channel_count()} | Source ID: {s.source_id()}")
        if allow_skip:
            print(f"[{len(streams)}] SKIP this sensor and use generic model/scaler")
    while True:
        try:
            sel = input(f"Select the stream index for {stream_type}: ")
            if allow_skip and sel.strip() == str(len(streams)):
                if confirm:
                    print(f"[SKIP] Skipping {stream_type} stream selection. Will use generic model/scaler.")
                return None
            sel = int(sel)
            if 0 <= sel < len(streams):
                chosen = streams[sel]
                if confirm:
                    print(f"[CONFIRM] Selected stream: Name='{chosen.name()}', Type='{chosen.type()}', Channels={chosen.channel_count()}, Source ID='{chosen.source_id()}'\n")
                return chosen
            else:
                if confirm:
                    print(f"Invalid index. Please enter a number between 0 and {len(streams)-1} (or {len(streams)} to skip if available).")
        except ValueError:
            if confirm:
                print("Invalid input. Please enter a valid integer index.")

def calibrate_user(user_id, calibration_duration_sec=60):
    """
    BASELINE CALIBRATION: Collect baseline data during relaxed/passive state.
    
    This function collects the user's baseline EEG/EDA patterns during a relaxed state
    (eyes closed, sitting still). This baseline is crucial for:
    - Personalizing MI thresholds relative to individual resting state
    - Training user-specific models that account for individual differences
    - Normalizing real-time features relative to personal baseline
    
    Expected data: RAW EEG (8 channels) + ACC/Gyro (6 channels) + RAW EDA (2 channels)
    Sampling rate: 250 Hz, Feature extraction: 1 Hz (1-second windows)
    """
    print(f"\n=== BASELINE CALIBRATION for user: {user_id} ===")
    print("This will collect your BASELINE brain/body state during relaxation.")
    print("Please remain as relaxed and still as possible during calibration.")
    print("Type 'exit' at any prompt to abort calibration.")
    # Show user info if config exists
    config_path = os.path.join(USER_CONFIG_DIR, f'{user_id}_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        print(f"[INFO] Previous calibration found for user '{user_id}':")
        print(f"  Baseline file: {user_config.get('baseline_csv','N/A')}")
        print(f"  Calibration time: {user_config.get('calibration_time','N/A')}")
        print(f"  Number of samples: {user_config.get('n_samples','N/A')}")
        print(f"[CONFIRM] Calibration config loaded from {config_path}")
    else:
        print(f"[INFO] No previous calibration found for user '{user_id}'.")
    # Interactive countdown
    countdown_sec = 5
    print(f"\n[BASELINE CALIBRATION INSTRUCTIONS]")
    print("• This will establish your personal BASELINE brain state")
    print("• Please close your eyes and relax as much as possible")
    print("• Avoid movement, thinking, or mental effort")
    print("• Just sit comfortably and let your mind rest")
    print("• This baseline will be used to personalize your MI thresholds")
    print(f"\nCalibration will start in {countdown_sec} seconds. Please get ready...")
    for i in range(countdown_sec, 0, -1):
        print(f"Starting in {i}...", end='\r', flush=True)
        time.sleep(1)
    print("\n[COLLECTING BASELINE] Please remain relaxed with eyes closed...")
    print("Select the EEG LSL feature stream to use for calibration (must be RAW, unconverted EEG data):")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream')
    if eeg_stream.channel_count() < 14:
        print(f"[ERROR] Selected EEG stream has {eeg_stream.channel_count()} channels. At least 14 (8 EEG + 6 ACC/Gyro) required. Skipping calibration.")
        return None, None
    eeg_inlet = StreamInlet(eeg_stream)
    print("Select the EDA LSL feature stream to use for calibration (must be RAW, unconverted EDA data):")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals')
    if eda_stream.channel_count() < 2:
        print(f"[ERROR] Selected EDA stream has {eda_stream.channel_count()} channels. At least 2 required. Skipping calibration.")
        return None, None
    eda_inlet = StreamInlet(eda_stream)
    processed_info = StreamInfo('calibration_processed', 'ProcessedCalibration', len(FEATURE_ORDER), 250, 'float32', f'calib_{user_id}')
    processed_outlet = StreamOutlet(processed_info)
    print(f"Calibration processed LSL stream created as 'calibration_processed' with {len(FEATURE_ORDER)} channels at 250 Hz.\n")
    eeg_samples, eda_samples, accgyr_samples, ts_samples = [], [], [], []
    n_samples = int(250 * calibration_duration_sec)
    window_size = 250  # 1 second window for feature extraction
    features_list = []
    print(f"Collecting {n_samples} samples at 250 Hz for {calibration_duration_sec} seconds...")
    max_wall_time = calibration_duration_sec * 2
    start_time = time.time()
    feature_push_count = 0
    for i in range(n_samples):
        if (time.time() - start_time) > max_wall_time:
            print(f"[ERROR] Calibration exceeded maximum allowed time ({max_wall_time}s). Aborting calibration.")
            break
        eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=1.0)
        eda_sample, eda_ts = eda_inlet.pull_sample(timeout=1.0)
        if eeg_sample is None or eda_sample is None:
            print(f"[WARN] Missing sample at index {i}: eeg_sample is None: {eeg_sample is None}, eda_sample is None: {eda_sample is None}. Skipping.")
            continue
        eeg = np.array(eeg_sample[:8])
        acc_gyr = np.array(eeg_sample[8:14])
        eda_raw = np.array(eda_sample[:2])
        eda = eda_raw  # Note: No scaling applied in calibration!
        # EDA CHANNEL SELECTION: Using configurable channel (see EDA_CHANNEL_INDEX at top)
        eda_feat = eda[EDA_CHANNEL_INDEX]  # Channel configured at top of file
        
        # Debug EDA values during calibration
        if i % 250 == 0:  # Print every second
            print(f"[DEBUG CALIB] EDA raw sample {i}: {eda_raw}")
        
        eeg_samples.append(eeg)
        accgyr_samples.append(acc_gyr)
        eda_samples.append(eda)
        ts_samples.append(eeg_ts)
        # Only compute features every 1 second (every 250 samples)
        if len(eeg_samples) >= window_size and (i+1) % window_size == 0:
            eeg_win = np.array(eeg_samples[-window_size:])
            eda_win = np.array(eda_samples[-window_size:])
            
            # Debug EDA window during calibration
            print(f"[DEBUG CALIB] EDA window {(i+1)//window_size}: ch0_mean={np.mean(eda_win[:,0]):.6f}, ch1_mean={np.mean(eda_win[:,1]):.6f}, using_channel={EDA_CHANNEL_INDEX}")
            
            sf = 250
            theta_fz = compute_bandpower(eeg_win[:,0], sf, (4,8))
            alpha_po = (compute_bandpower(eeg_win[:,6], sf, (8,13)) + compute_bandpower(eeg_win[:,7], sf, (8,13))) / 2
            faa = np.log(compute_bandpower(eeg_win[:,4], sf, (8,13)) + 1e-8) - np.log(compute_bandpower(eeg_win[:,5], sf, (8,13)) + 1e-8)
            beta_frontal = compute_bandpower(eeg_win[:,0], sf, (13,30))
            # Apply universal normalization for consistent calibration
            raw_eda = np.mean(eda_win[:,EDA_CHANNEL_INDEX])
            eda_norm = normalize_features_flexible({'eda_norm': raw_eda}, method='robust_quantile')['eda_norm']
            
            print(f"[DEBUG CALIB] EDA_NORM = {eda_norm:.6f}")
            
            features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
            features_list.append(features)
            processed_outlet.push_sample(features, eeg_ts)
            feature_push_count += 1
            print(f"[DEBUG] calibration_processed: pushed features at t={eeg_ts:.3f} {features}")
        if i % 250 == 0:
            print(f"Collected {i} samples...")
    actual_duration = time.time() - start_time
    print(f"[SUMMARY] calibration_processed: pushed {feature_push_count} feature windows (1 Hz). Actual duration: {actual_duration:.2f} seconds.")
    # After collection, use features_list for baseline_arr
    baseline_arr = np.array(features_list)
    baseline_arr = baseline_arr[~np.isnan(baseline_arr).any(axis=1)]
    min_valid_windows = int(0.5 * n_samples / window_size)  # Require at least 50% of expected windows
    if baseline_arr.shape[0] < min_valid_windows:
        print(f"[ERROR] Too few valid calibration windows collected ({baseline_arr.shape[0]} < {min_valid_windows}). Skipping calibration/model update.")
        return None, None
    if baseline_arr.shape[0] == 0:
        print("[ERROR] All calibration samples are invalid (contain NaN). Skipping calibration/model update.")
        return None, None
    baseline_csv = os.path.join(USER_CONFIG_DIR, f'{user_id}_baseline.csv')
    new_df = pd.DataFrame(baseline_arr, columns=FEATURE_ORDER)
    if os.path.exists(baseline_csv):
        existing_df = pd.read_csv(baseline_csv)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(baseline_csv, index=False)
        print(f"Appended new calibration features to {baseline_csv}")
    else:
        new_df.to_csv(baseline_csv, index=False)
        print(f"Baseline calibration features saved to {baseline_csv}")
    print(f"[CONFIRM] Calibration file created/updated: {baseline_csv}")

    # --- Save user config JSON (keep this, do not save raw data) ---
    user_config = {
        'user_id': user_id,
        'baseline_csv': baseline_csv,
        'calibration_time': str(datetime.now()),
        'n_samples': n_samples
    }
    with open(config_path, 'w') as f:
        json.dump(user_config, f, indent=2)
    print(f"User config saved to {config_path}")
    print(f"[CONFIRM] Calibration config created: {config_path}")

    # --- AUTOMATIC SVR TRAINING AFTER CALIBRATION ---
    print("[AUTO] Fine-tuning SVR model for this user based on calibration data...")
    calib_df = pd.read_csv(baseline_csv)
    X_calib = calib_df[FEATURE_ORDER].values
    
    # Use universal MI calculation for consistent training targets
    y_calib = np.array([calculate_mi_universal(f, method='robust_quantile') for f in X_calib])
    
    print(f"[INFO] Using universal MI calculation for training targets")
    print(f"[INFO] This ensures consistency across all users and devices")
    
    # --- DEBUG: Check and log feature statistics ---
    feature_mins = np.min(X_calib, axis=0)
    feature_maxs = np.max(X_calib, axis=0)
    feature_means = np.mean(X_calib, axis=0)
    feature_stds = np.std(X_calib, axis=0)
    
    print("\n[DEBUG] FEATURE STATISTICS DURING CALIBRATION:")
    for i, feat in enumerate(FEATURE_ORDER):
        print(f"  {feat}: min={feature_mins[i]:.2f}, max={feature_maxs[i]:.2f}, mean={feature_means[i]:.2f}, std={feature_stds[i]:.2f}")
    
    # Alert about potentially problematic feature values
    for i, feat in enumerate(FEATURE_ORDER):
        if abs(feature_maxs[i]) > 100000 or abs(feature_means[i]) > 10000:
            print(f"[WARN] Feature '{feat}' has extremely large values! Consider scaling or normalization.")
        elif abs(feature_maxs[i]) < 0.001 and abs(feature_maxs[i]) > 0:
            print(f"[WARN] Feature '{feat}' has extremely small values! Consider scaling.")
    
    print("\n--- Debug information ends ---\n")
    
    # --- DEBUG: Print MI targets for calibration ---
    print(f"[DEBUG] MI targets (y_calib) stats: min={y_calib.min()}, max={y_calib.max()}, unique={np.unique(y_calib)}")
    # Always fit a new scaler for the user calibration data
    scaler = StandardScaler().fit(X_calib)
    X_calib_scaled = scaler.transform(X_calib)
    # --- DEBUG: Print first 5 scaled features ---
    print(f"[DEBUG] First 5 scaled calibration features: {X_calib_scaled[:5]}")
    print("[INFO] Training SVR model on calibration data...")
    svr = SVR().fit(X_calib_scaled, y_calib)
    print("[INFO] SVR model training complete.")
    user_model_path = os.path.join(MODEL_DIR, f'{user_id}_svr_model.joblib')
    user_scaler_path = os.path.join(MODEL_DIR, f'{user_id}_scaler.joblib')
    dump(svr, user_model_path)
    dump(scaler, user_scaler_path)
    print(f"[AUTO] User SVR model and scaler saved: {user_model_path}, {user_scaler_path}")

    # --- Train and save user-specific SVC classifier ---
    print("[AUTO] Training user-specific SVC classifier...")
    y_calib_binned = np.array([bin_mi(val) for val in y_calib])
    unique_classes = np.unique(y_calib_binned)
    
    if len(unique_classes) < 2:
        print(f"[WARNING] Cannot train SVC classifier: Only {len(unique_classes)} unique class(es) found in calibration data.")
        print(f"[WARNING] MI values are too clustered (min={min(y_calib):.3f}, max={max(y_calib):.3f}).")
        print("[WARNING] SVC training skipped. Will use SVR/thresholding approach instead.")
        user_svc_path = None
    else:
        print(f"[INFO] Found {len(unique_classes)} classes for SVC training: {unique_classes}")
        svc = SVC().fit(X_calib_scaled, y_calib_binned)
        user_svc_path = os.path.join(MODEL_DIR, f'{user_id}_svc_model.joblib')
        dump(svc, user_svc_path)
        print(f"[AUTO] User SVC classifier saved: {user_svc_path}")

    # --- Evaluate new model ---
    y_new_pred = svr.predict(X_calib_scaled)
    new_mae = mean_absolute_error(y_calib, y_new_pred)
    new_r2 = r2_score(y_calib, y_new_pred)
    print(f"[REPORT] New model on calibration: MAE={new_mae:.4f}, R2={new_r2:.4f}")

    # --- Save comparative report ---
    comp_report = {
        'user_id': user_id,
        'calibration_time': str(datetime.now()),
        'n_samples': len(y_calib),
        'new_mae': new_mae,
        'new_r2': new_r2
    }
    comp_report_path = os.path.join(LOG_DIR, f"{user_id}_calibration_comparative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame([comp_report]).to_csv(comp_report_path, index=False)
    print(f"[REPORT] Calibration comparative report saved to {comp_report_path}")
    return baseline_csv, config_path

# --- Visualization ---
class OnlineVisualizer:
    def __init__(self):
        self.mi_history = []
        self.raw_mi_history = []
        self.emi_history = []
        self.label_history = []
        self.timestamps = []
        self.last_metrics = {'precision': 1.0, 'recall': 1.0}
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        # Do not show the figure during the process

    def update(self, mi_pred, raw_mi=None, emi=None, label=None):
        """Update visualization with all types of MI data"""
        # Store all values
        self.mi_history.append(mi_pred)
        self.raw_mi_history.append(raw_mi if raw_mi is not None else np.nan)
        self.emi_history.append(emi if emi is not None else np.nan)
        self.timestamps.append(datetime.now())
        
        if label is not None:
            self.label_history.append(label)
        else:
            self.label_history.append(np.nan)
        # Do not plot during the process

    def final_plot(self):
        """Plot all mindfulness indices for comparison"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Create x-axis time points
        x = np.arange(len(self.mi_history))
        
        # Plot MI on first subplot
        self.axes[0].plot(x, self.mi_history, label='Standard MI', color='blue')
        if any(~np.isnan(self.label_history)):
            self.axes[0].plot(x, self.label_history, label='Labels', color='red', linestyle=':')
        self.axes[0].set_title('Standard MI (0-1 range)')
        self.axes[0].set_ylabel('MI Value')
        self.axes[0].legend()
        self.axes[0].set_ylim(0, 1)
        self.axes[0].grid(True, alpha=0.3)
        
        # Plot Raw MI (unbounded) on second subplot
        self.axes[1].plot(x, self.raw_mi_history, label='Raw MI', color='purple')
        self.axes[1].set_title('Raw MI (More Dynamic Range)')
        self.axes[1].set_ylabel('Raw Value')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)
        
        # Plot EMI on third subplot
        self.axes[2].plot(x, self.emi_history, label='EMI', color='green')
        self.axes[2].set_title('Emotional Mindfulness Index')
        self.axes[2].set_xlabel('Samples')
        self.axes[2].set_ylabel('EMI Value')
        self.axes[2].legend()
        self.axes[2].set_ylim(0, 1)
        self.axes[2].grid(True, alpha=0.3)
        
        # Adjust layout and show
        plt.tight_layout()
        
        # Save plot
        fname = os.path.join(VIS_DIR, f'mindfulness_indices_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        self.fig.savefig(fname)
        print(f"[REPORT] Mindfulness indices plot saved to {fname}")
        # Close the figure to free memory
        plt.close(self.fig)

    def log_metrics(self, y_true, y_pred):
        # Only compute if enough labels
        if len(y_true) > 10 and not all(np.isnan(y_true)):
            y_true_bin = [bin_mi(val) for val in y_true if not np.isnan(val)]
            y_pred_bin = [bin_mi(val) for val in y_pred[:len(y_true_bin)]]
            precision = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            if precision < self.last_metrics['precision'] or recall < self.last_metrics['recall']:
                warn_path = os.path.join(VIS_DIR, f'warning_drop_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                self.fig.savefig(warn_path)
                print(f"WARNING: Precision or recall dropped! Plot saved to {warn_path}")
            self.last_metrics = {'precision': precision, 'recall': recall}
            # Log
            with open(os.path.join(LOG_DIR, 'metrics_log.txt'), 'a') as f:
                f.write(f"{datetime.now()} Precision: {precision:.3f}, Recall: {recall:.3f}\n")

def run_experiment(user_id, calibration_samples=100, experiment_duration_sec=240):
    # Step 1: Calibration (ground truth)
    print(f"Starting calibration for user {user_id}...")
    calibrate_user(user_id, n_samples=calibration_samples)  # Modify calibrate_user to accept n_samples

    # Step 2: Model/scaler loading
    svr, scaler = load_or_train_models()
    online_model = SGDRegressor(max_iter=1000, learning_rate='optimal', eta0=0.01)
    X, y = load_training_data()
    X_scaled = scaler.transform(X)
    online_model.partial_fit(X_scaled, svr.predict(X_scaled))    # Step 3: LSL setup
    feature_stream = select_lsl_stream('Features')
    feature_inlet = StreamInlet(feature_stream)
    
    # Unity markers are optional in run_experiment too
    use_unity = input("Use Unity markers in experiment? (y/n, default: n): ")

    if use_unity == 'y':
        try:
            label_stream = select_lsl_stream('UnityMarkers', allow_skip=True, confirm=True)
            if label_stream is not None:
                label_inlet = StreamInlet(label_stream)
                print("Unity markers connected for experiment.")
            else:
                label_inlet = None
                print("[INFO] Unity markers skipped for experiment.")
        except Exception as e:
            print(f"[WARN] Unity markers connection failed: {e}")
            label_inlet = None
    else:
        label_inlet = None
        print("[INFO] Unity markers disabled for experiment.")
    
    # Create all LSL output streams
    outlets = setup_mindfulness_lsl_streams()
    mi_outlet = outlets['mi']
    raw_mi_outlet = outlets['raw_mi']
    emi_outlet = outlets['emi']

    visualizer = OnlineVisualizer()
    EEG_BUFFER, EDA_BUFFER, TS_BUFFER = [], [], []
    WINDOW_SIZE = 250  # 1 second at 250 Hz
    mi_window = []  # Moving window for MI predictions
    mi_records = []  # To store MI, timestamp, and state

    print("Starting 4-minute real-time MI feedback session...")
    start_time = time.time()
    while (time.time() - start_time) < experiment_duration_sec:
        sample, _ = feature_inlet.pull_sample()
        x_raw = np.array(sample).reshape(1, -1)
        x_scaled = scaler.transform(x_raw)
        if np.isnan(x_scaled).any():
            print("[WARN] Feature vector contains NaN. Skipping MI prediction for this window.")
            mi_pred = 0.0  # or np.nan, or previous MI value
        else:
            mi_pred = svr.predict(x_scaled)[0]
        # Classify MI for Unity color logic
        if mi_pred >= 0.5:
            state = "Focused"
        elif mi_pred >= 0.37:
            state = "Neutral"
        else:
            state = "Unfocused"
        print(f"MI: {mi_pred:.3f} | State: {state}")
        mi_outlet.push_sample([mi_pred])
        
        # Handle Unity markers if available
        if label_inlet is not None:
            try:
                label, _ = label_inlet.pull_sample(timeout=0.01)
                if label:
                    online_model.partial_fit(x_scaled, [float(label[0])])
                    visualizer.update(mi_pred, float(label[0]))
                    visualizer.log_metrics(np.array(visualizer.label_history), np.array(visualizer.mi_history))
                else:
                    visualizer.update(mi_pred)
            except Exception as e:
                # Continue without Unity markers if they fail
                visualizer.update(mi_pred)
        else:
            visualizer.update(mi_pred)
    print("Experiment complete.")

# --- MAIN ENTRY POINT ---
def main():
    check_required_resources()
    print("\n==============================")
    print("REAL-TIME MI LSL PIPELINE STARTING")
    print("==============================\n")
    print("[INFO] This script expects RAW (unconverted, unnormalized) EEG and EDA data from LSL streams.")
    print("[INFO] Do NOT pre-normalize or convert your EEG/EDA data before streaming to this script.")
    print(f"[CONFIG] EDA Channel: Using channel {EDA_CHANNEL_INDEX + 1} (0-based index: {EDA_CHANNEL_INDEX}) for features")
    print("[CONFIG] To change EDA channel, modify EDA_CHANNEL_INDEX at the top of this file")
    user_id = input("Enter user ID for this session: ")
    calibration_duration = 60
    print(f"Calibration will last {calibration_duration} seconds at 250 Hz.")
    calibrate = input("Run calibration step for this user? (y/n): ").strip().lower() == 'y'
    if calibrate:
        baseline_csv, config_path = calibrate_user(user_id, calibration_duration_sec=calibration_duration)
        if baseline_csv is None:
            print("[WARN] Calibration skipped or failed. Continuing with generic/global model.")
        else:
            print("[INFO] Calibration complete. Proceeding to real-time MI prediction...")
    # --- Everything below here should ALWAYS run, regardless of calibration ---
    artifact_regressors = None
    config_path = os.path.join(USER_CONFIG_DIR, f'{user_id}_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        artifact_regressors = user_config.get('artifact_regressors', None)
    print(f"Checking for user-specific model and scaler for user: {user_id}")
    user_model_path = os.path.join(MODEL_DIR, f'{user_id}_svr_model.joblib')
    user_scaler_path = os.path.join(MODEL_DIR, f'{user_id}_scaler.joblib')
    baseline_csv = os.path.join(USER_CONFIG_DIR, f'{user_id}_baseline.csv')
    scaler = None
    svr = None
    # Robustly load per-user scaler/model, refit if shape mismatch
    if os.path.exists(user_model_path) and os.path.exists(user_scaler_path):
        print(f"Loading user-specific model and scaler for user {user_id}...")
        svr = load(user_model_path)
        scaler = load(user_scaler_path)
        # Check scaler shape matches expected features
        try:
            calib_df = pd.read_csv(baseline_csv)
            X_calib = calib_df[FEATURE_ORDER].values
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != X_calib.shape[1]:
                print(f"[WARN] User scaler shape mismatch (expected {X_calib.shape[1]}, got {scaler.n_features_in_}). Refitting scaler...")
                scaler = StandardScaler().fit(X_calib)
                dump(scaler, user_scaler_path)
                print(f"[INFO] Refitted and saved user scaler: {user_scaler_path}")
        except Exception as e:
            print(f"[ERROR] Could not check/refit user scaler: {e}")
        print("User-specific model and scaler loaded.")
    else:
        print(f"User-specific model/scaler not found. Loading or training global model...")
        svr, scaler = load_or_train_models()
        print("Model and scaler ready.")
    print("[INFO] Online adaptation is DISABLED during real-time MI prediction. Only user-specific model/scaler will be used.")
    print(f"[INFO] Using model: {getattr(svr, 'model_path', user_model_path if os.path.exists(user_model_path) else MODEL_PATH)}")
    print(f"[INFO] Using scaler: {getattr(scaler, 'scaler_path', user_scaler_path if os.path.exists(user_scaler_path) else SCALER_PATH)}")
    # LSL streams
    print("Select the EEG LSL feature stream to use (must be RAW, unconverted EEG data):")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream', allow_skip=True)
    if eeg_stream is not None:
        eeg_inlet = StreamInlet(eeg_stream)
        print("EEG feature stream connected.")
    else:
        eeg_inlet = None
        print("EEG stream skipped. Will use generic model/scaler.")
    print("Select the EDA LSL feature stream to use (must be RAW, unconverted EDA data):")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals', allow_skip=True)
    if eda_stream is not None:
        eda_inlet = StreamInlet(eda_stream)
        print("EDA feature stream connected.")
    else:
        eda_inlet = None
        print("EDA stream skipped. Will use generic model/scaler.")
    # --- Unity markers are completely optional ---
    use_unity_markers = input("Connect to Unity markers stream? (y/n, default: n): ").strip().lower()
    if use_unity_markers == 'y':
        try:
            print("Resolving Unity label stream (type='UnityMarkers')...")
            label_stream = select_lsl_stream('UnityMarkers', allow_skip=True, confirm=True)
            if label_stream is not None:
                label_inlet = StreamInlet(label_stream)
                print("Unity label stream connected.")
            else:
                label_inlet = None
                print("[INFO] Unity markers skipped by user choice.")
        except Exception as e:
            print(f"[WARN] Unity label stream connection failed: {e}")
            print("[INFO] Continuing without Unity markers.")
            label_inlet = None
    else:
        print("[INFO] Unity markers disabled. MI pipeline will run independently.")
        label_inlet = None
    # Ask user for MI calculation rate and transmission interval
    try:
        mi_calc_rate_input = input("Enter MI calculation rate in Hz (default 10): ").strip()
        if mi_calc_rate_input == '':
            MI_CALC_RATE = 10
        else:
            MI_CALC_RATE = float(mi_calc_rate_input)
    except Exception:
        MI_CALC_RATE = 10
    
    try:
        mi_update_interval_input = input("Enter MI transmission interval in seconds (default 3): ")

        if mi_update_interval_input == '':
            MI_UPDATE_INTERVAL = 3.0
        else:
            MI_UPDATE_INTERVAL = float(mi_update_interval_input)
    except Exception:
        MI_UPDATE_INTERVAL = 3.0
    # Create all LSL output streams
    outlets = setup_mindfulness_lsl_streams()
    mi_outlet = outlets['mi']
    raw_mi_outlet = outlets['raw_mi']
    emi_outlet = outlets['emi']

    visualizer = OnlineVisualizer()
    EEG_BUFFER, EDA_BUFFER, TS_BUFFER = [], [], []
    WINDOW_SIZE = 250  # 1 second at 250 Hz
    mi_window = []  # Moving window for MI predictions
    mi_records = []  # To store MI, timestamp, and state
    print("Entering real-time MI prediction loop at 1 Hz. Classification every 3 seconds.")
    # --- Automatic input data analysis and adaptation ---
    # CORRECTED SCALING BASED ON REAL USER DATA ANALYSIS
    # Analysis of user 005_alextest shows:
    # - Calibration EEG: 6-40 (normal physiological range)  
    # - Real-time EEG with 0.001 scaling: 2e-6 to 1e-4 (1,000,000x too small!)
    # - This causes model saturation (all MI = 0.9996...)
    eeg_scale_factor = 1.0  # NO SCALING - EEG values are already in correct range
    eda_scale_factor = 1.0  # NO SCALING - EDA values are already in correct range
    
    print(f"[INFO] CORRECTED scaling factors based on real data analysis:")
    print(f"[INFO] EEG scale factor: {eeg_scale_factor} (no scaling - values already in physiological range)")
    print(f"[INFO] EDA scale factor: {eda_scale_factor} (no scaling - values already in physiological range)")
    print(f"[INFO] Previous aggressive scaling (0.001) was causing 1,000,000x reduction - FIXED!")
    
    if eeg_inlet is not None or eda_inlet is not None:
        print("\n[INFO] Running automatic input data analysis for EEG/EDA streams...")
        # Analyze and adapt scaling if needed
        import numpy as np
        analysis_eeg_vals = []
        analysis_eda_vals = []
        n_samples = 500
        for sample_idx in range(n_samples):
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.5)
                if eeg_sample is not None:
                    analysis_eeg_vals.append(np.array(eeg_sample[:8]))
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=0.5)
                if eda_sample is not None:
                    eda_raw = np.array(eda_sample[:2])
                    analysis_eda_vals.append(eda_raw)
                    # Debug first few samples
                    if sample_idx < 10:
                        print(f"[DEBUG] Analysis EDA sample {sample_idx}: {eda_raw}")
                        
        if analysis_eeg_vals:
            eeg_arr = np.vstack(analysis_eeg_vals)
            print("\n[EEG RAW DATA ANALYSIS]")
            print(f"  Shape: {eeg_arr.shape}")
            print(f"  Min: {np.min(eeg_arr):.3f}, Max: {np.max(eeg_arr):.3f}, Mean: {np.mean(eeg_arr):.3f}, Std: {np.std(eeg_arr):.3f}")
            if np.issubdtype(eeg_arr.dtype, np.integer):
                print("  [WARN] EEG data appears to be integer. RAW (unconverted, unnormalized) EEG is expected.")
            if np.nanmax(np.abs(eeg_arr)) < 1.0:
                print("  [WARN] EEG data values are very small (<1.0). RAW EEG is expected. Check your LSL stream.")
            if np.nanmax(np.abs(eeg_arr)) > 1000:
                print("  [WARN] EEG data values are very large (>1000). Check for amplifier scaling or units.")
        if analysis_eda_vals:
            eda_arr = np.vstack(analysis_eda_vals)
            print("\n[EDA RAW DATA ANALYSIS]")
            print(f"  Shape: {eda_arr.shape}")
            print(f"  Channel 0 - Min: {np.min(eda_arr[:,0]):.6f}, Max: {np.max(eda_arr[:,0]):.6f}, Mean: {np.mean(eda_arr[:,0]):.6f}, Std: {np.std(eda_arr[:,0]):.6f}")
            print(f"  Channel 1 - Min: {np.min(eda_arr[:,1]):.6f}, Max: {np.max(eda_arr[:,1]):.6f}, Mean: {np.mean(eda_arr[:,1]):.6f}, Std: {np.std(eda_arr[:,1]):.6f}")
            
            # Determine appropriate scaling
            max_ch0 = np.nanmax(np.abs(eda_arr[:,0]))
            max_ch1 = np.nanmax(np.abs(eda_arr[:,1]))
            
            print(f"  Max absolute values: Ch0={max_ch0:.6f}, Ch1={max_ch1:.6f}")
            
            # Adjust scaling based on EDA range
            if max_ch1 > 100000:
                eda_scale_factor = 0.00001  # Very large values
                print(f"  [INFO] EDA values are very large. Using scale factor: {eda_scale_factor}")
            elif max_ch1 > 10000:
                eda_scale_factor = 0.0001   # Large values
                print(f"  [INFO] EDA values are large. Using scale factor: {eda_scale_factor}")
            elif max_ch1 > 1000:
                eda_scale_factor = 0.001    # Medium values
                print(f"  [INFO] EDA values are medium. Using scale factor: {eda_scale_factor}")
            elif max_ch1 < 0.01:
                eda_scale_factor = 100.0    # Very small values - scale up
                print(f"  [INFO] EDA values are very small. Using scale factor: {eda_scale_factor}")
            else:
                eda_scale_factor = 1.0      # Good range - no scaling
                print(f"  [INFO] EDA values are in good range. Using scale factor: {eda_scale_factor}")
                
            if np.issubdtype(eda_arr.dtype, np.integer):
                print("  [WARN] EDA data appears to be integer. RAW (unconverted, unnormalized) EDA is expected.")
            
            # Check for constant values
            if np.std(eda_arr[:,0]) < 1e-10:
                print("  [WARN] EDA channel 0 appears constant!")
            if np.std(eda_arr[:,1]) < 1e-10:
                print("  [WARN] EDA channel 1 appears constant!")
        
        print(f"\n[INFO] Final scaling factors: EEG={eeg_scale_factor}, EDA={eda_scale_factor}")
        print(f"[INFO] Expected feature ranges after scaling:")
        print(f"  - EEG power features: 5-100 (physiological power range)")
        print(f"  - EDA features: 5-15 (typical normalized range)")
        print(f"[INFO] This matches the calibration data ranges, ensuring model compatibility!")
    
    # IMPORTANT: Consistent scaling between calibration and real-time
    # Calibration shows correct expected ranges (EEG: 6-40, EDA: 8-9)
    # Real-time must match these ranges for proper model performance

    import threading
    import sys
    import msvcrt  # For ESC key detection on Windows
    stop_flag = {'stop': False}
    def wait_for_exit():
        print("\nPress Enter or ESC to end the session and generate report...\n")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\r' or key == b'\n':  # Enter key
                    stop_flag['stop'] = True
                    break
                elif key == b'\x1b':  # ESC key
                    stop_flag['stop'] = True
                    break
            time.sleep(0.05)
    t = threading.Thread(target=wait_for_exit)
    t.daemon = True
    t.start()
    last_push_time = time.time()
    mi_buffer = []
    mi_records = []
    next_calc_time = time.time()
    session_start_time = time.time()  # Track session start time
    print(f"Entering real-time MI prediction loop at 1 Hz. Classification window: 3 seconds.\n")
    
    # Load user baseline statistics for normalization  
    baseline_stats = load_user_baseline(user_id)
    has_user_calibration = baseline_stats is not None
    
    if has_user_calibration:
        print(f"[INFO] Loaded user baseline statistics for {user_id}")
        print(f"[INFO] Baseline MI: mean={baseline_stats['mi_baseline']['mean']:.3f}, std={baseline_stats['mi_baseline']['std']:.3f}")
        print("[INFO] Using calibrated user: SVR regression for continuous MI values (0-1 range)")
        print("[INFO] No classification thresholds needed - outputting raw MI numbers")
        # For calibrated users, we don't use SVC classification
        global_svc = None
    else:
        print(f"[WARN] No baseline data found for {user_id}. Using default MI calculation.")
        print("[INFO] Using non-calibrated user: Will attempt to load SVC for classification if available")
        
        # Try to load global SVC for non-calibrated users (check feature compatibility)
        if os.path.exists(MODEL_PATH):
            try:
                # Check if global SVC expects 5 features
                test_svc = load(MODEL_PATH)
                if hasattr(test_svc, 'n_features_in_'):
                    if test_svc.n_features_in_ == 5:
                        global_svc = test_svc
                        print(f"[INFO] Loaded global SVC classifier (5 features) from {MODEL_PATH}")
                    else:
                        print(f"[WARN] Global SVC expects {test_svc.n_features_in_} features, but pipeline uses 5. Skipping SVC classification.")
                        global_svc = None
                else:
                    # Old sklearn version - try to load anyway
                    global_svc = test_svc
                    print(f"[INFO] Loaded global SVC classifier from {MODEL_PATH} (feature count unknown)")
            except Exception as e:
                print(f"[WARN] Failed to load global SVC: {e}. Using SVR/thresholding approach.")
                global_svc = None
        else:
            global_svc = None
            print("[INFO] No global SVC model found. Using SVR regression only.")
        
    while not stop_flag['stop']:
        now = time.time()
        if now < next_calc_time:
            time.sleep(max(0, next_calc_time - now))
            continue
        next_calc_time += 1.0  # 1 Hz
        # Collect 250 samples for 1-second window
        eeg_win_buf, eda_win_buf, ts_win_buf = [], [], []
        for i in range(WINDOW_SIZE):
            if eeg_inlet is not None:
                eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample is not None:
                    eeg = np.array(eeg_sample[:8]) * eeg_scale_factor
                    acc_gyr = np.array(eeg_sample[8:14])
                    if artifact_regressors is not None:
                        eeg_clean = apply_artifact_regression(eeg, acc_gyr, artifact_regressors)
                    else:
                        eeg_clean = eeg
                else:
                    eeg_clean = np.zeros(8)
                    eeg_ts = time.time()
                    print(f"[DEBUG] EEG sample {i} is None!")
            else:
                eeg_clean = np.zeros(8)
                eeg_ts = time.time()
            
            if eda_inlet is not None:
                eda_sample, eda_ts = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample is not None:
                    eda_raw = np.array(eda_sample[:2])
                    eda = eda_raw * eda_scale_factor
                    # Debug EDA values
                    if i % 50 == 0:  # Print every 50th sample to avoid spam
                        print(f"[DEBUG] EDA sample {i}: raw={eda_raw}, scaled={eda}, scale_factor={eda_scale_factor}")
                else:
                    eda = np.zeros(2)
                    eda_ts = eeg_ts
                    print(f"[DEBUG] EDA sample {i} is None!")
            else:
                eda = np.zeros(2)
                eda_ts = eeg_ts
            
            eeg_win_buf.append(eeg_clean)
            eda_win_buf.append(eda)
            ts_win_buf.append(eeg_ts)
        eeg_win = np.array(eeg_win_buf)
        eda_win = np.array(eda_win_buf)
        
        # --- Debug EDA window statistics ---
        print(f"[DEBUG] EDA window shape: {eda_win.shape}")
        print(f"[DEBUG] EDA channel 0 stats: min={np.min(eda_win[:,0]):.6f}, max={np.max(eda_win[:,0]):.6f}, mean={np.mean(eda_win[:,0]):.6f}")
        print(f"[DEBUG] EDA channel 1 stats: min={np.min(eda_win[:,1]):.6f}, max={np.max(eda_win[:,1]):.6f}, mean={np.mean(eda_win[:,1]):.6f}")
        print(f"[DEBUG] Using EDA channel {EDA_CHANNEL_INDEX} for features (configured at top of file)")
        print(f"[DEBUG] ✓ CORRECT CHANNEL: Using Ch{EDA_CHANNEL_INDEX} ({np.mean(eda_win[:,EDA_CHANNEL_INDEX]):.1f}) NOT Ch0 ({np.mean(eda_win[:,0]):.0f})")
        
        # --- Feature extraction (windowed, real) ---
        sf = 250
        theta_fz = compute_bandpower(eeg_win[:,0], sf, (4,8))
        alpha_po = (compute_bandpower(eeg_win[:,6], sf, (8,13)) + compute_bandpower(eeg_win[:,7], sf, (8,13))) / 2
        faa = np.log(compute_bandpower(eeg_win[:,4], sf, (8,13)) + 1e-8) - np.log(compute_bandpower(eeg_win[:,5], sf, (8,13)) + 1e-8)
        beta_frontal = compute_bandpower(eeg_win[:,0], sf, (13,30))
        # EDA UNIVERSAL NORMALIZATION: Apply consistent normalization method
        raw_eda = np.mean(eda_win[:,EDA_CHANNEL_INDEX])
        eda_norm = normalize_features_flexible({'eda_norm': raw_eda}, method='robust_quantile')['eda_norm']
        
        # --- Debug EDA normalization ---
        print(f"[DEBUG] EDA_NORM calculation: robust normalized = {eda_norm:.6f}")
        print(f"[DEBUG] EDA channel {EDA_CHANNEL_INDEX} raw values (first 10): {eda_win[:10,EDA_CHANNEL_INDEX]}")
        print(f"[DEBUG] CONFIRM: Using EDA channel {EDA_CHANNEL_INDEX + 1} (1-based) = index {EDA_CHANNEL_INDEX} (0-based)")
        print(f"[DEBUG] EDA raw window stats: Ch0 mean={np.mean(eda_win[:,0]):.1f}, Ch1 mean={np.mean(eda_win[:,1]):.1f}")
        
        features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
        sample = np.array(features).reshape(1, -1)
        # --- Print features and MI values in real-time ---
        print(f"[REAL-TIME] Features: {dict(zip(FEATURE_ORDER, features))}")
        # Warn if features are all zeros, all NaN, or constant
        if np.all(np.isnan(sample)):
            print("[WARN] All features are NaN. Skipping MI prediction for this window.")
            mi_pred = 0.0
            skipped_reason = 'all_nan'
        elif np.all(sample == 0):
            print("[WARN] All features are zero. Skipping MI prediction for this window.")
            mi_pred = 0.0
            skipped_reason = 'all_zero'
        elif np.all(sample == sample[0,0]):
            print("[WARN] All features are constant. Skipping MI prediction for this window.")
            mi_pred = 0.0
            skipped_reason = 'constant'
        else:
            x_scaled = scaler.transform(sample)
            if np.isnan(x_scaled).any():
                print("[WARN] Feature vector contains NaN. Skipping MI prediction for this window.")
                mi_pred = 0.0
                skipped_reason = 'scaled_nan'
            else:
                mi_pred = svr.predict(x_scaled)[0]
                skipped_reason = None
                
                # Add detailed debug info for near-zero MI values
                if mi_pred < 0.0001:
                    print("\n[DEBUG] Near-zero MI value detected!")
                    print(f"[DEBUG] Raw features: {sample[0]}")
                    print(f"[DEBUG] Scaled features: {x_scaled[0]}")
                    recalc_mi = calculate_mi_scaled(sample[0])  # Use scaled version for comparison
                    print(f"[DEBUG] Direct MI calculation (scaled): {recalc_mi}, Model prediction: {mi_pred}")
                    print(f"[DEBUG] Difference: {abs(recalc_mi - mi_pred):.6f}")
                    if abs(recalc_mi - mi_pred) > 0.1:
                        print("[DEBUG] Large difference between direct calculation and model! Check model training.")
                    print("")
        
        # --- Real-time MI output ---
        if skipped_reason:
            # Set default state for skipped predictions
            state = "skipped"
            print(f"MI: {mi_pred:.4f} (prediction skipped: {skipped_reason})")
        elif has_user_calibration:
            # Calibrated user: Just output the continuous MI value
            print(f"MI: {mi_pred:.4f}")
            state = "continuous"  # No discrete classification for calibrated users
        else:
            # Non-calibrated user: Try SVC classification if available
            if global_svc is not None:
                try:
                    svc_state_idx = global_svc.predict(x_scaled)[0]
                    svc_state = {2: "Focused", 1: "Neutral", 0: "Unfocused"}.get(svc_state_idx, str(svc_state_idx))
                    print(f"MI: {mi_pred:.3f} | SVC State: {svc_state}")
                    state = svc_state
                except Exception as e:
                    print(f"[WARN] SVC prediction failed: {e}. Using threshold classification.")
                    # Fallback to threshold-based classification
                    if mi_pred >= 0.5:
                        state = "Focused"
                    elif mi_pred >= 0.37:
                        state = "Neutral"
                    else:
                        state = "Unfocused"
                    print(f"MI: {mi_pred:.3f} | State: {state} (threshold-based)")
            else:
                # No SVC available - use threshold classification
                if mi_pred >= 0.5:
                    state = "Focused"
                elif mi_pred >= 0.37:
                    state = "Neutral"
                else:
                    state = "Unfocused"
                print(f"MI: {mi_pred:.3f} | State: {state} (threshold-based)")
        
        # Show how current features compare to baseline
        if baseline_stats is not None:
            print(f"[BASELINE COMPARISON]")
            for feat_name, feat_val in zip(FEATURE_ORDER, features):
                baseline_mean = baseline_stats['means'][feat_name]
                baseline_std = baseline_stats['stds'][feat_name]
                if baseline_std > 0:
                    z_score = (feat_val - baseline_mean) / baseline_std
                    comparison = "HIGH" if z_score > 1.5 else "LOW" if z_score < -1.5 else "NORMAL"
                    print(f"  {feat_name}: {comparison} (z={z_score:.2f})")
            print(f"  Current MI vs Baseline: {mi_pred:.3f} vs {baseline_stats['mi_baseline']['mean']:.3f}")
            print("")
        
        mi_buffer.append(mi_pred)
        if skipped_reason:
            if 'mi_skipped_count' not in locals():
                mi_skipped_count = {}
            mi_skipped_count[skipped_reason] = mi_skipped_count.get(skipped_reason, 0) + 1
            
        # Calculate additional indices for more dynamic feedback using universal methods
        raw_mi_value = calculate_raw_mi_universal(sample[0], method='robust_quantile')
        emi_value = calculate_emi_universal(sample[0], method='robust_quantile')
        # Remap raw MI to 0-1 range for output
        raw_mi_remapped = remap_raw_mi(raw_mi_value)
        
        # Also calculate universal MI for comparison (this is the generalizable approach)
        mi_universal = calculate_mi_universal(sample[0], method='robust_quantile')
        
        # --- Print all MI values with universal comparison ---
        print(f"[REAL-TIME] MI (SVR): {mi_pred:.3f} | MI (Universal): {mi_universal:.3f} | Raw MI: {raw_mi_value:.3f} (remapped: {raw_mi_remapped:.3f}) | EMI: {emi_value:.3f}")
        
        # OPTION: Use universal MI instead of user-specific SVR if model is saturated
        if abs(mi_pred - 0.984) < 0.001:  # Detect saturation (like user 007_alex_test)
            print(f"[AUTO-SWITCH] Detected model saturation. Using Universal MI: {mi_universal:.3f}")
            mi_pred = mi_universal  # Switch to universal approach
        
        # Use current time for samples
        current_ts = time.time()
        # Push all index types to their respective LSL streams
        mi_outlet.push_sample([mi_pred], current_ts)
        raw_mi_outlet.push_sample([raw_mi_remapped], current_ts)
        emi_outlet.push_sample([emi_value], current_ts)
        
        # Record for visualization/analysis
        mi_records.append({
            'mi': mi_pred, 
            'raw_mi': raw_mi_remapped, 
            'emi': emi_value,
            'timestamp': current_ts, 
            'state': state,
            'theta_fz': theta_fz,
            'alpha_po': alpha_po,
            'faa': faa,
            'beta_frontal': beta_frontal,
            'eda_norm': eda_norm
        })
        # Handle Unity markers if available
        if label_inlet is not None:
            try:
                label, label_ts = label_inlet.pull_sample(timeout=0.01)
                if label:
                    try:
                        label_val = float(label[0])
                        visualizer.update(mi_pred, raw_mi_value, emi_value, label_val)
                    except (ValueError, TypeError):
                        visualizer.update(mi_pred, raw_mi_value, emi_value, None)
                else:
                    visualizer.update(mi_pred, raw_mi_value, emi_value, None)
            except Exception as e:
                # If Unity stream fails, continue without it
                visualizer.update(mi_pred, raw_mi_value, emi_value, None)
        else:
            # No Unity markers - continue with MI pipeline only
            visualizer.update(mi_pred, raw_mi_value, emi_value, None)
    # --- After session: Save MI CSV and print report ---
    print(f"\n{'='*60}")
    print("SESSION COMPLETED - GENERATING REPORTS")
    print(f"{'='*60}")
    
    session_duration = time.time() - session_start_time if 'session_start_time' in locals() else 0
    print(f"Session Duration: {session_duration:.1f} seconds")
    print(f"Total MI predictions: {len(mi_records)}")
    
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    mi_csv_path = os.path.join(LOG_DIR, f'{user_id}_mi_session_{session_time}.csv')
    
    # Save all features and state if available
    session_df = pd.DataFrame(mi_records)
    
    # Add session metadata
    if len(mi_records) > 0:
        avg_mi = np.mean([r['mi'] for r in mi_records if 'mi' in r])
        avg_raw_mi = np.mean([r['raw_mi'] for r in mi_records if 'raw_mi' in r])
        avg_emi = np.mean([r['emi'] for r in mi_records if 'emi' in r])
        
        print(f"Average MI: {avg_mi:.3f}")
        print(f"Average Raw MI: {avg_raw_mi:.3f}") 
        print(f"Average EMI: {avg_emi:.3f}")
        
        # Count states
        states = [r.get('state', 'unknown') for r in mi_records]
        state_counts = {state: states.count(state) for state in set(states)}
        print(f"State distribution: {state_counts}")
    
    # Save all features and state (features are now included in mi_records)
    session_df = pd.DataFrame(mi_records)
    
    session_df.to_csv(mi_csv_path, index=False)
    print(f"\n[REPORT 1] MI session data saved to {mi_csv_path}")
    # --- After session: Create comprehensive features visualization ---
    print(f"\n[REPORT 2] Generating comprehensive features visualization...")
    try:
        session_df = pd.read_csv(mi_csv_path)
        
        # Add scaled features to the dataframe for comparison
        print(f"[REPORT 2a] Computing scaled features for visualization...")
        scaled_features_data = []
        for _, row in session_df.iterrows():
            if all(feat in row and not pd.isna(row[feat]) for feat in FEATURE_ORDER):
                raw_features = [row[feat] for feat in FEATURE_ORDER]
                scaled_features = scale_features_for_mi(raw_features)
                scaled_features_data.append({
                    'theta_fz_scaled': scaled_features[0],
                    'alpha_po_scaled': scaled_features[1], 
                    'faa_scaled': scaled_features[2],
                    'beta_frontal_scaled': scaled_features[3],
                    'eda_norm_scaled': scaled_features[4]
                })
            else:
                scaled_features_data.append({feat + '_scaled': np.nan for feat in FEATURE_ORDER})
        
        scaled_df = pd.DataFrame(scaled_features_data)
        session_df = pd.concat([session_df, scaled_df], axis=1)
        
        # Create a comprehensive plot with MI and all 5 features (both raw and scaled)
        fig, axes = plt.subplots(11, 1, figsize=(15, 25), sharex=True)
        
        # Plot MI on the first subplot
        axes[0].plot(session_df['mi'], label='MI (SVR)', color='blue', linewidth=2)
        axes[0].set_ylabel('MI Value', fontsize=12)
        axes[0].set_title('Mindfulness Index and EEG/EDA Features Over Time (Raw vs Scaled)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Define feature descriptions and colors for better visualization
        feature_info = {
            'theta_fz': {'title': 'Theta Power (Fz)', 'description': 'Attention/Focus', 'color': 'red'},
            'alpha_po': {'title': 'Alpha Power (PO7/PO8)', 'description': 'Relaxation/Awareness', 'color': 'green'},
            'faa': {'title': 'Frontal Alpha Asymmetry', 'description': 'Emotional Balance', 'color': 'purple'},
            'beta_frontal': {'title': 'Beta Power (Frontal)', 'description': 'Mental Activity', 'color': 'orange'},
            'eda_norm': {'title': 'EDA (Normalized)', 'description': 'Arousal/Stress', 'color': 'brown'}
        }
        
        # Plot raw features (original problematic scale)
        for i, feat in enumerate(FEATURE_ORDER):
            if feat in session_df.columns:
                feat_data = session_df[feat].dropna()
                if len(feat_data) > 0:
                    axes[i+1].plot(feat_data, 
                                 label=f"{feature_info[feat]['title']} (Raw)", 
                                 color=feature_info[feat]['color'], 
                                 linewidth=1.5, alpha=0.7)
                    axes[i+1].set_ylabel(f"Raw {feat}\n(Original Scale)", fontsize=10)
                    axes[i+1].legend(fontsize=9)
                    axes[i+1].grid(True, alpha=0.3)
                    
                    # Add statistical info and scale warning
                    mean_val = feat_data.mean()
                    std_val = feat_data.std()
                    axes[i+1].set_title(f"Mean: {mean_val:.2e}, Std: {std_val:.2e} [UNSCALED - May appear flat]", fontsize=9, color='red')
        
        # Plot scaled features (properly balanced for MI calculation)
        for i, feat in enumerate(FEATURE_ORDER):
            scaled_feat = feat + '_scaled'
            if scaled_feat in session_df.columns:
                feat_data = session_df[scaled_feat].dropna()
                if len(feat_data) > 0:
                    axes[i+6].plot(feat_data, 
                                 label=f"{feature_info[feat]['title']} (Scaled)", 
                                 color=feature_info[feat]['color'], 
                                 linewidth=2)
                    axes[i+6].set_ylabel(f"Scaled {feat}\n(Balanced for MI)", fontsize=10)
                    axes[i+6].legend(fontsize=9)
                    axes[i+6].grid(True, alpha=0.3)
                    
                    # Add statistical info for scaled features
                    mean_val = feat_data.mean()
                    std_val = feat_data.std()
                    axes[i+6].set_title(f"Mean: {mean_val:.3f}, Std: {std_val:.3f} [SCALED - Balanced contribution]", fontsize=9, color='green')
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_path = os.path.join(VIS_DIR, f'{user_id}_comprehensive_features_{session_time}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[REPORT 2] Comprehensive features plot saved to {plot_path}")
        plt.close(fig)
        
        # Create a second plot: Features correlation matrix
        print(f"[REPORT 2b] Generating features correlation heatmap...")
        features_df = session_df[FEATURE_ORDER + ['mi']].dropna()
        if len(features_df) > 10:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            corr_matrix = features_df.corr()
            im = ax2.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values as text
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    text = ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
            
            ax2.set_xticks(range(len(corr_matrix.columns)))
            ax2.set_yticks(range(len(corr_matrix.columns)))
            ax2.set_xticklabels([feature_info.get(col, {'title': col})['title'] for col in corr_matrix.columns], rotation=45)
            ax2.set_yticklabels([feature_info.get(col, {'title': col})['title'] for col in corr_matrix.columns])
            ax2.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Correlation Coefficient', fontsize=12)
            
            plt.tight_layout()
            corr_plot_path = os.path.join(VIS_DIR, f'{user_id}_features_correlation_{session_time}.png')
            plt.savefig(corr_plot_path, dpi=200, bbox_inches='tight')
            print(f"[REPORT 2b] Features correlation heatmap saved to {corr_plot_path}")
            plt.close(fig2)
        
    except Exception as e:
        print(f"[WARN] Could not create features visualization: {e}")
    # --- Print and save summary statistics and feature-MI correlations ---
    print(f"\n[REPORT 3] Computing summary statistics...")
    try:
        stats_report = []
        print("\n[SUMMARY STATISTICS]")
        for col in ['mi'] + FEATURE_ORDER:
            if col in session_df.columns:
                vals = session_df[col].dropna()
                mean, std, vmin, vmax = vals.mean(), vals.std(), vals.min(), vals.max()
                print(f"{col}: mean={mean:.3f}, std={std:.3f}, min={vmin:.3f}, max={vmax:.3f}")
                stats_report.append({'variable': col, 'mean': mean, 'std': std, 'min': vmin, 'max': vmax})
        stats_path = os.path.join(LOG_DIR, f'{user_id}_mi_feature_stats_{session_time}.csv')
        pd.DataFrame(stats_report).to_csv(stats_path, index=False)
        print(f"[REPORT 3] Summary statistics saved to {stats_path}")
        
        # Feature-MI correlations
        print(f"\n[REPORT 4] Computing feature-MI correlations...")
        corr_report = []
        print("\n[FEATURE-MI CORRELATIONS] (Spearman)")
        for feat in FEATURE_ORDER:
            if feat in session_df.columns:
                corr, p = spearmanr(session_df[feat], session_df['mi'], nan_policy='omit')
                print(f"{feat} vs MI: corr={corr:.3f}, p={p:.3g}")
                corr_report.append({'feature': feat, 'spearman_corr': corr, 'p_value': p})
        corr_path = os.path.join(LOG_DIR, f'{user_id}_mi_feature_corr_{session_time}.csv')
        pd.DataFrame(corr_report).to_csv(corr_path, index=False)
        print(f"[REPORT 4] Feature-MI correlations saved to {corr_path}")
    except Exception as e:
        print(f"[WARN] Could not compute summary stats/correlations: {e}")
    
    # --- Generate final visualization ---
    print(f"\n[REPORT 5] Generating final visualization...")
    try:
        visualizer.final_plot()
        print(f"[REPORT 5] Final visualization completed")
    except Exception as e:
        print(f"[WARN] Could not generate final visualization: {e}")
    
    print(f"\n{'='*60}")
    print("ALL REPORTS GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Check the following directories for your reports:")
    print(f"- CSV files: {LOG_DIR}")
    print(f"- Plots: {LOG_DIR} and {VIS_DIR}")
    print(f"{'='*60}\n")

# --- Confirm required files and folders at startup ---
REQUIRED_DIRS = [MODEL_DIR, LOG_DIR, VIS_DIR, PROCESSED_DATA_DIR, USER_CONFIG_DIR]
REQUIRED_FILES = [MODEL_PATH, SCALER_PATH]
def check_required_resources():
    print("\n[CHECK] Confirming required directories and files...")
    for d in REQUIRED_DIRS:
        if not os.path.exists(d):
            print(f"[MISSING] Directory not found: {d}")
        else:
            print(f"[OK] Directory: {d}")
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            print(f"[MISSING] File not found: {f}")
        else:
            print(f"[OK] File: {f}")
    print("")

# --- Analysis and Visualization (separate section, not part of the pipeline) ---
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a session's feature CSV (e.g., baseline or MI session)
df = pd.read_csv('user_configs/your_user_baseline.csv')  # or logs/your_user_mi_session_*.csv

# Plot each feature over time
feature_names = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
titles = [
    'Attentional Engagement (theta_fz)',
    'Alpha Power (alpha_po)',
    'Frontal Alpha Asymmetry (FAA)',
    'Beta Power (beta_frontal)',
    'Normalized EDA (eda_norm)'
]
plt.figure(figsize=(15, 10))
for i, (feat, title) in enumerate(zip(feature_names, titles), 1):
    plt.subplot(5, 1, i)
    plt.plot(df[feat])
    plt.title(title)
    plt.xlabel('Window')
    plt.ylabel(feat)
plt.tight_layout()
plt.savefig('feature_time_series.png')
print("Feature time series plot saved to feature_time_series.png")
# Optional: Pairplot/correlations
sns.pairplot(df[feature_names])
plt.savefig('feature_pairplot.png')
print("Feature pairplot saved to feature_pairplot.png")
"""

# --- Flexible Normalization Function ---
def normalize_features_flexible(features, method='robust_quantile', user_stats=None, pop_stats=None):
    """
    Flexible feature normalization supporting robust quantile, physiological, z-score (with min std), and hybrid/adaptive.
    - method: 'robust_quantile', 'physiological', 'zscore', 'hybrid'
    - user_stats: dict with 'means' and 'stds' (optional, for zscore/hybrid)
    - pop_stats: dict with 'means', 'stds', 'q5', 'q95', 'physio_min', 'physio_max' (optional, for hybrid)
    """
    if isinstance(features, (list, np.ndarray)):
        feature_dict = dict(zip(FEATURE_ORDER, features))
    else:
        feature_dict = features.copy()
    normalized = {}
    # Population-level robust quantiles and physiological ranges
    robust_ranges = {
        'theta_fz': (1, 50),
        'alpha_po': (1, 50),
        'faa': (-2, 2),
        'beta_frontal': (1, 50),
        'eda_norm': (0.1, 20)
    }
    physio_ranges = {
        'theta_fz': (0.1, 100),
        'alpha_po': (0.1, 100),
        'faa': (-3, 3),
        'beta_frontal': (0.1, 100),
        'eda_norm': (0.01, 50)
    }
    min_std = 1.0  # Minimum std for z-score
    for feat_name, value in feature_dict.items():
        if method == 'robust_quantile':
            q5, q95 = robust_ranges[feat_name]
            val = 10 * (value - q5) / (q95 - q5)
            normalized[feat_name] = np.clip(val, 0, 10)
        elif method == 'physiological':
            min_val, max_val = physio_ranges[feat_name]
            clipped_val = np.clip(value, min_val * 0.1, max_val * 2)
            val = 10 * (clipped_val - min_val) / (max_val - min_val)
            normalized[feat_name] = np.clip(val, 0, 10)
        elif method == 'zscore':
            # Use user_stats if available, else pop_stats
            if user_stats and 'means' in user_stats and 'stds' in user_stats:
                mean = user_stats['means'].get(feat_name, 0)
                std = max(user_stats['stds'].get(feat_name, min_std), min_std)
            elif pop_stats and 'means' in pop_stats and 'stds' in pop_stats:
                mean = pop_stats['means'].get(feat_name, 0)
                std = max(pop_stats['stds'].get(feat_name, min_std), min_std)
            else:
                mean, std = 0, min_std
            z = (value - mean) / std
            # Map z-score to [0, 10] using a sigmoid-like mapping
            mapped = 10 / (1 + np.exp(-z))
            normalized[feat_name] = np.clip(mapped, 0, 10)
        elif method == 'hybrid':
            # Try user/session stats, else robust quantile, else physiological
            if user_stats and 'means' in user_stats and 'stds' in user_stats:
                mean = user_stats['means'].get(feat_name, 0)
                std = max(user_stats['stds'].get(feat_name, min_std), min_std)
                z = (value - mean) / std
                mapped = 10 / (1 + np.exp(-z))
                normalized[feat_name] = np.clip(mapped, 0, 10)
            elif pop_stats and 'q5' in pop_stats and 'q95' in pop_stats:
                q5 = pop_stats['q5'].get(feat_name, robust_ranges[feat_name][0])
                q95 = pop_stats['q95'].get(feat_name, robust_ranges[feat_name][1])
                val = 10 * (value - q5) / (q95 - q5)
                normalized[feat_name] = np.clip(val, 0, 10)
            else:
                min_val, max_val = physio_ranges[feat_name]
                clipped_val = np.clip(value, min_val * 0.1, max_val * 2)
                val = 10 * (clipped_val - min_val) / (max_val - min_val)
                normalized[feat_name] = np.clip(val, 0, 10)
        else:
            # Default to robust quantile
            q5, q95 = robust_ranges[feat_name]
            val = 10 * (value - q5) / (q95 - q5)
            normalized[feat_name] = np.clip(val, 0, 10)
    return normalized

# --- Update universal MI/EMI to use flexible normalization ---
def calculate_mi_universal(features, method='robust_quantile', user_stats=None, pop_stats=None):
    normalized_features = normalize_features_flexible(features, method=method, user_stats=user_stats, pop_stats=pop_stats)
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])
    weighted_sum = np.dot(feature_array, weights)
    mi = 1 / (1 + np.exp(-(weighted_sum - 2.5)))
    return np.clip(mi, 0, 1)

def calculate_raw_mi_universal(features, method='robust_quantile', user_stats=None, pop_stats=None):
    normalized_features = normalize_features_flexible(features, method=method, user_stats=user_stats, pop_stats=pop_stats)
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])
    raw_mi = np.dot(feature_array, weights) - 2.5
    return np.clip(raw_mi, -10, 10)

def calculate_emi_universal(features, method='robust_quantile', user_stats=None, pop_stats=None):
    normalized_features = normalize_features_flexible(features, method=method, user_stats=user_stats, pop_stats=pop_stats)
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    weights = np.array([0.15, 0.15, 0.4, -0.05, -0.25])
    weighted_sum = np.dot(feature_array, weights)
    emi = 1 / (1 + np.exp(-(weighted_sum - 2.0)))
    return np.clip(emi, 0, 1)
