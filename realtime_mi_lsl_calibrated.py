#!/usr/bin/env python3
"""
REAL-TIME MINDFULNESS INDEX (MI) LSL PIPELINE - CALIBRATED VERSION
==================================================================

This version implements a dual-calibration approach with:
1. RELAXED calibration period (20 seconds, eyes closed)
2. FOCUSED calibration period (30 seconds, eyes open, attention task)
3. Adaptive per-user thresholding based on calibration baselines
4. Peak suppression and artifact rejection during data acquisition
5. Clear user instructions and logging throughout the process

Key Features:
- Dual calibration periods for personalized MI mapping
- Robust artifact suppression (median filtering, outlier rejection)
- Adaptive normalization based on user's low/high baselines
- Real-time feedback with calibrated thresholds
- Comprehensive logging and session reports

Author: EEG-SVM Team
Date: June 2025
Version: 2.0 (Calibrated)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os
import json
import threading
if os.name == 'nt':  # Windows
    import msvcrt

from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream
from scipy.signal import welch, butter, filtfilt
from scipy.stats import spearmanr, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load

# ===========================
# CONFIGURATION & CONSTANTS
# ===========================

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'user_configs')
CALIBRATION_DIR = os.path.join(BASE_DIR, 'calibrations')

# Create directories if they don't exist
for directory in [MODEL_DIR, LOG_DIR, VIS_DIR, PROCESSED_DATA_DIR, USER_CONFIG_DIR, CALIBRATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model and scaler paths
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_mi_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Feature configuration
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
EDA_CHANNEL_INDEX = 1  # Use channel 1 (0-based indexing) for EDA features

# Calibration configuration
RELAXED_DURATION = 20  # seconds for relaxed calibration
FOCUSED_DURATION = 30  # seconds for focused calibration
SAMPLING_RATE = 250    # Hz

# Peak suppression configuration
PEAK_SUPPRESSION_CONFIG = {
    'median_window': 5,        # Median filter window size
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
    'max_change_rate': 0.5,    # Maximum allowed change rate between samples
    'smoothing_window': 3      # Moving average window for smoothing
}

# ===========================
# SIGNAL PROCESSING FUNCTIONS
# ===========================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_bandpower(data, sf, band, nperseg=None):
    """Compute average power in frequency band using Welch's method"""
    if nperseg is None:
        nperseg = min(len(data), sf * 2)
    
    freqs, psd = welch(data, sf, nperseg=nperseg)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])

def suppress_peaks_and_artifacts(data, config=None):
    """
    Advanced peak suppression and artifact rejection for EEG/EDA data
    
    Args:
        data: 1D array of signal data
        config: Dictionary with suppression parameters
    
    Returns:
        cleaned_data: Array with peaks suppressed and artifacts removed
        artifact_mask: Boolean mask indicating artifact locations
    """
    if config is None:
        config = PEAK_SUPPRESSION_CONFIG
    
    data = np.array(data, dtype=float)
    cleaned_data = data.copy()
    artifact_mask = np.zeros(len(data), dtype=bool)
    
    # Step 1: Median filtering to suppress sharp peaks
    from scipy.signal import medfilt
    if len(data) >= config['median_window']:
        cleaned_data = medfilt(cleaned_data, kernel_size=config['median_window'])
    
    # Step 2: Outlier detection using robust z-score
    if len(data) > 10:
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))  # Median Absolute Deviation
        if mad > 0:
            robust_zscore = 0.6745 * (data - median_val) / mad
            outliers = np.abs(robust_zscore) > config['outlier_threshold']
            artifact_mask |= outliers
            
            # Replace outliers with median values
            cleaned_data[outliers] = median_val
    
    # Step 3: Rate of change limiting
    if len(data) > 1:
        diff = np.diff(cleaned_data)
        std_diff = np.std(diff) if len(diff) > 1 else 1.0
        max_allowed_change = config['max_change_rate'] * std_diff
        
        for i in range(1, len(cleaned_data)):
            if abs(cleaned_data[i] - cleaned_data[i-1]) > max_allowed_change:
                # Limit the change rate
                sign = 1 if cleaned_data[i] > cleaned_data[i-1] else -1
                cleaned_data[i] = cleaned_data[i-1] + sign * max_allowed_change
                artifact_mask[i] = True
    
    # Step 4: Final smoothing with moving average
    window_size = config['smoothing_window']
    if len(cleaned_data) >= window_size:
        smoothed = np.convolve(cleaned_data, np.ones(window_size)/window_size, mode='same')
        # Preserve edges
        smoothed[:window_size//2] = cleaned_data[:window_size//2]
        smoothed[-(window_size//2):] = cleaned_data[-(window_size//2):]
        cleaned_data = smoothed
    
    return cleaned_data, artifact_mask

def extract_features_robust(eeg_window, eda_window, sf=250):
    """
    Extract features with robust preprocessing and peak suppression
    
    Args:
        eeg_window: EEG data array (samples x channels)
        eda_window: EDA data array (samples x channels)
        sf: Sampling frequency
    
    Returns:
        features: Dictionary with extracted features
        quality_metrics: Quality assessment of the extracted features
    """
    features = {}
    quality_metrics = {}
    
    # EEG feature extraction with artifact suppression
    try:
        # Apply peak suppression to each EEG channel
        eeg_clean = np.zeros_like(eeg_window)
        eeg_artifacts = np.zeros(eeg_window.shape, dtype=bool)
        
        for ch in range(eeg_window.shape[1]):
            eeg_clean[:, ch], eeg_artifacts[:, ch] = suppress_peaks_and_artifacts(eeg_window[:, ch])
        
        # Calculate artifact percentage
        artifact_percentage = np.mean(eeg_artifacts) * 100
        quality_metrics['eeg_artifact_percentage'] = artifact_percentage
        
        # Extract frequency domain features from cleaned EEG
        theta_fz = compute_bandpower(eeg_clean[:, 0], sf, (4, 8))
        alpha_po7 = compute_bandpower(eeg_clean[:, 6], sf, (8, 13))
        alpha_po8 = compute_bandpower(eeg_clean[:, 7], sf, (8, 13))
        alpha_po = (alpha_po7 + alpha_po8) / 2
        
        # Frontal Alpha Asymmetry (F3 vs F4)
        alpha_f3 = compute_bandpower(eeg_clean[:, 4], sf, (8, 13))
        alpha_f4 = compute_bandpower(eeg_clean[:, 5], sf, (8, 13))
        faa = np.log(alpha_f4 + 1e-8) - np.log(alpha_f3 + 1e-8)
        
        beta_frontal = compute_bandpower(eeg_clean[:, 0], sf, (13, 30))
        
        features.update({
            'theta_fz': theta_fz,
            'alpha_po': alpha_po,
            'faa': faa,
            'beta_frontal': beta_frontal
        })
        
    except Exception as e:
        print(f"[WARN] EEG feature extraction failed: {e}")
        features.update({
            'theta_fz': 0.0,
            'alpha_po': 0.0,
            'faa': 0.0,
            'beta_frontal': 0.0
        })
        quality_metrics['eeg_artifact_percentage'] = 100.0
    
    # EDA feature extraction with artifact suppression
    try:
        eda_channel = eda_window[:, EDA_CHANNEL_INDEX]
        eda_clean, eda_artifacts = suppress_peaks_and_artifacts(eda_channel)
        
        # Calculate EDA quality metrics
        eda_artifact_percentage = np.mean(eda_artifacts) * 100
        quality_metrics['eda_artifact_percentage'] = eda_artifact_percentage
        quality_metrics['eda_snr'] = np.mean(eda_clean) / (np.std(eda_clean) + 1e-8)
        
        # Robust EDA normalization
        eda_raw = np.mean(eda_clean)
        eda_norm = normalize_eda_robust(eda_raw)
        
        features['eda_norm'] = eda_norm
        
    except Exception as e:
        print(f"[WARN] EDA feature extraction failed: {e}")
        features['eda_norm'] = 0.0
        quality_metrics['eda_artifact_percentage'] = 100.0
        quality_metrics['eda_snr'] = 0.0
    
    return features, quality_metrics

def normalize_eda_robust(eda_raw):
    """Robust EDA normalization using population statistics"""
    # Updated robust ranges based on real data analysis
    eda_min, eda_max = 2.0, 12.0
    
    # Apply robust clipping and normalization
    eda_clipped = np.clip(eda_raw, eda_min * 0.5, eda_max * 1.5)
    eda_norm = 10 * (eda_clipped - eda_min) / (eda_max - eda_min)
    
    return np.clip(eda_norm, 0, 10)

# ===========================
# CALIBRATION SYSTEM
# ===========================

class CalibrationManager:
    """Manages dual-phase calibration and adaptive thresholding"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.relaxed_baseline = None
        self.focused_baseline = None
        self.adaptive_thresholds = None
        self.calibration_quality = {}
        
    def run_dual_calibration(self, eeg_inlet, eda_inlet):
        """
        Run complete dual-phase calibration
        
        Returns:
            success: Boolean indicating calibration success
            calibration_data: Dictionary with baseline data and thresholds
        """
        print(f"\n{'='*60}")
        print("DUAL-PHASE CALIBRATION STARTING")
        print(f"{'='*60}")
        print(f"User: {self.user_id}")
        print(f"Phase 1: Relaxed state ({RELAXED_DURATION}s)")
        print(f"Phase 2: Focused state ({FOCUSED_DURATION}s)")
        print(f"{'='*60}\n")
        
        # Phase 1: Relaxed calibration
        print("🧘 PHASE 1: RELAXED CALIBRATION")
        print("="*40)
        print("INSTRUCTIONS:")
        print("- Close your eyes")
        print("- Sit comfortably and relax")
        print("- Breathe naturally")
        print("- Let your mind wander freely")
        print("- Duration: 20 seconds")
        print("\nPress ENTER when ready to start...")
        input()
        
        relaxed_success, self.relaxed_baseline = self._run_calibration_phase(
            "RELAXED", RELAXED_DURATION, eeg_inlet, eda_inlet
        )
        
        if not relaxed_success:
            print("[ERROR] Relaxed calibration failed!")
            return False, None
            
        # Short break between phases
        print("\n" + "="*40)
        print("Take a 10-second break...")
        for i in range(10, 0, -1):
            print(f"Break: {i} seconds remaining", end="\r")
            time.sleep(1)
        print(" " * 30, end="\r")  # Clear the line
        
        # Phase 2: Focused calibration
        print("\n🎯 PHASE 2: FOCUSED CALIBRATION")
        print("="*40)
        print("INSTRUCTIONS:")
        print("- Keep your eyes open")
        print("- Focus on a fixed point ahead")
        print("- Count backwards from 100 by 7s (100, 93, 86...)")
        print("- Maintain concentration and alertness")
        print("- Duration: 30 seconds")
        print("\nPress ENTER when ready to start...")
        input()
        
        focused_success, self.focused_baseline = self._run_calibration_phase(
            "FOCUSED", FOCUSED_DURATION, eeg_inlet, eda_inlet
        )
        
        if not focused_success:
            print("[ERROR] Focused calibration failed!")
            return False, None
        
        # Calculate adaptive thresholds
        self._calculate_adaptive_thresholds()
        
        # Save calibration data
        calibration_data = self._save_calibration_data()
        
        # Print calibration summary
        self._print_calibration_summary()
        
        return True, calibration_data
    
    def _run_calibration_phase(self, phase_name, duration, eeg_inlet, eda_inlet):
        """Run a single calibration phase"""
        print(f"\n[{phase_name}] Starting in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"[{phase_name}] {i}...", end="\r")
            time.sleep(1)
        print(f"[{phase_name}] GO!     ")
        
        features_list = []
        quality_metrics_list = []
        n_samples = duration * SAMPLING_RATE
        window_size = SAMPLING_RATE  # 1-second windows
        
        start_time = time.time()
        sample_count = 0
        window_count = 0
        
        # Collect data for the calibration duration
        eeg_buffer = []
        eda_buffer = []
        
        print(f"[{phase_name}] Collecting {duration}s of data...")
        
        while time.time() - start_time < duration:
            # Collect EEG sample
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample is not None:
                    eeg_data = np.array(eeg_sample[:8])  # First 8 channels
                    eeg_buffer.append(eeg_data)
            
            # Collect EDA sample
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample is not None:
                    eda_data = np.array(eda_sample[:2])  # First 2 channels
                    eda_buffer.append(eda_data)
            
            sample_count += 1
            
            # Process complete windows
            if len(eeg_buffer) >= window_size and len(eda_buffer) >= window_size:
                eeg_window = np.array(eeg_buffer[-window_size:])
                eda_window = np.array(eda_buffer[-window_size:])
                
                # Extract features with peak suppression
                features, quality = extract_features_robust(eeg_window, eda_window)
                
                if all(feat in features for feat in FEATURE_ORDER):
                    features_list.append([features[feat] for feat in FEATURE_ORDER])
                    quality_metrics_list.append(quality)
                    window_count += 1
                    
                    # Progress indicator
                    elapsed = time.time() - start_time
                    progress = (elapsed / duration) * 100
                    print(f"[{phase_name}] Progress: {progress:.1f}% | Windows: {window_count} | Quality: {quality.get('eeg_artifact_percentage', 0):.1f}% artifacts", end="\r")
        
        elapsed_time = time.time() - start_time
        print(f"\n[{phase_name}] Complete! Duration: {elapsed_time:.1f}s | Windows collected: {window_count}")
        
        # Validate collected data
        if window_count < duration * 0.5:  # At least 50% of expected windows
            print(f"[ERROR] Insufficient data collected for {phase_name} phase!")
            print(f"Expected: ~{duration} windows, Got: {window_count}")
            return False, None
        
        # Convert to numpy arrays and clean
        features_array = np.array(features_list)
        quality_array = quality_metrics_list
        
        # Remove windows with excessive artifacts
        good_windows = []
        for i, quality in enumerate(quality_array):
            eeg_artifacts = quality.get('eeg_artifact_percentage', 0)
            eda_artifacts = quality.get('eda_artifact_percentage', 0)
            
            if eeg_artifacts < 50 and eda_artifacts < 50:  # Less than 50% artifacts
                good_windows.append(i)
        
        if len(good_windows) < window_count * 0.3:  # At least 30% good windows
            print(f"[ERROR] Too many artifacts in {phase_name} phase!")
            return False, None
        
        features_clean = features_array[good_windows]
        
        # Calculate baseline statistics
        baseline_stats = {
            'mean': np.mean(features_clean, axis=0),
            'std': np.std(features_clean, axis=0),
            'median': np.median(features_clean, axis=0),
            'q25': np.percentile(features_clean, 25, axis=0),
            'q75': np.percentile(features_clean, 75, axis=0),
            'quality': {
                'total_windows': window_count,
                'good_windows': len(good_windows),
                'avg_eeg_artifacts': np.mean([q.get('eeg_artifact_percentage', 0) for q in quality_array]),
                'avg_eda_artifacts': np.mean([q.get('eda_artifact_percentage', 0) for q in quality_array])
            }
        }
        
        print(f"[{phase_name}] Baseline calculated | Good windows: {len(good_windows)}/{window_count}")
        return True, baseline_stats
    
    def _calculate_adaptive_thresholds(self):
        """Calculate adaptive thresholds based on calibration baselines"""
        print("\n📊 CALCULATING ADAPTIVE THRESHOLDS")
        print("="*40)
        
        # Calculate MI values for both baselines
        relaxed_features = self.relaxed_baseline['mean']
        focused_features = self.focused_baseline['mean']
        
        # Calculate MI using universal method for consistency
        relaxed_mi = self._calculate_universal_mi(relaxed_features)
        focused_mi = self._calculate_universal_mi(focused_features)
        
        print(f"Relaxed baseline MI: {relaxed_mi:.3f}")
        print(f"Focused baseline MI: {focused_mi:.3f}")
        
        # Ensure focused > relaxed (if not, swap or adjust)
        if focused_mi <= relaxed_mi:
            print("[WARN] Focused MI not higher than relaxed MI. Adjusting...")
            # Add small offset to ensure proper range
            focused_mi = relaxed_mi + 0.2
        
        # Calculate adaptive thresholds
        mi_range = focused_mi - relaxed_mi
        
        self.adaptive_thresholds = {
            'low_baseline': relaxed_mi,
            'high_baseline': focused_mi,
            'mi_range': mi_range,
            'unfocused_threshold': relaxed_mi + 0.1 * mi_range,
            'neutral_threshold': relaxed_mi + 0.4 * mi_range,
            'focused_threshold': relaxed_mi + 0.7 * mi_range,
            'highly_focused_threshold': relaxed_mi + 0.9 * mi_range
        }
        
        print(f"MI range: {mi_range:.3f}")
        print(f"Adaptive thresholds:")
        print(f"  Unfocused: < {self.adaptive_thresholds['unfocused_threshold']:.3f}")
        print(f"  Neutral: {self.adaptive_thresholds['unfocused_threshold']:.3f} - {self.adaptive_thresholds['neutral_threshold']:.3f}")
        print(f"  Focused: {self.adaptive_thresholds['neutral_threshold']:.3f} - {self.adaptive_thresholds['focused_threshold']:.3f}")
        print(f"  Highly Focused: > {self.adaptive_thresholds['focused_threshold']:.3f}")
    
    def _calculate_universal_mi(self, features):
        """Calculate MI using universal normalization method"""
        # Use robust quantile normalization
        normalized_features = self._normalize_features_universal(features)
        
        # Universal MI weights
        weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])
        weighted_sum = np.dot(normalized_features, weights)
        
        # Map to 0.1-0.9 range
        raw_score = weighted_sum / 10.0
        mi = 0.1 + 0.8 * raw_score
        return np.clip(mi, 0.1, 0.9)
    
    def _normalize_features_universal(self, features):
        """Universal feature normalization"""
        robust_ranges = {
            'theta_fz': (2, 60),
            'alpha_po': (1, 30),
            'faa': (-2.5, 2.5),
            'beta_frontal': (2, 35),
            'eda_norm': (2, 12)
        }
        
        normalized = []
        for i, feat_name in enumerate(FEATURE_ORDER):
            q5, q95 = robust_ranges[feat_name]
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        return np.array(normalized)
    
    def _save_calibration_data(self):
        """Save calibration data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed calibration data
        calibration_data = {
            'user_id': self.user_id,
            'timestamp': timestamp,
            'relaxed_baseline': {
                'mean': self.relaxed_baseline['mean'].tolist(),
                'std': self.relaxed_baseline['std'].tolist(),
                'quality': self.relaxed_baseline['quality']
            },
            'focused_baseline': {
                'mean': self.focused_baseline['mean'].tolist(),
                'std': self.focused_baseline['std'].tolist(),
                'quality': self.focused_baseline['quality']
            },
            'adaptive_thresholds': self.adaptive_thresholds,
            'feature_order': FEATURE_ORDER
        }
        
        # Save to JSON file
        calib_path = os.path.join(CALIBRATION_DIR, f'{self.user_id}_calibration_{timestamp}.json')
        with open(calib_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save user config for quick loading
        user_config_path = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_config.json')
        user_config = {
            'user_id': self.user_id,
            'latest_calibration': calib_path,
            'adaptive_thresholds': self.adaptive_thresholds,
            'calibration_timestamp': timestamp
        }
        
        with open(user_config_path, 'w') as f:
            json.dump(user_config, f, indent=2)
        
        print(f"\n💾 CALIBRATION DATA SAVED")
        print(f"Detailed data: {calib_path}")
        print(f"User config: {user_config_path}")
        
        return calibration_data
    
    def _print_calibration_summary(self):
        """Print comprehensive calibration summary"""
        print(f"\n{'='*60}")
        print("CALIBRATION SUMMARY")
        print(f"{'='*60}")
        print(f"User: {self.user_id}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📈 BASELINE COMPARISON:")
        print("Feature".ljust(15) + "Relaxed".ljust(10) + "Focused".ljust(10) + "Difference")
        print("-" * 50)
        
        for i, feat in enumerate(FEATURE_ORDER):
            relaxed_val = self.relaxed_baseline['mean'][i]
            focused_val = self.focused_baseline['mean'][i]
            diff = focused_val - relaxed_val
            print(f"{feat:<15}{relaxed_val:<10.3f}{focused_val:<10.3f}{diff:>10.3f}")
        
        print(f"\n🎯 ADAPTIVE THRESHOLDS:")
        thresholds = self.adaptive_thresholds
        print(f"Low baseline (relaxed): {thresholds['low_baseline']:.3f}")
        print(f"High baseline (focused): {thresholds['high_baseline']:.3f}")
        print(f"Personalized MI range: {thresholds['mi_range']:.3f}")
        print(f"Unfocused threshold: {thresholds['unfocused_threshold']:.3f}")
        print(f"Focused threshold: {thresholds['focused_threshold']:.3f}")
        
        print(f"\n✅ QUALITY ASSESSMENT:")
        relaxed_quality = self.relaxed_baseline['quality']
        focused_quality = self.focused_baseline['quality']
        
        print(f"Relaxed phase:")
        print(f"  - Good windows: {relaxed_quality['good_windows']}/{relaxed_quality['total_windows']}")
        print(f"  - Avg artifacts: EEG {relaxed_quality['avg_eeg_artifacts']:.1f}%, EDA {relaxed_quality['avg_eda_artifacts']:.1f}%")
        
        print(f"Focused phase:")
        print(f"  - Good windows: {focused_quality['good_windows']}/{focused_quality['total_windows']}")
        print(f"  - Avg artifacts: EEG {focused_quality['avg_eeg_artifacts']:.1f}%, EDA {focused_quality['avg_eda_artifacts']:.1f}%")
        
        print(f"\n🎉 CALIBRATION COMPLETE! Ready for real-time MI with personalized thresholds.")
        print(f"{'='*60}\n")

def load_user_calibration(user_id):
    """Load user's calibration data"""
    user_config_path = os.path.join(USER_CONFIG_DIR, f'{user_id}_config.json')
    
    if not os.path.exists(user_config_path):
        return None
    
    try:
        with open(user_config_path, 'r') as f:
            user_config = json.load(f)
        
        # Load detailed calibration data
        calib_path = user_config['latest_calibration']
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                calibration_data = json.load(f)
            return calibration_data
        else:
            return None
    except Exception as e:
        print(f"[WARN] Failed to load calibration for {user_id}: {e}")
        return None

# ===========================
# ADAPTIVE MI CALCULATION
# ===========================

def calculate_adaptive_mi(features, calibration_data=None):
    """
    Calculate MI with adaptive normalization based on user calibration
    
    Args:
        features: List or array of feature values
        calibration_data: User's calibration data (if available)
    
    Returns:
        mi: Mindfulness index (0-1 range)
        raw_mi: Raw MI value for comparison
        state: Categorical state based on adaptive thresholds
    """
    if calibration_data is None:
        # Fallback to universal MI calculation
        return calculate_universal_mi(features), 0.0, "unknown"
    
    # Get user's baselines
    relaxed_mean = np.array(calibration_data['relaxed_baseline']['mean'])
    focused_mean = np.array(calibration_data['focused_baseline']['mean'])
    thresholds = calibration_data['adaptive_thresholds']
    
    # Adaptive normalization: map current features relative to user's range
    features_array = np.array(features)
    
    # Calculate relative position between relaxed and focused baselines
    baseline_range = focused_mean - relaxed_mean
    
    # Avoid division by zero
    baseline_range = np.where(np.abs(baseline_range) < 1e-6, 1e-6, baseline_range)
    
    # Normalize features relative to user's personal range
    relative_features = (features_array - relaxed_mean) / baseline_range
    
    # Clip to reasonable range and map to 0-10 scale
    relative_features_clipped = np.clip(relative_features, -1, 2)  # Allow some extrapolation
    normalized_features = 5 + 2.5 * relative_features_clipped  # Map to ~0-10 range
    
    # Calculate MI using adapted weights
    weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])
    weighted_sum = np.dot(normalized_features, weights)
    
    # Map to user's personal MI range
    raw_score = weighted_sum / 10.0
    mi_range = thresholds['mi_range']
    low_baseline = thresholds['low_baseline']
    
    # Scale MI to user's calibrated range
    mi = low_baseline + mi_range * raw_score
    mi = np.clip(mi, 0.05, 0.95)
    
    # Calculate raw MI for comparison
    raw_mi = (weighted_sum - 5.0) * 2.0  # Center around 0, expand range
    
    # Determine categorical state based on adaptive thresholds
    if mi < thresholds['unfocused_threshold']:
        state = "unfocused"
    elif mi < thresholds['neutral_threshold']:
        state = "neutral"
    elif mi < thresholds['focused_threshold']:
        state = "focused"
    else:
        state = "highly_focused"
    
    return mi, raw_mi, state

def calculate_universal_mi(features):
    """Universal MI calculation (fallback when no calibration available)"""
    # Robust quantile normalization
    robust_ranges = {
        'theta_fz': (2, 60),
        'alpha_po': (1, 30),
        'faa': (-2.5, 2.5),
        'beta_frontal': (2, 35),
        'eda_norm': (2, 12)
    }
    
    normalized = []
    for i, feat_name in enumerate(FEATURE_ORDER):
        q5, q95 = robust_ranges[feat_name]
        val = 10 * (features[i] - q5) / (q95 - q5)
        normalized.append(np.clip(val, 0, 10))
    
    # Calculate weighted sum
    weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])
    weighted_sum = np.dot(normalized, weights)
    
    # Map to 0.1-0.9 range
    raw_score = weighted_sum / 10.0
    mi = 0.1 + 0.8 * raw_score
    return np.clip(mi, 0.1, 0.9)

# ===========================
# LSL STREAM MANAGEMENT
# ===========================

def select_lsl_stream(stream_type, name_hint=None, allow_skip=False, confirm=True):
    """Select an LSL stream with user confirmation"""
    print(f"\nSearching for {stream_type} streams...")
    
    try:
        if stream_type == 'EEG':
            streams = resolve_stream('type', 'EEG', timeout=5)
        elif stream_type == 'EDA':
            streams = resolve_stream('type', 'EDA', timeout=5)
        elif stream_type == 'UnityMarkers':
            streams = resolve_stream('type', 'UnityMarkers', timeout=5)
        else:
            streams = resolve_stream('name', stream_type, timeout=5)
        
        if not streams:
            print(f"No {stream_type} streams found.")
            if allow_skip:
                skip = input(f"Skip {stream_type} stream? (y/n): ").lower() == 'y'
                if skip:
                    return None
            return None
        
        print(f"Found {len(streams)} {stream_type} stream(s):")
        for i, stream in enumerate(streams):
            print(f"  [{i}] {stream.name()} | {stream.channel_count()} channels | {stream.nominal_srate()} Hz")
        
        if len(streams) == 1 and not confirm:
            return streams[0]
        
        while True:
            try:
                if allow_skip:
                    choice = input(f"Select stream [0-{len(streams)-1}] or 's' to skip: ")
                    if choice.lower() == 's':
                        return None
                else:
                    choice = input(f"Select stream [0-{len(streams)-1}]: ")
                
                idx = int(choice)
                if 0 <= idx < len(streams):
                    return streams[idx]
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a number.")
                
    except Exception as e:
        print(f"Error resolving {stream_type} streams: {e}")
        return None

def setup_mi_output_streams():
    """Setup LSL output streams for MI data"""
    streams = {}
    
    # Standard MI stream
    mi_info = StreamInfo('mindfulness_index', 'MI', 1, 1, 'float32', 'mi_stream')
    streams['mi'] = StreamOutlet(mi_info)
    
    # Raw MI stream  
    raw_mi_info = StreamInfo('raw_mindfulness_index', 'RawMI', 1, 1, 'float32', 'raw_mi_stream')
    streams['raw_mi'] = StreamOutlet(raw_mi_info)
    
    # State stream
    state_info = StreamInfo('mindfulness_state', 'State', 1, 1, 'string', 'state_stream')
    streams['state'] = StreamOutlet(state_info)
    
    print("LSL output streams created:")
    print("  - mindfulness_index (1 Hz)")
    print("  - raw_mindfulness_index (1 Hz)")
    print("  - mindfulness_state (1 Hz)")
    
    return streams

# ===========================
# VISUALIZATION
# ===========================

class RealTimeVisualizer:
    """Real-time visualization for calibrated MI pipeline"""
    
    def __init__(self, calibration_data=None):
        self.calibration_data = calibration_data
        self.mi_history = []
        self.raw_mi_history = []
        self.state_history = []
        self.timestamps = []
        self.features_history = {feat: [] for feat in FEATURE_ORDER}
        
    def update(self, mi, raw_mi, state, features=None, timestamp=None):
        """Update visualization data"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.mi_history.append(mi)
        self.raw_mi_history.append(raw_mi)
        self.state_history.append(state)
        self.timestamps.append(timestamp)
        
        if features is not None:
            for i, feat_name in enumerate(FEATURE_ORDER):
                if i < len(features):
                    self.features_history[feat_name].append(features[i])
                else:
                    self.features_history[feat_name].append(np.nan)
    
    def create_session_report(self, user_id, session_duration):
        """Create comprehensive session report with visualizations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main MI plot with adaptive thresholds
        ax_main = fig.add_subplot(gs[0, :])
        
        x = np.arange(len(self.mi_history))
        ax_main.plot(x, self.mi_history, 'b-', linewidth=2, label='Mindfulness Index')
        
        # Add adaptive threshold lines if calibration data available
        if self.calibration_data and 'adaptive_thresholds' in self.calibration_data:
            thresholds = self.calibration_data['adaptive_thresholds']
            
            ax_main.axhline(y=thresholds['unfocused_threshold'], color='red', linestyle='--', 
                           alpha=0.7, label='Unfocused threshold')
            ax_main.axhline(y=thresholds['neutral_threshold'], color='orange', linestyle='--', 
                           alpha=0.7, label='Neutral threshold')
            ax_main.axhline(y=thresholds['focused_threshold'], color='green', linestyle='--', 
                           alpha=0.7, label='Focused threshold')
            
            # Add colored background regions
            ax_main.fill_between(x, 0, thresholds['unfocused_threshold'], 
                               alpha=0.1, color='red', label='Unfocused region')
            ax_main.fill_between(x, thresholds['unfocused_threshold'], thresholds['neutral_threshold'], 
                               alpha=0.1, color='orange', label='Neutral region')
            ax_main.fill_between(x, thresholds['neutral_threshold'], thresholds['focused_threshold'], 
                               alpha=0.1, color='green', label='Focused region')
            ax_main.fill_between(x, thresholds['focused_threshold'], 1, 
                               alpha=0.1, color='darkgreen', label='Highly focused region')
        
        ax_main.set_title(f'Mindfulness Index Over Time - {user_id}', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('MI Value')
        ax_main.set_xlabel('Time (seconds)')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_ylim(0, 1)
        
        # Raw MI plot
        ax_raw = fig.add_subplot(gs[1, 0])
        ax_raw.plot(x, self.raw_mi_history, 'purple', linewidth=1.5)
        ax_raw.set_title('Raw MI (Extended Range)')
        ax_raw.set_ylabel('Raw MI')
        ax_raw.grid(True, alpha=0.3)
        
        # State distribution
        ax_state = fig.add_subplot(gs[1, 1])
        state_counts = {}
        for state in self.state_history:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        if state_counts:
            states, counts = zip(*state_counts.items())
            colors = {'unfocused': 'red', 'neutral': 'orange', 'focused': 'green', 
                     'highly_focused': 'darkgreen', 'unknown': 'gray'}
            bar_colors = [colors.get(state, 'gray') for state in states]
            ax_state.bar(states, counts, color=bar_colors, alpha=0.7)
            ax_state.set_title('State Distribution')
            ax_state.set_ylabel('Count')
            plt.setp(ax_state.xaxis.get_majorticklabels(), rotation=45)
        
        # Feature trends
        ax_feat1 = fig.add_subplot(gs[2, 0])
        ax_feat1.plot(x, self.features_history['theta_fz'], label='Theta (Fz)', color='red')
        ax_feat1.plot(x, self.features_history['alpha_po'], label='Alpha (PO)', color='blue')
        ax_feat1.set_title('EEG Features')
        ax_feat1.legend()
        ax_feat1.grid(True, alpha=0.3)
        
        ax_feat2 = fig.add_subplot(gs[2, 1])
        ax_feat2.plot(x, self.features_history['faa'], label='FAA', color='purple')
        ax_feat2.plot(x, self.features_history['eda_norm'], label='EDA', color='brown')
        ax_feat2.set_title('Emotional Features')
        ax_feat2.legend()
        ax_feat2.grid(True, alpha=0.3)
        
        # Session statistics
        ax_stats = fig.add_subplot(gs[3, :])
        ax_stats.axis('off')
        
        # Calculate session statistics
        avg_mi = np.mean(self.mi_history) if self.mi_history else 0
        std_mi = np.std(self.mi_history) if self.mi_history else 0
        
        stats_text = f"""
SESSION STATISTICS:
Duration: {session_duration:.1f} seconds | Samples: {len(self.mi_history)}
Average MI: {avg_mi:.3f} ± {std_mi:.3f}
        """
        
        if state_counts:
            total_samples = sum(state_counts.values())
            stats_text += "State percentages: "
            for state, count in state_counts.items():
                percentage = (count / total_samples) * 100
                stats_text += f"{state}: {percentage:.1f}% | "
        
        ax_stats.text(0.05, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Save plot
        plot_path = os.path.join(VIS_DIR, f'{user_id}_calibrated_session_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save session data
        session_data = {
            'user_id': user_id,
            'timestamp': timestamp,
            'duration': session_duration,
            'mi_history': self.mi_history,
            'raw_mi_history': self.raw_mi_history,
            'state_history': self.state_history,
            'features_history': self.features_history,
            'statistics': {
                'avg_mi': avg_mi,
                'std_mi': std_mi,
                'state_counts': state_counts
            }
        }
        
        session_path = os.path.join(LOG_DIR, f'{user_id}_calibrated_session_{timestamp}.json')
        with open(session_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"\n📊 SESSION REPORT GENERATED")
        print(f"Plot: {plot_path}")
        print(f"Data: {session_path}")
        
        return plot_path, session_path

# ===========================
# MAIN PIPELINE
# ===========================

def main():
    """Main pipeline with dual calibration and adaptive thresholding"""
    print(f"\n{'='*80}")
    print("REAL-TIME MI LSL PIPELINE - CALIBRATED VERSION")
    print(f"{'='*80}")
    print("Features:")
    print("✓ Dual-phase calibration (relaxed + focused)")
    print("✓ Adaptive per-user thresholding")
    print("✓ Peak suppression and artifact rejection")
    print("✓ Real-time feedback with personalized baselines")
    print("✓ Comprehensive session reporting")
    print(f"{'='*80}\n")
    
    # Get user ID
    user_id = input("Enter user ID for this session: ").strip()
    if not user_id:
        print("[ERROR] User ID required!")
        return
    
    # Check for existing calibration
    calibration_data = load_user_calibration(user_id)
    
    if calibration_data:
        calib_time = calibration_data.get('timestamp', 'unknown')
        print(f"\n✅ Found existing calibration for {user_id} (created: {calib_time})")
        
        # Show calibration summary
        thresholds = calibration_data.get('adaptive_thresholds', {})
        if thresholds:
            print(f"Current adaptive thresholds:")
            print(f"  Low baseline: {thresholds.get('low_baseline', 0):.3f}")
            print(f"  High baseline: {thresholds.get('high_baseline', 0):.3f}")
            print(f"  MI range: {thresholds.get('mi_range', 0):.3f}")
        
        recalibrate = input("\nRun new calibration? (y/n, default: n): ").strip().lower()
        run_calibration = recalibrate == 'y'
    else:
        print(f"\n❌ No calibration found for {user_id}")
        print("Calibration is required for adaptive thresholding.")
        run_calibration = True
    
    # Setup LSL streams
    print("\n🔌 SETTING UP LSL STREAMS")
    print("="*40)
    
    print("Select EEG stream (raw, unconverted data):")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream')
    if eeg_stream is None:
        print("[ERROR] EEG stream required!")
        return
    
    if eeg_stream.channel_count() < 8:
        print(f"[ERROR] Need at least 8 EEG channels, got {eeg_stream.channel_count()}")
        return
    
    eeg_inlet = StreamInlet(eeg_stream)
    print(f"✓ EEG stream connected: {eeg_stream.name()} ({eeg_stream.channel_count()} channels)")
    
    print("\nSelect EDA stream (raw, unconverted data):")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals')
    if eda_stream is None:
        print("[ERROR] EDA stream required!")
        return
    
    if eda_stream.channel_count() < 2:
        print(f"[ERROR] Need at least 2 EDA channels, got {eda_stream.channel_count()}")
        return
    
    eda_inlet = StreamInlet(eda_stream)
    print(f"✓ EDA stream connected: {eda_stream.name()} ({eda_stream.channel_count()} channels)")
    
    # Run calibration if needed
    if run_calibration:
        calibration_manager = CalibrationManager(user_id)
        success, calibration_data = calibration_manager.run_dual_calibration(eeg_inlet, eda_inlet)
        
        if not success:
            print("[ERROR] Calibration failed! Cannot proceed with adaptive thresholding.")
            return
        
        print("🎉 Calibration completed successfully!")
    
    # Setup output streams
    print("\n📡 SETTING UP OUTPUT STREAMS")
    output_streams = setup_mi_output_streams()
    
    # Setup visualization
    visualizer = RealTimeVisualizer(calibration_data)
    
    # Real-time MI calculation
    print(f"\n{'='*60}")
    print("STARTING REAL-TIME MI CALCULATION")
    print(f"{'='*60}")
    print("Instructions:")
    print("- MI will be calculated every second")
    print("- Values are automatically adapted to your personal baselines")
    print("- Press Enter or ESC to stop and generate report")
    print(f"{'='*60}\n")
    
    # Setup exit detection
    stop_flag = {'stop': False}
    
    def wait_for_exit():
        print("💡 Press Enter or ESC to end session and generate report...\n")
        while not stop_flag['stop']:
            if os.name == 'nt':  # Windows
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in [b'\r', b'\n', b'\x1b']:  # Enter or ESC
                        stop_flag['stop'] = True
                        break
            time.sleep(0.1)
    
    exit_thread = threading.Thread(target=wait_for_exit)
    exit_thread.daemon = True
    exit_thread.start()
    
    # Main processing loop
    session_start_time = time.time()
    window_size = SAMPLING_RATE  # 1-second windows
    eeg_buffer = []
    eda_buffer = []
    
    print("🚀 Real-time processing started...")
    print("State legend: 🔴 Unfocused | 🟡 Neutral | 🟢 Focused | 🟢+ Highly Focused")
    print("-" * 80)
    
    try:
        while not stop_flag['stop']:
            loop_start = time.time()
            
            # Collect 1 second of data
            eeg_window_data = []
            eda_window_data = []
            
            for _ in range(window_size):
                # Get EEG sample
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample is not None:
                    eeg_window_data.append(eeg_sample[:8])
                
                # Get EDA sample
                eda_sample, _ = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample is not None:
                    eda_window_data.append(eda_sample[:2])
                
                if stop_flag['stop']:
                    break
            
            if stop_flag['stop']:
                break
            
            # Process if we have enough data
            if len(eeg_window_data) >= window_size * 0.8 and len(eda_window_data) >= window_size * 0.8:
                eeg_window = np.array(eeg_window_data[-window_size:])
                eda_window = np.array(eda_window_data[-window_size:])
                
                # Extract features with peak suppression
                features, quality = extract_features_robust(eeg_window, eda_window)
                
                if all(feat in features for feat in FEATURE_ORDER):
                    feature_values = [features[feat] for feat in FEATURE_ORDER]
                    
                    # Calculate adaptive MI
                    mi, raw_mi, state = calculate_adaptive_mi(feature_values, calibration_data)
                    
                    # Send to output streams
                    current_time = time.time()
                    output_streams['mi'].push_sample([mi], current_time)
                    output_streams['raw_mi'].push_sample([raw_mi], current_time)
                    output_streams['state'].push_sample([state], current_time)
                    
                    # Update visualization
                    visualizer.update(mi, raw_mi, state, feature_values)
                    
                    # Display progress
                    elapsed = time.time() - session_start_time
                    state_emoji = {'unfocused': '🔴', 'neutral': '🟡', 'focused': '🟢', 'highly_focused': '🟢+'}.get(state, '❓')
                    
                    artifacts_eeg = quality.get('eeg_artifact_percentage', 0)
                    artifacts_eda = quality.get('eda_artifact_percentage', 0)
                    
                    print(f"[{elapsed:6.1f}s] MI: {mi:.3f} | Raw: {raw_mi:+.2f} | {state_emoji} {state:<13} | Quality: EEG {artifacts_eeg:4.1f}%, EDA {artifacts_eda:4.1f}%")
                else:
                    print(f"[{time.time() - session_start_time:6.1f}s] Feature extraction failed - skipping window")
            
            # Maintain 1 Hz timing
            elapsed_loop = time.time() - loop_start
            if elapsed_loop < 1.0:
                time.sleep(1.0 - elapsed_loop)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Session interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
    
    # Generate session report
    session_duration = time.time() - session_start_time
    print(f"\n{'='*60}")
    print("GENERATING SESSION REPORT")
    print(f"{'='*60}")
    
    plot_path, data_path = visualizer.create_session_report(user_id, session_duration)
    
    print(f"\n✅ SESSION COMPLETED")
    print(f"Duration: {session_duration:.1f} seconds")
    print(f"Samples processed: {len(visualizer.mi_history)}")
    print(f"Report saved: {plot_path}")
    print(f"Data saved: {data_path}")
    
    if calibration_data:
        avg_mi = np.mean(visualizer.mi_history) if visualizer.mi_history else 0
        thresholds = calibration_data.get('adaptive_thresholds', {})
        low_baseline = thresholds.get('low_baseline', 0)
        high_baseline = thresholds.get('high_baseline', 1)
        
        # Calculate performance relative to calibration
        relative_performance = (avg_mi - low_baseline) / (high_baseline - low_baseline)
        relative_performance = np.clip(relative_performance, 0, 1)
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"Your average MI: {avg_mi:.3f}")
        print(f"Relative to your calibration: {relative_performance*100:.1f}%")
        print(f"(0% = relaxed baseline, 100% = focused baseline)")
    
    print(f"\n🎉 Thank you for using the Calibrated MI Pipeline!")

if __name__ == "__main__":
    main()
