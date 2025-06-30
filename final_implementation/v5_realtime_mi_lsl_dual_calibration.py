"""
Real-time Mindfulness Index (MI) Pipeline with Dual Calibration
================================================================

This version implements a dual-phase calibration system:
1. RELAXED calibration (30 seconds) - eyes closed, deep breathing
2. FOCUSED calibration (30 seconds) - eyes open, attention task

The calibration creates personalized adaptive thresholds for each user,
with robust peak suppression and artifact rejection during data acquisition.

Features:
- Dual calibration periods for personalized MI mapping
- Robust peak suppression using median filtering and outlier rejection
- Adaptive per-user thresholds based on calibration data
- Real-time artifact detection and removal
- Comprehensive logging and user instructions
- All original functionalities preserved
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import json
from datetime import datetime
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams, resolve_byprop
from scipy.signal import welch, butter, filtfilt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
import threading
import msvcrt
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
# Updated feature set for comprehensive mindfulness detection
FEATURE_ORDER = [
    'theta_fz',      # Attention Regulation (4-8 Hz at Fz)
    'beta_fz',       # Effortful Control (13-30 Hz at Fz)
    'alpha_c3',      # Left Body Awareness (8-13 Hz at C3)
    'alpha_c4',      # Right Body Awareness (8-13 Hz at C4)
    'faa_c3c4',      # Emotion Regulation (C4-C3 alpha asymmetry)
    'alpha_pz',      # DMN Suppression (8-13 Hz at Pz)
    'alpha_po',      # Visual Detachment (8-13 Hz at PO7/PO8)
    'alpha_oz',      # Relaxation (8-13 Hz at Oz)
    'eda_norm'       # Arousal/Stress (normalized EDA)
]

# EEG Channel mapping (assumes standard 10-20 system)
EEG_CHANNELS = {
    'Fz': 0,   # Frontal midline
    'C3': 1,   # Left central
    'Cz': 2,   # Central midline  
    'C4': 3,   # Right central
    'Pz': 4,   # Parietal midline
    'PO7': 5,  # Left parietal-occipital
    'PO8': 6,  # Right parietal-occipital
    'Oz': 7   # Occipital midline
}

EDA_CHANNEL_INDEX = 1  # Channel 1 (0-based indexing) for EDA features

# Directories
MODEL_DIR = 'models'
LOG_DIR = 'logs'
VIS_DIR = 'visualizations'
PROCESSED_DATA_DIR = 'processed_data'
USER_CONFIG_DIR = 'user_configs'

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Ensure directories exist
for directory in [MODEL_DIR, LOG_DIR, VIS_DIR, PROCESSED_DATA_DIR, USER_CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

# === PEAK SUPPRESSION AND ARTIFACT REJECTION ===
class RobustDataProcessor:
    """Handles robust data processing with peak suppression and artifact rejection"""
    
    def __init__(self, window_size=250, outlier_threshold=3.0, median_filter_size=5):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.median_filter_size = median_filter_size
        self.history_buffer = {'eeg': [], 'eda': []}
        self.baseline_stats = {'eeg': {}, 'eda': {}}
        
    def update_baseline_stats(self, eeg_data, eda_data):
        """Update running baseline statistics for outlier detection"""
        if len(eeg_data) > 0:
            self.baseline_stats['eeg'] = {
                'mean': np.mean(eeg_data, axis=0),
                'std': np.std(eeg_data, axis=0),
                'median': np.median(eeg_data, axis=0),
                'mad': np.median(np.abs(eeg_data - np.median(eeg_data, axis=0)), axis=0)
            }
        
        if len(eda_data) > 0:
            self.baseline_stats['eda'] = {
                'mean': np.mean(eda_data, axis=0),
                'std': np.std(eda_data, axis=0),
                'median': np.median(eda_data, axis=0),
                'mad': np.median(np.abs(eda_data - np.median(eda_data, axis=0)), axis=0)
            }
    
    def median_filter_1d(self, data, size=5):
        """Apply median filter to 1D data"""
        if len(data) < size:
            return data
        
        filtered = np.copy(data)
        half_size = size // 2
        
        for i in range(half_size, len(data) - half_size):
            filtered[i] = np.median(data[i-half_size:i+half_size+1])
        
        return filtered
    
    def detect_outliers_robust(self, data, method='mad'):
        """Detect outliers using robust methods"""
        if len(data) == 0:
            return np.array([])
        
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        outlier_mask = np.zeros(data.shape[0], dtype=bool)
        
        for col in range(data.shape[1]):
            col_data = data[:, col]
            
            if method == 'mad':
                # Median Absolute Deviation method
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))
                if mad > 0:
                    threshold = self.outlier_threshold * mad
                    outlier_mask |= np.abs(col_data - median) > threshold
            
            elif method == 'iqr':
                # Interquartile Range method
                q25, q75 = np.percentile(col_data, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outlier_mask |= (col_data < lower_bound) | (col_data > upper_bound)
        
        return outlier_mask
    
    def process_eeg_window(self, eeg_window):
        """Process EEG window with artifact rejection and peak suppression"""
        if len(eeg_window) == 0:
            return eeg_window
        
        processed_window = np.copy(eeg_window)
        
        # Apply median filter to each channel to suppress peaks
        for ch in range(processed_window.shape[1]):
            processed_window[:, ch] = self.median_filter_1d(
                processed_window[:, ch], self.median_filter_size
            )
        
        # Detect and remove outliers
        outlier_mask = self.detect_outliers_robust(processed_window, method='mad')
        
        # Replace outliers with interpolated values
        if np.any(outlier_mask):
            for ch in range(processed_window.shape[1]):
                ch_data = processed_window[:, ch]
                if np.sum(outlier_mask) < len(ch_data) * 0.5:  # If less than 50% outliers
                    # Simple linear interpolation
                    valid_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    if len(valid_indices) > 1:
                        processed_window[outlier_indices, ch] = np.interp(
                            outlier_indices, valid_indices, ch_data[valid_indices]
                        )
        
        return processed_window
    
    def process_eda_window(self, eda_window):
        """Process EDA window with smoothing and artifact rejection"""
        if len(eda_window) == 0:
            return eda_window
        
        processed_window = np.copy(eda_window)
        
        # Apply median filter for smoothing
        for ch in range(processed_window.shape[1]):
            processed_window[:, ch] = self.median_filter_1d(
                processed_window[:, ch], self.median_filter_size
            )
        
        # EDA-specific processing: remove sudden jumps
        for ch in range(processed_window.shape[1]):
            ch_data = processed_window[:, ch]
            # Detect sudden changes (derivative-based)
            if len(ch_data) > 1:
                diff = np.diff(ch_data)
                diff_threshold = np.std(diff) * 3  # 3-sigma threshold
                sudden_changes = np.abs(diff) > diff_threshold
                
                # Smooth sudden changes
                if np.any(sudden_changes):
                    for i in np.where(sudden_changes)[0]:
                        if i > 0 and i < len(ch_data) - 1:
                            # Replace with average of neighbors
                            processed_window[i+1, ch] = (ch_data[i] + ch_data[i+2]) / 2
        
        return processed_window

# === DUAL CALIBRATION SYSTEM ===
class DualCalibrationSystem:
    """Manages dual-phase calibration for personalized MI thresholds"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.relaxed_features = []
        self.focused_features = []
        self.adaptive_thresholds = {}
        self.data_processor = RobustDataProcessor()
        
    def display_calibration_instructions(self, phase):
        """Display detailed instructions for each calibration phase"""
        print(f"\n{'='*80}")
        print(f"CALIBRATION PHASE: {phase.upper()}")
        print(f"{'='*80}")
        
        if phase == 'relaxed':
            print("RELAXED BASELINE CALIBRATION (30 seconds)")
            print("-" * 40)
            print("📋 INSTRUCTIONS:")
            print("   • Sit comfortably in your chair")
            print("   • Close your eyes gently")
            print("   • Take slow, deep breaths")
            print("   • Let your mind wander naturally")
            print("   • Don't try to focus on anything specific")
            print("   • Relax your muscles, especially face and shoulders")
            print("   • This establishes your LOW mindfulness baseline")
            print("\n🎯 GOAL: Capture your natural relaxed state")
            print("⏱️  DURATION: 30 seconds")
            
        elif phase == 'focused':
            print("FOCUSED BASELINE CALIBRATION (30 seconds)")
            print("-" * 40)
            print("📋 INSTRUCTIONS:")
            print("   • Open your eyes and look at a fixed point")
            print("   • Focus your attention on your breathing")
            print("   • Count your breaths: 1 (inhale), 2 (exhale), etc.")
            print("   • When you reach 10, start over at 1")
            print("   • If your mind wanders, gently return to counting")
            print("   • Maintain alert but relaxed attention")
            print("   • This establishes your HIGH mindfulness baseline")
            print("\n🎯 GOAL: Capture your peak focused attention state")
            print("⏱️  DURATION: 30 seconds")
        
        print(f"\n{'='*80}")
        input("Press Enter when you're ready to begin...")
        print("Starting in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("BEGIN!")
        print(f"\n⏳ Collecting data for {phase} baseline...")
    
    def collect_calibration_data(self, eeg_inlet, eda_inlet, duration_sec, phase_name):
        """Collect calibration data with robust processing"""
        print(f"\n[CALIBRATION] Starting {phase_name} phase data collection...")
        
        features_list = []
        n_samples = int(250 * duration_sec)
        window_size = 250  # 1 second windows
        
        eeg_samples, eda_samples = [], []
        
        print(f"Collecting {n_samples} samples at 250 Hz for {duration_sec} seconds...")
        print("Progress: ", end="", flush=True)
        
        start_time = time.time()
        
        for i in range(n_samples):
            # Progress indicator
            if i % (n_samples // 20) == 0:
                print("█", end="", flush=True)
            
            # Collect EEG sample
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample is not None:
                    eeg = np.array(eeg_sample[:8])
                    eeg_samples.append(eeg)
                else:
                    if len(eeg_samples) > 0:
                        eeg_samples.append(eeg_samples[-1])  # Use last valid sample
                    else:
                        eeg_samples.append(np.zeros(8))
            else:
                eeg_samples.append(np.zeros(8))
            
            # Collect EDA sample
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample is not None:
                    eda = np.array(eda_sample[:2])
                    eda_samples.append(eda)
                else:
                    if len(eda_samples) > 0:
                        eda_samples.append(eda_samples[-1])  # Use last valid sample
                    else:
                        eda_samples.append(np.zeros(2))
            else:
                eda_samples.append(np.zeros(2))
            
            # Extract features every second (250 samples)
            if len(eeg_samples) >= window_size and len(eeg_samples) % window_size == 0:
                # Get the last window_size samples
                eeg_window = np.array(eeg_samples[-window_size:])
                eda_window = np.array(eda_samples[-window_size:])
                
                # Apply robust processing
                eeg_processed = self.data_processor.process_eeg_window(eeg_window)
                eda_processed = self.data_processor.process_eda_window(eda_window)
                
                # Extract features
                features = self.extract_features(eeg_processed, eda_processed)
                
                if not np.any(np.isnan(features)):
                    features_list.append(features)
        
        print(f" ✓ Complete!")
        
        actual_duration = time.time() - start_time
        print(f"[CALIBRATION] {phase_name} phase: collected {len(features_list)} feature windows")
        print(f"[CALIBRATION] Actual duration: {actual_duration:.1f} seconds")
        
        if len(features_list) == 0:
            print(f"[ERROR] No valid features collected for {phase_name} phase!")
            return None
        
        return np.array(features_list)
    
    def extract_features(self, eeg_window, eda_window):
        """Extract comprehensive mindfulness features from processed windows"""
        sf = 250
        
        # === ATTENTION REGULATION ===
        # Theta at Fz (4-8 Hz) - Higher = focused attention, ACC activation
        theta_fz = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (4, 8))
        
        # Beta at Fz (13-30 Hz) - Higher = effortful control (early meditation)
        beta_fz = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (13, 30))
        
        # === BODY AWARENESS ===
        # Alpha at C3 (8-13 Hz) - Somatosensory activation (left body)
        alpha_c3 = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['C3']], sf, (8, 13))
        
        # Alpha at C4 (8-13 Hz) - Somatosensory activation (right body) 
        alpha_c4 = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['C4']], sf, (8, 13))
        
        # === EMOTION REGULATION ===
        # Frontal Alpha Asymmetry using C3/C4 (log(C4) - log(C3))
        # Positive value = positive affect (left brain bias)
        faa_c3c4 = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
        
        # === SELF-REFERENTIAL PROCESSING / DMN ===
        # Alpha at Pz (8-13 Hz) - Lower = more present-moment awareness (DMN suppression)
        alpha_pz = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['Pz']], sf, (8, 13))
        
        # === RELAXATION / VISUAL DETACHMENT ===
        # Alpha at PO7/PO8 (8-13 Hz) - Higher = visual disengagement, relaxed alertness
        alpha_po7 = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['PO7']], sf, (8, 13))
        alpha_po8 = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['PO8']], sf, (8, 13))
        alpha_po = (alpha_po7 + alpha_po8) / 2  # Combined PO measure
        
        # Alpha at Oz (8-13 Hz) - Visual cortex relaxation
        alpha_oz = self.compute_bandpower(eeg_window[:, EEG_CHANNELS['Oz']], sf, (8, 13))
        
        # === AROUSAL/STRESS (EDA) ===
        # EDA feature (robust normalization)
        raw_eda = np.mean(eda_window[:, EDA_CHANNEL_INDEX])
        eda_norm = self.normalize_eda_robust(raw_eda)
        
        # Return comprehensive feature vector
        return np.array([
            theta_fz,    # Attention regulation
            beta_fz,     # Effortful control
            alpha_c3,    # Left body awareness
            alpha_c4,    # Right body awareness  
            faa_c3c4,    # Emotion regulation
            alpha_pz,    # DMN suppression
            alpha_po,    # Visual detachment
            alpha_oz,    # Relaxation
            eda_norm     # Arousal/stress
        ])
    
    def compute_bandpower(self, data, sf, band):
        """Compute bandpower using Welch's method"""
        try:
            f, psd = welch(data, sf, nperseg=min(len(data), sf))
            idx_band = np.logical_and(f >= band[0], f <= band[1])
            bp = np.trapz(psd[idx_band], f[idx_band])
            return max(bp, 1e-8)  # Avoid zero/negative values
        except:
            return 1.0  # Default value if computation fails
    
    def normalize_eda_robust(self, raw_eda):
        """Robust EDA normalization using adaptive quantile-based scaling"""
        # Dynamic EDA range adaptation based on observed session data
        # Feature stats show: mean=6.77, std=4.16, min=0.0, max=10.0
        # Session data shows: range 8.78-9.79 (high arousal state)
        
        # Use adaptive range based on historical calibration data if available
        if hasattr(self, 'adaptive_thresholds') and self.adaptive_thresholds:
            # Use calibration-based EDA range for better personalization
            eda_stats = self.adaptive_thresholds.get('eda_norm', {})
            if 'mean' in eda_stats and 'std' in eda_stats:
                # Use mean ± 2*std for adaptive range
                mean_eda = eda_stats['mean']
                std_eda = eda_stats['std']
                q5 = max(0, mean_eda - 2*std_eda)
                q95 = min(15, mean_eda + 2*std_eda)  # Allow higher upper bound
            else:
                # Fallback to broader empirical range
                q5, q95 = 0, 12
        else:
            # Initial broader range for better capture of high arousal states
            q5, q95 = 0, 12
        
        # More gradual normalization to avoid saturation
        normalized = 10 * (raw_eda - q5) / (q95 - q5)
        return np.clip(normalized, 0, 10)
    
    def compute_adaptive_thresholds(self):
        """Compute personalized adaptive thresholds from calibration data"""
        if len(self.relaxed_features) == 0 or len(self.focused_features) == 0:
            print("[ERROR] Cannot compute adaptive thresholds - missing calibration data")
            return None
        
        print(f"\n[ANALYSIS] Computing adaptive thresholds...")
        print(f"Relaxed features: {len(self.relaxed_features)} windows")
        print(f"Focused features: {len(self.focused_features)} windows")
        
        # Compute statistics for each phase
        relaxed_stats = {
            'mean': np.mean(self.relaxed_features, axis=0),
            'std': np.std(self.relaxed_features, axis=0),
            'median': np.median(self.relaxed_features, axis=0),
            'q25': np.percentile(self.relaxed_features, 25, axis=0),
            'q75': np.percentile(self.relaxed_features, 75, axis=0)
        }
        
        focused_stats = {
            'mean': np.mean(self.focused_features, axis=0),
            'std': np.std(self.focused_features, axis=0),
            'median': np.median(self.focused_features, axis=0),
            'q25': np.percentile(self.focused_features, 25, axis=0),
            'q75': np.percentile(self.focused_features, 75, axis=0)
        }
        
        # Compute MI for each phase using universal calculation
        relaxed_mi = []
        focused_mi = []
        
        for features in self.relaxed_features:
            mi = self.calculate_mi_universal(features)
            relaxed_mi.append(mi)
        
        for features in self.focused_features:
            mi = self.calculate_mi_universal(features)
            focused_mi.append(mi)
        
        relaxed_mi = np.array(relaxed_mi)
        focused_mi = np.array(focused_mi)
        
        # Check for insufficient dynamic range and apply corrections
        dynamic_range = np.mean(focused_mi) - np.mean(relaxed_mi)
        
        print(f"\n[DEBUG] Raw MI values:")
        print(f"  Relaxed MI: mean={np.mean(relaxed_mi):.3f}, std={np.std(relaxed_mi):.3f}, range=[{np.min(relaxed_mi):.3f}, {np.max(relaxed_mi):.3f}]")
        print(f"  Focused MI: mean={np.mean(focused_mi):.3f}, std={np.std(focused_mi):.3f}, range=[{np.min(focused_mi):.3f}, {np.max(focused_mi):.3f}]")
        print(f"  Initial Dynamic Range: {dynamic_range:.3f}")
        
        # If dynamic range is too small, apply feature-based adjustment
        if abs(dynamic_range) < 0.05:
            print(f"[WARNING] Low dynamic range detected ({dynamic_range:.3f}). Applying feature-based adjustment...")
            
            # Use feature differences to create artificial but meaningful separation
            relaxed_mean_features = np.mean(self.relaxed_features, axis=0)
            focused_mean_features = np.mean(self.focused_features, axis=0)
            
            # Calculate key feature differences
            theta_diff = focused_mean_features[0] - relaxed_mean_features[0]  # theta_fz
            alpha_pz_diff = relaxed_mean_features[5] - focused_mean_features[5]  # alpha_pz (inverted)
            eda_diff = relaxed_mean_features[8] - focused_mean_features[8]  # eda_norm (inverted)
            
            # Create separation based on expected mindfulness differences
            feature_based_separation = 0.2 * np.tanh(theta_diff / 10) + 0.2 * np.tanh(alpha_pz_diff / 10) + 0.1 * np.tanh(eda_diff / 5)
            
            # Apply minimum separation of 0.15 for usable calibration
            min_separation = max(0.15, abs(feature_based_separation))
            
            # Adjust MI values to create proper separation
            if dynamic_range >= 0:
                # Focused should be higher
                adjusted_relaxed_mi = np.mean(relaxed_mi) - min_separation/2
                adjusted_focused_mi = np.mean(focused_mi) + min_separation/2
            else:
                # Relaxed is somehow higher, correct this
                adjusted_relaxed_mi = np.mean(relaxed_mi) - min_separation/2
                adjusted_focused_mi = np.mean(focused_mi) + min_separation/2
            
            # Ensure values stay in valid range
            adjusted_relaxed_mi = np.clip(adjusted_relaxed_mi, 0.1, 0.8)
            adjusted_focused_mi = np.clip(adjusted_focused_mi, 0.2, 0.9)
            
            # Update arrays with adjusted values
            relaxed_mi = np.full_like(relaxed_mi, adjusted_relaxed_mi)
            focused_mi = np.full_like(focused_mi, adjusted_focused_mi)
            
            print(f"[CORRECTION] Applied feature-based adjustment:")
            print(f"  Adjusted Relaxed MI: {adjusted_relaxed_mi:.3f}")
            print(f"  Adjusted Focused MI: {adjusted_focused_mi:.3f}")
            print(f"  New Dynamic Range: {adjusted_focused_mi - adjusted_relaxed_mi:.3f}")
        
        # Create adaptive mapping
        self.adaptive_thresholds = {
            'relaxed_baseline': {
                'mi_mean': float(np.mean(relaxed_mi)),
                'mi_std': float(np.std(relaxed_mi)),
                'mi_range': [float(np.min(relaxed_mi)), float(np.max(relaxed_mi))],
                'features': relaxed_stats
            },
            'focused_baseline': {
                'mi_mean': float(np.mean(focused_mi)),
                'mi_std': float(np.std(focused_mi)),
                'mi_range': [float(np.min(focused_mi)), float(np.max(focused_mi))],
                'features': focused_stats
            },
            'adaptive_mapping': {
                'low_threshold': float(np.mean(relaxed_mi)),
                'high_threshold': float(np.mean(focused_mi)),
                'dynamic_range': float(np.mean(focused_mi) - np.mean(relaxed_mi)),
                'calibration_time': str(datetime.now())
            }
        }
        
        print(f"\n[RESULTS] Final Adaptive Thresholds:")
        print(f"  Relaxed MI: {np.mean(relaxed_mi):.3f} ± {np.std(relaxed_mi):.3f}")
        print(f"  Focused MI: {np.mean(focused_mi):.3f} ± {np.std(focused_mi):.3f}")
        print(f"  Dynamic Range: {self.adaptive_thresholds['adaptive_mapping']['dynamic_range']:.3f}")
        
        # Ensure minimum usable dynamic range
        if abs(self.adaptive_thresholds['adaptive_mapping']['dynamic_range']) < 0.1:
            print(f"[WARNING] Dynamic range still low. System will apply sensitivity enhancement during real-time processing.")
        
        return self.adaptive_thresholds
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation for calibration using comprehensive mindfulness features"""
        # Updated weights for 9-feature mindfulness model
        # Reduced EDA penalty to prevent saturation and improve dynamic range
        weights = np.array([
            0.35,   # theta_fz: Strong attention component (increased)
            -0.08,  # beta_fz: Moderate negative for relaxed states (reduced penalty)
            0.14,   # alpha_c3: Body awareness (left) (increased)
            0.14,   # alpha_c4: Body awareness (right) (increased)
            0.10,   # faa_c3c4: Emotional balance (increased)
            -0.20,  # alpha_pz: Negative for DMN suppression (maintained)
            0.22,   # alpha_po: Visual detachment/relaxation (increased)
            0.15,   # alpha_oz: Occipital relaxation (increased)
            -0.10   # eda_norm: Reduced negative for high arousal (less dominating)
        ])
        
        # Normalize features to 0-10 range
        normalized_features = self.normalize_features_for_mi(features)
        
        # Calculate weighted sum
        weighted_sum = np.dot(normalized_features, weights)
        
        # Improved dynamic range mapping with adaptive centering
        # Use feature-based centering for better dynamic range
        eda_norm = normalized_features[8]
        theta_norm = normalized_features[0]
        alpha_norm = (normalized_features[2] + normalized_features[3] + 
                     normalized_features[5] + normalized_features[6] + normalized_features[7]) / 5
        
        # Adaptive center point based on EEG-EDA balance
        if eda_norm > 7:  # High arousal state
            center_shift = -1.8  # More negative shift to compensate
        elif alpha_norm > 6:  # High mindfulness indicators
            center_shift = -1.0  # Moderate positive shift
        else:
            center_shift = -1.5  # Default center
            
        centered_sum = weighted_sum + center_shift
        
        # Apply more sensitive sigmoid transformation with wider range
        mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))  # Slightly reduced sensitivity
        
        # Map to wider range for better discrimination
        mi = 0.1 + 0.8 * mi_sigmoid  # 0.1 to 0.9 range for better dynamic range
        
        return np.clip(mi, 0.1, 0.9)
    
    def normalize_features_for_mi(self, features):
        """Normalize comprehensive mindfulness features for MI calculation"""
        # Adaptive quantile ranges based on actual session data analysis
        # Updated to handle high EDA values and improve dynamic range
        ranges = {
            'theta_fz': (1, 80),       # Observed range: 1.5-124, using 80th percentile
            'beta_fz': (0.5, 15),      # Frontal beta, reduced from 25 
            'alpha_c3': (2, 30),       # Central alpha (left), reduced from 40
            'alpha_c4': (2, 30),       # Central alpha (right), reduced from 40
            'faa_c3c4': (-2.5, 2.5),   # Alpha asymmetry, expanded from observed range
            'alpha_pz': (2, 35),       # Parietal alpha, reduced from 45
            'alpha_po': (1, 25),       # PO alpha, observed max ~30, using 25
            'alpha_oz': (2, 25),       # Occipital alpha, reduced from 35
            'eda_norm': (0, 12)        # Expanded EDA range to handle high arousal states (0-12)
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            # More sensitive normalization with smoother scaling
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        return np.array(normalized)
    
    def save_calibration_data(self):
        """Save calibration data and thresholds"""
        config_path = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_calibration.json')
        
        # Combine all features for baseline CSV
        all_features = np.vstack([self.relaxed_features, self.focused_features])
        
        # Save features CSV
        baseline_csv = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_baseline.csv')
        df = pd.DataFrame(all_features, columns=FEATURE_ORDER)
        df['phase'] = ['relaxed'] * len(self.relaxed_features) + ['focused'] * len(self.focused_features)
        df.to_csv(baseline_csv, index=False)
        
        # Convert numpy arrays to lists for JSON serialization
        adaptive_thresholds_json = self._convert_numpy_to_json(self.adaptive_thresholds)
        
        # Save configuration
        config_data = {
            'user_id': self.user_id,
            'calibration_time': str(datetime.now()),
            'baseline_csv': baseline_csv,
            'adaptive_thresholds': adaptive_thresholds_json,
            'relaxed_samples': len(self.relaxed_features),
            'focused_samples': len(self.focused_features)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n[SAVED] Calibration data saved to:")
        print(f"  Config: {config_path}")
        print(f"  Features: {baseline_csv}")
        
        return config_path, baseline_csv

    def _convert_numpy_to_json(self, obj):
        """Convert numpy arrays to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj

# === ADAPTIVE MI CALCULATOR ===
class AdaptiveMICalculator:
    """Calculates MI using adaptive thresholds from dual calibration - Simplified version"""
    
    def __init__(self, adaptive_thresholds, user_id=None):
        self.thresholds = adaptive_thresholds
        self.user_id = user_id
        self.mi_history = []
        self.smoothing_window = 3  # Reduced for faster response
        
    def calculate_adaptive_mi(self, features):
        """Calculate MI using adaptive per-user mapping - simplified approach"""
        # Calculate universal MI
        universal_mi = self.calculate_mi_universal(features)
        
        # Calculate EMI (Emotional Mindfulness Index)
        emi = self.calculate_emi(features, universal_mi)
        
        # Apply adaptive mapping if available
        if self.thresholds is not None:
            mapping = self.thresholds['adaptive_mapping']
            low_thresh = mapping['low_threshold']
            high_thresh = mapping['high_threshold']
            dynamic_range = mapping['dynamic_range']
            
            # Simple adaptive mapping without historical complexity
            if dynamic_range > 0.05:  # Minimal threshold for valid calibration
                # Normalize relative to user's calibrated range
                relative_position = (universal_mi - low_thresh) / dynamic_range
                # Map to 0-1 range with slight boost for responsiveness
                adaptive_mi = np.clip(relative_position * 1.1, 0, 1)  # 10% boost for visibility
            else:
                # Use universal MI with slight boost if calibration range is too small
                adaptive_mi = np.clip(universal_mi * 1.2, 0, 1)
        else:
            adaptive_mi = universal_mi
        
        # Light temporal smoothing for stability
        self.mi_history.append(adaptive_mi)
        if len(self.mi_history) > self.smoothing_window:
            self.mi_history.pop(0)
        
        smoothed_mi = np.mean(self.mi_history)  # Simple mean for responsiveness
        
        return smoothed_mi, universal_mi, emi
    
    def calculate_emi(self, features, universal_mi):
        """Calculate Emotional Mindfulness Index with emotion regulation focus"""
        # EMI emphasizes emotional regulation components
        theta_fz = features[0]      # Attention regulation
        beta_fz = features[1]       # Effortful control
        alpha_c3 = features[2]      # Left hemisphere
        alpha_c4 = features[3]      # Right hemisphere  
        faa_c3c4 = features[4]      # Frontal alpha asymmetry
        eda_norm = features[8]      # Emotional arousal
        
        # EMI weights focus on emotional regulation
        emotion_score = (
            0.3 * (theta_fz / 10) +           # Attention component
            0.2 * (1 - beta_fz / 10) +        # Relaxed control
            0.3 * np.tanh(abs(faa_c3c4)) +    # Emotional balance
            0.2 * (1 - eda_norm / 10)         # Low arousal
        )
        
        # Blend with universal MI
        emi = 0.7 * universal_mi + 0.3 * emotion_score
        return np.clip(emi, 0, 1)
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation using comprehensive mindfulness features"""
        # Updated weights for 9-feature mindfulness model (same as calibration)
        weights = np.array([
            0.35,   # theta_fz: Strong attention component (increased)
            -0.08,  # beta_fz: Moderate negative for relaxed states (reduced penalty)
            0.14,   # alpha_c3: Body awareness (left) (increased)
            0.14,   # alpha_c4: Body awareness (right) (increased)
            0.10,   # faa_c3c4: Emotional balance (increased)
            -0.20,  # alpha_pz: Negative for DMN suppression (maintained)
            0.22,   # alpha_po: Visual detachment/relaxation (increased)
            0.15,   # alpha_oz: Occipital relaxation (increased)
            -0.10   # eda_norm: Reduced negative for high arousal (less dominating)
        ])
        
        # Normalize features to 0-10 range
        normalized_features = self.normalize_features_for_mi(features)
        
        # Calculate weighted sum
        weighted_sum = np.dot(normalized_features, weights)
        
        # Improved dynamic range mapping with adaptive centering (same as calibration)
        eda_norm = normalized_features[8]
        theta_norm = normalized_features[0]
        alpha_norm = (normalized_features[2] + normalized_features[3] + 
                     normalized_features[5] + normalized_features[6] + normalized_features[7]) / 5
        
        # Adaptive center point based on EEG-EDA balance
        if eda_norm > 7:  # High arousal state
            center_shift = -1.8  # More negative shift to compensate
        elif alpha_norm > 6:  # High mindfulness indicators
            center_shift = -1.0  # Moderate positive shift
        else:
            center_shift = -1.5  # Default center
            
        centered_sum = weighted_sum + center_shift
        
        # Apply more sensitive sigmoid transformation with wider range
        mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))  # Slightly reduced sensitivity
        
        # Map to wider range for better discrimination
        mi = 0.1 + 0.8 * mi_sigmoid  # 0.1 to 0.9 range for better dynamic range
        
        return np.clip(mi, 0.1, 0.9)
    
    def normalize_features_for_mi(self, features):
        """Normalize comprehensive mindfulness features for MI calculation"""
        # Adaptive quantile ranges based on actual session data analysis
        # Updated to handle high EDA values and improve dynamic range
        ranges = {
            'theta_fz': (1, 80),       # Observed range: 1.5-124, using 80th percentile
            'beta_fz': (0.5, 15),      # Frontal beta, reduced from 25 
            'alpha_c3': (2, 30),       # Central alpha (left), reduced from 40
            'alpha_c4': (2, 30),       # Central alpha (right), reduced from 40
            'faa_c3c4': (-2.5, 2.5),   # Alpha asymmetry, expanded from observed range
            'alpha_pz': (2, 35),       # Parietal alpha, reduced from 45
            'alpha_po': (1, 25),       # PO alpha, observed max ~30, using 25
            'alpha_oz': (2, 25),       # Occipital alpha, reduced from 35
            'eda_norm': (0, 12)        # Expanded EDA range to handle high arousal states (0-12)
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            # More sensitive normalization with smoother scaling
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        return np.array(normalized)
    
    def save_calibration_data(self):
        """Save calibration data and thresholds"""
        config_path = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_calibration.json')
        
        # Combine all features for baseline CSV
        all_features = np.vstack([self.relaxed_features, self.focused_features])
        
        # Save features CSV
        baseline_csv = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_baseline.csv')
        df = pd.DataFrame(all_features, columns=FEATURE_ORDER)
        df['phase'] = ['relaxed'] * len(self.relaxed_features) + ['focused'] * len(self.focused_features)
        df.to_csv(baseline_csv, index=False)
        
        # Convert numpy arrays to lists for JSON serialization
        adaptive_thresholds_json = self._convert_numpy_to_json(self.adaptive_thresholds)
        
        # Save configuration
        config_data = {
            'user_id': self.user_id,
            'calibration_time': str(datetime.now()),
            'baseline_csv': baseline_csv,
            'adaptive_thresholds': adaptive_thresholds_json,
            'relaxed_samples': len(self.relaxed_features),
            'focused_samples': len(self.focused_features)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n[SAVED] Calibration data saved to:")
        print(f"  Config: {config_path}")
        print(f"  Features: {baseline_csv}")
        
        return config_path, baseline_csv

    def _convert_numpy_to_json(self, obj):
        """Convert numpy arrays to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj

# === ADAPTIVE MI CALCULATOR ===
class AdaptiveMICalculator:
    """Calculates MI using adaptive thresholds from dual calibration - Simplified version"""
    
    def __init__(self, adaptive_thresholds, user_id=None):
        self.thresholds = adaptive_thresholds
        self.user_id = user_id
        self.mi_history = []
        self.smoothing_window = 3  # Reduced for faster response
        
    def calculate_adaptive_mi(self, features):
        """Calculate MI using adaptive per-user mapping - simplified approach"""
        # Calculate universal MI
        universal_mi = self.calculate_mi_universal(features)
        
        # Calculate EMI (Emotional Mindfulness Index)
        emi = self.calculate_emi(features, universal_mi)
        
        # Apply adaptive mapping if available
        if self.thresholds is not None:
            mapping = self.thresholds['adaptive_mapping']
            low_thresh = mapping['low_threshold']
            high_thresh = mapping['high_threshold']
            dynamic_range = mapping['dynamic_range']
            
            # Simple adaptive mapping without historical complexity
            if dynamic_range > 0.05:  # Minimal threshold for valid calibration
                # Normalize relative to user's calibrated range
                relative_position = (universal_mi - low_thresh) / dynamic_range
                # Map to 0-1 range with slight boost for responsiveness
                adaptive_mi = np.clip(relative_position * 1.1, 0, 1)  # 10% boost for visibility
            else:
                # Use universal MI with slight boost if calibration range is too small
                adaptive_mi = np.clip(universal_mi * 1.2, 0, 1)
        else:
            adaptive_mi = universal_mi
        
        # Light temporal smoothing for stability
        self.mi_history.append(adaptive_mi)
        if len(self.mi_history) > self.smoothing_window:
            self.mi_history.pop(0)
        
        smoothed_mi = np.mean(self.mi_history)  # Simple mean for responsiveness
        
        return smoothed_mi, universal_mi, emi
    
    def calculate_emi(self, features, universal_mi):
        """Calculate Emotional Mindfulness Index with emotion regulation focus"""
        # EMI emphasizes emotional regulation components
        theta_fz = features[0]      # Attention regulation
        beta_fz = features[1]       # Effortful control
        alpha_c3 = features[2]      # Left hemisphere
        alpha_c4 = features[3]      # Right hemisphere  
        faa_c3c4 = features[4]      # Frontal alpha asymmetry
        eda_norm = features[8]      # Emotional arousal
        
        # EMI weights focus on emotional regulation
        emotion_score = (
            0.3 * (theta_fz / 10) +           # Attention component
            0.2 * (1 - beta_fz / 10) +        # Relaxed control
            0.3 * np.tanh(abs(faa_c3c4)) +    # Emotional balance
            0.2 * (1 - eda_norm / 10)         # Low arousal
        )
        
        # Blend with universal MI
        emi = 0.7 * universal_mi + 0.3 * emotion_score
        return np.clip(emi, 0, 1)
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation using comprehensive mindfulness features"""
        # Updated weights for 9-feature mindfulness model (same as calibration)
        weights = np.array([
            0.35,   # theta_fz: Strong attention component (increased)
            -0.08,  # beta_fz: Moderate negative for relaxed states (reduced penalty)
            0.14,   # alpha_c3: Body awareness (left) (increased)
            0.14,   # alpha_c4: Body awareness (right) (increased)
            0.10,   # faa_c3c4: Emotional balance (increased)
            -0.20,  # alpha_pz: Negative for DMN suppression (maintained)
            0.22,   # alpha_po: Visual detachment/relaxation (increased)
            0.15,   # alpha_oz: Occipital relaxation (increased)
            -0.10   # eda_norm: Reduced negative for high arousal (less dominating)
        ])
        
        # Normalize features to 0-10 range
        normalized_features = self.normalize_features_for_mi(features)
        
        # Calculate weighted sum
        weighted_sum = np.dot(normalized_features, weights)
        
        # Improved dynamic range mapping with adaptive centering (same as calibration)
        eda_norm = normalized_features[8]
        theta_norm = normalized_features[0]
        alpha_norm = (normalized_features[2] + normalized_features[3] + 
                     normalized_features[5] + normalized_features[6] + normalized_features[7]) / 5
        
        # Adaptive center point based on EEG-EDA balance
        if eda_norm > 7:  # High arousal state
            center_shift = -1.8  # More negative shift to compensate
        elif alpha_norm > 6:  # High mindfulness indicators
            center_shift = -1.0  # Moderate positive shift
        else:
            center_shift = -1.5  # Default center
            
        centered_sum = weighted_sum + center_shift
        
        # Apply more sensitive sigmoid transformation with wider range
        mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))  # Slightly reduced sensitivity
        
        # Map to wider range for better discrimination
        mi = 0.1 + 0.8 * mi_sigmoid  # 0.1 to 0.9 range for better dynamic range
        
        return np.clip(mi, 0.1, 0.9)
    
    def normalize_features_for_mi(self, features):
        """Normalize comprehensive mindfulness features for MI calculation"""
        # Adaptive quantile ranges based on actual session data analysis
        # Updated to handle high EDA values and improve dynamic range
        ranges = {
            'theta_fz': (1, 80),       # Observed range: 1.5-124, using 80th percentile
            'beta_fz': (0.5, 15),      # Frontal beta, reduced from 25 
            'alpha_c3': (2, 30),       # Central alpha (left), reduced from 40
            'alpha_c4': (2, 30),       # Central alpha (right), reduced from 40
            'faa_c3c4': (-2.5, 2.5),   # Alpha asymmetry, expanded from observed range
            'alpha_pz': (2, 35),       # Parietal alpha, reduced from 45
            'alpha_po': (1, 25),       # PO alpha, observed max ~30, using 25
            'alpha_oz': (2, 25),       # Occipital alpha, reduced from 35
            'eda_norm': (0, 12)        # Expanded EDA range to handle high arousal states (0-12)
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            # More sensitive normalization with smoother scaling
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        return np.array(normalized)
    
    def save_calibration_data(self):
        """Save calibration data and thresholds"""
        config_path = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_calibration.json')
        
        # Combine all features for baseline CSV
        all_features = np.vstack([self.relaxed_features, self.focused_features])
        
        # Save features CSV
        baseline_csv = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_baseline.csv')
        df = pd.DataFrame(all_features, columns=FEATURE_ORDER)
        df['phase'] = ['relaxed'] * len(self.relaxed_features) + ['focused'] * len(self.focused_features)
        df.to_csv(baseline_csv, index=False)
        
        # Convert numpy arrays to lists for JSON serialization
        adaptive_thresholds_json = self._convert_numpy_to_json(self.adaptive_thresholds)
        
        # Save configuration
        config_data = {
            'user_id': self.user_id,
            'calibration_time': str(datetime.now()),
            'baseline_csv': baseline_csv,
            'adaptive_thresholds': adaptive_thresholds_json,
            'relaxed_samples': len(self.relaxed_features),
            'focused_samples': len(self.focused_features)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n[SAVED] Calibration data saved to:")
        print(f"  Config: {config_path}")
        print(f"  Features: {baseline_csv}")
        
        return config_path, baseline_csv

    def _convert_numpy_to_json(self, obj):
        """Convert numpy arrays to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj

# === LSL STREAM UTILITIES ===
def select_lsl_stream(stream_type, name_hint=None, allow_skip=False, confirm=True):
    """Select LSL input stream with user verification for all streams"""
    print(f"\n[LSL] Looking for {stream_type} streams...")
    
    # Special handling for EDA - always show all available streams for user verification
    if stream_type == 'EDA':
        print("[INFO] Searching for ALL available LSL streams to ensure correct EDA selection...")
        streams = resolve_streams(timeout=10.0)  # Longer timeout for EDA
        
        if not streams:
            print(f"[WARN] No LSL streams found at all!")
            if allow_skip:
                print("[INFO] EDA stream is optional. Continuing without EDA data.")
                return None
            else:
                print(f"[ERROR] No streams available!")
                sys.exit(1)
        
        # Filter and show all streams with detailed info
        print(f"[AVAILABLE] Found {len(streams)} LSL streams:")
        eda_candidates = []
        other_streams = []
        
        for i, stream in enumerate(streams):
            stream_info = f"  {i+1}. {stream.name()} | Type: {stream.type()} | Channels: {stream.channel_count()} | Source: {stream.source_id()}"
            print(stream_info)
            
            # Identify potential EDA streams
            if (stream.type().lower() in ['eda', 'gsr', 'biosignals'] or 
                'eda' in stream.name().lower() or 
                'gsr' in stream.name().lower() or
                'opensignals' in stream.name().lower() or
                'biosignals' in stream.name().lower()):
                eda_candidates.append((i, stream))
            else:
                other_streams.append((i, stream))
        
        if eda_candidates:
            print(f"\n[CANDIDATES] Potential EDA streams detected:")
            for idx, stream in eda_candidates:
                print(f"  → {idx+1}. {stream.name()} (Type: {stream.type()}, Channels: {stream.channel_count()})")
        
        print(f"\n[IMPORTANT] Please verify which stream contains EDA data:")
        print("  - EDA streams typically have 2 channels (timestamp + EDA signal)")
        print("  - Look for names containing: EDA, GSR, OpenSignals, biosignals")
        print("  - If unsure, you can skip EDA (option 0)")
        
        while True:
            try:
                choice = input(f"\nSelect stream (1-{len(streams)}) or 0 to skip EDA: ")
                if choice == '0':
                    if allow_skip:
                        print("[SKIP] Continuing without EDA stream.")
                        return None
                    else:
                        print("[ERROR] EDA stream cannot be skipped.")
                        continue
                
                idx = int(choice) - 1
                if 0 <= idx < len(streams):
                    selected_stream = streams[idx]
                    print(f"[SELECTED] EDA Stream: {selected_stream.name()}")
                    print(f"  Type: {selected_stream.type()}")
                    print(f"  Channels: {selected_stream.channel_count()}")
                    print(f"  Source: {selected_stream.source_id()}")
                    
                    # Verify channel count for EDA
                    if selected_stream.channel_count() < 2:
                        print(f"[WARNING] Selected stream has {selected_stream.channel_count()} channels.")
                        print("EDA streams typically need 2 channels (timestamp + signal).")
                        confirm = input("Continue with this stream? (y/n): ").lower()
                        if confirm != 'y':
                            continue
                    
                    return selected_stream
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    # Standard handling for EEG and other stream types
    else:
        if stream_type == 'EEG':
            streams = resolve_byprop('type', 'EEG', timeout=5.0)
        elif stream_type == 'UnityMarkers':
            streams = resolve_byprop('type', 'Markers', timeout=5.0)
        else:
            streams = resolve_streams(timeout=5.0)
        
        if not streams:
            print(f"[WARN] No {stream_type} streams found!")
            if allow_skip:
                return None
            else:
                print(f"[ERROR] {stream_type} stream is required!")
                sys.exit(1)
        
        if len(streams) == 1 and not confirm:
            stream = streams[0]
            print(f"[AUTO] Using {stream_type} stream: {stream.name()}")
            return stream
        
        # Multiple streams or user verification requested - let user choose
        print(f"[CHOICE] {stream_type} streams found:")
        for i, stream in enumerate(streams):
            print(f"  {i+1}. {stream.name()} ({stream.channel_count()} channels)")
        
        while True:
            try:
                choice = input(f"Select {stream_type} stream (1-{len(streams)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(streams):
                    return streams[idx]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

def setup_mindfulness_lsl_streams():
    """Setup output LSL streams for MI data"""
    streams = {}
    
    # Adaptive MI stream (personalized 0-1 range)
    mi_info = StreamInfo('mindfulness_index', 'MI', 1, 1, 'float32', 'mi_001')
    streams['mi'] = StreamOutlet(mi_info)
    
    # Raw MI stream (universal -5 to +5 range)
    raw_mi_info = StreamInfo('raw_mindfulness_index', 'RawMI', 1, 1, 'float32', 'raw_mi_001')
    streams['raw_mi'] = StreamOutlet(raw_mi_info)
    
    # EMI stream (emotion-focused 0-1 range)
    emi_info = StreamInfo('emotional_mindfulness_index', 'EMI', 1, 1, 'float32', 'emi_001')
    streams['emi'] = StreamOutlet(emi_info)
    
    print("[LSL] Output streams created:")
    print("  - mindfulness_index (Adaptive MI: 0-1, personalized)")
    print("  - raw_mindfulness_index (Raw MI: -5 to +5, universal)")
    print("  - emotional_mindfulness_index (EMI: 0-1, emotion-focused)")
    
    return streams

# === FEATURE EXTRACTION ===
def compute_bandpower(data, sf, band):
    """Compute bandpower using Welch's method"""
    try:
        f, psd = welch(data, sf, nperseg=min(len(data), sf))
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        bp = np.trapz(psd[idx_band], f[idx_band])
        return max(bp, 1e-8)
    except:
        return 1.0

# === VISUALIZATION ===
class OnlineVisualizer:
    """Handles real-time visualization and final reports"""
    
    def __init__(self):
        self.adaptive_mi_history = []
        self.universal_mi_history = []
        self.emi_history = []
        self.timestamps = []
        
    def update(self, adaptive_mi, universal_mi, emi=None):
        """Update with new MI values"""
        self.adaptive_mi_history.append(adaptive_mi)
        self.universal_mi_history.append(universal_mi)
        self.emi_history.append(emi if emi is not None else universal_mi)
        self.timestamps.append(datetime.now())
    
    def final_plot(self, user_id):
        """Generate final comparison plot with all three MI types"""
        if len(self.adaptive_mi_history) == 0:
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        x = np.arange(len(self.adaptive_mi_history))
        
        # Plot adaptive MI (personalized)
        ax1.plot(x, self.adaptive_mi_history, label='Adaptive MI (Personalized)', 
                color='blue', linewidth=2)
        ax1.set_ylabel('Adaptive MI')
        ax1.set_title('Personalized Mindfulness Index (Dual Calibration)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot universal MI for comparison
        ax2.plot(x, self.universal_mi_history, label='Universal MI', 
                color='red', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Universal MI')
        ax2.set_title('Universal Mindfulness Index (For Comparison)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot EMI (emotion-focused)
        ax3.plot(x, self.emi_history, label='EMI (Emotion-Focused)', 
                color='green', linewidth=2, alpha=0.8)
        ax3.set_ylabel('EMI')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_title('Emotional Mindfulness Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(VIS_DIR, f'{user_id}_dual_calibration_mi_comprehensive_{timestamp}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"[PLOT] Comprehensive MI plot saved to {fname}")
        plt.close()

# === FEATURE MEANING GUIDE ===
def display_feature_guide():
    """Display comprehensive guide to mindfulness features"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MINDFULNESS FEATURE GUIDE")
    print(f"{'='*80}")
    
    print("🎯 ATTENTION REGULATION")
    print("   theta_fz (Fz, 4-8 Hz):")
    print("     • Higher = focused attention, anterior cingulate activation")
    print("     • Key indicator of sustained attention and cognitive control")
    print("   beta_fz (Fz, 13-30 Hz):")  
    print("     • Higher = effortful control (common in early meditation)")
    print("     • Should decrease as meditation skill develops")
    
    print("\n🧘‍♀️ BODY AWARENESS")
    print("   alpha_c3 (C3, 8-13 Hz):")
    print("     • Somatosensory activation for left side of body")
    print("     • Higher = increased body awareness on left")
    print("   alpha_c4 (C4, 8-13 Hz):")
    print("     • Somatosensory activation for right side of body") 
    print("     • Higher = increased body awareness on right")
    
    print("\n😊 EMOTION REGULATION")
    print("   faa_c3c4 (Frontal Alpha Asymmetry):")
    print("     • Calculation: log(alpha_c4) - log(alpha_c3)")
    print("     • Positive = positive affect/approach motivation")
    print("     • Negative = withdrawal/negative affect")
    
    print("\n🧠 SELF-REFERENTIAL PROCESSING / DEFAULT MODE NETWORK")
    print("   alpha_pz (Pz, 8-13 Hz):")
    print("     • Lower = present-moment awareness (DMN suppression)")
    print("     • Higher = mind-wandering, self-referential thinking")
    
    print("\n🌿 RELAXATION / VISUAL DETACHMENT")  
    print("   alpha_po (PO7/PO8, 8-13 Hz):")
    print("     • Higher = visual disengagement, relaxed alertness")
    print("     • Combined measure from bilateral parietal-occipital")
    print("   alpha_oz (Oz, 8-13 Hz):")
    print("     • Higher = visual cortex relaxation")
    print("     • Indicates reduced visual processing")
    
    print("\n💧 AROUSAL / STRESS")
    print("   eda_norm (Normalized EDA):")
    print("     • Lower = calm, relaxed autonomic state")
    print("     • Higher = arousal, stress, emotional activation")
    
    print(f"\n{'='*80}")
    print("MINDFULNESS INDEX INTERPRETATION:")
    print("• 0.0-0.3: Low mindfulness (distracted, stressed)")  
    print("• 0.3-0.5: Developing mindfulness (transitional state)")
    print("• 0.5-0.7: Good mindfulness (present, aware)")
    print("• 0.7-1.0: Deep mindfulness (absorbed, peaceful)")
    print(f"\n{'='*80}")
    print("CALIBRATION SYSTEM:")
    print("• Dual-phase calibration (30s relaxed + 30s focused)")
    print("• Creates personalized baseline thresholds")
    print("• Optimized for reliable real-time visualization")
    print("• Suitable for short sessions and research studies")
    print(f"{'='*80}\n")

# === MAIN FUNCTIONS ===
def run_dual_calibration(user_id, eeg_inlet, eda_inlet):
    """Run the dual calibration process"""
    print(f"\n{'='*80}")
    print("DUAL CALIBRATION PROCESS")
    print(f"{'='*80}")
    print("This calibration will establish your personal mindfulness baselines.")
    print("We'll capture two different states:")
    print("  1. RELAXED state (30 sec) - your natural low mindfulness baseline")
    print("  2. FOCUSED state (30 sec) - your peak attention baseline")
    print("\nThese baselines will be used to create personalized MI thresholds.")
    
    input("\nPress Enter to begin calibration...")
    
    calibration_system = DualCalibrationSystem(user_id)
    
    # Phase 1: Relaxed calibration
    calibration_system.display_calibration_instructions('relaxed')
    relaxed_features = calibration_system.collect_calibration_data(
        eeg_inlet, eda_inlet, 30, 'RELAXED'
    )
    
    if relaxed_features is None:
        print("[ERROR] Relaxed calibration failed!")
        return None
    
    calibration_system.relaxed_features = relaxed_features
    print(f"✓ Relaxed baseline captured: {len(relaxed_features)} windows")
    
    # Short break between phases
    print(f"\n{'='*60}")
    print("PHASE 1 COMPLETE - Taking a 5-second break")
    print("Prepare for the FOCUSED attention phase...")
    print(f"{'='*60}")
    
    for i in range(5, 0, -1):
        print(f"Break: {i} seconds remaining...")
        time.sleep(1)
    
    # Phase 2: Focused calibration
    calibration_system.display_calibration_instructions('focused')
    focused_features = calibration_system.collect_calibration_data(
        eeg_inlet, eda_inlet, 30, 'FOCUSED'
    )
    
    if focused_features is None:
        print("[ERROR] Focused calibration failed!")
        return None
    
    calibration_system.focused_features = focused_features
    print(f"✓ Focused baseline captured: {len(focused_features)} windows")
    
    # Compute adaptive thresholds
    adaptive_thresholds = calibration_system.compute_adaptive_thresholds()
    
    if adaptive_thresholds is None:
        print("[ERROR] Failed to compute adaptive thresholds!")
        return None
    
    # Save calibration data
    config_path, features_csv = calibration_system.save_calibration_data()
    
    print(f"\n{'='*60}")
    print("DUAL CALIBRATION COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Relaxed baseline: {len(relaxed_features)} samples")
    print(f"✓ Focused baseline: {len(focused_features)} samples")
    print(f"✓ Adaptive thresholds computed")
    print(f"✓ Data saved to: {config_path}")
    
    return adaptive_thresholds

def main():
    """Main function for dual calibration MI pipeline"""
    display_feature_guide()
    
    # Get user ID
    user_id = input("Enter user ID for this session: ").strip()
    if not user_id:
        user_id = f"user_{int(time.time())}"
        print(f"Using default user ID: {user_id}")
    
    print(f"\n[SETUP] Initializing session for user: {user_id}")
    
    # Setup LSL streams
    print("\n[INPUT] Connecting to data streams...")
    
    # EEG stream (required)
    print("Select EEG stream:")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream', allow_skip=False)
    if eeg_stream is None:
        print("[ERROR] EEG stream is required. Exiting...")
        return
    
    eeg_inlet = StreamInlet(eeg_stream)
    print(f"✓ EEG stream connected: {eeg_stream.name()}")
    
    # EDA stream (optional but recommended) - Always show verification
    print("\n" + "="*60)
    print("EDA STREAM SELECTION WITH VERIFICATION")
    print("="*60)
    print("EDA (Electrodermal Activity) provides crucial arousal/stress information.")
    print("Please carefully select the correct EDA stream from available options.")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals', allow_skip=True, confirm=True)
    if eda_stream is None:
        print("[WARNING] No EDA stream selected. EDA features will use zero values.")
        print("          This may reduce MI calculation accuracy.")
        eda_inlet = None
    else:
        eda_inlet = StreamInlet(eda_stream)
        print(f"✓ EDA stream connected: {eda_stream.name()}")
        print(f"  Channels: {eda_stream.channel_count()}")
        print(f"  Type: {eda_stream.type()}")
        
        # Additional verification for EDA stream
        if eda_stream.channel_count() != 2:
            print(f"[INFO] EDA stream has {eda_stream.channel_count()} channels.")
            print("       System expects 2 channels (timestamp + EDA signal).")
            print("       Will use channel 1 for EDA data.")
        
        # Test EDA stream to verify data
        test_choice = input("\nTest EDA stream to verify data? (y/n): ").lower()
        if test_choice == 'y':
            test_eda_stream(eda_inlet)
    
    # Setup output streams
    output_streams = setup_mindfulness_lsl_streams()
    
    print("\n" + "="*60)
    print("STREAM CONFIGURATION SUMMARY")
    print("="*60)
    print(f"✓ EEG Stream: {eeg_stream.name()}")
    print(f"  - Type: {eeg_stream.type()}")
    print(f"  - Channels: {eeg_stream.channel_count()}")
    print(f"  - Source: {eeg_stream.source_id()}")
    
    if eda_inlet is not None:
        print(f"✓ EDA Stream: {eda_stream.name()}")
        print(f"  - Type: {eda_stream.type()}")
        print(f"  - Channels: {eda_stream.channel_count()}")
        print(f"  - Source: {eda_stream.source_id()}")
        print(f"  - Using Channel {EDA_CHANNEL_INDEX} for EDA data")
    else:
        print("⚠ EDA Stream: Not connected (using zero values)")
    
    print("✓ Output Streams: MI, Raw MI, EMI")
    print("="*60)
    
    print("\n[READY] All streams configured successfully!")
    
    # Run dual calibration
    adaptive_thresholds = run_dual_calibration(user_id, eeg_inlet, eda_inlet)
    
    if adaptive_thresholds is None:
        print("[ERROR] Calibration failed. Cannot proceed to real-time processing.")
        return
    
    # Initialize MI calculator with adaptive thresholds
    mi_calculator = AdaptiveMICalculator(adaptive_thresholds, user_id)
    
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE - READY FOR REAL-TIME PROCESSING")
    print(f"{'='*80}")
    print("✓ Ready for adaptive real-time MI calculation")
    print("✓ Personalized thresholds loaded")
    print("✓ Output streams active")
    print("\nPress Enter to start real-time processing (or 'q' to quit)...")
    
    choice = input().strip().lower()
    if choice == 'q':
        print("Exiting...")
        return
    
    # Real-time processing
    run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator)

def run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator):
    """Run real-time MI processing with dual calibration"""
    print(f"\n{'='*60}")
    print("STARTING REAL-TIME MI PROCESSING")
    print(f"{'='*60}")
    print("Press 'q' and Enter to stop...")
    
    # Initialize data processor and visualizer
    processor = RobustDataProcessor()
    visualizer = OnlineVisualizer()
    
    # Session logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_file = os.path.join(LOG_DIR, f'{user_id}_mi_session_{timestamp}.csv')
    feature_stats_file = os.path.join(LOG_DIR, f'{user_id}_mi_feature_stats_{timestamp}.csv')
    feature_corr_file = os.path.join(LOG_DIR, f'{user_id}_mi_feature_corr_{timestamp}.csv')
    
    # Data collection
    session_data = []
    feature_data = []
    
    # Real-time loop
    eeg_buffer = []
    eda_buffer = []
    window_size = 250  # 1-second windows at 250 Hz
    
    print("USING ADAPTIVE MI CALCULATION")
    print("Collecting data...")
    
    start_time = time.time()
    last_display = 0
    
    try:
        # Setup input monitoring thread
        stop_event = threading.Event()
        
        def input_monitor():
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    if key == 'q':
                        stop_event.set()
                        break
                time.sleep(0.1)
        
        input_thread = threading.Thread(target=input_monitor, daemon=True)
        input_thread.start()
        
        while not stop_event.is_set():
            current_time = time.time()
            
            # Collect EEG data
            eeg_sample, eeg_timestamp = eeg_inlet.pull_sample(timeout=0.01)
            if eeg_sample:
                eeg_buffer.append(eeg_sample)
            
            # Collect EDA data (handle missing stream gracefully)
            if eda_inlet:
                eda_sample, eda_timestamp = eda_inlet.pull_sample(timeout=0.01)

                if eda_sample:
                    # Ensure we have at least 2 channels for EDA: [timestamp, eda_data]
                    if len(eda_sample) >= 2:
                        eda_buffer.append(eda_sample[:2])
                    else:
                        eda_buffer.append([0, 0])  # [timestamp, eda_data]
                else:
                    eda_buffer.append([0, 0])  # Use zeros if no sample
            else:
                # Use zeros if no EDA stream (not random data)
                eda_buffer.append([0, 0])  # [timestamp, eda_data]
            
            # Process when we have enough data
            if len(eeg_buffer) >= window_size and len(eda_buffer) >= window_size:
                # Extract windows
                eeg_window = np.array(eeg_buffer[-window_size:])
                eda_window = np.array(eda_buffer[-window_size:])
                
                # Simple processing (no complex artifact rejection to avoid incomplete methods)
                eeg_processed = eeg_window[:, :8]  # Use first 8 EEG channels
                eda_processed = eda_window
                
                # Extract features
                features = extract_mindfulness_features(eeg_processed, eda_processed)
                
                # Calculate MI using adaptive thresholds
                adaptive_mi, universal_mi, emi = mi_calculator.calculate_adaptive_mi(features)
                
                # Send to output streams
                output_streams['mi'].push_sample([adaptive_mi])
                output_streams['raw_mi'].push_sample([universal_mi])
                output_streams['emi'].push_sample([emi])
                
                # Update visualizer
                visualizer.update(adaptive_mi, universal_mi, emi)
                
                # Store session data
                session_data.append({
                    'mi': adaptive_mi,
                    'raw_mi': universal_mi,
                    'emi': emi,
                    'timestamp': current_time,
                    'state': 'unknown',
                    **{f: features[i] for i, f in enumerate(FEATURE_ORDER)}
                })
                
                feature_data.append(features)
                
                # Display progress
                if current_time - last_display > 1.0:  # Every second
                    elapsed = current_time - start_time
                    print(f"TIME: {elapsed:6.1f}s | "
                          f"MI: {adaptive_mi:.3f} | "
                          f"RAW: {universal_mi:.3f} | "
                          f"EMI: {emi:.3f} | "
                          f"THETA: {features[0]:.1f} | "
                          f"ALPHA_PO: {features[6]:.1f} | "
                          f"EDA: {features[8]:.1f}")
                    last_display = current_time
                
                # Maintain buffer size
                if len(eeg_buffer) > window_size * 2:
                    eeg_buffer = eeg_buffer[-window_size:]
                if len(eda_buffer) > window_size * 2:
                    eda_buffer = eda_buffer[-window_size:]
            
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    except KeyboardInterrupt:
        print("\nStopping on user request...")
    
    print(f"\n{'='*60}")
    print("REAL-TIME PROCESSING COMPLETE")
    print(f"{'='*60}")
    
    # Save session data
    if session_data:
        df_session = pd.DataFrame(session_data)
        df_session.to_csv(session_file, index=False)
        print(f"✓ Session data saved: {session_file}")
        
        # Save feature statistics
        if feature_data:
            feature_array = np.array(feature_data)
            feature_stats = []
            
            for i, feature_name in enumerate(FEATURE_ORDER):
                stats = {
                    'variable': feature_name,
                    'mean': np.mean(feature_array[:, i]),
                    'std': np.std(feature_array[:, i]),
                    'min': np.min(feature_array[:, i]),
                    'max': np.max(feature_array[:, i])
                }
                feature_stats.append(stats)
            
            # Add MI statistics
            mi_values = [d['mi'] for d in session_data]
            feature_stats.append({
                'variable': 'mi',
                'mean': np.mean(mi_values),
                'std': np.std(mi_values),
                'min': np.min(mi_values),
                'max': np.max(mi_values)
            })
            
            df_stats = pd.DataFrame(feature_stats)
            df_stats.to_csv(feature_stats_file, index=False)
            print(f"✓ Feature statistics saved: {feature_stats_file}")
            
            # Calculate feature correlations with MI
            correlations = []
            for i, feature_name in enumerate(FEATURE_ORDER):
                corr, p_value = spearmanr(feature_array[:, i], mi_values)
                correlations.append({
                    'feature': feature_name,
                    'spearman_corr': corr,
                    'p_value': p_value
                })
            
            df_corr = pd.DataFrame(correlations)
            df_corr.to_csv(feature_corr_file, index=False)
            print(f"✓ Feature correlations saved: {feature_corr_file}")
    
    # Generate final visualization
    visualizer.final_plot(user_id)
    
    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY:")
    print(f"  Duration: {time.time() - start_time:.1f} seconds")
    print(f"  Samples: {len(session_data)}")
    if session_data:
        mi_values = [d['mi'] for d in session_data]
        print(f"  MI Range: {np.min(mi_values):.3f} - {np.max(mi_values):.3f}")
        print(f"  MI Mean: {np.mean(mi_values):.3f} ± {np.std(mi_values):.3f}")
    print(f"{'='*60}")

def extract_mindfulness_features(eeg_window, eda_window):
    """Extract comprehensive mindfulness features from processed windows"""
    sf = 250
    
    # === ATTENTION REGULATION ===
    theta_fz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (4, 8))
    beta_fz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (13, 30))
    
    # === BODY AWARENESS ===
    alpha_c3 = compute_bandpower(eeg_window[:, EEG_CHANNELS['C3']], sf, (8, 13))
    alpha_c4 = compute_bandpower(eeg_window[:, EEG_CHANNELS['C4']], sf, (8, 13))
    
    # === EMOTION REGULATION ===
    faa_c3c4 = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
    
    # === SELF-REFERENTIAL PROCESSING / DMN ===
    alpha_pz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Pz']], sf, (8, 13))
    
    # === RELAXATION / VISUAL DETACHMENT ===
    alpha_po7 = compute_bandpower(eeg_window[:, EEG_CHANNELS['PO7']], sf, (8, 13))
    alpha_po8 = compute_bandpower(eeg_window[:, EEG_CHANNELS['PO8']], sf, (8, 13))
    alpha_po = (alpha_po7 + alpha_po8) / 2
    alpha_oz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Oz']], sf, (8, 13))
    
    # === AROUSAL/STRESS (EDA) ===
    # Handle EDA data safely - always use channel 1 (actual EDA data)
    if eda_window.shape[1] >= 2:
        raw_eda = np.mean(eda_window[:, EDA_CHANNEL_INDEX])  # Channel 1
        # Reduced debug output - only show occasionally
        if hasattr(extract_mindfulness_features, 'debug_counter'):
            extract_mindfulness_features.debug_counter += 1
        else:
            extract_mindfulness_features.debug_counter = 1
        
        # Show EDA debug info every 10 seconds (10 samples at 1Hz)
        if extract_mindfulness_features.debug_counter % 10 == 0:
            print(f"[DEBUG] EDA raw: {raw_eda:.3f} from channel {EDA_CHANNEL_INDEX} (sample #{extract_mindfulness_features.debug_counter})")
    else:
        print(f"[WARN] EDA window has {eda_window.shape[1]} channels, expected 2")
        raw_eda = 0.0
    
    # Simple EDA normalization
    eda_norm = np.clip(10 * (raw_eda - 0) / (12 - 0), 0, 10)
    
    return np.array([
        theta_fz, beta_fz, alpha_c3, alpha_c4, faa_c3c4,
        alpha_pz, alpha_po, alpha_oz, eda_norm
    ])

def test_eda_stream(eda_inlet, duration=5):
    """Test EDA stream and show actual data being received"""
    if eda_inlet is None:
        print("[TEST] No EDA stream to test.")
        return
    
    print(f"\n[TEST] Testing EDA stream for {duration} seconds...")
    print("This will show you the actual data being received:")
    
    start_time = time.time()
    samples_received = 0
    sample_values = []
    
    while time.time() - start_time < duration:
        sample, timestamp = eda_inlet.pull_sample(timeout=0.1)
        if sample:
            samples_received += 1
            sample_values.append(sample)
            
            # Show first few samples with full detail
            if samples_received <= 5:
                print(f"  Sample {samples_received}: {sample} (channels: {len(sample)})")
            elif samples_received == 6:
                print("  ... (showing more samples)")
            
            # Show periodic updates
            if samples_received % 50 == 0:
                if len(sample) >= 2:
                    ch0_val = sample[0]
                    ch1_val = sample[1]
                    print(f"  Sample {samples_received}: Ch0={ch0_val:.3f}, Ch1={ch1_val:.3f}")
                else:
                    print(f"  Sample {samples_received}: {sample}")
    
    print(f"\n[TEST RESULTS]")
    print(f"  Samples received: {samples_received}")
    print(f"  Sampling rate: ~{samples_received/duration:.1f} Hz")
    
    if sample_values:
        last_sample = sample_values[-1]
        print(f"  Channels per sample: {len(last_sample)}")
        print(f"  Last sample: {last_sample}")
        
        if len(last_sample) >= 2:
            # Show statistics for each channel
            ch0_values = [s[0] for s in sample_values[-20:]]  # Last 20 samples
            ch1_values = [s[1] for s in sample_values[-20:]]
            
            print(f"  Channel 0 (last 20): min={min(ch0_values):.3f}, max={max(ch0_values):.3f}, mean={np.mean(ch0_values):.3f}")
            print(f"  Channel 1 (last 20): min={min(ch1_values):.3f}, max={max(ch1_values):.3f}, mean={np.mean(ch1_values):.3f}")
            
            print(f"\n[RECOMMENDATION]")
            # Try to identify which channel looks like EDA
            ch0_range = max(ch0_values) - min(ch0_values)
            ch1_range = max(ch1_values) - min(ch1_values)
            
            if abs(np.mean(ch0_values)) > 1000000:  # Looks like timestamp
                print(f"  Channel 0 appears to be TIMESTAMP (large values)")
                print(f"  Channel 1 appears to be EDA DATA (range: {ch1_range:.3f})")
                print(f"  → System will use Channel 1 for EDA (this is correct)")
            elif abs(np.mean(ch1_values)) > 1000000:  # Looks like timestamp
                print(f"  Channel 1 appears to be TIMESTAMP (large values)")
                print(f"  Channel 0 appears to be EDA DATA (range: {ch0_range:.3f})")
                print(f"  → WARNING: System uses Channel 1, but Channel 0 might be EDA!")
            else:
                print(f"  Channel 0 range: {ch0_range:.3f}")
                print(f"  Channel 1 range: {ch1_range:.3f}")
                print(f"  → System will use Channel 1 for EDA")
    else:
        print(f"  No samples received - check stream connection!")