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
import json
from datetime import datetime
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream
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
        """Process EEG window with peak suppression and artifact rejection"""
        if len(eeg_window) == 0:
            return np.zeros((self.window_size, 8))
        
        eeg_window = np.array(eeg_window)
        if eeg_window.ndim == 1:
            eeg_window = eeg_window.reshape(-1, 8)
        
        processed = np.copy(eeg_window)
        
        # Apply median filtering to suppress peaks
        for ch in range(processed.shape[1]):
            processed[:, ch] = self.median_filter_1d(processed[:, ch], self.median_filter_size)
        
        # Detect and handle outliers
        outlier_mask = self.detect_outliers_robust(processed)
        
        if np.any(outlier_mask):
            # Replace outliers with median values
            for ch in range(processed.shape[1]):
                ch_data = processed[:, ch]
                if len(ch_data[~outlier_mask]) > 0:
                    median_val = np.median(ch_data[~outlier_mask])
                    processed[outlier_mask, ch] = median_val
        
        # Ensure we have the right shape
        if processed.shape[0] != self.window_size:
            if processed.shape[0] > self.window_size:
                processed = processed[:self.window_size, :]
            else:
                # Pad with last values if too short
                padding = np.tile(processed[-1, :], (self.window_size - processed.shape[0], 1))
                processed = np.vstack([processed, padding])
        
        return processed
    
    def process_eda_window(self, eda_window):
        """Process EDA window with peak suppression and artifact rejection"""
        if len(eda_window) == 0:
            return np.zeros((self.window_size, 2))
        
        eda_window = np.array(eda_window)
        if eda_window.ndim == 1:
            eda_window = eda_window.reshape(-1, 2)
        
        processed = np.copy(eda_window)
        
        # Apply median filtering to suppress peaks
        for ch in range(processed.shape[1]):
            processed[:, ch] = self.median_filter_1d(processed[:, ch], self.median_filter_size)
        
        # Detect and handle outliers
        outlier_mask = self.detect_outliers_robust(processed)
        
        if np.any(outlier_mask):
            # Replace outliers with median values
            for ch in range(processed.shape[1]):
                ch_data = processed[:, ch]
                if len(ch_data[~outlier_mask]) > 0:
                    median_val = np.median(ch_data[~outlier_mask])
                    processed[outlier_mask, ch] = median_val
        
        # Ensure we have the right shape
        if processed.shape[0] != self.window_size:
            if processed.shape[0] > self.window_size:
                processed = processed[:self.window_size, :]
            else:
                # Pad with last values if too short
                padding = np.tile(processed[-1, :], (self.window_size - processed.shape[0], 1))
                processed = np.vstack([processed, padding])
        
        return processed

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
        """Robust EDA normalization using quantile-based scaling"""
        # Population-based robust quantiles for EDA
        q5, q95 = 2, 12  # Based on typical EDA ranges
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
        
        # Create adaptive mapping
        self.adaptive_thresholds = {
            'relaxed_baseline': {
                'mi_mean': np.mean(relaxed_mi),
                'mi_std': np.std(relaxed_mi),
                'mi_range': [np.min(relaxed_mi), np.max(relaxed_mi)],
                'features': relaxed_stats
            },
            'focused_baseline': {
                'mi_mean': np.mean(focused_mi),
                'mi_std': np.std(focused_mi),
                'mi_range': [np.min(focused_mi), np.max(focused_mi)],
                'features': focused_stats
            },
            'adaptive_mapping': {
                'low_threshold': np.mean(relaxed_mi),
                'high_threshold': np.mean(focused_mi),
                'dynamic_range': np.mean(focused_mi) - np.mean(relaxed_mi),
                'calibration_time': str(datetime.now())
            }
        }
        
        print(f"\n[RESULTS] Adaptive Thresholds Computed:")
        print(f"  Relaxed MI: {np.mean(relaxed_mi):.3f} ± {np.std(relaxed_mi):.3f}")
        print(f"  Focused MI: {np.mean(focused_mi):.3f} ± {np.std(focused_mi):.3f}")
        print(f"  Dynamic Range: {self.adaptive_thresholds['adaptive_mapping']['dynamic_range']:.3f}")
        
        return self.adaptive_thresholds
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation for calibration using comprehensive mindfulness features"""
        # Updated weights for 9-feature mindfulness model
        # Based on neuroscience literature for meditation states
        weights = np.array([
            0.25,   # theta_fz: Strong attention component
            -0.05,  # beta_fz: Negative for relaxed states (too much effort)
            0.15,   # alpha_c3: Body awareness (left)
            0.15,   # alpha_c4: Body awareness (right)  
            0.10,   # faa_c3c4: Emotional balance
            -0.20,  # alpha_pz: Negative for DMN suppression (lower alpha = better)
            0.20,   # alpha_po: Visual detachment/relaxation
            0.15,   # alpha_oz: Occipital relaxation
            -0.15   # eda_norm: Negative for low arousal (calm state)
        ])
        
        # Normalize features to 0-10 range
        normalized_features = self.normalize_features_for_mi(features)
        
        # Calculate weighted sum
        weighted_sum = np.dot(normalized_features, weights)
        
        # Map to 0-1 range with improved scaling for 9 features
        # Adjust baseline since we have different feature count and weights
        mi = 0.1 + 0.8 * ((weighted_sum + 2.0) / 4.0)  # Shifted and scaled for new range
        return np.clip(mi, 0.1, 0.9)
    
    def normalize_features_for_mi(self, features):
        """Normalize comprehensive mindfulness features for MI calculation"""
        # Updated robust quantile ranges for all 9 features
        # Based on empirical EEG/EDA data analysis
        ranges = {
            'theta_fz': (1, 50),      # Frontal theta power
            'beta_fz': (0.5, 25),     # Frontal beta power  
            'alpha_c3': (2, 40),      # Central alpha (left)
            'alpha_c4': (2, 40),      # Central alpha (right)
            'faa_c3c4': (-2.0, 2.0),  # Alpha asymmetry ratio
            'alpha_pz': (3, 45),      # Parietal alpha
            'alpha_po': (1, 30),      # Parietal-occipital alpha
            'alpha_oz': (2, 35),      # Occipital alpha
            'eda_norm': (2, 12)       # Normalized EDA
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
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
        
        # Save configuration
        config_data = {
            'user_id': self.user_id,
            'calibration_time': str(datetime.now()),
            'baseline_csv': baseline_csv,
            'adaptive_thresholds': self.adaptive_thresholds,
            'relaxed_samples': len(self.relaxed_features),
            'focused_samples': len(self.focused_features)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n[SAVED] Calibration data saved to:")
        print(f"  Config: {config_path}")
        print(f"  Features: {baseline_csv}")
        
        return config_path, baseline_csv

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
        # Updated weights for 9-feature mindfulness model
        weights = np.array([
            0.25,   # theta_fz: Strong attention component
            -0.05,  # beta_fz: Negative for relaxed states
            0.15,   # alpha_c3: Body awareness (left)
            0.15,   # alpha_c4: Body awareness (right)  
            0.10,   # faa_c3c4: Emotional balance
            -0.20,  # alpha_pz: Negative for DMN suppression
            0.20,   # alpha_po: Visual detachment/relaxation
            0.15,   # alpha_oz: Occipital relaxation
            -0.15   # eda_norm: Negative for low arousal
        ])
        
        normalized_features = self.normalize_features_for_mi(features)
        weighted_sum = np.dot(normalized_features, weights)
        mi = 0.1 + 0.8 * ((weighted_sum + 2.0) / 4.0)
        return np.clip(mi, 0.1, 0.9)
    
    def normalize_features_for_mi(self, features):
        """Normalize comprehensive mindfulness features for MI calculation"""
        ranges = {
            'theta_fz': (1, 50),      # Frontal theta power
            'beta_fz': (0.5, 25),     # Frontal beta power  
            'alpha_c3': (2, 40),      # Central alpha (left)
            'alpha_c4': (2, 40),      # Central alpha (right)
            'faa_c3c4': (-2.0, 2.0),  # Alpha asymmetry ratio
            'alpha_pz': (3, 45),      # Parietal alpha
            'alpha_po': (1, 30),      # Parietal-occipital alpha
            'alpha_oz': (2, 35),      # Occipital alpha
            'eda_norm': (2, 12)       # Normalized EDA
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        return np.array(normalized)

# === LSL STREAM UTILITIES ===
def select_lsl_stream(stream_type, name_hint=None, allow_skip=False, confirm=True):
    """Select an LSL stream with user confirmation"""
    print(f"\nResolving {stream_type} streams...")
    streams = resolve_stream('type', stream_type, timeout=5.0)
    
    if not streams:
        print(f"No {stream_type} streams found.")
        if allow_skip:
            return None
        else:
            input("Please start your LSL stream and press Enter to retry...")
            return select_lsl_stream(stream_type, name_hint, allow_skip, confirm)
    
    if len(streams) == 1:
        stream = streams[0]
        if confirm:
            print(f"Found {stream_type} stream: {stream.name()} ({stream.channel_count()} channels)")
            if input("Use this stream? (y/n): ").lower() != 'y':
                return None
        return stream
    
    # Multiple streams - let user choose
    print(f"Found {len(streams)} {stream_type} streams:")
    for i, stream in enumerate(streams):
        print(f"  {i+1}. {stream.name()} ({stream.channel_count()} channels)")
    
    while True:
        try:
            choice = int(input("Select stream number: ")) - 1
            if 0 <= choice < len(streams):
                return streams[choice]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

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
    config_path, baseline_csv = calibration_system.save_calibration_data()
    
    print(f"\n{'='*80}")
    print("DUAL CALIBRATION COMPLETE!")
    print(f"{'='*80}")
    print("✓ Personalized MI thresholds have been established")
    print("✓ Calibration data saved for future sessions")
    print("✓ Ready for adaptive real-time MI calculation")
    print(f"{'='*80}")
    
    # Display calibration results and pause for user confirmation
    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS SUMMARY")
    print(f"{'='*60}")
    mapping = adaptive_thresholds['adaptive_mapping']
    print(f"🎯 Your Personal MI Baselines:")
    print(f"   • Relaxed State MI: {mapping['low_threshold']:.3f}")
    print(f"   • Focused State MI: {mapping['high_threshold']:.3f}")
    print(f"   • Dynamic Range: {mapping['dynamic_range']:.3f}")
    print(f"   • Calibration Quality: {'Excellent' if mapping['dynamic_range'] > 0.3 else 'Good' if mapping['dynamic_range'] > 0.15 else 'Fair'}")
    
    if mapping['dynamic_range'] < 0.15:
        print(f"\n⚠️  NOTE: Low dynamic range detected. The system will apply")
        print(f"   sensitivity enhancement during real-time processing.")
    
    print(f"\n{'='*60}")
    print("WHAT HAPPENS NEXT:")
    print("• Real-time MI processing will begin")
    print("• You'll see 3 MI values: Adaptive, Universal, and EMI")
    print("• All features will be displayed in real-time")
    print("• Session data will be saved and analyzed")
    print("• Press Enter during processing to stop and generate reports")
    print(f"{'='*60}")
    
    # Wait for user confirmation
    proceed = input("\nReady to start real-time MI processing? (y/n, default: y): ").strip().lower()
    if proceed == 'n':
        print("Session paused. You can restart the process anytime.")
        return None, None
    
    print("\n🚀 Starting real-time MI processing...")
    
    return adaptive_thresholds, config_path

def load_user_calibration(user_id):
    """Load existing user calibration data"""
    config_path = os.path.join(USER_CONFIG_DIR, f'{user_id}_dual_calibration.json')
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        adaptive_thresholds = config_data.get('adaptive_thresholds')
        if adaptive_thresholds is None:
            return None
        
        print(f"[LOADED] Existing calibration for user {user_id}")
        print(f"  Calibrated: {config_data.get('calibration_time', 'Unknown')}")
        print(f"  Relaxed samples: {config_data.get('relaxed_samples', 0)}")
        print(f"  Focused samples: {config_data.get('focused_samples', 0)}")
        
        return adaptive_thresholds
    
    except Exception as e:
        print(f"[ERROR] Failed to load calibration: {e}")
        return None

def main():
    """Main function with comprehensive mindfulness feature support"""
    print("\n" + "="*80)
    print("DUAL CALIBRATION MINDFULNESS INDEX PIPELINE")
    print("="*80)
    print("This version uses comprehensive mindfulness features:")
    print("• Attention Regulation (theta_fz, beta_fz)")
    print("• Body Awareness (alpha_c3, alpha_c4)")  
    print("• Emotion Regulation (faa_c3c4)")
    print("• DMN Suppression (alpha_pz)")
    print("• Visual Detachment (alpha_po, alpha_oz)")
    print("• Arousal/Stress (eda_norm)")
    print("\n🎯 FEATURES:")
    print("• Dual calibration for personalized thresholds")
    print("• Stable real-time visualization")
    print("• Three MI outputs: Adaptive, Universal, and EMI")
    print("• Optimized for short sessions and small user groups")
    print("="*80)
    
    # Display comprehensive feature guide
    show_guide = input("Show detailed feature guide? (y/n, default: n): ").strip().lower()
    if show_guide == 'y':
        display_feature_guide()
    
    # Get user ID
    user_id = input("Enter user ID for this session: ").strip()
    if not user_id:
        print("User ID is required!")
        return
    
    # Check for existing calibration
    existing_calibration = load_user_calibration(user_id)
    
    if existing_calibration is not None:
        print(f"\n[FOUND] Existing calibration for user {user_id}")
        use_existing = input("Use existing calibration? (y/n, default: y): ").strip().lower()
        
        if use_existing != 'n':
            adaptive_thresholds = existing_calibration
            print("[INFO] Using existing calibration data")
        else:
            adaptive_thresholds = None
    else:
        adaptive_thresholds = None
        print(f"[INFO] No existing calibration found for user {user_id}")
    
    # Setup LSL streams
    print(f"\n{'='*60}")
    print("LSL STREAM SETUP")
    print(f"{'='*60}")
    
    print("Setting up EEG stream...")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream', allow_skip=True)
    eeg_inlet = StreamInlet(eeg_stream) if eeg_stream is not None else None
    
    print("Setting up EDA stream...")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals', allow_skip=True)
    eda_inlet = StreamInlet(eda_stream) if eda_stream is not None else None
    
    if eeg_inlet is None and eda_inlet is None:
        print("[ERROR] No input streams available!")
        return
    
    # Validate EEG channel count
    if eeg_stream is not None:
        required_channels = max(EEG_CHANNELS.values()) + 1  # 0-based indexing
        available_channels = eeg_stream.channel_count()
        
        print(f"\n[VALIDATION] EEG Channel Check:")
        print(f"  Required channels: {required_channels} (for comprehensive features)")
        print(f"  Available channels: {available_channels}")
        
        if available_channels < required_channels:
            print(f"\n[WARNING] Insufficient EEG channels!")
            print(f"Required mapping: {EEG_CHANNELS}")
            print("Some features may not be available or may use substitute channels.")
            
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Session cancelled. Please ensure proper EEG setup.")
                return
        else:
            print("✓ Channel count sufficient for comprehensive feature extraction")

    # Run calibration if needed
    if adaptive_thresholds is None:
        print(f"\n[REQUIRED] Running dual calibration for user {user_id}")
        adaptive_thresholds, config_path = run_dual_calibration(user_id, eeg_inlet, eda_inlet)
        
        if adaptive_thresholds is None:
            print("[ERROR] Calibration failed! Cannot proceed.")
            return
    
    # Setup output streams
    outlets = setup_mindfulness_lsl_streams()
    
    # Initialize components
    data_processor = RobustDataProcessor()
    mi_calculator = AdaptiveMICalculator(adaptive_thresholds, user_id)
    visualizer = OnlineVisualizer()
    
    # Display calibration information
    if adaptive_thresholds is not None:
        print(f"\n{'='*60}")
        print("CALIBRATION STATUS")
        print(f"{'='*60}")
        mapping = adaptive_thresholds['adaptive_mapping']
        print(f"✓ Using personalized calibration for {user_id}")
        print(f"  Relaxed baseline: {mapping['low_threshold']:.3f}")
        print(f"  Focused baseline: {mapping['high_threshold']:.3f}")
        print(f"  Dynamic range: {mapping['dynamic_range']:.3f}")
        quality = "Excellent" if mapping['dynamic_range'] > 0.2 else "Good" if mapping['dynamic_range'] > 0.1 else "Fair"
        print(f"  Calibration quality: {quality}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("USING UNIVERSAL MI CALCULATION")
        print(f"{'='*60}")
    
    # Real-time processing
    print(f"\n{'='*60}")
    print("REAL-TIME MI PROCESSING")
    print(f"{'='*60}")
    print("• Using personalized adaptive thresholds")
    print("• Peak suppression and artifact rejection enabled")
    print("• Optimized for stable real-time visualization")
    print("• Outputting 3 MI values: Adaptive, Universal, EMI")
    print("• Press Enter to stop and generate report")
    print(f"{'='*60}")
    
    # Setup stop mechanism
    stop_flag = {'stop': False}
    
    def wait_for_exit():
        input("\nPress Enter to stop session and generate report...\n")
        stop_flag['stop'] = True
    
    stop_thread = threading.Thread(target=wait_for_exit)
    stop_thread.daemon = True
    stop_thread.start()
    
    # Processing loop
    window_size = 250
    eeg_buffer, eda_buffer = [], []
    session_data = []
    
    print("Starting real-time processing...")
    session_start = time.time()
    
    while not stop_flag['stop']:
        # Collect window of data
        eeg_window, eda_window = [], []
        
        for i in range(window_size):
            # EEG sample
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample is not None:
                    eeg_window.append(np.array(eeg_sample[:8]))
                else:
                    if len(eeg_window) > 0:
                        eeg_window.append(eeg_window[-1])
                    else:
                        eeg_window.append(np.zeros(8))
            else:
                eeg_window.append(np.zeros(8))
            
            # EDA sample
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample is not None:
                    eda_window.append(np.array(eda_sample[:2]))
                else:
                    if len(eda_window) > 0:
                        eda_window.append(eda_window[-1])
                    else:
                        eda_window.append(np.zeros(2))
            else:
                eda_window.append(np.zeros(2))
        
        # Process data with peak suppression
        eeg_processed = data_processor.process_eeg_window(eeg_window)
        eda_processed = data_processor.process_eda_window(eda_window)
        
        # Extract comprehensive mindfulness features
        sf = 250
        
        # === ATTENTION REGULATION ===
        theta_fz = compute_bandpower(eeg_processed[:, EEG_CHANNELS['Fz']], sf, (4, 8))
        beta_fz = compute_bandpower(eeg_processed[:, EEG_CHANNELS['Fz']], sf, (13, 30))
        
        # === BODY AWARENESS ===
        alpha_c3 = compute_bandpower(eeg_processed[:, EEG_CHANNELS['C3']], sf, (8, 13))
        alpha_c4 = compute_bandpower(eeg_processed[:, EEG_CHANNELS['C4']], sf, (8, 13))
        
        # === EMOTION REGULATION ===
        faa_c3c4 = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
        
        # === DMN SUPPRESSION ===
        alpha_pz = compute_bandpower(eeg_processed[:, EEG_CHANNELS['Pz']], sf, (8, 13))
        
        # === VISUAL DETACHMENT/RELAXATION ===
        alpha_po7 = compute_bandpower(eeg_processed[:, EEG_CHANNELS['PO7']], sf, (8, 13))
        alpha_po8 = compute_bandpower(eeg_processed[:, EEG_CHANNELS['PO8']], sf, (8, 13))
        alpha_po = (alpha_po7 + alpha_po8) / 2
        alpha_oz = compute_bandpower(eeg_processed[:, EEG_CHANNELS['Oz']], sf, (8, 13))
        
        # === AROUSAL/STRESS ===
        raw_eda = np.mean(eda_processed[:, EDA_CHANNEL_INDEX])
        q5, q95 = 2, 12
        eda_norm = np.clip(10 * (raw_eda - q5) / (q95 - q5), 0, 10)
        
        # Comprehensive feature vector
        features = np.array([
            theta_fz, beta_fz, alpha_c3, alpha_c4, faa_c3c4,
            alpha_pz, alpha_po, alpha_oz, eda_norm
        ])
        
        # Calculate MI with all 3 values
        adaptive_mi, universal_mi, emi = mi_calculator.calculate_adaptive_mi(features)
        
        # Calculate raw MI (scaled version for backward compatibility)
        raw_mi = (universal_mi - 0.5) * 10  # Convert to -5 to +5 range
        
        # Output to LSL streams - all 3 MI values
        current_time = time.time()
        outlets['mi'].push_sample([adaptive_mi], current_time)           # Adaptive MI (0-1)
        outlets['raw_mi'].push_sample([raw_mi], current_time)            # Raw MI (-5 to +5)
        outlets['emi'].push_sample([emi], current_time)                  # EMI (0-1)
        
        # Update visualization
        visualizer.update(adaptive_mi, universal_mi, emi)
        
        # Store session data with comprehensive features
        session_data.append({
            'timestamp': current_time,
            'adaptive_mi': adaptive_mi,
            'universal_mi': universal_mi,
            'raw_mi': raw_mi,
            'emi': emi,
            'theta_fz': theta_fz,
            'beta_fz': beta_fz,
            'alpha_c3': alpha_c3,
            'alpha_c4': alpha_c4,
            'faa_c3c4': faa_c3c4,
            'alpha_pz': alpha_pz,
            'alpha_po': alpha_po,
            'alpha_oz': alpha_oz,
            'eda_norm': eda_norm
        })
        
        # Print comprehensive progress with feature breakdown
        elapsed = current_time - session_start
        print(f"\n[{elapsed:6.1f}s] === MINDFULNESS ANALYSIS ===")
        print(f"Adaptive MI: {adaptive_mi:.3f} | Universal MI: {universal_mi:.3f} | EMI: {emi:.3f}")
        print(f"Raw MI: {raw_mi:+.1f} (range: -5 to +5)")
        print(f"ATTENTION:  θ_Fz={theta_fz:.1f}  β_Fz={beta_fz:.1f}")
        print(f"BODY:       α_C3={alpha_c3:.1f}  α_C4={alpha_c4:.1f}")
        print(f"EMOTION:    FAA={faa_c3c4:+.2f}")
        print(f"DMN:        α_Pz={alpha_pz:.1f}")
        print(f"VISUAL:     α_PO={alpha_po:.1f}  α_Oz={alpha_oz:.1f}")
        print(f"AROUSAL:    EDA={eda_norm:.1f}")
    
    # Session complete - generate reports
    print(f"\n{'='*60}")
    print("SESSION COMPLETE - GENERATING REPORTS")
    print(f"{'='*60}")
    
    session_duration = time.time() - session_start
    print(f"Session duration: {session_duration:.1f} seconds")
    print(f"Total samples: {len(session_data)}")
    
    # Save session data
    if len(session_data) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_csv = os.path.join(LOG_DIR, f'{user_id}_dual_calibration_session_{timestamp}.csv')
        
        df = pd.DataFrame(session_data)
        df.to_csv(session_csv, index=False)
        print(f"[SAVED] Session data: {session_csv}")
        
        # Print statistics
        print(f"\n[STATISTICS]")
        print(f"  Adaptive MI: {df['adaptive_mi'].mean():.3f} ± {df['adaptive_mi'].std():.3f}")
        print(f"  Universal MI: {df['universal_mi'].mean():.3f} ± {df['universal_mi'].std():.3f}")
        print(f"  EMI: {df['emi'].mean():.3f} ± {df['emi'].std():.3f}")
        print(f"  Raw MI: {df['raw_mi'].mean():.1f} ± {df['raw_mi'].std():.1f}")
        print(f"  Dynamic Range: {df['adaptive_mi'].max() - df['adaptive_mi'].min():.3f}")
        
        # Generate visualization
        visualizer.final_plot(user_id)
    
    print(f"\n{'='*60}")
    print("DUAL CALIBRATION SESSION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
