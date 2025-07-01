"""
Real-time Mindfulness Index (MI) Pipeline with Dual Calibration - Clean Version
==============================================================================

This version implements a dual-phase calibration system:
1. RELAXED calibration (30 seconds) - eyes closed, deep breathing
2. FOCUSED calibration (30 seconds) - eyes open, attention task

Features:
- Dual calibration periods for personalized MI mapping
- Robust peak suppression and artifact rejection
- Real-time processing with LSL streams
- ATT (Attention Index) output stream
"""

import os
import sys
import time
import json
import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
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
    'Fz': 0, 'C3': 1, 'Cz': 2, 'C4': 3,
    'Pz': 4, 'PO7': 5, 'PO8': 6, 'Oz': 7
}

EDA_CHANNEL_INDEX = 1  # Channel 1 for EDA features

# Directories
MODEL_DIR = 'models'
LOG_DIR = 'logs'
VIS_DIR = 'visualizations'
USER_CONFIG_DIR = 'user_configs'

# Ensure directories exist
for directory in [MODEL_DIR, LOG_DIR, VIS_DIR, USER_CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

# === ROBUST DATA PROCESSOR ===
class RobustDataProcessor:
    """Handles robust data processing with feature extraction"""
    
    def __init__(self):
        self.median_filter_size = 5
    
    def median_filter_1d(self, data, size=5):
        """Apply median filter to 1D data"""
        if len(data) < size:
            return data
        
        filtered = np.copy(data)
        half_size = size // 2
        
        for i in range(half_size, len(data) - half_size):
            filtered[i] = np.median(data[i-half_size:i+half_size+1])
        
        return filtered
    
    def process_eeg_window(self, eeg_window):
        """Process EEG window with artifact rejection"""
        if len(eeg_window) == 0:
            return eeg_window
        
        processed_window = np.copy(eeg_window)
        
        # Apply median filter to each channel
        for ch in range(processed_window.shape[1]):
            processed_window[:, ch] = self.median_filter_1d(
                processed_window[:, ch], self.median_filter_size
            )
        
        return processed_window
    
    def process_eda_window(self, eda_window):
        """Process EDA window with smoothing"""
        if len(eda_window) == 0:
            return eda_window
        
        processed_window = np.copy(eda_window)
        
        # Apply median filter for smoothing
        for ch in range(processed_window.shape[1]):
            processed_window[:, ch] = self.median_filter_1d(
                processed_window[:, ch], self.median_filter_size
            )
        
        return processed_window
    
    def compute_bandpower(self, data, sf, band):
        """Compute bandpower using Welch's method"""
        try:
            f, psd = welch(data, sf, nperseg=min(len(data), sf))
            idx_band = np.logical_and(f >= band[0], f <= band[1])
            bp = np.trapz(psd[idx_band], f[idx_band])
            return max(bp, 1e-8)
        except:
            return 1.0
    
    def normalize_eda_robust(self, raw_eda):
        """Robust EDA normalization"""
        # Use adaptive range: 0-12 for high arousal states
        eda_min, eda_max = 0, 12
        eda_clipped = np.clip(raw_eda, eda_min * 0.5, eda_max * 1.5)
        eda_norm = 10 * (eda_clipped - eda_min) / max(eda_max - eda_min, 0.1)
        return np.clip(eda_norm, 0, 10)
    
    def extract_features(self, eeg_window, eda_window):
        """Extract comprehensive mindfulness features"""
        sf = 250
        
        # Process windows
        eeg_processed = self.process_eeg_window(eeg_window)
        eda_processed = self.process_eda_window(eda_window)
        
        # === ATTENTION REGULATION ===
        theta_fz = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['Fz']], sf, (4, 8))
        beta_fz = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['Fz']], sf, (13, 30))
        
        # === BODY AWARENESS ===
        alpha_c3 = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['C3']], sf, (8, 13))
        alpha_c4 = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['C4']], sf, (8, 13))
        
        # === EMOTION REGULATION ===
        faa_c3c4 = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
        
        # === DMN SUPPRESSION ===
        alpha_pz = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['Pz']], sf, (8, 13))
        
        # === RELAXATION ===
        alpha_po7 = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['PO7']], sf, (8, 13))
        alpha_po8 = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['PO8']], sf, (8, 13))
        alpha_po = (alpha_po7 + alpha_po8) / 2
        alpha_oz = self.compute_bandpower(eeg_processed[:, EEG_CHANNELS['Oz']], sf, (8, 13))
        
        # === AROUSAL/STRESS ===
        raw_eda = np.mean(eda_processed[:, EDA_CHANNEL_INDEX])
        eda_norm = self.normalize_eda_robust(raw_eda)
        
        return np.array([
            theta_fz, beta_fz, alpha_c3, alpha_c4, faa_c3c4,
            alpha_pz, alpha_po, alpha_oz, eda_norm
        ])

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
            print("• Sit comfortably, close your eyes")
            print("• Take slow, deep breaths")
            print("• Let your mind wander naturally")
            print("• This establishes your LOW mindfulness baseline")
        elif phase == 'focused':
            print("FOCUSED BASELINE CALIBRATION (30 seconds)")
            print("• Open your eyes, look at a fixed point")
            print("• Focus attention on your breathing")
            print("• Count breaths: 1 (inhale), 2 (exhale), etc.")
            print("• This establishes your HIGH mindfulness baseline")
        
        input("\nPress Enter when ready...")
        print("Starting in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("BEGIN!")
    
    def collect_calibration_data(self, eeg_inlet, eda_inlet, duration_sec, phase_name):
        """Collect calibration data with real-time duration control"""
        print(f"\n[CALIBRATION] Starting {phase_name} phase...")
        
        features_list = []
        window_size = 250  # 1 second windows
        eeg_samples, eda_samples = [], []
        
        start_time = time.time()
        progress_interval = duration_sec / 20
        next_progress = start_time + progress_interval
        
        print("Progress: ", end="", flush=True)
        
        while (time.time() - start_time) < duration_sec:
            loop_start = time.time()
            
            # Collect EEG sample
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.01)
                if eeg_sample is not None:
                    eeg_samples.append(np.array(eeg_sample[:8]))
                else:
                    if len(eeg_samples) > 0:
                        eeg_samples.append(eeg_samples[-1])
                    else:
                        eeg_samples.append(np.zeros(8))
            else:
                eeg_samples.append(np.zeros(8))
            
            # Collect EDA sample
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=0.01)
                if eda_sample is not None:
                    eda_samples.append(np.array(eda_sample[:2]))
                else:
                    if len(eda_samples) > 0:
                        eda_samples.append(eda_samples[-1])
                    else:
                        eda_samples.append(np.zeros(2))
            else:
                eda_samples.append(np.zeros(2))
            
            # Extract features every second
            if len(eeg_samples) >= window_size and len(eeg_samples) % window_size == 0:
                eeg_window = np.array(eeg_samples[-window_size:])
                eda_window = np.array(eda_samples[-window_size:])
                
                features = self.data_processor.extract_features(eeg_window, eda_window)
                
                if not np.any(np.isnan(features)):
                    features_list.append(features)
            
            # Progress indicator
            if time.time() >= next_progress:
                print("█", end="", flush=True)
                next_progress += progress_interval
            
            # Maintain 250 Hz sampling rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0/250) - elapsed)
            time.sleep(sleep_time)
        
        print(" ✓ Complete!")
        actual_duration = time.time() - start_time
        print(f"[CALIBRATION] {phase_name}: {len(features_list)} windows in {actual_duration:.1f}s")
        
        return np.array(features_list) if features_list else None
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation"""
        weights = np.array([
            0.35, -0.08, 0.14, 0.14, 0.10,  # theta_fz, beta_fz, alpha_c3, alpha_c4, faa_c3c4
            -0.20, 0.22, 0.15, -0.10        # alpha_pz, alpha_po, alpha_oz, eda_norm
        ])
        
        # Normalize features to 0-10 range
        ranges = {
            'theta_fz': (1, 80), 'beta_fz': (0.5, 15), 'alpha_c3': (2, 30),
            'alpha_c4': (2, 30), 'faa_c3c4': (-2.5, 2.5), 'alpha_pz': (2, 35),
            'alpha_po': (1, 25), 'alpha_oz': (2, 25), 'eda_norm': (0, 12)
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        normalized_features = np.array(normalized)
        
        # Calculate weighted sum with adaptive centering
        weighted_sum = np.dot(normalized_features, weights)
        eda_norm = normalized_features[8]
        alpha_norm = np.mean(normalized_features[2:8])  # alpha channels
        
        if eda_norm > 7:
            center_shift = -1.8
        elif alpha_norm > 6:
            center_shift = -1.0
        else:
            center_shift = -1.5
        
        centered_sum = weighted_sum + center_shift
        mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))
        mi = 0.1 + 0.8 * mi_sigmoid
        
        return np.clip(mi, 0.1, 0.9)
    
    def compute_adaptive_thresholds(self):
        """Compute personalized adaptive thresholds"""
        if len(self.relaxed_features) == 0 or len(self.focused_features) == 0:
            print("[ERROR] Missing calibration data!")
            return None
        
        print(f"\n[ANALYSIS] Computing adaptive thresholds...")
        print(f"Relaxed: {len(self.relaxed_features)} windows")
        print(f"Focused: {len(self.focused_features)} windows")
        
        # Compute MI for each phase
        relaxed_mi = [self.calculate_mi_universal(f) for f in self.relaxed_features]
        focused_mi = [self.calculate_mi_universal(f) for f in self.focused_features]
        
        relaxed_mi = np.array(relaxed_mi)
        focused_mi = np.array(focused_mi)
        
        # Check dynamic range
        dynamic_range = np.mean(focused_mi) - np.mean(relaxed_mi)
        
        print(f"Relaxed MI: {np.mean(relaxed_mi):.3f} ± {np.std(relaxed_mi):.3f}")
        print(f"Focused MI: {np.mean(focused_mi):.3f} ± {np.std(focused_mi):.3f}")
        print(f"Dynamic Range: {dynamic_range:.3f}")
        
        # Ensure minimum separation
        if abs(dynamic_range) < 0.15:
            print("[ADJUSTMENT] Applying minimum separation...")
            adjusted_relaxed = np.mean(relaxed_mi) - 0.075
            adjusted_focused = np.mean(focused_mi) + 0.075
            relaxed_mi = np.full_like(relaxed_mi, adjusted_relaxed)
            focused_mi = np.full_like(focused_mi, adjusted_focused)
            dynamic_range = adjusted_focused - adjusted_relaxed
        
        self.adaptive_thresholds = {
            'relaxed_baseline': {
                'mi_mean': float(np.mean(relaxed_mi)),
                'mi_std': float(np.std(relaxed_mi))
            },
            'focused_baseline': {
                'mi_mean': float(np.mean(focused_mi)),
                'mi_std': float(np.std(focused_mi))
            },
            'adaptive_mapping': {
                'low_threshold': float(np.mean(relaxed_mi)),
                'high_threshold': float(np.mean(focused_mi)),
                'dynamic_range': float(dynamic_range),
                'calibration_time': str(datetime.now())
            }
        }
        
        print(f"[RESULTS] Final thresholds computed!")
        return self.adaptive_thresholds
    
    def save_calibration_data(self):
        """Save calibration data"""
        config_path = os.path.join(USER_CONFIG_DIR, f'{self.user_id}_dual_calibration.json')
        
        with open(config_path, 'w') as f:
            json.dump(self.adaptive_thresholds, f, indent=2)
        
        print(f"[SAVED] Configuration: {config_path}")
        return config_path

# === ADAPTIVE MI CALCULATOR ===
class AdaptiveMICalculator:
    """Calculates MI using adaptive thresholds from dual calibration"""
    
    def __init__(self, adaptive_thresholds, user_id=None):
        self.thresholds = adaptive_thresholds
        self.user_id = user_id
        self.mi_history = []
        self.smoothing_window = 3
        self.data_processor = RobustDataProcessor()
    
    def calculate_adaptive_mi(self, features):
        """Calculate adaptive MI using personalized thresholds"""
        # Calculate universal MI
        universal_mi = self.calculate_mi_universal(features)
        
        # Calculate EMI (emotion-focused)
        emi = self.calculate_emi(features, universal_mi)
        
        # Apply adaptive mapping
        if self.thresholds is not None:
            mapping = self.thresholds['adaptive_mapping']
            low_thresh = mapping['low_threshold']
            high_thresh = mapping['high_threshold']
            dynamic_range = mapping['dynamic_range']
            
            if dynamic_range > 0.05:
                relative_position = (universal_mi - low_thresh) / dynamic_range
                adaptive_mi = np.clip(relative_position * 1.1, 0, 1)
            else:
                adaptive_mi = np.clip(universal_mi * 1.2, 0, 1)
        else:
            adaptive_mi = universal_mi
        
        # Temporal smoothing
        self.mi_history.append(adaptive_mi)
        if len(self.mi_history) > self.smoothing_window:
            self.mi_history.pop(0)
        
        smoothed_mi = np.mean(self.mi_history)
        
        return smoothed_mi, universal_mi, emi
    
    def calculate_mi_universal(self, features):
        """Universal MI calculation (same as calibration)"""
        weights = np.array([
            0.35, -0.08, 0.14, 0.14, 0.10,
            -0.20, 0.22, 0.15, -0.10
        ])
        
        ranges = {
            'theta_fz': (1, 80), 'beta_fz': (0.5, 15), 'alpha_c3': (2, 30),
            'alpha_c4': (2, 30), 'faa_c3c4': (-2.5, 2.5), 'alpha_pz': (2, 35),
            'alpha_po': (1, 25), 'alpha_oz': (2, 25), 'eda_norm': (0, 12)
        }
        
        normalized = []
        for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
            val = 10 * (features[i] - q5) / (q95 - q5)
            normalized.append(np.clip(val, 0, 10))
        
        normalized_features = np.array(normalized)
        weighted_sum = np.dot(normalized_features, weights)
        
        eda_norm = normalized_features[8]
        alpha_norm = np.mean(normalized_features[2:8])
        
        if eda_norm > 7:
            center_shift = -1.8
        elif alpha_norm > 6:
            center_shift = -1.0
        else:
            center_shift = -1.5
        
        centered_sum = weighted_sum + center_shift
        mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))
        mi = 0.1 + 0.8 * mi_sigmoid
        
        return np.clip(mi, 0.1, 0.9)
    
    def calculate_emi(self, features, universal_mi):
        """Calculate Emotional Mindfulness Index"""
        theta_fz = features[0]
        beta_fz = features[1]
        faa_c3c4 = features[4]
        eda_norm = features[8]
        
        emotion_score = (
            0.3 * (theta_fz / 10) +
            0.2 * (1 - beta_fz / 10) +
            0.3 * np.tanh(abs(faa_c3c4)) +
            0.2 * (1 - eda_norm / 10)
        )
        
        emi = 0.7 * universal_mi + 0.3 * emotion_score
        return np.clip(emi, 0, 1)

# === LSL UTILITIES ===
def select_lsl_stream(stream_type, allow_skip=False):
    """Select LSL stream with user input"""
    print(f"Searching for {stream_type} streams...")
    streams = resolve_streams()
    
    if not streams:
        if allow_skip:
            print(f"No streams found. Skipping {stream_type}.")
            return None
        else:
            raise RuntimeError("No LSL streams found!")
    
    print("Available streams:")
    for idx, s in enumerate(streams):
        print(f"[{idx}] {s.name()} | {s.type()} | {s.channel_count()} channels")
    
    if allow_skip:
        print(f"[{len(streams)}] SKIP this sensor")
    
    while True:
        try:
            sel = input(f"Select {stream_type} stream index: ")
            if allow_skip and sel.strip() == str(len(streams)):
                return None
            
            sel = int(sel)
            if 0 <= sel < len(streams):
                chosen = streams[sel]
                print(f"Selected: {chosen.name()}")
                return chosen
            else:
                print(f"Invalid index. Enter 0-{len(streams)-1}")
        except ValueError:
            print("Invalid input. Enter a number.")

def setup_output_streams():
    """Setup LSL output streams"""
    streams = {}
    
    # Mindfulness Index (adaptive)
    mi_info = StreamInfo('mindfulness_index', 'MI', 1, 1, 'float32', 'mi_001')
    streams['mi'] = StreamOutlet(mi_info)
    
    # Raw MI (universal)
    raw_mi_info = StreamInfo('raw_mindfulness_index', 'RawMI', 1, 1, 'float32', 'raw_mi_001')
    streams['raw_mi'] = StreamOutlet(raw_mi_info)
    
    # EMI (emotion-focused)
    emi_info = StreamInfo('emotional_mindfulness_index', 'EMI', 1, 1, 'float32', 'emi_001')
    streams['emi'] = StreamOutlet(emi_info)
    
    # ATT (attention index)
    att_info = StreamInfo('attention_index', 'ATT', 1, 1, 'float32', 'att_001')
    streams['att'] = StreamOutlet(att_info)
    
    print("[LSL] Output streams created:")
    print("  - mindfulness_index (Adaptive MI: 0-1)")
    print("  - raw_mindfulness_index (Universal MI: 0-1)")
    print("  - emotional_mindfulness_index (EMI: 0-1)")
    print("  - attention_index (ATT: 0-1)")
    
    return streams

# === MAIN FUNCTIONS ===
def run_dual_calibration(user_id, eeg_inlet, eda_inlet):
    """Run dual calibration process"""
    print(f"\n{'='*60}")
    print("DUAL CALIBRATION PROCESS")
    print(f"{'='*60}")
    print("Two phases: RELAXED (30s) + FOCUSED (30s)")
    
    input("Press Enter to begin...")
    
    calibration_system = DualCalibrationSystem(user_id)
    
    # Phase 1: Relaxed
    calibration_system.display_calibration_instructions('relaxed')
    relaxed_features = calibration_system.collect_calibration_data(
        eeg_inlet, eda_inlet, 30, 'RELAXED'
    )
    
    if relaxed_features is None:
        print("[ERROR] Relaxed calibration failed!")
        return None
    
    calibration_system.relaxed_features = relaxed_features
    print(f"✓ Relaxed baseline: {len(relaxed_features)} windows")
    
    # Short break
    print("\n5-second break...")
    for i in range(5, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print()
    
    # Phase 2: Focused
    calibration_system.display_calibration_instructions('focused')
    focused_features = calibration_system.collect_calibration_data(
        eeg_inlet, eda_inlet, 30, 'FOCUSED'
    )
    
    if focused_features is None:
        print("[ERROR] Focused calibration failed!")
        return None
    
    calibration_system.focused_features = focused_features
    print(f"✓ Focused baseline: {len(focused_features)} windows")
    
    # Compute thresholds
    adaptive_thresholds = calibration_system.compute_adaptive_thresholds()
    
    if adaptive_thresholds is None:
        print("[ERROR] Failed to compute thresholds!")
        return None
    
    # Save data
    calibration_system.save_calibration_data()
    
    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE!")
    print(f"{'='*60}")
    
    return adaptive_thresholds

def run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator):
    """Run real-time MI processing"""
    print(f"\n{'='*60}")
    print("REAL-TIME MI PROCESSING")
    print(f"{'='*60}")
    print("Press 'q' + Enter to stop...")
    
    processor = RobustDataProcessor()
    
    # Data buffers
    eeg_buffer = []
    eda_buffer = []
    window_size = 250  # 1 second windows
    
    session_data = []
    start_time = time.time()
    last_display = 0
    
    # Input monitoring
    stop_event = threading.Event()
    
    def input_monitor():
        while True:
            if input().strip().lower() == 'q':
                stop_event.set()
                break
    
    input_thread = threading.Thread(target=input_monitor, daemon=True)
    input_thread.start()
    
    print("Collecting data...")
    print("\n" + "="*80)
    print("📡 REAL-TIME LSL TRANSMISSION VALUES")
    print("="*80)
    print("  MI   = Adaptive Mindfulness Index (0-1, personalized)")
    print("  Raw  = Universal Mindfulness Index (0-1, standard)")
    print("  EMI  = Emotional Mindfulness Index (0-1, emotion-focused)")
    print("  ATT  = Attention Index (0-1, theta-based alertness)")
    print("="*80)
    print("Press 'q' + Enter to stop transmission...\n")
    
    try:
        while not stop_event.is_set():
            # Collect EEG
            eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.01)
            if eeg_sample is not None:
                eeg_buffer.append(eeg_sample[:8])
            
            # Collect EDA
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=0.01)
                if eda_sample is not None:
                    eda_buffer.append(eda_sample[:2])
            
            # Process when window is full
            if len(eeg_buffer) >= window_size:
                eeg_window = np.array(eeg_buffer[:window_size])
                eeg_buffer = eeg_buffer[window_size:]
                
                if eda_inlet is not None and len(eda_buffer) >= window_size:
                    eda_window = np.array(eda_buffer[:window_size])
                    eda_buffer = eda_buffer[window_size:]
                else:
                    eda_window = np.zeros((window_size, 2))
                
                # Extract features
                features = processor.extract_features(eeg_window, eda_window)
                
                # Calculate MI values
                adaptive_mi, universal_mi, emi = mi_calculator.calculate_adaptive_mi(features)
                
                # Calculate ATT (Attention Index)
                theta_fz = features[0]
                att = np.clip(theta_fz / 50, 0, 1)  # Normalize theta for attention
                
                # Push to LSL streams
                output_streams['mi'].push_sample([adaptive_mi])
                output_streams['raw_mi'].push_sample([universal_mi])
                output_streams['emi'].push_sample([emi])
                output_streams['att'].push_sample([att])
                
                # Log data
                session_data.append([adaptive_mi, universal_mi, emi, att])
                
                # Real-time display of transmitted LSL values
                elapsed_time = time.time() - start_time
                print(f"\r[{elapsed_time:6.1f}s] LSL OUT → MI: {adaptive_mi:.3f} | Raw: {universal_mi:.3f} | EMI: {emi:.3f} | ATT: {att:.3f}", end="", flush=True)
                
                # Detailed display every 10 seconds
                if time.time() - last_display > 10:
                    print(f"\n[{elapsed_time:6.1f}s] ── LSL Transmission Status ──")
                    print(f"  📊 Mindfulness Index (MI):     {adaptive_mi:.4f}")
                    print(f"  🧠 Raw Universal MI:           {universal_mi:.4f}")
                    print(f"  💭 Emotional MI (EMI):         {emi:.4f}")
                    print(f"  🎯 Attention Index (ATT):      {att:.4f}")
                    print(f"  📈 Session samples:            {len(session_data)}")
                    print(f"  🔄 Sampling rate:              ~{len(session_data)/elapsed_time:.1f} Hz")
                    last_display = time.time()
    
    except KeyboardInterrupt:
        print("\n[STOPPED] Processing interrupted.")
    
    # Final transmission summary
    total_duration = time.time() - start_time
    print(f"\n\n{'='*60}")
    print("📡 LSL TRANSMISSION SUMMARY")
    print("="*60)
    print(f"  ⏱️  Total duration:        {total_duration:.1f} seconds")
    print(f"  📊  Samples transmitted:   {len(session_data)}")
    print(f"  🔄  Average rate:          {len(session_data)/total_duration:.2f} Hz")
    if session_data:
        data_array = np.array(session_data)
        print(f"  📈  MI range:              {data_array[:, 0].min():.3f} - {data_array[:, 0].max():.3f}")
        print(f"  🧠  Raw MI range:          {data_array[:, 1].min():.3f} - {data_array[:, 1].max():.3f}")
        print(f"  💭  EMI range:             {data_array[:, 2].min():.3f} - {data_array[:, 2].max():.3f}")
        print(f"  🎯  ATT range:             {data_array[:, 3].min():.3f} - {data_array[:, 3].max():.3f}")
    print("="*60)
    
    # Save session data
    if session_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_file = os.path.join(LOG_DIR, f'{user_id}_session_{timestamp}.csv')
        df = pd.DataFrame(session_data, columns=['Adaptive_MI', 'Universal_MI', 'EMI', 'ATT'])
        df.to_csv(session_file, index=False)
        print(f"[SAVED] Session data: {session_file}")
    
    print("[COMPLETE] Real-time processing ended.")

def main():
    """Main function"""
    print("Real-time Mindfulness Index Pipeline with Dual Calibration")
    print("=" * 60)
    
    # Get user ID
    user_id = input("Enter user ID: ").strip()
    if not user_id:
        user_id = f"user_{int(time.time())}"
        print(f"Using ID: {user_id}")
    
    # Connect to streams
    print("\n[SETUP] Connecting to streams...")
    
    # EEG stream (required)
    eeg_stream = select_lsl_stream('EEG', allow_skip=False)
    if eeg_stream is None:
        print("[ERROR] EEG stream required!")
        return
    eeg_inlet = StreamInlet(eeg_stream)
    
    # EDA stream (optional)
    eda_stream = select_lsl_stream('EDA', allow_skip=True)
    eda_inlet = StreamInlet(eda_stream) if eda_stream else None
    
    # Setup output streams
    output_streams = setup_output_streams()
    
    print("\n[READY] All streams connected!")
    
    # Run calibration
    adaptive_thresholds = run_dual_calibration(user_id, eeg_inlet, eda_inlet)
    if adaptive_thresholds is None:
        print("[ERROR] Calibration failed!")
        return
    
    # Initialize MI calculator
    mi_calculator = AdaptiveMICalculator(adaptive_thresholds, user_id)
    
    print("\n[READY] Starting real-time processing...")
    input("Press Enter to continue (or 'q' to quit): ")
    
    # Real-time processing
    run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator)

if __name__ == "__main__":
    main()
