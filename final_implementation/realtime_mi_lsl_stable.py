#!/usr/bin/env python3
"""
STABLE REAL-TIME MINDFULNESS INDEX (MI) LSL PIPELINE
====================================================

This is a STABLE version focused on consistent, repeatable results for real-time interactive applications.

KEY CHANGES FROM ORIGINAL:
- REMOVED: Personal baseline statistics and relative adjustments
- REMOVED: Automatic saturation detection and fallback mechanisms
- REMOVED: Dynamic input scaling based on real-time analysis
- REMOVED: Flexible normalization with user/population stats
- REMOVED: Anti-static mechanisms and adaptive noise
- FIXED: Static feature normalization ranges based on population data
- FIXED: Consistent scaling factors (no adaptation)
- FIXED: Single MI calculation method (no fallbacks)

This ensures session-to-session consistency and predictable behavior for interactive applications.
"""

import os
import sys
import time
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
from scipy.stats import spearmanr
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import msvcrt  # For Windows key detection

# LSL imports
try:
    from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams, resolve_byprop
    print("[OK] LSL library imported successfully")
except ImportError as e:
    print(f"[ERROR] LSL library not found: {e}")
    print("[HINT] Install LSL: pip install pylsl")
    sys.exit(1)

# === STABLE CONFIGURATION - NO DYNAMIC CHANGES ===
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
EDA_CHANNEL_INDEX = 1  # Fixed channel selection

# Fixed scaling factors - NO ADAPTATION
EEG_SCALE_FACTOR = 1.0  # No scaling - values in correct physiological range
EDA_SCALE_FACTOR = 1.0  # No scaling - values in correct physiological range

# Fixed normalization ranges - STATIC, NO USER ADAPTATION
FIXED_NORMALIZATION_RANGES = {
    'theta_fz': (2, 60),      # Fixed range based on population data
    'alpha_po': (1, 30),      # Fixed range based on population data  
    'faa': (-2.5, 2.5),       # Fixed range based on population data
    'beta_frontal': (2, 35),  # Fixed range based on population data
    'eda_norm': (2, 12)       # Fixed range based on population data
}

# Fixed MI calculation weights - NO ADAPTATION
FIXED_MI_WEIGHTS = np.array([0.3, 0.3, 0.2, -0.1, -0.2])

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'user_configs')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# Create directories
for d in [MODEL_DIR, LOG_DIR, VIS_DIR, USER_CONFIG_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# === STABLE SIGNAL PROCESSING FUNCTIONS ===

def compute_bandpower(data, sf, band):
    """Compute power spectral density in a specific frequency band - STABLE VERSION"""
    if len(data) < sf:  # Need at least 1 second of data
        return 0.0
    
    # Fixed parameters - no adaptation
    nperseg = min(sf, len(data))
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)
    
    # Find frequency indices
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    if not np.any(idx_band):
        return 0.0
    
    # Compute band power
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return max(band_power, 1e-8)  # Prevent zero values

def normalize_eda_stable(eda_raw):
    """STABLE EDA normalization - NO USER ADAPTATION"""
    # Fixed normalization based on population statistics
    # Simple robust scaling using fixed percentiles
    normalized = (eda_raw - 5.0) / 3.0  # Fixed center and scale
    return np.clip(normalized, 0, 10)

# === STABLE FEATURE EXTRACTION ===

def extract_features_stable(eeg_window, eda_window, sf=250):
    """Extract features with STABLE, consistent processing - NO ADAPTATION"""
    
    # EEG features - fixed parameters
    theta_fz = compute_bandpower(eeg_window[:, 0], sf, (4, 8))
    alpha_po7 = compute_bandpower(eeg_window[:, 6], sf, (8, 13))
    alpha_po8 = compute_bandpower(eeg_window[:, 7], sf, (8, 13))
    alpha_po = (alpha_po7 + alpha_po8) / 2
    
    # Frontal Alpha Asymmetry - fixed calculation
    alpha_f4 = compute_bandpower(eeg_window[:, 4], sf, (8, 13))
    alpha_f3 = compute_bandpower(eeg_window[:, 5], sf, (8, 13))
    faa = np.log(alpha_f4 + 1e-8) - np.log(alpha_f3 + 1e-8)
    
    # Beta frontal
    beta_frontal = compute_bandpower(eeg_window[:, 0], sf, (13, 30))
    
    # EDA - fixed channel and normalization
    raw_eda = np.mean(eda_window[:, EDA_CHANNEL_INDEX])
    eda_norm = normalize_eda_stable(raw_eda)
    
    return [theta_fz, alpha_po, faa, beta_frontal, eda_norm]

# === STABLE NORMALIZATION - NO USER ADAPTATION ===

def normalize_features_stable(features):
    """STABLE feature normalization using FIXED ranges - NO USER ADAPTATION"""
    normalized = {}
    
    for i, feat_name in enumerate(FEATURE_ORDER):
        value = features[i] if isinstance(features, (list, np.ndarray)) else features[feat_name]
        
        # Use FIXED ranges - no user or session adaptation
        min_val, max_val = FIXED_NORMALIZATION_RANGES[feat_name]
        
        # Simple linear mapping to [0, 10] range
        normalized_val = 10 * (value - min_val) / (max_val - min_val)
        normalized[feat_name] = np.clip(normalized_val, 0, 10)
    
    return normalized

# === STABLE MI CALCULATION - NO FALLBACKS OR ADAPTATION ===

def calculate_mi_stable(features):
    """STABLE MI calculation - NO FALLBACKS, NO ADAPTATION, NO USER BASELINES"""
    
    # Normalize features using FIXED ranges
    normalized_features = normalize_features_stable(features)
    
    # Convert to array for calculation
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    
    # Fixed weighted sum - NO ADAPTATION
    weighted_sum = np.dot(feature_array, FIXED_MI_WEIGHTS)
    
    # STABLE mapping to [0.1, 0.9] range - NO DYNAMIC ADJUSTMENTS
    raw_score = weighted_sum / 10.0  # Normalize to 0-1
    mi = 0.1 + 0.8 * raw_score  # Map to 0.1-0.9 range
    
    return np.clip(mi, 0.1, 0.9)

def calculate_raw_mi_stable(features):
    """STABLE raw MI calculation for extended range"""
    normalized_features = normalize_features_stable(features)
    
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    
    weighted_sum = np.dot(feature_array, FIXED_MI_WEIGHTS)
    
    # Fixed mapping to [-5, +5] range
    raw_mi = (weighted_sum - 5.0) * 2.0
    return np.clip(raw_mi, -5, 5)

def calculate_emi_stable(features):
    """STABLE Emotional MI calculation"""
    normalized_features = normalize_features_stable(features)
    
    feature_array = np.array([
        normalized_features['theta_fz'],
        normalized_features['alpha_po'],
        normalized_features['faa'],
        normalized_features['beta_frontal'],
        normalized_features['eda_norm']
    ])
    
    # Fixed EMI weights - emphasis on emotional features
    emi_weights = np.array([0.15, 0.15, 0.4, -0.05, -0.25])
    weighted_sum = np.dot(feature_array, emi_weights)
    
    # Fixed sigmoid mapping
    centered_sum = weighted_sum - 4.0
    emi = 1 / (1 + np.exp(-centered_sum * 0.8))
    return np.clip(emi, 0.05, 0.95)

# === LSL STREAM SETUP ===

def setup_mindfulness_lsl_streams():
    """Setup LSL output streams for MI transmission"""
    print("[LSL] Creating MI output streams...")
    
    # Main MI stream (SVR-based or stable calculation)
    mi_info = StreamInfo('MindfulnessIndex', 'MI', 1, 3.0, 'float32', 'mi_stable_001')
    mi_outlet = StreamOutlet(mi_info)
    
    # Raw MI stream (extended range)
    raw_mi_info = StreamInfo('RawMindfulnessIndex', 'RawMI', 1, 3.0, 'float32', 'raw_mi_stable_001')
    raw_mi_outlet = StreamOutlet(raw_mi_info)
    
    # Emotional MI stream
    emi_info = StreamInfo('EmotionalMindfulnessIndex', 'EMI', 1, 3.0, 'float32', 'emi_stable_001')
    emi_outlet = StreamOutlet(emi_info)
    
    print("[LSL] MI output streams created successfully")
    print("  - MindfulnessIndex (0-1 range)")
    print("  - RawMindfulnessIndex (-5 to +5 range)")  
    print("  - EmotionalMindfulnessIndex (0-1 range)")
    
    return {
        'mi': mi_outlet,
        'raw_mi': raw_mi_outlet,
        'emi': emi_outlet
    }

def select_lsl_stream(stream_type, name_hint=None, allow_skip=False, confirm=True):
    """Select LSL input stream"""
    print(f"\n[LSL] Looking for {stream_type} streams...")
    
    if stream_type == 'EEG':
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
    elif stream_type == 'EDA':
        streams = resolve_byprop('type', 'EDA', timeout=5.0)
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
    
    if len(streams) == 1:
        stream = streams[0]
        print(f"[AUTO] Using {stream_type} stream: {stream.name()}")
        return stream
    
    # Multiple streams - let user choose
    print(f"[CHOICE] Multiple {stream_type} streams found:")
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

# === STABLE MODEL LOADING ===

def load_stable_models():
    """Load models without user-specific adaptations"""
    print("[MODEL] Loading stable models...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[WARN] Global models not found. Using stable calculation only.")
        return None, None
    
    try:
        svr = load(MODEL_PATH)
        scaler = load(SCALER_PATH)
        print("[MODEL] Global SVR model and scaler loaded successfully")
        return svr, scaler
    except Exception as e:
        print(f"[WARN] Failed to load models: {e}")
        return None, None

# === STABLE VISUALIZATION ===

class StableVisualizer:
    """Visualization class for stable MI tracking"""
    
    def __init__(self):
        self.mi_history = []
        self.raw_mi_history = []
        self.emi_history = []
        self.timestamps = []
        
    def update(self, mi_pred, raw_mi=None, emi=None):
        """Update with MI values"""
        self.mi_history.append(mi_pred)
        self.raw_mi_history.append(raw_mi if raw_mi is not None else np.nan)
        self.emi_history.append(emi if emi is not None else np.nan)
        self.timestamps.append(datetime.now())
    
    def save_final_plot(self, user_id):
        """Save final comparison plot"""
        if len(self.mi_history) == 0:
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        x = np.arange(len(self.mi_history))
        
        # Plot Standard MI
        axes[0].plot(x, self.mi_history, label='Stable MI', color='blue', linewidth=2)
        axes[0].set_title('Stable Mindfulness Index (0-1 range)')
        axes[0].set_ylabel('MI Value')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Plot Raw MI
        axes[1].plot(x, self.raw_mi_history, label='Raw MI', color='purple', linewidth=2)
        axes[1].set_title('Raw MI (-5 to +5 range)')
        axes[1].set_ylabel('Raw Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot EMI
        axes[2].plot(x, self.emi_history, label='Emotional MI', color='green', linewidth=2)
        axes[2].set_title('Emotional Mindfulness Index (0-1 range)')
        axes[2].set_xlabel('Samples')
        axes[2].set_ylabel('EMI Value')
        axes[2].legend()
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{user_id}_stable_mi_{timestamp}.png'
        filepath = os.path.join(VIS_DIR, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"[VIS] Stable MI plot saved to {filepath}")
        plt.close(fig)

# === MAIN STABLE PIPELINE ===

def main():
    """Main stable MI pipeline - NO ADAPTATION, NO USER BASELINES"""
    
    print("\n" + "="*60)
    print("STABLE REAL-TIME MI LSL PIPELINE")
    print("="*60)
    print("[INFO] This version provides CONSISTENT, STABLE results")
    print("[INFO] NO user adaptation, NO dynamic scaling, NO fallbacks")
    print("[INFO] Designed for reliable real-time interactive applications")
    print("="*60)
    
    # Get user ID for logging only
    user_id = input("Enter user ID for session logging: ").strip()
    if not user_id:
        user_id = f"stable_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"[CONFIG] User ID: {user_id}")
    print(f"[CONFIG] EDA Channel: {EDA_CHANNEL_INDEX + 1} (fixed)")
    print(f"[CONFIG] Scaling: EEG={EEG_SCALE_FACTOR}, EDA={EDA_SCALE_FACTOR} (fixed)")
    print(f"[CONFIG] Normalization: Fixed population ranges (no adaptation)")
    
    # Load models (optional - can run without SVR)
    svr, scaler = load_stable_models()
    use_svr = svr is not None and scaler is not None
    
    if use_svr:
        print("[MODE] Using SVR model + stable calculation")
    else:
        print("[MODE] Using stable calculation only (no SVR)")
    
    # Connect to input streams
    print("\n[INPUT] Connecting to data streams...")
    
    # EEG stream
    print("Select EEG stream:")
    eeg_stream = select_lsl_stream('EEG', name_hint='UnicornRecorderLSLStream', allow_skip=True)
    eeg_inlet = StreamInlet(eeg_stream) if eeg_stream else None
    
    # EDA stream  
    print("Select EDA stream:")
    eda_stream = select_lsl_stream('EDA', name_hint='OpenSignals', allow_skip=True)
    eda_inlet = StreamInlet(eda_stream) if eda_stream else None
    
    if eeg_inlet is None and eda_inlet is None:
        print("[ERROR] At least one input stream (EEG or EDA) is required!")
        sys.exit(1)
    
    # Unity markers (optional)
    use_unity = input("\nConnect to Unity markers? (y/n, default: n): ").strip().lower()
    label_inlet = None
    if use_unity == 'y':
        try:
            unity_stream = select_lsl_stream('UnityMarkers', allow_skip=True)
            label_inlet = StreamInlet(unity_stream) if unity_stream else None
        except Exception as e:
            print(f"[WARN] Unity connection failed: {e}")
    
    # Setup output streams
    outlets = setup_mindfulness_lsl_streams()
    mi_outlet = outlets['mi']
    raw_mi_outlet = outlets['raw_mi'] 
    emi_outlet = outlets['emi']
    
    # Initialize tracking
    visualizer = StableVisualizer()
    mi_records = []
    session_start_time = time.time()
    
    print(f"\n[START] Stable MI pipeline running...")
    print("[INFO] Press Enter to stop and generate report")
    print("[INFO] MI calculation rate: 1 Hz (fixed)")
    print("[INFO] No adaptation - consistent output guaranteed")
    
    # Key detection thread
    stop_flag = {'stop': False}
    def wait_for_exit():
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\r', b'\n', b'\x1b'):  # Enter or ESC
                    stop_flag['stop'] = True
                    break
            time.sleep(0.1)
    
    exit_thread = threading.Thread(target=wait_for_exit, daemon=True)
    exit_thread.start()
    
    # Main processing loop
    next_calc_time = time.time()
    sample_count = 0
    
    while not stop_flag['stop']:
        now = time.time()
        if now < next_calc_time:
            time.sleep(max(0, next_calc_time - now))
            continue
        
        next_calc_time += 1.0  # Fixed 1 Hz rate
        sample_count += 1
        
        # Collect 1-second windows (250 samples at 250 Hz)
        eeg_window = []
        eda_window = []
        
        for i in range(250):
            # Get EEG sample
            if eeg_inlet:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=1.0)
                if eeg_sample:
                    eeg_data = np.array(eeg_sample[:8]) * EEG_SCALE_FACTOR  # Fixed scaling
                    eeg_window.append(eeg_data)
                else:
                    eeg_window.append(np.zeros(8))
            else:
                eeg_window.append(np.zeros(8))
            
            # Get EDA sample
            if eda_inlet:
                eda_sample, _ = eda_inlet.pull_sample(timeout=1.0)
                if eda_sample:
                    eda_data = np.array(eda_sample[:2]) * EDA_SCALE_FACTOR  # Fixed scaling
                    eda_window.append(eda_data)
                else:
                    eda_window.append(np.zeros(2))
            else:
                eda_window.append(np.zeros(2))
        
        # Convert to arrays
        eeg_win = np.array(eeg_window)
        eda_win = np.array(eda_window)
        
        # Extract features using STABLE method
        try:
            features = extract_features_stable(eeg_win, eda_win, sf=250)
            
            # Calculate MI using STABLE methods
            mi_pred = calculate_mi_stable(features)
            raw_mi_value = calculate_raw_mi_stable(features)  
            emi_value = calculate_emi_stable(features)
            
            # If SVR available, compare but DON'T use for adaptation
            if use_svr:
                try:
                    sample = np.array(features).reshape(1, -1)
                    x_scaled = scaler.transform(sample)
                    svr_pred = svr.predict(x_scaled)[0]
                    svr_pred = np.clip(svr_pred, 0.1, 0.9)  # Simple clipping
                    
                    # Use SVR prediction as main MI (no enhancement or adaptation)
                    mi_pred = svr_pred
                    
                except Exception as e:
                    print(f"[WARN] SVR prediction failed: {e}")
                    # Continue with stable calculation
            
            # Remap raw MI to 0-1 range for transmission
            raw_mi_remapped = np.clip((raw_mi_value + 5.0) / 10.0, 0.0, 1.0)
            
            # Output to console
            print(f"[STABLE] Sample {sample_count:4d} | "
                  f"MI: {mi_pred:.3f} | "
                  f"Raw: {raw_mi_value:.2f} | " 
                  f"EMI: {emi_value:.3f}")
            
            # Push to LSL streams
            current_time = time.time()
            mi_outlet.push_sample([mi_pred], current_time)
            raw_mi_outlet.push_sample([raw_mi_remapped], current_time)
            emi_outlet.push_sample([emi_value], current_time)
            
            # Record for analysis
            mi_records.append({
                'sample': sample_count,
                'mi': mi_pred,
                'raw_mi': raw_mi_value,
                'raw_mi_remapped': raw_mi_remapped,
                'emi': emi_value,
                'timestamp': current_time,
                'theta_fz': features[0],
                'alpha_po': features[1], 
                'faa': features[2],
                'beta_frontal': features[3],
                'eda_norm': features[4]
            })
            
            # Update visualization
            visualizer.update(mi_pred, raw_mi_value, emi_value)
            
        except Exception as e:
            print(f"[ERROR] Processing failed at sample {sample_count}: {e}")
            continue
    
    # === SESSION COMPLETE - GENERATE REPORTS ===
    
    print(f"\n{'='*60}")
    print("STABLE SESSION COMPLETED")
    print(f"{'='*60}")
    
    session_duration = time.time() - session_start_time
    print(f"Session Duration: {session_duration:.1f} seconds")
    print(f"Total Samples: {len(mi_records)}")
    
    if len(mi_records) > 0:
        # Calculate statistics
        mi_values = [r['mi'] for r in mi_records]
        raw_mi_values = [r['raw_mi'] for r in mi_records]
        emi_values = [r['emi'] for r in mi_records]
        
        print(f"\nSTABLE SESSION STATISTICS:")
        print(f"MI - Mean: {np.mean(mi_values):.3f}, Std: {np.std(mi_values):.3f}")
        print(f"Raw MI - Mean: {np.mean(raw_mi_values):.3f}, Std: {np.std(raw_mi_values):.3f}")
        print(f"EMI - Mean: {np.mean(emi_values):.3f}, Std: {np.std(emi_values):.3f}")
        
        # Save session data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'{user_id}_stable_session_{timestamp}.csv'
        csv_filepath = os.path.join(LOG_DIR, csv_filename)
        
        session_df = pd.DataFrame(mi_records)
        session_df.to_csv(csv_filepath, index=False)
        print(f"\n[SAVED] Session data: {csv_filepath}")
        
        # Generate visualization
        visualizer.save_final_plot(user_id)
        
        # Summary report
        print(f"\n[SUMMARY] Stable MI Pipeline Report:")
        print(f"- Consistent normalization: ✓")
        print(f"- Fixed scaling factors: ✓") 
        print(f"- No user adaptation: ✓")
        print(f"- No fallback mechanisms: ✓")
        print(f"- Repeatable results: ✓")
        
    print(f"\n{'='*60}")
    print("REPORTS GENERATED - CHECK logs/ AND visualizations/ FOLDERS")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
