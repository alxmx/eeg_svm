#!/usr/bin/env python3
"""
STREAMLINED Real-time Mindfulness Index (MI) Pipeline with Dual Calibration
==========================================================================

FAST VERSION - Removes unnecessary complexity while maintaining robust EDA handling.

Key Features:
- Robust EDA stream detection and handling
- Dual calibration (relaxed + focused)
- Real-time MI calculation with adaptive thresholds
- Proper error handling for missing EDA streams
- Simplified code structure for faster execution
"""

import os
import sys
import time
import json
import threading
import warnings
import numpy as np
try:
    import pandas as pd
except ImportError:
    print("[WARN] pandas not available - some features may be limited")
    pd = None

try:
    from scipy.signal import welch
except ImportError:
    print("[WARN] scipy not available - using basic bandpower calculation")
    welch = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[WARN] matplotlib not available - no plotting")
    plt = None

from datetime import datetime

# LSL imports
try:
    from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams, resolve_byprop
    print("[OK] LSL library imported successfully")
except ImportError as e:
    print(f"[ERROR] LSL library not found: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
FEATURE_ORDER = ['theta_fz', 'beta_fz', 'alpha_c3', 'alpha_c4', 'faa_c3c4', 'alpha_pz', 'alpha_po', 'alpha_oz', 'eda_norm']

EEG_CHANNELS = {'Fz': 0, 'C3': 1, 'Cz': 2, 'C4': 3, 'Pz': 4, 'PO7': 5, 'PO8': 6, 'Oz': 7}
EDA_CHANNEL_INDEX = 1  # Channel 1 contains actual EDA data; Channel 0 contains timestamps

# Directories
for directory in ['models', 'logs', 'visualizations', 'user_configs']:
    os.makedirs(directory, exist_ok=True)

# === ROBUST LSL STREAM SELECTION ===
def select_lsl_stream(stream_type, allow_skip=False):
    """Robust LSL stream selection with detailed debugging"""
    print(f"\n[LSL] Looking for {stream_type} streams...")
    
    streams = []
    
    if stream_type == 'EEG':
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
        if not streams:
            print("[DEBUG] No 'EEG' type streams, trying name-based detection...")
            all_streams = resolve_streams(timeout=5.0)
            streams = [s for s in all_streams if 'unicorn' in s.name().lower() or s.channel_count() >= 8]
    
    elif stream_type == 'EDA':
        # Try multiple EDA detection methods
        print("[DEBUG] Trying resolve_byprop('type', 'EDA')...")
        streams = resolve_byprop('type', 'EDA', timeout=5.0)
        
        if not streams:
            print("[DEBUG] No 'EDA' type found, trying alternative types...")
            for alt_type in ['BioSignals', 'GSR', 'Physiological', 'OpenSignals']:
                print(f"[DEBUG] Trying type '{alt_type}'...")
                streams = resolve_byprop('type', alt_type, timeout=2.0)
                if streams:
                    print(f"[INFO] Found EDA stream with type '{alt_type}'")
                    break
        
        if not streams:
            print("[DEBUG] Trying name-based detection...")
            all_streams = resolve_streams(timeout=5.0)
            print(f"[DEBUG] Found {len(all_streams)} total streams")
            streams = [s for s in all_streams if any(keyword in s.name().lower() 
                      for keyword in ['eda', 'opensignals', 'gsr', 'bio', 'physiological'])]
        
        if not streams:
            print("[DEBUG] Last resort: checking all 2-channel streams...")
            all_streams = resolve_streams(timeout=5.0)
            streams = [s for s in all_streams if s.channel_count() == 2]
            if streams:
                print(f"[INFO] Found {len(streams)} 2-channel stream(s) that might be EDA:")
                for i, s in enumerate(streams):
                    print(f"  {i+1}. {s.name()} (type: {s.type()}, channels: {s.channel_count()})")
    
    if not streams:
        print(f"[WARN] No {stream_type} streams found!")
        if allow_skip:
            return None
        else:
            print(f"[ERROR] {stream_type} stream is required!")
            sys.exit(1)
    
    if len(streams) == 1:
        stream = streams[0]
        print(f"[AUTO] Using {stream_type} stream: {stream.name()} (channels: {stream.channel_count()})")
        if stream_type == 'EDA' and stream.channel_count() != 2:
            print(f"[WARN] EDA stream has {stream.channel_count()} channels, expected 2")
        return stream
    
    # Multiple streams - let user choose
    print(f"[CHOICE] Multiple {stream_type} streams found:")
    for i, stream in enumerate(streams):
        print(f"  {i+1}. {stream.name()} (type: {stream.type()}, channels: {stream.channel_count()})")
    
    while True:
        try:
            choice = input(f"Select {stream_type} stream (1-{len(streams)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(streams):
                selected = streams[idx]
                print(f"[SELECTED] {stream_type}: {selected.name()}")
                return selected
        except ValueError:
            print("Please enter a valid number.")

# === FEATURE EXTRACTION ===
def compute_bandpower(data, sf, band):
    """Compute bandpower using Welch's method or simple FFT fallback"""
    try:
        if welch is not None:
            f, psd = welch(data, sf, nperseg=min(len(data), sf))
            idx_band = np.logical_and(f >= band[0], f <= band[1])
            return max(np.trapz(psd[idx_band], f[idx_band]), 1e-8)
        else:
            # Simple FFT-based bandpower calculation
            fft_data = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data), 1/sf)
            power = np.abs(fft_data) ** 2
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
            return max(np.mean(power[idx_band]), 1e-8)
    except:
        return 1.0

def extract_features(eeg_window, eda_window):
    """Extract mindfulness features"""
    sf = 250
    
    # Safety checks
    if len(eeg_window) == 0 or eeg_window.shape[1] < 8:
        return np.zeros(9)
    if len(eda_window) == 0:
        return np.zeros(9)
    
    # EEG features
    theta_fz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (4, 8))
    beta_fz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Fz']], sf, (13, 30))
    alpha_c3 = compute_bandpower(eeg_window[:, EEG_CHANNELS['C3']], sf, (8, 13))
    alpha_c4 = compute_bandpower(eeg_window[:, EEG_CHANNELS['C4']], sf, (8, 13))
    faa_c3c4 = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
    alpha_pz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Pz']], sf, (8, 13))
    alpha_po7 = compute_bandpower(eeg_window[:, EEG_CHANNELS['PO7']], sf, (8, 13))
    alpha_po8 = compute_bandpower(eeg_window[:, EEG_CHANNELS['PO8']], sf, (8, 13))
    alpha_po = (alpha_po7 + alpha_po8) / 2
    alpha_oz = compute_bandpower(eeg_window[:, EEG_CHANNELS['Oz']], sf, (8, 13))
    
    # EDA feature - ALWAYS use channel 1 (actual EDA data)
    if eda_window.shape[1] >= 2:
        raw_eda = np.mean(eda_window[:, EDA_CHANNEL_INDEX])  # Channel 1
        print(f"[DEBUG] EDA raw value: {raw_eda:.3f} (from channel {EDA_CHANNEL_INDEX})")
    else:
        print(f"[ERROR] EDA window has {eda_window.shape[1]} channels, expected 2")
        raw_eda = 0.0
    
    # Simple EDA normalization
    eda_norm = np.clip((raw_eda - 0) / 12, 0, 10)  # 0-12 range mapped to 0-10
    
    return np.array([theta_fz, beta_fz, alpha_c3, alpha_c4, faa_c3c4, alpha_pz, alpha_po, alpha_oz, eda_norm])

# === MI CALCULATION ===
def calculate_mi_universal(features):
    """Universal MI calculation"""
    weights = np.array([0.35, -0.08, 0.14, 0.14, 0.10, -0.20, 0.22, 0.15, -0.10])
    
    # Normalize features to 0-10 range
    ranges = {'theta_fz': (1, 80), 'beta_fz': (0.5, 15), 'alpha_c3': (2, 30), 'alpha_c4': (2, 30),
              'faa_c3c4': (-2.5, 2.5), 'alpha_pz': (2, 35), 'alpha_po': (1, 25), 'alpha_oz': (2, 25), 'eda_norm': (0, 12)}
    
    normalized = []
    for i, (feat_name, (q5, q95)) in enumerate(ranges.items()):
        val = 10 * (features[i] - q5) / (q95 - q5)
        normalized.append(np.clip(val, 0, 10))
    
    normalized = np.array(normalized)
    weighted_sum = np.dot(normalized, weights)
    
    # Adaptive centering based on EDA/EEG balance
    eda_norm = normalized[8]
    alpha_norm = np.mean(normalized[2:8])
    
    if eda_norm > 7:
        center_shift = -1.8
    elif alpha_norm > 6:
        center_shift = -1.0
    else:
        center_shift = -1.5
    
    centered_sum = weighted_sum + center_shift
    mi_sigmoid = 1 / (1 + np.exp(-2.5 * centered_sum))
    return np.clip(0.1 + 0.8 * mi_sigmoid, 0.1, 0.9)

# === CALIBRATION SYSTEM ===
def collect_calibration_phase(eeg_inlet, eda_inlet, duration_sec, phase_name):
    """Collect calibration data for one phase"""
    print(f"\n[CALIBRATION] {phase_name.upper()} PHASE ({duration_sec}s)")
    
    if phase_name == 'relaxed':
        print("📋 INSTRUCTIONS:")
        print("• Close your eyes and relax completely")
        print("• Take slow, deep breaths")
        print("• Let your mind rest - don't try to meditate")
        print("• Just sit comfortably and be at ease")
    else:
        print("📋 INSTRUCTIONS:")  
        print("• Open your eyes and focus on this text")
        print("• Maintain steady, focused attention")
        print("• Count your breaths from 1 to 10, then repeat")
        print("• Stay alert and engaged")
    
    input("Press Enter when ready...")
    print("Starting in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1) 
    print("1...")
    time.sleep(1)
    print("BEGIN!")
    
    features_list = []
    eeg_buffer, eda_buffer = [], []
    window_size = 250
    
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        # Collect EEG
        if eeg_inlet:
            eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.1)
            if eeg_sample:
                eeg_buffer.append(eeg_sample[:8])
        
        # Collect EDA (handle missing stream gracefully)
        if eda_inlet:
            eda_sample, _ = eda_inlet.pull_sample(timeout=0.1)
            if eda_sample:
                eda_buffer.append(eda_sample[:2])  # [timestamp, eda_data]
            else:
                eda_buffer.append([0, 0])
        else:
            eda_buffer.append([0, 0])  # Use zeros if no EDA stream
        
        # Extract features every second
        if len(eeg_buffer) >= window_size and len(eda_buffer) >= window_size:
            eeg_window = np.array(eeg_buffer[-window_size:])
            eda_window = np.array(eda_buffer[-window_size:])
            
            features = extract_features(eeg_window, eda_window)
            if not np.any(np.isnan(features)):
                features_list.append(features)
            
            # Show progress
            elapsed = time.time() - start_time
            print(f"Progress: {elapsed:.0f}s/{duration_sec}s", end='\r')
    
    print(f"\n✓ {phase_name.capitalize()} calibration complete: {len(features_list)} windows")
    return np.array(features_list) if features_list else None

def run_dual_calibration(user_id, eeg_inlet, eda_inlet):
    """Run dual calibration process"""
    print(f"\n{'='*60}")
    print("DUAL CALIBRATION PROCESS")
    print("="*60)
    
    # Phase 1: Relaxed
    relaxed_features = collect_calibration_phase(eeg_inlet, eda_inlet, 30, 'relaxed')
    if relaxed_features is None:
        print("[ERROR] Relaxed calibration failed!")
        return None
    
    time.sleep(2)  # Brief pause
    
    # Phase 2: Focused  
    focused_features = collect_calibration_phase(eeg_inlet, eda_inlet, 30, 'focused')
    if focused_features is None:
        print("[ERROR] Focused calibration failed!")
        return None
    
    # Compute adaptive thresholds
    relaxed_mi = [calculate_mi_universal(f) for f in relaxed_features]
    focused_mi = [calculate_mi_universal(f) for f in focused_features]
    
    relaxed_mean = np.mean(relaxed_mi)
    focused_mean = np.mean(focused_mi)
    dynamic_range = focused_mean - relaxed_mean
    
    print(f"\n[RESULTS] Calibration Summary:")
    print(f"  Relaxed MI: {relaxed_mean:.3f}")
    print(f"  Focused MI: {focused_mean:.3f}")
    print(f"  Dynamic Range: {dynamic_range:.3f}")
    
    # Create adaptive mapping
    adaptive_thresholds = {
        'relaxed_baseline': {'mi_mean': float(relaxed_mean)},
        'focused_baseline': {'mi_mean': float(focused_mean)},
        'adaptive_mapping': {
            'low_threshold': float(relaxed_mean),
            'high_threshold': float(focused_mean),
            'dynamic_range': float(dynamic_range)
        }
    }
    
    # Save calibration
    config_path = os.path.join('user_configs', f'{user_id}_calibration.json')
    with open(config_path, 'w') as f:
        json.dump(adaptive_thresholds, f, indent=2)
    
    print(f"✓ Calibration saved to {config_path}")
    return adaptive_thresholds

# === ADAPTIVE MI CALCULATOR ===
class AdaptiveMICalculator:
    def __init__(self, adaptive_thresholds):
        self.thresholds = adaptive_thresholds
        self.mi_history = []
        
    def calculate_adaptive_mi(self, features):
        universal_mi = calculate_mi_universal(features)
        
        if self.thresholds:
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
        
        # Light smoothing
        self.mi_history.append(adaptive_mi)
        if len(self.mi_history) > 3:
            self.mi_history.pop(0)
        
        return np.mean(self.mi_history), universal_mi

# === OUTPUT STREAMS ===
def setup_output_streams():
    """Setup LSL output streams"""
    mi_info = StreamInfo('mindfulness_index', 'MI', 1, 1, 'float32', 'mi_001')
    raw_mi_info = StreamInfo('raw_mindfulness_index', 'RawMI', 1, 1, 'float32', 'raw_mi_001')
    
    return {
        'mi': StreamOutlet(mi_info),
        'raw_mi': StreamOutlet(raw_mi_info)
    }

# === REAL-TIME PROCESSING ===
def run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator):
    """Run real-time MI processing"""
    print(f"\n{'='*60}")
    print("REAL-TIME MI PROCESSING")
    print("="*60)
    print("Press 'q' + Enter to stop...")
    
    eeg_buffer, eda_buffer = [], []
    window_size = 250
    mi_data = []
    
    start_time = time.time()
    
    try:
        # Simple input monitoring for Windows
        import msvcrt
        
        while True:
            # Check for quit command (Windows compatible)
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'q':
                    break
            
            # Collect data
            if eeg_inlet:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.01)
                if eeg_sample:
                    eeg_buffer.append(eeg_sample[:8])
            
            if eda_inlet:
                eda_sample, _ = eda_inlet.pull_sample(timeout=0.01)
                if eda_sample:
                    eda_buffer.append(eda_sample[:2])
            else:
                eda_buffer.append([0, 0])
            
            # Process when enough data
            if len(eeg_buffer) >= window_size and len(eda_buffer) >= window_size:
                eeg_window = np.array(eeg_buffer[-window_size:])
                eda_window = np.array(eda_buffer[-window_size:])
                
                features = extract_features(eeg_window, eda_window)
                adaptive_mi, universal_mi = mi_calculator.calculate_adaptive_mi(features)
                
                # Output to streams
                current_time = time.time()
                output_streams['mi'].push_sample([adaptive_mi], current_time)
                output_streams['raw_mi'].push_sample([universal_mi], current_time)
                
                # Display
                elapsed = time.time() - start_time
                print(f"Time: {elapsed:6.1f}s | Adaptive MI: {adaptive_mi:.3f} | Universal MI: {universal_mi:.3f}")
                
                # Log data
                mi_data.append({
                    'time': elapsed,
                    'adaptive_mi': adaptive_mi,
                    'universal_mi': universal_mi
                })
                
                # Trim buffers
                if len(eeg_buffer) > window_size * 2:
                    eeg_buffer = eeg_buffer[-window_size:]
                    eda_buffer = eda_buffer[-window_size:]
    
    except KeyboardInterrupt:
        pass
    
    # Save session data
    if mi_data:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_file = os.path.join('logs', f'{user_id}_session_{timestamp}.csv')
        pd.DataFrame(mi_data).to_csv(session_file, index=False)
        print(f"\n✓ Session data saved to {session_file}")

# === MAIN FUNCTION ===
def main():
    """Main function"""
    print("\n" + "="*60)
    print("STREAMLINED MINDFULNESS INDEX PIPELINE")
    print("="*60)
    
    # Get user ID
    user_id = input("Enter user ID: ").strip() or f"user_{int(time.time())}"
    print(f"User ID: {user_id}")
    
    # Connect to streams
    print("\n[SETUP] Connecting to data streams...")
    
    eeg_stream = select_lsl_stream('EEG', allow_skip=False)
    eeg_inlet = StreamInlet(eeg_stream)
    print(f"✓ EEG connected: {eeg_stream.name()}")
    
    eda_stream = select_lsl_stream('EDA', allow_skip=True)
    if eda_stream:
        eda_inlet = StreamInlet(eda_stream)
        print(f"✓ EDA connected: {eda_stream.name()} (channels: {eda_stream.channel_count()})")
        if eda_stream.channel_count() != 2:
            print(f"[WARN] Expected 2 EDA channels, got {eda_stream.channel_count()}")
    else:
        eda_inlet = None
        print("[WARN] No EDA stream - using zero values")
    
    # Setup output streams
    output_streams = setup_output_streams()
    print("✓ Output streams created")
    
    # Run calibration
    adaptive_thresholds = run_dual_calibration(user_id, eeg_inlet, eda_inlet)
    if not adaptive_thresholds:
        print("[ERROR] Calibration failed!")
        return
    
    # Initialize MI calculator
    mi_calculator = AdaptiveMICalculator(adaptive_thresholds)
    
    # Start real-time processing
    input("\nPress Enter to start real-time processing...")
    run_realtime_processing(user_id, eeg_inlet, eda_inlet, output_streams, mi_calculator)
    
    print("\n✓ Session complete!")

if __name__ == "__main__":
    main()
