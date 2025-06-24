"""
Real-time Mindfulness Index (MI) LSL Pipeline

- Expects RAW (unconverted, unnormalized) EEG and EDA data from LSL streams.
- Loads existing SVM/SVR model and scaler if available, else trains from EEG/EDA data.
- Sets up LSL streams for features (input), Unity labels (input), and MI output (output).
- Uses SGDRegressor for online MI adaptation.

Usage:
    python realtime_mi_lsl.py

Unity Integration Instructions:
- The script creates an LSL stream named 'processed_MI' (type 'MI', 1 channel, float32) that outputs the real-time Mindfulness Index (MI) for the current user/session.
- In your Unity project, use an LSL receiver (e.g., LSL4Unity or LabStreamingLayer.NET) to connect to the 'processed_MI' stream.
- The MI value is sent as a single float per sample. You can use this value for real-time feedback, visualization, or adaptive game logic.
- The MI stream is available as long as the Python script is running and processing data.
- If you want to record the MI stream, use LabRecorder or a similar LSL-compatible tool.

Dependencies:
    pip install numpy pandas scikit-learn pylsl joblib
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import load, dump
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
import scipy.signal
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'user_configs')
for d in [MODEL_DIR, LOG_DIR, VIS_DIR, PROCESSED_DATA_DIR, USER_CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
EEG_DIR = 'data/_eeg'
EDA_DIR = 'data/_eda'
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
MI_THRESHOLDS = {'focused': 0.5, 'neutral': 0.37}
# --- Online adaptation config ---
ONLINE_UPDATE_WINDOW = 10  # Number of samples in moving window
ONLINE_UPDATE_RATIO = 0.7  # Ratio of samples above/below threshold to trigger update
ONLINE_MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, '{}_online_model.joblib')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    Calibration step: Collect baseline (ground truth) data for a new user at 250 Hz.
    Expects RAW EEG (first 8 channels) and RAW ACC/Gyro (next 6 channels), and RAW EDA (2 channels).
    Uses LSL timestamps for all samples. EDA is downsampled/interpolated to 250 Hz.
    Also fits artifact regression models for each EEG channel using ACC/Gyro as regressors.
    """
    print(f"\n=== Calibration for user: {user_id} ===")
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
    print(f"Calibration will start in {countdown_sec} seconds. Please get ready...")
    for i in range(countdown_sec, 0, -1):
        print(f"Starting in {i}...", end='\r', flush=True)
        time.sleep(1)
    print("\nPlease relax (e.g., eyes closed) and remain still. Collecting baseline samples...")
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
        eda = np.array(eda_sample[:2])
        eda_feat = eda[1]
        eeg_samples.append(eeg)
        accgyr_samples.append(acc_gyr)
        eda_samples.append(eda)
        ts_samples.append(eeg_ts)
        # Only compute features every 1 second (every 250 samples)
        if len(eeg_samples) >= window_size and (i+1) % window_size == 0:
            eeg_win = np.array(eeg_samples[-window_size:])
            eda_win = np.array(eda_samples[-window_size:])
            sf = 250
            theta_fz = compute_bandpower(eeg_win[:,0], sf, (4,8))
            alpha_po = (compute_bandpower(eeg_win[:,6], sf, (8,13)) + compute_bandpower(eeg_win[:,7], sf, (8,13))) / 2
            faa = np.log(compute_bandpower(eeg_win[:,4], sf, (8,13)) + 1e-8) - np.log(compute_bandpower(eeg_win[:,5], sf, (8,13)) + 1e-8)
            beta_frontal = compute_bandpower(eeg_win[:,0], sf, (13,30))
            eda_norm = np.mean(eda_win[:,1])
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
    y_calib = np.array([calculate_mi(f) for f in X_calib])
    
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
        self.label_history = []
        self.timestamps = []
        self.last_metrics = {'precision': 1.0, 'recall': 1.0}
        self.fig, self.ax = plt.subplots()
        # Do not show the figure during the process

    def update(self, mi_pred, label=None):
        self.mi_history.append(mi_pred)
        self.timestamps.append(datetime.now())
        if label is not None:
            self.label_history.append(label)
        else:
            self.label_history.append(np.nan)
        # Do not plot during the process

    def final_plot(self):
        self.ax.clear()
        self.ax.plot(self.mi_history, label='MI Prediction')
        if any(~np.isnan(self.label_history)):
            self.ax.plot(self.label_history, label='Labels', linestyle='dashed')
        self.ax.set_title('Online MI Prediction')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('MI')
        self.ax.legend()
        plt.tight_layout()
        fname = os.path.join(VIS_DIR, f'final_online_mi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        self.fig.savefig(fname)
        print(f"[REPORT] Final MI plot saved to {fname}")
        plt.show()

    def log_metrics(self, y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score
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
    online_model.partial_fit(X_scaled, svr.predict(X_scaled))

    # Step 3: LSL setup
    feature_stream = select_lsl_stream('Features')
    feature_inlet = StreamInlet(feature_stream)
    label_stream = select_lsl_stream('UnityMarkers', confirm=False)
    label_inlet = StreamInlet(label_stream)
    # Create LSL output stream for MI
    mi_info = StreamInfo('processed_MI', 'MI', 1, 1, 'float32', 'mi_stream')
    mi_outlet = StreamOutlet(mi_info)
    print("Created LSL output stream 'processed_MI' for MI values")

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
        label, _ = label_inlet.pull_sample(timeout=0.01)
        if label:
            online_model.partial_fit(x_scaled, [float(label[0])])
            visualizer.update(mi_pred, float(label[0]))
            visualizer.log_metrics(np.array(visualizer.label_history), np.array(visualizer.mi_history))
        else:
            visualizer.update(mi_pred)
    print("Experiment complete.")

# --- MAIN ENTRY ---
def main():
    print("\n==============================")
    print("REAL-TIME MI LSL PIPELINE STARTING")
    print("==============================\n")
    print("[INFO] This script expects RAW (unconverted, unnormalized) EEG and EDA data from LSL streams.")
    print("[INFO] Do NOT pre-normalize or convert your EEG/EDA data before streaming to this script.")
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
    # --- New: Option to continue in offline/report mode if no LSL streams are found ---
    try:
        print("Resolving Unity label stream (type='UnityMarkers')...")
        label_stream = select_lsl_stream('UnityMarkers', confirm=False)
        label_inlet = StreamInlet(label_stream)
        print("Unity label stream connected.")
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        choice = input("No LSL label stream found. Type 'offline' to continue in offline/report mode, or press Enter to exit: ").strip().lower()
        if choice == 'offline':
            generate_offline_report()
            return
        else:
            print("Exiting.")
            return
    # Ask user for MI calculation rate and transmission interval
    try:
        mi_calc_rate_input = input("Enter MI calculation rate in Hz (default 10): ").strip()
        if mi_calc_rate_input == '':
            MI_CALC_RATE = 10
        else:
            MI_CALC_RATE = float(mi_calc_rate_input)
    except Exception:        MI_CALC_RATE = 10
    try:
        mi_update_interval_input = input("Enter MI transmission interval in seconds (default 3): ").strip()
        if mi_update_interval_input == '':
            MI_UPDATE_INTERVAL = 3.0
        else:
            MI_UPDATE_INTERVAL = float(mi_update_interval_input)
    except Exception:
        MI_UPDATE_INTERVAL = 3.0
        
    # Create LSL output stream for MI
    mi_info = StreamInfo('processed_MI', 'MI', 1, 1, 'float32', 'mi_stream')
    mi_outlet = StreamOutlet(mi_info)
    print("Created LSL output stream 'processed_MI' for MI values")

    visualizer = OnlineVisualizer()
    EEG_BUFFER, EDA_BUFFER, TS_BUFFER = [], [], []
    WINDOW_SIZE = 250  # 1 second at 250 Hz
    mi_window = []  # Moving window for MI predictions
    mi_records = []  # To store MI, timestamp, and state
    print("Entering real-time MI prediction loop at 1 Hz. Classification every 3 seconds.")
    # --- Automatic input data analysis and adaptation ---
    eeg_scale_factor = 0.001  # Scale down large EEG values by 1000x
    eda_scale_factor = 0.0001  # Scale down large EDA values by 10000x
    if eeg_inlet is not None or eda_inlet is not None:
        print("\n[INFO] Running automatic input data analysis for EEG/EDA streams...")
        # Analyze and adapt scaling if needed
        # Import numpy here to avoid scope issues
        import numpy as np
        analysis_eeg_vals = []
        analysis_eda_vals = []
        n_samples = 500
        for _ in range(n_samples):
            if eeg_inlet is not None:
                eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.5)
                if eeg_sample is not None:
                    analysis_eeg_vals.append(np.array(eeg_sample[:8]))
            if eda_inlet is not None:
                eda_sample, _ = eda_inlet.pull_sample(timeout=0.5)
                if eda_sample is not None:
                    analysis_eda_vals.append(np.array(eda_sample[:2]))
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
            print(f"  Min: {np.min(eda_arr):.5f}, Max: {np.max(eda_arr):.5f}, Mean: {np.mean(eda_arr):.5f}, Std: {np.std(eda_arr):.5f}")
            if np.issubdtype(eda_arr.dtype, np.integer):
                print("  [WARN] EDA data appears to be integer. RAW (unconverted, unnormalized) EDA is expected.")
            if np.nanmax(np.abs(eda_arr)) < 0.01:
                print("  [WARN] EDA data values are very small (<0.01). RAW EDA is expected. Check your LSL stream.")
            if np.nanmax(np.abs(eda_arr)) > 10:
                print("  [WARN] EDA data values are very large (>10). Check for scaling or units.")

    # In the real-time MI prediction loop, apply scaling before feature extraction
    # Replace in the loop:
    # eeg = np.array(eeg_sample[:8])
    # ...
    # eda = np.array(eda_sample[:2])
    # ...
    # With:
    # eeg = np.array(eeg_sample[:8]) * eeg_scale_factor
    # ...
    # eda = np.array(eda_sample[:2]) * eda_scale_factor
    # ...
    # (Apply this in both calibration and real-time prediction)

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
    print(f"Entering real-time MI prediction loop at 1 Hz. Classification window: 3 seconds.\n")
    while not stop_flag['stop']:
        now = time.time()
        if now < next_calc_time:
            time.sleep(max(0, next_calc_time - now))
            continue
        next_calc_time += 1.0  # 1 Hz
        # Collect 250 samples for 1-second window
        eeg_win_buf, eda_win_buf, ts_win_buf = [], [], []
        for _ in range(WINDOW_SIZE):
            if eeg_inlet is not None:
                eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=1.0)
                eeg = np.array(eeg_sample[:8]) * eeg_scale_factor
                acc_gyr = np.array(eeg_sample[8:14])
                if artifact_regressors is not None:
                    eeg_clean = apply_artifact_regression(eeg, acc_gyr, artifact_regressors)
                else:
                    eeg_clean = eeg
            else:
                eeg_clean = np.zeros(8)
                eeg_ts = time.time()
            if eda_inlet is not None:
                eda_sample, eda_ts = eda_inlet.pull_sample(timeout=1.0)
                eda = np.array(eda_sample[:2]) * eda_scale_factor
            else:
                eda = np.zeros(2)
                eda_ts = eeg_ts
            eeg_win_buf.append(eeg_clean)
            eda_win_buf.append(eda)
            ts_win_buf.append(eeg_ts)
        eeg_win = np.array(eeg_win_buf)
        eda_win = np.array(eda_win_buf)
        # --- Feature extraction (windowed, real) ---
        sf = 250
        theta_fz = compute_bandpower(eeg_win[:,0], sf, (4,8))
        alpha_po = (compute_bandpower(eeg_win[:,6], sf, (8,13)) + compute_bandpower(eeg_win[:,7], sf, (8,13))) / 2
        faa = np.log(compute_bandpower(eeg_win[:,4], sf, (8,13)) + 1e-8) - np.log(compute_bandpower(eeg_win[:,5], sf, (8,13)) + 1e-8)
        beta_frontal = compute_bandpower(eeg_win[:,0], sf, (13,30))
        eda_norm = np.mean(eda_win[:,1])
        features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
        sample = np.array(features).reshape(1, -1)
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
                    # Recalculate with the calculate_mi formula to compare with model
                    recalc_mi = calculate_mi(sample[0])
                    print(f"[DEBUG] Direct MI calculation: {recalc_mi}, Model prediction: {mi_pred}")
                    print(f"[DEBUG] Difference: {abs(recalc_mi - mi_pred):.6f}")
                    if abs(recalc_mi - mi_pred) > 0.1:
                        print("[DEBUG] Large difference between direct calculation and model! Check model training.")
                    print("")
        # --- Real-time classification printout ---
        if mi_pred >= 0.5:
            state = "Focused"
        elif mi_pred >= 0.37:
            state = "Neutral"
        else:
            state = "Unfocused"
        print(f"MI: {mi_pred:.3f} | State: {state}")
        mi_buffer.append(mi_pred)
        if skipped_reason:
            if 'mi_skipped_count' not in locals():
                mi_skipped_count = {}
            mi_skipped_count[skipped_reason] = mi_skipped_count.get(skipped_reason, 0) + 1
        # Only push to LSL once per second (1 Hz)
        ts = ts_win_buf[-1]
        print(f"Pushed MI: {mi_pred:.3f} | State: {state} (to processed_MI stream) [DEBUG]")
        mi_outlet.push_sample([mi_pred], ts)
        mi_records.append({'mi': mi_pred, 'timestamp': ts, 'state': state})
        label, label_ts = label_inlet.pull_sample(timeout=0.01)
        if label:
            try:
                label_val = float(label[0])
                visualizer.update(mi_pred, label_val)
            except (ValueError, TypeError):
                visualizer.update(mi_pred, None)
        else:
            visualizer.update(mi_pred)
    # --- After session: Save MI CSV and print report ---
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    mi_csv_path = os.path.join(LOG_DIR, f'{user_id}_mi_session_{session_time}.csv')
    # Save all features and state if available
    session_df = pd.DataFrame(mi_records)
    # If features are available, add them to the DataFrame
    if 'features' in locals() and isinstance(features, list) and len(features) == 5:
        # If features were collected in mi_records, add them
        for i, feat_name in enumerate(FEATURE_ORDER):
            session_df[feat_name] = [rec.get(feat_name, None) if isinstance(rec, dict) else None for rec in mi_records]
    session_df.to_csv(mi_csv_path, index=False)
    print(f"\n[REPORT] MI session data saved to {mi_csv_path}")
    # Print summary of MI prediction skips if any
    if 'mi_skipped_count' in locals() and mi_skipped_count:
        print("\n[SUMMARY] MI predictions skipped due to feature issues:")
        for reason, count in mi_skipped_count.items():
            print(f"  {reason}: {count} windows skipped")
    mi_vals = [r['mi'] for r in mi_records]
    if mi_vals:
        mi_arr = np.array(mi_vals)
        summary = {
            'user_id': user_id,
            'session_time': session_time,
            'n_samples': len(mi_arr),
            'mi_mean': np.mean(mi_arr),
            'mi_std': np.std(mi_arr),
            'mi_min': np.min(mi_arr),
            'mi_max': np.max(mi_arr),
            'focused_pct': 100 * sum(v >= 0.5 for v in mi_arr) / len(mi_arr),
            'neutral_pct': 100 * sum((v >= 0.37) and (v < 0.5) for v in mi_arr),
            'unfocused_pct': 100 * sum(v < 0.37 for v in mi_arr) / len(mi_arr)
        }
        report_path = os.path.join(LOG_DIR, f'{user_id}_mi_report_{session_time}.csv')
        pd.DataFrame([summary]).to_csv(report_path, index=False)
        print(f"\n[REPORT] Session summary saved to {report_path}")

        # --- Wilcoxon Signed-Rank Test for phases (if available) ---
        try:
            import scipy.stats as stats
            if 'phase' in session_df.columns:
                phases = session_df['phase'].dropna().unique()
                if len(phases) == 2:
                    vals1 = session_df[session_df['phase'] == phases[0]]['mi'].dropna()
                    vals2 = session_df[session_df['phase'] == phases[1]]['mi'].dropna()
                    if len(vals1) > 0 and len(vals2) > 0:
                        stat, p = stats.wilcoxon(vals1, vals2)
                        wilcoxon_result = {
                            'phase1': phases[0],
                            'phase2': phases[1],
                            'wilcoxon_stat': stat,
                            'wilcoxon_p': p
                        }
                        wilcoxon_path = os.path.join(LOG_DIR, f'{user_id}_wilcoxon_{session_time}.csv')
                        pd.DataFrame([wilcoxon_result]).to_csv(wilcoxon_path, index=False)
                        print(f"[REPORT] Wilcoxon Signed-Rank Test saved to {wilcoxon_path}")
        except Exception as e:
            print(f"[WARN] Wilcoxon test not computed: {e}")

        # --- Spearman's rank correlation (features vs MI) ---
        try:
            spearman_results = []
            for feat in FEATURE_ORDER:
                if feat in session_df.columns:
                    corr, p = stats.spearmanr(session_df[feat], session_df['mi'], nan_policy='omit')
                    spearman_results.append({'feature': feat, 'spearman_corr': corr, 'spearman_p': p})
            if spearman_results:
                spearman_path = os.path.join(LOG_DIR, f'{user_id}_spearman_{session_time}.csv')
                pd.DataFrame(spearman_results).to_csv(spearman_path, index=False)
                print(f"[REPORT] Spearman correlations saved to {spearman_path}")
        except Exception as e:
            print(f"[WARN] Spearman correlation not computed: {e}")

        # --- Real-time classification comparative (only at end) ---
        # If Unity labels were received, compare MI predictions to labels
        label_vals = [l for l in visualizer.label_history if not np.isnan(l)]
        if label_vals:
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_true_bin = [bin_mi(val) for val in label_vals]
            y_pred_bin = [bin_mi(val) for val in visualizer.mi_history[:len(y_true_bin)]]
            precision = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            print(f"[REAL-TIME CLASSIFICATION REPORT] Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

            # --- Save real-time classification report ---
            rt_report_path = os.path.join(LOG_DIR, f'{user_id}_mi_realtime_classification_report_{session_time}.csv')
            pd.DataFrame([{
                'user_id': user_id,
                'session_time': session_time,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }]).to_csv(rt_report_path, index=False)
            print(f"[REPORT] Real-time classification report saved to {rt_report_path}")

        # --- Show and save final MI plot and features plot ---
        visualizer.final_plot()
        # Additional: Plot MI, features, and state/label over time for the session
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            # Reload session_df to ensure all columns
            session_df = pd.read_csv(mi_csv_path)
            fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
            axes[0].plot(session_df['mi'], label='MI', color='blue')
            axes[0].set_ylabel('MI')
            axes[0].legend()
            for i, feat in enumerate(FEATURE_ORDER):
                if feat in session_df.columns:
                    axes[i+1].plot(session_df[feat], label=feat)
                    axes[i+1].set_ylabel(feat)
                    axes[i+1].legend()
            if 'state' in session_df.columns:
                axes[-1].plot(session_df['state'], label='State', color='purple')
                axes[-1].set_ylabel('State')
                axes[-1].legend()
            elif 'label' in session_df.columns:
                axes[-1].plot(session_df['label'], label='Label', color='purple')
                axes[-1].set_ylabel('Label')
                axes[-1].legend()
            axes[-1].set_xlabel('Window')
            plt.tight_layout()
            plot_path = os.path.join(LOG_DIR, f'{user_id}_mi_features_state_{session_time}.png')
            plt.savefig(plot_path)
            print(f"[REPORT] MI, features, and state/label plot saved to {plot_path}")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Could not generate MI/features/state plot: {e}")
def apply_artifact_regression(eeg, acc_gyr, artifact_regressors):
    """
    Remove predicted artifact from each EEG channel using linear regression coefficients.
    Args:
        eeg: np.array, shape (8,) - raw EEG channels
        acc_gyr: np.array, shape (6,) - raw ACC/Gyro channels
        artifact_regressors: list of dicts or np.array, len 8, each with 'coef' and 'intercept'
    Returns:
        eeg_clean: np.array, shape (8,)
    """
    eeg_clean = np.zeros_like(eeg)
    for ch in range(8):
        if artifact_regressors is not None and len(artifact_regressors) > ch:
            reg = artifact_regressors[ch]
            # Support both dict and sklearn LinearRegression
            if isinstance(reg, dict):
                coef = np.array(reg.get('coef', np.zeros(6)))
                intercept = reg.get('intercept', 0.0)
            else:
                coef = np.array(getattr(reg, 'coef_', np.zeros(6)))
                intercept = getattr(reg, 'intercept_', 0.0)
            predicted_artifact = np.dot(coef, acc_gyr) + intercept
            eeg_clean[ch] = eeg[ch] - predicted_artifact
        else:
            eeg_clean[ch] = eeg[ch]
    return eeg_clean

def compute_bandpower(data, sf, band, window_sec=None, relative=False):
    """
    Compute the average power of the signal x in a specific frequency band.
    data: 1D numpy array (samples,)
    sf: float, sampling frequency
    band: tuple, (low, high)
    window_sec: float or None
    relative: bool, return relative power
    """
    # Handle extremely large values - auto-scale if necessary
    max_value = np.max(np.abs(data))
    if max_value > 100000:  # Extremely high values
        scale_factor = 10000.0 / max_value
        scaled_data = data * scale_factor
        print(f"[WARN] Auto-scaling EEG data (max={max_value:.1f}) by factor {scale_factor:.6f}")
    elif max_value < 0.01 and max_value > 0:  # Extremely low values
        scale_factor = 1.0 / max_value 
        scaled_data = data * scale_factor
        print(f"[WARN] Auto-scaling EEG data (max={max_value:.6f}) by factor {scale_factor:.2f}")
    else:
        scaled_data = data
    
    band = np.asarray(band)
    low, high = band
    if window_sec is not None:
        nperseg = int(window_sec * sf)
    else:
        nperseg = min(256, len(scaled_data))
    
    try:
        freqs, psd = scipy.signal.welch(scaled_data, sf, nperseg=nperseg)
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        
        # Using scipy.integrate.trapezoid instead of deprecated np.trapz
        try:
            from scipy.integrate import trapezoid
            bp = trapezoid(psd[idx_band], dx=freq_res)
            if relative:
                bp /= trapezoid(psd, dx=freq_res)
        except ImportError:
            # Fall back to np.trapz if needed
            bp = np.trapz(psd[idx_band], dx=freq_res)
            if relative:
                bp /= np.trapz(psd, dx=freq_res)
    
        # Use log scale for large power values to reduce range
        if bp > 10000:
            bp = np.log10(bp)
            
        return bp
    
    except Exception as e:
        print(f"[ERROR] Error in compute_bandpower: {e}")
        return 1.0  # Return small default value instead of error

def resample_eda(eda_buffer, eda_timestamps, target_timestamps):
    """
    Resample/interpolated EDA to match target timestamps (EEG timestamps).
    eda_buffer: list of [2,] arrays
    eda_timestamps: list of floats
    target_timestamps: list of floats (EEG timestamps)
    Returns: np.array of shape (len(target_timestamps), 2)
    """
    eda_buffer = np.array(eda_buffer)
    eda_timestamps = np.array(eda_timestamps)
    target_timestamps = np.array(target_timestamps)
    eda_resampled = np.zeros((len(target_timestamps), 2))
    for ch in range(2):
        eda_resampled[:, ch] = np.interp(target_timestamps, eda_timestamps, eda_buffer[:, ch])
    return eda_resampled

# --- Analyze input EEG and EDA ranges at the start ---
def analyze_input_streams(eeg_inlet, eda_inlet, n_samples=500):
    eeg_vals = []
    eda_vals = []
    for _ in range(n_samples):
        if eeg_inlet is not None:
            eeg_sample, _ = eeg_inlet.pull_sample(timeout=0.5)
            if eeg_sample is not None:
                eeg_vals.append(np.array(eeg_sample[:8]))
        if eda_inlet is not None:
            eda_sample, _ = eda_inlet.pull_sample(timeout=0.5)
            if eda_sample is not None:
                eda_vals.append(np.array(eda_sample[:2]))
    if eeg_vals:
        eeg_arr = np.vstack(eeg_vals)
        print("\n[EEG RAW DATA ANALYSIS]")
        print(f"  Shape: {eeg_arr.shape}")
        print(f"  Min: {np.min(eeg_arr):.3f}, Max: {np.max(eeg_arr):.3f}, Mean: {np.mean(eeg_arr):.3f}, Std: {np.std(eeg_arr):.3f}")
        if np.issubdtype(eeg_arr.dtype, np.integer):
            print("  [WARN] EEG data appears to be integer. RAW (unconverted, unnormalized) EEG is expected.")
        if np.nanmax(np.abs(eeg_arr)) < 1.0:
            print("  [WARN] EEG data values are very small (<1.0). RAW EEG is expected. Check your LSL stream.")
        if np.nanmax(np.abs(eeg_arr)) > 1000:
            print("  [WARN] EEG data values are very large (>1000). Check for amplifier scaling or units.")
    if eda_vals:
        eda_arr = np.vstack(eda_vals)
        print("\n[EDA RAW DATA ANALYSIS]")
        print(f"  Shape: {eda_arr.shape}")
        print(f"  Min: {np.min(eda_arr):.5f}, Max: {np.max(eda_arr):.5f}, Mean: {np.mean(eda_arr):.5f}, Std: {np.std(eda_arr):.5f}")
        if np.issubdtype(eda_arr.dtype, np.integer):
            print("  [WARN] EDA data appears to be integer. RAW (unconverted, unnormalized) EDA is expected.")
        if np.nanmax(np.abs(eda_arr)) < 0.01:
            print("  [WARN] EDA data values are very small (<0.01). RAW EDA is expected. Check your LSL stream.")
        if np.nanmax(np.abs(eda_arr)) > 10:
            print("  [WARN] EDA data values are very large (>10). Check for scaling or units.")

def generate_offline_report():
    print("\n[OFFLINE MODE] Generating report from available files...")
    import glob
    import pandas as pd
    import os
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    USER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_configs')
    # MI session summary
    mi_files = glob.glob(os.path.join(LOG_DIR, '*_mi_session_*.csv'))
    if mi_files:
        print(f"Found {len(mi_files)} MI session files. Generating summary...")
        summary = []
        for f in mi_files:
            try:
                df = pd.read_csv(f)
                if df.empty or not set(['mi']).issubset(df.columns):
                    print(f"[WARN] Skipping empty or invalid MI session file: {os.path.basename(f)}")
                    continue
                mi_mean = df['mi'].mean()
                mi_std = df['mi'].std()
                focused_pct = (df['mi'] >= 0.5).mean() * 100
                neutral_pct = ((df['mi'] >= 0.37) & (df['mi'] < 0.5)).mean() * 100
                unfocused_pct = (df['mi'] < 0.37).mean() * 100
                summary.append({'file': os.path.basename(f), 'mi_mean': mi_mean, 'mi_std': mi_std,
                                'focused_pct': focused_pct, 'neutral_pct': neutral_pct, 'unfocused_pct': unfocused_pct})
            except Exception as e:
                print(f"[WARN] Could not process {os.path.basename(f)}: {e}")
        if summary:
            summary_path = os.path.join(LOG_DIR, 'mi_summary_report.csv')
            pd.DataFrame(summary).to_csv(summary_path, index=False)
            print(f"MI session summary saved to {summary_path}")
        else:
            print("No valid MI session files found for summary.")
    else:
        print("No MI session files found in logs/.")
    # Calibration summary
    baseline_files = glob.glob(os.path.join(USER_CONFIG_DIR, '*_baseline.csv'))
    if baseline_files:
        print(f"Found {len(baseline_files)} calibration baseline files.")
    else:
        print("No calibration baseline files found in user_configs/.")
    # Model files
    model_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', '*.joblib'))
    if model_files:
        print(f"Found {len(model_files)} model/scaler files in models/.")
    else:
        print("No model/scaler files found in models/.")
    print("[OFFLINE MODE] Report generation complete.\n")

def merge_reports_to_excel_and_pdf(user_id=None):
    import pandas as pd
    import glob
    import os
    from datetime import datetime
    try:
        from fpdf import FPDF
    except ImportError:
        print("[WARN] fpdf not installed. PDF report will not be generated. Install with 'pip install fpdf'.")
        FPDF = None

    log_dir = LOG_DIR
    user = user_id if user_id else ''
    # Find latest session_time for this user
    mi_session_files = sorted(glob.glob(os.path.join(log_dir, f'{user}_mi_session_*.csv')))
    if not mi_session_files:
        print("[MERGE] No MI session files found to merge.")
        return
    latest_mi_session = mi_session_files[-1]
    session_time = latest_mi_session.split('_mi_session_')[-1].replace('.csv','')
    # Paths
    summary_path = os.path.join(log_dir, f'{user}_mi_report_{session_time}.csv')
    rt_class_path = os.path.join(log_dir, f'{user}_mi_realtime_classification_report_{session_time}.csv')
    calib_comp_files = sorted(glob.glob(os.path.join(log_dir, f'{user}_calibration_comparative_*.csv')))
    calib_comp_path = calib_comp_files[-1] if calib_comp_files else None

    # Model file info
    user_model_path = os.path.join(MODEL_DIR, f'{user}_svr_model.joblib')
    global_model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')
    if os.path.exists(user_model_path):
        model_used = user_model_path
    elif os.path.exists(global_model_path):
        model_used = global_model_path
    else:
        model_used = None
    model_note = ""
    if model_used:
        model_time = datetime.fromtimestamp(os.path.getmtime(model_used)).strftime('%Y-%m-%d %H:%M:%S')
        model_note = f"Model file: {os.path.basename(model_used)} (created: {model_time})"
    else:
        model_note = "Model file: Not found"

    # Ensure all output files are labeled with user and date/time to avoid overwriting
    excel_path = os.path.join(log_dir, f'{user}_merged_report_{session_time}.xlsx')
    # Write Excel
    with pd.ExcelWriter(excel_path) as writer:
        pd.read_csv(latest_mi_session).to_excel(writer, sheet_name='MI Session', index=False)
        if os.path.exists(summary_path):
            pd.read_csv(summary_path).to_excel(writer, sheet_name='Session Summary', index=False)
        if os.path.exists(rt_class_path):
            pd.read_csv(rt_class_path).to_excel(writer, sheet_name='Real-time Classification', index=False)
        if calib_comp_path and os.path.exists(calib_comp_path):
            pd.read_csv(calib_comp_path).to_excel(writer, sheet_name='Calibration Comparative', index=False)
        # Add model note as a separate sheet
        pd.DataFrame([{"Model Info": model_note}]).to_excel(writer, sheet_name='Model Info', index=False)
    print(f"[MERGE] Merged Excel report saved to {excel_path}")

    # PDF summary
    if FPDF is not None:
        pdf_path = os.path.join(log_dir, f'{user}_merged_report_{session_time}.pdf')
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Mindfulness Index Session Report', ln=1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'User: {user}', ln=1)
        pdf.cell(0, 10, f'Session Time: {session_time}', ln=1)
        pdf.cell(0, 10, model_note, ln=1)
        # Add summaries
        if os.path.exists(summary_path):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Session Summary:', ln=1)
            pdf.set_font('Arial', '', 12)
            df = pd.read_csv(summary_path)
            for col in df.columns:
                pdf.cell(0, 8, f'{col}: {df.iloc[0][col]}', ln=1)
        if os.path.exists(rt_class_path):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Real-time Classification:', ln=1)
            pdf.set_font('Arial', '', 12)
            df = pd.read_csv(rt_class_path)
            for col in df.columns:
                pdf.cell(0, 8, f'{col}: {df.iloc[0][col]}', ln=1)
        # Compare user-specific and global calibration if both exist
        global_calib_files = sorted(glob.glob(os.path.join(log_dir, f'calibration_comparative_*.csv')))
        user_calib_mae, user_calib_r2 = None, None
        global_calib_mae, global_calib_r2 = None, None
        if calib_comp_path and os.path.exists(calib_comp_path):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Calibration Comparative:', ln=1)
            pdf.set_font('Arial', '', 12)
            df = pd.read_csv(calib_comp_path)
            for col in df.columns:
                pdf.cell(0, 8, f'{col}: {df.iloc[0][col]}', ln=1)
            if 'new_mae' in df.columns:
                user_calib_mae = df['new_mae'].iloc[0]
            if 'new_r2' in df.columns:
                user_calib_r2 = df['new_r2'].iloc[0]
        # Find global calibration for comparison
        for f in global_calib_files:
            if user not in os.path.basename(f):  # crude filter for global
                gdf = pd.read_csv(f)
                if 'new_mae' in gdf.columns:
                    global_calib_mae = gdf['new_mae'].iloc[0]
                if 'new_r2' in gdf.columns:
                    global_calib_r2 = gdf['new_r2'].iloc[0]
                break
        if user_calib_mae is not None and global_calib_mae is not None:
            pdf.set_font('Arial', 'B', 13)
            pdf.cell(0, 10, 'User vs Global Model Comparison:', ln=1)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'User Model MAE: {user_calib_mae:.4f} | Global Model MAE: {global_calib_mae:.4f}', ln=1)
            pdf.cell(0, 8, f'User Model R²: {user_calib_r2:.4f} | Global Model R²: {global_calib_r2:.4f}', ln=1)
        pdf.output(pdf_path)
        print(f"[MERGE] Merged PDF summary saved to {pdf_path}")
    else:
        print("[MERGE] PDF summary not generated (fpdf not installed).")

# IMPORTANT: Always use the global scaler for all users.
# Only the SVR model is adapted per user after calibration.
# This ensures consistent feature scaling and reliable predictions.

if __name__ == "__main__":
    # Ask if user wants to run diagnostics
    diagnostics_choice = input("\nDo you want to run MI calculation diagnostics? (y/n): ").strip().lower()
    if diagnostics_choice == 'y':
        diagnose_mi_calculation()
    main()

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
plt.show()
# Optional: Pairplot/correlations
sns.pairplot(df[feature_names])
plt.savefig('feature_pairplot.png')
plt.show()
"""

def diagnose_mi_calculation():
    """
    Run diagnostic tests on MI calculation with sample data to identify issues.
    Use this function when getting consistent 0 values for MI.
    """
    print("\n====== MI CALCULATION DIAGNOSTIC REPORT ======\n")
    
    # 1. Test with ideal values that should give high MI
    ideal_features = np.array([100, 50, 0.5, 10, 1])  # High theta, alpha, moderate FAA, etc.
    print("TEST CASE 1: Ideal values that should give high MI")
    ideal_mi = calculate_mi_debug(ideal_features)
    print(f"Result: MI = {ideal_mi}\n")
    
    # 2. Test with zero values
    zero_features = np.array([0, 0, 0, 0, 0])
    print("TEST CASE 2: All zero values")
    zero_mi = calculate_mi_debug(zero_features)
    print(f"Result: MI = {zero_mi}\n")
    
    # 3. Test with typical real data ranges
    typical_features = np.array([25, 10, 0.2, 15, 5])
    print("TEST CASE 3: Typical real data ranges")
    typical_mi = calculate_mi_debug(typical_features)
    print(f"Result: MI = {typical_mi}\n")
    
    # 4. Test with custom weight sets to see what produces better variations
    print("TEST CASE 4: Testing different weight configurations")
    test_weights = [
        [0.25, 0.25, 0.2, -0.15, -0.1],  # Default
        [0.5, 0.5, 0.5, -0.25, -0.25],   # Stronger weights
        [0.1, 0.1, 0.1, -0.05, -0.05],   # Weaker weights
    ]
    
    for i, weights in enumerate(test_weights):
        print(f"Weight set {i+1}: {weights}")
        weights = np.array(weights)
        raw_mi = np.dot(typical_features, weights) - 1
        mi = 1 / (1 + np.exp(-raw_mi))
        print(f"Result with typical features: MI = {mi}\n")
    
    # 5. Test with different offset values
    print("TEST CASE 5: Testing different offset values")
    offsets = [-2, -1, -0.5, 0, 0.5]
    for offset in offsets:
        raw_mi = np.dot(typical_features, np.array([0.25, 0.25, 0.2, -0.15, -0.1])) - offset
        mi = 1 / (1 + np.exp(-raw_mi))
        print(f"Offset {offset}: MI = {mi}")
    
    print("\n====== END OF DIAGNOSTIC REPORT ======\n")
    
    print("RECOMMENDATIONS:")
    print("1. If all test cases give 0, check for calculation issues or Python errors")
    print("2. If only your real data gives 0, your feature values may be outside expected ranges")
    print("3. Consider adjusting weights or offset based on the tests above")
    print("4. Add print statements in your feature extraction code to verify values")
    print("5. Examine your calibration data CSV to verify feature quality")
    
    # Return whether any of our tests produced reasonable values
    has_variation = abs(ideal_mi - zero_mi) > 0.1
    return has_variation

# You can run this by adding the following to your main() function or console:
# diagnose_mi_calculation()
