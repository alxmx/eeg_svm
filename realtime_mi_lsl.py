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
import glob
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
import time
import scipy.signal

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
    weights = np.array([0.25, 0.25, 0.2, -0.15, -0.1])
    raw_mi = np.dot(features, weights) - 1
    mi = 1 / (1 + np.exp(-raw_mi))
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
    print(f"Collecting {n_samples} samples at 250 Hz for {calibration_duration_sec} seconds...")
    for i in range(n_samples):
        eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=1.0)
        eda_sample, eda_ts = eda_inlet.pull_sample(timeout=1.0)
        eeg = np.array(eeg_sample[:8])
        acc_gyr = np.array(eeg_sample[8:14])
        eda = np.array(eda_sample[:2])
        eeg_samples.append(eeg)
        accgyr_samples.append(acc_gyr)
        eda_samples.append(eda)
        ts_samples.append(eeg_ts)  # Use EEG timestamp as reference
        # Feature extraction placeholder (replace with your method)
        # theta_fz = np.mean(eeg)
        # alpha_po = np.mean(eeg)
        # faa = np.mean(eeg)
        # beta_frontal = np.mean(eeg)
        # eda_norm = np.mean(eda)
        # features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
        # processed_outlet.push_sample(features, eeg_ts)
        # Instead, use real features:
        # Assume EEG channels: [Fz, Cz, Pz, Oz, C3, C4, PO7, PO8] (adjust as needed)
        eeg_arr = np.array(eeg)
        sf = 250
        theta_fz = compute_bandpower(eeg_arr[0], sf, (4,8))  # Fz
        alpha_po = (compute_bandpower(eeg_arr[6], sf, (8,13)) + compute_bandpower(eeg_arr[7], sf, (8,13))) / 2  # PO7, PO8
        faa = np.log(compute_bandpower(eeg_arr[4], sf, (8,13)) + 1e-8) - np.log(compute_bandpower(eeg_arr[5], sf, (8,13)) + 1e-8)  # C3 - C4
        beta_frontal = compute_bandpower(eeg_arr[0], sf, (13,30))  # Fz
        eda_norm = np.mean(eda)
        features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
        processed_outlet.push_sample(features, eeg_ts)
        if i % 250 == 0:
            print(f"Collected {i} samples...")
    # Downsample/interpolate EDA to match EEG timestamps if needed
    eeg_samples = np.array(eeg_samples)  # shape (n_samples, 8)
    accgyr_samples = np.array(accgyr_samples)  # shape (n_samples, 6)
    eda_samples = np.array(eda_samples)
    ts_samples = np.array(ts_samples)
    # Save features as before
    baseline_arr = np.column_stack([
        eeg_samples.mean(axis=1),  # placeholder for theta_fz
        eeg_samples.mean(axis=1),  # placeholder for alpha_po
        eeg_samples.mean(axis=1),  # placeholder for faa
        eeg_samples.mean(axis=1),  # placeholder for beta_frontal
        eda_samples.mean(axis=1)   # placeholder for eda_norm
    ])
    baseline_arr = baseline_arr[~np.isnan(baseline_arr).any(axis=1)]
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
    print("[AUTO] Fine-tuning SVR model for this user based on calibration data, starting from generic model...")
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    calib_df = pd.read_csv(baseline_csv)
    X_calib = calib_df[FEATURE_ORDER].values
    y_calib = np.array([calculate_mi(f) for f in X_calib])

    # Warn if all MI values are 0 or 1 (likely feature/coding error)
    if np.all(y_calib == 0) or np.all(y_calib == 1):
        print("[ERROR] All calibration MI values are 0 or 1. This suggests a feature extraction or coding error. Skipping model update.")
        return baseline_csv, config_path

    # Filter out NaN samples before training
    valid_idx = ~np.isnan(X_calib).any(axis=1) & ~np.isnan(y_calib)
    X_calib = X_calib[valid_idx]
    y_calib = y_calib[valid_idx]
    if len(X_calib) == 0:
        print("[ERROR] All calibration samples are invalid (NaN). Skipping calibration and model update.")
        return baseline_csv, config_path
    # Load generic scaler and model as base
    generic_scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    generic_model_path = os.path.join(MODEL_DIR, 'svm_model.joblib')
    if os.path.exists(generic_scaler_path) and os.path.exists(generic_model_path):
        base_scaler = load(generic_scaler_path)
        base_model = load(generic_model_path)
        print("[INFO] Loaded generic model and scaler as base for user fine-tuning.")
    else:
        print("[WARN] Generic model/scaler not found. Training from scratch.")
        base_scaler = StandardScaler().fit(X_calib)
        base_model = SVR()

    # Fit scaler on user data (optionally could use partial_fit if available)
    scaler = StandardScaler().fit(X_calib)
    X_calib_scaled = scaler.transform(X_calib)

    # Fine-tune SVR: re-fit on user data, starting from generic model's parameters
    # (SVR does not support partial_fit, so we re-fit, but you could use warm_start for some estimators)
    svr = SVR()
    # Only set valid SVR parameters from base_model
    valid_params = svr.get_params().keys()
    base_params = {k: v for k, v in base_model.get_params().items() if k in valid_params}
    svr.set_params(**base_params)
    svr.fit(X_calib_scaled, y_calib)

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
    mi_info = StreamInfo('MI_Output', 'MI', 1, 10, 'float32', 'mi_stream')
    mi_outlet = StreamOutlet(mi_info)
    visualizer = OnlineVisualizer()

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
    # Calibration duration is always 60 seconds
    calibration_duration = 60
    print(f"Calibration will last {calibration_duration} seconds at 250 Hz.")
    calibrate = input("Run calibration step for this user? (y/n): ").strip().lower() == 'y'
    if calibrate:
        baseline_csv, config_path = calibrate_user(user_id, calibration_duration_sec=calibration_duration)
        if baseline_csv is None:
            print("[WARN] Calibration skipped or failed. Continuing with generic/global model.")
    # Load artifact regressors from user config if available
    artifact_regressors = None
    config_path = os.path.join(USER_CONFIG_DIR, f'{user_id}_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        artifact_regressors = user_config.get('artifact_regressors', None)
    print(f"Checking for user-specific model and scaler for user: {user_id}")
    user_model_path = os.path.join(MODEL_DIR, f'{user_id}_svr_model.joblib')
    user_scaler_path = os.path.join(MODEL_DIR, f'{user_id}_scaler.joblib')
    if os.path.exists(user_model_path) and os.path.exists(user_scaler_path):
        print(f"Loading user-specific model and scaler for user {user_id}...")
        svr = load(user_model_path)
        scaler = load(user_scaler_path)
        print("User-specific model and scaler loaded.")
    else:
        print(f"User-specific model/scaler not found. Loading or training global model...")
        svr, scaler = load_or_train_models()
        print("Model and scaler ready.")
    # Online adaptation is DISABLED in real-time operation per user request.
    print("[INFO] Online adaptation is DISABLED during real-time MI prediction. Only user-specific model/scaler will be used.")

    # Print which model/scaler version is being used for this session
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
    print("Resolving Unity label stream (type='UnityMarkers')...")
    label_stream = select_lsl_stream('UnityMarkers', confirm=False)
    label_inlet = StreamInlet(label_stream)
    print("Unity label stream connected.")
    print("Creating MI output LSL stream (type='processed_MI')...")
    mi_info = StreamInfo('processed_MI', 'MI', 1, 10, 'float32', 'mi_stream')
    mi_outlet = StreamOutlet(mi_info)
    print("MI output stream created as 'processed_MI'.\n")

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
        mi_update_interval_input = input("Enter MI transmission interval in seconds (default 3): ").strip()
        if mi_update_interval_input == '':
            MI_UPDATE_INTERVAL = 3.0
        else:
            MI_UPDATE_INTERVAL = float(mi_update_interval_input)
    except Exception:
        MI_UPDATE_INTERVAL = 3.0

    visualizer = OnlineVisualizer()
    EEG_BUFFER, EDA_BUFFER, TS_BUFFER = [], [], []
    WINDOW_SIZE = 3 * 250  # 3 seconds at 250 Hz
    mi_window = []  # Moving window for MI predictions
    mi_records = []  # To store MI, timestamp, and state
    print("Entering real-time MI prediction loop at 250 Hz. Classification every 3 seconds.")
    # ...existing code for threading, stop_flag, etc...
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
    mi_records = []  # To store MI, timestamp, and state
    next_calc_time = time.time()
    print(f"Entering real-time MI prediction loop at {MI_CALC_RATE} Hz. Classification window: 3 seconds.\n")
    while not stop_flag['stop']:
        now = time.time()
        if now < next_calc_time:
            time.sleep(max(0, next_calc_time - now))
            continue
        next_calc_time += 1.0 / MI_CALC_RATE
        if eeg_inlet is not None:
            eeg_sample, eeg_ts = eeg_inlet.pull_sample(timeout=1.0)
            eeg = np.array(eeg_sample[:8])
            acc_gyr = np.array(eeg_sample[8:14])
            # --- RAW DATA CHECK ---
            if np.issubdtype(eeg.dtype, np.integer):
                print("[WARN] EEG data appears to be integer. RAW (unconverted, unnormalized) EEG is expected.")
            elif np.nanmax(np.abs(eeg)) < 1.0:
                print("[WARN] EEG data values are very small (<1.0). RAW EEG is expected. Check your LSL stream.")
            if artifact_regressors is not None:
                eeg_clean = apply_artifact_regression(eeg, acc_gyr, artifact_regressors)
            else:
                eeg_clean = eeg  # fallback: no artifact regression
        else:
            eeg_clean = np.zeros(8)
            eeg_ts = time.time()
        if eda_inlet is not None:
            eda_sample, eda_ts = eda_inlet.pull_sample(timeout=1.0)
            eda = np.array(eda_sample[:2])
            # --- RAW DATA CHECK ---
            if np.issubdtype(eda.dtype, np.integer):
                print("[WARN] EDA data appears to be integer. RAW (unconverted, unnormalized) EDA is expected.")
            elif np.nanmax(np.abs(eda)) < 0.01:
                print("[WARN] EDA data values are very small (<0.01). RAW EDA is expected. Check your LSL stream.")
        else:
            eda = np.zeros(2)
            eda_ts = eeg_ts
        EEG_BUFFER.append(eeg_clean)
        EDA_BUFFER.append(eda)
        TS_BUFFER.append(eeg_ts)
        if len(EEG_BUFFER) > WINDOW_SIZE:
            EEG_BUFFER = EEG_BUFFER[-WINDOW_SIZE:]
            EDA_BUFFER = EDA_BUFFER[-WINDOW_SIZE:]
            TS_BUFFER = TS_BUFFER[-WINDOW_SIZE:]
        # Run MI calculation at MI_CALC_RATE
        if len(EEG_BUFFER) == WINDOW_SIZE:
            eeg_win = np.array(EEG_BUFFER)
            eda_win = np.array(EDA_BUFFER)
            # Feature extraction placeholder (replace with your method)
            # theta_fz = np.mean(eeg_win)
            # alpha_po = np.mean(eeg_win)
            # faa = np.mean(eeg_win)
            # beta_frontal = np.mean(eeg_win)
            # eda_norm = np.mean(eda_win)
            # features = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]
            # sample = np.array(features).reshape(1, -1)
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
            # Only push to LSL at MI_UPDATE_INTERVAL
            if (now - last_push_time) >= MI_UPDATE_INTERVAL:
                mean_mi = float(np.mean(mi_buffer)) if mi_buffer else 0.0
                ts = TS_BUFFER[-1]
                # State already determined above for last mi_pred
                print(f"Pushed mean MI: {mean_mi:.3f} | State: {state} (to processed_MI stream)")
                mi_outlet.push_sample([mean_mi], ts)
                mi_records.append({'mi': mean_mi, 'timestamp': ts, 'state': state})
                mi_buffer = []
                last_push_time = now
            label, label_ts = label_inlet.pull_sample(timeout=0.01)
            if label:
                try:
                    label_val = float(label[0])
                    visualizer.update(mi_pred, label_val)
                except (ValueError, TypeError):
                    # Non-numeric label, skip updating label history
                    visualizer.update(mi_pred, None)
            else:
                visualizer.update(mi_pred)
    # --- After session: Save MI CSV and print report ---
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    mi_csv_path = os.path.join(LOG_DIR, f'{user_id}_mi_session_{session_time}.csv')
    pd.DataFrame(mi_records).to_csv(mi_csv_path, index=False)
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
        print("\n[SESSION SUMMARY REPORT]")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print(f"\n[REPORT] Session summary saved to {report_path}")

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

        # --- Show and save final MI plot ---
        visualizer.final_plot()

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
    band = np.asarray(band)
    low, high = band
    if window_sec is not None:
        nperseg = int(window_sec * sf)
    else:
        nperseg = min(256, len(data))
    freqs, psd = scipy.signal.welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.trapz(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapz(psd, dx=freq_res)
    return bp

def resample_eda(eda_buffer, eda_timestamps, target_timestamps):
    """
    Resample/interpolate EDA to match target timestamps (EEG timestamps).
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

if __name__ == "__main__":
    main()
