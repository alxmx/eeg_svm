"""
Real-time Mindfulness Index (MI) LSL Pipeline

- Loads existing SVM/SVR model and scaler if available, else trains from EEG/EDA data.
- Sets up LSL streams for features (input), Unity labels (input), and MI output (output).
- Uses SGDRegressor for online MI adaptation.

Usage:
    python realtime_mi_lsl.py

Dependencies:
    pip install numpy pandas scikit-learn pylsl joblib
"""
import os
import glob
import numpy as np
import pandas as pd
from joblib import load, dump
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

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

def select_lsl_stream(stream_type):
    print(f"Searching for available LSL streams of type '{stream_type}'...")
    streams = resolve_byprop('type', stream_type)
    if not streams:
        raise RuntimeError(f"No LSL streams of type '{stream_type}' found.")
    print("Available streams:")
    for idx, s in enumerate(streams):
        print(f"[{idx}] Name: {s.name()}, Type: {s.type()}, Channel count: {s.channel_count()}, Source ID: {s.source_id()}")
    sel = input(f"Select stream index (0-{len(streams)-1}): ")
    try:
        sel = int(sel)
        assert 0 <= sel < len(streams)
    except Exception:
        print("Invalid selection, defaulting to 0.")
        sel = 0
    return streams[sel]

def calibrate_user(user_id, n_samples=100):
    """
    Calibration step: Collect baseline (ground truth) data for a new user.
    Prompts the user to relax (e.g., eyes closed), collects N samples, and saves as ground truth.
    Creates a config file for the user with calibration info.
    Now includes an interactive countdown, exit command, user info display, model update summary, and cumulative calibration report.
    """
    import glob
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
    feature_stream = select_lsl_stream('Features')
    feature_inlet = StreamInlet(feature_stream)
    baseline_samples = []
    N = n_samples
    i = 0
    while i < N:
        user_input = input(f"Press Enter to collect sample {i+1}/{N} or type 'exit' to abort: ").strip().lower()
        if user_input == 'exit':
            print("Calibration aborted by user.")
            return None, None
        sample, _ = feature_inlet.pull_sample()
        baseline_samples.append(sample)
        print(f"Collected baseline sample {i+1}/{N}")
        i += 1
    baseline_arr = np.array(baseline_samples)
    # Save baseline as CSV for this user
    baseline_csv = os.path.join(USER_CONFIG_DIR, f'{user_id}_baseline.csv')
    pd.DataFrame(baseline_arr, columns=FEATURE_ORDER).to_csv(baseline_csv, index=False)
    print(f"Baseline calibration data saved to {baseline_csv}")
    print(f"[CONFIRM] Calibration file created: {baseline_csv}")
    # Save user config JSON
    user_config = {
        'user_id': user_id,
        'baseline_csv': baseline_csv,
        'calibration_time': str(datetime.now()),
        'n_samples': N
    }
    with open(config_path, 'w') as f:
        json.dump(user_config, f, indent=2)
    print(f"User config saved to {config_path}")
    print(f"[CONFIRM] Calibration config created: {config_path}")
    # Update cumulative calibration report
    report_path = os.path.join(LOG_DIR, 'calibration_report.csv')
    import pandas as pd
    new_row = {
        'user_id': user_id,
        'calibration_time': user_config['calibration_time'],
        'n_samples': N,
        'baseline_csv': baseline_csv
    }
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        # Remove previous entry for this user if exists
        df = df[df['user_id'] != user_id]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(report_path, index=False)
    print(f"Cumulative calibration report updated: {report_path}")
    # Show model update summary (from logs)
    metrics_log = os.path.join(LOG_DIR, 'metrics_log.txt')
    if os.path.exists(metrics_log):
        with open(metrics_log, 'r') as f:
            lines = f.readlines()
        user_updates = [line for line in lines if user_id in line or 'Precision' in line]
        print(f"[MODEL UPDATE SUMMARY]")
        print(f"  Total updates: {len(user_updates)}")
        if user_updates:
            print(f"  Last update: {user_updates[-1].strip()}")
        else:
            print("  No updates found for this user.")
    else:
        print("[MODEL UPDATE SUMMARY] No model update log found.")
    return baseline_csv, config_path

# --- Visualization ---
class OnlineVisualizer:
    def __init__(self):
        self.mi_history = []
        self.label_history = []
        self.timestamps = []
        self.last_metrics = {'precision': 1.0, 'recall': 1.0}
        self.fig, self.ax = plt.subplots()
        self.fig.show()

    def update(self, mi_pred, label=None):
        self.mi_history.append(mi_pred)
        self.timestamps.append(datetime.now())
        if label is not None:
            self.label_history.append(label)
        else:
            self.label_history.append(np.nan)
        self.plot()

    def plot(self):
        self.ax.clear()
        self.ax.plot(self.mi_history, label='MI Prediction')
        if any(~np.isnan(self.label_history)):
            self.ax.plot(self.label_history, label='Labels', linestyle='dashed')
        self.ax.set_title('Online MI Prediction')
        self.ax.set_xlabel('Sample')
        self.ax.set_ylabel('MI')
        self.ax.legend()
        plt.pause(0.01)
        # Save plot
        fname = os.path.join(VIS_DIR, f'online_mi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        self.fig.savefig(fname)

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
    label_stream = select_lsl_stream('UnityMarkers')
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
        mi_pred = online_model.predict(x_scaled)[0]
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
    user_id = input("Enter user ID for this session: ")
    # Ask for calibration duration
    try:
        duration_input = input("Enter calibration duration in seconds (default 60): ").strip()
        if duration_input == '':
            calibration_duration = 60
        else:
            calibration_duration = int(duration_input)
    except Exception:
        calibration_duration = 60
    print(f"Calibration will last {calibration_duration} seconds.")
    calibrate = input("Run calibration step for this user? (y/n): ").strip().lower() == 'y'
    if calibrate:
        calibrate_user(user_id, n_samples=calibration_duration)
    print(f"Checking for existing model: {MODEL_PATH} and scaler: {SCALER_PATH}")
    svr, scaler = load_or_train_models()
    print("Model and scaler ready.")

    # Set up online learner
    print("Initializing online adaptive model (SGDRegressor)...")
    online_model = SGDRegressor(max_iter=1000, learning_rate='optimal', eta0=0.01)
    print("Loading training data for online model warm start...")
    X, y = load_training_data()
    X_scaled = scaler.transform(X)
    print("Fitting online model with SVR predictions...")
    online_model.partial_fit(X_scaled, svr.predict(X_scaled))
    print("Online model initialized.")

    # Try to load per-user online model if exists
    user_online_model_path = ONLINE_MODEL_PATH_TEMPLATE.format(user_id)
    if os.path.exists(user_online_model_path):
        print(f"Loading previous online model for user {user_id}...")
        online_model = load(user_online_model_path)
        print("User-specific online model loaded.")

    # LSL streams
    print("Select the EEG LSL feature stream to use:")
    eeg_stream = select_lsl_stream('EEG')
    eeg_inlet = StreamInlet(eeg_stream)
    print("EEG feature stream connected.")
    print("Select the EDA LSL feature stream to use:")
    eda_stream = select_lsl_stream('EDA')
    eda_inlet = StreamInlet(eda_stream)
    print("EDA feature stream connected.")
    print("Resolving Unity label stream (type='UnityMarkers')...")
    label_stream = select_lsl_stream('UnityMarkers')
    label_inlet = StreamInlet(label_stream)
    print("Unity label stream connected.")
    print("Creating MI output LSL stream (type='processed_MI')...")
    mi_info = StreamInfo('processed_MI', 'MI', 1, 10, 'float32', 'mi_stream')
    mi_outlet = StreamOutlet(mi_info)
    print("MI output stream created as 'processed_MI'.\n")

    visualizer = OnlineVisualizer()
    print("Entering real-time MI prediction loop. Press Ctrl+C to stop.")
    mi_window = []  # Moving window for MI predictions
    while True:
        print("\nWaiting for new EEG and EDA feature samples from LSL...")
        eeg_sample, _ = eeg_inlet.pull_sample()
        eda_sample, _ = eda_inlet.pull_sample()
        print(f"Received EEG sample: {eeg_sample}")
        print(f"Received EDA sample: {eda_sample}")
        # Combine EEG and EDA features (assuming both are lists/arrays)
        sample = np.array(list(eeg_sample) + list(eda_sample)).reshape(1, -1)
        x_scaled = scaler.transform(sample)
        mi_pred = online_model.predict(x_scaled)[0]
        print(f"Predicted MI: {mi_pred:.3f} (pushed to processed_MI stream)")
        mi_outlet.push_sample([mi_pred])
        # Add to moving window
        mi_window.append(mi_pred)
        if len(mi_window) > ONLINE_UPDATE_WINDOW:
            mi_window.pop(0)
        # Optional: update with Unity label if threshold trigger is met
        label, _ = label_inlet.pull_sample(timeout=0.01)
        update_triggered = False
        if len(mi_window) == ONLINE_UPDATE_WINDOW and label:
            # Check if MI is consistently above or below thresholds
            above = sum([v >= MI_THRESHOLDS['focused'] for v in mi_window])
            below = sum([v < MI_THRESHOLDS['neutral'] for v in mi_window])
            if above / ONLINE_UPDATE_WINDOW >= ONLINE_UPDATE_RATIO:
                print(f"[AUTO-UPDATE] MI above focused threshold for {ONLINE_UPDATE_RATIO*100:.0f}% of window. Updating model with label {label[0]}.")
                online_model.partial_fit(x_scaled, [float(label[0])])
                update_triggered = True
            elif below / ONLINE_UPDATE_WINDOW >= ONLINE_UPDATE_RATIO:
                print(f"[AUTO-UPDATE] MI below neutral threshold for {ONLINE_UPDATE_RATIO*100:.0f}% of window. Updating model with label {label[0]}.")
                online_model.partial_fit(x_scaled, [float(label[0])])
                update_triggered = True
        if update_triggered:
            dump(online_model, user_online_model_path)
            print(f"User online model updated and saved to {user_online_model_path}")
            visualizer.update(mi_pred, float(label[0]))
            visualizer.log_metrics(np.array(visualizer.label_history), np.array(visualizer.mi_history))
        else:
            if label:
                print("Label received from Unity, but threshold not met. No update.")
            visualizer.update(mi_pred)

if __name__ == '__main__':
    main()
