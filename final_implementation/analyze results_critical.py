import os
import pandas as pd
from datetime import datetime

def find_latest_session_file(user_id, log_dir, file_prefix):
    """Find the latest session file for the given user."""
    session_files = [
        f for f in os.listdir(log_dir)
        if f.startswith(f"{user_id}_{file_prefix}_") and f.endswith(".csv")
    ]
    if not session_files:
        print(f"[ERROR] No session files found for user {user_id} with prefix {file_prefix} in {log_dir}.")
        return None

    # Sort files by timestamp in their filenames
    session_files.sort(key=lambda x: datetime.strptime(x.split('_')[-1].split('.')[0], '%H%M%S'), reverse=True)
    latest_file = session_files[0]
    print(f"[INFO] Latest session file for user {user_id} with prefix {file_prefix}: {latest_file}")
    return os.path.join(log_dir, latest_file)

def analyze_calibration(file_path):
    """Analyze calibration results."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"[INFO] Loading calibration data from {file_path}...")
    calibration_df = pd.read_csv(file_path)

    avg_mae = calibration_df['new_mae'].mean()
    avg_r2 = calibration_df['new_r2'].mean()
    total_samples = calibration_df['n_samples'].sum()

    print(f"Average MAE: {avg_mae:.3f}")
    print(f"Average R2: {avg_r2:.3f}")
    print(f"Total Samples: {total_samples}")

def analyze_features(file_path):
    """Analyze feature correlations and statistics."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"[INFO] Loading feature data from {file_path}...")
    feature_df = pd.read_csv(file_path)

    print("Feature Correlations:")
    print(feature_df[['feature', 'spearman_corr', 'p_value']])

def analyze_session(file_path):
    """Analyze the session file and generate a report."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    print(f"[INFO] Loading session data from {file_path}...")
    session_df = pd.read_csv(file_path)

    avg_mi = session_df['mi'].mean()
    avg_raw_mi = session_df['raw_mi'].mean()
    avg_emi = session_df['emi'].mean()

    print(f"Average MI: {avg_mi:.3f}")
    print(f"Average Raw MI: {avg_raw_mi:.3f}")
    print(f"Average EMI: {avg_emi:.3f}")

if __name__ == "__main__":
    USER_ID = "009_alex_test"  # Change this to the desired user ID
    LOG_DIR = "c:\\Users\\lenin\\Documents\\GitHub\\eeg_svm\\final_implementation\\logs"  # Update if log directory changes

    calibration_file = find_latest_session_file(USER_ID, LOG_DIR, "calibration_comparative")
    if calibration_file:
        analyze_calibration(calibration_file)

    feature_corr_file = find_latest_session_file(USER_ID, LOG_DIR, "mi_feature_corr")
    if feature_corr_file:
        analyze_features(feature_corr_file)

    session_file = find_latest_session_file(USER_ID, LOG_DIR, "mi_session")
    if session_file:
        analyze_session(session_file)