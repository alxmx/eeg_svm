"""
SVM Regression using Mindfulness Index (MI) as a continuous target

This script:
- Loads EEG (and EDA if available) data from a folder
- Extracts features and computes MI for each window
- Uses MI (0-1) as the regression target
- Builds a dataset and trains/evaluates an SVM regressor
- Saves results and metrics for reporting
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from eeg_mindfulness_index import load_eeg_csv, bandpass_filter, compute_band_power, WINDOW_SEC, FS, MI_WEIGHTS, calculate_mi, THRESHOLDS, load_eda_csv, normalize_eda

# Path to the single EDA file to use for all EEG files
EDA_FILE = 'data\eda_data\opensignals_0007808C0700_2025-04-30_16-32-15_V.txt'  # <-- Update this to your actual EDA file path

def extract_mi_features_with_eda_regression(eeg, fs, eda_data, eda_fs):
    win_size = int(fs * WINDOW_SEC)
    n_windows = eeg.shape[0] // win_size
    mi_features = []
    mi_targets = []
    eda_warned = False
    for w in range(n_windows):
        seg = eeg[w*win_size:(w+1)*win_size, :]
        features = {}
        features['theta_fz'] = compute_band_power(seg[:, 0], fs, (4, 7.99))
        features['alpha_po'] = np.mean([
            compute_band_power(seg[:, 5], fs, (8, 12.99)),
            compute_band_power(seg[:, 7], fs, (8, 12.99))
        ])
        alpha_c3 = compute_band_power(seg[:, 1], fs, (8, 12.99))
        alpha_c4 = compute_band_power(seg[:, 3], fs, (8, 12.99))
        features['faa'] = np.log(alpha_c4 + 1e-8) - np.log(alpha_c3 + 1e-8)
        features['beta_frontal'] = np.mean([
            compute_band_power(seg[:, 0], fs, (13, 30)),
            compute_band_power(seg[:, 1], fs, (13, 30)),
            compute_band_power(seg[:, 3], fs, (13, 30))
        ])
        # EDA alignment
        if eda_data is not None and len(eda_data) > 0:
            eda_idx = int((w * win_size) * eda_fs / fs)
            if eda_idx < len(eda_data):
                eda_value = eda_data[eda_idx]
            else:
                eda_value = eda_data[-1]
        else:
            eda_value = None
            if not eda_warned:
                print("No EDA data available for this file, using 0 for all windows")
                eda_warned = True
        mi = calculate_mi(features, eda_value)
        mi_features.append(list(features.values()) + [eda_value if eda_value is not None else 0])
        mi_targets.append(mi)
    return np.array(mi_features), np.array(mi_targets)

def build_mi_regression_dataset_with_eda(data_dir, eda_data, eda_fs):
    X, y = [], []
    for f in os.listdir(data_dir):
        if not f.endswith('.csv'):
            continue
        eeg = load_eeg_csv(os.path.join(data_dir, f))
        eeg = bandpass_filter(eeg, 4, 30, FS)
        feats, targets = extract_mi_features_with_eda_regression(eeg, FS, eda_data, eda_fs)
        X.append(feats)
        y.append(targets)
    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

def main():
    DATA_DIR = 'data/toClasify'
    print(f"Loading EDA file: {EDA_FILE}")
    eda_raw = load_eda_csv(EDA_FILE)
    if eda_raw is None or len(eda_raw) == 0:
        print("WARNING: EDA file missing or empty. Proceeding without EDA.")
        eda_data = None
        eda_fs = FS
    else:
        eda_data = normalize_eda(eda_raw)
        eda_fs = FS
    print(f"Building MI regression dataset from {DATA_DIR} using EDA from {EDA_FILE}...")
    X, y = build_mi_regression_dataset_with_eda(DATA_DIR, eda_data, eda_fs)
    print(f"Dataset: {X.shape[0]} windows, {X.shape[1]} features (including EDA)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    reg = SVR(kernel='rbf', C=1, gamma='scale')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regression Metrics:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}")
    # Save results
    df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(X_test.shape[1])])
    df['true_mi'] = y_test
    df['pred_mi'] = y_pred
    df.to_csv('svm_mi_regression_results.csv', index=False)
    # Save metrics
    with open('svm_mi_regression_metrics.txt', 'w') as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}\n")
    print("Results and metrics saved for regression SVM.")

if __name__ == "__main__":
    main()
