"""
EEG Signal Processing Pipeline

This script implements advanced signal processing for EEG data analysis, including:
- Data loading and preprocessing
- Signal filtering (bandpass and notch)
- Spectral analysis using Welch's method
- Band power calculations
- SVM classification of emotions (based on "data/eeg_by_emotion" folder) for a 2 second window.
- Use the classification of SVM to create a statistical representation of the stimuli periods
- Advanced visualization

Electrode Key:
    Frontal: Fz (ch1)
    Left Central: C3 (ch2)
    Central Midline: Cz (ch3)
    Right Central: C4 (ch4)
    Parietal Midline: Pz (ch5)
    Left Parietal-Occipital: PO7 (ch6)
    Occipital: Oz (ch7)
    Right Parietal-Occipital: PO8 (ch8)
    Accelerometers (ch9–11): Not relevant for EEG analysis.

Emotion EEG Patterns:
- Excited: ↑ Beta (Fz, C3, Cz, C4), ↓ Alpha (PO7, Oz, PO8)
- Angry: ↑ Beta (C4, Fz)
- Sad: ↑ Alpha (PO8)
- Calm: ↑ Alpha (PO7)

Stimulus Pattern (200s):
- 0-20s: white neutral
- 20-50s: warm color
- 50-80s: cold color
- 80-110s: warm color
- 110-140s: cold color
- 140-170s: warm color
- 170-200s: cold color

"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import glob
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
import sys
from tqdm import tqdm
import joblib
import datetime

def print_progress(message):
    print(f"[PROGRESS] {message}")

# --- Parameters ---
FS = 250  # Sampling frequency (Hz), adjust if needed
BANDPASS = (1, 40)  # Hz
NOTCH_FREQ = 50  # Hz (powerline)
WINDOW_SEC = 2  # Window size for features (seconds)
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CH_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# --- Updated Parameters ---
EPOCH_SEC = 2
EPOCH_OVERLAP = 1  # seconds
N_STATS = 5  # mean, var, std, kurtosis, skewness

# --- Helper Functions ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def notch_filter(data, freq, fs, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return lfilter(b, a, data, axis=0)

def compute_band_power(data, fs, band):
    fmin, fmax = band
    f, Pxx = welch(data, fs=fs, nperseg=fs//2)
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    return np.mean(Pxx[idx_band])

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

def windowed_epochs(data, fs, epoch_sec, overlap_sec):
    step = int(fs * (epoch_sec - overlap_sec))
    win_size = int(fs * epoch_sec)
    epochs = []
    for start in range(0, data.shape[0] - win_size + 1, step):
        epochs.append(data[start:start+win_size, :])
    return np.array(epochs)

def extract_stat_features(epoch, fs):
    feats = []
    # For each channel and each band, compute stats (mean, var, std, kurtosis, skewness)
    for ch in range(epoch.shape[1]):
        for band in BANDS.values():
            psd = compute_band_power(epoch[:, ch], fs, band)
            # For each epoch, for each channel and band, we only have one value (psd),
            # so stats are not meaningful unless we use the whole epoch's bandpower time series.
            # Instead, let's collect all bandpowers for the epoch, then compute stats per band per channel.
            # But with Welch, we get a single value per band per channel per epoch.
            # To get 200 features, we need to use the raw epoch for stats per band per channel.
            # So, let's compute the bandpass-filtered signal for each band, then compute stats on that.
            # For each band, filter the signal and compute stats.
            from scipy.signal import butter, filtfilt
            low, high = band
            nyq = 0.5 * fs
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            band_sig = filtfilt(b, a, epoch[:, ch])
            feats.extend([
                np.mean(band_sig),
                np.var(band_sig),
                np.std(band_sig),
                kurtosis(band_sig),
                skew(band_sig)
            ])
    return feats

def zscore_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    zX = (X - mu) / (sigma + 1e-8)
    return zX, mu, sigma

def label_high_low(zX):
    # Returns a binary mask for high (>1) and low (<-1) features
    high = (zX > 1).astype(int)
    low = (zX < -1).astype(int)
    return high, low

# --- Data Loading ---
def load_eeg_csv(filepath):
    df = pd.read_csv(filepath, dtype=str)  # Read as string to avoid dtype warning
    # Convert all EEG channel columns to float, coerce errors to NaN
    eeg_data = df.iloc[:, 1:9].apply(pd.to_numeric, errors='coerce').values
    # Remove rows with any NaN (from conversion errors)
    eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
    return eeg_data

def extract_features(eeg, fs):
    features = []
    for ch in range(eeg.shape[1]):
        ch_feats = []
        for band in BANDS.values():
            ch_feats.append(compute_band_power(eeg[:, ch], fs, band))
        features.extend(ch_feats)
    return features

def windowed_features(data, fs, window_sec):
    win_size = int(fs * window_sec)
    n_windows = data.shape[0] // win_size
    feats = []
    for w in range(n_windows):
        seg = data[w*win_size:(w+1)*win_size, :]
        feats.append(extract_features(seg, fs))
    return np.array(feats)

# --- Prepare Dataset ---
def build_dataset(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for f in glob.glob(os.path.join(label_dir, '*.csv')):
            eeg = load_eeg_csv(f)
            eeg = bandpass_filter(eeg, *BANDPASS, FS)
            eeg = notch_filter(eeg, NOTCH_FREQ, FS)
            feats = windowed_features(eeg, FS, WINDOW_SEC)
            X.append(feats)
            y.extend([label]*feats.shape[0])
    X = np.vstack(X)
    y = np.array(y)
    return X, y

# --- Updated Dataset Builder ---
def build_dataset_advanced(data_dir):
    X, y, groups = [], [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for f in glob.glob(os.path.join(label_dir, '*.csv')):
            try:
                eeg = load_eeg_csv(f)
                eeg = bandpass_filter(eeg, *BANDPASS, FS)
                eeg = notch_filter(eeg, NOTCH_FREQ, FS)
                epochs = windowed_epochs(eeg, FS, EPOCH_SEC, EPOCH_OVERLAP)
                for ep in epochs:
                    feats = extract_stat_features(ep, FS)
                    X.append(feats)
                    y.append(label)
                    groups.append(f)  # group by file for LOSO
            except Exception as e:
                print(f"Error processing {f}: {e}")
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    zX, mu, sigma = zscore_features(X)
    return zX, y, groups, mu, sigma

# --- SVM Classification ---
def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_scaled, y)
    return clf, scaler

def classify_windows(clf, scaler, eeg):
    eeg = bandpass_filter(eeg, *BANDPASS, FS)
    eeg = notch_filter(eeg, NOTCH_FREQ, FS)
    feats = windowed_features(eeg, FS, WINDOW_SEC)
    X_scaled = scaler.transform(feats)
    return clf.predict(X_scaled)

def classify_windows_progress(clf, scaler, feats):
    preds = []
    for i in tqdm(range(len(feats)), desc='Classifying windows', ncols=80):
        X_scaled = scaler.transform([feats[i]])
        preds.append(clf.predict(X_scaled)[0])
    return np.array(preds)

# --- SVM with PCA and LOSO ---
def train_svm_pca_loso(X, y, groups, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    logo = LeaveOneGroupOut()
    all_reports = []
    for train_idx, test_idx in logo.split(X_pca, y, groups):
        clf = SVC(kernel='rbf', C=1, class_weight='balanced', gamma='scale')
        try:
            clf.fit(X_pca[train_idx], y[train_idx])
            y_pred = clf.predict(X_pca[test_idx])
            report = classification_report(y[test_idx], y_pred, output_dict=True)
            all_reports.append(report)
        except Exception as e:
            print(f"SVM training error: {e}")
    return all_reports, pca

# --- Stimulus Periods ---
STIM_PERIODS = [
    (20, 50, 'cold'),   # 20-50s: cold (30s)
    (50, 80, 'warm'),   # 50-80s: warm (30s)
    (80, 110, 'cold'),  # 80-110s: cold (30s)
    (110, 140, 'warm'), # 110-140s: warm (30s)
    (140, 170, 'cold'), # 140-170s: cold (30s)
    (170, 200, 'warm')  # 170-200s: warm (30s)
]

def get_stimulus_labels(n_windows, fs, window_sec):
    labels = ['neutral'] * n_windows
    for start, end, stim in STIM_PERIODS:
        s_win = int(start // window_sec)
        e_win = int(end // window_sec)
        for i in range(s_win, e_win):
            if i < n_windows:
                labels[i] = stim
    return labels

# --- Visualization ---
def plot_stimulus_stats(preds, stim_labels):
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    stats = pd.crosstab(df['stimulus'], df['emotion'], normalize='index')
    stats.plot(kind='bar', stacked=True)
    plt.title('Emotion Distribution by Stimulus Period')
    plt.ylabel('Proportion')
    plt.xlabel('Stimulus')
    plt.legend(title='Emotion')
    plt.tight_layout()
    plt.show()

def plot_stimulus_stats_time(preds, stim_labels, epoch_sec=2):
    from collections import Counter
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['window'] = df.index
    df['time_sec'] = df['window'] * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    # Fix: ensure normalization is always between 0 and 1
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    pivot_prop = pivot_prop.clip(0, 1)
    plt.figure(figsize=(14, 7))
    pivot_prop.plot.area(ax=plt.gca(), cmap='tab10', alpha=0.85)
    # Overlay stimulus periods
    legend_labels = set()
    for stim, color in zip(['warm', 'cold', 'neutral'], ['#FFDDC1', '#B5D8FA', '#EEEEEE']):
        for start, end, stim_type in STIM_PERIODS:
            if stim_type == stim:
                label = stim if stim not in legend_labels else None
                plt.axvspan(start, end, color=color, alpha=0.18, label=label)
                legend_labels.add(stim)
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion')
    plt.title('Emotion Proportion Over Time by Stimulus Period (Smoothed)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    # Fix: Save and close instead of plt.show() to avoid blocking
    plt.savefig('emotion_proportion_over_time.png')
    plt.close()

def add_side_text_to_plot(ax, text):
    """Add a descriptive text box to the right side of a plot."""
    ax.text(1.02, 0.5, text, va='center', ha='left', fontsize=11, transform=ax.transAxes, wrap=True, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

def plot_stimulus_stats_time_with_text(preds, stim_labels, epoch_sec=2, filename='emotion_proportion_over_time.png', file_label=None):
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['window'] = df.index
    df['time_sec'] = df['window'] * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    pivot_prop = pivot_prop.clip(0, 1)
    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=300)
    pivot_prop.plot.area(ax=ax, cmap='tab10', alpha=0.85)
    legend_labels = set()
    for stim, color in zip(['warm', 'cold', 'neutral'], ['#FFDDC1', '#B5D8FA', '#EEEEEE']):
        for start, end, stim_type in STIM_PERIODS:
            if stim_type == stim:
                label = stim if stim not in legend_labels else None
                ax.axvspan(start, end, color=color, alpha=0.18, label=label)
                legend_labels.add(stim)
    ax.set_xlabel('Time (s)', fontsize=22, labelpad=12)
    ax.set_ylabel('Proportion', fontsize=22, labelpad=12)
    ax.set_title('Emotion Proportion Over Time by Stimulus Period (Smoothed)' + (f"\nFile: {file_label}" if file_label else ''), fontsize=26, pad=18)
    ax.legend(loc='upper right', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    text = (
        "This plot represents how the proportion of each predicted emotion changes over time during the EEG recording.\n\n"
        f"File evaluated: {file_label if file_label else 'N/A'}.\n\n"
        "What it shows:\n"
        "- The x-axis is time (in seconds).\n"
        "- The y-axis is the proportion (from 0 to 1) of each emotion in each time window.\n"
        "- Each colored area corresponds to a different emotion label (e.g., calm, angry, excited, sad).\n"
        "- At each time point, the height of each colored region shows the fraction of windows classified as that emotion at that time.\n"
        "- The plot overlays the stimulus periods (e.g., warm, cold, neutral) as shaded regions, so you can see how emotion proportions change in response to different stimuli.\n\n"
        "Purpose:\n"
        "- To visualize the dynamics of emotional state estimates over the course of the experiment.\n"
        "- To see how the subject’s predicted emotions respond to different color stimuli or time periods."
    )
    add_side_text_to_plot(ax, text)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(filename)
    plt.close(fig)

def plot_band_power_over_time(epochs, fs, filename, file_label):
    """
    Plot band power (delta, theta, alpha, beta, gamma) over time for all channels in a given file.
    """
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    band_powers = np.zeros((n_epochs, len(band_names)))
    for i in range(n_epochs):
        for b, band in enumerate(band_names):
            # Average band power across all channels for this band
            band_powers[i, b] = np.mean([
                compute_band_power(epochs[i, :, ch], fs, BANDS[band])
                for ch in range(n_channels)
            ])
    # Plotting
    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=300)
    for b, band in enumerate(band_names):
        ax.plot(np.arange(n_epochs), band_powers[:, b], label=band.capitalize(), linewidth=2)
    ax.set_xlabel('Epoch (Time)', fontsize=22, labelpad=12)
    ax.set_ylabel('Average Band Power', fontsize=22, labelpad=12)
    ax.set_title(f'EEG Band Power Over Time\nFile: {file_label}', fontsize=26, pad=18)
    ax.legend(fontsize=18, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Explanatory text under plot
    text = (
        "This plot shows the average power of each EEG band (delta, theta, alpha, beta, gamma) "
        "across all electrodes for each epoch.\n\n"
        "Each colored line represents a different brainwave band.\n"
        "Comparing these lines allows you to see how the relative strength of each band changes over time, "
        "and to directly compare the dynamics of different brainwave frequencies.\n\n"
        "Higher values indicate greater power in that frequency band, averaged across all channels."
    )
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.05, 0.04, text, ha='left', va='bottom', fontsize=17, wrap=True, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.4'))
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return band_powers

def plot_alpha_beta_ratio_over_time(epochs, fs, filename, file_label):
    n_epochs, _, n_channels = epochs.shape
    ratios = np.zeros((n_epochs, n_channels))
    for i in range(n_epochs):
        for ch in range(n_channels):
            alpha = compute_band_power(epochs[i, :, ch], fs, BANDS['alpha'])
            beta = compute_band_power(epochs[i, :, ch], fs, BANDS['beta'])
            ratios[i, ch] = alpha / (beta + 1e-8)
    plt.figure(figsize=(14, 2*n_channels))
    for ch in range(n_channels):
        plt.plot(ratios[:, ch], label=CHANNELS[ch])
    plt.xlabel('Epoch (Time)')
    plt.ylabel('Alpha/Beta Ratio')
    plt.title(f'Alpha/Beta Power Ratio Over Time\nFile: {file_label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return ratios

# --- Save alpha/beta ratio to CSV ---
def save_alpha_beta_ratio_csv(ratios, filename, file_label):
    df = pd.DataFrame(ratios, columns=CHANNELS)
    df['file'] = file_label
    df['epoch'] = np.arange(len(df))
    df.to_csv(filename, index=False)
    return df

def save_bandpower_csv(epochs, fs, filename, file_label):
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    data = []
    for i in range(n_epochs):
        row = {'file': file_label, 'epoch': i}
        for ch in range(n_channels):
            for b, band in enumerate(band_names):
                val = compute_band_power(epochs[i, :, ch], fs, BANDS[band])
                row[f'{CHANNELS[ch]}_{band}'] = val
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

# --- Utility: Timestamped filename for outputs ---
def timestamped_filename(base, file_label, ext):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{file_label}_{ts}.{ext}"

# --- Comparative Band Power Plot ---
def plot_comparative_band_power(bandpower_dfs, out_filename, file_labels=None):
    """
    bandpower_dfs: list of (file_label, DataFrame) where each DataFrame has columns like 'Fz_alpha', ...
    Plots average band power per channel/band across all files for comparison.
    file_labels: list of file names included in the comparison (for explicit labeling)
    """
    import matplotlib.pyplot as plt
    band_names = list(BANDS.keys())
    n_files = len(bandpower_dfs)
    n_channels = len(CHANNELS)
    fig, axes = plt.subplots(len(band_names), 1, figsize=(16, 2.5*len(band_names)), sharex=True)
    for b, band in enumerate(band_names):
        for file_label, df in bandpower_dfs:
            means = [df[f"{ch}_{band}"].mean() for ch in CHANNELS]
            axes[b].plot(CHANNELS, means, marker='o', label=file_label)
        axes[b].set_ylabel(f'{band} Power', fontsize=14)
        axes[b].set_title(f'Average {band} Band Power Across Files', fontsize=16)
        axes[b].legend(fontsize=12)
    axes[-1].set_xlabel('Channel', fontsize=14)
    # Add explicit file list and context-rich description
    if file_labels is None:
        file_labels = [file_label for file_label, _ in bandpower_dfs]
    file_list_str = ', '.join(file_labels)
    fig.suptitle('Comparative Band Power Across Files', fontsize=18)
    # Explanatory/context text under plot
    text = (
        f"This plot compares the average band power for each EEG band (delta, theta, alpha, beta, gamma) "
        f"across all channels for each file.\n\n"
        f"Files included in this analysis: {file_list_str}.\n\n"
        "Each line represents a different file. Use this plot to compare the spectral characteristics of EEG signals across subjects or sessions."
    )
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.05, 0.04, text, ha='left', va='bottom', fontsize=13, wrap=True, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.4'))
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(out_filename, dpi=300)
    plt.close()

# --- Main Pipeline Update: Per-file analysis and comparative report ---
if __name__ == "__main__":
    # --- Model/Scaler/PCA file names ---
    SVM_MODEL_FILE = 'svm_model.joblib'
    SCALER_FILE = 'scaler.joblib'
    PCA_FILE = 'pca.joblib'

    # --- Check for existing models ---
    model_exists = os.path.exists(SVM_MODEL_FILE) and os.path.exists(SCALER_FILE)
    pca_exists = os.path.exists(PCA_FILE)

    # --- Model Verification and Loading ---
    import os
    MODEL_PATH = 'svm_model.joblib'
    SCALER_PATH = 'scaler.joblib'

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print_progress("Loading existing SVM model and scaler from disk...")
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Load training stats for z-scoring
        # If you saved mu/sigma, load them here; else, recompute as needed
        # For now, we need to build the dataset to get mu/sigma for feature normalization
        zX, y, groups, mu, sigma = build_dataset_advanced('data/eeg_by_emotion')
        mask = y != 'base'
        zX = zX[mask]
        y = y[mask]
        groups = groups[mask]
        print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (after removing 'base' class)")
    else:
        print_progress("Building advanced dataset and training SVM (200 features per epoch)...")
        zX, y, groups, mu, sigma = build_dataset_advanced('data/eeg_by_emotion')
        mask = y != 'base'
        zX = zX[mask]
        y = y[mask]
        groups = groups[mask]
        print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (after removing 'base' class)")
        print_progress("Splitting data and training SVM classifier...")
        X_train, X_test, y_train, y_test = train_test_split(zX, y, test_size=0.2, stratify=y, random_state=42)
        clf, scaler = train_svm(X_train, y_train)
        print_progress("Evaluating SVM classifier...")
        y_pred = clf.predict(scaler.transform(X_test))
        print(classification_report(y_test, y_pred, zero_division=0))
        print(confusion_matrix(y_test, y_pred))
        # Save the trained model and scaler
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        # If using PCA:
        # joblib.dump(pca, 'pca.joblib')

    print_progress("Analyzing new EEG file (toClasify) using advanced features...")
    test_file = glob.glob('data/toClasify/*.csv')[0]
    eeg_raw = load_eeg_csv(test_file)
    eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, FS)
    eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, FS)
    epochs = windowed_epochs(eeg_filt, FS, EPOCH_SEC, EPOCH_OVERLAP)
    feats = [extract_stat_features(ep, FS) for ep in epochs]
    feats = np.array(feats)
    feats = (feats - mu) / (sigma + 1e-8)  # z-score using training stats
    print_progress("Visualizing band power over time for all channels...")
    plot_band_power_over_time(epochs, FS, 'band_power_over_time.png', 'toClasify')
    print_progress("Classifying windows for new EEG file (progress bar)...")
    preds = classify_windows_progress(clf, scaler, feats)
    n_windows = len(preds)
    stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)
    print_progress("Plotting emotion proportions over time (improved)...")
    plot_stimulus_stats_time_with_text(preds, stim_labels, epoch_sec=EPOCH_SEC)
    print("Done.")

    print_progress("Saving statistics and plots to PDF and CSV...")
    save_filtering_stats(eeg_raw, eeg_filt, 'filtering_stats.csv')
    save_feature_stats(feats, 'feature_stats.csv')
    print_progress("Running LOSO cross-validation...")
    reports, pca = train_svm_pca_loso(zX, y, groups)
    save_loso_reports(reports, 'loso_reports.csv')
    plot_and_save_all(eeg_raw, eeg_filt, feats, preds, stim_labels, reports, out_prefix='EEG_Analysis')
    print("PDF and CSV reports saved.")

    print_progress("Generating all emotion visualizations...")
    plot_emotion_labels(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig1_emotion_labels.png')
    plot_emotion_avg_per_stimulus(preds, stim_labels, filename='fig2_emotion_avg_per_stimulus.png')
    plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig2_emotion_avg_per_stimulus_over_time.png', file_label='test_file')
    plot_stacked_area(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig3_stacked_area.png')
    plot_line_chart(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig4_line_chart.png')
    plot_stacked_bar(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig5_stacked_bar.png')
    plot_heatmap(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig6_heatmap.png')
    plot_small_multiples(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig7_small_multiples.png')
    plot_streamgraph(preds, stim_labels, epoch_sec=EPOCH_SEC, filename='fig8_streamgraph.png')
    print_progress("All emotion visualizations saved as PNG files.")

    # Save all figures to a single PDF
    fig_files = [
        'fig1_emotion_labels.png',
        'fig2_emotion_avg_per_stimulus.png',
        'fig2_emotion_avg_per_stimulus_over_time.png',
        'fig3_stacked_area.png',
        'fig4_line_chart.png',
        'fig5_stacked_bar.png',
        'fig6_heatmap.png',
        'fig7_small_multiples.png',
        'fig8_streamgraph.png',
    ]
    save_all_figures_to_pdf(fig_files, 'EEG_Emotion_Visualizations.pdf')
    print_progress("All emotion visualizations saved to EEG_Emotion_Visualizations.pdf.")

    print_progress("LOSO cross-validation reports:")
    print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (for LOSO)")
    for r in reports:
        print(r)

    # Save PCA if available
    if pca is not None:
        joblib.dump(pca, PCA_FILE)
        print_progress("PCA model saved.")

    print_progress("Generating pattern-based emotion labels visualization...")
    plot_eeg_pattern_based_labels(epochs, FS, filename='pattern_based_emotion_labels.png', file_label='test_file')
    print_progress("Pattern-based emotion labels visualization saved.")

    # --- Analyze all files in data/toClasify/ ---
    test_files = sorted(glob.glob('data/toClasify/*.csv'))
    print_progress(f"Files to analyze: {test_files}")
    bandpower_csvs = []
    bandpower_dfs = []
    bandpower_heatmap_pngs = []
    alpha_beta_ratio_csvs = []
    alpha_beta_ratio_pngs = []
    emotion_prop_csvs = []
    emotion_bar_pngs = []
    pattern_label_pngs = []
    for idx, test_file in enumerate(test_files):
        file_label = os.path.splitext(os.path.basename(test_file))[0]
        print_progress(f"Processing file {idx+1}/{len(test_files)}: {file_label}")
        eeg_raw = load_eeg_csv(test_file)
        eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, FS)
        eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, FS)
        epochs = windowed_epochs(eeg_filt, FS, EPOCH_SEC, EPOCH_OVERLAP)
        feats = [extract_stat_features(ep, FS) for ep in epochs]
        feats = np.array(feats)
        feats = (feats - mu) / (sigma + 1e-8)
        preds = classify_windows_progress(clf, scaler, feats)
        n_windows = len(preds)
        stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)
        # --- Per-file band power heatmap (all bands, all channels together) ---
        bandpower_heatmap_png = timestamped_filename('band_power_bands_all_channels', file_label, 'png')
        plot_band_power_bands_all_channels_together(epochs, FS, bandpower_heatmap_png, file_label)
        bandpower_heatmap_pngs.append(bandpower_heatmap_png)
        # --- Save bandpower CSV ---
        bandpower_csv = timestamped_filename('bandpower', file_label, 'csv')
        bandpower_df = save_bandpower_csv(epochs, FS, bandpower_csv, file_label)
        bandpower_csvs.append(bandpower_csv)
        bandpower_dfs.append((file_label, bandpower_df))
        # --- Alpha/Beta ratio visualization and CSV ---
        alpha_beta_ratio_png = timestamped_filename('alpha_beta_ratio', file_label, 'png')
        ratios = plot_alpha_beta_ratio_over_time(epochs, FS, alpha_beta_ratio_png, file_label)
        alpha_beta_ratio_pngs.append(alpha_beta_ratio_png)
        alpha_beta_ratio_csv = timestamped_filename('alpha_beta_ratio', file_label, 'csv')
        save_alpha_beta_ratio_csv(ratios, alpha_beta_ratio_csv, file_label)
        alpha_beta_ratio_csvs.append(alpha_beta_ratio_csv)
        # --- Emotion proportions CSV ---
        emotion_prop_csv = timestamped_filename('emotion_proportions', file_label, 'csv')
        save_emotion_proportions_csv(preds, stim_labels, emotion_prop_csv, file_label)
        emotion_prop_csvs.append(emotion_prop_csv)
        # --- Per-stimulus window emotion bar plot ---
        emotion_bar_png = timestamped_filename('emotion_bar_per_stimulus', file_label, 'png')
        plot_emotion_avg_per_stimulus_window(preds, stim_labels, EPOCH_SEC, emotion_bar_png, file_label)
        emotion_bar_pngs.append(emotion_bar_png)
        # --- Pattern-based emotion label plot ---
        pattern_label_png = timestamped_filename('pattern_based_emotion_labels', file_label, 'png')
        plot_eeg_pattern_based_labels(epochs, FS, filename=pattern_label_png, file_label=file_label)
        pattern_label_pngs.append(pattern_label_png)
        # --- Bandpower by stimulus period (all channels) ---
        bandpower_stimulus_png = timestamped_filename('bandpower_by_stimulus_period', file_label, 'png')
        plot_bandpower_by_stimulus_period_all_channels(epochs, stim_labels, FS, bandpower_stimulus_png, file_label)
        bandpower_heatmap_pngs.append(bandpower_stimulus_png)
    # --- Comparative band power plot ---
    comparative_bandpower_png = timestamped_filename('comparative_band_power', 'ALL', 'png')
    file_labels = [file_label for file_label, _ in bandpower_dfs]
    plot_comparative_band_power(bandpower_dfs, comparative_bandpower_png, file_labels=file_labels)
    # --- Comparative alpha/beta ratio plot ---
    comparative_alpha_beta_png = timestamped_filename('comparative_alpha_beta_ratio', 'ALL', 'png')
    alpha_beta_file_labels = []
    for f in alpha_beta_ratio_csvs:
        try:
            import pandas as pd
            df = pd.read_csv(f)
            alpha_beta_file_labels.append(df['file'][0] if 'file' in df.columns else 'unknown')
        except Exception:
            alpha_beta_file_labels.append('unknown')
    plot_comparative_alpha_beta_ratio(alpha_beta_ratio_csvs, comparative_alpha_beta_png, file_labels=alpha_beta_file_labels)
    # --- Comparative emotion bar plot ---
    comparative_emotion_bar_png = timestamped_filename('comparative_emotion_bar', 'ALL', 'png')
    emotion_bar_file_labels = []
    for f in emotion_prop_csvs:
        try:
            import pandas as pd
            df = pd.read_csv(f)
            emotion_bar_file_labels.append(df['file'][0] if 'file' in df.columns else 'unknown')
        except Exception:
            emotion_bar_file_labels.append('unknown')
    plot_comparative_emotion_bar(emotion_prop_csvs, comparative_emotion_bar_png, file_labels=emotion_bar_file_labels)
    # --- Save all per-file and comparative plots to a PDF report ---
    all_report_pngs = bandpower_heatmap_pngs + alpha_beta_ratio_pngs + emotion_bar_pngs + pattern_label_pngs + [comparative_bandpower_png, comparative_alpha_beta_png, comparative_emotion_bar_png]
    pdf_report_name = timestamped_report_filename('EEG_Analysis_report')
    # Add a cover page with explanation and classification statistics
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(pdf_report_name) as pdf:
        # Cover page: Explanation and statistics
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        text = (
            "EEG Signal Processing Pipeline\n\n"
            "This report summarizes the EEG signal processing and emotion classification pipeline.\n\n"
            "- Data from all files in 'data/toClasify/' were analyzed.\n"
            "- For each file, the pipeline computed band power (delta, theta, alpha, beta, gamma) for all channels, alpha/beta ratio, and emotion proportions per stimulus window.\n"
            "- An SVM classifier (RBF kernel, C=1, gamma='scale', class_weight='balanced') was trained on labeled data and used to classify emotional state per window.\n"
            "- Pattern-based emotion labels were also computed using EEG band power rules and z-score thresholding.\n\n"
            "Classification statistics (LOSO cross-validation):\n"
        )
        plt.text(0.01, 0.98, text, fontsize=12, va='top', ha='left', wrap=True)
        # Insert LOSO stats table (from loso_reports.csv)
        try:
            loso_df = pd.read_csv('loso_reports.csv')
            table_text = loso_df[['Fold','Label','precision','recall','f1-score','support']].to_string(index=False)
            plt.text(0.01, 0.5, table_text, fontsize=10, va='top', ha='left', family='monospace')
        except Exception as e:
            plt.text(0.01, 0.5, f"Could not load LOSO stats: {e}", fontsize=10, va='top', ha='left')
        # Add a note
        note = ("\n\nAll subsequent pages show per-file and comparative visualizations. "
                "See the project README or markdown report for further details.")
        plt.text(0.01, 0.1, note, fontsize=11, va='top', ha='left')
        pdf.savefig(fig)
        plt.close(fig)
        # Add all PNGs as pages
        for fname in all_report_pngs:
            fig = plt.figure()
            img = plt.imread(fname)
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
    print_progress(f"All per-file and comparative visualizations saved to {pdf_report_name}")

# --- Utility: Timestamped PDF report name ---
def timestamped_report_filename(base):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{ts}.pdf"

# --- Pattern-based emotion labels visualization (ensure defined before main) ---
def plot_eeg_pattern_based_labels(epochs, fs, filename, file_label):
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    band_powers = np.zeros((n_epochs, len(band_names)))
    for i in range(n_epochs):
        for ch in range(n_channels):
            for b, band in enumerate(BANDS.values()):
                band_powers[i, ch, b] = compute_band_power(epochs[i, :, ch], fs, band)
    mu = band_powers.mean(axis=0)
    sigma = band_powers.std(axis=0) + 1e-8
    z_band_powers = (band_powers - mu) / sigma
    labels = []
    for i in range(n_epochs):
        excited = (
            (z_band_powers[i, CH_IDX['Fz'], band_names.index('beta')] > 1) and
            (z_band_powers[i, CH_IDX['C3'], band_names.index('beta')] > 1) and
            (z_band_powers[i, CH_IDX['Cz'], band_names.index('beta')] > 1) and
            (z_band_powers[i, CH_IDX['C4'], band_names.index('beta')] > 1) and
            (z_band_powers[i, CH_IDX['PO7'], band_names.index('alpha')] < -1) and
            (z_band_powers[i, CH_IDX['Oz'], band_names.index('alpha')] < -1) and
            (z_band_powers[i, CH_IDX['PO8'], band_names.index('alpha')] < -1)
        )
        angry = (
            (z_band_powers[i, CH_IDX['C4'], band_names.index('beta')] > 1) and
            (z_band_powers[i, CH_IDX['Fz'], band_names.index('beta')] > 1)
        )
        sad = (z_band_powers[i, CH_IDX['PO8'], band_names.index('alpha')] > 1)
        calm = (z_band_powers[i, CH_IDX['PO7'], band_names.index('alpha')] > 1)
        if excited:
            labels.append('excited')
        elif angry:
            labels.append('angry')
        elif sad:
            labels.append('sad')
        elif calm:
            labels.append('calm')
        else:
            labels.append('neutral')
    fig, ax = plt.subplots(figsize=(16, 3))
    color_map = {'excited': '#FFB347', 'angry': '#FF6961', 'sad': '#779ECB', 'calm': '#77DD77', 'neutral': '#CCCCCC'}
    x = np.arange(n_epochs)
    y = np.zeros(n_epochs)
    for emo in ['excited', 'angry', 'sad', 'calm', 'neutral']:
        idx = [i for i, l in enumerate(labels) if l == emo]
        ax.scatter(x[idx], y[idx], color=color_map[emo], label=emo, s=60)
    ax.set_yticks([])
    ax.set_xlabel('Epoch (Time)')
    ax.set_title(f'Pattern-based Emotion Labels per Epoch\nFile: {file_label}')
    ax.legend(loc='upper right')
    text = (
        "Pattern-based emotion labels are assigned using EEG band power rules and z-score thresholding.\n"
        "Rules:\n"
        "- Excited: High beta (Fz, C3, Cz, C4), low alpha (PO7, Oz, PO8)\n"
        "- Angry: High beta (C4, Fz)\n"
        "- Sad: High alpha (PO8)\n"
        "- Calm: High alpha (PO7)\n"
        "- Neutral: None of the above\n\n"
        "Z-score threshold: >1 (high), <-1 (low) relative to mean band power per channel.\n"
        "This plot shows the detected emotion label for each epoch based on these rules."
    )
    add_side_text_to_plot(ax, text)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(filename)
    plt.close(fig)
    return labels

def plot_bandpower_by_stimulus_period_all_channels(epochs, stim_labels, fs, filename, file_label):
    """
    Plot average bandpower for each EEG band (delta, theta, alpha, beta, gamma) per stimulus period,
    averaged across all electrodes and all epochs in that period.
    Now explicitly states the evaluated file in the title and in the explanatory text.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    band_names = list(BANDS.keys())
    stim_types = ['neutral', 'warm', 'cold']
    # Map each epoch to its stimulus period
    df = pd.DataFrame({'stimulus': stim_labels})
    n_epochs, _, n_channels = epochs.shape
    bandpower_by_period = {stim: [] for stim in stim_types}
    for stim in stim_types:
        idx = df.index[df['stimulus'] == stim].to_numpy()
        if len(idx) == 0:
            continue
        # For all epochs in this period, compute mean bandpower across all channels
        bandpower = np.zeros((len(idx), len(band_names)))
        for i, ep_idx in enumerate(idx):
            for b, band in enumerate(band_names):
                bandpower[i, b] = np.mean([
                    compute_band_power(epochs[ep_idx, :, ch], fs, BANDS[band])
                    for ch in range(n_channels)
                ])
        bandpower_by_period[stim] = bandpower
    # Compute mean and std for each band in each period
    means = np.zeros((len(stim_types), len(band_names)))
    stds = np.zeros((len(stim_types), len(band_names)))
    for s, stim in enumerate(stim_types):
        if len(bandpower_by_period[stim]) > 0:
            means[s, :] = np.mean(bandpower_by_period[stim], axis=0)
            stds[s, :] = np.std(bandpower_by_period[stim], axis=0)
        else:
            means[s, :] = np.nan
            stds[s, :] = np.nan
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.15
    x = np.arange(len(stim_types))
    for b, band in enumerate(band_names):
        ax.bar(x + b*width - (len(band_names)/2-0.5)*width, means[:, b], width, yerr=stds[:, b], label=band.capitalize(), capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in stim_types], fontsize=14)
    ax.set_ylabel('Average Band Power (across all channels)', fontsize=14)
    ax.set_xlabel('Stimulus Period', fontsize=14)
    ax.set_title(f'EEG Band Power by Stimulus Period (All Channels)\nFile: {file_label}', fontsize=16)
    ax.legend(fontsize=12)
    text = (
        f"This plot shows the average power of each EEG band (delta, theta, alpha, beta, gamma) "
        f"across all electrodes, grouped by stimulus period (neutral, warm, cold).\n\n"
        f"File evaluated: {file_label}.\n\n"
        "Each group of bars represents a stimulus period. Each colored bar is a different brainwave band.\n"
        "Error bars show the standard deviation across epochs in each period.\n\n"
        "This allows you to compare how brainwave activity changes in response to different color stimuli."
    )
    add_side_text_to_plot(ax, text)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(filename)
    plt.close(fig)
    return means, stds

def analyze_and_export_emotion_results(eeg_file, clf, scaler, mu, sigma, out_prefix, fs=FS, epoch_sec=EPOCH_SEC, epoch_overlap=EPOCH_OVERLAP):
    """
    1. Evaluate emotion classification per epoch
    2. Evaluate emotion per stimulus period (proportion per period)
    3. Create band power and emotion plots
    4. Export CSV with emotion proportions per period
    5. Export TXT with explanation and summary
    """
    import os
    # 1. Load and preprocess EEG
    eeg_raw = load_eeg_csv(eeg_file)
    eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, fs)
    eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, fs)
    epochs = windowed_epochs(eeg_filt, fs, epoch_sec, epoch_overlap)
    feats = [extract_stat_features(ep, fs) for ep in epochs]
    feats = np.array(feats)
    feats = (feats - mu) / (sigma + 1e-8)
    # 2. Classify per epoch
    preds = classify_windows_progress(clf, scaler, feats)
    n_epochs = len(preds)
    stim_labels = get_stimulus_labels(n_epochs, fs, epoch_sec)
    # 3. Bandpower plot
    bandpower_png = f"{out_prefix}_bandpower.png"
    plot_band_power_bands_all_channels_together(epochs, fs, bandpower_png, out_prefix)
    # 4. Line plot of emotion proportions over time
    lineplot_png = f"{out_prefix}_emotion_lineplot.png"
    plot_line_chart(preds, stim_labels, epoch_sec=epoch_sec, filename=lineplot_png)
    # 5. Bar plot of emotion proportions per stimulus period
    barplot_png = f"{out_prefix}_emotion_barplot.png"
    plot_stacked_bar(preds, stim_labels, epoch_sec=epoch_sec, filename=barplot_png)
    # 6. Export CSV: emotion proportions per period
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    periods = df['stimulus'].unique()
    emotions = df['emotion'].unique()
    prop_table = []
    for period in periods:
        sub = df[df['stimulus'] == period]
        total = len(sub)
        row = {'stimulus': period}
        for emo in emotions:
            if total > 0:
                pct = 100 * (sub['emotion'] == emo).sum() / total
            else:
                pct = 0
            row[emo] = pct
        prop_table.append(row)
    prop_df = pd.DataFrame(prop_table)
    prop_df.to_csv(f"{out_prefix}_emotion_proportions.csv", index=False)
    # 7. Export TXT: summary of results
    with open(f"{out_prefix}_summary.txt", "w") as txt_file:
        txt_file.write(f"EEG Emotion Analysis Report\nFile: {eeg_file}\n\n")
        txt_file.write("Emotion Proportions by Stimulus Period:\n")
        for period in periods:
            txt_file.write(f"{period}:\n")
            for emo in emotions:
                pct = prop_df.loc[prop_df['stimulus'] == period, emo].values
                if len(pct) > 0:
                    txt_file.write(f"  {emo}: {pct[0]:.2f}%\n")
                else:
                    txt_file.write(f"  {emo}: 0.00%\n")
            txt_file.write("\n")
        txt_file.write("Note: Proportions are based on the classified emotions for each epoch.\n")
    print(f"Results exported: {out_prefix}_emotion_proportions.csv, {out_prefix}_summary.txt")

def plot_and_save_all(eeg_raw, eeg_filt, feats, preds, stim_labels, reports, out_prefix='EEG_Analysis'):
    """
    Create a summary figure and combine key visualizations and reports.
    Save all to disk with a timestamp, return list of generated files.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    import datetime
    # Generate timestamped base name
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = f"{out_prefix}_{ts}"
    files = []
    # 1. Save filtering stats
    filtering_stats_csv = f"{base}_filtering_stats.csv"
    save_filtering_stats(eeg_raw, eeg_filt, filtering_stats_csv)
    files.append(filtering_stats_csv)
    # 2. Save feature stats
    feature_stats_csv = f"{base}_feature_stats.csv"
    save_feature_stats(feats, feature_stats_csv)
    files.append(feature_stats_csv)
    # 3. Save LOSO reports
    loso_reports_csv = f"{base}_loso_reports.csv"
    save_loso_reports(reports, loso_reports_csv)
    files.append(loso_reports_csv)
    # 4. Generate and save all key plots (with timestamped names)
    fig_files = []
    fig1 = f"{base}_fig1_emotion_labels.png"
    plot_emotion_labels(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig1)
    fig_files.append(fig1)
    fig2 = f"{base}_fig2_emotion_avg_per_stimulus.png"
    plot_emotion_avg_per_stimulus(preds, stim_labels, filename=fig2)
    fig_files.append(fig2)
    fig3 = f"{base}_fig2_emotion_avg_per_stimulus_over_time.png"
    plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig3, file_label=out_prefix)
    fig_files.append(fig3)
    fig4 = f"{base}_fig3_stacked_area.png"
    plot_stacked_area(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig4)
    fig_files.append(fig4)
    fig5 = f"{base}_fig4_line_chart.png"
    plot_line_chart(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig5)
    fig_files.append(fig5)
    fig6 = f"{base}_fig5_stacked_bar.png"
    plot_stacked_bar(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig6)
    fig_files.append(fig6)
    fig7 = f"{base}_fig6_heatmap.png"
    plot_heatmap(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig7)
    fig_files.append(fig7)
    fig8 = f"{base}_fig7_small_multiples.png"
    plot_small_multiples(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig8)
    fig_files.append(fig8)
    fig9 = f"{base}_fig8_streamgraph.png"
    plot_streamgraph(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=fig9)
    fig_files.append(fig9)
    files.extend(fig_files)
    # 5. Save all figures to a single PDF
    pdf_file = f"{base}_visualizations.pdf"
    save_all_figures_to_pdf(fig_files, pdf_file)
    files.append(pdf_file)
    # 6. Create a summary figure (grid of key plots)
    summary_fig = f"{base}_summary.png"
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, fname in enumerate(fig_files[:9]):
        img = plt.imread(fname)
        ax = axes.flat[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(fname), fontsize=8)
    plt.tight_layout()
    plt.savefig(summary_fig)
    plt.close(fig)
    files.append(summary_fig)
    # 7. Return list of all generated files
    return files

def save_filtering_stats(eeg_raw, eeg_filt, filename, label='EEG File'):
    """
    Save filtering statistics (mean, std, min, max) for each channel before and after filtering,
    along with filter parameters and frequency response, and a label about the data source.
    """
    import pandas as pd
    import numpy as np
    from scipy.signal import freqz
    # Channel names
    channels = CHANNELS
    stats = []
    for i, ch in enumerate(channels):
        raw = eeg_raw[:, i]
        filt = eeg_filt[:, i]
        stats.append({
            'label': label,
            'channel': ch,
            'raw_mean': np.mean(raw),
            'raw_std': np.std(raw),
            'raw_min': np.min(raw),
            'raw_max': np.max(raw),
            'filt_mean': np.mean(filt),
            'filt_std': np.std(filt),
            'filt_min': np.min(filt),
            'filt_max': np.max(filt),
        })
    df = pd.DataFrame(stats)
    # Add filter parameters
    filter_params = {
        'bandpass_low': BANDPASS[0],
        'bandpass_high': BANDPASS[1],
        'notch_freq': NOTCH_FREQ,
        'fs': FS,
        'order': 4,
        'notch_Q': 30
    }
    # Compute frequency response for bandpass
    from scipy.signal import butter, iirnotch
    b_bp, a_bp = butter(4, [BANDPASS[0]/(0.5*FS), BANDPASS[1]/(0.5*FS)], btype='band')
    w_bp, h_bp = freqz(b_bp, a_bp, worN=512, fs=FS)
    b_notch, a_notch = iirnotch(NOTCH_FREQ, 30, FS)
    w_notch, h_notch = freqz(b_notch, a_notch, worN=512, fs=FS)
    # Save stats
    with pd.ExcelWriter(filename.replace('.csv', '.xlsx'), engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Stats', index=False)
        # Filter parameters
        pd.DataFrame([filter_params]).to_excel(writer, sheet_name='FilterParams', index=False)
        # Frequency response
        pd.DataFrame({'freq': w_bp, 'bandpass_mag': np.abs(h_bp)}).to_excel(writer, sheet_name='BandpassResponse', index=False)
        pd.DataFrame({'freq': w_notch, 'notch_mag': np.abs(h_notch)}).to_excel(writer, sheet_name='NotchResponse', index=False)
    # Also save as CSV for quick view
    df.to_csv(filename, index=False)
