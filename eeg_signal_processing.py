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
import matplotlib.pyplot as plt
import glob
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut

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
    (20, 50, 'warm'),
    (50, 80, 'cold'),
    (80, 110, 'warm'),
    (110, 140, 'cold'),
    (140, 170, 'warm'),
    (170, 200, 'cold')
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

 mean, variance, standard deviation, kurtosis, skewness.def plot_stimulus_stats_time(preds, stim_labels, epoch_sec=2):
    import seaborn as sns
    import matplotlib.dates as mdates
    from datetime import timedelta
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['window'] = df.index
    df['time_sec'] = df['window'] * epoch_sec
    # Group by stimulus and time, get emotion counts
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    # Also get the dominant emotion per window
    df['dominant_emotion'] = df['emotion']
    # Plot stacked area chart for emotion proportions over time
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(12, 6))
    pivot_prop.plot.area(ax=plt.gca(), cmap='tab10', alpha=0.8)
    # Overlay stimulus periods
    for stim, color in zip(['warm', 'cold', 'neutral'], ['#FFDDC1', '#B5D8FA', '#EEEEEE']):
        for start, end, stim_type in STIM_PERIODS:
            if stim_type == stim:
                plt.axvspan(start, end, color=color, alpha=0.2, label=stim if plt.gca().get_legend() is None or stim not in [t.get_text() for t in plt.gca().get_legend().texts] else None)
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion')
    plt.title('Emotion Proportion Over Time by Stimulus Period')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# --- Main Pipeline ---
if __name__ == "__main__":
    # 1. Build advanced dataset and train SVM (200 features per epoch)
    zX, y, groups, mu, sigma = build_dataset_advanced('data/eeg_by_emotion')
    # Remove 'base' class if present
    mask = y != 'base'
    zX = zX[mask]
    y = y[mask]
    groups = groups[mask]
    print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (after removing 'base' class)")
    X_train, X_test, y_train, y_test = train_test_split(zX, y, test_size=0.2, stratify=y, random_state=42)
    clf, scaler = train_svm(X_train, y_train)
    y_pred = clf.predict(scaler.transform(X_test))
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # 2. Analyze a new EEG file (example: toClasify folder) using advanced features
    test_file = glob.glob('data/toClasify/*.csv')[0]
    eeg = load_eeg_csv(test_file)
    epochs = windowed_epochs(eeg, FS, EPOCH_SEC, EPOCH_OVERLAP)
    feats = [extract_stat_features(ep, FS) for ep in epochs]
    feats = np.array(feats)
    feats = (feats - mu) / (sigma + 1e-8)  # z-score using training stats
    preds = clf.predict(feats)
    n_windows = len(preds)
    stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)
    plot_stimulus_stats_time(preds, stim_labels, epoch_sec=EPOCH_SEC)
    print("Done.")

    # 3. Build advanced dataset and train SVM with PCA and LOSO
    print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (for LOSO)")
    reports, pca = train_svm_pca_loso(zX, y, groups)
    print("LOSO cross-validation reports:")
    for r in reports:
        print(r)
