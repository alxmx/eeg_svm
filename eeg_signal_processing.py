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
def timestamped_filename(prefix, file_label, ext):
    """
    Generates a timestamped filename for plot outputs.
    Example: bandpower_UnicornRecorder_12_05_2025_12_37_430_20250521_153021.png
    """
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{file_label}_{ts}.{ext}"

def relative_timestamped_filename(prefix, file_label, ext):
    """
    Generates a timestamped filename for plot outputs without timestamp.
    This is useful when we're already saving to a dated folder.
    Example: bandpower_UnicornRecorder_12_05_2025_12_37_430.png
    """
    return f"{prefix}_{file_label}.{ext}"

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
    # joblib.dump(pca, 'pca.joblib')    # Create a dated output folder for all results
test_file = glob.glob('data/toClasify/*.csv')[0]
file_basename = os.path.splitext(os.path.basename(test_file))[0]
output_folder = create_output_folder(file_basename)
    
print_progress(f"Analyzing EEG file: {test_file}, results will be saved to: {output_folder}")
eeg_raw = load_eeg_csv(test_file)
eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, FS)
eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, FS)
epochs = windowed_epochs(eeg_filt, FS, EPOCH_SEC, EPOCH_OVERLAP)
feats = [extract_stat_features(ep, FS) for ep in epochs]
feats = np.array(feats)
feats = (feats - mu) / (sigma + 1e-8)  # z-score using training stats
    
print_progress("Visualizing band power over time for all channels...")
bp_filename = os.path.join(output_folder, 'band_power_over_time.png')
plot_band_power_over_time(epochs, FS, bp_filename, file_basename)
    
print_progress("Classifying windows for new EEG file (progress bar)...")
preds = classify_windows_progress(clf, scaler, feats)
n_windows = len(preds)
stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)    print_progress("Plotting emotion proportions over time (improved)...")
    emotion_plot_filename = os.path.join(output_folder, 'emotion_proportion_over_time.png')
    plot_stimulus_stats_time_with_text(preds, stim_labels, epoch_sec=EPOCH_SEC, filename=emotion_plot_filename, file_label=file_basename)
    print("Done.")
    
    print_progress("Saving statistics and plots to PDF and CSV...")
filtering_stats_csv = os.path.join(output_folder, 'filtering_stats.csv')
save_filtering_stats(eeg_raw, eeg_filt, filtering_stats_csv)
    
feature_stats_csv = os.path.join(output_folder, 'feature_stats.csv')
save_feature_stats(feats, feature_stats_csv)
    
print_progress("Running LOSO cross-validation...")
reports, pca = train_svm_pca_loso(zX, y, groups)
    
loso_reports_csv = os.path.join(output_folder, 'loso_reports.csv')
save_loso_reports(reports, loso_reports_csv)
    
report_prefix = os.path.join(output_folder, 'EEG_Analysis')    plot_and_save_all(eeg_raw, eeg_filt, feats, preds, stim_labels, reports, out_prefix=report_prefix)
    print("PDF and CSV reports saved.")
    
    print_progress("Generating all emotion visualizations...")
emotion_visuals_folder = os.path.join(output_folder, 'emotion_visualizations')
if not os.path.exists(emotion_visuals_folder):
    os.makedirs(emotion_visuals_folder)
    
# Generate all visualizations in the output folder
plot_emotion_labels(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                    filename=os.path.join(emotion_visuals_folder, 'fig1_emotion_labels.png'))
    
plot_emotion_avg_per_stimulus(preds, stim_labels, 
                             filename=os.path.join(emotion_visuals_folder, 'fig2_emotion_avg_per_stimulus.png'))
    
plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                                       filename=os.path.join(emotion_visuals_folder, 'fig2_emotion_avg_per_stimulus_over_time.png'),
                                       file_label=file_basename)
    
plot_stacked_area(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                 filename=os.path.join(emotion_visuals_folder, 'fig3_stacked_area.png'))
    
plot_line_chart(preds, stim_labels, epoch_sec=EPOCH_SEC, 
               filename=os.path.join(emotion_visuals_folder, 'fig4_line_chart.png'))
    
plot_stacked_bar(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                filename=os.path.join(emotion_visuals_folder, 'fig5_stacked_bar.png'))
    
plot_heatmap(preds, stim_labels, epoch_sec=EPOCH_SEC, 
            filename=os.path.join(emotion_visuals_folder, 'fig6_heatmap.png'))
    
plot_small_multiples(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                    filename=os.path.join(emotion_visuals_folder, 'fig7_small_multiples.png'))
    
plot_streamgraph(preds, stim_labels, epoch_sec=EPOCH_SEC, 
                filename=os.path.join(emotion_visuals_folder, 'fig8_streamgraph.png'))
    
print_progress("All emotion visualizations saved as PNG files.")    # Save all figures to a single PDF
fig_files = [
    os.path.join(emotion_visuals_folder, 'fig1_emotion_labels.png'),
    os.path.join(emotion_visuals_folder, 'fig2_emotion_avg_per_stimulus.png'),
    os.path.join(emotion_visuals_folder, 'fig2_emotion_avg_per_stimulus_over_time.png'),
    os.path.join(emotion_visuals_folder, 'fig3_stacked_area.png'),
    os.path.join(emotion_visuals_folder, 'fig4_line_chart.png'),
    os.path.join(emotion_visuals_folder, 'fig5_stacked_bar.png'),
    os.path.join(emotion_visuals_folder, 'fig6_heatmap.png'),
    os.path.join(emotion_visuals_folder, 'fig7_small_multiples.png'),
    os.path.join(emotion_visuals_folder, 'fig8_streamgraph.png'),
]
pdf_filename = os.path.join(output_folder, 'EEG_Emotion_Visualizations.pdf')
save_all_figures_to_pdf(fig_files, pdf_filename)
print_progress(f"All emotion visualizations saved to {pdf_filename}")

print_progress("LOSO cross-validation reports:")
print(f"Dataset: {zX.shape[0]} epochs, {zX.shape[1]} features (for LOSO)")
for r in reports:
    print(r)    # Save PCA if available
    if pca is not None:
        joblib.dump(pca, PCA_FILE)
        print_progress("PCA model saved.")
        
    print_progress("Generating pattern-based emotion labels visualization...")
pattern_labels_filename = os.path.join(output_folder, 'pattern_based_emotion_labels.png')
plot_eeg_pattern_based_labels(epochs, FS, filename=pattern_labels_filename, file_label=file_basename)
print_progress("Pattern-based emotion labels visualization saved.")    # --- Analyze all files in data/toClasify/ ---
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
    
    # Create a main output folder for comparative analysis
    comparative_folder = create_output_folder("EEG_Comparative_Analysis")
    
    for idx, test_file in enumerate(test_files):
        file_label = os.path.splitext(os.path.basename(test_file))[0]
        print_progress(f"Processing file {idx+1}/{len(test_files)}: {file_label}")
        
        # Create a dedicated output folder for this file
        output_folder = create_output_folder(file_label)
        print_progress(f"Results will be saved to: {output_folder}")
        
        # Create a subfolder for emotion visualizations
        emotion_visuals_folder = os.path.join(output_folder, 'emotion_visualizations')
        if not os.path.exists(emotion_visuals_folder):
            os.makedirs(emotion_visuals_folder)
        
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
        bandpower_heatmap_png = os.path.join(output_folder, f'band_power_bands_all_channels_{file_label}.png')
        plot_band_power_bands_all_channels_together(epochs, FS, bandpower_heatmap_png, file_label)
        bandpower_heatmap_pngs.append(bandpower_heatmap_png)
        
        # --- Save bandpower CSV ---
        bandpower_csv = os.path.join(output_folder, f'bandpower_{file_label}.csv')
        bandpower_df = save_bandpower_csv(epochs, FS, bandpower_csv, file_label)
        bandpower_csvs.append(bandpower_csv)
        bandpower_dfs.append((file_label, bandpower_df))
        
        # --- Alpha/Beta ratio visualization and CSV ---
        alpha_beta_ratio_png = os.path.join(output_folder, f'alpha_beta_ratio_{file_label}.png')
        ratios = plot_alpha_beta_ratio_over_time(epochs, FS, alpha_beta_ratio_png, file_label)
        alpha_beta_ratio_pngs.append(alpha_beta_ratio_png)
        
        alpha_beta_ratio_csv = os.path.join(output_folder, f'alpha_beta_ratio_{file_label}.csv')
        save_alpha_beta_ratio_csv(ratios, alpha_beta_ratio_csv, file_label)
        alpha_beta_ratio_csvs.append(alpha_beta_ratio_csv)
        
        # --- Emotion proportions CSV ---
        emotion_prop_csv = os.path.join(output_folder, f'emotion_proportions_{file_label}.csv')
        save_emotion_proportions_csv(preds, stim_labels, emotion_prop_csv, file_label)
        emotion_prop_csvs.append(emotion_prop_csv)
        
        # --- Per-stimulus window emotion bar plot ---
        emotion_bar_png = os.path.join(emotion_visuals_folder, f'emotion_bar_per_stimulus_{file_label}.png')
        plot_emotion_avg_per_stimulus_window(preds, stim_labels, EPOCH_SEC, emotion_bar_png, file_label)
        emotion_bar_pngs.append(emotion_bar_png)
        
        # --- Pattern-based emotion label plot ---
        pattern_label_png = os.path.join(emotion_visuals_folder, f'pattern_based_emotion_labels_{file_label}.png')
        plot_eeg_pattern_based_labels(epochs, FS, filename=pattern_label_png, file_label=file_label)
        pattern_label_pngs.append(pattern_label_png)
        
        # --- Bandpower by stimulus period (all channels) ---
        bandpower_stimulus_png = os.path.join(output_folder, f'bandpower_by_stimulus_period_{file_label}.png')
        plot_bandpower_by_stimulus_period_all_channels(epochs, stim_labels, FS, bandpower_stimulus_png, file_label)
        bandpower_heatmap_pngs.append(bandpower_stimulus_png)
        
        # Save a report PDF for this specific file
        file_report_pdf = os.path.join(output_folder, f'{file_label}_Analysis_Report.pdf')
        file_report_pngs = [
            bandpower_heatmap_png,
            alpha_beta_ratio_png,
            emotion_bar_png,
            pattern_label_png,
            bandpower_stimulus_png
        ]
        save_all_figures_to_pdf(file_report_pngs, file_report_pdf)
        print_progress(f"Analysis report for {file_label} saved to {file_report_pdf}")    # --- Comparative band power plot ---
    comparative_bandpower_png = os.path.join(comparative_folder, 'comparative_band_power_ALL.png')
    file_labels = [file_label for file_label, _ in bandpower_dfs]
    plot_comparative_band_power(bandpower_dfs, comparative_bandpower_png, file_labels=file_labels)
    
    # --- Comparative alpha/beta ratio plot ---
    comparative_alpha_beta_png = os.path.join(comparative_folder, 'comparative_alpha_beta_ratio_ALL.png')
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
    comparative_emotion_bar_png = os.path.join(comparative_folder, 'comparative_emotion_bar_ALL.png')
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
    pdf_report_name = os.path.join(comparative_folder, 'EEG_Analysis_report.pdf')
    
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
            row[f'{emo}_count'] = (sub['emotion'] == emo).sum()
            row[f'{emo}_pct'] = pct
            
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

def save_feature_stats(feats, filename):
    """
    Save statistics about features (mean, std, min, max, quartiles)
    for each feature dimension across all samples.
    """
    import pandas as pd
    import numpy as np
    stats = []
    for i in range(feats.shape[1]):
        feat = feats[:, i]
        stats.append({
            'feature_idx': i,
            'mean': np.mean(feat),
            'std': np.std(feat),
            'min': np.min(feat),
            'max': np.max(feat),
            '25%': np.percentile(feat, 25),
            'median': np.median(feat),
            '75%': np.percentile(feat, 75)
        })
    df = pd.DataFrame(stats)
    df.to_csv(filename, index=False)
    return df

def save_loso_reports(reports, filename):
    """
    Save Leave-One-Subject-Out cross-validation reports to CSV.
    """
    import pandas as pd
    rows = []
    for i, report in enumerate(reports):
        for label in report.keys():
            if isinstance(report[label], dict):
                metrics = report[label]
                rows.append({
                    'Fold': i+1,
                    'Label': label,
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1-score': metrics.get('f1-score', 0),
                    'support': metrics.get('support', 0)
                })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df

def save_all_figures_to_pdf(fig_files, pdf_filename):
    """
    Combine multiple PNG figures into a single PDF file.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(pdf_filename) as pdf:
        for fig_file in fig_files:
            try:
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(fig_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(fig_file), fontsize=10)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Error adding {fig_file} to PDF: {e}")
    return pdf_filename

def save_emotion_proportions_csv(preds, stim_labels, filename, file_label):
    """
    Save the proportions of each emotion in each stimulus period to a CSV file.
    """
    import pandas as pd
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['file'] = file_label
    
    # Per stimulus proportions
    periods = df['stimulus'].unique()
    emotions = df['emotion'].unique()
    rows = []
    
    for period in periods:
        sub = df[df['stimulus'] == period]
        total = len(sub)
        row = {'file': file_label, 'stimulus': period, 'total_epochs': total}
        
        for emotion in emotions:
            count = (sub['emotion'] == emotion).sum()
            pct = (count / total * 100) if total > 0 else 0
            row[f'{emotion}_count'] = count
            row[f'{emotion}_pct'] = pct
            
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(filename, index=False)
    return result_df

def plot_emotion_labels(preds, stim_labels, epoch_sec=2, filename='emotion_labels.png'):
    """
    Plot emotion labels for each epoch with stimulus period background.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Define colors for emotions and stimulus periods
    emotion_colors = {
        'angry': '#FF6961',   # Red
        'calm': '#77DD77',    # Green
        'excited': '#FFB347', # Orange
        'sad': '#779ECB',     # Blue
        'neutral': '#CCCCCC', # Gray
        'base': '#EEEEEE'     # Light Gray
    }
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(16, 3))
    
    # Plot each emotion label as a colored dot
    n_epochs = len(preds)
    x = np.arange(n_epochs)
    y = np.zeros(n_epochs)
    
    unique_emotions = sorted(set(preds))
    for emotion in unique_emotions:
        idx = [i for i, p in enumerate(preds) if p == emotion]
        if emotion in emotion_colors:
            color = emotion_colors[emotion]
        else:
            color = '#999999'
        ax.scatter(x[idx], y[idx], color=color, label=emotion, s=50)
    
    # Overlay stimulus periods
    stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
    legend_labels = set()
    for stim, color in zip(['warm', 'cold', 'neutral'], ['#FFDDC1', '#B5D8FA', '#EEEEEE']):
        for i, label in enumerate(stim_labels):
            if label == stim:
                ax.axvspan(i-0.5, i+0.5, color=color, alpha=0.3, linewidth=0)
                
    # Customize the plot
    ax.set_yticks([])
    ax.set_xlabel('Epoch')
    ax.set_title('Emotion Labels per Epoch with Stimulus Periods')
    
    # Create stimulus period legend separately
    from matplotlib.patches import Patch
    stim_legend_elements = [
        Patch(facecolor=stim_colors[s], alpha=0.3, label=s) 
        for s in ['warm', 'cold', 'neutral']
    ]
    ax.legend(loc='upper right', handles=[
        *[Patch(facecolor=emotion_colors[e], label=e) for e in unique_emotions],
        *stim_legend_elements
    ])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_emotion_avg_per_stimulus(preds, stim_labels, filename='emotion_avg_per_stimulus.png'):
    """
    Plot the average proportion of each emotion per stimulus period as a stacked bar chart.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    stats = pd.crosstab(df['stimulus'], df['emotion'], normalize='index')
    
    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
    
    ax.set_title('Average Emotion Proportion per Stimulus Period', fontsize=14)
    ax.set_xlabel('Stimulus Period', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.legend(title='Emotion', title_fontsize=12, fontsize=10)
    
    # Add percentage labels on bars
    for i, stimulus in enumerate(stats.index):
        total = 0
        for emotion, prop in stats.loc[stimulus].items():
            if prop > 0.05:  # Only show label if proportion is significant
                ax.text(i, total + prop/2, f"{prop:.0%}", 
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            total += prop
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, epoch_sec=2, 
                                          filename='emotion_avg_per_stimulus_over_time.png',
                                          file_label=None):
    """
    Plot the average proportion of each emotion per stimulus period over time.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Unique stimulus periods and emotions
    stim_periods = sorted(df['stimulus'].unique())
    emotions = sorted(df['emotion'].unique())
    
    # Create figure with subplots for each stimulus period
    fig, axes = plt.subplots(len(stim_periods), 1, figsize=(14, 3*len(stim_periods)),
                             sharex=True, constrained_layout=True)
    
    if len(stim_periods) == 1:
        axes = [axes]  # Make it iterable when only one stimulus period
    
    for i, stim in enumerate(stim_periods):
        stim_data = df[df['stimulus'] == stim]
        window_size = max(1, len(stim_data) // 10)  # Adjust window size based on data
        
        # Compute rolling average of emotion proportions
        emotion_props = {}
        for emotion in emotions:
            stim_data[f'{emotion}_flag'] = (stim_data['emotion'] == emotion).astype(int)
            if len(stim_data) > window_size:
                emotion_props[emotion] = stim_data[f'{emotion}_flag'].rolling(window=window_size).mean()
            else:
                emotion_props[emotion] = stim_data[f'{emotion}_flag']
        
        # Plot for this stimulus
        for emotion in emotions:
            axes[i].plot(stim_data['time_sec'], emotion_props[emotion], label=emotion, linewidth=2)
        
        axes[i].set_title(f'Stimulus: {stim.capitalize()}', fontsize=12)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Proportion', fontsize=10)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    # Set x-axis label on the last subplot
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    # Main title for the plot
    if file_label:
        fig.suptitle(f'Emotion Proportions Over Time by Stimulus Period\nFile: {file_label}', 
                     fontsize=16, y=1.02)
    else:
        fig.suptitle('Emotion Proportions Over Time by Stimulus Period', fontsize=16, y=1.02)
    
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_stacked_area(preds, stim_labels, epoch_sec=2, filename='stacked_area.png'):
    """
    Plot emotion proportions over time as a stacked area chart.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Pivot to get emotion proportions
    pivot = pd.crosstab(index=df['time_sec'], columns=df['emotion'], normalize='index')
    pivot = pivot.fillna(0)
    
    # Plot as stacked area
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot.area(ax=ax, stacked=True, alpha=0.7, colormap='tab10')
    
    # Overlay stimulus periods
    legend_labels = set()
    stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
    for stim in ['warm', 'cold', 'neutral']:
        color = stim_colors[stim]
        for start, end, stim_type in STIM_PERIODS:
            if stim_type == stim:
                ax.axvspan(start, end, color=color, alpha=0.2, label=stim if stim not in legend_labels else None)
                legend_labels.add(stim)
    
    # Add stimulus legend
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    emotion_handles = handles[:len(pivot.columns)]
    stim_handles = [Patch(facecolor=stim_colors[s], alpha=0.2, label=s) for s in stim_colors]
    ax.legend(handles=emotion_handles + stim_handles, 
             labels=list(pivot.columns) + list(stim_colors.keys()),
             loc='upper right')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Emotion Proportions Over Time (Stacked Area)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_line_chart(preds, stim_labels, epoch_sec=2, filename='line_chart.png'):
    """
    Plot emotion proportions over time as a line charts.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Compute smoothed emotion proportions over time
    emotions = sorted(df['emotion'].unique())
    time_points = sorted(df['time_sec'].unique())
    emotion_data = {emotion: [] for emotion in emotions}
    
    window_size = max(1, len(time_points) // 20)  # Adjustable window size
    
    for t in time_points:
        window = df[(df['time_sec'] >= t - epoch_sec/2) & 
                    (df['time_sec'] < t + epoch_sec/2)]
        total = len(window)
        
        for emotion in emotions:
            if total > 0:
                prop = (window['emotion'] == emotion).sum() / total
            else:
                prop = 0
            emotion_data[emotion].append(prop)
    
    # Apply Gaussian smoothing
    for emotion in emotions:
        if len(emotion_data[emotion]) > 5:  # Only smooth if enough data points
            emotion_data[emotion] = gaussian_filter1d(emotion_data[emotion], sigma=1.5)
    
    # Plot as line chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for emotion in emotions:
        ax.plot(time_points, emotion_data[emotion], label=emotion, linewidth=2)
    
    # Overlay stimulus periods
    stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
    legend_labels = set()
    for stim in ['warm', 'cold', 'neutral']:
        color = stim_colors[stim]
        for start, end, stim_type in STIM_PERIODS:
            if stim_type == stim:
                ax.axvspan(start, end, color=color, alpha=0.2, label=stim if stim not in legend_labels else None)
                legend_labels.add(stim)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Emotion Proportions Over Time (Smoothed Line Chart)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_stacked_bar(preds, stim_labels, epoch_sec=2, filename='stacked_bar.png'):
    """
    Plot emotion proportions per time window as a stacked bar chart.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Group by time windows (e.g., 10-second windows)
    window_size = 10  # seconds
    df['time_window'] = (df['time_sec'] // window_size) * window_size
    df['time_label'] = df['time_window'].apply(lambda x: f"{x}-{x+window_size}")
    
    # Compute emotion proportions per window
    window_stats = pd.crosstab(df['time_label'], df['emotion'], normalize='index')
    
    # Get stimulus for each time window (most frequent)
    window_stim = df.groupby('time_label')['stimulus'].agg(lambda x: x.value_counts().index[0])
    
    # Plot as stacked bar
    fig, ax = plt.subplots(figsize=(14, 6))
    window_stats.plot.bar(stacked=True, ax=ax, width=0.8, colormap='tab10')
    
    # Color bars by stimulus period
    bar_positions = np.arange(len(window_stats.index))
    stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
    
    for i, (window, stim) in enumerate(window_stim.items()):
        if stim in stim_colors:
            ax.axvspan(i-0.4, i+0.4, color=stim_colors[stim], alpha=0.2)
    
    # Add stimulus legend
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    stim_handles = [Patch(facecolor=stim_colors[s], alpha=0.2, label=s) for s in stim_colors]
    ax.legend(handles=list(handles) + stim_handles, 
             labels=list(labels) + list(stim_colors.keys()),
             loc='upper right')
    
    ax.set_xlabel('Time Window (seconds)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Emotion Proportions by Time Window', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_heatmap(preds, stim_labels, epoch_sec=2, filename='heatmap.png'):
    """
    Plot emotion proportions over time as a heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Group by time windows
    window_size = 10  # seconds
    df['time_window'] = (df['time_sec'] // window_size) * window_size
    df['time_label'] = df['time_window'].apply(lambda x: f"{x}")
    
    # Compute emotion proportions per window
    window_stats = pd.crosstab(df['time_label'], df['emotion'], normalize='index').fillna(0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(window_stats, cmap='viridis', annot=False, linewidths=0.5, ax=ax)
    
    # Add stimulus period labels
    stim_windows = df.groupby('time_label')['stimulus'].agg(lambda x: x.value_counts().index[0])
    
    # Create colored tick labels based on stimulus
    tick_colors = {'warm': 'red', 'cold': 'blue', 'neutral': 'gray'}
    
    time_ticks = [int(t) for t in window_stats.index]
    plt.yticks(np.arange(len(time_ticks))+0.5, time_ticks, fontsize=10)
    
    # Color tick labels by stimulus
    for i, (time, stim) in enumerate(stim_windows.items()):
        if stim in tick_colors:
            plt.gca().get_yticklabels()[i].set_color(tick_colors[stim])
    
    ax.set_title('Emotion Proportions Heatmap by Time Window', fontsize=14)
    ax.set_ylabel('Time Window (seconds)', fontsize=12)
    ax.set_xlabel('Emotion', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_small_multiples(preds, stim_labels, epoch_sec=2, filename='small_multiples.png'):
    """
    Plot small multiples of emotion proportions over time, one plot per emotion.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Create a DataFrame with time index
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Get unique emotions
    emotions = sorted(df['emotion'].unique())
    
    # Create small multiples plot
    n_emotions = len(emotions)
    fig, axes = plt.subplots(n_emotions, 1, figsize=(14, 3*n_emotions), sharex=True)
    
    if n_emotions == 1:
        axes = [axes]  # Make it iterable if only one emotion
    
    # Compute smoothed emotion proportions over time
    time_points = sorted(df['time_sec'].unique())
    window_size = max(1, len(time_points) // 20)  # Adjustable window size
    
    for i, emotion in enumerate(emotions):
        emotion_data = []
        for t in time_points:
            window = df[(df['time_sec'] >= t - epoch_sec/2) & 
                        (df['time_sec'] < t + epoch_sec/2)]
            total = len(window)
            
            if total > 0:
                prop = (window['emotion'] == emotion).sum() / total
            else:
                prop = 0
            emotion_data.append(prop)
        
        # Apply Gaussian smoothing
        if len(emotion_data) > 5:  # Only smooth if enough data points
            emotion_data = gaussian_filter1d(emotion_data, sigma=1.5)
        
        # Plot for this emotion
        axes[i].plot(time_points, emotion_data, linewidth=2)
        axes[i].set_title(f'Emotion: {emotion.capitalize()}', fontsize=12)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('Proportion', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Overlay stimulus periods
        stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
        for stim in ['warm', 'cold', 'neutral']:
            color = stim_colors[stim]
            for start, end, stim_type in STIM_PERIODS:
                if stim_type == stim:
                    axes[i].axvspan(start, end, color=color, alpha=0.2, label=stim)
        
        # Add legend but only for first plot
        if i == 0:
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=stim_colors[s], alpha=0.2, label=s) for s in stim_colors]
            axes[i].legend(handles=handles, loc='upper right')
    
    # Set x-axis label on the last subplot
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    # Main title
    fig.suptitle('Emotion Proportions Over Time (One per Emotion)', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename

def plot_streamgraph(preds, stim_labels, epoch_sec=2, filename='streamgraph.png'):
    """
    Plot emotion proportions over time as a streamgraph.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib import cm
    
    try:
        # This is a more advanced plot that requires scipy
        from scipy.ndimage import gaussian_filter1d
        
        # Create a DataFrame with time index
        df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
        df['epoch'] = df.index
        df['time_sec'] = df['epoch'] * epoch_sec
        
        # Get unique emotions and time points
        emotions = sorted(df['emotion'].unique())
        time_points = sorted(df['time_sec'].unique())
        
        # Compute emotion proportions per time point
        proportions = np.zeros((len(time_points), len(emotions)))
        
        for i, t in enumerate(time_points):
            window = df[df['time_sec'] == t]
            total = len(window)
            
            for j, emotion in enumerate(emotions):
                if total > 0:
                    proportions[i, j] = (window['emotion'] == emotion).sum() / total
                else:
                    proportions[i, j] = 0
        
        # Apply smoothing
        for j in range(proportions.shape[1]):
            if proportions.shape[0] > 5:  # Only smooth if enough data points
                proportions[:, j] = gaussian_filter1d(proportions[:, j], sigma=2)
        
        # Create streamgraph
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Get colormap
        colors = cm.tab10(np.linspace(0, 1, len(emotions)))
        
        # Plot each layer
        layers = ax.stackplot(time_points, proportions.T, colors=colors, labels=emotions, alpha=0.8)
        
        # Make it look like a streamgraph (centered around 0)
        baseline = np.zeros_like(time_points)
        for i in range(len(emotions)):
            baseline = baseline - proportions[:, i] / 2
        
        ax.stackplot(time_points, [baseline], colors=['none'])
        
        # Overlay stimulus periods
        stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
        for stim in ['warm', 'cold', 'neutral']:
            color = stim_colors[stim]
            for start, end, stim_type in STIM_PERIODS:
                if stim_type == stim:
                    ax.axvspan(start, end, color=color, alpha=0.2)
        
        # Add stimulus legend separately
        from matplotlib.patches import Patch
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2 = [Patch(facecolor=stim_colors[s], alpha=0.2, label=s) for s in stim_colors]
        
        # Combine legends
        ax.legend(handles=handles1 + handles2, 
                 labels=labels1 + list(stim_colors.keys()),
                 loc='upper right')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        ax.set_title('Emotion Proportions Over Time (Streamgraph)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
    except Exception as e:
        # Fallback if streamgraph fails
        print(f"Error generating streamgraph: {e}. Falling back to stacked area plot.")
        plot_stacked_area(preds, stim_labels, epoch_sec, filename)
    
    return filename

def plot_band_power_bands_all_channels_together(epochs, fs, filename, file_label):
    """
    Plot a heatmap of band power for all bands and all channels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    
    # Calculate band power for each epoch, channel, and band
    bandpowers = np.zeros((n_epochs, n_channels, len(band_names)))
    
    for i in range(n_epochs):
        for ch in range(n_channels):
            for b, band in enumerate(band_names):
                bandpowers[i, ch, b] = compute_band_power(epochs[i, :, ch], fs, band)
    
    # Normalize band powers per channel (z-score)
    mean_bp = np.mean(bandpowers, axis=0, keepdims=True)
    std_bp = np.std(bandpowers, axis=0, keepdims=True) + 1e-8
    z_bandpowers = (bandpowers - mean_bp) / std_bp
    
    # Reshape to create a DataFrame for plotting
    data = []
    for e in range(n_epochs):
        for ch in range(n_channels):
            for b, band in enumerate(band_names):
                data.append({
                    'Epoch': e,
                    'Channel': CHANNELS[ch],
                    'Band': band,
                    'Power': bandpowers[e, ch, b],
                    'Z-Power': z_bandpowers[e, ch, b]
                })
    
    df = pd.DataFrame(data)
    
    # Create heatmap plots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Raw band power
    pivot1 = df.pivot_table(
        index=['Channel', 'Band'], 
        columns='Epoch', 
        values='Power',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot1, ax=axes[0], cmap='viridis', linewidths=0.1, 
                cbar_kws={'label': 'Band Power'})
    axes[0].set_title(f'Band Power Heatmap - Raw\nFile: {file_label}', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    
    # Plot 2: Z-scored band power
    pivot2 = df.pivot_table(
        index=['Channel', 'Band'], 
        columns='Epoch', 
        values='Z-Power',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot2, ax=axes[1], cmap='coolwarm', center=0, linewidths=0.1,
                cbar_kws={'label': 'Z-scored Band Power'})
    axes[1].set_title(f'Band Power Heatmap - Z-scored\nFile: {file_label}', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    
    # Add descriptive text
    fig.suptitle(f'EEG Band Power Across All Channels and Bands\nFile: {file_label}', 
                fontsize=16, y=0.98)
    
    text = (
        "This visualization shows band power for all EEG frequency bands across all channels and epochs.\n\n"
        "Top plot: Raw band power values.\n"
        "Bottom plot: Z-scored band power (normalized per channel and band).\n\n"
        "The heatmap rows represent each channel-band combination.\n"
        "The columns represent time progression (epochs).\n\n"
        "Color intensity indicates the relative power in that frequency band at that point in time.\n"
        "Use this plot to identify patterns of brain activity across different regions and frequencies."
    )
    
    fig.text(0.1, 0.01, text, wrap=True, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return filename

def plot_comparative_alpha_beta_ratio(ratio_csvs, filename, file_labels=None):
    """
    Plot a comparison of alpha/beta ratios across multiple files.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Load all ratio CSVs
    dfs = []
    if file_labels is None:
        file_labels = []
        
    for csv_file in ratio_csvs:
        try:
            df = pd.read_csv(csv_file)
            if 'file' in df.columns:
                file_label = df['file'].iloc[0]
            else:
                file_label = os.path.basename(csv_file).split('_')[0]
                
            file_labels.append(file_label)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        print("No valid ratio CSVs found")
        return filename
    
    # Create plot with one subplot per channel
    n_channels = len(CHANNELS)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]  # Make it iterable if only one channel
    
    # Plot alpha/beta ratio for each file and channel
    for i, channel in enumerate(CHANNELS):
        for j, df in enumerate(dfs):
            if channel in df.columns:
                axes[i].plot(df['epoch'], df[channel], label=file_labels[j], alpha=0.7)
        
        axes[i].set_title(f'Channel: {channel}', fontsize=12)
        axes[i].set_ylabel('Alpha/Beta Ratio', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add legend only for first plot
        if i == 0:
            axes[i].legend(fontsize=9, loc='upper right')
    
    # Set x-axis label on the last subplot
    axes[-1].set_xlabel('Epoch', fontsize=12)
    
    # Main title
    fig.suptitle('Comparative Alpha/Beta Ratio Across Files', fontsize=16, y=1.02)
    
    # Add explanatory text
    text = (
        "This plot compares the alpha/beta power ratio across different EEG recordings.\n\n"
        f"Files included: {', '.join(file_labels)}\n\n"
        "The alpha/beta ratio is an indicator of relaxation vs. active processing:\n"
        "- Higher ratio (more alpha) suggests relaxed, meditative states\n"
        "- Lower ratio (more beta) suggests active cognitive processing, focus or stress\n\n"
        "Each subplot shows one electrode channel, allowing comparison of the ratio's dynamics across different brain regions."
    )
    
    fig.text(0.1, 0.01, text, wrap=True, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return filename

def plot_comparative_emotion_bar(emotion_csvs, filename, file_labels=None):
    """
    Plot a comparison of emotion proportions across multiple files.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Load all emotion CSVs
    all_data = []
    if file_labels is None:
        file_labels = []
    
    for csv_file in emotion_csvs:
        try:
            df = pd.read_csv(csv_file)
            if 'file' in df.columns:
                file_label = df['file'].iloc[0]
            else:
                file_label = os.path.basename(csv_file).split('_')[0]
                
            file_labels.append(file_label)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No valid emotion CSVs found")
        return filename
    
    # Combine all data
    combined = pd.concat(all_data)
    
    # Get all possible emotions and stimuli
    emotion_cols = [col for col in combined.columns if col.endswith('_pct')]
    emotions = [col.split('_')[0] for col in emotion_cols]
    stimuli = sorted(combined['stimulus'].unique())
    
    # Create grouped bar chart
    n_files = len(file_labels)
    n_emotions = len(emotions)
    n_stimuli = len(stimuli)
    
    # Set up the figure
    fig, axes = plt.subplots(n_stimuli, 1, figsize=(14, 4*n_stimuli), sharex=True)
    
    if n_stimuli == 1:
        axes = [axes]  # Make it iterable if only one stimulus
    
    # Plot bars for each stimulus
    bar_width = 0.8 / n_files
    
    for s, stimulus in enumerate(stimuli):
        x = np.arange(n_emotions)
        
        for f, (file_label, df) in enumerate(zip(file_labels, all_data)):
            stim_data = df[df['stimulus'] == stimulus]
            
            if len(stim_data) > 0:
                values = []
                for emotion in emotions:
                    col = f"{emotion}_pct"
                    if col in stim_data.columns:
                        values.append(stim_data[col].values[0])
                    else:
                        values.append(0)
                
                pos = x - 0.4 + (f + 0.5) * bar_width
                axes[s].bar(pos, values, width=bar_width, label=file_label if s == 0 else "")
        
        axes[s].set_title(f'Stimulus: {stimulus.capitalize()}', fontsize=12)
        axes[s].set_ylabel('Percentage (%)', fontsize=10)
        axes[s].set_xticks(x)
        axes[s].set_xticklabels([e.capitalize() for e in emotions])
        axes[s].grid(True, alpha=0.3, axis='y')
        
        # Add legend only for first plot
        if s == 0:
            axes[s].legend(fontsize=10, loc='upper right')
    
    # Main title
    fig.suptitle('Comparative Emotion Distribution by Stimulus Across Files', fontsize=16, y=1.02)
    
    # Add explanatory text
    text = (
        "This plot compares the emotion classification distribution across different EEG recordings.\n\n"
        f"Files included: {', '.join(file_labels)}\n\n"
        "Each subplot shows a different stimulus condition (warm, cold, neutral).\n"
        "For each stimulus, the bars represent the percentage of epochs classified as each emotion.\n"
        "This allows comparison of emotional responses between different subjects or sessions."
    )
    
    fig.text(0.1, 0.01, text, wrap=True, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return filename

def plot_emotion_avg_per_stimulus_window(preds, stim_labels, epoch_sec, filename, file_label=None):
    """
    Plot the average proportion of each emotion per stimulus window.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Create DataFrame
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['epoch'] = df.index
    df['time_sec'] = df['epoch'] * epoch_sec
    
    # Get stimulus periods
    stimulus_periods = []
    for start, end, stim_type in STIM_PERIODS:
        stimulus_periods.append({
            'start': start, 
            'end': end, 
            'type': stim_type,
            'label': f"{stim_type} ({start}-{end}s)"
        })
    
    # Calculate emotion percentages for each stimulus period
    results = []
    for period in stimulus_periods:
        mask = (df['time_sec'] >= period['start']) & (df['time_sec'] < period['end'])
        period_data = df[mask]
        
        if len(period_data) > 0:
            emotion_counts = period_data['emotion'].value_counts(normalize=True) * 100
            
            for emotion, pct in emotion_counts.items():
                results.append({
                    'Period': period['label'],
                    'Type': period['type'],
                    'Emotion': emotion,
                    'Percentage': pct
                })
    
    if not results:
        print("No data for stimulus periods")
        return filename
    
    result_df = pd.DataFrame(results)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot grouped bar chart
    period_types = sorted(result_df['Type'].unique())
    emotions = sorted(result_df['Emotion'].unique())
    periods = sorted(result_df['Period'].unique(), 
                    key=lambda x: int(x.split('(')[1].split('-')[0]))
    
    # Set position for bars
    positions = np.arange(len(periods))
    width = 0.8 / len(emotions)
    
    # Plot bars for each emotion
    for i, emotion in enumerate(emotions):
        mask = result_df['Emotion'] == emotion
        data = result_df[mask]
        
        # Prepare data in correct order
        values = []
        for period in periods:
            val = data[data['Period'] == period]['Percentage'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        pos = positions - 0.4 + (i + 0.5) * width
        ax.bar(pos, values, width=width, label=emotion)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)', fontsize=12)
    
    # Add title with file label if provided
    if file_label:
        ax.set_title(f'Emotion Distribution by Stimulus Period\nFile: {file_label}', fontsize=14)
    else:
        ax.set_title('Emotion Distribution by Stimulus Period', fontsize=14)
    
    ax.legend(fontsize=10, title='Emotion')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color the background by stimulus type
    stim_colors = {'warm': '#FFDDC1', 'cold': '#B5D8FA', 'neutral': '#EEEEEE'}
    for i, period in enumerate(periods):
        stim_type = result_df[result_df['Period'] == period]['Type'].values[0]
        if stim_type in stim_colors:
            ax.axvspan(i-0.5, i+0.5, color=stim_colors[stim_type], alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    return filename

def create_output_folder(base_name="EEG_Analysis"):
    """
    Creates a new output folder with the format 'base_name_dd_mm_yyyy' and returns the path.
    This ensures all outputs from a single run go to the same dedicated folder.
    """
    import os
    import datetime
    
    # Get current date in dd_mm_yyyy format
    current_date = datetime.datetime.now().strftime('%d_%m_%Y')
    folder_name = f"{base_name}_{current_date}"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print_progress(f"Created output folder: {folder_name}")
    else:
        # If folder already exists, create a unique one by adding a counter
        counter = 1
        while os.path.exists(f"{folder_name}_{counter}"):
            counter += 1
        folder_name = f"{folder_name}_{counter}"
        os.makedirs(folder_name)
        print_progress(f"Created output folder: {folder_name}")
    
    return folder_name
