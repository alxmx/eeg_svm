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

def save_filtering_stats(eeg_raw, eeg_filt, filename):
    stats = []
    for ch in range(eeg_raw.shape[1]):
        stats.append({
            'Channel': CHANNELS[ch],
            'Mean (raw)': np.mean(eeg_raw[:, ch]),
            'Std (raw)': np.std(eeg_raw[:, ch]),
            'Mean (filt)': np.mean(eeg_filt[:, ch]),
            'Std (filt)': np.std(eeg_filt[:, ch])
        })
    df = pd.DataFrame(stats)
    df.to_csv(filename, index=False)
    return df

def save_feature_stats(X, filename):
    df = pd.DataFrame(X, columns=[f'{ch}_{band}_{stat}' for ch in CHANNELS for band in BANDS for stat in ['mean','var','std','kurt','skew']])
    stats = df.describe().T[['mean','std','min','max']]
    stats.to_csv(filename)
    return stats

def save_loso_reports(reports, filename):
    # reports is a list of dicts from classification_report(..., output_dict=True)
    all_rows = []
    for i, rep in enumerate(reports):
        for label, metrics in rep.items():
            if isinstance(metrics, dict):
                row = {'Fold': i, 'Label': label}
                row.update(metrics)
                all_rows.append(row)
    df = pd.DataFrame(all_rows)
    df.to_csv(filename, index=False)
    return df

def plot_and_save_all(eeg_raw, eeg_filt, X, preds, stim_labels, reports, out_prefix):
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{out_prefix}_report.pdf")
    # Time domain plot for all channels
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(12, 2*len(CHANNELS)), sharex=True)
    for ch in range(len(CHANNELS)):
        axes[ch].plot(eeg_raw[:1000, ch], label='Raw', alpha=0.7)
        axes[ch].plot(eeg_filt[:1000, ch], label='Filtered', alpha=0.7)
        axes[ch].set_ylabel(CHANNELS[ch])
        axes[ch].legend(loc='upper right')
    axes[-1].set_xlabel('Sample Index')
    fig.suptitle('EEG Example: Raw vs. Filtered (All Channels, First 1000 Samples)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig); plt.close(fig)
    # Info text for time-domain
    info_text = (
        "This figure shows the raw and filtered EEG signals for all channels. "
        "Filtering includes a 1-40 Hz bandpass and 50 Hz notch filter. "
        "Visual inspection helps identify noise/artifacts and the effect of filtering."
    )
    fig = plt.figure(figsize=(10, 1))
    plt.text(0.01, 0.5, info_text, fontsize=12, va='center')
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    # Frequency domain plot for all channels
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(12, 2*len(CHANNELS)), sharex=True)
    for ch in range(len(CHANNELS)):
        f, Pxx = welch(eeg_filt[:, ch], fs=FS, nperseg=FS*2)
        axes[ch].semilogy(f, Pxx)
        axes[ch].set_ylabel(CHANNELS[ch])
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle('Power Spectral Density (PSD) of Filtered EEG (All Channels)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig); plt.close(fig)
    # Info text for frequency-domain
    info_text = (
        "This figure shows the power spectral density (PSD) for each EEG channel after filtering. "
        "PSD reveals the distribution of signal power across frequencies, highlighting dominant rhythms (e.g., alpha, beta)."
    )
    fig = plt.figure(figsize=(10, 1))
    plt.text(0.01, 0.5, info_text, fontsize=12, va='center')
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    # Feature distribution
    fig = plt.figure(figsize=(14, 6))
    sns.violinplot(data=pd.DataFrame(X))
    plt.title('Feature Distribution (All Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Z-scored Value')
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)
    info_text = (
        "Violin plot of all extracted features (z-scored). "
        "Each feature represents a statistical metric (mean, var, std, kurtosis, skewness) for a channel and band. "
        "Distribution width indicates feature variability across epochs."
    )
    fig = plt.figure(figsize=(10, 1))
    plt.text(0.01, 0.5, info_text, fontsize=12, va='center')
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    # Correlation matrix
    fig = plt.figure(figsize=(10, 8))
    corr = pd.DataFrame(X).corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)
    info_text = (
        "Correlation matrix of all features. "
        "High correlation may indicate redundancy; low correlation suggests complementary information."
    )
    fig = plt.figure(figsize=(10, 1))
    plt.text(0.01, 0.5, info_text, fontsize=12, va='center')
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    # Emotion over time
    plot_stimulus_stats_time(preds, stim_labels, epoch_sec=EPOCH_SEC)
    fig = plt.figure()
    img = plt.imread('emotion_proportion_over_time.png')
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    info_text = (
        "Proportion of predicted emotions over time, with stimulus periods highlighted. "
        "Shows how emotional state estimates change in response to stimuli."
    )
    fig = plt.figure(figsize=(10, 1))
    plt.text(0.01, 0.5, info_text, fontsize=12, va='center')
    plt.axis('off')
    pdf.savefig(fig); plt.close(fig)
    pdf.close()

def print_progress(message):
    print(f"[PROGRESS] {message}")
    sys.stdout.flush()

# Ensure all plotting functions are defined before main
def plot_emotion_labels(preds, stim_labels, epoch_sec=2, filename='fig1_emotion_labels.png'):
    """Fig 1: Emotional Estimation with Labels"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    plt.figure(figsize=(16, 4))
    unique_emotions = df['emotion'].unique()
    colors = {e: c for e, c in zip(unique_emotions, sns.color_palette('tab10', n_colors=len(unique_emotions)))}
    for e in unique_emotions:
        idx = df['emotion'] == e
        plt.scatter(df.loc[idx, 'time_sec'], np.zeros(sum(idx)), color=colors[e], label=e, s=60)
    plt.title('Fig 1: Emotion Estimation per Window (Labels)')
    plt.xlabel('Time (s)')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig1_desc = "Fig 1 shows the predicted emotion for each time window, with the emotion label and color-coded points. This allows us to see the sequence and transitions of emotional states over time."


def plot_emotion_avg_per_stimulus(preds, stim_labels, filename='fig2_emotion_avg_per_stimulus.png'):
    """Fig 2: Average of Emotions per Stimulus Period"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    avg = df.groupby('stimulus')['emotion'].value_counts(normalize=True).unstack().fillna(0)
    avg.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10,6))
    plt.title('Fig 2: Average Emotion Proportion per Stimulus Period')
    plt.ylabel('Proportion')
    plt.xlabel('Stimulus Period')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig2_desc = "Fig 2 displays the average proportion of each emotion within each stimulus period, highlighting how emotional responses differ between warm, cold, and neutral periods."


def plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, epoch_sec, filename, file_label):
    """
    Plot average emotion proportion per stimulus period, one bar per window (sampling the stimulus), over time.
    Each bar shows the average emotion for the corresponding stimulus period.
    """
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['window'] = df.index
    df['time_sec'] = df['window'] * epoch_sec
    # Find unique stimulus periods and their window indices
    periods = []
    last_stim = None
    start_idx = 0
    for idx, stim in enumerate(df['stimulus']):
        if stim != last_stim:
            if last_stim is not None:
                periods.append((last_stim, start_idx, idx))
            last_stim = stim
            start_idx = idx
    periods.append((last_stim, start_idx, len(df)))
    # For each period, compute average emotion proportion
    period_labels = []
    period_props = []
    for stim, start, end in periods:
        sub = df.iloc[start:end]
        prop = sub['emotion'].value_counts(normalize=True).reindex(df['emotion'].unique(), fill_value=0)
        period_labels.append(f"{stim} ({start*epoch_sec}-{end*epoch_sec}s)")
        period_props.append(prop.values)
    period_props = np.array(period_props)
    plt.figure(figsize=(14, 6))
    bottom = np.zeros(period_props.shape[0])
    emotions = df['emotion'].unique()
    for i, emo in enumerate(emotions):
        plt.bar(np.arange(len(period_labels)), period_props[:, i], bottom=bottom, label=emo)
        bottom += period_props[:, i]
    plt.xticks(np.arange(len(period_labels)), period_labels, rotation=30, ha='right')
    plt.ylabel('Proportion')
    plt.xlabel('Stimulus Period')
    plt.title(f'Average Emotion Proportion per Stimulus Period (File: {file_label})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_stacked_area(preds, stim_labels, epoch_sec=2, filename='fig3_stacked_area.png'):
    """Fig 3: Stacked Area Chart"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)
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
    plt.title('Fig 3: Stacked Area Chart of Emotion Proportions Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig3_desc = "Fig 3 is a stacked area chart showing the proportion of each emotion over time, emphasizing overall trends and the composition of emotional states."


def plot_line_chart(preds, stim_labels, epoch_sec=2, filename='fig4_line_chart.png'):
    """Fig 4: Line Chart"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(14, 7))
    for col in pivot_prop.columns:
        plt.plot(pivot_prop.index, pivot_prop[col], label=col)
    plt.title('Fig 4: Line Chart of Emotion Proportions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig4_desc = "Fig 4 is a line chart tracking the proportion of each emotion over time, making it easy to compare individual emotion trends."


def plot_stacked_bar(preds, stim_labels, epoch_sec=2, filename='fig5_stacked_bar.png'):
    """Fig 5: Stacked Bar Chart"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    # Use every Nth window for clarity
    N = max(1, len(df)//30)
    df_sub = df.iloc[::N]
    pivot = df_sub.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    pivot.plot(kind='bar', stacked=True, colormap='tab10', figsize=(14,7))
    plt.title('Fig 5: Stacked Bar Chart of Emotion Counts at Intervals')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig5_desc = "Fig 5 is a stacked bar chart showing the count of each emotion at selected time intervals, allowing for discrete comparison of emotion contributions."


def plot_heatmap(preds, stim_labels, epoch_sec=2, filename='fig6_heatmap.png'):
    """Fig 6: Heatmap"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot.T, cmap='YlGnBu', cbar=True)
    plt.title('Fig 6: Heatmap of Emotion Counts Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig6_desc = "Fig 6 is a heatmap showing the intensity of each emotion over time, making it easy to spot patterns and periods of high or low emotional presence."


def plot_small_multiples(preds, stim_labels, epoch_sec=2, filename='fig7_small_multiples.png'):
    """Fig 7: Small Multiples"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    n = len(pivot.columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2*n), sharex=True)
    for i, col in enumerate(pivot.columns):
        axes[i].plot(pivot.index, pivot[col], label=col, color=sns.color_palette('tab10')[i])
        axes[i].set_ylabel(col)
        axes[i].legend()
    plt.xlabel('Time (s)')
    plt.suptitle('Fig 7: Small Multiples of Emotion Counts Over Time')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename)
    plt.close()

fig7_desc = "Fig 7 shows small multiples: a grid of line charts, one per emotion, for detailed inspection of each emotion's trend over time."


def plot_streamgraph(preds, stim_labels, epoch_sec=2, filename='fig8_streamgraph.png'):
    """Fig 8: Streamgraph (stylized area chart)"""
    df = pd.DataFrame({'emotion': preds, 'stimulus': stim_labels})
    df['time_sec'] = df.index * epoch_sec
    pivot = df.pivot_table(index='time_sec', columns='emotion', aggfunc='size', fill_value=0)
    pivot_prop = pivot.div(pivot.sum(axis=1), axis=0)
    # Streamgraph effect: center baseline
    y = pivot_prop.values.T
    y_offset = y - y.mean(axis=0)
    plt.figure(figsize=(14, 7))
    plt.stackplot(pivot_prop.index, y_offset, labels=pivot_prop.columns, colors=sns.color_palette('tab10', n_colors=y.shape[0]), alpha=0.85)
    plt.title('Fig 8: Streamgraph of Emotion Proportions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Centered Proportion')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

fig8_desc = "Fig 8 is a streamgraph, a stylized area chart with a flowing baseline, providing an aesthetic view of shifting emotion proportions over time."

def save_all_figures_to_pdf(fig_filenames, pdf_filename):
    """Save all listed figure PNGs into a single PDF file."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    pdf = PdfPages(pdf_filename)
    for fname in fig_filenames:
        fig = plt.figure()
        img = plt.imread(fname)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()

# --- Artifact Detection ---
def detect_artifacts(acc_data, gyr_data, threshold=0.3):
    """Detect motion artifacts using accelerometer and gyroscope data.
    Args:
        acc_data: Accelerometer data (channels 8-10, zero-based index 8:11)
        gyr_data: Gyroscope data (channels 11-13, zero-based index 11:14)
        threshold: Motion detection threshold
    Returns:
        boolean array: True for clean data, False for artifacts
    """
    acc_magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
    gyr_magnitude = np.sqrt(np.sum(gyr_data**2, axis=1))
    motion_score = acc_magnitude + gyr_magnitude
    return motion_score <= threshold

def plot_band_power_over_time(epochs, fs, filename_prefix='band_power_over_time'):
    """
    Plot a heatmap of band power over time for each channel.
    epochs: shape (n_epochs, n_samples, n_channels)
    fs: sampling frequency
    """
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    band_powers = np.zeros((n_epochs, n_channels, len(band_names)))
    for i in range(n_epochs):
        for ch in range(n_channels):
            for b, band in enumerate(BANDS.values()):
                band_powers[i, ch, b] = compute_band_power(epochs[i, :, ch], fs, band)
    # Plot for each channel
    for ch in range(n_channels):
        plt.figure(figsize=(10, 4))
        plt.imshow(band_powers[:, ch, :].T, aspect='auto', origin='lower',
                   extent=[0, n_epochs, 0, len(band_names)], cmap='viridis')
        plt.yticks(np.arange(len(band_names)) + 0.5, band_names)
        plt.colorbar(label='Power')
        plt.xlabel('Epoch (Time)')
        plt.ylabel('Band')
        plt.title(f'Band Power Over Time - Channel {CHANNELS[ch]}')
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{CHANNELS[ch]}.png')
        plt.close()

def plot_band_power_over_time_all_channels(epochs, fs, filename, file_label):
    """
    Plot a heatmap of band power over time for all channels in one plot.
    epochs: shape (n_epochs, n_samples, n_channels)
    fs: sampling frequency
    filename: output filename (png)
    file_label: name of the file being visualized
    """
    n_epochs, _, n_channels = epochs.shape
    band_names = list(BANDS.keys())
    band_powers = np.zeros((n_epochs, n_channels, len(band_names)))
    for i in range(n_epochs):
        for ch in range(n_channels):
            for b, band in enumerate(BANDS.values()):
                band_powers[i, ch, b] = compute_band_power(epochs[i, :, ch], fs, band)
    # band_powers: (n_epochs, n_channels, n_bands)
    # For each channel, plot band power over time (one subplot per channel)
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(14, 2*len(CHANNELS)), sharex=True)
    for ch in range(n_channels):
        im = axes[ch].imshow(band_powers[:, ch, :].T, aspect='auto', origin='lower',
                            extent=[0, n_epochs, 0, len(band_names)], cmap='viridis')
        axes[ch].set_yticks(np.arange(len(band_names)) + 0.5)
        axes[ch].set_yticklabels(band_names)
        axes[ch].set_ylabel(CHANNELS[ch])
    axes[-1].set_xlabel('Epoch (Time)')
    fig.suptitle(f'Band Power Over Time (All Channels)\nFile: {file_label}')
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.01, pad=0.01)
    cbar.set_label('Power')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename)
    plt.close(fig)

# --- Main Pipeline ---
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
    plot_band_power_over_time(epochs, FS)
    print_progress("Classifying windows for new EEG file (progress bar)...")
    preds = classify_windows_progress(clf, scaler, feats)
    n_windows = len(preds)
    stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)
    print_progress("Plotting emotion proportions over time (improved)...")
    plot_stimulus_stats_time(preds, stim_labels, epoch_sec=EPOCH_SEC)
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

# --- Per-file visualizations ---
    test_files = glob.glob('data/toClasify/*.csv')
    for test_file in test_files:
        file_label = os.path.basename(test_file)
        print_progress(f"Analyzing EEG file: {file_label} using advanced features...")
        eeg_raw = load_eeg_csv(test_file)
        eeg_filt = bandpass_filter(eeg_raw, *BANDPASS, FS)
        eeg_filt = notch_filter(eeg_filt, NOTCH_FREQ, FS)
        epochs = windowed_epochs(eeg_filt, FS, EPOCH_SEC, EPOCH_OVERLAP)
        feats = [extract_stat_features(ep, FS) for ep in epochs]
        feats = np.array(feats)
        feats = (feats - mu) / (sigma + 1e-8)  # z-score using training stats
        print_progress(f"Visualizing band power over time for all channels in {file_label}...")
        plot_band_power_over_time_all_channels(epochs, FS, filename=f'band_power_over_time_all_{file_label}.png', file_label=file_label)
        print_progress(f"Classifying windows for {file_label} (progress bar)...")
        preds = classify_windows_progress(clf, scaler, feats)
        n_windows = len(preds)
        stim_labels = get_stimulus_labels(n_windows, FS, EPOCH_SEC)
        print_progress(f"Plotting average emotion per stimulus period for {file_label}...")
        plot_avg_emotion_per_stimulus_over_time(preds, stim_labels, EPOCH_SEC, filename=f'avg_emotion_per_stimulus_{file_label}.png', file_label=file_label)
        print_progress(f"Plotting pattern-based emotion labels for {file_label}...")
        plot_eeg_pattern_based_labels(epochs, FS, filename=f'pattern_based_emotion_labels_{file_label}.png', file_label=file_label)
        # --- PDF Report: add new per-file plots and text blocks ---
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"EEG_Analysis_report.pdf")
    for test_file in test_files:
        file_label = os.path.basename(test_file)
        # Band power all channels
        img = plt.imread(f'band_power_over_time_all_{file_label}.png')
        fig = plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Band Power Over Time (All Channels)\nFile: {file_label}')
        pdf.savefig(fig)
        plt.close(fig)
        add_text_block_to_pdf(pdf, f"Band power over time for all electrodes. File visualized: {file_label}")
        # Avg emotion per stimulus
        img = plt.imread(f'avg_emotion_per_stimulus_{file_label}.png')
        fig = plt.figure(figsize=(14, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Average Emotion Proportion per Stimulus Period\nFile: {file_label}')
        pdf.savefig(fig)
        plt.close(fig)
        add_text_block_to_pdf(pdf, f"Average emotion proportion per stimulus period, one bar per window of the stimulus. File visualized: {file_label}")
        # Pattern-based labels
        img = plt.imread(f'pattern_based_emotion_labels_{file_label}.png')
        fig = plt.figure(figsize=(14, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Pattern-based Emotion Labels per Epoch\nFile: {file_label}')
        pdf.savefig(fig)
        plt.close(fig)
        add_text_block_to_pdf(pdf, f"Pattern-based emotion labels using EEG band power rules and z-score thresholding. File visualized: {file_label}")
    pdf.close()
