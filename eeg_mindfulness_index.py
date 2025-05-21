"""
EEG Mindfulness Index Pipeline

This script implements a mindfulness index calculation pipeline for EEG data, including:
- Data loading and preprocessing
- Signal filtering (bandpass)
- Feature extraction: Frontal Theta, Posterior Alpha, Frontal Alpha Asymmetry, Frontal Beta, EDA
- Mindfulness Index (MI) calculation
- Behavioral state classification

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

Mindfulness Index Formula:
    MI = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)
    
    Default weights:
    w1 = 0.25 (Frontal Theta)
    w2 = 0.25 (Posterior Alpha)
    w3 = 0.20 (Frontal Alpha Asymmetry)
    w4 = 0.15 (Frontal Beta)
    w5 = 0.15 (EDA)
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt
import json
import glob
from datetime import datetime

# --- Parameters ---
FS = 250  # Sampling frequency (Hz)
BANDPASS = (4, 30)  # Hz, for general filtering
THETA_BAND = (4, 7)  # Hz
ALPHA_BAND = (8, 12)  # Hz
BETA_BAND = (13, 30)  # Hz
WINDOW_SEC = 3  # Window size for features (seconds)
OVERLAP = 0.5  # 50% overlap between windows
CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
CH_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

# MI calculation weights
MI_WEIGHTS = {
    'theta_fz': 0.25,
    'alpha_po': 0.25,
    'faa': 0.20,
    'beta_frontal': -0.15,
    'eda_norm': -0.15
}

# Behavioral state thresholds
THRESHOLDS = {
    'focused': 0.6,  # MI >= 0.6
    'neutral': 0.4,  # 0.4 <= MI < 0.6
    # 'unfocused': MI < 0.4
}

# --- Helper Functions ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def compute_band_power(data, fs, band):
    """Calculate power in a frequency band using Welch's method"""
    fmin, fmax = band
    f, Pxx = welch(data, fs=fs, nperseg=min(len(data), fs*2))
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    return np.mean(Pxx[idx_band]) if np.any(idx_band) else 0

def load_eeg_csv(filepath):
    """Load EEG data from CSV file"""
    print(f"Loading EEG data from: {filepath}")
    df = pd.read_csv(filepath, dtype=str)  # Read as string to avoid dtype warning
    # Convert all EEG channel columns to float, coerce errors to NaN
    eeg_data = df.iloc[:, 1:9].apply(pd.to_numeric, errors='coerce').values
    # Remove rows with any NaN (from conversion errors)
    eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
    return eeg_data

def load_eda_csv(filepath):
    """Load EDA data from CSV file if available"""
    try:
        # Adapt this based on your actual EDA data format
        eda_df = pd.read_csv(filepath, dtype=str)
        eda_data = eda_df.iloc[:, 1].apply(pd.to_numeric, errors='coerce').values
        return eda_data
    except Exception as e:
        print(f"No EDA data found or error loading EDA: {e}")
        return None

def create_windows(data, fs, window_sec, overlap):
    """Create overlapping windows from continuous data"""
    n_samples = data.shape[0]
    window_samples = int(fs * window_sec)
    step_samples = int(window_samples * (1 - overlap))
    
    windows = []
    timestamps = []
    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(data[start:end])
        timestamps.append(start / fs)  # Convert to seconds
    
    return np.array(windows), np.array(timestamps)

def normalize_eda(eda_data, method='zscore'):
    """Normalize EDA data using z-score or min-max scaling"""
    if eda_data is None:
        return None
    
    if method == 'zscore':
        return (eda_data - np.mean(eda_data)) / (np.std(eda_data) + 1e-8)
    elif method == 'minmax':
        return (eda_data - np.min(eda_data)) / (np.max(eda_data) - np.min(eda_data) + 1e-8)
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def preprocess_eeg(eeg_data, fs):
    """Apply preprocessing to EEG data"""
    # Apply bandpass filter
    eeg_filtered = bandpass_filter(eeg_data, *BANDPASS, fs)
    
    # Artifact rejection (remove windows with amplitude > ±100 µV)
    # This is simplified - in practice, you might want to keep track of rejected segments
    amplitude_threshold = 100  # µV
    valid_indices = np.all(np.abs(eeg_filtered) < amplitude_threshold, axis=1)
    eeg_cleaned = eeg_filtered[valid_indices]
    
    return eeg_cleaned

def extract_features(window, fs):
    """Extract relevant features from an EEG window"""
    features = {}
    
    # 1. Frontal Theta (Fz)
    theta_fz = compute_band_power(window[:, CH_IDX["Fz"]], fs, THETA_BAND)
    features['theta_fz'] = theta_fz
    
    # 2. Posterior Alpha (PO7/PO8)
    alpha_po7 = compute_band_power(window[:, CH_IDX["PO7"]], fs, ALPHA_BAND)
    alpha_po8 = compute_band_power(window[:, CH_IDX["PO8"]], fs, ALPHA_BAND)
    alpha_po = (alpha_po7 + alpha_po8) / 2
    features['alpha_po'] = alpha_po
    
    # 3. Frontal Alpha Asymmetry (FAA)
    alpha_c3 = compute_band_power(window[:, CH_IDX["C3"]], fs, ALPHA_BAND)
    alpha_c4 = compute_band_power(window[:, CH_IDX["C4"]], fs, ALPHA_BAND)
    # Use log transformation (avoid log(0))
    alpha_c3_log = np.log(alpha_c3 + 1e-8)
    alpha_c4_log = np.log(alpha_c4 + 1e-8)
    faa = alpha_c4_log - alpha_c3_log
    features['faa'] = faa
    
    # 4. Frontal Beta (Fz/C3/C4)
    beta_fz = compute_band_power(window[:, CH_IDX["Fz"]], fs, BETA_BAND)
    beta_c3 = compute_band_power(window[:, CH_IDX["C3"]], fs, BETA_BAND)
    beta_c4 = compute_band_power(window[:, CH_IDX["C4"]], fs, BETA_BAND)
    beta_frontal = (beta_fz + beta_c3 + beta_c4) / 3
    features['beta_frontal'] = beta_frontal
    
    return features

def calculate_mi(features, eda_value=None):
    """Calculate Mindfulness Index (MI) from features"""
    # Set EDA to 0 if not available
    if eda_value is None:
        eda_norm = 0
    else:
        eda_norm = eda_value
    features['eda_norm'] = eda_norm
    
    # Calculate MI using weights
    mi = 0
    for feature, weight in MI_WEIGHTS.items():
        if feature in features:
            mi += weight * features[feature]
    
    return mi

def classify_behavioral_state(mi):
    """Classify behavioral state based on MI value"""
    if mi >= THRESHOLDS['focused']:
        return "Focused"
    elif mi >= THRESHOLDS['neutral']:
        return "Neutral"
    else:
        return "Unfocused"

def process_eeg_file(eeg_filepath, eda_filepath=None):
    """Process a single EEG file and return results"""
    # Load data
    eeg_data = load_eeg_csv(eeg_filepath)
    eda_data = load_eda_csv(eda_filepath) if eda_filepath else None
    
    # Preprocess EEG
    eeg_cleaned = preprocess_eeg(eeg_data, FS)
    
    # Create windows
    windows, timestamps = create_windows(eeg_cleaned, FS, WINDOW_SEC, OVERLAP)
    
    # Normalize EDA (if available)
    eda_normalized = normalize_eda(eda_data) if eda_data is not None else None
    
    # Process each window
    results = []
    for i, (window, timestamp) in enumerate(zip(windows, timestamps)):
        # Extract features
        features = extract_features(window, FS)
        
        # Get EDA value for this window (if available)
        if eda_normalized is not None:
            # Find corresponding EDA value for this timestamp
            # This is simplified and assumes EDA is sampled at the same rate as EEG
            eda_idx = int(timestamp * FS)
            if eda_idx < len(eda_normalized):
                eda_value = eda_normalized[eda_idx]
            else:
                eda_value = None
        else:
            eda_value = None
        
        # Calculate MI
        mi = calculate_mi(features, eda_value)
        
        # Classify state
        behavioral_state = classify_behavioral_state(mi)
        
        # Create result entry
        result = {
            'timestamp': timestamp,
            'features': features,
            'eda_value': eda_value,
            'mi_score': mi,
            'behavioral_state': behavioral_state
        }
        results.append(result)
    
    return results

def save_results_to_json(results, output_filepath):
    """Save processing results to JSON file"""
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_filepath}")

def plot_mi_timeseries(results, output_filepath):
    """Plot MI values over time"""
    timestamps = [r['timestamp'] for r in results]
    mi_values = [r['mi_score'] for r in results]
    states = [r['behavioral_state'] for r in results]
    
    # Create color map for states
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    
    colors = [state_colors[state] for state in states]
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot MI values
    scatter = plt.scatter(timestamps, mi_values, c=colors, alpha=0.7)
    plt.plot(timestamps, mi_values, 'k-', alpha=0.3)
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7, label='Focused Threshold')
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7, label='Neutral Threshold')
    
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI)')
    plt.title('Mindfulness Index Over Time')
    
    # Create legend for states
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=state, markersize=10)
        for state, color in state_colors.items()
    ]
    legend_elements.append(Line2D([0], [0], linestyle='--', color='g', label='Focused Threshold'))
    legend_elements.append(Line2D([0], [0], linestyle='--', color='b', label='Neutral Threshold'))
    
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"MI plot saved to {output_filepath}")

def plot_feature_contributions(results, output_filepath):
    """Plot contributions of each feature to the MI score"""
    # Extract data
    timestamps = [r['timestamp'] for r in results]
    feature_names = list(MI_WEIGHTS.keys())
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate weighted feature values
    weighted_features = []
    for result in results:
        weighted_vals = {}
        for feature in feature_names:
            if feature == 'eda_norm' and result['eda_value'] is not None:
                val = MI_WEIGHTS[feature] * result['eda_value']
            elif feature != 'eda_norm':
                val = MI_WEIGHTS[feature] * result['features'].get(feature, 0)
            else:
                val = 0
            weighted_vals[feature] = val
        weighted_features.append(weighted_vals)
    
    # Create subplots
    plt.subplot(2, 1, 1)
    
    # Plot MI values
    mi_values = [r['mi_score'] for r in results]
    plt.plot(timestamps, mi_values, 'k-', linewidth=2, label='MI Score')
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7, label='Focused Threshold')
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7, label='Neutral Threshold')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI)')
    plt.title('Mindfulness Index Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature contributions
    plt.subplot(2, 1, 2)
    
    # Get weighted contribution of each feature
    for feature in feature_names:
        values = [wf[feature] for wf in weighted_features]
        plt.plot(timestamps, values, label=f"{feature} (w={MI_WEIGHTS[feature]})")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Weighted Contribution')
    plt.title('Feature Contributions to MI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"Feature contributions plot saved to {output_filepath}")

def plot_behavioral_state_summary(results, output_filepath):
    """Plot summary of behavioral states"""
    states = [r['behavioral_state'] for r in results]
    unique_states = ['Focused', 'Neutral', 'Unfocused']
    state_counts = {state: states.count(state) for state in unique_states}
    
    # Calculate percentages
    total = len(states)
    state_pcts = {state: (count / total) * 100 for state, count in state_counts.items()}
    
    # Create colors
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    colors = [state_colors[state] for state in unique_states]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.subplot(1, 2, 1)
    plt.pie([state_counts[state] for state in unique_states], 
            labels=unique_states, 
            autopct='%1.1f%%', 
            colors=colors,
            explode=[0.05] * len(unique_states))
    plt.title('Distribution of Behavioral States')
    
    # Create bar chart
    plt.subplot(1, 2, 2)
    plt.bar(unique_states, [state_pcts[state] for state in unique_states], color=colors)
    plt.xlabel('Behavioral State')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage of Time in Each State')
    
    # Add exact percentages on top of bars
    for i, state in enumerate(unique_states):
        plt.text(i, state_pcts[state] + 1, f"{state_pcts[state]:.1f}%", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()
    
    print(f"Behavioral state summary plot saved to {output_filepath}")

def generate_report(results, file_label, output_dir='results'):
    """Generate comprehensive report with visualization and CSV output"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{file_label}_{timestamp}"
    
    # Save JSON results
    json_filepath = os.path.join(output_dir, f"{base_filename}_mindfulness_results.json")
    save_results_to_json(results, json_filepath)
    
    # Save CSV summary
    csv_data = []
    for r in results:
        row = {
            'timestamp': r['timestamp'],
            'mi_score': r['mi_score'],
            'behavioral_state': r['behavioral_state']
        }
        for feature, value in r['features'].items():
            row[feature] = value
        if r['eda_value'] is not None:
            row['eda_norm'] = r['eda_value']
        csv_data.append(row)
    
    csv_filepath = os.path.join(output_dir, f"{base_filename}_mindfulness_data.csv")
    pd.DataFrame(csv_data).to_csv(csv_filepath, index=False)
    print(f"CSV data saved to {csv_filepath}")
    
    # Create plots
    plot_mi_filepath = os.path.join(output_dir, f"{base_filename}_mi_timeseries.png")
    plot_mi_timeseries(results, plot_mi_filepath)
    
    feature_plot_filepath = os.path.join(output_dir, f"{base_filename}_feature_contributions.png")
    plot_feature_contributions(results, feature_plot_filepath)
    
    state_plot_filepath = os.path.join(output_dir, f"{base_filename}_behavioral_states.png")
    plot_behavioral_state_summary(results, state_plot_filepath)
    
    # Generate summary text file
    summary_filepath = os.path.join(output_dir, f"{base_filename}_summary.txt")
    
    with open(summary_filepath, 'w') as f:
        f.write("=== Mindfulness Index (MI) Analysis Summary ===\n\n")
        f.write(f"File: {file_label}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parameters
        f.write("Parameters:\n")
        f.write(f"- Window Size: {WINDOW_SEC} seconds\n")
        f.write(f"- Window Overlap: {OVERLAP * 100}%\n")
        f.write(f"- Frequency Bands: Theta {THETA_BAND}, Alpha {ALPHA_BAND}, Beta {BETA_BAND}\n")
        f.write(f"- MI Weights: {MI_WEIGHTS}\n")
        f.write(f"- State Thresholds: Focused >= {THRESHOLDS['focused']}, " 
                f"Neutral >= {THRESHOLDS['neutral']}, Unfocused < {THRESHOLDS['neutral']}\n\n")
        
        # Results summary
        states = [r['behavioral_state'] for r in results]
        mi_values = [r['mi_score'] for r in results]
        
        f.write("Results Summary:\n")
        f.write(f"- Total Windows Analyzed: {len(results)}\n")
        f.write(f"- Average MI Score: {np.mean(mi_values):.4f}\n")
        f.write(f"- MI Score Range: {np.min(mi_values):.4f} to {np.max(mi_values):.4f}\n\n")
        
        f.write("Behavioral States:\n")
        for state in ['Focused', 'Neutral', 'Unfocused']:
            count = states.count(state)
            percentage = (count / len(states)) * 100
            f.write(f"- {state}: {count} windows ({percentage:.1f}%)\n")
        
        f.write("\nOutput Files:\n")
        f.write(f"- JSON Data: {json_filepath}\n")
        f.write(f"- CSV Data: {csv_filepath}\n")
        f.write(f"- MI Plot: {plot_mi_filepath}\n")
        f.write(f"- Feature Plot: {feature_plot_filepath}\n")
        f.write(f"- State Plot: {state_plot_filepath}\n")
    
    print(f"Summary report saved to {summary_filepath}")
    
    return {
        'json': json_filepath,
        'csv': csv_filepath,
        'mi_plot': plot_mi_filepath,
        'feature_plot': feature_plot_filepath,
        'state_plot': state_plot_filepath,
        'summary': summary_filepath
    }

def process_directory(input_dir, output_dir='results'):
    """Process all EEG files in a directory"""
    # Get all CSV files in the directory
    eeg_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not eeg_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(eeg_files)} EEG files to process")
    
    # Process each file
    for eeg_file in eeg_files:
        file_label = os.path.splitext(os.path.basename(eeg_file))[0]
        print(f"\nProcessing {file_label}...")
        
        # Look for corresponding EDA file (if your EDA files follow a naming convention)
        # This is just a placeholder - adjust based on your actual file naming
        eda_file = os.path.join(input_dir, f"{file_label}_eda.csv")
        if not os.path.exists(eda_file):
            eda_file = None
        
        # Process EEG file
        results = process_eeg_file(eeg_file, eda_file)
        
        # Generate report
        generate_report(results, file_label, output_dir)
        
        print(f"Completed processing {file_label}")
    
    print(f"\nAll {len(eeg_files)} files processed")

if __name__ == "__main__":
    # Process a single file example
    # Replace with your actual file path
    data_dir = "data/toClasify"
    
    # Create results directory if it doesn't exist
    results_dir = "results/mindfulness_analysis"
    os.makedirs(results_dir, exist_ok=True)
    
    # Process all files in directory
    process_directory(data_dir, results_dir)
    
    print("Mindfulness index analysis complete.")
