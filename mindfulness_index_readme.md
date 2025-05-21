# EEG Mindfulness Index Calculator

## Overview

This tool calculates a Mindfulness Index (MI) from EEG and EDA data. The MI is a composite measure that quantifies mindfulness levels based on several neurophysiological markers:

1. Frontal Theta power (Fz electrode)
2. Posterior Alpha power (PO7 and PO8 electrodes)
3. Frontal Alpha Asymmetry (C3 and C4 electrodes)
4. Frontal Beta power (Fz, C3, and C4 electrodes)
5. Electrodermal Activity (EDA)

The MI value ranges from 0 to 1, where higher values indicate higher mindfulness levels. The behavioral states are classified as:
- **Focused**: MI ≥ 0.5
- **Neutral**: 0.37 ≤ MI < 0.5
- **Unfocused**: MI < 0.37

## Features

- Processes EEG data from CSV files
- Supports EDA data in both CSV and OpenSignals (.txt) formats
- Automatic matching of EEG and EDA files
- Calculation of Mindfulness Index using a weighted formula
- Behavioral state classification
- Comprehensive reports and visualizations

## Formula

The Mindfulness Index is calculated using the following formula:

```
MI_raw = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)
MI = 1 / (1 + exp(-MI_raw + 1))  # Normalized to 0-1 range
```

Default weights:
- w1 = 0.25 (Frontal Theta)
- w2 = 0.25 (Posterior Alpha)
- w3 = 0.20 (Frontal Alpha Asymmetry)
- w4 = 0.15 (Frontal Beta)
- w5 = 0.15 (EDA)

## EEG Data Format

The tool expects EEG data in CSV format with the following channel layout:
1. Fz (Frontal)
2. C3 (Left Central)
3. Cz (Central Midline)
4. C4 (Right Central)
5. Pz (Parietal Midline)
6. PO7 (Left Parietal-Occipital)
7. Oz (Occipital)
8. PO8 (Right Parietal-Occipital)

## EDA Data Formats

The tool supports multiple EDA data formats:

1. **CSV Format**:
   - Regular CSV with EDA values in the second column
   - Named as `[filename]_eda.csv` where `[filename]` matches the EEG filename

2. **OpenSignals Format**:
   - Tab-separated text files (.txt)
   - Named as `opensignals_*.txt`
   - Contains header lines starting with `#` or `//`
   - EDA data is auto-detected from the header if possible

## EDA File Detection

The tool uses the following methods to match EDA files with EEG files:

1. **Exact name matching** (in order of preference):
   - `[filename]_eda.csv`
   - `[filename]_eda.txt`
   - `opensignals_[filename].txt`
   - `[filename].txt`

2. **Date-based matching**:
   - For EEG files named `UnicornRecorder_DD_MM_YYYY_*`, the tool will search for OpenSignals files containing the same date in format `YYYY-MM-DD`

3. **Fallback method**:
   - If no exact match is found, the tool will use the most recent OpenSignals file in the EDA directory

## Updated Features (May 2025)

- **Enhanced EDA Detection**: Improved algorithm for detecting and matching EDA files with EEG recordings
- **OpenSignals Format Support**: Better handling of OpenSignals (.txt) format, including header parsing
- **EDA Normalization**: Improved normalization to ensure valid EDA data within expected ranges
- **Sampling Rate Detection**: Automatic detection of EDA sampling rate from filenames
- **Comprehensive Error Handling**: Better validation and error reporting for data quality issues
- **Extended Documentation**: Clear explanation of supported formats and detection methods

## Usage

Run the script with:

```
python eeg_mindfulness_index.py
```

This will:
1. Process all EEG files in the `data/toClasify` directory
2. Look for corresponding EDA files in the `data/eda_data` directory
3. Calculate MI for each file
4. Generate reports and visualizations in the `results/mindfulness_analysis` directory

## Output

For each processed file, the tool generates:
- JSON data file with full analysis results
- CSV data file with timestamp, MI scores, and features
- Time series plot of MI values
- Feature contributions plot
- Behavioral state summary plot
- Text summary report

## Configuration

Key parameters can be adjusted at the top of the `eeg_mindfulness_index.py` file:

- `FS`: Sampling frequency (Hz)
- `BANDPASS`: General filtering range (Hz)
- `EDA_DATA_FOLDER`: Location of EDA data files
- `THETA_BAND`, `ALPHA_BAND`, `BETA_BAND`: Frequency band definitions
- `WINDOW_SEC`: Window size for feature extraction (seconds)
- `OVERLAP`: Overlap between windows (0.5 = 50%)
- `MI_WEIGHTS`: Weights for each component in the MI formula
- `THRESHOLDS`: Thresholds for behavioral state classification

## Testing

To verify EDA detection and processing, run:

```
python test_improved_eda_detection.py
```
