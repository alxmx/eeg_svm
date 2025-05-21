# EEG Mindfulness Index Calculator - User Guide

This document provides a guide for using the EEG Mindfulness Index Calculator to analyze EEG data and compute a mindfulness score.

## Overview

The Mindfulness Index (MI) Calculator processes EEG data to quantify mindfulness states based on neurophysiological features. The algorithm combines multiple EEG biomarkers known to correlate with focused attention and meditative states.

## Features

- **EEG Signal Processing**: Bandpass filtering, artifact rejection, and windowing
- **Feature Extraction**: Frontal Theta, Posterior Alpha, Frontal Alpha Asymmetry, Frontal Beta
- **Mindfulness Index Calculation**: Weighted combination of EEG features (and EDA if available)
- **Behavioral State Classification**: Focused, Neutral, or Unfocused
- **Visualization**: Time series plots, feature contribution analysis, state summaries
- **Reporting**: JSON, CSV, and TXT formats with comprehensive metrics

## Technical Details

The mindfulness index is calculated using the following formula:

```
MI = (0.25 * Theta_Fz) + (0.25 * Alpha_PO) + (0.20 * FAA) - (0.15 * Beta_Frontal) - (0.15 * EDA_norm)
```

Where:
- **Theta_Fz**: 4-7 Hz power at frontal electrode Fz (ch1)
- **Alpha_PO**: 8-12 Hz power averaged across posterior electrodes PO7 (ch6) and PO8 (ch8)
- **FAA**: Frontal Alpha Asymmetry = log(alpha at C4) - log(alpha at C3)
- **Beta_Frontal**: 13-30 Hz power averaged across Fz, C3, and C4
- **EDA_norm**: Normalized electrodermal activity (if available)

## How to Use

### Installation Requirements

This tool requires Python 3.6+ and the following libraries:
- numpy
- pandas
- scipy
- matplotlib

Install them using pip:
```
pip install numpy pandas scipy matplotlib
```

### Basic Usage

1. Place your EEG data (CSV format) in the `data/toClasify` directory
2. Run the script:
   ```
   python eeg_mindfulness_index.py
   ```
3. Results will be stored in the `results/mindfulness_analysis` directory

### Input Data Format

The script expects CSV files with EEG data organized in columns:
- Column 1: Timestamp
- Columns 2-9: EEG channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8)

EDA data (optional) should be in a similarly named file with `_eda.csv` suffix.

### Understanding Output

For each processed file, you'll receive:
- **JSON Results**: Comprehensive data for each time window
- **CSV Data**: Tabular format of all features and scores
- **MI Plot**: Time series of Mindfulness Index with state classification
- **Feature Contributions**: Visualization of how each feature affects the MI
- **State Summary**: Distribution of behavioral states in the recording
- **Text Summary**: Overview of results and statistics

## Customization

You can modify the following parameters in the script:
- `WINDOW_SEC`: Window size (default: 3 seconds)
- `OVERLAP`: Window overlap (default: 0.5 or 50%)
- `MI_WEIGHTS`: Change the weighting of different features
- `THRESHOLDS`: Adjust boundary values for state classification

## Examples

### Example 1: Processing a Single File

```python
# Inside eeg_mindfulness_index.py
if __name__ == "__main__":
    eeg_file = "data/toClasify/your_eeg_file.csv"
    results = process_eeg_file(eeg_file)
    generate_report(results, "your_eeg_file", "results")
```

### Example 2: Custom Weights

To adjust the importance of different features, modify the MI_WEIGHTS dictionary:

```python
# Custom weights that emphasize theta activity
MI_WEIGHTS = {
    'theta_fz': 0.35,      # Increased from 0.25
    'alpha_po': 0.20,      # Decreased from 0.25
    'faa': 0.20,
    'beta_frontal': -0.15,
    'eda_norm': -0.10      # Decreased from -0.15
}
```

## Troubleshooting

### Common Issues

1. **Missing Data Error**: Ensure your CSV files have the correct format with 8 EEG channels
2. **No Results Generated**: Check that your data directory path is correct
3. **Visualization Error**: Make sure matplotlib is properly installed

### Getting Help

If you encounter issues, check:
1. Python and package versions
2. Data format compatibility
3. File permissions in the results directory

## Future Development

Planned enhancements:
- Real-time processing via UDP
- Adjustable preprocessing parameters
- Integration with visualization dashboards
- Support for additional EEG features

---

## References

1. Lomas, T., Ivtzan, I., & Fu, C. H. (2015). A systematic review of the neurophysiology of mindfulness on EEG oscillations. Neuroscience & Biobehavioral Reviews, 57, 401-410.
2. Brandmeyer, T., & Delorme, A. (2018). Meditation and neurofeedback. Frontiers in psychology, 9, 1403.
3. Chow, T., Javan, T., Ros, T., & Frewen, P. (2017). EEG dynamics of mindfulness meditation versus alpha neurofeedback: a sham-controlled study. Mindfulness, 8(3), 572-584.
