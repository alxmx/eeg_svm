# Mindfulness Index (MI) Real-Time LSL Pipeline

This folder contains a self-contained, ready-to-run experiment for real-time Mindfulness Index (MI) estimation using LSL streams.

## Contents
- `realtime_mi_lsl.py`: Main experiment script
- `environment.yml`: Conda environment for all dependencies
- `models/`: Pre-trained and per-user models/scalers
- `user_configs/`: Example user config and baseline CSV
- `logs/`, `visualizations/`, `data/processed/`, `data/_eeg/`, `data/_eda/`: Output and data folders

## Usage
1. Create the conda environment:
   ```
   conda env create -f environment.yml
   conda activate eeg_mi
   ```
2. Run the experiment:
   ```
   python realtime_mi_lsl.py
   ```

## Notes
- LSL streams must be available on the network for EEG and EDA.
- LabRecorder or similar can be used to record LSL streams externally.
- The script will guide you through calibration and experiment steps.

---

For more details, see comments in `realtime_mi_lsl.py`.
