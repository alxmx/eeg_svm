# Real-Time Mindfulness Index (MI) LSL Pipeline

## Project Goals
- **Estimate Mindfulness Index (MI) in real-time** from raw EEG and EDA sensor data using robust, user-calibrated ML models.
- **Provide per-user calibration, artifact correction, and cumulative reporting** for research and neurofeedback applications.
- **Stream MI values via LSL** for real-time feedback, visualization, or integration with external tools (e.g., Unity, LabRecorder).
- Error handling, workflow automation, and portability for easy setup and reliable operation.

## Pipeline Overview
1. **LSL Stream Selection:**
   - Selects correct EEG (18 channels: 8 EEG, 6 ACC/Gyro, 4 unused) and EDA (2 channels) LSL streams.
   - Handles missing sensors and allows fallback to generic models.
2. **Calibration:**
   - Per-user, 60 seconds at 250 Hz (15,000 samples).
   - Collects raw EEG, ACC/Gyro, and EDA data.
   - Computes features using bandpower (theta, alpha, beta, FAA) and EDA mean.
   - Fits artifact regression models for EEG using ACC/Gyro.
   - Fine-tunes SVR model and scaler for the user.
   - Saves user config, baseline features, and cumulative calibration reports.
3. **Real-Time MI Prediction:**
   - Buffers incoming EEG/EDA at 250 Hz.
   - Computes features on 3-second windows (750 samples).
   - Applies artifact regression and feature scaling.
   - Predicts MI using user-specific or generic SVR model.
   - Classifies MI (Focused/Neutral/Unfocused) and prints to console.
   - Streams MI to LSL (`processed_MI`) at user-configurable rate/interval.
   - Handles missing/invalid data robustly.
4. **Reporting & Visualization:**
   - Saves MI session data and summary reports.
   - Compares MI predictions to Unity labels (if available).
   - Generates and saves final MI plots.

## Setup Instructions
### 1. Environment
- Python 3.8+
- Install dependencies:
  ```sh
  pip install numpy pandas scikit-learn pylsl joblib matplotlib scipy
  ```
- (Optional) Use `environment.yml` for conda setup.

### 2. Folder Structure
- `models/` — Stores trained SVR models and scalers.
- `logs/` — Session logs, reports, and metrics.
- `visualizations/` — Saved MI plots.
- `user_configs/` — Per-user calibration/config files.
- `data/processed/`, `data/_eeg/`, `data/_eda/` — Data storage (no raw data saved by default).

### 3. Running the Pipeline
- Start your EEG and EDA LSL streams (ensure they are RAW, unnormalized).
- Run the main script:
  ```sh
  python realtime_mi_lsl.py
  ```
- Follow prompts to select streams, calibrate user, and configure MI output rates.
- MI values will be streamed to LSL (`processed_MI`) and saved to logs.
- Use LabRecorder or LSL4Unity to record or visualize MI in real time.

## Key Features
- **Per-user calibration** (60s, 250 Hz) with artifact regression.
- **Robust feature extraction** (bandpower, FAA, EDA mean).
- **Real-time MI prediction and LSL streaming** with user-configurable rates.
- **Cumulative reporting and visualization** at session end.
- **Handles EDA at 500 Hz** (downsamples/interpolates to 250 Hz).
- **No raw data saved** (only features, configs, and models).
- **Graceful fallback and error handling** for missing sensors or invalid calibration.

## Notes
- The pipeline expects **RAW, unconverted, unnormalized EEG and EDA** data.
- All timing is based on LSL timestamps.
- Online adaptation is disabled by default in real-time operation.
- Feature extraction is performed on sliding windows (3s for real-time, 1s for calibration).

## Troubleshooting
- If you see warnings about constant/NaN features, check your sensor connections and data quality.
- If EDA is at 500 Hz, it will be automatically resampled to 250 Hz.
- For any issues, consult the logs in the `logs/` folder.

## Contact
For questions or contributions, please contact the project maintainer or open an issue in the repository.
