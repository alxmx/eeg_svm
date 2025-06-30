# Dual Calibration Mindfulness Index Pipeline - User Guide

## Overview

This pipeline implements a **dual-phase calibration system** that creates personalized mindfulness index (MI) thresholds for each user. Unlike the stable version that uses population-based thresholds, this system adapts to your individual physiological patterns through two distinct calibration phases.

## Key Features

### 🎯 Dual Calibration System
- **RELAXED Phase (20 seconds)**: Establishes your natural low mindfulness baseline
- **FOCUSED Phase (30 seconds)**: Establishes your peak attention baseline
- **Adaptive Mapping**: Creates personalized MI thresholds based on your unique patterns

### 🛡️ Robust Data Processing
- **Peak Suppression**: Median filtering to reduce signal artifacts
- **Outlier Rejection**: Automatic detection and removal of data anomalies
- **Artifact Detection**: Real-time identification of corrupted signals
- **Temporal Smoothing**: Median-based smoothing for stable MI output

### 📊 Personalized MI Calculation
- **User-Specific Thresholds**: MI range mapped to your calibrated baselines
- **Dynamic Range Optimization**: Maximizes sensitivity within your personal range
- **Session Consistency**: Uses saved calibration data across sessions

## Scientific Rationale

### Why Dual Calibration?

1. **Individual Differences**: People have vastly different baseline EEG/EDA patterns
2. **State-Dependent Calibration**: Captures both relaxed and focused states for full range mapping
3. **Improved Sensitivity**: Adaptive thresholds provide better discrimination than population norms
4. **Ecological Validity**: Calibration states mirror real meditation practice (relaxation → focused attention)

### Calibration States

**RELAXED State (Low Mindfulness Baseline)**:
- Eyes closed, natural mind-wandering
- Represents your default, non-meditative state
- Captures individual EEG alpha patterns and EDA levels

**FOCUSED State (High Mindfulness Baseline)**:
- Eyes open, sustained attention to breathing
- Represents your peak attentional engagement
- Captures individual concentration-related neural signatures

## Usage Instructions

### 1. Initial Setup

```bash
# Navigate to your project directory
cd c:\Users\lenin\Documents\GitHub\eeg_svm

# Run the dual calibration pipeline
python realtime_mi_lsl_dual_calibration.py
```

### 2. User Identification

When prompted, enter a unique user ID. This will be used to:
- Save your personal calibration data
- Load existing calibration in future sessions
- Generate personalized reports

### 3. Calibration Process

#### Phase 1: RELAXED Baseline (20 seconds)

**Preparation**:
- Sit comfortably in your chair
- Ensure a quiet environment
- Have your EEG/EDA equipment properly connected

**Instructions During Calibration**:
- ✅ Close your eyes gently
- ✅ Take slow, deep breaths
- ✅ Let your mind wander naturally
- ✅ Don't try to focus on anything specific
- ✅ Relax your muscles, especially face and shoulders
- ❌ Don't try to meditate or concentrate
- ❌ Don't engage in mental tasks

**Goal**: Capture your natural, relaxed baseline state

#### Phase 2: FOCUSED Baseline (30 seconds)

**Preparation**:
- Keep the same comfortable position
- Have a fixed point to look at (wall, object)

**Instructions During Calibration**:
- ✅ Open your eyes and look at a fixed point
- ✅ Focus your attention on your breathing
- ✅ Count your breaths: 1 (inhale), 2 (exhale), etc.
- ✅ When you reach 10, start over at 1
- ✅ If your mind wanders, gently return to counting
- ✅ Maintain alert but relaxed attention
- ❌ Don't force concentration
- ❌ Don't get frustrated if you lose count

**Goal**: Capture your peak focused attention state

### 4. Real-Time MI Processing

After calibration, the system will:
- Calculate personalized MI values using your adaptive thresholds
- Apply robust peak suppression and artifact rejection
- Output smooth, consistent MI values via LSL streams
- Display real-time progress and statistics

### 5. Session Reports

At the end of each session, you'll receive:
- **CSV Data File**: Complete session data with timestamps
- **Visualization Plots**: Comparison of adaptive vs. universal MI
- **Statistics Summary**: Mean, standard deviation, and dynamic range
- **Calibration Persistence**: Your thresholds are saved for future sessions

## Output Streams

The pipeline creates three LSL output streams:

1. **mindfulness_index**: Personalized adaptive MI (0-1 range)
2. **raw_mindfulness_index**: Raw MI (-5 to +5 range)
3. **emotional_mindfulness_index**: EMI variant

## File Structure

### User Configuration Files
- `user_configs/{user_id}_dual_calibration.json`: Your personal thresholds
- `user_configs/{user_id}_dual_baseline.csv`: Raw calibration feature data

### Session Data
- `logs/{user_id}_dual_calibration_session_{timestamp}.csv`: Complete session data
- `visualizations/{user_id}_dual_calibration_mi_{timestamp}.png`: Session plots

## Troubleshooting

### Common Issues

**1. Calibration Failed - No Valid Features**
- **Cause**: Poor signal quality during calibration
- **Solution**: 
  - Check electrode connections
  - Ensure proper skin contact
  - Minimize movement during calibration
  - Retry calibration process

**2. Low Dynamic Range Warning**
- **Cause**: Similar features between relaxed and focused states
- **Solution**:
  - Follow calibration instructions more carefully
  - Ensure genuine state differences (relaxed vs. focused)
  - Consider recalibrating with clearer mental states

**3. Inconsistent MI Values**
- **Cause**: Signal artifacts or poor calibration
- **Solution**:
  - Check for loose electrodes
  - Recalibrate if necessary
  - Ensure stable recording environment

### Best Practices

1. **Calibration Environment**:
   - Quiet, distraction-free room
   - Consistent lighting
   - Comfortable seating
   - Stable electrode placement

2. **Mental State Preparation**:
   - Practice relaxation techniques before RELAXED phase
   - Understand breathing-counting task before FOCUSED phase
   - Take breaks between phases if needed

3. **Signal Quality**:
   - Check electrode impedances before starting
   - Ensure proper skin preparation
   - Minimize head movement during calibration

## Technical Details

### Feature Extraction
The system extracts five key features:
- **theta_fz**: Frontal theta power (4-8 Hz)
- **alpha_po**: Posterior alpha power (8-13 Hz)
- **faa**: Frontal alpha asymmetry
- **beta_frontal**: Frontal beta power (13-30 Hz)
- **eda_norm**: Normalized EDA activity

### Adaptive Threshold Calculation
```
adaptive_mi = (universal_mi - relaxed_baseline) / (focused_baseline - relaxed_baseline)
```

Where:
- `universal_mi`: Population-based MI calculation
- `relaxed_baseline`: Your personal relaxed state MI
- `focused_baseline`: Your personal focused state MI

### Peak Suppression Algorithm
1. **Median Filtering**: 5-point median filter on raw signals
2. **Outlier Detection**: Median Absolute Deviation (MAD) method
3. **Artifact Replacement**: Outliers replaced with local median values
4. **Temporal Smoothing**: Median-based smoothing of final MI values

## Comparison with Other Versions

| Feature | Adaptive | Stable | Dual Calibration |
|---------|----------|--------|------------------|
| Personalization | ✅ Dynamic | ❌ None | ✅ Calibrated |
| Session Consistency | ❌ Variable | ✅ High | ✅ High |
| Setup Time | Fast | Fast | Slow (50s calibration) |
| Sensitivity | High | Medium | Highest |
| Robustness | Medium | High | Highest |

## Literature Support

### Calibration-Based Approaches
- **Deiber et al. (2020)**: Individual alpha frequency calibration improves BCI performance
- **Klimesch (1999)**: Personal alpha bands vary significantly between individuals
- **Bazanova & Vernon (2014)**: Individual EEG patterns require personalized thresholds

### Mindfulness State Detection
- **Lomas et al. (2015)**: Dual-state calibration (rest vs. attention) for meditation classification
- **Brandmeyer & Delorme (2013)**: Personalized baselines improve mindfulness detection accuracy

### Robust Signal Processing
- **Mognon et al. (2011)**: Artifact rejection improves EEG-based classification
- **Nolan et al. (2010)**: Robust preprocessing essential for reliable BCI systems

## Advanced Usage

### Recalibration
```python
# Force recalibration for existing user
python realtime_mi_lsl_dual_calibration.py
# When prompted, answer 'n' to "Use existing calibration?"
```

### Batch Analysis
```python
# Load and analyze saved session data
import pandas as pd
data = pd.read_csv('logs/{user_id}_dual_calibration_session_{timestamp}.csv')
print(data.describe())
```

### Custom Thresholds
Advanced users can manually edit the JSON configuration files to adjust thresholds, though this is not recommended for typical use.

## Support

For technical issues or questions:
1. Check this guide first
2. Verify LSL stream setup
3. Confirm hardware connections
4. Review calibration procedure

The dual calibration system provides the most robust and personalized mindfulness detection available in this pipeline suite, suitable for research applications requiring high accuracy and individual adaptation.
