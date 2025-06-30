# Dual Calibration MI Pipeline - Technical Specification

## System Architecture

### Overview
The Dual Calibration Mindfulness Index (MI) Pipeline is a physiologically-aware, personalized mindfulness detection system that establishes user-specific thresholds through a two-phase calibration protocol. Unlike population-based approaches, this system adapts to individual EEG/EDA patterns to provide maximally sensitive and consistent mindfulness measurements.

### Core Components

#### 1. RobustDataProcessor
**Purpose**: Real-time artifact suppression and signal conditioning
**Key Features**:
- Median filtering (5-point kernel) for peak suppression
- Median Absolute Deviation (MAD) outlier detection
- Robust baseline statistics computation
- Adaptive window processing with padding/truncation

**Technical Details**:
```python
# Outlier detection threshold
outlier_threshold = 3.0 * MAD

# Median filter implementation
filtered[i] = median(data[i-2:i+3])

# Outlier replacement
outliers = |data - median| > threshold
data[outliers] = median(data[~outliers])
```

#### 2. DualCalibrationSystem
**Purpose**: Two-phase baseline establishment and threshold computation
**Phases**:
- **RELAXED** (20s, 250 Hz = 5000 samples): Eyes closed, natural mind-wandering
- **FOCUSED** (30s, 250 Hz = 7500 samples): Eyes open, attention to breathing

**Mathematical Foundation**:
```python
# Feature extraction every 1 second (250 samples)
n_windows_relaxed = 20
n_windows_focused = 30

# Adaptive threshold calculation
low_threshold = mean(relaxed_MI)
high_threshold = mean(focused_MI)
dynamic_range = high_threshold - low_threshold

# Personalized MI mapping
adaptive_MI = (universal_MI - low_threshold) / dynamic_range
adaptive_MI = clip(adaptive_MI, 0, 1)
```

#### 3. AdaptiveMICalculator
**Purpose**: Real-time MI computation using personalized thresholds
**Algorithm**:
1. Calculate universal MI using population norms
2. Apply user-specific threshold mapping
3. Temporal smoothing via median filter
4. Output normalized MI (0-1 range)

### Feature Extraction Pipeline

#### EEG Features (5 total)
1. **Theta Frontal (theta_fz)**: Bandpower 4-8 Hz at Fz
2. **Alpha Posterior (alpha_po)**: Average bandpower 8-13 Hz at PO7/PO8
3. **Frontal Alpha Asymmetry (faa)**: ln(F4_alpha) - ln(F3_alpha)
4. **Beta Frontal (beta_frontal)**: Bandpower 13-30 Hz at Fz

#### EDA Features (1 total)
5. **Normalized EDA (eda_norm)**: Robust quantile normalization (0-10 scale)

#### Bandpower Calculation
```python
# Welch's method with adaptive window
f, psd = welch(data, sf=250, nperseg=min(len(data), 250))
idx_band = (f >= low_freq) & (f <= high_freq)
bandpower = trapz(psd[idx_band], f[idx_band])
```

### Calibration Protocol

#### Phase 1: RELAXED Baseline
**Duration**: 20 seconds
**Instructions**:
- Eyes closed
- Natural breathing
- Mind-wandering allowed
- No meditation or concentration

**Physiological Target**: Default Mode Network activation, high alpha power, low attention-related signatures

#### Phase 2: FOCUSED Baseline  
**Duration**: 30 seconds
**Instructions**:
- Eyes open, fixed gaze
- Breath counting (1-10, repeat)
- Sustained attention
- Return to counting if mind wanders

**Physiological Target**: Executive attention networks, reduced alpha, increased frontal theta

### Quality Metrics

#### Calibration Quality Assessment
1. **Feature Separability**: Cohen's d between relaxed and focused states
2. **Statistical Significance**: t-test p-values for state differences
3. **Overlap Coefficient**: Histogram overlap between states (lower is better)
4. **Dynamic Range**: |focused_MI - relaxed_MI| (>0.3 preferred)
5. **Stability**: Coefficient of variation within each phase (<0.5 preferred)

#### Overall Quality Score
```python
separability_score = abs(cohens_d) * (1 - overlap) * significance_factor
range_quality = min(dynamic_range / 0.3, 1.0)
stability_penalty = max(0, 1 - avg_cv / 0.5)

overall_quality = (
    mean(separability_scores) * 0.5 +
    range_quality * 0.3 +
    stability_penalty * 0.2
)
```

### Adaptive Threshold Computation

#### Universal MI Calculation
```python
# Feature normalization to 0-10 scale
normalized_features = []
for i, (feature, (q5, q95)) in enumerate(feature_ranges.items()):
    norm_val = 10 * (features[i] - q5) / (q95 - q5)
    normalized_features.append(clip(norm_val, 0, 10))

# Weighted combination
weights = [0.3, 0.3, 0.2, -0.1, -0.2]  # theta, alpha, faa, beta, eda
weighted_sum = dot(normalized_features, weights)

# Scale to 0.1-0.9 range
universal_MI = 0.1 + 0.8 * (weighted_sum / 10.0)
```

#### Personalized Mapping
```python
# Apply user-specific calibration
if dynamic_range > 0.1:  # Sufficient range
    relative_position = (universal_MI - low_threshold) / dynamic_range
    adaptive_MI = clip(relative_position, 0, 1)
else:
    adaptive_MI = universal_MI  # Fallback to universal
```

### Peak Suppression Algorithm

#### Multi-Stage Artifact Rejection
1. **Pre-filtering**: 5-point median filter on raw signals
2. **Outlier Detection**: MAD-based threshold (3σ equivalent)
3. **Replacement Strategy**: Local median substitution
4. **Temporal Smoothing**: 5-sample median filter on final MI

#### Mathematical Implementation
```python
# Median filtering
def median_filter_1d(data, size=5):
    half_size = size // 2
    for i in range(half_size, len(data) - half_size):
        data[i] = median(data[i-half_size:i+half_size+1])

# MAD outlier detection
median_val = median(data)
mad = median(abs(data - median_val))
threshold = 3.0 * mad
outliers = abs(data - median_val) > threshold

# Robust replacement
data[outliers] = median(data[~outliers])
```

### Data Persistence and Configuration

#### User Configuration Storage
```json
{
  "user_id": "string",
  "calibration_time": "ISO datetime",
  "baseline_csv": "path/to/features.csv",
  "adaptive_thresholds": {
    "relaxed_baseline": {
      "mi_mean": "float",
      "mi_std": "float",
      "mi_range": "[min, max]",
      "features": "statistics_dict"
    },
    "focused_baseline": {
      "mi_mean": "float", 
      "mi_std": "float",
      "mi_range": "[min, max]",
      "features": "statistics_dict"
    },
    "adaptive_mapping": {
      "low_threshold": "float",
      "high_threshold": "float", 
      "dynamic_range": "float",
      "calibration_time": "ISO datetime"
    }
  },
  "relaxed_samples": "int",
  "focused_samples": "int"
}
```

#### Session Data Schema
```csv
timestamp,adaptive_mi,universal_mi,raw_mi,emi,theta_fz,alpha_po,faa,beta_frontal,eda_norm
```

### LSL Stream Specification

#### Output Streams
1. **mindfulness_index**: Personalized adaptive MI (0-1 range, 1 Hz)
2. **raw_mindfulness_index**: Raw MI (-5 to +5 range, 1 Hz)  
3. **emotional_mindfulness_index**: EMI variant (0-1 range, 1 Hz)

#### Stream Properties
```python
# Standard MI stream
StreamInfo(name='mindfulness_index', 
          type='MI', 
          channel_count=1, 
          nominal_srate=1, 
          channel_format='float32',
          source_id='mi_001')
```

### Performance Specifications

#### Temporal Characteristics
- **Sampling Rate**: 250 Hz (EEG/EDA)
- **Analysis Window**: 1 second (250 samples)
- **Output Rate**: 1 Hz
- **Processing Latency**: <50 ms
- **Memory Usage**: <100 MB baseline

#### Accuracy Metrics
- **Calibration SNR**: >3.0 dB feature separation
- **Temporal Stability**: CV <0.3 within sessions
- **Cross-session Consistency**: r >0.7 for MI means
- **Dynamic Range**: >0.3 MI units preferred

### Validation and Quality Control

#### Real-time Quality Monitoring
```python
# Signal quality indicators
electrode_impedance_check()  # <10 kΩ preferred
signal_to_noise_ratio()      # >20 dB
artifact_percentage()        # <15% outliers
```

#### Post-session Analysis
```python
# Effectiveness metrics
range_enhancement = adaptive_range / universal_range  # >1.5 good
scale_utilization = full_scale_usage()               # >0.8 good
responsiveness_ratio = adaptive_response / universal_response
personalization_score = weighted_effectiveness()     # >0.7 excellent
```

### Error Handling and Robustness

#### Calibration Failure Cases
1. **Insufficient data**: <15 windows per phase
2. **Low signal quality**: >30% outliers detected
3. **No state difference**: Cohen's d <0.3
4. **Technical failure**: Stream disconnection

#### Fallback Strategies
```python
if calibration_failed:
    if previous_calibration_exists:
        load_previous_calibration()
    else:
        use_population_defaults()
        log_warning("Using non-personalized thresholds")
```

#### Runtime Error Recovery
```python
# Stream disconnection handling
try:
    sample = inlet.pull_sample(timeout=1.0)
except:
    use_last_valid_sample()
    log_warning("Stream interruption")

# Outlier saturation protection
if outlier_percentage > 50:
    increase_filter_strength()
    log_warning("High artifact rate detected")
```

### Computational Complexity

#### Real-time Processing
- **Feature Extraction**: O(N log N) per window (FFT-based)
- **Outlier Detection**: O(N) per window
- **MI Calculation**: O(1) (linear combination)
- **Memory**: O(N) for sliding windows

#### Calibration Phase
- **Complexity**: O(M × N log N) where M=calibration_windows, N=window_size
- **Storage**: O(M × F) where F=feature_count
- **Threshold Computation**: O(M) statistics calculation

### Integration Interfaces

#### LSL Compatibility
- Compatible with OpenViBE, BCILAB, MNE-Python
- Standard LSL stream discovery protocol
- Automatic stream type resolution

#### File I/O Formats
- **Configuration**: JSON (human-readable)
- **Session Data**: CSV (analysis-friendly)
- **Baseline Features**: CSV with phase labels
- **Visualizations**: PNG (publication-quality)

### Deployment Considerations

#### Hardware Requirements
- **CPU**: Multi-core recommended for real-time processing
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 1GB for data/models, 10GB for long-term sessions
- **EEG**: 8+ channels, 250+ Hz sampling
- **EDA**: 2+ channels, 250+ Hz sampling

#### Software Dependencies
```python
# Core dependencies
numpy >= 1.20.0
scipy >= 1.7.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
pylsl >= 1.15.0

# Optional enhancements
seaborn >= 0.11.0  # Enhanced visualizations
joblib >= 1.1.0    # Model persistence
```

### Validation Protocol

#### Clinical Validation Steps
1. **Technical Validation**: Signal processing accuracy
2. **Physiological Validation**: Known-state discrimination
3. **Behavioral Validation**: Meditation task correlation
4. **Longitudinal Validation**: Multi-session consistency

#### Recommended Test Protocol
```python
# Phase 1: Technical validation
test_signal_processing_accuracy()
test_artifact_suppression()
test_real_time_performance()

# Phase 2: Physiological validation  
test_eyes_open_vs_closed()
test_relaxation_vs_attention()
test_meditation_vs_rest()

# Phase 3: User acceptance
test_calibration_usability()
test_real_time_feedback()
test_session_consistency()
```

### Research Applications

#### Suitable Use Cases
- **Meditation Training**: Real-time feedback systems
- **Attention Research**: Sustained attention measurement
- **Clinical Studies**: Mindfulness intervention assessment
- **BCI Applications**: Attention-based control interfaces

#### Limitations and Considerations
- **Individual Differences**: Some users may show limited calibration quality
- **State Dependency**: Calibration quality depends on user's ability to achieve target states
- **Environmental Factors**: Electrical noise, movement artifacts
- **Learning Effects**: User adaptation to calibration procedure over time

### Future Enhancements

#### Planned Improvements
1. **Multi-modal Integration**: ECG, respiration, eye tracking
2. **Adaptive Recalibration**: Automatic threshold updates
3. **Machine Learning**: Deep learning feature extraction
4. **Real-time Feedback**: Visual/auditory biofeedback integration

#### Research Directions
- **Cross-cultural Validation**: Multi-population studies
- **Longitudinal Modeling**: Long-term personalization
- **Transfer Learning**: Cross-user generalization
- **Multi-state Calibration**: Extended state repertoires

This technical specification provides comprehensive implementation details for the dual calibration MI pipeline, enabling reproducible research and clinical applications requiring high-accuracy, personalized mindfulness detection.
