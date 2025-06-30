# Stable MI Pipeline - Quick Start Guide

## Overview

The **stable MI pipeline** (`realtime_mi_lsl_stable.py`) has been created to address consistency and robustness issues in real-time Mindfulness Index calculations. This version removes all adaptive/dynamic mechanisms that can cause session-to-session variability.

## Key Differences: Adaptive vs Stable Pipeline

| Feature | Original (Adaptive) | Stable Version |
|---------|-------------------|----------------|
| **Personal Baselines** | ✅ User-specific adjustments | ❌ **REMOVED** - Uses population baselines only |
| **Dynamic Scaling** | ✅ Real-time input scaling | ❌ **REMOVED** - Fixed scaling factors |
| **Saturation Detection** | ✅ Automatic fallback mechanisms | ❌ **REMOVED** - Simple range clipping |
| **Flexible Normalization** | ✅ User/population stat blending | ❌ **REMOVED** - Fixed population ranges |
| **Anti-Static Logic** | ✅ Prevents frozen outputs | ❌ **REMOVED** - Natural variation only |
| **Session Consistency** | ⚠️ Varies between sessions | ✅ **CONSISTENT** across sessions |

## Files Available

### 🎯 **Main Pipeline**
- **Original (Adaptive)**: `final_implementation/realtime_mi_lsl.py`
- **Stable Version**: `final_implementation/realtime_mi_lsl_stable.py` ⭐

### 🔧 **XDF Analysis Tools**
- **XDF Reader**: `xdf_reader.py`
- **Instructions**: `XDF_READER_INSTRUCTIONS.md`

### 📋 **Documentation**
- **Version Comparison**: `final_implementation/MI_LSL_VERSION_COMPARISON.md`
- **Stream Summary**: `final_implementation/MI_LSL_STREAMS_SUMMARY.md`
- **EDA Improvements**: `final_implementation/EDA_IMPROVEMENT_RECOMMENDATIONS.md`

## Quick Start

### 1. Run the Stable Pipeline
```cmd
cd final_implementation
python realtime_mi_lsl_stable.py
```

### 2. Connect to LSL Streams
The pipeline will automatically detect and connect to:
- **EEG Stream** (type: 'EEG') - requires minimum 8 channels
- **EDA Stream** (type: 'EDA') - requires minimum 2 channels  
- **Unity Markers** (optional, type: 'UnityMarkers')

### 3. LSL Output Streams
The stable pipeline creates three fixed-rate output streams:
- `mindfulness_index` - Standard MI (0.1-0.9 range)
- `raw_mindfulness_index` - Raw MI (-5 to +5, remapped to 0-1 for LSL)
- `emotional_mindfulness_index` - EMI (0.05-0.95 range)

All streams output at **10 Hz** with **fixed parameters**.

## Stable Version Features

### ✅ **Fixed Normalization Ranges**
```python
FIXED_NORMALIZATION_RANGES = {
    'theta_fz': (2, 60),      # Population-based, never changes
    'alpha_po': (1, 30),      # Population-based, never changes  
    'faa': (-2.5, 2.5),       # Population-based, never changes
    'beta_frontal': (2, 35),  # Population-based, never changes
    'eda_norm': (2, 12)       # Population-based, never changes
}
```

### ✅ **Fixed Scaling Factors**
```python
FIXED_EEG_SCALE = 1.0  # No real-time adaptation
FIXED_EDA_SCALE = 1.0  # No real-time adaptation
```

### ✅ **Consistent MI Calculation**
```python
# Fixed weights (never change)
weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])

# Fixed mapping to [0.1, 0.9] range
mi = 0.1 + 0.8 * raw_score
```

## Expected Results

### **Session-to-Session Consistency**
- Same input data → Same MI output
- No drift due to personal baselines
- No sudden jumps from saturation detection
- Predictable behavior for interactive applications

### **Trade-offs**
- ⚠️ **Less personalized** - No user-specific adjustments
- ⚠️ **May not handle extreme outliers** as gracefully
- ✅ **More stable** for interactive applications
- ✅ **Reproducible** results for research/validation

## Validation Tools

### **XDF File Analysis**
To analyze recorded LSL sessions:
```cmd
python xdf_reader.py your_recording.xdf
```

This provides:
- Stream information and channel counts
- Data quality assessment
- Sample rate validation
- Missing data detection

### **Session Comparison**
Both versions log detailed session data to:
- `final_implementation/logs/` - CSV files with all metrics
- `final_implementation/visualizations/` - Time-series plots

## Recommendations

### **For Interactive Applications**
✅ **Use the Stable Version** - Provides consistent, predictable MI values suitable for real-time feedback systems.

### **For Research/Clinical Use**
⚠️ **Consider the Original** - May provide better personalization and adaptation to individual differences.

### **For Validation/Testing**
✅ **Use Both** - Compare outputs to understand the impact of adaptive mechanisms on your specific use case.

## Next Steps

1. **Test both versions** with your specific EEG/EDA setup
2. **Record XDF files** during testing for offline analysis
3. **Compare MI outputs** between adaptive and stable versions
4. **Adjust fixed parameters** in stable version if needed based on your population
5. **Document any required modifications** for your specific hardware/application

---

**Contact**: This implementation prioritizes **stability and consistency** over **personalization and adaptation**. Choose the version that best fits your application requirements.
