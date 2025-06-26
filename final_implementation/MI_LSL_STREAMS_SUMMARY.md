# MI Values Transmitted via LSL - Summary

## Three LSL Streams are Created:

### 1. **MindfulnessIndex** Stream
- **Stream Name**: `MindfulnessIndex`
- **Type**: `MI`
- **Value Transmitted**: `mi_pred` 
- **Description**: Enhanced SVR prediction (primary MI output)
- **Range**: 0.05 - 0.95 (clipped)
- **Calculation**: 
  - Uses user-specific SVR model if available
  - Enhanced with `enhance_svr_prediction()` function
  - Falls back to universal MI if SVR is static/saturated
  - Auto-switches to universal MI if saturation detected

### 2. **RawMindfulnessIndex** Stream  
- **Stream Name**: `RawMindfulnessIndex`
- **Type**: `RawMI`
- **Value Transmitted**: `raw_mi_remapped`
- **Description**: Raw MI value remapped to 0-1 range
- **Range**: 0.0 - 1.0 (remapped from -5 to +5)
- **Calculation**:
  - Uses `calculate_raw_mi_universal()` → gives -5 to +5 range
  - Then `remap_raw_mi()` → converts to 0-1 range
  - Provides more dynamic/sensitive range than standard MI

### 3. **EmotionalMindfulnessIndex** Stream
- **Stream Name**: `EmotionalMindfulnessIndex` 
- **Type**: `EMI`
- **Value Transmitted**: `emi_value`
- **Description**: Emotional Mindfulness Index
- **Range**: 0.05 - 0.95 (clipped)
- **Calculation**:
  - Uses `calculate_emi_universal()` 
  - Emphasizes FAA (Frontal Alpha Asymmetry) and EDA features
  - Weights: [0.15, 0.15, 0.4, -0.05, -0.25] for emotional states

## Stream Configuration:
- **Sample Rate**: 10 Hz (but actual data is sent at ~1 Hz)
- **Channels**: 1 channel per stream
- **Data Type**: float32
- **Source IDs**: 'mi_12345', 'raw_mi_12345', 'emi_12345'

## Real-time Console Output:
The script prints all three values in real-time:
```
[REAL-TIME] MI (SVR): 0.456 | MI (Universal): 0.423 | Raw MI: -1.234 (remapped: 0.377) | EMI: 0.512
```

## Key Features:
1. **Dynamic Output**: Enhanced SVR prevents static values
2. **Fallback Mechanisms**: Universal MI if SVR fails/saturates  
3. **Multiple Perspectives**: Standard, Raw (sensitive), Emotional
4. **Real-time Monitoring**: Values updated every second
5. **User-specific**: Uses calibrated models when available
