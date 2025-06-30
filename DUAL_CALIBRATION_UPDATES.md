# Dual Calibration System Updates

## Summary of Changes Made

### 1. Calibration Timing Update
- **Changed**: Both calibration phases now use 30 seconds duration
- **Before**: RELAXED (20s) + FOCUSED (30s)
- **After**: RELAXED (30s) + FOCUSED (30s)
- **Impact**: More consistent data collection for both baseline states

### 2. Enhanced MI Calculation - Three MI Values
The system now calculates and outputs all three MI values as in previous versions:

#### A. **Adaptive MI** (0-1 range, personalized)
- Uses dual calibration data to create user-specific thresholds
- Applies dynamic scaling based on session history
- Optimized for individual responsiveness

#### B. **Universal MI** (0-1 range, universal)
- Research-based universal calculation
- Uses comprehensive 9-feature mindfulness model
- Consistent across all users for comparison

#### C. **EMI - Emotional Mindfulness Index** (0-1 range, emotion-focused)
- Emphasizes emotion regulation components
- Weights frontal alpha asymmetry and arousal measures
- Blends with universal MI for emotional awareness

#### D. **Raw MI** (-5 to +5 range, backward compatibility)
- Scaled version of Universal MI
- Maintains compatibility with previous systems

### 3. Dynamic Scaling and Mapping
Added intelligent scaling based on previous session data:

#### **Session History Analysis**
- Loads up to 5 most recent sessions
- Analyzes dynamic range and responsiveness patterns
- Calculates trend-based scaling factors

#### **Dynamic Scaling Factors**
- **Low responsiveness** (dynamic range < 0.3): Increase sensitivity by 20%
- **High responsiveness** (dynamic range > 0.7): Decrease sensitivity by 10%
- **Normal responsiveness**: No scaling (1.0x)

#### **Benefits**
- More consistent MI responsiveness across sessions
- Adapts to individual user characteristics
- Improves long-term user experience

### 4. Enhanced LSL Stream Output
Updated stream descriptions for clarity:

```
- mindfulness_index (Adaptive MI: 0-1, personalized)
- raw_mindfulness_index (Raw MI: -5 to +5, universal)  
- emotional_mindfulness_index (EMI: 0-1, emotion-focused)
```

### 6. Enhanced User Experience
Updated real-time feedback shows:
- All three MI values: Adaptive, Universal, EMI
- Raw MI value for backward compatibility
- Dynamic scaling factor (when active)
- Comprehensive feature breakdown
- **Post-calibration pause with results summary**
- **User confirmation before starting real-time processing**

### 6. Enhanced Visualization
Updated final plot to show:
- Three separate subplots for each MI type
- Clear labeling and color coding
- Comprehensive comparison view

### 7. Improved User Experience
- **Post-calibration pause**: System shows calibration results and waits for user confirmation
- **Calibration quality assessment**: Displays whether calibration quality is Excellent/Good/Fair
- **Session learning transparency**: Shows how previous sessions influence current sensitivity
- **Real-time learning feedback**: Periodic display of dynamic scaling adjustments

### 8. Session Statistics
Enhanced session statistics include:
- All three MI values with means and standard deviations
- Dynamic scaling factor applied
- Number of previous sessions used for scaling
- Dynamic range analysis

## Technical Implementation

### Core Classes Updated

#### **AdaptiveMICalculator**
- Added `user_id` parameter for session history loading
- Implemented `load_previous_sessions()` method
- Added `get_dynamic_scaling_factor()` method
- Enhanced `calculate_adaptive_mi()` to return 3 values
- Added dedicated `calculate_emi()` method

#### **OnlineVisualizer**
- Added EMI history tracking
- Updated `update()` method to accept 3 MI values
- Enhanced `final_plot()` with 3-subplot layout

### Key Features

#### **Robust Data Processing**
- Peak suppression with median filtering
- Artifact rejection using robust outlier detection
- Comprehensive EEG channel validation

#### **Comprehensive Feature Set**
9 research-based mindfulness features:
1. `theta_fz` - Attention Regulation
2. `beta_fz` - Effortful Control  
3. `alpha_c3` - Left Body Awareness
4. `alpha_c4` - Right Body Awareness
5. `faa_c3c4` - Emotion Regulation (Frontal Alpha Asymmetry)
6. `alpha_pz` - DMN Suppression
7. `alpha_po` - Visual Detachment
8. `alpha_oz` - Relaxation
9. `eda_norm` - Arousal/Stress

## Usage Guidelines

### For New Users
1. System automatically runs dual calibration (30s + 30s)
2. Creates personalized adaptive thresholds
3. Saves calibration data for future sessions

### For Returning Users
1. System loads existing calibration automatically
2. Applies dynamic scaling based on session history
3. Option to recalibrate if needed

### Real-time Operation
1. All three MI values are calculated and streamed simultaneously
2. Dynamic scaling factor shows adaptation to user's historical responsiveness
3. Comprehensive feature breakdown available in real-time

## Files Modified
- `realtime_mi_lsl_dual_calibration.py` - Main implementation
- Created: `DUAL_CALIBRATION_UPDATES.md` - This documentation

## Backward Compatibility
- All previous MI output streams maintained
- Raw MI output preserved for legacy systems
- Existing calibration files compatible
- Previous session data can be used for dynamic scaling

## Next Steps
1. Test with real EEG/EDA hardware
2. Validate dynamic scaling effectiveness
3. Fine-tune EMI calculation weights
4. Add optional advanced scaling parameters
