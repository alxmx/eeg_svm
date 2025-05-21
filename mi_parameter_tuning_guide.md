# MI Parameter Tuning Guide

This guide explains how to fine-tune the Mindfulness Index (MI) parameters to optimize discrimination between different cognitive states using the EEG Mindfulness Index pipeline.

## Understanding the MI Formula

The Mindfulness Index is calculated using a weighted combination of EEG features, followed by normalization:

```
MI_raw = (w1 * Theta_Fz) + (w2 * Alpha_PO) + (w3 * FAA) - (w4 * Beta_Frontal) - (w5 * EDA_norm)
MI = 1 / (1 + exp(-MI_raw + 1))  # Normalized to 0-1 range
```

Where:
- **Theta_Fz**: 4-7 Hz power at frontal electrode Fz (higher in focused meditation)
- **Alpha_PO**: 8-12 Hz power at posterior electrodes PO7/PO8 (higher in relaxed focus)
- **FAA**: Frontal Alpha Asymmetry (right-left alpha power, positive in approach states)
- **Beta_Frontal**: 13-30 Hz power at frontal sites (higher in arousal/alertness)
- **EDA_norm**: Normalized electrodermal activity (higher in arousal/stress)

## Understanding Parameter Effects

### Feature Weights

The feature weights determine how strongly each EEG feature influences the final MI score:

| Feature | Default Weight | Effect of Increasing | Effect of Decreasing |
|---------|---------------|----------------------|----------------------|
| `theta_fz` | +0.25 | Increases MI when theta power is high (meditative states) | Decreases influence of frontal theta |
| `alpha_po` | +0.25 | Increases MI when posterior alpha is high (relaxed focus) | Decreases influence of posterior alpha |
| `faa` | +0.20 | Increases MI when right-side alpha is higher (approach/positive states) | Decreases influence of asymmetry |
| `beta_frontal` | -0.15 | Decreases MI when beta power is high (arousal/alertness) | Reduces the negative impact of beta activity |
| `eda_norm` | -0.15 | Decreases MI when EDA is high (arousal/stress) | Reduces the negative impact of EDA |

### Normalization Effect

After computing the weighted sum, the raw MI value is passed through a sigmoid-like normalization function that ensures the MI score falls between 0 and 1. This normalization has the following properties:

- Raw MI = 0.0 → Normalized MI ≈ 0.27
- Raw MI = 0.5 → Normalized MI ≈ 0.38
- Raw MI = 1.0 → Normalized MI = 0.50
- Raw MI = 2.0 → Normalized MI ≈ 0.73
- Raw MI = 3.0 → Normalized MI ≈ 0.88
- Raw MI = 4.0 → Normalized MI ≈ 0.95

### Threshold Values

The thresholds determine how MI scores are mapped to cognitive states:

- **Focused**: MI ≥ 0.5 (raw MI ≥ 1.0)
- **Neutral**: 0.37 ≤ MI < 0.5 (raw MI between 0.47 and 1.0)
- **Unfocused**: MI < 0.37 (raw MI < 0.47)

## Tuning Recommendations

### When to Adjust Weights

Consider adjusting weights when:

1. You observe false positives/negatives in state classification
2. You want to emphasize certain features over others
3. You have domain knowledge about specific EEG correlates for your use case
4. You see correlation between specific features and desired cognitive states

### Tuning Process

1. **Start with Default Weights**: Begin with the default weights and observe the baseline performance.

2. **Identify Key Features**: Determine which features correlate most strongly with the desired cognitive states in your data.

3. **Gradual Adjustments**: Make small adjustments (±0.05) to weights and observe the effects.

4. **Maintain Balance**: Ensure the sum of positive weights and absolute sum of negative weights remain similar to prevent skewing.

5. **Validate Results**: Test your adjusted weights across multiple recordings and subjects.

### Example Tuning Scenarios

#### For Meditation Focus Detection

```python
# Emphasize theta and alpha, reduce beta influence
MI_WEIGHTS = {
    'theta_fz': 0.30,    # Increased from 0.25
    'alpha_po': 0.30,    # Increased from 0.25
    'faa': 0.20,         # Unchanged
    'beta_frontal': -0.10, # Reduced from -0.15
    'eda_norm': -0.10    # Reduced from -0.15
}
```

#### For Stress/Relaxation Discrimination

```python
# Emphasize beta and EDA (stress indicators)
MI_WEIGHTS = {
    'theta_fz': 0.20,    # Reduced from 0.25
    'alpha_po': 0.20,    # Reduced from 0.25
    'faa': 0.10,         # Reduced from 0.20
    'beta_frontal': -0.25, # Increased from -0.15
    'eda_norm': -0.25    # Increased from -0.15
}
```

## Using the Parameter Adjuster

The `mindfulness_parameter_adjuster.py` script provides a GUI for interactively adjusting weights and thresholds:

1. Run the script: `python mindfulness_parameter_adjuster.py`
2. Load an EEG file for analysis
3. Adjust the sliders for feature weights and thresholds
4. Click "Apply Changes" to see the effects in real-time

This allows you to immediately visualize how parameter changes affect MI values and state classification.

## Best Practices

1. **Document Changes**: Keep records of weight changes and their effects for future reference.

2. **Use Multiple Files**: Test your tuned parameters on several recordings to ensure robustness.

3. **Consider Individual Differences**: Different individuals may require different optimal weights.

4. **Iterative Approach**: Make incremental changes and validate each step.

5. **Maintain Interpretability**: Avoid extreme weights that make the index difficult to interpret.

## Advanced Customization

For more advanced customization, consider:

- Creating subject-specific parameter profiles
- Developing context-specific weight sets (e.g., meditation vs. work focus)
- Implementing an automated calibration procedure at the start of new recordings
- Comparing feature importance using visualization tools to guide weight selection

Remember that the goal of tuning is to optimize the detection of meaningful cognitive/attentional states while maintaining a scientifically sound and interpretable index.
