"""
Test script to demonstrate the effect of MI normalization

This script shows how the new MI normalization affects the MI values
compared to the old calculation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eeg_mindfulness_index import calculate_mi, MI_WEIGHTS, THRESHOLDS

def normalize_mi_value(mi_raw):
    """Normalize a raw MI value to 0-1 range using sigmoid-like normalization."""
    import numpy as np
    return 1 / (1 + np.exp(-mi_raw + 1))

def test_mi_normalization():
    # Create test features with different ranges
    test_feature_sets = [
        # Normal range features (expected MI close to 0.5)
        {
            'theta_fz': 1.0,
            'alpha_po': 1.0,
            'faa': 0.0,
            'beta_frontal': 1.0,
            'eda_norm': 0.0
        },
        # High features (would produce high MI > 1 in old method)
        {
            'theta_fz': 10.0,
            'alpha_po': 12.0,
            'faa': 5.0,
            'beta_frontal': 0.5,
            'eda_norm': 0.2
        },
        # Very high features (would produce MI > 5 in old method)
        {
            'theta_fz': 20.0,
            'alpha_po': 25.0,
            'faa': 10.0,
            'beta_frontal': 1.0,
            'eda_norm': 0.5
        },
        # Extreme features (would produce MI > 10 in old method)
        {
            'theta_fz': 40.0,
            'alpha_po': 45.0,
            'faa': 20.0,
            'beta_frontal': 2.0,
            'eda_norm': 1.0
        }
    ]
    
    # Calculate MI with old method (manual calculation)
    def old_calculate_mi(features):
        mi = 0
        for feature, weight in MI_WEIGHTS.items():
            if feature in features:
                mi += weight * features[feature]
        return mi
    
    results = []
    for i, features in enumerate(test_feature_sets):
        # Calculate MI with old method
        mi_old = old_calculate_mi(features)
        
        # Calculate MI with new method
        mi_new = calculate_mi(features)
        
        # Get behavioral state with old method
        state_old = "Focused" if mi_old >= THRESHOLDS['focused'] else "Neutral" if mi_old >= THRESHOLDS['neutral'] else "Unfocused"
        
        # Get behavioral state with new method
        state_new = "Focused" if mi_new >= THRESHOLDS['focused'] else "Neutral" if mi_new >= THRESHOLDS['neutral'] else "Unfocused"
        
        # Store results
        results.append({
            'Set': f"Set {i+1}",
            'MI (Old)': mi_old,
            'MI (New)': mi_new,
            'State (Old)': state_old,
            'State (New)': state_new
        })
    
    # Convert to DataFrame and print
    df = pd.DataFrame(results)
    print("Effect of MI Normalization:")
    print(df)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(results))
    
    plt.bar(x - bar_width/2, df['MI (Old)'], bar_width, label='MI (Old)', color='red')
    plt.bar(x + bar_width/2, df['MI (New)'], bar_width, label='MI (New)', color='green')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=THRESHOLDS['focused'], color='black', linestyle='--', label=f"Focused Threshold ({THRESHOLDS['focused']})")
    plt.axhline(y=THRESHOLDS['neutral'], color='gray', linestyle='--', label=f"Neutral Threshold ({THRESHOLDS['neutral']})")
    plt.axhline(y=1.0, color='blue', linestyle=':', label="Max Normalized Value (1.0)")
    
    plt.xlabel('Feature Sets')
    plt.ylabel('MI Value')
    plt.title('Comparison of Original vs Normalized MI')
    plt.xticks(x, df['Set'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    output_dir = "results"
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "mi_normalization_comparison.png"))
    
    print(f"Plot saved to {os.path.join(output_dir, 'mi_normalization_comparison.png')}")
    
    # Plot a range of values to show the normalization curve
    raw_values = np.linspace(-2, 15, 100)
    normalized_values = [normalize_mi_value(x) for x in raw_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(raw_values, normalized_values, 'b-', linewidth=2)
    plt.xlabel('Raw MI Value')
    plt.ylabel('Normalized MI Value')
    plt.title('MI Normalization Function')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', label="Mid-point (0.5)")
    plt.axhline(y=THRESHOLDS['focused'], color='green', linestyle='--', label=f"Focused Threshold ({THRESHOLDS['focused']})")
    plt.axhline(y=THRESHOLDS['neutral'], color='orange', linestyle='--', label=f"Neutral Threshold ({THRESHOLDS['neutral']})")
    plt.axvline(x=1.0, color='red', linestyle='--', label="Input = 1.0")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "mi_normalization_function.png"))
    print(f"Normalization function plot saved to {os.path.join(output_dir, 'mi_normalization_function.png')}")
    
    # Also print what raw MI values map to important thresholds
    raw_for_neutral = -np.log(1/THRESHOLDS['neutral'] - 1) + 1
    raw_for_focused = -np.log(1/THRESHOLDS['focused'] - 1) + 1
    
    print(f"\nRaw MI Value for Neutral Threshold ({THRESHOLDS['neutral']}): {raw_for_neutral:.4f}")
    print(f"Raw MI Value for Focused Threshold ({THRESHOLDS['focused']}): {raw_for_focused:.4f}")

if __name__ == "__main__":
    test_mi_normalization()
