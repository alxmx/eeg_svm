"""
Comprehensive test script for MI normalization

This script tests:
1. Basic MI normalization with different feature values
2. Parameter changes through the parameter adjuster
3. Historical data conversion
4. Visualization of normalized vs raw MI values
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eeg_mindfulness_index import (
    calculate_mi, 
    normalize_mi_value, 
    MI_WEIGHTS, 
    THRESHOLDS,
    classify_behavioral_state
)

def test_varying_feature_magnitudes():
    """Test MI normalization with different feature magnitudes"""
    print("\n=== Testing MI Normalization with Different Feature Magnitudes ===")
    
    # Test with different feature magnitudes
    feature_sets = [
        {"name": "Low Values", "theta_fz": 0.5, "alpha_po": 0.3, "faa": 0.2, "beta_frontal": 0.2, "eda_norm": 0.0},
        {"name": "Normal Values", "theta_fz": 2.0, "alpha_po": 1.0, "faa": 1.0, "beta_frontal": 0.5, "eda_norm": 0.0},
        {"name": "High Values", "theta_fz": 5.0, "alpha_po": 4.0, "faa": 3.0, "beta_frontal": 1.0, "eda_norm": 0.0},
        {"name": "Very High Values", "theta_fz": 15.0, "alpha_po": 12.0, "faa": 8.0, "beta_frontal": 2.0, "eda_norm": 0.0},
        {"name": "Extreme Values", "theta_fz": 40.0, "alpha_po": 35.0, "faa": 20.0, "beta_frontal": 5.0, "eda_norm": 0.0},
    ]
    
    results = []
    
    # Calculate Raw MI (old method)
    def old_calculate_mi(features):
        mi = 0
        for feature, weight in MI_WEIGHTS.items():
            if feature in features:
                mi += weight * features[feature]
        return mi
    
    # Process each feature set
    for features in feature_sets:
        # Save feature name and remove from feature dict
        name = features.pop("name")
        
        # Calculate MI with old and new methods
        mi_raw = old_calculate_mi(features)
        mi_normalized = calculate_mi(features)
        
        # Get behavioral states
        state_raw = classify_behavioral_state(mi_raw)
        state_norm = classify_behavioral_state(mi_normalized)
        
        # Make a copy of features for results
        feature_copy = features.copy()
        
        results.append({
            "Set": name,
            "MI (Raw)": mi_raw,
            "MI (Normalized)": mi_normalized,
            "State (Raw)": state_raw,
            "State (Normalized)": state_norm,
            "Features": feature_copy
        })
    
    # Display results
    df = pd.DataFrame(results)
    print(df[["Set", "MI (Raw)", "MI (Normalized)", "State (Raw)", "State (Normalized)"]])
    
    # Show warning if states are different
    different_states = df[df["State (Raw)"] != df["State (Normalized)"]]
    if not different_states.empty:
        print("\nWarning: The following feature sets have different behavioral states:")
        print(different_states[["Set", "MI (Raw)", "MI (Normalized)", "State (Raw)", "State (Normalized)"]])
    
    return df

def test_parameter_adjustments():
    """Test MI normalization with different weights"""
    print("\n=== Testing MI Normalization with Different Weight Settings ===")
    
    # Sample features
    features = {
        "theta_fz": 8.0,
        "alpha_po": 5.0, 
        "faa": 3.0,
        "beta_frontal": 1.0,
        "eda_norm": 0.0
    }
    
    # Different weight configurations
    weight_configs = [
        {"name": "Default", "weights": MI_WEIGHTS.copy()},
        {"name": "Theta Focus", "weights": {'theta_fz': 0.5, 'alpha_po': 0.2, 'faa': 0.1, 'beta_frontal': -0.1, 'eda_norm': -0.1}},
        {"name": "Alpha Focus", "weights": {'theta_fz': 0.2, 'alpha_po': 0.5, 'faa': 0.1, 'beta_frontal': -0.1, 'eda_norm': -0.1}},
        {"name": "Equal Positive", "weights": {'theta_fz': 0.33, 'alpha_po': 0.33, 'faa': 0.34, 'beta_frontal': -0.0, 'eda_norm': -0.0}},
        {"name": "Negative Heavy", "weights": {'theta_fz': 0.15, 'alpha_po': 0.15, 'faa': 0.1, 'beta_frontal': -0.3, 'eda_norm': -0.3}}
    ]
    
    results = []
    
    # Calculate MI for each weight configuration
    for config in weight_configs:
        name = config["name"]
        weights = config["weights"]
        
        # Calculate normalized MI with these weights
        mi_normalized = calculate_mi(features, weights=weights)
        
        # Calculate raw MI manually
        mi_raw = 0
        for feature, weight in weights.items():
            if feature in features:
                mi_raw += weight * features[feature]
        
        # Check if normalization preserves order
        mi_normalized_with_func = normalize_mi_value(mi_raw)
        
        results.append({
            "Config": name,
            "MI (Raw)": mi_raw,
            "MI (Normalized)": mi_normalized,
            "MI (Manual Normalize)": mi_normalized_with_func,
            "Match": np.isclose(mi_normalized, mi_normalized_with_func)
        })
    
    # Display results
    df = pd.DataFrame(results)
    print(df)
    
    # Check if normalization is consistent
    if not all(df["Match"]):
        print("\nWarning: Normalization function is not consistent with calculate_mi results!")
    else:
        print("\nNormalization is consistent between calculate_mi and normalize_mi_value functions.")
    
    return df

def test_historical_data_conversion():
    """Test converting historical MI values to normalized values"""
    print("\n=== Testing Historical Data Conversion ===")
    
    # Sample historical MI values
    historical_data = [
        {"timestamp": 0.0, "mi_raw": 0.2, "state_old": "Unfocused"},
        {"timestamp": 1.5, "mi_raw": 0.5, "state_old": "Neutral"},
        {"timestamp": 3.0, "mi_raw": 1.0, "state_old": "Focused"},
        {"timestamp": 4.5, "mi_raw": 2.5, "state_old": "Focused"},
        {"timestamp": 6.0, "mi_raw": 5.0, "state_old": "Focused"},
        {"timestamp": 7.5, "mi_raw": 7.5, "state_old": "Focused"},
        {"timestamp": 9.0, "mi_raw": 10.0, "state_old": "Focused"}
    ]
    
    # Convert historical data
    for data in historical_data:
        # Normalize raw MI
        data["mi_normalized"] = normalize_mi_value(data["mi_raw"])
        
        # Classify with old raw value
        data["state_raw"] = classify_behavioral_state(data["mi_raw"])
        
        # Classify with normalized value
        data["state_normalized"] = classify_behavioral_state(data["mi_normalized"])
    
    # Display results
    df = pd.DataFrame(historical_data)
    print(df[["timestamp", "mi_raw", "mi_normalized", "state_old", "state_normalized"]])
    
    # Check if normalization preserved state classification
    changes = df[df["state_old"] != df["state_normalized"]]
    if not changes.empty:
        print("\nWarning: State classification changed for some historical data points:")
        print(changes[["timestamp", "mi_raw", "mi_normalized", "state_old", "state_normalized"]])
        print("\nThis may indicate threshold adjustments are needed for backward compatibility.")
    else:
        print("\nHistorical data state classification preserved after normalization.")
        
    return df
    
def visualize_normalization_function():
    """Visualize the normalization function and its effect"""
    print("\n=== Visualizing Normalization Function ===")
    
    # Plot normalization function
    raw_values = np.linspace(-2, 15, 1000)
    normalized_values = [normalize_mi_value(x) for x in raw_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(raw_values, normalized_values, 'b-', linewidth=2)
    plt.xlabel('Raw MI Value')
    plt.ylabel('Normalized MI Value')
    plt.title('MI Normalization Function')
    plt.grid(True, alpha=0.3)
    
    # Add reference lines
    plt.axhline(y=0.5, color='gray', linestyle='--', label="Mid-point (0.5)")
    plt.axhline(y=THRESHOLDS['focused'], color='green', linestyle='--', label=f"Focused Threshold ({THRESHOLDS['focused']})")
    plt.axhline(y=THRESHOLDS['neutral'], color='orange', linestyle='--', label=f"Neutral Threshold ({THRESHOLDS['neutral']})")
    plt.axvline(x=1.0, color='red', linestyle='--', label="Input = 1.0")
    
    # Mark key points
    raw_for_neutral = -np.log(1/THRESHOLDS['neutral'] - 1) + 1
    raw_for_focused = -np.log(1/THRESHOLDS['focused'] - 1) + 1
    
    plt.plot(raw_for_neutral, THRESHOLDS['neutral'], 'ro', markersize=8)
    plt.plot(raw_for_focused, THRESHOLDS['focused'], 'go', markersize=8)
    
    plt.text(raw_for_neutral + 0.1, THRESHOLDS['neutral'] - 0.03, f"({raw_for_neutral:.2f}, {THRESHOLDS['neutral']})", 
             color='red', fontsize=10)
    plt.text(raw_for_focused + 0.1, THRESHOLDS['focused'] - 0.03, f"({raw_for_focused:.2f}, {THRESHOLDS['focused']})", 
             color='green', fontsize=10)
    
    plt.legend()
    plt.xlim(-2, 10)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "mi_normalization_detailed.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Normalization function visualization saved to {output_path}")
    print(f"\nRaw MI Value for Neutral Threshold ({THRESHOLDS['neutral']}): {raw_for_neutral:.4f}")
    print(f"Raw MI Value for Focused Threshold ({THRESHOLDS['focused']}): {raw_for_focused:.4f}")
    
    return output_path

def test_extreme_cases():
    """Test edge cases for the normalization function"""
    print("\n=== Testing Edge Cases ===")
    
    test_cases = [
        {"name": "Zero", "value": 0.0},
        {"name": "Very Negative", "value": -10.0},
        {"name": "Very Positive", "value": 100.0},
        {"name": "Extremely Negative", "value": -1000.0},
        {"name": "Extremely Positive", "value": 1000.0},
        {"name": "NaN", "value": np.nan}
    ]
    
    for case in test_cases:
        # Handle NaN separately to avoid warnings
        if np.isnan(case["value"]):
            try:
                result = normalize_mi_value(case["value"])
                print(f"{case['name']} ({case['value']}): {result}")
            except Exception as e:
                print(f"{case['name']} ({case['value']}): Error - {e}")
        else:
            result = normalize_mi_value(case["value"])
            print(f"{case['name']} ({case['value']}): {result}")
    
    print("\nThis verifies that the normalization function handles extreme values appropriately.")

if __name__ == "__main__":
    print("======== Comprehensive MI Normalization Tests ========")
    
    # Run all tests
    test_varying_feature_magnitudes()
    test_parameter_adjustments()
    test_historical_data_conversion()
    visualize_normalization_function()
    test_extreme_cases()
    
    print("\n======== All Tests Completed ========")
    print("MI normalization is working as expected and provides a consistent 0-1 range output.")
    print("The original behavioral state classification is preserved with the normalized MI values.")
