#!/usr/bin/env python3
"""
Test script to validate SVR MI enhancement functions work correctly.
This script simulates the static SVR output problem and tests our enhancement pipeline.
"""

import numpy as np
import time
import sys
import os

# Add the current directory to the path to import from realtime_mi_lsl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhancement functions
from realtime_mi_lsl import (
    enhance_svr_prediction, 
    enhance_static_prediction,
    calculate_mi_universal,
    FEATURE_ORDER
)

def test_static_svr_enhancement():
    """Test the enhancement of static SVR predictions."""
    
    print("=" * 60)
    print("Testing SVR MI Enhancement Pipeline")
    print("=" * 60)
    
    # Create test data that mimics the problematic session
    # Using data from 014_alex_test session that shows static SVR output
    test_features = [
        [15.055045552274514, 11.272985363049578, 0.7234814214527019, 16.394104512816075, 12.133246644337973],
        [8.98581585828131, 11.020747833756584, -0.22070237820376315, 5.544038613320335, 12.526752026875814],
        [5.594112650316004, 8.640564496576873, 0.5487699256731392, 3.845561762490709, 13.947867952982586],
        [4.412330176358929, 3.7639398207386745, -1.2453898601457336, 10.325958499649008, 15.0],
        [43.007160962316604, 10.084079198235248, 0.46383123228919665, 3.785774445017105, 15.0],
        [57.89627965175828, 8.545794022687597, -0.25849611659192906, 13.313592420009842, 15.0],
    ]
    
    # Simulate the static SVR prediction problem
    static_svr_value = 0.40930194734359193
    
    # Create mock baseline stats
    baseline_stats = {
        'means': {
            'theta_fz': 17.0,
            'alpha_po': 9.3,
            'faa': 0.15,
            'beta_frontal': 12.5,
            'eda_norm': 10.5
        },
        'stds': {
            'theta_fz': 14.9,
            'alpha_po': 4.8,
            'faa': 0.7,
            'beta_frontal': 6.3,
            'eda_norm': 3.6
        },
        'mi_baseline': {
            'mean': 0.45,
            'std': 0.12
        }
    }
    
    print(f"Testing with static SVR value: {static_svr_value}")
    print(f"Testing {len(test_features)} feature samples")
    print()
    
    enhanced_values = []
    universal_values = []
    
    for i, features in enumerate(test_features):
        # Simulate scaled features (dummy values)
        scaled_features = np.array(features) / 10.0  # Simple scaling for test
        
        print(f"Sample {i+1}:")
        print(f"  Raw features: {dict(zip(FEATURE_ORDER, features))}")
        
        # Test the enhancement function
        enhanced_mi = enhance_svr_prediction(
            static_svr_value, 
            np.array(features), 
            scaled_features, 
            baseline_stats
        )
        
        # Also calculate universal MI for comparison
        universal_mi = calculate_mi_universal(np.array(features), method='robust_quantile')
        
        enhanced_values.append(enhanced_mi)
        universal_values.append(universal_mi)
        
        print(f"  Static SVR: {static_svr_value:.6f}")
        print(f"  Enhanced MI: {enhanced_mi:.6f}")
        print(f"  Universal MI: {universal_mi:.6f}")
        print(f"  Difference: {abs(enhanced_mi - static_svr_value):.6f}")
        print()
    
    # Analyze results
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    enhanced_std = np.std(enhanced_values)
    universal_std = np.std(universal_values)
    enhanced_range = np.max(enhanced_values) - np.min(enhanced_values)
    universal_range = np.max(universal_values) - np.min(universal_values)
    
    print(f"Enhanced MI values: {enhanced_values}")
    print(f"Universal MI values: {universal_values}")
    print()
    print(f"Enhanced MI - Std Dev: {enhanced_std:.6f}, Range: {enhanced_range:.6f}")
    print(f"Universal MI - Std Dev: {universal_std:.6f}, Range: {universal_range:.6f}")
    print()
    
    # Check if enhancement is working
    if enhanced_std > 0.01:  # Should have some variation
        print("✅ SUCCESS: Enhanced MI shows dynamic behavior!")
        print(f"   Standard deviation: {enhanced_std:.6f} (target: > 0.01)")
    else:
        print("❌ ISSUE: Enhanced MI still appears static")
        print(f"   Standard deviation: {enhanced_std:.6f} (target: > 0.01)")
    
    # Check if values differ from static input
    static_differences = [abs(val - static_svr_value) for val in enhanced_values]
    avg_difference = np.mean(static_differences)
    
    if avg_difference > 0.01:
        print(f"✅ SUCCESS: Enhanced values differ from static input!")
        print(f"   Average difference: {avg_difference:.6f} (target: > 0.01)")
    else:
        print(f"❌ ISSUE: Enhanced values too similar to static input")
        print(f"   Average difference: {avg_difference:.6f} (target: > 0.01)")
    
    print()
    return enhanced_values, universal_values

def test_time_variation():
    """Test that time-based variations work."""
    
    print("=" * 60)
    print("Testing Time-Based Variation")
    print("=" * 60)
    
    # Same features, tested over time
    test_features = [15.0, 9.0, 0.0, 12.0, 14.0]
    static_svr_value = 0.40930194734359193
    
    values_over_time = []
    
    for i in range(10):
        enhanced_mi = enhance_svr_prediction(
            static_svr_value,
            np.array(test_features),
            np.array(test_features) / 10.0,
            None  # No baseline stats
        )
        values_over_time.append(enhanced_mi)
        print(f"Time step {i+1}: {enhanced_mi:.6f}")
        time.sleep(0.1)  # Small delay to see time variation
    
    time_std = np.std(values_over_time)
    time_range = np.max(values_over_time) - np.min(values_over_time)
    
    print()
    print(f"Time-based variation - Std Dev: {time_std:.6f}, Range: {time_range:.6f}")
    
    if time_std > 0.005:  # Should have some time-based variation
        print("✅ SUCCESS: Time-based variation is working!")
    else:
        print("❌ ISSUE: Time-based variation insufficient")
    
    return values_over_time

if __name__ == "__main__":
    try:
        # Run tests
        enhanced_vals, universal_vals = test_static_svr_enhancement()
        time_vals = test_time_variation()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("If the enhancement is working correctly, you should see:")
        print("1. Enhanced MI values that vary between samples")
        print("2. Values different from the static SVR input (0.409...)")
        print("3. Time-based variation when features are the same")
        print("4. Values generally in a reasonable range (0.1 - 0.8)")
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
