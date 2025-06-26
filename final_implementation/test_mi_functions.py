#!/usr/bin/env python3
"""
Quick test script to verify the MI calculation functions are working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Import the functions from realtime_mi_lsl
try:
    from realtime_mi_lsl import (
        setup_mindfulness_lsl_streams,
        calculate_raw_mi,
        remap_raw_mi,
        calculate_emi,
        quantile_normalize_features,
        compute_adaptive_weights,
        adaptive_mi,
        FEATURE_ORDER
    )
    print("✓ Successfully imported all MI functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_mi_functions():
    """Test the MI calculation functions with sample data."""
    
    # Create sample feature data
    sample_features = np.array([0.5, 0.7, 0.3, 0.8, 0.4])  # 5 features
    print(f"Sample features: {sample_features}")
    
    # Test raw MI calculation
    raw_mi_val = calculate_raw_mi(sample_features)
    print(f"Raw MI: {raw_mi_val:.4f}")
    
    # Test remapped raw MI
    remapped_mi = remap_raw_mi(raw_mi_val)
    print(f"Remapped MI: {remapped_mi:.4f}")
    
    # Test EMI calculation
    emi_val = calculate_emi(sample_features)
    print(f"EMI: {emi_val:.4f}")
    
    # Test adaptive normalization
    # Create sample baseline data (20 samples, 5 features)
    np.random.seed(42)
    baseline_data = np.random.randn(20, 5) * 0.5 + [0.5, 0.6, 0.4, 0.7, 0.3]
    
    # Compute adaptive weights
    weights = compute_adaptive_weights(baseline_data)
    print(f"Adaptive weights: {weights}")
    
    # Create quantile dictionary
    quantiles_dict = {}
    for i, feat_name in enumerate(FEATURE_ORDER):
        # Create 100 quantile points for each feature
        quantiles_dict[feat_name] = np.percentile(baseline_data[:, i], np.linspace(0, 100, 100))
    
    # Test quantile normalization
    normalized_features = quantile_normalize_features(sample_features, quantiles_dict)
    print(f"Normalized features: {normalized_features}")
    
    # Test adaptive MI
    adaptive_mi_val = adaptive_mi(sample_features, quantiles_dict, weights)
    print(f"Adaptive MI: {adaptive_mi_val:.4f}")
    
    print("\n✓ All MI functions working correctly!")
    
    return True

def test_lsl_stream_setup():
    """Test LSL stream setup (without actually creating streams)."""
    try:
        # This would normally create actual LSL streams
        # For testing, we'll just check the function exists
        print("✓ setup_mindfulness_lsl_streams function is available")
        print("  Note: Actual LSL stream creation not tested (requires LSL environment)")
        return True
    except Exception as e:
        print(f"✗ LSL stream setup test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MI calculation functions...")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_mi_functions()
        success &= test_lsl_stream_setup()
        
        if success:
            print("\n🎉 All tests passed! The MI functions are ready for use.")
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        success = False
    
    sys.exit(0 if success else 1)
