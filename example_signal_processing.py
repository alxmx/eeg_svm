"""
Minimal Example: EEG Signal Processing Workflow

This script demonstrates how to use the signal processing workflow
with a real EEG file from the repository.
"""

import signal_processing_workflow as spw
import numpy as np
import os

def main():
    """Run example signal processing workflow."""
    
    print("=" * 60)
    print("EEG SIGNAL PROCESSING WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Find an available test file
    test_files = [
        "data/toClasify/UnicornRecorder_12_05_2025_12_24_420.csv",
        "data/toClasify/UnicornRecorder_12_05_2025_12_31_050.csv",
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if test_file is None:
        print("No test files found. Please check the data directory.")
        return
    
    print(f"Using test file: {test_file}")
    print()
    
    # Process the EEG file
    results = spw.process_eeg_signal(test_file)
    
    # Display detailed results
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Data dimensions
    print(f"Raw EEG data shape: {results['raw_data'].shape}")
    print(f"Filtered data shape: {results['filtered_data'].shape}")
    print(f"Epochs shape: {results['epochs'].shape}")
    print(f"Features shape: {results['features'].shape}")
    print(f"Normalized features shape: {results['features_normalized'].shape}")
    print()
    
    # Processing statistics
    duration_sec = results['raw_data'].shape[0] / spw.FS
    print(f"Recording duration: {duration_sec:.1f} seconds")
    print(f"Number of epochs: {results['n_epochs']}")
    print(f"Features per epoch: {results['n_features']}")
    print(f"Total features extracted: {results['n_epochs'] * results['n_features']}")
    print()
    
    # Sample statistics
    raw_mean = np.mean(results['raw_data'])
    filtered_mean = np.mean(results['filtered_data'])
    features_mean = np.mean(results['features'])
    features_norm_mean = np.mean(results['features_normalized'])
    
    print("Data Statistics:")
    print(f"  Raw data mean: {raw_mean:.6f}")
    print(f"  Filtered data mean: {filtered_mean:.6f}")
    print(f"  Features mean: {features_mean:.6f}")
    print(f"  Normalized features mean: {features_norm_mean:.6f}")
    print()
    
    # Show feature distribution
    print("Feature Distribution (first epoch, first 10 features):")
    print(f"  Raw features: {results['features'][0][:10]}")
    print(f"  Normalized:   {results['features_normalized'][0][:10]}")
    print()
    
    # Normalization statistics
    mu, sigma = results['normalization_stats']
    print(f"Normalization statistics shape: μ={mu.shape}, σ={sigma.shape}")
    print(f"Mean of means: {np.mean(mu):.6f}")
    print(f"Mean of stds: {np.mean(sigma):.6f}")
    print()
    
    print("=" * 60)
    print("WORKFLOW VALIDATION")
    print("=" * 60)
    
    # Validate that normalization worked correctly
    norm_mean = np.mean(results['features_normalized'])
    norm_std = np.std(results['features_normalized'])
    
    print(f"Normalized features mean: {norm_mean:.6f} (should be ~0)")
    print(f"Normalized features std: {norm_std:.6f} (should be ~1)")
    
    if abs(norm_mean) < 0.1 and abs(norm_std - 1.0) < 0.1:
        print("✅ Normalization validation PASSED")
    else:
        print("⚠️  Normalization validation FAILED")
    
    # Validate epoch dimensions
    expected_samples = int(spw.EPOCH_SEC * spw.FS)  # 2s * 250Hz = 500 samples
    actual_samples = results['epochs'].shape[1]
    
    print(f"Expected samples per epoch: {expected_samples}")
    print(f"Actual samples per epoch: {actual_samples}")
    
    if expected_samples == actual_samples:
        print("✅ Epoch dimensions validation PASSED")
    else:
        print("⚠️  Epoch dimensions validation FAILED")
    
    # Validate number of features
    expected_features = len(spw.CHANNELS) * len(spw.BANDS) * 5  # 8 channels * 5 bands * 5 stats = 200
    actual_features = results['n_features']
    
    print(f"Expected features per epoch: {expected_features}")
    print(f"Actual features per epoch: {actual_features}")
    
    if expected_features == actual_features:
        print("✅ Feature extraction validation PASSED")
    else:
        print("⚠️  Feature extraction validation FAILED")
    
    print()
    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("The signal processing workflow has been successfully demonstrated!")
    print("You can now use this workflow in your own EEG analysis pipeline.")
    

if __name__ == "__main__":
    main()