"""
Test the improved EDA detection and normalization in the EEG Mindfulness Index calculator.

This script tests the EDA detection and processing functionality with the OpenSignals format.
"""

import os
import sys
import glob
import numpy as np
import json
from datetime import datetime

# Import from the main script
from eeg_mindfulness_index import (
    EDA_DATA_FOLDER, 
    load_eda_csv, 
    normalize_eda, 
    process_directory,
    calculate_mi
)

def test_eda_loading():
    """Test loading EDA data from OpenSignals format"""
    print("\n=== Testing EDA Loading ===")
    
    # Get available EDA files
    if not os.path.exists(EDA_DATA_FOLDER):
        print(f"EDA data folder {EDA_DATA_FOLDER} does not exist")
        return
    
    eda_files = glob.glob(os.path.join(EDA_DATA_FOLDER, "*.txt"))
    if not eda_files:
        print(f"No .txt EDA files found in {EDA_DATA_FOLDER}")
        return
    
    # Test loading each EDA file
    for eda_file in eda_files:
        print(f"\nTesting EDA file: {os.path.basename(eda_file)}")
        try:
            eda_data = load_eda_csv(eda_file)
            if eda_data is None:
                print(f"Failed to load EDA data from {eda_file}")
                continue
                
            print(f"Successfully loaded {len(eda_data)} samples")
            print(f"Data range: {np.min(eda_data)} to {np.max(eda_data)}")
            print(f"Mean: {np.mean(eda_data)}, Std: {np.std(eda_data)}")
            
            # Test normalization
            print("\nTesting normalization:")
            eda_norm = normalize_eda(eda_data, method='zscore')
            print(f"Z-score normalized range: {np.min(eda_norm)} to {np.max(eda_norm)}")
            
            eda_norm_minmax = normalize_eda(eda_data, method='minmax')
            print(f"Min-max normalized range: {np.min(eda_norm_minmax)} to {np.max(eda_norm_minmax)}")
            
            # Test using EDA in MI calculation
            print("\nTesting EDA in MI calculation:")
            # Create dummy features
            features = {
                'theta_fz': 0.5,
                'alpha_po': 0.5,
                'faa': 0.5,
                'beta_frontal': 0.5
            }
            
            # Calculate MI with and without EDA
            mi_without_eda = calculate_mi(features, eda_value=None)
            print(f"MI without EDA: {mi_without_eda:.4f}")
            
            # Try with a few different EDA values
            if len(eda_norm) > 0:
                test_indices = [0, len(eda_norm) // 2, len(eda_norm) - 1]
                for idx in test_indices:
                    eda_value = eda_norm[idx]
                    mi_with_eda = calculate_mi(features, eda_value=eda_value)
                    print(f"MI with EDA[{idx}]={eda_value:.4f}: {mi_with_eda:.4f}")
            
        except Exception as e:
            print(f"Error processing {eda_file}: {str(e)}")

def test_directory_processing():
    """Test processing the entire directory"""
    print("\n=== Testing Directory Processing ===")
    
    data_dir = "data/toClasify"
    test_results_dir = "results/test_improved_eda"
    
    # Create test results directory if it doesn't exist
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Process the directory
    process_directory(data_dir, test_results_dir)
    
    # Check results files
    result_files = glob.glob(os.path.join(test_results_dir, "*.json"))
    if result_files:
        print(f"\nResults summary ({len(result_files)} files processed):")
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                results = data.get('results', [])
                
                print(f"\nFile: {os.path.basename(result_file)}")
                print(f"EEG file: {os.path.basename(metadata.get('file_path', 'unknown'))}")
                print(f"EDA file: {os.path.basename(metadata.get('eda_path', 'None'))}")
                print(f"Has EDA data: {metadata.get('has_eda', False)}")
                print(f"Windows analyzed: {len(results)}")
                
                if results:
                    mi_values = [r.get('mi_score', 0) for r in results]
                    print(f"MI range: {min(mi_values):.4f} to {max(mi_values):.4f}")
                    print(f"MI mean: {sum(mi_values)/len(mi_values):.4f}")
                    
                    # Count behavioral states
                    states = {}
                    for r in results:
                        state = r.get('behavioral_state', 'Unknown')
                        states[state] = states.get(state, 0) + 1
                    
                    print("Behavioral states:")
                    for state, count in states.items():
                        print(f"  - {state}: {count} ({count/len(results)*100:.1f}%)")
                        
            except Exception as e:
                print(f"Error reading {result_file}: {str(e)}")

if __name__ == "__main__":
    print("=== EDA Detection and Normalization Test ===")
    
    # Run individual tests
    test_eda_loading()
    
    # Run full directory processing test
    test_directory_processing()
