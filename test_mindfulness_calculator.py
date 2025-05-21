"""
Test script for the EEG Mindfulness Index Calculator

This script demonstrates how to use the mindfulness index calculator
on a specific EEG file.
"""

import os
from eeg_mindfulness_index import (
    process_eeg_file,
    generate_report,
    MI_WEIGHTS,
    WINDOW_SEC,
    THRESHOLDS
)

def test_mindfulness_calculator():
    # Print configuration for verification
    print("=== EEG Mindfulness Index Calculator Test ===")
    print(f"Window Size: {WINDOW_SEC} seconds")
    print(f"MI Weights: {MI_WEIGHTS}")
    print(f"Thresholds: Focused >= {THRESHOLDS['focused']}, Neutral >= {THRESHOLDS['neutral']}")
    
    # Get list of available EEG files
    data_dir = "data/toClasify"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return
    
    eeg_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not eeg_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    # Print available files
    print("\nAvailable EEG files:")
    for i, file in enumerate(eeg_files):
        print(f"{i+1}. {file}")
    
    # Process the first file as an example
    sample_file = os.path.join(data_dir, eeg_files[0])
    file_label = os.path.splitext(eeg_files[0])[0]
    
    print(f"\nProcessing file: {file_label}")
    
    # Process the EEG file
    results = process_eeg_file(sample_file)
    
    # Print basic results
    print(f"Processed {len(results)} windows")
    
    # Count states
    states = [r['behavioral_state'] for r in results]
    focused_count = states.count("Focused")
    neutral_count = states.count("Neutral")
    unfocused_count = states.count("Unfocused")
    
    total = len(states)
    print(f"Focused: {focused_count} ({focused_count/total*100:.1f}%)")
    print(f"Neutral: {neutral_count} ({neutral_count/total*100:.1f}%)")
    print(f"Unfocused: {unfocused_count} ({unfocused_count/total*100:.1f}%)")
    
    # Generate complete report
    print("\nGenerating report...")
    report_files = generate_report(results, file_label, "results/test_mindfulness")
    
    # Print output file locations
    print("\nOutput files:")
    for file_type, filepath in report_files.items():
        print(f"- {file_type}: {filepath}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results/test_mindfulness", exist_ok=True)
    
    # Run the test
    test_mindfulness_calculator()
