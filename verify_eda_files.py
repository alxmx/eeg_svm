"""
Verify EDA file detection for the EEG signal processing pipeline

This script checks if the EDA files are properly detected for the EEG files.
"""

import os
import glob
import json
from datetime import datetime

# Import the EDA data folder configuration from the main script
from eeg_mindfulness_index import EDA_DATA_FOLDER

def check_eda_files():
    """Check if EDA files are properly detected for EEG files"""
    # Path to EEG files
    data_dir = "data/toClasify"
    
    # Get all CSV files in the directory
    eeg_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not eeg_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(eeg_files)} EEG files")
    
    # Check EDA data folder
    if not os.path.exists(EDA_DATA_FOLDER):
        print(f"WARNING: EDA data folder {EDA_DATA_FOLDER} does not exist")
        return
        
    # List available files in EDA folder
    eda_files = os.listdir(EDA_DATA_FOLDER)
    print(f"Found {len(eda_files)} files in EDA data folder: {EDA_DATA_FOLDER}")
    print("Available files:")
    for file in eda_files:
        file_path = os.path.join(EDA_DATA_FOLDER, file)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  - {file} ({file_size:.1f} KB, modified: {file_modified})")
    
    print("\nChecking EDA file detection for each EEG file:")
    # Check each EEG file
    for eeg_file in eeg_files:
        file_label = os.path.splitext(os.path.basename(eeg_file))[0]
        print(f"\nEEG file: {file_label}")
        
        # Check for different EDA file possibilities
        possible_eda_files = [
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.csv"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.txt"),
            os.path.join(EDA_DATA_FOLDER, f"opensignals_{file_label}.txt"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}.txt")
        ]
        
        eda_file = None
        for potential_file in possible_eda_files:
            if os.path.exists(potential_file):
                eda_file = potential_file
                print(f"  ✓ Found exact matching EDA file: {os.path.basename(eda_file)}")
                break
        
        # If no exact match found, look for any OpenSignals file
        if eda_file is None:
            opensignals_files = glob.glob(os.path.join(EDA_DATA_FOLDER, "opensignals_*.txt"))
            if opensignals_files:
                # Use the most recent OpenSignals file if multiple exist
                opensignals_files.sort(key=os.path.getmtime, reverse=True)
                eda_file = opensignals_files[0]
                print(f"  ✓ Using most recent OpenSignals file: {os.path.basename(eda_file)}")
            else:
                print("  ✗ No compatible EDA file found")
        
        # Report result
        if eda_file:
            print(f"  Result: Will use {os.path.basename(eda_file)}")
        else:
            print("  Result: No EDA data will be used")

if __name__ == "__main__":
    print("=== EDA File Detection Verification ===")
    check_eda_files()
