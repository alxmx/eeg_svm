"""
Manual detection of EDA files for EEG processing

This script manually checks the directory structure and files
to help diagnose issues with EDA file detection.
"""

import os
import glob
import json
from datetime import datetime

# Configuration
EDA_DATA_FOLDER = "data/eda_data"
DATA_DIR = "data/toClasify"

def manual_check():
    """Manually check directory structure and files"""
    print("\n=== Directory Structure Check ===")
    
    # Check if directories exist
    print(f"Checking directories:")
    print(f"  DATA_DIR ({DATA_DIR}): {'Exists' if os.path.exists(DATA_DIR) else 'MISSING'}")
    print(f"  EDA_DATA_FOLDER ({EDA_DATA_FOLDER}): {'Exists' if os.path.exists(EDA_DATA_FOLDER) else 'MISSING'}")
    
    # List data directory contents
    print("\nContents of DATA_DIR:")
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {file} ({file_size:.1f} KB, modified: {file_modified})")
    else:
        print("  Directory does not exist")
    
    # List EDA directory contents
    print("\nContents of EDA_DATA_FOLDER:")
    if os.path.exists(EDA_DATA_FOLDER):
        for file in os.listdir(EDA_DATA_FOLDER):
            file_path = os.path.join(EDA_DATA_FOLDER, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {file} ({file_size:.1f} KB, modified: {file_modified})")
    else:
        print("  Directory does not exist")
    
    # Check EEG files
    eeg_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"\nFound {len(eeg_files)} EEG files in DATA_DIR")
    
    # Check OpenSignals files
    opensignals_files = glob.glob(os.path.join(EDA_DATA_FOLDER, "opensignals_*.txt"))
    print(f"Found {len(opensignals_files)} OpenSignals files in EDA_DATA_FOLDER")
    
    # Print findings
    print("\n=== EDA File Detection for Each EEG File ===")
    for eeg_file in eeg_files:
        file_basename = os.path.basename(eeg_file)
        file_label = os.path.splitext(file_basename)[0]
        print(f"\nEEG file: {file_label}")
        
        # Check for different EDA file possibilities
        possible_eda_files = [
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.csv"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}_eda.txt"),
            os.path.join(EDA_DATA_FOLDER, f"opensignals_{file_label}.txt"),
            os.path.join(EDA_DATA_FOLDER, f"{file_label}.txt")
        ]
        
        print("Looking for exact matches:")
        for potential_file in possible_eda_files:
            if os.path.exists(potential_file):
                print(f"  ✓ FOUND: {os.path.basename(potential_file)}")
            else:
                print(f"  ✗ NOT FOUND: {os.path.basename(potential_file)}")
        
        # If using the fallback to any OpenSignals file
        print("\nFallback option:")
        if opensignals_files:
            # Use the most recent OpenSignals file if multiple exist
            opensignals_files.sort(key=os.path.getmtime, reverse=True)
            most_recent = opensignals_files[0]
            print(f"  ✓ Will use most recent OpenSignals file: {os.path.basename(most_recent)}")
            
            # Check the content of this file
            print(f"\nFirst few lines of {os.path.basename(most_recent)}:")
            try:
                with open(most_recent, 'r') as f:
                    lines = f.readlines()[:10]  # First 10 lines
                    for line in lines:
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"  Error reading file: {e}")
        else:
            print("  ✗ No OpenSignals files found as fallback")
    
    print("\n=== Recommendations ===")
    print("1. Make sure the EDA file is in the correct directory: data/eda_data")
    print("2. If the EDA file is in OpenSignals format, ensure it has the .txt extension")
    print("3. You may need to adjust the EDA_DATA_FOLDER path in eeg_mindfulness_index.py")

if __name__ == "__main__":
    print("=== Manual EDA File Detection ===")
    manual_check()
