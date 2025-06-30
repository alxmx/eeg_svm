#!/usr/bin/env python3
"""
Test Script for Improved MI Mapping
===================================

This script tests the improved MI calculation on the session data
to verify that the new mapping provides better dynamic range.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import our MI calculation functions
from v3_realtime_mi_lsl_dual_calibration import DualCalibrationSystem

def test_improved_mapping():
    """Test the improved MI mapping on existing session data"""
    
    # Load the session data
    session_file = "logs/022_mi_session_20250626_224154.csv"
    
    if not os.path.exists(session_file):
        print(f"Session file not found: {session_file}")
        return
    
    print("Loading session data...")
    df = pd.read_csv(session_file)
    print(f"Loaded {len(df)} samples from session")
    
    # Initialize calibration system for MI calculation
    calibration_system = DualCalibrationSystem("test_user")
    
    # Extract features from the dataframe
    feature_columns = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
    
    # Map to our 9-feature format (pad missing features with defaults)
    print("Converting features to 9-feature format...")
    
    improved_mi_values = []
    original_mi_values = df['mi'].values
    
    for idx, row in df.iterrows():
        # Create 9-feature vector from available data
        features = np.array([
            row['theta_fz'],      # theta_fz
            row['beta_frontal'],  # beta_fz (using beta_frontal as proxy)
            10.0,                 # alpha_c3 (default)
            10.0,                 # alpha_c4 (default)
            row['faa'],           # faa_c3c4
            15.0,                 # alpha_pz (default)
            row['alpha_po'],      # alpha_po
            12.0,                 # alpha_oz (default)
            row['eda_norm']       # eda_norm
        ])
        
        # Calculate improved MI
        improved_mi = calibration_system.calculate_mi_universal(features)
        improved_mi_values.append(improved_mi)
    
    improved_mi_values = np.array(improved_mi_values)
    
    # Compare results
    print(f"\n{'='*60}")
    print("MI MAPPING COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"Original MI:")
    print(f"  Range: {np.min(original_mi_values):.3f} - {np.max(original_mi_values):.3f}")
    print(f"  Mean:  {np.mean(original_mi_values):.3f} ± {np.std(original_mi_values):.3f}")
    print(f"  Dynamic Range: {np.max(original_mi_values) - np.min(original_mi_values):.3f}")
    
    print(f"\nImproved MI:")
    print(f"  Range: {np.min(improved_mi_values):.3f} - {np.max(improved_mi_values):.3f}")
    print(f"  Mean:  {np.mean(improved_mi_values):.3f} ± {np.std(improved_mi_values):.3f}")
    print(f"  Dynamic Range: {np.max(improved_mi_values) - np.min(improved_mi_values):.3f}")
    
    # Calculate improvement metrics
    original_range = np.max(original_mi_values) - np.min(original_mi_values)
    improved_range = np.max(improved_mi_values) - np.min(improved_mi_values)
    range_improvement = (improved_range - original_range) / original_range * 100
    
    original_std = np.std(original_mi_values)
    improved_std = np.std(improved_mi_values)
    std_improvement = (improved_std - original_std) / original_std * 100
    
    print(f"\nImprovements:")
    print(f"  Dynamic Range: {range_improvement:+.1f}%")
    print(f"  Standard Deviation: {std_improvement:+.1f}%")
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    x = np.arange(len(original_mi_values))
    
    # Plot original MI
    ax1.plot(x, original_mi_values, 'b-', linewidth=1, label='Original MI', alpha=0.8)
    ax1.set_ylabel('Original MI')
    ax1.set_title('Original MI Values (Narrow Range)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot improved MI
    ax2.plot(x, improved_mi_values, 'r-', linewidth=1, label='Improved MI', alpha=0.8)
    ax2.set_ylabel('Improved MI')
    ax2.set_title('Improved MI Values (Enhanced Dynamic Range)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Plot both together for comparison
    ax3.plot(x, original_mi_values, 'b-', linewidth=1, label='Original MI', alpha=0.7)
    ax3.plot(x, improved_mi_values, 'r-', linewidth=1, label='Improved MI', alpha=0.7)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('MI Value')
    ax3.set_title('Direct Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"logs/mi_mapping_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {plot_filename}")
    plt.show()
    
    # Save improved results
    df_improved = df.copy()
    df_improved['mi_improved'] = improved_mi_values
    df_improved['mi_original'] = original_mi_values
    
    output_file = f"logs/022_mi_session_improved_{timestamp}.csv"
    df_improved.to_csv(output_file, index=False)
    print(f"✓ Improved session data saved: {output_file}")
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    if range_improvement > 50:
        print("✓ SIGNIFICANT improvement in dynamic range!")
    elif range_improvement > 20:
        print("✓ Good improvement in dynamic range")
    elif range_improvement > 0:
        print("✓ Modest improvement in dynamic range")
    else:
        print("⚠ No improvement in dynamic range")
    
    if improved_range > 0.2:
        print("✓ Dynamic range is now usable for feedback applications")
    else:
        print("⚠ Dynamic range still limited - may need further tuning")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    test_improved_mapping()
