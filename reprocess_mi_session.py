#!/usr/bin/env python3
"""
Reprocess historical session CSV with improved MI mapping
========================================================

This script takes an existing MI session CSV and recalculates the MI values
using the improved mapping algorithm with better EDA handling and dynamic range.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

def normalize_eda_robust_improved(raw_eda, adaptive_range=None):
    """Improved EDA normalization with adaptive range"""
    if adaptive_range:
        q5, q95 = adaptive_range
    else:
        # Broader range for high arousal states
        q5, q95 = 0, 12
    
    normalized = 10 * (raw_eda - q5) / (q95 - q5)
    return np.clip(normalized, 0, 10)

def normalize_features_for_mi_improved(features):
    """Improved feature normalization"""
    ranges = {
        'theta_fz': (1, 80),       # Expanded from observed range
        'alpha_po': (1, 25),       # Reduced from 30 to better capture variation
        'faa': (-2.5, 2.5),        # Expanded asymmetry range
        'beta_frontal': (0.5, 15), # Reduced from 25
        'eda_norm': (0, 12)        # Expanded EDA range for high arousal
    }
    
    feature_names = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
    normalized = []
    
    for i, feat_name in enumerate(feature_names):
        q5, q95 = ranges[feat_name]
        val = 10 * (features[i] - q5) / (q95 - q5)
        normalized.append(np.clip(val, 0, 10))
    
    return np.array(normalized)

def calculate_mi_improved(features):
    """Improved MI calculation with better EDA handling"""
    # Improved weights with reduced EDA penalty
    weights = np.array([
        0.40,   # theta_fz: Increased attention component
        0.25,   # alpha_po: Increased relaxation component
        0.12,   # faa: Emotional balance
        -0.08,  # beta_frontal: Reduced penalty
        -0.08   # eda_norm: Significantly reduced penalty
    ])
    
    # Normalize features
    normalized_features = normalize_features_for_mi_improved(features)
    
    # Calculate weighted sum
    weighted_sum = np.dot(normalized_features, weights)
    
    # Adaptive centering based on EDA state
    eda_norm = normalized_features[4]
    theta_norm = normalized_features[0]
    alpha_norm = normalized_features[1]
    
    # Adaptive center point
    if eda_norm > 7:  # High arousal
        center_shift = -1.6  # Less negative than before
    elif alpha_norm > 6 or theta_norm > 6:  # High mindfulness
        center_shift = -0.8
    else:
        center_shift = -1.2  # More moderate default
        
    centered_sum = weighted_sum + center_shift
    
    # Sigmoid transformation with improved sensitivity
    mi_sigmoid = 1 / (1 + np.exp(-2.0 * centered_sum))  # Reduced sensitivity
    
    # Wider output range
    mi = 0.15 + 0.7 * mi_sigmoid  # 0.15 to 0.85 range
    
    return np.clip(mi, 0.15, 0.85)

def enhance_mi_dynamic_range(mi_value, features):
    """Post-process MI to enhance dynamic range"""
    theta_fz = features[0]
    alpha_po = features[1]
    eda_norm = features[4]
    
    enhanced_mi = mi_value
    
    # Anti-saturation correction for high EDA
    if eda_norm > 8:
        theta_boost = min(0.12, 0.015 * (theta_fz - 10) / 10) if theta_fz > 10 else 0
        alpha_boost = min(0.08, 0.01 * (alpha_po - 8) / 10) if alpha_po > 8 else 0
        enhanced_mi += theta_boost + alpha_boost
        
    return np.clip(enhanced_mi, 0.1, 0.9)

def reprocess_session_csv(input_path, output_path=None):
    """Reprocess a session CSV with improved MI mapping"""
    print(f"Reading session data from: {input_path}")
    
    # Read the CSV
    df = pd.read_csv(input_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original MI range: {df['mi'].min():.3f} - {df['mi'].max():.3f}")
    print(f"Original MI std: {df['mi'].std():.3f}")
    
    # Extract features and recalculate MI
    new_mi_values = []
    enhanced_mi_values = []
    
    for _, row in df.iterrows():
        features = [
            row['theta_fz'],
            row['alpha_po'], 
            row['faa'],
            row['beta_frontal'],
            row['eda_norm']
        ]
        
        # Calculate improved MI
        new_mi = calculate_mi_improved(features)
        enhanced_mi = enhance_mi_dynamic_range(new_mi, features)
        
        new_mi_values.append(new_mi)
        enhanced_mi_values.append(enhanced_mi)
    
    # Update dataframe
    df['mi_improved'] = new_mi_values
    df['mi_enhanced'] = enhanced_mi_values
    
    print(f"\nImproved MI range: {np.min(new_mi_values):.3f} - {np.max(new_mi_values):.3f}")
    print(f"Improved MI std: {np.std(new_mi_values):.3f}")
    print(f"Enhanced MI range: {np.min(enhanced_mi_values):.3f} - {np.max(enhanced_mi_values):.3f}")
    print(f"Enhanced MI std: {np.std(enhanced_mi_values):.3f}")
    
    # Save output
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_improved_mapping.csv"
    
    df.to_csv(output_path, index=False)
    print(f"\nReprocessed data saved to: {output_path}")
    
    # Generate summary stats
    stats = {
        'original_mi_mean': df['mi'].mean(),
        'original_mi_std': df['mi'].std(),
        'original_mi_range': df['mi'].max() - df['mi'].min(),
        'improved_mi_mean': np.mean(new_mi_values),
        'improved_mi_std': np.std(new_mi_values), 
        'improved_mi_range': np.max(new_mi_values) - np.min(new_mi_values),
        'enhanced_mi_mean': np.mean(enhanced_mi_values),
        'enhanced_mi_std': np.std(enhanced_mi_values),
        'enhanced_mi_range': np.max(enhanced_mi_values) - np.min(enhanced_mi_values)
    }
    
    return df, stats

if __name__ == "__main__":
    # Process the session file
    input_file = "final_implementation/logs/022_mi_session_20250626_224154.csv"
    
    if os.path.exists(input_file):
        df_processed, stats = reprocess_session_csv(input_file)
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Original MI  - Mean: {stats['original_mi_mean']:.3f}, Std: {stats['original_mi_std']:.3f}, Range: {stats['original_mi_range']:.3f}")
        print(f"Improved MI  - Mean: {stats['improved_mi_mean']:.3f}, Std: {stats['improved_mi_std']:.3f}, Range: {stats['improved_mi_range']:.3f}")
        print(f"Enhanced MI  - Mean: {stats['enhanced_mi_mean']:.3f}, Std: {stats['enhanced_mi_std']:.3f}, Range: {stats['enhanced_mi_range']:.3f}")
        
        # Calculate improvement ratios
        std_improvement = stats['improved_mi_std'] / stats['original_mi_std']
        range_improvement = stats['improved_mi_range'] / stats['original_mi_range']
        enhanced_std_improvement = stats['enhanced_mi_std'] / stats['original_mi_std']
        enhanced_range_improvement = stats['enhanced_mi_range'] / stats['original_mi_range']
        
        print(f"\nImprovement Ratios:")
        print(f"Std deviation improvement: {std_improvement:.2f}x")
        print(f"Range improvement: {range_improvement:.2f}x")
        print(f"Enhanced std improvement: {enhanced_std_improvement:.2f}x")
        print(f"Enhanced range improvement: {enhanced_range_improvement:.2f}x")
        
    else:
        print(f"File not found: {input_file}")
        print("Please check the file path.")
