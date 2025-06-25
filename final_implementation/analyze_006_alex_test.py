"""
Analysis script for user 006_alex_test to check if scaling issues are resolved.
"""

import pandas as pd
import numpy as np

# Load baseline data
baseline_file = r"c:\Users\lenin\Documents\GitHub\eeg_svm\final_implementation\user_configs\006_alex_test_baseline.csv"
baseline_df = pd.read_csv(baseline_file)

# Load session data
session_file = r"c:\Users\lenin\Documents\GitHub\eeg_svm\final_implementation\logs\006_alex_test_mi_session_20250625_182624.csv"
session_df = pd.read_csv(session_file)

# Feature columns
feature_cols = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']

print("=== FEATURE RANGE COMPARISON FOR USER 006_alex_test ===")
print()

for feature in feature_cols:
    baseline_stats = baseline_df[feature].describe()
    session_stats = session_df[feature].describe()
    
    print(f"Feature: {feature}")
    print(f"  Baseline - Mean: {baseline_stats['mean']:.3f}, Std: {baseline_stats['std']:.3f}, Range: [{baseline_stats['min']:.3f}, {baseline_stats['max']:.3f}]")
    print(f"  Session  - Mean: {session_stats['mean']:.3f}, Std: {session_stats['std']:.3f}, Range: [{session_stats['min']:.3f}, {session_stats['max']:.3f}]")
    print(f"  Mean ratio (session/baseline): {session_stats['mean'] / baseline_stats['mean']:.6f}")
    print(f"  Std ratio (session/baseline): {session_stats['std'] / baseline_stats['std']:.6f}")
    print()

# Check MI variation
print("=== MI VARIATION ANALYSIS ===")
mi_stats = session_df['mi'].describe()
print(f"MI - Mean: {mi_stats['mean']:.6f}, Std: {mi_stats['std']:.6f}, Range: [{mi_stats['min']:.6f}, {mi_stats['max']:.6f}]")
print(f"MI Coefficient of Variation: {mi_stats['std'] / mi_stats['mean']:.6f}")

# Check if MI values are stuck
unique_mi_count = session_df['mi'].nunique()
total_samples = len(session_df)
print(f"Unique MI values: {unique_mi_count} out of {total_samples} samples ({unique_mi_count/total_samples*100:.1f}%)")

if mi_stats['std'] < 0.01:
    print("⚠️  WARNING: MI has very low variation (std < 0.01) - may indicate model saturation")
else:
    print("✅ MI shows reasonable variation")

# Check raw_mi values (should be sigmoid scaled)
raw_mi_stats = session_df['raw_mi'].describe()
print(f"\nRaw MI - Mean: {raw_mi_stats['mean']:.6f}, Std: {raw_mi_stats['std']:.6f}, Range: [{raw_mi_stats['min']:.6f}, {raw_mi_stats['max']:.6f}]")

# Check if raw_mi is saturated at 1.0
saturated_count = (session_df['raw_mi'] >= 0.999999).sum()
print(f"Raw MI values near saturation (≥0.999999): {saturated_count} out of {total_samples} ({saturated_count/total_samples*100:.1f}%)")

if saturated_count > total_samples * 0.8:
    print("⚠️  WARNING: Raw MI is saturated (>80% of values near 1.0) - model may still have scaling issues")
else:
    print("✅ Raw MI shows good distribution")
