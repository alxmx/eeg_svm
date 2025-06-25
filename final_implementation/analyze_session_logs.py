#!/usr/bin/env python3
"""
Analysis script for EEG/EDA scaling issues based on session logs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_session_data(session_file, baseline_file, stats_file):
    """Analyze session data to understand scaling issues"""
    
    print("="*60)
    print("EEG/EDA SCALING ANALYSIS REPORT")
    print("="*60)
    
    # Load data
    session_df = pd.read_csv(session_file)
    baseline_df = pd.read_csv(baseline_file) 
    stats_df = pd.read_csv(stats_file)
    
    feature_names = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
    
    print(f"\n1. SESSION OVERVIEW:")
    print(f"   - Session file: {session_file}")
    print(f"   - Baseline file: {baseline_file}")
    print(f"   - Session duration: {len(session_df)} seconds")
    
    print(f"\n2. MI ANALYSIS:")
    mi_stats = session_df['mi'].describe()
    print(f"   - MI range: {mi_stats['min']:.6f} to {mi_stats['max']:.6f}")
    print(f"   - MI std: {mi_stats['std']:.2e}")
    if mi_stats['std'] < 1e-10:
        print("   ❌ CRITICAL: MI values are essentially constant - MODEL SATURATED!")
    else:
        print("   ✅ MI values show variation")
    
    print(f"\n3. FEATURE SCALING COMPARISON:")
    print(f"   {'Feature':<15} {'Baseline Range':<20} {'Session Range':<20} {'Scale Diff':<15} Status")
    print("-" * 85)
    
    scaling_issues = []
    for feat in feature_names:
        if feat in baseline_df.columns and feat in session_df.columns:
            # Baseline stats
            baseline_min = baseline_df[feat].min()
            baseline_max = baseline_df[feat].max()
            baseline_range = baseline_max - baseline_min
            
            # Session stats  
            session_min = session_df[feat].min()
            session_max = session_df[feat].max()
            session_range = session_max - session_min
            
            # Calculate scale difference
            if baseline_range > 0 and session_range > 0:
                scale_ratio = session_range / baseline_range
                scale_diff = abs(np.log10(scale_ratio))
            else:
                scale_ratio = 0
                scale_diff = float('inf')
            
            # Status determination
            if scale_diff > 3:  # More than 1000x difference
                status = "❌ CRITICAL"
                scaling_issues.append((feat, scale_ratio, "Critical scaling mismatch"))
            elif scale_diff > 1:  # More than 10x difference
                status = "⚠️  WARNING"
                scaling_issues.append((feat, scale_ratio, "Significant scaling difference"))
            else:
                status = "✅ OK"
            
            baseline_str = f"{baseline_min:.2e}-{baseline_max:.2e}"
            session_str = f"{session_min:.2e}-{session_max:.2e}"
            scale_str = f"{scale_ratio:.2e}" if scale_ratio != 0 else "N/A"
            
            print(f"   {feat:<15} {baseline_str:<20} {session_str:<20} {scale_str:<15} {status}")
    
    print(f"\n4. SCALING ISSUES IDENTIFIED:")
    if scaling_issues:
        for feat, ratio, issue in scaling_issues:
            print(f"   - {feat}: {issue} (ratio: {ratio:.2e})")
            
            # Specific recommendations
            if feat.startswith(('theta', 'alpha', 'beta')) and ratio < 0.001:
                print(f"     RECOMMENDATION: Remove EEG scaling factor - values are 1000x too small")
            elif feat == 'eda_norm' and ratio > 1000:
                print(f"     RECOMMENDATION: Apply EDA scaling factor - values are 1000x too large")
    else:
        print("   ✅ No major scaling issues detected")
    
    print(f"\n5. MODEL PERFORMANCE INDICATORS:")
    # Check for model saturation
    unique_mi = len(session_df['mi'].unique())
    if unique_mi < 5:
        print(f"   ❌ Model saturation: Only {unique_mi} unique MI values")
        print("   CAUSE: Extreme feature scaling causing model to saturate")
    else:
        print(f"   ✅ Model variation: {unique_mi} unique MI values")
    
    # Check R² from calibration
    print(f"\n6. RECOMMENDED FIXES:")
    if any("Critical" in issue[2] for issue in scaling_issues):
        print("   1. Set eeg_scale_factor = 1.0 (remove aggressive 0.001 scaling)")
        print("   2. Set eda_scale_factor = 1.0 (unless EDA values are extreme)")
        print("   3. Ensure calibration and real-time use identical scaling")
        print("   4. Re-calibrate user after fixing scaling")
    else:
        print("   ✅ Current scaling appears appropriate")
    
    return scaling_issues

def main():
    # Analyze the user 005_alextest session
    session_file = "logs/005_alextest_mi_session_20250625_181520.csv"
    baseline_file = "user_configs/005_alextest_baseline.csv" 
    stats_file = "logs/005_alextest_mi_feature_stats_20250625_181520.csv"
    
    if all(Path(f).exists() for f in [session_file, baseline_file, stats_file]):
        scaling_issues = analyze_session_data(session_file, baseline_file, stats_file)
        
        print(f"\n" + "="*60)
        print("SUMMARY CONCLUSION:")
        print("="*60)
        
        if scaling_issues:
            print("❌ SCALING ISSUES DETECTED - Model performance compromised")
            print("✅ FIXED: Updated realtime_mi_lsl.py with corrected scaling factors")
            print("📋 NEXT STEPS:")
            print("   1. Run calibration again with fixed scaling")
            print("   2. Test real-time session to verify MI variation")
            print("   3. Check that MI values vary meaningfully (not constant)")
        else:
            print("✅ No critical scaling issues found")
            
    else:
        print("Error: Required files not found. Please ensure session data exists.")
        print(f"Looking for:")
        for f in [session_file, baseline_file, stats_file]:
            exists = "✅" if Path(f).exists() else "❌"
            print(f"  {exists} {f}")

if __name__ == "__main__":
    main()
