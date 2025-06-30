#!/usr/bin/env python3
"""
MI Pipeline Analysis and Comparison Tool
========================================

This script analyzes and compares the adaptive vs stable MI pipeline versions,
providing insights into their differences and helping validate improvements.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def analyze_session_logs(log_directory="final_implementation/logs"):
    """Analyze MI session logs to compare consistency between versions"""
    
    print(f"Analyzing session logs in: {log_directory}")
    
    if not os.path.exists(log_directory):
        print(f"Log directory not found: {log_directory}")
        return None
    
    # Find log files
    log_files = []
    for file in os.listdir(log_directory):
        if file.endswith('.csv') and ('mi_session' in file or 'realtime_mi' in file):
            log_files.append(os.path.join(log_directory, file))
    
    if not log_files:
        print("No MI session log files found")
        return None
    
    print(f"Found {len(log_files)} session log files:")
    for file in log_files:
        print(f"  - {os.path.basename(file)}")
    
    # Analyze each session
    analysis_results = {}
    
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            session_name = os.path.basename(log_file).replace('.csv', '')
            
            # Basic statistics
            stats = {
                'duration_samples': len(df),
                'mi_mean': df['mi'].mean() if 'mi' in df.columns else np.nan,
                'mi_std': df['mi'].std() if 'mi' in df.columns else np.nan,
                'mi_range': df['mi'].max() - df['mi'].min() if 'mi' in df.columns else np.nan,
                'raw_mi_mean': df['raw_mi'].mean() if 'raw_mi' in df.columns else np.nan,
                'raw_mi_std': df['raw_mi'].std() if 'raw_mi' in df.columns else np.nan,
                'emi_mean': df['emi'].mean() if 'emi' in df.columns else np.nan,
                'emi_std': df['emi'].std() if 'emi' in df.columns else np.nan,
                'is_stable_version': 'stable' in session_name.lower()
            }
            
            # Consistency metrics
            if 'mi' in df.columns:
                # Calculate jumps/discontinuities
                mi_diff = np.diff(df['mi'])
                stats['mi_max_jump'] = np.max(np.abs(mi_diff))
                stats['mi_avg_change'] = np.mean(np.abs(mi_diff))
                
                # Calculate stability (inverse of variance)
                stats['mi_stability'] = 1.0 / (stats['mi_std'] + 1e-8)
            
            analysis_results[session_name] = stats
            
            print(f"\nSession: {session_name}")
            print(f"  Samples: {stats['duration_samples']}")
            print(f"  MI: {stats['mi_mean']:.3f} ± {stats['mi_std']:.3f}")
            print(f"  Max Jump: {stats['mi_max_jump']:.3f}")
            print(f"  Version: {'Stable' if stats['is_stable_version'] else 'Adaptive'}")
            
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
    
    return analysis_results

def compare_mi_equations():
    """Compare MI calculation equations between versions"""
    
    print("\n" + "="*60)
    print("MI CALCULATION COMPARISON")
    print("="*60)
    
    print("\n1. ADAPTIVE VERSION (Original):")
    print("   - Uses personal baseline statistics")
    print("   - Dynamic normalization ranges")
    print("   - Saturation detection with fallback")
    print("   - Anti-static noise injection")
    print("   - User/population stat blending")
    
    print("\n   MI Calculation (Adaptive):")
    print("   normalized_features = adaptive_normalize(features, user_stats, pop_stats)")
    print("   if saturation_detected:")
    print("       mi = fallback_calculation(features)")
    print("   else:")
    print("       weighted_sum = dot(normalized_features, weights)")
    print("       mi = sigmoid_or_linear_mapping(weighted_sum)")
    print("   mi = apply_anti_static_noise(mi)")
    
    print("\n2. STABLE VERSION (New):")
    print("   - Fixed population-based normalization")
    print("   - No saturation detection")
    print("   - No anti-static mechanisms")
    print("   - No user-specific adjustments")
    
    print("\n   MI Calculation (Stable):")
    print("   normalized_features = fixed_normalize(features, FIXED_RANGES)")
    print("   feature_array = [theta_fz, alpha_po, faa, beta_frontal, eda_norm]")
    print("   weights = [0.3, 0.3, 0.2, -0.1, -0.2]")
    print("   weighted_sum = dot(feature_array, weights)")
    print("   raw_score = weighted_sum / 10.0")
    print("   mi = 0.1 + 0.8 * raw_score")
    print("   mi = clip(mi, 0.1, 0.9)")
    
    print("\n3. FIXED NORMALIZATION RANGES:")
    ranges = {
        'theta_fz': (2, 60),
        'alpha_po': (1, 30), 
        'faa': (-2.5, 2.5),
        'beta_frontal': (2, 35),
        'eda_norm': (2, 12)
    }
    
    for feature, (min_val, max_val) in ranges.items():
        print(f"   {feature}: [{min_val}, {max_val}]")
    
    print("\n4. EXPECTED CONSISTENCY IMPROVEMENTS:")
    print("   ✅ Same input → Same output (repeatability)")
    print("   ✅ No session-to-session drift")
    print("   ✅ No sudden jumps from saturation detection")
    print("   ✅ Predictable behavior for interactive apps")
    print("   ⚠️  Less personalized (trade-off)")

def generate_comparison_report(analysis_results=None):
    """Generate a comprehensive comparison report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"MI_PIPELINE_COMPARISON_REPORT_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# MI Pipeline Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares the **Adaptive** vs **Stable** MI pipeline versions.\n\n")
        
        f.write("### Key Differences\n\n")
        f.write("| Mechanism | Adaptive Version | Stable Version |\n")
        f.write("|-----------|------------------|----------------|\n")
        f.write("| Personal Baselines | ✅ User-specific adjustments | ❌ Population-based only |\n")
        f.write("| Dynamic Scaling | ✅ Real-time adaptation | ❌ Fixed scaling factors |\n")
        f.write("| Saturation Detection | ✅ Automatic fallback | ❌ Simple range clipping |\n")
        f.write("| Normalization | ✅ User/population blending | ❌ Fixed population ranges |\n")
        f.write("| Anti-Static Logic | ✅ Prevents frozen outputs | ❌ Natural variation only |\n")
        f.write("| Session Consistency | ⚠️ Variable between sessions | ✅ Consistent across sessions |\n\n")
        
        f.write("## Technical Implementation\n\n")
        f.write("### Stable Version Changes\n\n")
        f.write("```python\n")
        f.write("# Fixed normalization ranges (never change)\n")
        f.write("FIXED_NORMALIZATION_RANGES = {\n")
        f.write("    'theta_fz': (2, 60),\n")
        f.write("    'alpha_po': (1, 30),\n")
        f.write("    'faa': (-2.5, 2.5),\n")
        f.write("    'beta_frontal': (2, 35),\n")
        f.write("    'eda_norm': (2, 12)\n")
        f.write("}\n\n")
        f.write("# Fixed MI calculation\n")
        f.write("weights = np.array([0.3, 0.3, 0.2, -0.1, -0.2])\n")
        f.write("weighted_sum = np.dot(normalized_features, weights)\n")
        f.write("mi = 0.1 + 0.8 * (weighted_sum / 10.0)\n")
        f.write("mi = np.clip(mi, 0.1, 0.9)\n")
        f.write("```\n\n")
        
        if analysis_results:
            f.write("## Session Analysis Results\n\n")
            
            # Separate stable vs adaptive sessions
            stable_sessions = {k: v for k, v in analysis_results.items() if v['is_stable_version']}
            adaptive_sessions = {k: v for k, v in analysis_results.items() if not v['is_stable_version']}
            
            f.write(f"- **Stable Sessions**: {len(stable_sessions)}\n")
            f.write(f"- **Adaptive Sessions**: {len(adaptive_sessions)}\n\n")
            
            if stable_sessions and adaptive_sessions:
                # Compare consistency metrics
                stable_stds = [s['mi_std'] for s in stable_sessions.values() if not np.isnan(s['mi_std'])]
                adaptive_stds = [s['mi_std'] for s in adaptive_sessions.values() if not np.isnan(s['mi_std'])]
                
                if stable_stds and adaptive_stds:
                    f.write("### Consistency Comparison\n\n")
                    f.write(f"- **Stable Version** - Average MI std: {np.mean(stable_stds):.4f}\n")
                    f.write(f"- **Adaptive Version** - Average MI std: {np.mean(adaptive_stds):.4f}\n\n")
                    
                    if np.mean(stable_stds) < np.mean(adaptive_stds):
                        f.write("✅ **Stable version shows improved consistency** (lower standard deviation)\n\n")
                    else:
                        f.write("⚠️ **Adaptive version shows higher consistency** in this dataset\n\n")
        
        f.write("## Usage Recommendations\n\n")
        f.write("### For Interactive Applications\n")
        f.write("✅ **Use Stable Version** - Provides predictable, consistent MI values\n\n")
        f.write("### For Research/Clinical Applications\n")
        f.write("⚠️ **Consider Adaptive Version** - May provide better personalization\n\n")
        f.write("### For Validation/Testing\n")
        f.write("✅ **Use Both Versions** - Compare outputs to understand differences\n\n")
        
        f.write("## Files\n\n")
        f.write("- **Adaptive Pipeline**: `final_implementation/realtime_mi_lsl.py`\n")
        f.write("- **Stable Pipeline**: `final_implementation/realtime_mi_lsl_stable.py`\n")
        f.write("- **XDF Reader**: `xdf_reader.py`\n")
        f.write("- **Quick Start Guide**: `STABLE_MI_QUICK_START.md`\n\n")
    
    print(f"\nComparison report generated: {report_path}")
    return report_path

def main():
    """Main analysis function"""
    
    print("MI Pipeline Analysis and Comparison Tool")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('final_implementation'):
        print("Warning: 'final_implementation' directory not found.")
        print("Please run this script from the project root directory.")
    
    # Analyze session logs
    analysis_results = analyze_session_logs()
    
    # Show MI equation comparison
    compare_mi_equations()
    
    # Generate comprehensive report
    report_path = generate_comparison_report(analysis_results)
    
    print(f"\n{'='*50}")
    print("Analysis Complete!")
    print(f"{'='*50}")
    print(f"Report saved: {report_path}")
    print("\nNext steps:")
    print("1. Test both pipeline versions with your hardware")
    print("2. Record sessions and compare MI outputs")
    print("3. Use XDF reader to analyze recorded data")
    print("4. Adjust fixed parameters if needed for your use case")

if __name__ == "__main__":
    main()
