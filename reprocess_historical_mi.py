"""
Historical Mindfulness Index Reprocessing Tool

This script helps reprocess historical Mindfulness Index data to ensure
consistent visualization and analysis using the new normalization function.

It performs the following functions:
1. Reprocesses JSON result files in the mindfulness_analysis directory
2. Updates visualizations to reflect the normalized 0-1 scale
3. Generates updated summary files with the new threshold meanings
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

from eeg_mindfulness_index import (
    normalize_mi_value,
    classify_behavioral_state,
    MI_WEIGHTS,
    THRESHOLDS
)

# Paths
RESULTS_DIR = "results/mindfulness_analysis"
REPROCESSED_DIR = "results/mindfulness_analysis_normalized"

# Ensure output directory exists
os.makedirs(REPROCESSED_DIR, exist_ok=True)

def extract_file_label(filepath):
    """Extract original file label from result file path"""
    basename = os.path.basename(filepath)
    match = re.match(r'([^_]+(?:_[^_]+)*?)_\d{8}_\d{6}', basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]

def reprocess_json_file(filepath):
    """Reprocess a historical JSON results file using the new normalization"""
    print(f"Reprocessing {os.path.basename(filepath)}...")
    
    # Load the original results
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract results and metadata
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
        metadata = data.get('metadata', {})
    else:
        # For backward compatibility
        results = data
        metadata = {}
    
    # Track changes
    original_state_counts = {"Focused": 0, "Neutral": 0, "Unfocused": 0}
    new_state_counts = {"Focused": 0, "Neutral": 0, "Unfocused": 0}
    
    # Normalize each result's MI score and update behavioral state
    for result in results:
        # Store original values for comparison
        original_mi = result['mi_score']
        original_state = result['behavioral_state']
        original_state_counts[original_state] += 1
        
        # Apply normalization to the historical MI value
        normalized_mi = normalize_mi_value(original_mi)
        
        # Update the result with normalized values
        result['mi_raw'] = original_mi  # Store original as raw
        result['mi_score'] = normalized_mi  # Replace with normalized
        
        # Reclassify behavioral state with new thresholds
        new_state = classify_behavioral_state(normalized_mi)
        result['behavioral_state'] = new_state
        new_state_counts[new_state] += 1
    
    # Update metadata to include normalization info
    if metadata:
        metadata['normalized'] = True
        metadata['normalization_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['normalization_function'] = "1 / (1 + np.exp(-mi_raw + 1))"
        metadata['thresholds'] = THRESHOLDS.copy()
    
    # Create the updated data structure
    updated_data = {'results': results, 'metadata': metadata} if metadata else results
    
    return updated_data, original_state_counts, new_state_counts

def plot_mi_timeseries(results, output_filepath):
    """Create updated MI timeseries plot with normalized values"""
    timestamps = [r['timestamp'] for r in results]
    mi_values = [r['mi_score'] for r in results]
    states = [r['behavioral_state'] for r in results]
    
    # Create color map for states
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    
    colors = [state_colors[state] for state in states]
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot MI values
    scatter = plt.scatter(timestamps, mi_values, c=colors, alpha=0.7)
    plt.plot(timestamps, mi_values, 'k-', alpha=0.3)
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7)
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI) - Normalized Scale')
    plt.title('Mindfulness Index Over Time (Normalized 0-1 Scale)')
    
    # Create legend for states
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=state, markersize=10)
        for state, color in state_colors.items()
    ]
    legend_elements.append(Line2D([0], [0], linestyle='--', color='g', 
                                  label=f'Focused Threshold ({THRESHOLDS["focused"]})'))
    legend_elements.append(Line2D([0], [0], linestyle='--', color='b', 
                                  label=f'Neutral Threshold ({THRESHOLDS["neutral"]})'))
    
    # Set y-axis limits for 0-1 scale with a little padding
    plt.ylim(-0.05, 1.05)
    
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

def plot_feature_contributions(results, output_filepath):
    """Create updated feature contributions plot with normalized MI values"""
    # Extract data
    timestamps = [r['timestamp'] for r in results]
    feature_names = list(MI_WEIGHTS.keys())
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate weighted feature values
    weighted_features = []
    for result in results:
        weighted_vals = {}
        for feature in feature_names:
            if feature == 'eda_norm' and result.get('eda_value') is not None:
                val = MI_WEIGHTS[feature] * result['eda_value']
            elif feature != 'eda_norm':
                val = MI_WEIGHTS[feature] * result['features'].get(feature, 0)
            else:
                val = 0
            weighted_vals[feature] = val
        weighted_features.append(weighted_vals)
    
    # Create subplots
    plt.subplot(2, 1, 1)
    
    # Plot MI values
    mi_values = [r['mi_score'] for r in results]
    plt.plot(timestamps, mi_values, 'k-', linewidth=2, label='MI Score (Normalized)')
    
    # Add threshold lines
    plt.axhline(y=THRESHOLDS['focused'], color='g', linestyle='--', alpha=0.7, 
                label=f'Focused Threshold ({THRESHOLDS["focused"]})')
    plt.axhline(y=THRESHOLDS['neutral'], color='b', linestyle='--', alpha=0.7, 
                label=f'Neutral Threshold ({THRESHOLDS["neutral"]})')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mindfulness Index (MI)')
    plt.title('Normalized Mindfulness Index Over Time (0-1 Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)  # Set y-axis limits for 0-1 scale
    
    # Plot feature contributions
    plt.subplot(2, 1, 2)
    
    # Get weighted contribution of each feature
    for feature in feature_names:
        values = [wf[feature] for wf in weighted_features]
        plt.plot(timestamps, values, label=f"{feature} (w={MI_WEIGHTS[feature]})")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Weighted Raw Contribution')
    plt.title('Feature Contributions to Raw MI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

def plot_behavioral_state_summary(results, output_filepath):
    """Create updated behavioral state summary plot"""
    states = [r['behavioral_state'] for r in results]
    unique_states = ['Focused', 'Neutral', 'Unfocused']
    state_counts = {state: states.count(state) for state in unique_states}
    
    # Calculate percentages
    total = len(states)
    state_pcts = {state: (count / total) * 100 for state, count in state_counts.items()}
    
    # Create colors
    state_colors = {
        'Focused': 'green',
        'Neutral': 'blue',
        'Unfocused': 'red'
    }
    colors = [state_colors[state] for state in unique_states]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.subplot(1, 2, 1)
    plt.pie([state_counts[state] for state in unique_states], 
            labels=unique_states, 
            autopct='%1.1f%%', 
            colors=colors,
            explode=[0.05] * len(unique_states))
    plt.title('Distribution of Behavioral States (Normalized)')
    
    # Create bar chart
    plt.subplot(1, 2, 2)
    plt.bar(unique_states, [state_pcts[state] for state in unique_states], color=colors)
    plt.xlabel('Behavioral State')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage of Time in Each State')
    
    # Add exact percentages on top of bars
    for i, state in enumerate(unique_states):
        plt.text(i, state_pcts[state] + 1, f"{state_pcts[state]:.1f}%", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

def generate_updated_summary(file_label, results_data, original_counts, new_counts, output_path):
    """Generate updated summary file with normalization information"""
    # Extract results and metadata
    if isinstance(results_data, dict) and 'results' in results_data:
        results = results_data['results']
        metadata = results_data.get('metadata', {})
    else:
        # For backward compatibility
        results = results_data
        metadata = {}
    
    with open(output_path, 'w') as f:
        f.write("=== UPDATED Mindfulness Index (MI) Analysis Summary ===\n\n")
        f.write(f"File: {file_label}\n")
        f.write(f"Original Analysis Date: {metadata.get('analysis_date', 'Unknown')}\n")
        f.write(f"Normalization Applied: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Normalization info
        f.write("NORMALIZATION INFORMATION:\n")
        f.write("The MI values have been normalized using a sigmoid-like function:\n")
        f.write("   MI_normalized = 1 / (1 + exp(-MI_raw + 1))\n")
        f.write("This ensures all MI values fall within the 0-1 range while preserving their rank order.\n\n")
        
        # Cognitive state interpretation
        f.write("COGNITIVE STATE INTERPRETATION:\n")
        f.write(f"- Focused (MI ≥ {THRESHOLDS['focused']}): High mindfulness, concentrated attention,\n")
        f.write(f"  characterized by increased theta activity at Fz and alpha synchronization at\n")
        f.write(f"  posterior sites. Associated with meditative states and deep focus.\n\n")
        
        f.write(f"- Neutral ({THRESHOLDS['neutral']} ≤ MI < {THRESHOLDS['focused']}): Regular attentional state,\n")
        f.write(f"  balanced brain activity without strong indicators of either focused attention\n")
        f.write(f"  or distraction. Normal waking consciousness.\n\n")
        
        f.write(f"- Unfocused (MI < {THRESHOLDS['neutral']}): Low mindfulness or distracted state,\n")
        f.write(f"  characterized by increased beta activity and decreased theta/alpha coherence.\n")
        f.write(f"  Associated with mind wandering and distractibility.\n\n")
        
        # Parameters
        f.write("Parameters:\n")
        f.write(f"- Window Size: {metadata.get('window_size_sec', 3)} seconds\n")
        f.write(f"- Window Overlap: {metadata.get('overlap_pct', 50)}%\n")
        f.write(f"- MI Weights: {MI_WEIGHTS}\n")
        f.write(f"- UPDATED State Thresholds: Focused ≥ {THRESHOLDS['focused']}, " 
                f"Neutral ≥ {THRESHOLDS['neutral']}, Unfocused < {THRESHOLDS['neutral']}\n\n")
        
        # Results summary
        mi_values = [r['mi_score'] for r in results]
        mi_raw_values = [r.get('mi_raw', 0) for r in results if 'mi_raw' in r]
        
        f.write("Results Summary:\n")
        f.write(f"- Total Windows Analyzed: {len(results)}\n")
        f.write(f"- Average Normalized MI Score: {np.mean(mi_values):.4f}\n")
        f.write(f"- Normalized MI Range: {np.min(mi_values):.4f} to {np.max(mi_values):.4f}\n")
        if mi_raw_values:
            f.write(f"- Original Raw MI Range: {np.min(mi_raw_values):.4f} to {np.max(mi_raw_values):.4f}\n")
        f.write("\n")
        
        # State distribution
        f.write("Behavioral States After Normalization:\n")
        for state in ['Focused', 'Neutral', 'Unfocused']:
            count = new_counts[state]
            percentage = (count / len(results)) * 100
            f.write(f"- {state}: {count} windows ({percentage:.1f}%)\n")
        
        f.write("\nComparison with Original Classification:\n")
        f.write(f"{'State':<10} {'Original':<10} {'Normalized':<10} {'Change':<10}\n")
        f.write(f"{'-'*40}\n")
        for state in ['Focused', 'Neutral', 'Unfocused']:
            orig_count = original_counts[state]
            new_count = new_counts[state]
            change = new_count - orig_count
            change_pct = (change / len(results)) * 100 if results else 0
            change_str = f"{change:+d} ({change_pct:+.1f}%)" if change != 0 else "No change"
            f.write(f"{state:<10} {orig_count:<10} {new_count:<10} {change_str:<15}\n")

def process_single_file(json_filepath):
    """Process a single JSON results file"""
    # Determine file label from path
    file_label = extract_file_label(json_filepath)
    
    # Compute output paths
    base_output_dir = REPROCESSED_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{file_label}_normalized_{timestamp}"
    
    # Create output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Output paths
    json_output_path = os.path.join(base_output_dir, f"{base_filename}_results.json")
    mi_plot_path = os.path.join(base_output_dir, f"{base_filename}_mi_timeseries.png")
    feature_plot_path = os.path.join(base_output_dir, f"{base_filename}_feature_contributions.png")
    state_plot_path = os.path.join(base_output_dir, f"{base_filename}_behavioral_states.png")
    summary_path = os.path.join(base_output_dir, f"{base_filename}_summary.txt")
    
    # Reprocess the JSON file
    updated_data, original_counts, new_counts = reprocess_json_file(json_filepath)
    
    # Extract results
    results = updated_data['results'] if isinstance(updated_data, dict) and 'results' in updated_data else updated_data
    
    # Save the updated JSON
    with open(json_output_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    print(f"Updated JSON saved to {json_output_path}")
    
    # Create updated visualizations
    plot_mi_timeseries(results, mi_plot_path)
    plot_feature_contributions(results, feature_plot_path)
    plot_behavioral_state_summary(results, state_plot_path)
    
    # Generate summary file
    generate_updated_summary(file_label, updated_data, original_counts, new_counts, summary_path)
    print(f"Updated summary saved to {summary_path}")
    
    # Also create a CSV file with the normalized data
    csv_data = []
    for r in results:
        row = {
            'timestamp': r['timestamp'],
            'mi_raw': r.get('mi_raw', 'N/A'),
            'mi_score': r['mi_score'],
            'behavioral_state': r['behavioral_state']
        }
        for feature, value in r['features'].items():
            row[feature] = value
        csv_data.append(row)
    
    csv_path = os.path.join(base_output_dir, f"{base_filename}_data.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Normalized CSV data saved to {csv_path}")
    
    return {
        'json': json_output_path,
        'mi_plot': mi_plot_path,
        'feature_plot': feature_plot_path,
        'state_plot': state_plot_path,
        'summary': summary_path,
        'csv': csv_path
    }

def process_all_files():
    """Process all JSON result files in the results directory"""
    # Find all JSON results files
    json_pattern = os.path.join(RESULTS_DIR, "*mindfulness_results.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No result files found in {RESULTS_DIR}")
        return
    
    print(f"Found {len(json_files)} result files to process")
    
    # Process each file
    processed_files = []
    for json_file in json_files:
        output_paths = process_single_file(json_file)
        processed_files.append({
            'original': json_file,
            'outputs': output_paths
        })
        print(f"Completed processing {os.path.basename(json_file)}")
        print("-" * 40)
    
    print(f"\nAll {len(processed_files)} files processed and normalized")
    print(f"Updated results saved to {REPROCESSED_DIR}")
    
    # Generate overall summary
    generate_overall_summary(processed_files)

def generate_overall_summary(processed_files):
    """Generate an overall summary of the normalization process"""
    summary_path = os.path.join(REPROCESSED_DIR, f"normalization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== MI Normalization Process Summary ===\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Files Processed: {len(processed_files)}\n\n")
        
        f.write("NORMALIZATION METHOD:\n")
        f.write("All Mindfulness Index (MI) values have been normalized using a sigmoid function:\n")
        f.write("   MI_normalized = 1 / (1 + exp(-MI_raw + 1))\n\n")
        
        f.write("This ensures all MI values fall within the 0-1 range while preserving their relative ordering.\n")
        f.write("The function has the following properties:\n")
        f.write("- Raw MI of 0 maps to normalized MI of approximately 0.27\n")
        f.write("- Raw MI of 1 maps to normalized MI of 0.5\n")
        f.write("- Raw MI of 2 maps to normalized MI of approximately 0.73\n")
        f.write("- Very high raw MI values asymptotically approach 1.0\n")
        f.write("- Very low raw MI values asymptotically approach 0.0\n\n")
        
        f.write("UPDATED THRESHOLD VALUES:\n")
        f.write(f"- Focused: MI ≥ {THRESHOLDS['focused']} (corresponds to raw MI ≥ 1.0)\n")
        f.write(f"- Neutral: {THRESHOLDS['neutral']} ≤ MI < {THRESHOLDS['focused']} (corresponds to raw MI between 0.47 and 1.0)\n")
        f.write(f"- Unfocused: MI < {THRESHOLDS['neutral']} (corresponds to raw MI < 0.47)\n\n")
        
        f.write("PROCESSED FILES:\n")
        for i, file_info in enumerate(processed_files):
            orig_file = os.path.basename(file_info['original'])
            f.write(f"{i+1}. Original: {orig_file}\n")
            f.write(f"   Outputs in: {REPROCESSED_DIR}\n")
        
        f.write("\nIMPORTANT NOTES:\n")
        f.write("1. The normalization process preserves the relative ordering of MI values but changes their scale.\n")
        f.write("2. Some recordings may show different behavioral state classifications due to the new thresholds.\n")
        f.write("3. Future MI calculations will automatically use this normalization method for consistency.\n")
        f.write("4. When comparing historical data to new recordings, always use the normalized MI values.\n")
    
    print(f"\nOverall summary saved to {summary_path}")

if __name__ == "__main__":
    print("=== Mindfulness Index Historical Data Reprocessing Tool ===\n")
    print("This script will reprocess historical MI data using the new normalization function.")
    print(f"Original data from: {RESULTS_DIR}")
    print(f"Normalized data to: {REPROCESSED_DIR}\n")
    
    choice = input("Do you want to continue? (y/n): ").lower()
    if choice == 'y':
        process_all_files()
        print("\nReprocessing complete! The normalized data maintains the same ranking of states")
        print("but provides consistent 0-1 scale values that are easier to interpret.")
    else:
        print("Operation cancelled.")
