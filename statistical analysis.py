import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_rel, f_oneway
from sklearn.metrics import confusion_matrix, classification_report

# Set your data directory
DATA_DIR = "final_implementation/logs"  # e.g., "./logs" or "./data"
PDF_REPORT = "experiment_summary.pdf"

# 2. Descriptive Statistics
def descriptive_stats(df, index_cols=['MI', 'EMI', 'ATT'], phase_col='phase'):
    stats = {}
    for col in index_cols:
        stats[col] = df.groupby(phase_col)[col].agg(['mean', 'std', 'min', 'max', 'median'])
    return stats

# 3. Time Series and Phase Analysis
def plot_time_series(df, pdf, index_cols=['MI', 'EMI', 'ATT'], period_sec=20):
    for col in index_cols:
        plt.figure(figsize=(10, 4))
        plt.plot(df['timestamp'], df[col], label=col)
        # Color background every period_sec
        tmax = df['timestamp'].max()
        for i in range(0, int(tmax), period_sec):
            plt.axvspan(i, i+period_sec, color='lightgrey' if (i//period_sec)%2==0 else 'white', alpha=0.3)
        plt.title(f"{col} over Time (shaded every {period_sec}s)")
        plt.xlabel("Time (s)")
        plt.ylabel(col)
        plt.legend()
        pdf.savefig()
        plt.close()

def phase_comparison(df, pdf, index_col='MI', phase_col='phase'):
    phases = df[phase_col].unique()
    phase_data = [df[df[phase_col] == phase][index_col] for phase in phases]
    plt.figure()
    plt.boxplot(phase_data, labels=phases)
    plt.title(f"{index_col} by Phase")
    plt.ylabel(index_col)
    pdf.savefig()
    plt.close()
    if len(phases) == 2:
        stat, p = ttest_rel(phase_data[0], phase_data[1])
        result = f"Paired t-test between {phases[0]} and {phases[1]}: p={p:.4f}"
    elif len(phases) > 2:
        stat, p = f_oneway(*phase_data)
        result = f"ANOVA across phases: p={p:.4f}"
    else:
        result = "Not enough phases for statistical test."
    return result

def confusion_matrix_report(df, pdf, true_col='true_state', pred_col='pred_state'):
    cm = confusion_matrix(df[true_col], df[pred_col])
    report = classification_report(df[true_col], df[pred_col])
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    pdf.savefig()
    plt.close()
    return f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}"

def feature_importance(df, pdf, feature_cols, mi_col='MI'):
    corr = df[feature_cols].corrwith(df[mi_col])
    plt.figure()
    corr.plot(kind='bar', title='Feature Contribution to MI')
    plt.ylabel('Correlation')
    pdf.savefig()
    plt.close()
    return corr.to_string()

def calibration_quality(df, pdf, mi_col='MI', cal_col='calibration_phase'):
    stats = df.groupby(cal_col)[mi_col].agg(['mean', 'std', 'min', 'max'])
    plt.figure()
    stats[['mean', 'std']].plot(kind='bar')
    plt.title("Calibration Quality")
    plt.ylabel(mi_col)
    pdf.savefig()
    plt.close()
    return stats.to_string()

def protocol_adherence(df):
    return f"Event Markers Detected: {df['event_marker'].unique()}"

# 8. Statistical Reporting (already included in above functions)
# 9. Limitations and Recommendations (manual, based on output)

# Main analysis loop
def analyze_all_sessions():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))  # Adjust extension as needed
    with PdfPages(PDF_REPORT) as pdf:
        summary_text = []
        for file in files:
            df = pd.read_csv(file)
            summary_text.append(f"\nAnalyzing {file}\n")
            # 2. Descriptive stats
            if set(['variable', 'mean', 'std', 'min', 'max']).issubset(df.columns):
                summary_text.append("Summary statistics file detected.")
                summary_text.append(str(df))
                df.plot(x='variable', y='mean', kind='bar', title=f"Mean values in {file}")
                pdf.savefig()
                plt.close()
            else:
                summary_text.append("Raw feature time series detected.")
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats = df[col].agg(['mean', 'std', 'min', 'max', 'median'])
                        summary_text.append(f"{col} stats:\n{stats}\n")
                        plt.figure()
                        plt.plot(range(len(df)), df[col])
                        plt.title(f"{col} over Time (1 Hz) in {file}")
                        plt.xlabel("Time (s)")
                        plt.ylabel(col)
                        pdf.savefig()
                        plt.close()
                    else:
                        summary_text.append(f"{col} is non-numeric, skipping stats and plot.\n")
        # Add summary text as a PDF page
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0, 1, "\n\n".join(summary_text), fontsize=8, va='top')
        pdf.savefig()
        plt.close()

        session_dfs = []
        session_lengths = []
        session_files = []

        for file in files:
            df = pd.read_csv(file)
            if not set(['variable', 'mean', 'std', 'min', 'max']).issubset(df.columns):
                session_dfs.append(df)
                session_lengths.append(len(df))
                session_files.append(file)

        # Group sessions by length
        less3_idx = [i for i, l in enumerate(session_lengths) if l < 180]
        more5_idx = [i for i, l in enumerate(session_lengths) if l > 300]

        def get_common_numeric_cols(dfs, idxs):
            if not idxs:
                return []
            numeric_cols = [set([c for c in dfs[i].columns if pd.api.types.is_numeric_dtype(dfs[i][c])]) for i in idxs]
            return set.intersection(*numeric_cols) if numeric_cols else set()

        # Plot for <3 min sessions
        common_cols_less3 = get_common_numeric_cols(session_dfs, less3_idx)
        for col in common_cols_less3:
            plt.figure()
            for i in less3_idx:
                plt.plot(range(len(session_dfs[i])), session_dfs[i][col], label=os.path.basename(session_files[i]))
            plt.title(f"{col} comparison (<3 min sessions)")
            plt.xlabel("Time (s)")
            plt.ylabel(col)
            plt.legend()
            pdf.savefig()
            plt.close()

        # Plot for >5 min sessions
        common_cols_more5 = get_common_numeric_cols(session_dfs, more5_idx)
        for col in common_cols_more5:
            plt.figure()
            for i in more5_idx:
                plt.plot(range(len(session_dfs[i])), session_dfs[i][col], label=os.path.basename(session_files[i]))
            plt.title(f"{col} comparison (>5 min sessions)")
            plt.xlabel("Time (s)")
            plt.ylabel(col)
            plt.legend()
            pdf.savefig()
            plt.close()

        # Plot all <3 min sessions together
        if less3_idx:
            plt.figure()
            for col in common_cols_less3:
                for i in less3_idx:
                    plt.plot(range(len(session_dfs[i])), session_dfs[i][col], label=f"{col}-{os.path.basename(session_files[i])}")
            plt.title("All <3 min sessions (all parameters)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()
            pdf.savefig()
            plt.close()

        # Plot all >5 min sessions together
        if more5_idx:
            plt.figure()
            for col in common_cols_more5:
                for i in more5_idx:
                    plt.plot(range(len(session_dfs[i])), session_dfs[i][col], label=f"{col}-{os.path.basename(session_files[i])}")
            plt.title("All >5 min sessions (all parameters)")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    analyze_all_sessions()
    print(f"PDF report saved as {PDF_REPORT}")

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "final_implementation/logs"
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

for file in files:
    df = pd.read_csv(file)
    print(f"\nFile: {file}")
    # Detect summary statistics file
    if set(['variable', 'mean', 'std', 'min', 'max']).issubset(df.columns):
        print("Summary statistics file detected.")
        print(df)
        # Optionally, plot bar charts for means/stds
        df.plot(x='variable', y='mean', kind='bar', title=f"Mean values in {file}")
        plt.show()
    else:
        print("Raw feature time series detected.")
        time = range(len(df))
        for col in df.columns:
            plt.figure()
            plt.plot(time, df[col])
            plt.title(f"{col} over Time (1 Hz) in {file}")
            plt.xlabel("Time (s)")
            plt.ylabel(col)
            plt.show()
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].agg(['mean', 'std', 'min', 'max', 'median'])
                print(f"{col} stats:\n{stats}\n")
            else:
                print(f"{col} is non-numeric, skipping stats.\n")