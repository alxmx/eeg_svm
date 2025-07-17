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
            stats = descriptive_stats(df)
            summary_text.append("Descriptive Stats:\n" + str(stats))
            # 3. Time series and phase analysis
            plot_time_series(df, pdf)
            phase_result = phase_comparison(df, pdf)
            summary_text.append(phase_result)
            # 4. Classification performance
            if 'true_state' in df.columns and 'pred_state' in df.columns:
                cm_report = confusion_matrix_report(df, pdf)
                summary_text.append(cm_report)
            # 5. Feature contribution
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            if feature_cols:
                feat_imp = feature_importance(df, pdf, feature_cols)
                summary_text.append("Feature-MI Correlation:\n" + feat_imp)
            # 6. Calibration quality
            if 'calibration_phase' in df.columns:
                cal_qual = calibration_quality(df, pdf)
                summary_text.append("Calibration Quality:\n" + cal_qual)
            # 7. Protocol adherence
            if 'event_marker' in df.columns:
                prot = protocol_adherence(df)
                summary_text.append(prot)
        # Add summary text as a PDF page
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0, 1, "\n\n".join(summary_text), fontsize=8, va='top')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    analyze_all_sessions()
    print(f"PDF report saved as {PDF_REPORT}")