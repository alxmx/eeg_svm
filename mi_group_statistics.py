"""
Group-Level Mindfulness Index (MI) Statistics and Diagnostics

This script aggregates and analyzes MI results and model performance across all participants.
It is intended to be run after all user sessions are complete.

Features:
- Computes descriptive statistics (mean, std, min, max) for MI per participant.
- Calculates percentage of time in Focused, Neutral, Unfocused states.
- Performs Wilcoxon Signed-Rank Test for phase comparisons (if phase labels exist).
- Computes Spearman correlations between features and MI.
- Evaluates model performance (MAE, R², macro-precision, recall, F1, confusion matrix).
- Summarizes session-level info: total MI predictions, state proportions, skipped windows.
- Runs diagnostics for missing values, artifacts, and constant/invalid streams.
- Outputs summary tables (CSV/Excel) and visualizations (PDF/PNG).

Usage:
    python mi_group_statistics.py
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
USER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'user_configs')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']
MI_THRESHOLDS = {'focused': 0.5, 'neutral': 0.37}

# --- Helper functions ---
def bin_mi(mi):
    if mi >= MI_THRESHOLDS['focused']:
        return 2  # Focused
    elif mi >= MI_THRESHOLDS['neutral']:
        return 1  # Neutral
    else:
        return 0  # Unfocused

def load_all_sessions():
    mi_files = glob.glob(os.path.join(LOG_DIR, '*_mi_session_*.csv'))
    sessions = {}
    for f in mi_files:
        user = os.path.basename(f).split('_mi_session_')[0]
        try:
            df = pd.read_csv(f)
            if not df.empty and 'mi' in df.columns:
                sessions.setdefault(user, []).append(df)
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")
    return sessions

def load_all_calibrations():
    calib_files = glob.glob(os.path.join(USER_CONFIG_DIR, '*_baseline.csv'))
    calibs = {}
    for f in calib_files:
        user = os.path.basename(f).split('_baseline')[0]
        try:
            df = pd.read_csv(f)
            if not df.empty:
                calibs[user] = df
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")
    return calibs

def load_model_reports():
    report_files = glob.glob(os.path.join(LOG_DIR, '*calibration_comparative_*.csv'))
    reports = {}
    for f in report_files:
        user = os.path.basename(f).split('_calibration_comparative_')[0]
        try:
            df = pd.read_csv(f)
            if not df.empty:
                reports[user] = df.iloc[-1]  # Use latest
        except Exception as e:
            print(f"[WARN] Could not load {f}: {e}")
    return reports

def compute_state_percentages(mi_arr):
    n = len(mi_arr)
    focused = np.sum(mi_arr >= MI_THRESHOLDS['focused']) / n * 100
    neutral = np.sum((mi_arr >= MI_THRESHOLDS['neutral']) & (mi_arr < MI_THRESHOLDS['focused'])) / n * 100
    unfocused = np.sum(mi_arr < MI_THRESHOLDS['neutral']) / n * 100
    return focused, neutral, unfocused

def main():
    sessions = load_all_sessions()
    calibs = load_all_calibrations()
    reports = load_model_reports()
    group_summary = []
    for user, user_sessions in sessions.items():
        for i, df in enumerate(user_sessions):
            mi_arr = df['mi'].dropna().values
            # Descriptive stats
            mi_mean = np.mean(mi_arr)
            mi_std = np.std(mi_arr)
            mi_min = np.min(mi_arr)
            mi_max = np.max(mi_arr)
            focused, neutral, unfocused = compute_state_percentages(mi_arr)
            # Skipped windows
            skipped = df.shape[0] - len(mi_arr)
            # Phase comparison (if 'phase' column exists)
            phase_stats = {}
            if 'phase' in df.columns:
                phases = df['phase'].unique()
                if len(phases) == 2:
                    vals1 = df[df['phase'] == phases[0]]['mi'].dropna()
                    vals2 = df[df['phase'] == phases[1]]['mi'].dropna()
                    if len(vals1) > 0 and len(vals2) > 0:
                        try:
                            stat, p = wilcoxon(vals1, vals2)
                            phase_stats = {'phase1': phases[0], 'phase2': phases[1], 'wilcoxon_stat': stat, 'wilcoxon_p': p}
                        except Exception as e:
                            phase_stats = {'phase1': phases[0], 'phase2': phases[1], 'wilcoxon_stat': None, 'wilcoxon_p': None}
            # Feature correlations (if features available)
            feature_corrs = {}
            for feat in FEATURE_ORDER:
                if feat in df.columns:
                    try:
                        corr, p = spearmanr(df[feat], df['mi'], nan_policy='omit')
                        feature_corrs[feat] = corr
                    except Exception:
                        feature_corrs[feat] = np.nan
            # Model performance (if report available)
            model_mae = model_r2 = model_prec = model_rec = model_f1 = None
            if user in reports:
                model_mae = reports[user].get('new_mae', None)
                model_r2 = reports[user].get('new_r2', None)
            # Classification metrics (if label column exists)
            prec = rec = f1 = None
            confmat = None
            if 'label' in df.columns:
                y_true = df['label'].dropna().values
                y_pred = df['mi'].dropna().values
                if len(y_true) == len(y_pred):
                    y_true_bin = [bin_mi(val) for val in y_true]
                    y_pred_bin = [bin_mi(val) for val in y_pred]
                    prec = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
                    rec = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
                    f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
                    confmat = confusion_matrix(y_true_bin, y_pred_bin)
            # Diagnostics
            n_missing = df['mi'].isna().sum()
            n_const = int(np.all(mi_arr == mi_arr[0])) if len(mi_arr) > 0 else 0
            # Summary row
            row = {
                'user': user,
                'session': i+1,
                'n_predictions': len(mi_arr),
                'mi_mean': mi_mean,
                'mi_std': mi_std,
                'mi_min': mi_min,
                'mi_max': mi_max,
                'focused_pct': focused,
                'neutral_pct': neutral,
                'unfocused_pct': unfocused,
                'skipped_windows': skipped,
                'n_missing': n_missing,
                'constant_stream': n_const,
                'model_mae': model_mae,
                'model_r2': model_r2,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'phase_stats': phase_stats,
                'feature_corrs': feature_corrs,
                'confusion_matrix': confmat.tolist() if confmat is not None else None
            }
            group_summary.append(row)
    # Save summary
    now_str = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    summary_df = pd.DataFrame(group_summary)
    out_csv = os.path.join(LOG_DIR, f'group_mi_statistics_{now_str}.csv')
    summary_df.to_csv(out_csv, index=False)
    print(f"[REPORT] Group MI statistics saved to {out_csv}")
    # Optional: Excel and visualizations
    try:
        out_xlsx = os.path.join(LOG_DIR, f'group_mi_statistics_{now_str}.xlsx')
        summary_df.to_excel(out_xlsx, index=False)
        print(f"[REPORT] Group MI statistics saved to {out_xlsx}")
    except Exception as e:
        print(f"[WARN] Could not save Excel: {e}")
    # Visualize MI distributions
    plt.figure(figsize=(10,6))
    for user, user_sessions in sessions.items():
        for df in user_sessions:
            plt.hist(df['mi'].dropna(), bins=30, alpha=0.5, label=user)
    plt.xlabel('Mindfulness Index (MI)')
    plt.ylabel('Count')
    plt.title('MI Distribution Across Participants')
    plt.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(LOG_DIR, f'group_mi_distribution_{now_str}.png')
    plt.savefig(dist_plot_path)
    print(f"[REPORT] MI distribution plot saved to {dist_plot_path}")

    # --- PDF summary report ---
    try:
        from fpdf import FPDF
        pdf_path = os.path.join(LOG_DIR, f'group_mi_statistics_{now_str}.pdf')
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Group Mindfulness Index (MI) Summary', ln=1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Report generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
        pdf.ln(5)
        # --- Interpretation Guide ---
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 9, 'Interpretation Guide', ln=1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 7, (
            "This report summarizes Mindfulness Index (MI) results and model performance for all participants.\n"
            "Key metrics:\n"
            "- MI (Mindfulness Index): Higher values indicate greater mindfulness (range: 0-1).\n"
            "- Focused/Neutral/Unfocused %: Proportion of time spent in each state.\n"
            "- Model MAE/R²: Model accuracy (lower MAE, higher R² are better).\n"
            "- Precision/Recall/F1: Classification performance for state prediction.\n"
            "- Skipped Windows: Data segments excluded due to artifacts or missing data.\n"
            "Interpretation: High MI and Focused % suggest strong mindfulness. High Unfocused % or low MI may indicate distraction. Model metrics help assess reliability."
        ))
        pdf.ln(4)
        # For each user/session, print vertical table and interpretation
        col_names = ['user','session','n_predictions','mi_mean','mi_std','mi_min','mi_max','focused_pct','neutral_pct','unfocused_pct','skipped_windows','model_mae','model_r2','precision','recall','f1']
        for idx, row in summary_df.iterrows():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f"Participant: {row['user']} | Session: {row['session']}", ln=1)
            pdf.set_font('Arial', '', 11)
            for col in col_names:
                if col in ['user', 'session']:
                    continue
                label = col.replace('_', ' ').capitalize()
                val = row.get(col, '')
                if isinstance(val, float):
                    val = f"{val:.3f}"
                pdf.cell(50, 8, f"{label}", border=0)
                pdf.cell(0, 8, str(val), border=0, ln=1)
            pdf.ln(1)
            # --- Interpretation Template ---
            mi_mean = row.get('mi_mean', None)
            focused = row.get('focused_pct', None)
            unfocused = row.get('unfocused_pct', None)
            model_r2 = row.get('model_r2', None)
            model_mae = row.get('model_mae', None)
            f1 = row.get('f1', None)
            interpretation = "Session Interpretation: "
            if mi_mean is not None:
                if mi_mean >= 0.5:
                    interpretation += "High average MI suggests strong mindfulness during this session. "
                elif mi_mean >= 0.37:
                    interpretation += "Moderate MI indicates a balanced or neutral state. "
                else:
                    interpretation += "Low MI suggests periods of distraction or low mindfulness. "
            if focused is not None and unfocused is not None:
                if focused > 50:
                    interpretation += f"Participant spent {focused:.1f}% of time in the Focused state. "
                elif unfocused > 40:
                    interpretation += f"Unfocused state was predominant ({unfocused:.1f}%). "
            if model_r2 is not None and model_mae is not None:
                interpretation += f"Model R²={model_r2:.2f}, MAE={model_mae:.2f}. "
                if model_r2 > 0.5:
                    interpretation += "Model fit is strong. "
                elif model_r2 > 0.2:
                    interpretation += "Model fit is moderate. "
                else:
                    interpretation += "Model fit is weak; interpret MI with caution. "
            if f1 is not None:
                interpretation += f"Classification F1={f1:.2f}. "
            pdf.set_font('Arial', 'I', 10)
            pdf.multi_cell(0, 7, interpretation)
            pdf.ln(3)
        # Add MI distribution plot
        if os.path.exists(dist_plot_path):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'MI Distribution Across Participants', ln=1)
            pdf.image(dist_plot_path, w=180)
        pdf.output(pdf_path)
        print(f"[REPORT] Group MI statistics PDF saved to {pdf_path}")
    except ImportError:
        print("[WARN] fpdf not installed. PDF report not generated. Install with 'pip install fpdf'.")
    except Exception as e:
        print(f"[WARN] Could not generate PDF: {e}")
    # --- END ---

if __name__ == "__main__":
    main()
