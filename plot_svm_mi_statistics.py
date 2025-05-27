"""
Plotting Statistics of SVM Classification with MI-based Labels Over Time

This script loads the results from svm_mi_classification.py and visualizes:
- The distribution of predicted labels over time
- The moving average of classification confidence (if available)
- The proportion of each class in sliding windows

Assumes you have access to the features, labels, and predictions from the SVM MI classification.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from eeg_mindfulness_index import load_eeg_csv, bandpass_filter, WINDOW_SEC, FS
import glob

# --- Parameters ---
DATA_DIR = 'data/toClasify'
EDA_FILE = 'data/eda_data/opensignals_0007808C0700_2025-04-30_16-32-15_V.txt'
RESULTS_CSV = 'svm_mi_classification_results.csv'  # Will be created if not present

# --- Helper: Run classification and save results if not present ---
def run_and_save_results():
    from svm_mi_classification import build_mi_dataset_with_eda, load_eda_csv, normalize_eda
    eda_raw = load_eda_csv(EDA_FILE)
    if eda_raw is None or len(eda_raw) == 0:
        eda_data = None
        eda_fs = FS
    else:
        eda_data = normalize_eda(eda_raw)
        eda_fs = FS
    X, y = build_mi_dataset_with_eda(DATA_DIR, eda_data, eda_fs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    # Save results to CSV
    df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(X_test.shape[1])])
    df['true_label'] = y_test
    df['pred_label'] = y_pred
    df['confidence'] = np.max(y_proba, axis=1)
    df['window_idx'] = np.arange(len(df))
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to {RESULTS_CSV}")
    return df

# --- Main plotting function ---
def plot_statistics():
    if not os.path.exists(RESULTS_CSV):
        df = run_and_save_results()
    else:
        df = pd.read_csv(RESULTS_CSV)
    # Plot predicted label over time
    plt.figure(figsize=(14, 5))
    plt.plot(df['window_idx'], df['pred_label'], 'o-', alpha=0.7)
    plt.xlabel('Window Index (Time)')
    plt.ylabel('Predicted Label')
    plt.title('Predicted SVM MI Label Over Time')
    plt.tight_layout()
    plt.savefig('svm_mi_predicted_label_over_time.png')
    plt.close()
    # Plot moving average of confidence
    plt.figure(figsize=(14, 5))
    plt.plot(df['window_idx'], df['confidence'], '-', alpha=0.7, label='Confidence')
    plt.xlabel('Window Index (Time)')
    plt.ylabel('Classification Confidence')
    plt.title('SVM MI Classification Confidence Over Time')
    plt.tight_layout()
    plt.savefig('svm_mi_confidence_over_time.png')
    plt.close()
    # Plot class proportions in sliding window
    window_size = 20
    proportions = []
    for i in range(len(df) - window_size + 1):
        window = df['pred_label'].iloc[i:i+window_size]
        counts = window.value_counts(normalize=True)
        proportions.append({k: counts.get(k, 0) for k in ['Focused', 'Neutral', 'Unfocused']})
    prop_df = pd.DataFrame(proportions)
    plt.figure(figsize=(14, 5))
    for label in ['Focused', 'Neutral', 'Unfocused']:
        plt.plot(prop_df.index, prop_df[label], label=label)
    plt.xlabel(f'Window Start Index (sliding window size={window_size})')
    plt.ylabel('Proportion')
    plt.title('Proportion of Each Class in Sliding Window')
    plt.legend()
    plt.tight_layout()
    plt.savefig('svm_mi_class_proportion_over_time.png')
    plt.close()
    print("Plots saved: svm_mi_predicted_label_over_time.png, svm_mi_confidence_over_time.png, svm_mi_class_proportion_over_time.png")

# --- New: Compare all result files and plot together ---
def plot_comparison_all_files(results_pattern='svm_mi_classification_results*.csv', window_size=20, save_path='svm_mi_comparison_over_time.png'):
    result_files = glob.glob(results_pattern)
    if not result_files:
        print(f"No result files matching {results_pattern} found.")
        return
    n_files = len(result_files)
    fig, axes = plt.subplots(n_files, 3, figsize=(18, 5 * n_files), squeeze=False)
    for idx, file in enumerate(result_files):
        df = pd.read_csv(file)
        file_label = os.path.basename(file)
        # Predicted label over time
        axes[idx, 0].plot(df['window_idx'], df['pred_label'], 'o-', alpha=0.7)
        axes[idx, 0].set_xlabel('Window Index (Time)')
        axes[idx, 0].set_ylabel('Predicted Label')
        axes[idx, 0].set_title(f'{file_label}: Predicted Label')
        # Confidence over time
        axes[idx, 1].plot(df['window_idx'], df['confidence'], '-', alpha=0.7)
        axes[idx, 1].set_xlabel('Window Index (Time)')
        axes[idx, 1].set_ylabel('Confidence')
        axes[idx, 1].set_title(f'{file_label}: Confidence')
        # Class proportions in sliding window
        proportions = []
        for i in range(len(df) - window_size + 1):
            window = df['pred_label'].iloc[i:i+window_size]
            counts = window.value_counts(normalize=True)
            proportions.append({k: counts.get(k, 0) for k in ['Focused', 'Neutral', 'Unfocused']})
        prop_df = pd.DataFrame(proportions)
        for label in ['Focused', 'Neutral', 'Unfocused']:
            axes[idx, 2].plot(prop_df.index, prop_df.get(label, 0), label=label)
        axes[idx, 2].set_xlabel(f'Window Start (size={window_size})')
        axes[idx, 2].set_ylabel('Proportion')
        axes[idx, 2].set_title(f'{file_label}: Class Proportions')
        axes[idx, 2].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Combined comparison plot saved as {save_path}")

# --- Helper: Run regression and save results if not present ---
def run_and_save_regression_results():
    from svm_mi_classification import build_mi_regression_dataset_with_eda, load_eda_csv, normalize_eda
    eda_raw = load_eda_csv(EDA_FILE)
    if eda_raw is None or len(eda_raw) == 0:
        eda_data = None
        eda_fs = FS
    else:
        eda_data = normalize_eda(eda_raw)
        eda_fs = FS
    X, y = build_mi_regression_dataset_with_eda(DATA_DIR, eda_data, eda_fs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    reg = SVR(kernel='rbf', C=1, gamma='scale')
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(X_test.shape[1])])
    df['true_mi'] = y_test
    df['pred_mi'] = y_pred
    df.to_csv('svm_mi_regression_results.csv', index=False)
    with open('svm_mi_regression_metrics.txt', 'w') as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}\n")
    print(f"Regression results and metrics saved. MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
    return df, mse, mae, r2

# --- Plot regression results and metrics ---
def plot_regression_statistics():
    # Ensure plt is imported in this scope
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as backend_pdf
    if not os.path.exists('svm_mi_regression_results.csv'):
        df, mse, mae, r2 = run_and_save_regression_results()
    else:
        df = pd.read_csv('svm_mi_regression_results.csv')
        with open('svm_mi_regression_metrics.txt') as f:
            lines = f.readlines()
            mse = float(lines[0].split(':')[1])
            mae = float(lines[1].split(':')[1])
            r2 = float(lines[2].split(':')[1])
    # Plot true vs predicted MI over time
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, df['true_mi'], label='True MI', alpha=0.7)
    plt.plot(df.index, df['pred_mi'], label='Predicted MI', alpha=0.7)
    plt.xlabel('Window Index (Time)')
    plt.ylabel('Mindfulness Index (MI)')
    plt.title('True vs Predicted MI Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('svm_mi_regression_mi_over_time.png')
    # Scatter plot true vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(df['true_mi'], df['pred_mi'], alpha=0.5)
    plt.xlabel('True MI')
    plt.ylabel('Predicted MI')
    plt.title('True vs Predicted MI (Scatter)')
    plt.tight_layout()
    plt.savefig('svm_mi_regression_scatter.png')
    # Metrics bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(['MSE', 'MAE', 'R^2'], [mse, mae, r2], color=['red', 'orange', 'green'])
    plt.title('Regression Metrics')
    plt.tight_layout()
    plt.savefig('svm_mi_regression_metrics.png')
    # Save all to PDF
    pdf = backend_pdf.PdfPages('svm_mi_regression_report.pdf')
    for fname in ['svm_mi_regression_mi_over_time.png', 'svm_mi_regression_scatter.png', 'svm_mi_regression_metrics.png']:
        img = plt.imread(fname)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig()
        plt.close()
    # Add pipeline explanation as a text page
    fig = plt.figure(figsize=(8.5, 11))
    text = (
        "Mindfulness Index Regression Pipeline Using SVM\n\n"
        "Rationale:\n"
        "The Mindfulness Index (MI) is a continuous value between 0 and 1, representing the degree of mindfulness in each EEG window. "
        "Instead of classifying windows into discrete states, we use regression to predict the actual MI value. "
        "This approach preserves the granularity of the MI and allows for more nuanced feedback and analysis.\n\n"
        "Pipeline Steps:\n"
        "1. Feature Extraction: For each EEG window, extract relevant features.\n"
        "2. MI Calculation: Compute the MI for each window, resulting in a continuous target value between 0 and 1.\n"
        "3. Data Preparation: Construct a dataset where each row contains EEG features and the corresponding MI value.\n"
        "4. Model Training: Use Support Vector Regression (SVR) to model the relationship between EEG features and MI.\n"
        "5. Prediction & Evaluation: Predict MI values for the test set. Evaluate performance using regression metrics.\n"
        "6. Visualization: Plot true vs. predicted MI over time, scatter plot, and regression metrics.\n\n"
        "Advantages:\n"
        "- Maintains the full information content of the MI.\n"
        "- Enables subtle, continuous feedback for neurofeedback or research.\n"
        "- Regression metrics provide a direct measure of prediction quality."
    )
    fig.text(0.1, 0.9, text, va='top', wrap=True, fontsize=10)
    pdf.savefig(fig)
    pdf.close()
    print("Regression report PDF generated: svm_mi_regression_report.pdf")

if __name__ == "__main__":
    plot_statistics()
    plot_comparison_all_files()
    plot_regression_statistics()
