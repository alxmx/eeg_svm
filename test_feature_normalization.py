import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
# Replace with your actual file paths
calib_csv = 'user_configs/009_alex_test_baseline.csv'  # Calibration (relaxed)
session_csv = 'logs/009_alex_test_mi_session_20250625_205413.csv'  # Real session
feature_names = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']

# --- LOAD DATA ---
calib_df = pd.read_csv(calib_csv)
session_df = pd.read_csv(session_csv)

# --- GET CALIBRATION MEANS/STDS ---
calib_means = calib_df[feature_names].mean()
calib_stds = calib_df[feature_names].std().replace(0, 1e-6)  # Avoid div by zero

# --- Z-SCORE NORMALIZATION (using calibration stats) ---
def zscore_norm(df, means, stds):
    return (df[feature_names] - means) / stds

session_z = zscore_norm(session_df, calib_means, calib_stds)

# --- ROBUST QUANTILE NORMALIZATION (as in your pipeline) ---
robust_ranges = {
    'theta_fz': (1, 50),
    'alpha_po': (1, 50),
    'faa': (-2, 2),
    'beta_frontal': (1, 50),
    'eda_norm': (0.1, 20)
}
def robust_quantile_norm(row):
    normed = []
    for feat in feature_names:
        q5, q95 = robust_ranges[feat]
        val = 10 * (row[feat] - q5) / (q95 - q5)
        normed.append(np.clip(val, 0, 10))
    return pd.Series(normed, index=feature_names)
session_robust = session_df[feature_names].apply(robust_quantile_norm, axis=1)

# --- PLOT COMPARISON ---
plt.figure(figsize=(15, 10))
for i, feat in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    plt.plot(session_z[feat], label='Z-score (calib)', color='red', alpha=0.7)
    plt.plot(session_robust[feat], label='Robust Quantile', color='blue', alpha=0.7)
    plt.title(feat)
    plt.legend()
plt.tight_layout()
plt.savefig('feature_normalization_comparison.png')
print('Saved feature_normalization_comparison.png')

# --- PRINT STATS ---
print('Calibration means:')
print(calib_means)
print('Calibration stds:')
print(calib_stds)
print('Session z-score stats:')
print(session_z.describe())
print('Session robust quantile stats:')
print(session_robust.describe())
