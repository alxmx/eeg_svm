import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# --- CONFIG ---
FEATURE_ORDER = ['theta_fz', 'alpha_po', 'faa', 'beta_frontal', 'eda_norm']

# --- Robust quantile normalization ---
def robust_quantile_normalize(df, feature, q_low=0.05, q_high=0.95, out_min=0, out_max=10):
    ql = df[feature].quantile(q_low)
    qh = df[feature].quantile(q_high)
    def norm(x):
        return np.clip((x - ql) / (qh - ql), 0, 1) * (out_max - out_min) + out_min
    return df[feature].apply(norm)

# --- Adaptive weights based on Spearman correlation ---
def get_session_adaptive_weights(df, feature_names, mi_col='mi', min_weight=0.1, max_weight=0.5):
    corrs = []
    for feat in feature_names:
        if feat in df.columns:
            corr, _ = spearmanr(df[feat], df[mi_col], nan_policy='omit')
            corrs.append(corr)
        else:
            corrs.append(0)
    abs_corrs = np.abs(corrs)
    if abs_corrs.max() > 0:
        scaled = min_weight + (max_weight - min_weight) * (abs_corrs / abs_corrs.max())
    else:
        scaled = np.full_like(abs_corrs, min_weight)
    weights = np.sign(corrs) * scaled
    weights[np.abs(corrs) < 0.1] = 0
    return weights

# --- Universal MI calculation ---
def calculate_mi_universal(row, weights, offset=None):
    vals = np.array([row[f] for f in FEATURE_ORDER])
    mi_raw = np.dot(vals, weights)
    if offset is None:
        offset = np.dot(np.ones_like(weights)*5, weights)
    mi = 1 / (1 + np.exp(-(mi_raw - offset)))
    return np.clip(mi, 0, 1)

# --- Load session data ---
session_path = 'logs/alex_2241_mi_session_20250625_224604.csv'
df = pd.read_csv(session_path)

# --- Normalize features ---
for feat in FEATURE_ORDER:
    df[feat + '_norm'] = robust_quantile_normalize(df, feat)

# --- Compute adaptive weights ---
adaptive_weights = get_session_adaptive_weights(df, FEATURE_ORDER, mi_col='mi')
print('Adaptive weights:', dict(zip(FEATURE_ORDER, adaptive_weights)))

# --- Calculate MI using normalized features and adaptive weights ---
mi_universal = []
for _, row in df.iterrows():
    norm_row = {f: row[f + '_norm'] for f in FEATURE_ORDER}
    mi = calculate_mi_universal(norm_row, adaptive_weights)
    mi_universal.append(mi)
df['mi_universal'] = mi_universal

# --- Print summary ---
print('\nUniversal MI summary:')
print('Mean:', np.mean(mi_universal))
print('Std:', np.std(mi_universal))
print('Min:', np.min(mi_universal))
print('Max:', np.max(mi_universal))

# --- Compare to SVR MI ---
print('\nOriginal MI summary:')
print('Mean:', df['mi'].mean())
print('Std:', df['mi'].std())
print('Min:', df['mi'].min())
print('Max:', df['mi'].max())

# --- Print first 10 values for visual check ---
print('\nFirst 10 MI (SVR):', df['mi'].head(10).values)
print('First 10 MI (Universal):', df['mi_universal'].head(10).values)

# --- Save for further analysis ---
df.to_csv('logs/alex_2241_mi_session_20250625_224604_adaptive_universal.csv', index=False)
print('\nSaved: logs/alex_2241_mi_session_20250625_224604_adaptive_universal.csv')
