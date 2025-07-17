import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_implementation/final_logs')

# Parameters
SMOOTH_WINDOW = 20
SCALING = 'zscore'
MIN_REQUIRED = 240
COLUMN = 'mi'

# Collect all MI session files
mi_files = [f for f in glob.glob(os.path.join(LOG_DIR, '*_mi_session_*.csv'))]
series_list = []
min_len = None

for file in mi_files:
    try:
        df = pd.read_csv(file)
        if COLUMN not in df.columns:
            continue
        y = df[COLUMN]
        # Smoothing
        y = y.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()
        # Z-score scaling
        y = (y - y.mean()) / y.std() if y.std() > 0 else y - y.mean()
        # Only use files with enough samples
        if len(y) < MIN_REQUIRED:
            continue
        y = y.iloc[:MIN_REQUIRED].reset_index(drop=True)
        series_list.append(y)
        if min_len is None or len(y) < min_len:
            min_len = len(y)
    except Exception:
        continue

if not series_list:
    print("No valid MI session files found with enough samples.")
    exit(1)

# Truncate all to min_len
truncated_series = [s.iloc[:min_len] for s in series_list]
arr = np.vstack([s.values for s in truncated_series])
median = np.nanmedian(arr, axis=0)

# Peak detection
peaks, properties = find_peaks(median, height=1.5)

# Plot
plt.figure(figsize=(12,6))
plt.plot(median, label='Median MI (z-score, MA=20)')
plt.scatter(peaks, median[peaks], color='red', s=60, label='Peaks')
plt.title('Median MI (z-score, MA=20) with Peaks')
plt.xlabel('Time (s)')
plt.ylabel('MI (z-score)')
plt.legend()
plt.tight_layout()

# Summary interpretation
summary = f"""
Statistical Interpretation:
- All MI session files with >= {MIN_REQUIRED} samples were included.
- Each series was smoothed (moving average window=20) and z-score normalized.
- The median MI curve shows group-level dynamics.
- Peaks (z > 1.5) indicate shared moments of heightened MI.
- Number of detected peaks: {len(peaks)}
- Peak times (s): {peaks.tolist()}
- Peak heights: {median[peaks].round(2).tolist()}
"""

# Print to PDF
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pdf_path = f"mi_median_peaks_report_{timestamp}.pdf"
with PdfPages(pdf_path) as pdf:
    plt.savefig(pdf, format='pdf')
    plt.close()
    # Add summary page
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.01, 0.99, f"MI Median Peaks Report\nGenerated: {timestamp}", fontsize=14, va='top')
    plt.text(0.01, 0.95, summary, fontsize=12, va='top')
    pdf.savefig()
    plt.close()

print(f"Report saved to {pdf_path}")
