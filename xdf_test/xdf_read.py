import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import csv

# Load XDF file
xdf_path = "xdf_recordings/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
data, header = pyxdf.load_xdf(xdf_path)

# Prepare summary table
summary = []

# Prepare for CSV output
csv_rows = []

plt.figure(figsize=(12, 8))
for idx, stream in enumerate(data):
    name = stream['info']['name'][0]
    stype = stream['info']['type'][0]
    n_channels = int(stream['info']['channel_count'][0])
    try:
        srate = float(stream['info']['nominal_srate'][0])
    except Exception:
        srate = 'NA'
    y = stream['time_series']
    t = stream['time_stamps']
    dtype = type(y)
    if isinstance(y, np.ndarray):
        shape = y.shape
        dtype_str = str(y.dtype)
        # Compute stats per channel
        minv = np.nanmin(y, axis=0)
        maxv = np.nanmax(y, axis=0)
        meanv = np.nanmean(y, axis=0)
        stdv = np.nanstd(y, axis=0)
        n_nan = np.isnan(y).sum(axis=0) if y.ndim > 1 else np.isnan(y).sum()
        # Print stats
        print(f"\nStream: {name} | Type: {stype} | Channels: {n_channels} | Rate: {srate} | Dtype: {dtype_str} | Shape: {shape}")
        print(f"  Min:   {minv}")
        print(f"  Max:   {maxv}")
        print(f"  Mean:  {meanv}")
        print(f"  Std:   {stdv}")
        print(f"  NaNs:  {n_nan}")
        summary.append([name, stype, n_channels, srate, dtype_str, shape])
        # For CSV: flatten stats to semicolon-separated if multi-channel
        def arr2str(arr):
            if isinstance(arr, np.ndarray) and arr.shape != ():
                return ';'.join([str(x) for x in arr])
            return str(arr)
        csv_rows.append([
            name, stype, n_channels, srate, dtype_str, shape,
            arr2str(minv), arr2str(maxv), arr2str(meanv), arr2str(stdv), arr2str(n_nan), ''
        ])
        # Plot all channels
        if y.ndim == 1:
            plt.plot(t, y, label=f"{name} ({stype})")
        else:
            for ch in range(y.shape[1]):
                plt.plot(t, y[:, ch], label=f"{name} ({stype}) ch{ch+1}")
    elif isinstance(y, list):
        # Marker stream
        print(f"\nStream: {name} | Type: {stype} | Channels: {n_channels} | Rate: {srate} | Dtype: list | Shape: {len(y)}")
        print(f"  Markers: {len(y)} events")
        summary.append([name, stype, n_channels, srate, 'list', len(y)])
        csv_rows.append([
            name, stype, n_channels, srate, 'list', len(y), '', '', '', '', '', len(y)
        ])
        for timestamp, marker in zip(t, y):
            plt.axvline(x=timestamp, color='k', linestyle='--', alpha=0.5)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    else:
        print(f"Unknown stream format for {name}")

# Print summary table
print("\n=== Stream Summary Table ===")
print(f"{'Name':20} {'Type':12} {'Ch':>3} {'Rate':>6} {'Dtype':>10} {'Shape':>15}")
for row in summary:
    print(f"{row[0]:20} {row[1]:12} {row[2]:3} {str(row[3]):>6} {row[4]:>10} {str(row[5]):>15}")

# Write CSV summary for machine use
csv_path = "xdf_analysis_summary.csv"
csv_header = [
    'stream_name','stream_type','channels','rate','dtype','shape',
    'min','max','mean','std','n_nans','marker_count'
]
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    for row in csv_rows:
        writer.writerow(row)
print(f"\n[INFO] Stream analysis summary written to {csv_path}")

plt.title("All Streams (channels labeled)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()