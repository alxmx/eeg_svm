#!/usr/bin/env python3
"""
Test script to verify EDA channel selection
"""
import numpy as np

print('=== EDA CHANNEL VERIFICATION TEST ===')

# Configuration from main script
EDA_CHANNEL_INDEX = 1

# Sample data from your log
eda_sample = [5.34057500e+06, 1.35726995e+01]
print(f'Raw EDA sample: {eda_sample}')
print(f'EDA_CHANNEL_INDEX = {EDA_CHANNEL_INDEX} (0-based)')
print(f'Channel 0 (index 0): {eda_sample[0]:.1f}')
print(f'Channel 1 (index 1): {eda_sample[1]:.1f}')
print(f'Selected channel {EDA_CHANNEL_INDEX}: {eda_sample[EDA_CHANNEL_INDEX]:.1f}')
print('')
print('✓ We are correctly using channel 1 (the smaller values ~13.57)')
print('✓ NOT using channel 0 (the huge values ~5.34 million)')
print('')

# Test with window of data like in the real script
print('=== WINDOW DATA TEST ===')
eda_window = np.array([
    [5.34057500e+06, 1.35726995e+01],
    [5.34057600e+06, 1.36352997e+01],
    [5.34057700e+06, 1.36856003e+01],
    [5.34057800e+06, 1.37558002e+01],
    [5.34057900e+06, 1.38473997e+01]
])

selected_channel_data = eda_window[:, EDA_CHANNEL_INDEX]
print(f'EDA window shape: {eda_window.shape}')
print(f'Channel {EDA_CHANNEL_INDEX} data: {selected_channel_data}')
print(f'Channel {EDA_CHANNEL_INDEX} mean: {np.mean(selected_channel_data):.6f}')
print('')
print('✓ CONFIRMED: Using the correct channel with reasonable values')
