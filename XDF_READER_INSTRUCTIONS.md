# XDF File Reader - Usage Instructions

## Overview
This script analyzes XDF files to extract LSL stream information, channel details, and data overview.

## Installation
```bash
pip install pyxdf numpy pandas
```

## Usage

### 1. Basic Analysis (Full Report)
```bash
python xdf_reader.py your_recording.xdf
```

### 2. Quick Stream Check
```bash
python xdf_reader.py your_recording.xdf 1    # Check stream 1 data samples
```

### 3. Python Script Usage
```python
from xdf_reader import analyze_xdf_file, quick_stream_check

# Full analysis
results = analyze_xdf_file('recording.xdf')

# Quick stream check
quick_stream_check('recording.xdf', stream_id=1)
```

## Output Information

### 📁 File Information
- File size and path
- Number of streams detected

### 📋 XDF Header Information  
- Recording metadata
- Software version info
- Recording parameters

### 🌊 LSL Streams Analysis
For each stream:
- **Stream name and type**
- **Stream ID (UID)**
- **Channel count and names**
- **Sampling rates** (nominal vs actual)
- **Data duration and sample count**
- **Data range and statistics**

### 🔍 Stream Type Identification
Automatically identifies:
- **EEG streams** (brain activity)
- **EDA/GSR streams** (skin conductance) 
- **Marker streams** (triggers/events)
- **Mindfulness Index streams** (MI outputs)

### 📊 Summary Table
CSV export with all stream parameters for easy comparison

## Expected Output Example
```
📁 FILE INFORMATION
   File: my_session.xdf
   Size: 45.23 MB
   Streams found: 4

🌊 LSL STREAMS ANALYSIS
================================================================================

STREAM 1: UnicornRecorderLSLStream
   Type: EEG
   Stream ID: d8f7e2c1-4b9a-4d3e-8f1a-2c3e4f5a6b7c
   Channels: 8
   Nominal Sample Rate: 250.0 Hz
   Actual Sample Rate: 249.98 Hz
   Samples: 15,000
   Duration: 60.01 seconds
   Channel Names: ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Oz', 'PO8', 'AccX']
   Data Range (first 3 channels):
     Fz: [-12.345, 18.765] (mean: 2.134, std: 4.567)
     C3: [-15.234, 21.876] (mean: 1.897, std: 5.123)
     Cz: [-13.456, 19.234] (mean: 2.456, std: 4.789)

STREAM 2: OpenSignals
   Type: EDA
   Stream ID: a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6
   Channels: 2
   Nominal Sample Rate: 1000.0 Hz
   Actual Sample Rate: 999.87 Hz
   Samples: 60,000
   Duration: 60.01 seconds
   Channel Names: ['Channel_0', 'Channel_1']
   Data Range (first 3 channels):
     Channel_0: [0.000, 4095.000] (mean: 2047.500, std: 567.123)
     Channel_1: [1200.000, 1850.000] (mean: 1456.789, std: 89.456)

🔍 POTENTIAL STREAM IDENTIFICATION
================================================================================
EEG Streams: ['UnicornRecorderLSLStream (ID: 1)']
EDA/GSR Streams: ['OpenSignals (ID: 2)']
```

## Troubleshooting

### Common Issues:
1. **"pyxdf not installed"** → Run: `pip install pyxdf`
2. **"File not found"** → Check file path and extension (.xdf)
3. **Large files slow to load** → This is normal for XDF files (can take 1-2 minutes)

### File Size Guidelines:
- **< 50 MB**: Fast loading (few seconds)
- **50-200 MB**: Medium loading (30-60 seconds)  
- **> 200 MB**: Slow loading (1-3 minutes)

## Next Steps
After analyzing your XDF file, you can:
1. **Identify MI streams** for analysis
2. **Extract specific channel data** for processing
3. **Verify sampling rates** match expectations
4. **Check data quality** (range, artifacts, gaps)
5. **Plan data preprocessing** pipeline
