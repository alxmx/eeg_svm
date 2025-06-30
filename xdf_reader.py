"""
XDF File Reader and LSL Stream Analyzer
======================================

This script reads .xdf files and provides detailed information about:
- Header information
- LSL streams present in the file
- Channel details for each stream
- Data samples overview

Usage:
------
python xdf_reader.py path/to/your/file.xdf

Requirements:
-------------
pip install pyxdf numpy pandas

"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import pyxdf
except ImportError:
    print("ERROR: pyxdf not installed. Install with: pip install pyxdf")
    sys.exit(1)

def analyze_xdf_file(xdf_path):
    """
    Analyze an XDF file and extract comprehensive information about LSL streams
    
    Parameters:
    -----------
    xdf_path : str
        Path to the .xdf file
    
    Returns:
    --------
    dict : Analysis results
    """
    if not os.path.exists(xdf_path):
        print(f"ERROR: File not found: {xdf_path}")
        return None
    
    print(f"Loading XDF file: {xdf_path}")
    print("This may take a moment for large files...\n")
    
    try:
        # Load XDF file
        data, header = pyxdf.load_xdf(xdf_path)
        
        # File information
        file_size = os.path.getsize(xdf_path) / (1024 * 1024)  # MB
        print(f"📁 FILE INFORMATION")
        print(f"   File: {os.path.basename(xdf_path)}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Streams found: {len(data)}")
        
        # Header information
        print(f"\n📋 XDF HEADER INFORMATION")
        if header:
            for key, value in header.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"     {sub_key}: {sub_value}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("   No header information found")
        
        # Stream analysis
        print(f"\n🌊 LSL STREAMS ANALYSIS")
        print("=" * 80)
        
        stream_summary = []
        
        for i, stream in enumerate(data):
            stream_data = stream['time_series']
            stream_info = stream['info']
            timestamps = stream['time_stamps']
            
            # Basic stream info
            stream_name = stream_info.get('name', [{'#text': 'Unknown'}])[0]['#text']
            stream_type = stream_info.get('type', [{'#text': 'Unknown'}])[0]['#text']
            stream_id = stream_info.get('uid', [{'#text': 'Unknown'}])[0]['#text']
            
            # Channel information
            channels = stream_info.get('desc', [{}])[0].get('channels', [{}])
            if isinstance(channels, list) and len(channels) > 0:
                channel_info = channels[0].get('channel', [])
                if not isinstance(channel_info, list):
                    channel_info = [channel_info] if channel_info else []
                n_channels = len(channel_info)
                channel_names = [ch.get('label', [{'#text': f'Ch{j}'}])[0]['#text'] 
                               for j, ch in enumerate(channel_info)]
            else:
                n_channels = stream_data.shape[1] if len(stream_data.shape) > 1 else 1
                channel_names = [f'Channel_{j}' for j in range(n_channels)]
            
            # Sampling rate
            nominal_srate = stream_info.get('nominal_srate', [{'#text': 'Unknown'}])[0]['#text']
            try:
                nominal_srate = float(nominal_srate)
            except:
                nominal_srate = 'Unknown'
            
            # Calculate actual sampling rate from timestamps
            if len(timestamps) > 1:
                actual_srate = 1.0 / np.median(np.diff(timestamps))
            else:
                actual_srate = 'N/A'
            
            # Data statistics
            n_samples = len(stream_data)
            duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
            
            # Data range
            if n_samples > 0 and len(stream_data.shape) > 1:
                data_min = np.min(stream_data, axis=0)
                data_max = np.max(stream_data, axis=0)
                data_mean = np.mean(stream_data, axis=0)
                data_std = np.std(stream_data, axis=0)
            else:
                data_min = data_max = data_mean = data_std = 'N/A'
            
            # Print stream details
            print(f"\nSTREAM {i+1}: {stream_name}")
            print(f"   Type: {stream_type}")
            print(f"   Stream ID: {stream_id}")
            print(f"   Channels: {n_channels}")
            print(f"   Nominal Sample Rate: {nominal_srate} Hz")
            print(f"   Actual Sample Rate: {actual_srate:.2f} Hz" if isinstance(actual_srate, float) else f"   Actual Sample Rate: {actual_srate}")
            print(f"   Samples: {n_samples:,}")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Time Range: {timestamps[0]:.2f} - {timestamps[-1]:.2f}")
            
            # Channel details
            print(f"   Channel Names: {channel_names[:5]}{'...' if len(channel_names) > 5 else ''}")
            
            # Data statistics (first few channels)
            if isinstance(data_min, np.ndarray) and len(data_min) > 0:
                print(f"   Data Range (first 3 channels):")
                for ch_idx in range(min(3, len(data_min))):
                    ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'Ch{ch_idx}'
                    print(f"     {ch_name}: [{data_min[ch_idx]:.3f}, {data_max[ch_idx]:.3f}] (mean: {data_mean[ch_idx]:.3f}, std: {data_std[ch_idx]:.3f})")
            
            # Store summary
            stream_summary.append({
                'stream_id': i+1,
                'name': stream_name,
                'type': stream_type,
                'uid': stream_id,
                'channels': n_channels,
                'nominal_srate': nominal_srate,
                'actual_srate': actual_srate if isinstance(actual_srate, float) else np.nan,
                'samples': n_samples,
                'duration': duration,
                'start_time': timestamps[0] if len(timestamps) > 0 else np.nan,
                'end_time': timestamps[-1] if len(timestamps) > 0 else np.nan
            })
        
        # Summary table
        print(f"\n📊 SUMMARY TABLE")
        print("=" * 80)
        summary_df = pd.DataFrame(stream_summary)
        print(summary_df.to_string(index=False))
        
        # Potential LSL stream identification
        print(f"\n🔍 POTENTIAL STREAM IDENTIFICATION")
        print("=" * 80)
        
        eeg_streams = [s for s in stream_summary if 'eeg' in s['type'].lower() or 'eeg' in s['name'].lower()]
        eda_streams = [s for s in stream_summary if 'eda' in s['type'].lower() or 'eda' in s['name'].lower() or 'gsr' in s['name'].lower()]
        marker_streams = [s for s in stream_summary if 'marker' in s['type'].lower() or 'trigger' in s['type'].lower()]
        mi_streams = [s for s in stream_summary if 'mindfulness' in s['name'].lower() or 'mi' in s['name'].lower()]
        
        if eeg_streams:
            print(f"EEG Streams: {[f\"{s['name']} (ID: {s['stream_id']})\" for s in eeg_streams]}")
        if eda_streams:
            print(f"EDA/GSR Streams: {[f\"{s['name']} (ID: {s['stream_id']})\" for s in eda_streams]}")
        if marker_streams:
            print(f"Marker/Trigger Streams: {[f\"{s['name']} (ID: {s['stream_id']})\" for s in marker_streams]}")
        if mi_streams:
            print(f"Mindfulness Index Streams: {[f\"{s['name']} (ID: {s['stream_id']})\" for s in mi_streams]}")
        
        # Save summary to CSV
        csv_path = xdf_path.replace('.xdf', '_stream_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\n💾 Stream summary saved to: {csv_path}")
        
        return {
            'file_info': {
                'path': xdf_path,
                'size_mb': file_size,
                'n_streams': len(data)
            },
            'header': header,
            'streams': data,
            'summary': stream_summary
        }
        
    except Exception as e:
        print(f"ERROR reading XDF file: {e}")
        return None

def quick_stream_check(xdf_path, stream_id=None):
    """
    Quick check of specific stream data
    
    Parameters:
    -----------
    xdf_path : str
        Path to XDF file
    stream_id : int, optional
        Specific stream to analyze (1-based index)
    """
    print(f"\n🔬 QUICK STREAM DATA CHECK")
    print("=" * 50)
    
    try:
        data, header = pyxdf.load_xdf(xdf_path)
        
        if stream_id is None:
            print("Available streams:")
            for i, stream in enumerate(data):
                stream_name = stream['info'].get('name', [{'#text': 'Unknown'}])[0]['#text']
                print(f"  {i+1}: {stream_name}")
            return
        
        if stream_id < 1 or stream_id > len(data):
            print(f"ERROR: Stream ID {stream_id} not found. Available: 1-{len(data)}")
            return
        
        stream = data[stream_id - 1]
        stream_data = stream['time_series']
        timestamps = stream['time_stamps']
        
        print(f"Stream {stream_id} - First 5 samples:")
        print(f"Timestamps: {timestamps[:5]}")
        print(f"Data shape: {stream_data.shape}")
        if len(stream_data.shape) > 1:
            print("Data preview:")
            for i in range(min(5, stream_data.shape[0])):
                print(f"  Sample {i}: {stream_data[i]}")
        else:
            print(f"Data preview: {stream_data[:5]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python xdf_reader.py <path_to_xdf_file> [stream_id]")
        print("\nExample:")
        print("  python xdf_reader.py recording.xdf        # Analyze all streams")
        print("  python xdf_reader.py recording.xdf 1      # Quick check stream 1")
        return
    
    xdf_path = sys.argv[1]
    
    # Full analysis
    results = analyze_xdf_file(xdf_path)
    
    # Quick stream check if stream ID provided
    if len(sys.argv) > 2:
        try:
            stream_id = int(sys.argv[2])
            quick_stream_check(xdf_path, stream_id)
        except ValueError:
            print(f"ERROR: Invalid stream ID '{sys.argv[2]}'. Must be a number.")

if __name__ == "__main__":
    main()
