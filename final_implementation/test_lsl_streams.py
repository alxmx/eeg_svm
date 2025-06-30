#!/usr/bin/env python3
"""
Quick test for LSL stream detection and selection
"""
try:
    from pylsl import resolve_streams, resolve_byprop
    print('[TEST] LSL imports successful')
    
    print('[TEST] Looking for any LSL streams...')
    streams = resolve_streams(timeout=2.0)
    if streams:
        print(f'[FOUND] {len(streams)} streams detected:')
        for i, stream in enumerate(streams):
            print(f'  {i+1}. {stream.name()} (type: {stream.type()}, channels: {stream.channel_count()})')
    else:
        print('[INFO] No LSL streams currently running (this is expected if no devices are streaming)')
    
    print('[TEST] Testing EEG stream detection...')
    eeg_streams = resolve_byprop('type', 'EEG', timeout=2.0)
    if eeg_streams:
        print(f'[FOUND] {len(eeg_streams)} EEG streams:')
        for i, stream in enumerate(eeg_streams):
            print(f'  {i+1}. {stream.name()} ({stream.channel_count()} channels)')
    else:
        print('[INFO] No EEG streams found')
    
    print('[TEST] Testing EDA stream detection...')
    eda_streams = resolve_byprop('type', 'EDA', timeout=2.0)
    if eda_streams:
        print(f'[FOUND] {len(eda_streams)} EDA streams:')
        for i, stream in enumerate(eda_streams):
            print(f'  {i+1}. {stream.name()} ({stream.channel_count()} channels)')
    else:
        print('[INFO] No EDA streams found')
        
    print('[SUCCESS] LSL stream detection is working properly')
    print('[INFO] Stream selection will work in console when real streams are available')
        
except ImportError as e:
    print(f'[ERROR] LSL import failed: {e}')
    print('[HINT] Make sure pylsl is installed: pip install pylsl')
except Exception as e:
    print(f'[ERROR] Test failed: {e}')
    import traceback
    traceback.print_exc()
