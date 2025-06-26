#!/usr/bin/env python3
"""
LSL MI Values Receiver Test
This script connects to the MI LSL streams and displays the transmitted values
to confirm what MI values are being sent by the realtime_mi_lsl.py script.
"""

import time
from pylsl import StreamInlet, resolve_stream

def test_mi_lsl_streams():
    """Connect to and monitor all MI LSL streams"""
    
    print("🔍 Looking for MI LSL streams...")
    print("=" * 60)
    
    # Try to resolve each MI stream
    streams_to_monitor = {
        'MindfulnessIndex': None,
        'RawMindfulnessIndex': None,  
        'EmotionalMindfulnessIndex': None
    }
    
    # Resolve streams
    for stream_name in streams_to_monitor.keys():
        try:
            print(f"🔎 Resolving {stream_name}...")
            streams = resolve_stream('name', stream_name, timeout=2.0)
            if streams:
                inlet = StreamInlet(streams[0])
                streams_to_monitor[stream_name] = inlet
                print(f"✅ Connected to {stream_name}")
                print(f"   - Type: {streams[0].type()}")
                print(f"   - Channels: {streams[0].channel_count()}")
                print(f"   - Sample Rate: {streams[0].nominal_srate()} Hz")
                print(f"   - Source ID: {streams[0].source_id()}")
            else:
                print(f"❌ {stream_name} not found")
        except Exception as e:
            print(f"❌ Error resolving {stream_name}: {e}")
        print()
    
    # Check if any streams were found
    connected_streams = {k: v for k, v in streams_to_monitor.items() if v is not None}
    
    if not connected_streams:
        print("❌ No MI streams found. Make sure realtime_mi_lsl.py is running.")
        return
    
    print(f"📡 Monitoring {len(connected_streams)} MI streams:")
    for name in connected_streams.keys():
        print(f"   - {name}")
    print()
    print("📊 Real-time MI values (press Ctrl+C to stop):")
    print("-" * 80)
    print(f"{'Time':<12} {'MI (SVR)':<10} {'Raw MI':<10} {'EMI':<10}")
    print("-" * 80)
    
    try:
        while True:
            current_time = time.strftime('%H:%M:%S')
            mi_val = "N/A"
            raw_mi_val = "N/A" 
            emi_val = "N/A"
            
            # Pull latest samples from each stream
            if 'MindfulnessIndex' in connected_streams:
                inlet = connected_streams['MindfulnessIndex']
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    mi_val = f"{sample[0]:.3f}"
            
            if 'RawMindfulnessIndex' in connected_streams:
                inlet = connected_streams['RawMindfulnessIndex']
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    raw_mi_val = f"{sample[0]:.3f}"
            
            if 'EmotionalMindfulnessIndex' in connected_streams:
                inlet = connected_streams['EmotionalMindfulnessIndex']
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    emi_val = f"{sample[0]:.3f}"
            
            # Print current values
            print(f"{current_time:<12} {mi_val:<10} {raw_mi_val:<10} {emi_val:<10}")
            time.sleep(0.5)  # Update every 0.5 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")
    
    print("\n✅ MI LSL monitoring complete")

if __name__ == "__main__":
    print("🧠 MI LSL Streams Monitor")
    print("This script monitors the Mindfulness Index LSL streams")
    print("Make sure realtime_mi_lsl.py is running first!\n")
    
    test_mi_lsl_streams()
