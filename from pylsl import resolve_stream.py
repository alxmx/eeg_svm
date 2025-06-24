from pylsl import resolve_stream

print("Waiting for LSL streams...")
streams = resolve_stream()
for idx, s in enumerate(streams):
    print(f"[{idx}] Name: {s.name()}, Type: {s.type()}, Channels: {s.channel_count()}, Source ID: {s.source_id()}")