import os
import pandas as pd
import csv
import matplotlib.pyplot as plt

# User settings
TARGET_DIR = "data/eda_data/files"  # Change this to the folder you want to analyze
OUTPUT_CSV = "EDA_file_analysis_summary.csv"
SAMPLE_LINES = 5
DEFAULT_SAMPLING_RATE = 250  # Hz, change if your files use a different rate


def analyze_file(filepath):
    info = {
        'filename': os.path.basename(filepath),
        'path': filepath,
        'extension': os.path.splitext(filepath)[1].lower(),
        'size_kb': round(os.path.getsize(filepath) / 1024, 2),
        'n_lines': None,
        'n_columns': None,
        'has_header': None,
        'sample_lines': [],
        'estimated_duration_sec': None,
    }
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [next(f) for _ in range(SAMPLE_LINES)]
            info['sample_lines'] = [l.strip() for l in lines]
    except Exception:
        pass
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            info['n_lines'] = sum(1 for _ in f)
    except Exception:
        info['n_lines'] = None
    # CSV/TXT structure
    if info['extension'] in ['.csv', '.txt']:
        try:
            df = pd.read_csv(filepath, nrows=10)
            info['n_columns'] = len(df.columns)
            info['has_header'] = not all(str(c).startswith('Unnamed') for c in df.columns)
            # Estimate duration if time/sample columns present
            if info['n_lines'] and info['n_columns']:
                n_samples = info['n_lines'] - (1 if info['has_header'] else 0)
                info['estimated_duration_sec'] = round(n_samples / DEFAULT_SAMPLING_RATE, 2)
        except Exception:
            info['n_columns'] = None
            info['has_header'] = None
    return info

def walk_and_analyze(target_dir):
    summary = []
    for root, dirs, files in os.walk(target_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            info = analyze_file(fpath)
            summary.append(info)
    return summary

def print_and_save_summary(summary, output_csv):
    df = pd.DataFrame(summary)
    print(df[['filename', 'extension', 'size_kb', 'n_lines', 'n_columns', 'has_header', 'estimated_duration_sec']])
    df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to {output_csv}")
    # Optionally print sample lines for each file
    for entry in summary:
        print(f"\nFile: {entry['filename']}")
        print("Sample lines:")
        for line in entry['sample_lines']:
            print(f"  {line}")

def plot_frequency_over_time(summary):
    # Only .csv and .txt files
    filtered = [f for f in summary if f['extension'] in ['.csv', '.txt'] and f['estimated_duration_sec'] is not None]
    if not filtered:
        print("No .csv or .txt files with estimated duration found.")
        return
    # Sort by filename for consistent color
    filtered = sorted(filtered, key=lambda x: x['filename'])
    plt.figure(figsize=(12, 6))
    for entry in filtered:
        try:
            # For OpenSignals .txt, skip header lines
            if entry['extension'] == '.txt':
                with open(entry['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    data_lines = []
                    header_ended = False
                    for line in f:
                        if not header_ended:
                            if line.strip().startswith('# EndOfHeader'):
                                header_ended = True
                            continue
                        if line.strip() == '':
                            continue
                        data_lines.append(line)
                # Now read the data lines as a DataFrame
                if data_lines:
                    import io
                    df = pd.read_csv(io.StringIO(''.join(data_lines)), delim_whitespace=True, header=None)
                    times = df.iloc[:, 0].astype(float)
                    plt.plot(times, range(len(times)), label=entry['filename'])
            else:
                # For CSV, assume no header lines to skip
                if entry['has_header']:
                    df = pd.read_csv(entry['path'], usecols=[0])
                else:
                    df = pd.read_csv(entry['path'], header=None, usecols=[0])
                times = df.iloc[:, 0].astype(float)
                plt.plot(times, range(len(times)), label=entry['filename'])
        except Exception as e:
            print(f"Could not plot {entry['filename']}: {e}")
    plt.xlabel('Time (from first column)')
    plt.ylabel('Sample Index')
    plt.title('Frequency Plot Over Time for All .csv and .txt Files')
    plt.legend(fontsize=8)
    plt.tight_layout()
    # Save plot as image in the same folder as the script
    output_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frequency_plot_over_time.png')
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")
    plt.show()

def plot_eda_phasic_activity(summary):
    # Only .csv and .txt files
    filtered = [f for f in summary if f['extension'] in ['.csv', '.txt'] and f['estimated_duration_sec'] is not None]
    if not filtered:
        print("No .csv or .txt files with estimated duration found.")
        return
    filtered = sorted(filtered, key=lambda x: x['filename'])
    plt.figure(figsize=(14, 7))
    for entry in filtered:
        try:
            if entry['extension'] == '.txt':
                with open(entry['path'], 'r', encoding='utf-8', errors='ignore') as f:
                    data_lines = []
                    header_ended = False
                    for line in f:
                        if not header_ended:
                            if line.strip().startswith('# EndOfHeader'):
                                header_ended = True
                            continue
                        if line.strip() == '':
                            continue
                        data_lines.append(line)
                if data_lines:
                    import io
                    df = pd.read_csv(io.StringIO(''.join(data_lines)), delim_whitespace=True, header=None)
                    times = df.iloc[:, 0].astype(float)
                    eda = df.iloc[:, -1].astype(float)
                    plt.plot(times, eda, label=entry['filename'])
            else:
                if entry['has_header']:
                    df = pd.read_csv(entry['path'])
                else:
                    df = pd.read_csv(entry['path'], header=None)
                times = df.iloc[:, 0].astype(float)
                eda = df.iloc[:, -1].astype(float)
                plt.plot(times, eda, label=entry['filename'])
        except Exception as e:
            print(f"Could not plot EDA for {entry['filename']}: {e}")
    plt.xlabel('Time (from first column)')
    plt.ylabel('EDA (phasic activity, last column)')
    plt.title('Comparative EDA Phasic Activity Over Time')
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eda_phasic_activity_comparison.png')
    plt.savefig(output_img)
    print(f"EDA phasic activity plot saved to {output_img}")
    plt.show()

def main():
    summary = walk_and_analyze(TARGET_DIR)
    print_and_save_summary(summary, OUTPUT_CSV)
    plot_frequency_over_time(summary)
    plot_eda_phasic_activity(summary)

if __name__ == "__main__":
    main()
