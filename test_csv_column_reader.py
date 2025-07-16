import os
import glob
import pandas as pd

# Directory containing your CSV files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_implementation', 'logs')

files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

print(f"Found {len(files)} CSV files.")

for file in files:
    try:
        df = pd.read_csv(file)
        print(f"\nFile: {os.path.basename(file)}")
        print(f"Columns: {list(df.columns)}")
        print(f"First 2 rows:\n{df.head(2)}")
    except Exception as e:
        print(f"Error reading {file}: {e}")
