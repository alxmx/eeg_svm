"""
MI Pipeline Dashboard - Simple Web UI for Sensor/Report/File Status

Run this script to launch a local web server (http://localhost:8050) that shows:
- Online/offline status of available LSL sensors (EEG, EDA, UnityMarkers)
- List of available report/log/model files, with summary info
- Report settings and options
- Visual feedback for errors, missing files, or configuration issues

Requirements:
    pip install dash dash-bootstrap-components pylsl pandas

Usage:
    python mi_pipeline_dashboard.py
"""
import os
import glob
import pandas as pd
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from pylsl import resolve_streams
from dash.dependencies import Input, Output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'user_configs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- Helper functions ---
def get_lsl_status():
    status = []
    try:
        streams = resolve_streams()
        for s in streams:
            status.append({
                'Name': s.name(),
                'Type': s.type(),
                'Channels': s.channel_count(),
                'Source ID': s.source_id()
            })
    except Exception as e:
        status.append({'Name': 'ERROR', 'Type': str(e), 'Channels': '', 'Source ID': ''})
    return status

def get_file_table(folder, pattern, columns=None, summary_func=None):
    files = glob.glob(os.path.join(folder, pattern))
    rows = []
    for f in files:
        row = {'File': os.path.basename(f), 'Size (KB)': round(os.path.getsize(f)/1024, 1)}
        if summary_func:
            try:
                row.update(summary_func(f))
            except Exception as e:
                row['Summary'] = f'Error: {e}'
        rows.append(row)
    if columns:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)

def mi_session_summary(f):
    df = pd.read_csv(f)
    if df.empty or 'mi' not in df.columns:
        return {'Summary': 'Empty/invalid'}
    return {
        'n_samples': len(df),
        'mi_mean': round(df['mi'].mean(), 3),
        'mi_std': round(df['mi'].std(), 3)
    }

def baseline_summary(f):
    df = pd.read_csv(f)
    return {'n_windows': len(df)}

def model_summary(f):
    return {}  # Could add joblib inspection if needed

# --- Dash App ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Mindfulness Index Pipeline Dashboard"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H4("LSL Sensor Status (Online)"),
            dcc.Interval(id='interval-lsl', interval=3000, n_intervals=0),
            dash_table.DataTable(id='lsl-table', style_table={'overflowX': 'auto'})
        ], width=6),
        dbc.Col([
            html.H4("Report/Log Files (Offline)"),
            dash_table.DataTable(
                id='mi-files',
                columns=[{"name": i, "id": i} for i in ['File','Size (KB)','n_samples','mi_mean','mi_std','Summary']],
                style_table={'overflowX': 'auto'}
            ),
            html.Br(),
            html.H4("Calibration Baselines"),
            dash_table.DataTable(
                id='baseline-files',
                columns=[{"name": i, "id": i} for i in ['File','Size (KB)','n_windows']],
                style_table={'overflowX': 'auto'}
            ),
            html.Br(),
            html.H4("Model/Scaler Files"),
            dash_table.DataTable(
                id='model-files',
                columns=[{"name": i, "id": i} for i in ['File','Size (KB)']],
                style_table={'overflowX': 'auto'}
            )
        ], width=6)
    ]),
    html.Hr(),
    html.H4("Report Settings (Placeholder)"),
    html.Div("(Add options for report generation, file selection, etc. here.)"),
    html.Hr(),
    html.Div(id='error-msg', style={'color':'red'})
], fluid=True)

@app.callback(
    Output('lsl-table', 'data'),
    Input('interval-lsl', 'n_intervals')
)
def update_lsl_table(n):
    return get_lsl_status()

@app.callback(
    Output('mi-files', 'data'),
    Output('baseline-files', 'data'),
    Output('model-files', 'data'),
    Output('error-msg', 'children'),
    Input('interval-lsl', 'n_intervals')
)
def update_file_tables(n):
    errors = []
    mi_df = get_file_table(LOG_DIR, '*_mi_session_*.csv',
        columns=['File','Size (KB)','n_samples','mi_mean','mi_std','Summary'], summary_func=mi_session_summary)
    baseline_df = get_file_table(USER_CONFIG_DIR, '*_baseline.csv',
        columns=['File','Size (KB)','n_windows'], summary_func=baseline_summary)
    model_df = get_file_table(MODEL_DIR, '*.joblib', columns=['File','Size (KB)'], summary_func=model_summary)
    if mi_df.empty:
        errors.append("No MI session files found.")
    if baseline_df.empty:
        errors.append("No calibration baseline files found.")
    if model_df.empty:
        errors.append("No model/scaler files found.")
    return mi_df.to_dict('records'), baseline_df.to_dict('records'), model_df.to_dict('records'), ' | '.join(errors)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
