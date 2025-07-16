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
from dash import Dash, html, dcc, dash_table, callback_context
import dash_bootstrap_components as dbc
from pylsl import resolve_streams
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'final_implementation/final_logs')
USER_CONFIG_DIR = os.path.join(BASE_DIR, 'final_implementation/user_configs')
MODEL_DIR = os.path.join(BASE_DIR, 'final_implementation/models')


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

def get_all_csv_columns(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    all_cols = set()
    for f in files:
        try:
            df = pd.read_csv(f, nrows=1)
            all_cols.update(df.columns)
        except Exception:
            continue
    return sorted(list(all_cols))

def get_file_table_dynamic(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    all_cols = set(['File', 'Size (KB)'])
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=1)
            row = {'File': os.path.basename(f), 'Size (KB)': round(os.path.getsize(f)/1024, 1)}
            for col in df.columns:
                row[col] = df.iloc[0][col]
            dfs.append(row)
            all_cols.update(df.columns)
        except Exception as e:
            dfs.append({'File': os.path.basename(f), 'Size (KB)': round(os.path.getsize(f)/1024, 1), 'Error': str(e)})
            all_cols.add('Error')
    return pd.DataFrame(dfs, columns=sorted(list(all_cols)))

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

def serve_layout():
    # Get all columns for MI session files
    mi_cols = get_all_csv_columns(LOG_DIR, '*_mi_session_*.csv')
    return dbc.Container([
        html.H2("Mindfulness Index Pipeline Dashboard"),
        html.Hr(),
        dcc.Tabs(id='main-tabs', value='tab-files', children=[
            dcc.Tab(label='File Tables', value='tab-files', children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("LSL Sensor Status (Online)"),
                        dcc.Interval(id='interval-lsl', interval=3000, n_intervals=0),
                        dash_table.DataTable(id='lsl-table', style_table={'overflowX': 'auto', 'marginBottom': '30px'}),
                        html.H4("MI Session Files (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='mi-files', style_table={'overflowX': 'auto', 'marginBottom': '30px'}),
                        html.H4("Calibration Baselines (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='baseline-files', style_table={'overflowX': 'auto', 'marginBottom': '30px'}),
                        html.H4("Model/Scaler Files (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='model-files', style_table={'overflowX': 'auto', 'marginBottom': '30px'}),
                        html.Hr(),
                        html.H4("Report Settings (Placeholder)"),
                        html.Div("(Add options for report generation, file selection, etc. here.)"),
                        html.Hr(),
                        html.Div(id='error-msg', style={'color':'red'})
                    ], width=6, style={'minWidth': '500px', 'maxWidth': '700px', 'marginLeft': '0px'})
                ], justify='start')
            ]),
            dcc.Tab(label='Comparative Plot', value='tab-plot', children=[
                html.Br(),
                html.Div([
                    html.Label("Select MI Session File(s):"),
                    dcc.Dropdown(
                        id='plot-file-dropdown',
                        options=[{'label': os.path.basename(f), 'value': f} for f in glob.glob(os.path.join(LOG_DIR, '*_mi_session_*.csv'))],
                        multi=True
                    ),
                    html.Br(),
                    html.Label("Select Column(s) to Plot:"),
                    dcc.Dropdown(
                        id='plot-col-dropdown',
                        options=[{'label': c, 'value': c} for c in mi_cols],
                        multi=True
                    ),
                    html.Br(),
                    html.Label("Scaling Method:"),
                    dcc.Dropdown(
                        id='plot-scaling-dropdown',
                        options=[
                            {'label': 'None (raw)', 'value': 'none'},
                            {'label': 'Subtract Initial Value', 'value': 'subtract_initial'},
                            {'label': 'Min-Max [0,1]', 'value': 'minmax'},
                            {'label': 'Z-score', 'value': 'zscore'}
                        ],
                        value='subtract_initial',
                        clearable=False
                    ),
                    html.Br(),
                    html.Label("Plot Downsampling (show every Nth sample, 1 Hz data):"),
                    dcc.Slider(id='plot-freq-slider', min=1, max=10, step=1, value=1,
                               marks={i: f'Every {i}' for i in range(1, 11)}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Smoothing Window (samples, moving average):"),
                    dcc.Slider(id='plot-smooth-slider', min=1, max=50, step=1, value=1,
                               marks={1: 'None', 5: '5', 10: '10', 20: '20', 50: '50'}, tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    dcc.Graph(id='comparative-plot')
                ], style={'maxWidth': '900px', 'marginLeft': '0px'})
            ])
        ])
    ], fluid=True)

app.layout = serve_layout


# LSL table update
@app.callback(
    Output('lsl-table', 'data'),
    Input('interval-lsl', 'n_intervals')
)
def update_lsl_table(n):
    return get_lsl_status()


# Dynamic file tables update
@app.callback(
    Output('mi-files', 'data'),
    Output('mi-files', 'columns'),
    Output('baseline-files', 'data'),
    Output('baseline-files', 'columns'),
    Output('model-files', 'data'),
    Output('model-files', 'columns'),
    Output('error-msg', 'children'),
    Input('interval-lsl', 'n_intervals')
)
def update_file_tables(n):
    errors = []
    mi_df = get_file_table_dynamic(LOG_DIR, '*_mi_session_*.csv')
    baseline_df = get_file_table_dynamic(USER_CONFIG_DIR, '*_baseline.csv')
    model_df = get_file_table_dynamic(MODEL_DIR, '*.joblib')
    if mi_df.empty:
        errors.append("No MI session files found.")
    if baseline_df.empty:
        errors.append("No calibration baseline files found.")
    if model_df.empty:
        errors.append("No model/scaler files found.")
    return (
        mi_df.to_dict('records'), [{'name': c, 'id': c} for c in mi_df.columns],
        baseline_df.to_dict('records'), [{'name': c, 'id': c} for c in baseline_df.columns],
        model_df.to_dict('records'), [{'name': c, 'id': c} for c in model_df.columns],
        ' | '.join(errors)
    )

# Comparative plot callback
@app.callback(
    Output('comparative-plot', 'figure'),
    Input('plot-file-dropdown', 'value'),
    Input('plot-col-dropdown', 'value'),
    Input('plot-scaling-dropdown', 'value'),
    Input('plot-freq-slider', 'value'),
    Input('plot-smooth-slider', 'value')
)
def update_comparative_plot(selected_files, selected_cols, scaling, freq, smooth):
    if not selected_files or not selected_cols:
        return go.Figure()
    fig = go.Figure()
    import numpy as np
    # For each column, align, truncate, normalize, and collect for median
    for col in selected_cols:
        series_list = []
        x_list = []
        min_len = None
        # First pass: collect all series, align, normalize, and find min length
        for file in selected_files:
            try:
                df = pd.read_csv(file)
                # Normalize x-axis: if timestamp, subtract first value; else use index
                if 'timestamp' in df.columns:
                    x = df['timestamp'] - df['timestamp'].iloc[0]
                else:
                    x = df.index
                # Downsample by skipping samples (1 Hz data)
                step = max(1, int(freq))
                idx = range(0, len(df), step)
                x_ds = x.iloc[idx] if hasattr(x, 'iloc') else x[idx]
                if col in df.columns:
                    y = df[col]
                    # Smoothing (moving average)
                    if smooth > 1:
                        y = y.rolling(window=smooth, min_periods=1, center=True).mean()
                    # Scaling
                    if scaling == 'subtract_initial':
                        y = y - y.iloc[0]
                    elif scaling == 'minmax':
                        minv, maxv = y.min(), y.max()
                        y = (y - minv) / (maxv - minv) if maxv > minv else y - minv
                    elif scaling == 'zscore':
                        y = (y - y.mean()) / y.std() if y.std() > 0 else y - y.mean()
                    # else: 'none' (raw)
                    y_ds = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                    # Truncate to min length later
                    series_list.append(y_ds.reset_index(drop=True))
                    x_list.append(x_ds.reset_index(drop=True))
                    if min_len is None or len(y_ds) < min_len:
                        min_len = len(y_ds)
            except Exception as e:
                continue
        # Second pass: truncate all to min_len
        truncated_series = [s.iloc[:min_len] for s in series_list if len(s) >= min_len]
        truncated_x = x_list[0].iloc[:min_len] if x_list else []
        # Plot all individual series
        for i, s in enumerate(truncated_series):
            fig.add_trace(go.Scatter(x=truncated_x, y=s, mode='lines',
                                     name=f"{os.path.basename(selected_files[i])}: {col}",
                                     line=dict(width=1)))
        # Plot median if there are at least 2 series
        if len(truncated_series) > 1:
            arr = np.vstack([s.values for s in truncated_series])
            median = np.nanmedian(arr, axis=0)
            fig.add_trace(go.Scatter(x=truncated_x, y=median, mode='lines',
                                     name=f"Median: {col}",
                                     line=dict(width=4, dash='dash', color='black')))
    fig.update_layout(title="Comparative Plot (Aligned, Scaled, Median)", xaxis_title="Time (s)", yaxis_title="Value (scaled)", legend_title="File:Column")
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
