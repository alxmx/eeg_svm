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
LOG_DIR = os.path.join(BASE_DIR, 'final_implementation/logs')
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
                        dash_table.DataTable(id='lsl-table', style_table={'overflowX': 'auto', 'marginBottom': '30px', 'width': '100%'}),
                        html.H4("MI Session Files (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='mi-files', style_table={'overflowX': 'auto', 'marginBottom': '30px', 'width': '100%'}),
                        html.H4("Calibration Baselines (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='baseline-files', style_table={'overflowX': 'auto', 'marginBottom': '30px', 'width': '100%'}),
                        html.H4("Model/Scaler Files (Dynamic Columns)", style={'marginTop': '30px'}),
                        dash_table.DataTable(id='model-files', style_table={'overflowX': 'auto', 'marginBottom': '30px', 'width': '100%'}),
                        html.Hr(),
                        html.H4("Report Settings (Placeholder)"),
                        html.Div("(Add options for report generation, file selection, etc. here.)"),
                        html.Hr(),
                        html.Div(id='error-msg', style={'color':'red'})
                    ], width=12, style={'marginLeft': '0px'})
                ], justify='start')
            ]),
            dcc.Tab(label='Comparative Plot', value='tab-plot', children=[
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.Label("Select MI Session File(s):"),
                        dcc.Dropdown(
                            id='plot-file-dropdown',
                            options=[{'label': os.path.basename(f), 'value': f} for f in glob.glob(os.path.join(LOG_DIR, '*_mi_session_*.csv'))],
                            multi=True
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Select Column(s) to Plot:"),
                        dcc.Dropdown(
                            id='plot-col-dropdown',
                            options=[{'label': c, 'value': c} for c in mi_cols],
                            multi=True
                        ),
                    ], width=4)
                ], justify='start', style={'marginBottom': '20px'}),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='comparative-plot', style={'height': '700px', 'width': '100%'})
                    ], width=12)
                ])
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
    Input('plot-col-dropdown', 'value')
)
def update_comparative_plot(selected_files, selected_cols):
    if not selected_files or not selected_cols:
        return go.Figure()
    fig = go.Figure()
    for file in selected_files:
        try:
            df = pd.read_csv(file)
            # Normalize x-axis: if timestamp, subtract first value; else use index
            if 'timestamp' in df.columns:
                x = df['timestamp'] - df['timestamp'].iloc[0]
            else:
                x = df.index
            for col in selected_cols:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=x, y=df[col], mode='lines', name=f"{os.path.basename(file)}: {col}"))
        except Exception as e:
            continue
    fig.update_layout(title="Comparative Plot", xaxis_title="Time (s)", yaxis_title="Value", legend_title="File:Column")
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
