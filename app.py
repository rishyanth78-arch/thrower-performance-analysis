from pathlib import Path
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from preprocess import process_and_clean_data

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title='Athlete Performance Dashboard', page_icon='🏅', layout='wide')
st.title('🏅 Athlete Training & Performance Dashboard')
st.caption('Explore throwing performance, training workload, recovery measurements, and regression results.')


@st.cache_data(show_spinner=False)
def load_data():
    """
    Robust loader that checks for cleaned data, falls back to raw data with
    automatic on-the-fly preprocessing, and supports fallback locations.
    """
    cleaned_candidates = [
        BASE / 'data/cleaned/training_records_cleaned.csv',
        Path('data/cleaned/training_records_cleaned.csv'),
        BASE / 'training_records_cleaned.csv',
        Path('training_records_cleaned.csv'),
    ]

    for p in cleaned_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df, f"Loaded cleaned data from: {p}"
            except Exception:
                continue

    # Fallback to raw data with on-the-fly preprocessing
    raw_candidates = [
        BASE / 'data/raw/training_records.csv',
        Path('data/raw/training_records.csv'),
        BASE / 'training_records.csv',
        Path('training_records.csv'),
    ]

    for raw_p in raw_candidates:
        if raw_p.exists():
            clean_out = BASE / 'data/cleaned/training_records_cleaned.csv'
            try:
                df = process_and_clean_data(source=raw_p, output_path=clean_out)
                return df, f"Automatically cleaned raw data from: {raw_p}"
            except Exception as e:
                try:
                    df = process_and_clean_data(source=raw_p, output_path=None)
                    return df, f"Cleaned raw data in memory from: {raw_p}"
                except Exception:
                    continue

    return None, "No data file found."


# Sidebar Data Controls
st.sidebar.header('Data Source & Filters')

uploaded_file = st.sidebar.file_uploader('Upload CSV (if not in repo)', type=['csv'])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if 'is_throwing_session' in raw_df.columns and 'data_quality_flag' in raw_df.columns:
            df = raw_df
            st.sidebar.success('Loaded uploaded cleaned dataset.')
        else:
            df = process_and_clean_data(source=raw_df, output_path=None)
            st.sidebar.success('Uploaded data successfully preprocessed on the fly!')
    except Exception as e:
        st.sidebar.error(f"Error processing uploaded file: {e}")
        df = None
else:
    df, status_msg = load_data()
    if df is not None:
        st.sidebar.info(status_msg)

if df is None or df.empty:
    st.error('Cleaned or raw data was not found in the deployment directory.')
    st.info('Please upload your `training_records.csv` or `training_records_cleaned.csv` using the file uploader in the sidebar.')
    st.stop()

required = ['date', 'training_phase', 'session_type', 'workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score', 'release_velocity_ms', 'max_throw_distance_m']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f'Missing required columns in dataset: {missing}')
    st.stop()

df['date'] = pd.to_datetime(df['date'], errors='coerce')
for c in ['workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score', 'release_velocity_ms', 'max_throw_distance_m']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['date']).sort_values('date').copy()
throwing = df[df['max_throw_distance_m'] > 0].copy()
if throwing.empty:
    st.error('No sessions with positive throwing distances were found.')
    st.stop()

# Filters
phase_order = ['General Preparation', 'Specific Preparation', 'Competition / Peak', 'Transition / Taper']
phases = [p for p in phase_order if p in throwing['training_phase'].unique()]
selected_phases = st.sidebar.multiselect('Training phase', phases, default=phases)

min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input('Date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

filtered_all = df[df['date'].dt.date.between(start_date, end_date) & df['training_phase'].isin(selected_phases)].copy()
filtered_throws = filtered_all[filtered_all['max_throw_distance_m'] > 0].copy()
if filtered_throws.empty:
    st.warning('No throwing sessions match these filters.')
    st.stop()

# 1. Performance Overview
st.header('Performance Overview')
c1, c2, c3, c4 = st.columns(4)
c1.metric('Throwing sessions', len(filtered_throws))
c2.metric('Average distance', f'{filtered_throws["max_throw_distance_m"].mean():.2f} m')
c3.metric('Average release velocity', f'{filtered_throws["release_velocity_ms"].mean():.2f} m/s')
c4.metric('Average workload', f'{filtered_all["workload_au"].mean():.1f} AU')
st.caption('Distance/velocity from positive throws only. Workload includes all filtered sessions.')


def show_time_series(data, column, title, ylabel):
    plot_data = data[['date', column]].dropna().sort_values('date')
    if plot_data.empty:
        st.info(f'No data available for {title}.')
        return
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(plot_data['date'], plot_data[column], linewidth=1.5, marker='.', markersize=3)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# 2. Performance Trends
st.header('Performance Trends')
show_time_series(filtered_throws, 'max_throw_distance_m', 'Throwing Distance Over Time', 'Distance (m)')
show_time_series(filtered_throws, 'release_velocity_ms', 'Release Velocity Over Time', 'Velocity (m/s)')

# 3. Training Load & Recovery
st.header('Training Load & Recovery')
st.caption('Each measurement has its own chart and independent y-axis. Associations, not causation.')
show_time_series(filtered_all, 'workload_au', 'Training Workload', 'Workload (AU)')
show_time_series(filtered_all, 'sleep_hours', 'Sleep Duration', 'Sleep (hours)')
show_time_series(filtered_all, 'hrv_ms', 'Heart Rate Variability (HRV)', 'HRV (ms)')
show_time_series(filtered_all, 'soreness_score', 'Subjective Soreness', 'Soreness score')

# 4. Performance by Training Phase
st.header('Performance by Training Phase')
phase_throws = throwing[throwing['date'].dt.date.between(start_date, end_date) & throwing['training_phase'].isin(phase_order)].copy()
phase_summary = (
    phase_throws.groupby('training_phase')
    .agg(
        sessions=('max_throw_distance_m', 'count'),
        valid_distance_count=('max_throw_distance_m', 'count'),
        average_distance=('max_throw_distance_m', 'mean'),
        average_velocity=('release_velocity_ms', 'mean'),
        average_workload=('workload_au', 'mean')
    )
    .reindex(phase_order)
    .reset_index()
    .round(2)
)
st.dataframe(phase_summary, use_container_width=True, hide_index=True)
st.caption('Distance/velocity from positive throws; workload averaged over sessions in phase for filtered range.')

if not phase_throws.empty:
    phase_plot = phase_throws.groupby('training_phase')['max_throw_distance_m'].mean().reindex(phase_order).dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    phase_plot.plot(kind='bar', ax=ax)
    ax.set_title('Average Throwing Distance by Training Phase')
    ax.set_xlabel('Training phase')
    ax.set_ylabel('Average distance (m)')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 5. Regression Model Evaluation
st.header('Regression Model Evaluation')

def find_candidate_file(rel_path):
    candidates = [
        BASE / rel_path,
        Path(rel_path),
        BASE / Path(rel_path).name,
        Path(Path(rel_path).name),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

metrics_file = find_candidate_file('outputs/model_metrics.csv')
predictions_file = find_candidate_file('outputs/test_predictions.csv')

if metrics_file is not None:
    metrics = pd.read_csv(metrics_file, index_col=0)
    st.dataframe(metrics.round(3), use_container_width=True)
    st.warning('Negative R² means worse than mean baseline. Persistence baseline improved but still weak; results are experimental.')
else:
    st.info('Precomputed model metrics not found. Run python test_model.py or python train_tune.py.')

if predictions_file is not None:
    predictions = pd.read_csv(predictions_file)
    predictions['date'] = pd.to_datetime(predictions['date'], errors='coerce')
    predictions = predictions.dropna(subset=['date']).sort_values('date')
    predictions = predictions[predictions['date'].dt.date.between(start_date, end_date)]
    st.subheader('Actual vs Predicted Test Distances')
    if not predictions.empty:
        chart_cols = ['actual_distance_m', 'baseline_prediction_m', 'random_forest_prediction_m']
        if all(c in predictions.columns for c in chart_cols):
            plot_df = predictions[['date'] + chart_cols].dropna()
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(plot_df['date'], plot_df['actual_distance_m'], label='Actual distance', marker='o')
            ax.plot(plot_df['date'], plot_df['baseline_prediction_m'], label='Baseline (mean)', marker='x')
            ax.plot(plot_df['date'], plot_df['random_forest_prediction_m'], label='Random Forest', marker='s')
            ax.set_xlabel('Date')
            ax.set_ylabel('Distance (m)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            all_vals = plot_df[chart_cols].values.flatten()
            ymin, ymax = all_vals.min(), all_vals.max()
            margin = max(2, (ymax - ymin) * 0.1)
            ax.set_ylim(ymin - margin, ymax + margin)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(predictions, use_container_width=True)
    else:
        st.info('No test predictions fall within the selected date range.')

st.caption('Educational analysis only. Not validated for training/medical advice.')
