# Athlete Training & Performance Dashboard

## Project Objectives
- Analyze daily training data for a single thrower (365 sessions, 2025) to understand performance trends across training phases.
- Clean and preprocess raw training records, distinguishing throwing sessions from rest days.
- Engineer leakage-safe features (lags, rolling means over prior sessions, days since previous session) for throw-distance prediction.
- Compare baseline models (mean, persistence) with linear and tree-based regressors under a strict chronological train/test split.
- Provide an interactive Streamlit dashboard to explore performance, workload/recovery, phase summaries, and regression evaluation.

## Data Source
- Source: local CSV `data/raw/training_records.csv` (365 daily sessions, 2025-01-01 to 2025-12-31), single athlete, no athlete/event/id columns.
- Cleaned: `data/cleaned/training_records_cleaned.csv` (365 rows). 
- Columns (cleaned): `date`, `training_phase`, `session_type`, `workload_au`, `sleep_hours`, `hrv_ms`, `soreness_score`, `release_velocity_ms`, `max_throw_distance_m`, `is_throwing_session`, `data_quality_flag`.
- Throwing sessions: sessions with positive `max_throw_distance_m` and `is_throwing_session` true (327 throwing sessions). Rest days (38) have zero/absent performance values and are preserved for workload/recovery analysis.

## Preprocessing
- Standardized column names (lowercase, underscores), parsed dates, coerced numerics, removed exact duplicates, sorted chronologically.
- Identified rest days (e.g. `session_type` == 'Rest') and set `is_throwing_session` accordingly.
- Flagged potential data quality issues (negatives/rest-with-performance) via `data_quality_flag`.
- Rest days preserved for load/recovery visualizations; excluded from throwing-distance averages where specified.

## Feature Engineering (leakage-safe)
- **Lags (shifted)**: previous session values for distance, velocity, workload, sleep, HRV, soreness.
- **Rolling means (shifted)**: rolling means of previous sessions only (computed on `shift(1)`) over windows 3 and 5 for distance, velocity, workload.
- **Time-based**: `days_since_prev` (days since previous session).
- **Categorical**: `training_phase`, `session_type` (one-hot encoded in pipelines). Numeric features imputed/scaled as appropriate.
- **Note**: Current-session `release_velocity_ms` is measured during the throw and is excluded from pre-session prediction features in the leakage-safe models.

## Model Comparison (chronological test set)
Test set: 66 sessions (2025-10-19 to 2025-12-31). Train set: 261 sessions (2025-01-01 to 2025-10-18).

| Model | MAE (m) | RMSE (m) | R² |
|---|---|---|---|
| Baseline_mean | 4.890 | 5.581 | -2.448 |
| Baseline_persistence | 3.029 | 3.694 | -0.510 |
| Ridge | 4.290 | 5.403 | -2.230 |
| RandomForest_tuned | 2.645 | 3.204 | -0.136 |

Best model by RMSE: RandomForest_tuned. R² remains negative for all models, indicating limited predictive signal from pre-session features alone.

## Evaluation Metrics
- **MAE (m)**: Mean Absolute Error in meters.
- **RMSE (m)**: Root Mean Squared Error in meters.
- **R²**: Coefficient of determination (negative = worse than predicting the mean).

## Limitations
- R² remains negative even for the best model - results are not reliable for forecasting.
- Single athlete, single year (2025); no athlete/event split available.
- Pre-session features alone have limited signal for throw distance.
- Some analyses/visuals may use different feature sets depending on script; the dashboard reports saved metrics/predictions.

## Installation
```bash
pip install -r requirements.txt
```

## Run Commands
```bash
# Preprocess raw data
python preprocess.py

# Generate analysis outputs/plots
python analysis.py

# Train/tune models (produces outputs/model_metrics.csv, outputs/test_predictions.csv, models/distance_model.joblib)
python train_tune.py

# Test and validate model on held-out test set
python test_model.py

# Run dashboard
streamlit run app.py
```

Other training scripts are available (`train_model.py`, `train_model_improved.py`) for comparison; `train_tune.py` produces the results reflected in the dashboard and README. `test_model.py` runs the comprehensive model testing and validation suite. 
