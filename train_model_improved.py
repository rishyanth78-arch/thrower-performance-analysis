from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path('data/cleaned/training_records_cleaned.csv')
OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError('Cleaned dataset not found. Run python preprocess.py first.')

df = pd.read_csv(DATA_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Ensure numeric
for c in ['max_throw_distance_m', 'release_velocity_ms', 'workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Mark throwing sessions
if 'is_throwing_session' in df.columns:
    is_throw = df['is_throwing_session'].astype(str).str.lower().eq('true')
else:
    is_throw = df['max_throw_distance_m'] > 0

# Work on throwing sessions only for target modeling
throws = df[is_throw & (df['max_throw_distance_m'] > 0)].copy()
throws = throws.sort_values('date').reset_index(drop=True)

# Feature engineering with shift to avoid leakage
# Previous session values (by date order)
throws['prev_distance'] = throws['max_throw_distance_m'].shift(1)
throws['prev_velocity'] = throws['release_velocity_ms'].shift(1)
throws['prev_workload'] = throws['workload_au'].shift(1)
throws['prev_sleep'] = throws['sleep_hours'].shift(1)
throws['prev_hrv'] = throws['hrv_ms'].shift(1)
throws['prev_soreness'] = throws['soreness_score'].shift(1)

# Rolling means over previous 5 sessions (shifted)
for w in [3, 5]:
    throws[f'roll_dist_{w}'] = throws['max_throw_distance_m'].shift(1).rolling(window=w, min_periods=1).mean()
    throws[f'roll_vel_{w}'] = throws['release_velocity_ms'].shift(1).rolling(window=w, min_periods=1).mean()
    throws[f'roll_workload_{w}'] = throws['workload_au'].shift(1).rolling(window=w, min_periods=1).mean()

# Days since previous session
throws['days_since_prev'] = throws['date'].diff().dt.days

numeric_features = [
    'workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score',
    'prev_distance', 'prev_velocity', 'prev_workload', 'prev_sleep', 'prev_hrv', 'prev_soreness',
    'roll_dist_3', 'roll_vel_3', 'roll_workload_3',
    'roll_dist_5', 'roll_vel_5', 'roll_workload_5',
    'days_since_prev',
]
categorical_features = ['training_phase', 'session_type']
features = numeric_features + categorical_features
target = 'max_throw_distance_m'

# Drop rows with missing target/date
throws = throws.dropna(subset=['date', target]).reset_index(drop=True)
if len(throws) < 30:
    raise ValueError('Too few throwing sessions for modeling.')

# Chronological split
split_idx = int(len(throws) * 0.8)
train = throws.iloc[:split_idx].copy()
test = throws.iloc[split_idx:].copy()

X_train = train[features].copy()
y_train = train[target].copy()
X_test = test[features].copy()
y_test = test[target].copy()

# Preprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features),
])

# Baselines
baseline_mean = DummyRegressor(strategy='mean')
baseline_mean.fit(X_train, y_train)
pred_mean = baseline_mean.predict(X_test)

# Persistence baseline (predict previous distance)
pred_persist = test['prev_distance'].fillna(train[target].mean()).values

# Models
ridge = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))])
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)

rf = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1))])
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

def eval_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE_m': mae, 'RMSE_m': rmse, 'R2': r2}

results = pd.DataFrame({
    'Baseline_mean': eval_model(y_test, pred_mean),
    'Baseline_persistence': eval_model(y_test, pred_persist),
    'Ridge': eval_model(y_test, pred_ridge),
    'RandomForest': eval_model(y_test, pred_rf),
}).T

print('=== TEST RESULTS (chronological) ===')
print(results.round(3))
print(f'Train: {len(train)} ({train["date"].min().date()} to {train["date"].max().date()}) | Test: {len(test)} ({test["date"].min().date()} to {test["date"].max().date()})')

results.to_csv(OUTPUT_DIR / 'model_metrics.csv')
pred_df = pd.DataFrame({
    'date': test['date'].values,
    'actual_distance_m': y_test.values,
    'baseline_prediction_m': pred_mean,
    'baseline_persistence_m': pred_persist,
    'ridge_prediction_m': pred_ridge,
    'random_forest_prediction_m': pred_rf,
})
pred_df.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)

# Save best model (by RMSE)
best_name = results['RMSE_m'].idxmin()
if best_name == 'Ridge':
    joblib.dump(ridge, MODEL_DIR / 'distance_model.joblib')
elif best_name == 'RandomForest':
    joblib.dump(rf, MODEL_DIR / 'distance_model.joblib')
else:
    joblib.dump(rf, MODEL_DIR / 'distance_model.joblib')

print(f'Best model: {best_name}, saved to models/distance_model.joblib')
