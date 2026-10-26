from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path('data/cleaned/training_records_cleaned.csv')
OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
for c in ['max_throw_distance_m','release_velocity_ms','workload_au','sleep_hours','hrv_ms','soreness_score']:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

# Single athlete, no event split - use all throws
if 'is_throwing_session' in df.columns:
    is_throw = df['is_throwing_session'].astype(str).str.lower().eq('true')
else:
    is_throw = df['max_throw_distance_m'] > 0
throws = df[is_throw & (df['max_throw_distance_m']>0)].copy().sort_values('date').reset_index(drop=True)

# Feature eng with shift
throws['prev_distance'] = throws['max_throw_distance_m'].shift(1)
throws['prev_velocity'] = throws['release_velocity_ms'].shift(1)
throws['prev_workload'] = throws['workload_au'].shift(1)
throws['prev_sleep'] = throws['sleep_hours'].shift(1)
throws['prev_hrv'] = throws['hrv_ms'].shift(1)
throws['prev_soreness'] = throws['soreness_score'].shift(1)
for w in [3,5]:
    throws[f'roll_dist_{w}'] = throws['max_throw_distance_m'].shift(1).rolling(window=w, min_periods=1).mean()
    throws[f'roll_vel_{w}'] = throws['release_velocity_ms'].shift(1).rolling(window=w, min_periods=1).mean()
    throws[f'roll_workload_{w}'] = throws['workload_au'].shift(1).rolling(window=w, min_periods=1).mean()
throws['days_since_prev'] = throws['date'].diff().dt.days

numeric_features = ['workload_au','sleep_hours','hrv_ms','soreness_score','prev_distance','prev_velocity','prev_workload','prev_sleep','prev_hrv','prev_soreness','roll_dist_3','roll_vel_3','roll_workload_3','roll_dist_5','roll_vel_5','roll_workload_5','days_since_prev']
categorical_features = ['training_phase','session_type']
features = numeric_features + categorical_features

throws = throws.dropna(subset=['date','max_throw_distance_m']).reset_index(drop=True)
split_idx = int(len(throws)*0.8)
train = throws.iloc[:split_idx].copy()
test = throws.iloc[split_idx:].copy()

X_train = train[features].copy(); y_train = train['max_throw_distance_m'].copy()
X_test = test[features].copy(); y_test = test['max_throw_distance_m'].copy()

numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('numeric', numeric_pipeline, numeric_features), ('categorical', categorical_pipeline, categorical_features)])

# Chrono CV on train only
tscv = TimeSeriesSplit(n_splits=3)
rf = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])
# quick tune
from sklearn.model_selection import GridSearchCV
param_grid = {'regressor__n_estimators': [200, 500], 'regressor__max_depth': [4, 6, 8], 'regressor__min_samples_leaf': [3, 5]}
gcv = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
gcv.fit(X_train, y_train)
print('Best params:', gcv.best_params_)
best_rf = gcv.best_estimator_

# Baselines
pred_mean = DummyRegressor(strategy='mean').fit(X_train,y_train).predict(X_test)
# persistence
pred_persist = test['prev_distance'].fillna(train['max_throw_distance_m'].mean()).values
# ridge
ridge = Pipeline([('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))]).fit(X_train,y_train)
pred_ridge = ridge.predict(X_test)
pred_rf = best_rf.predict(X_test)

def eval_m(y_true,y_pred):
    return {'MAE_m': mean_absolute_error(y_true,y_pred), 'RMSE_m': np.sqrt(mean_squared_error(y_true,y_pred)), 'R2': r2_score(y_true,y_pred)}

res = pd.DataFrame({
    'Baseline_mean': eval_m(y_test,pred_mean),
    'Baseline_persistence': eval_m(y_test,pred_persist),
    'Ridge': eval_m(y_test,pred_ridge),
    'RandomForest_tuned': eval_m(y_test,pred_rf),
}).T
print(res.round(3))
res.to_csv(OUTPUT_DIR / 'model_metrics.csv')
pred_df = pd.DataFrame({'date': test['date'].values,'actual_distance_m': y_test.values,'baseline_prediction_m': pred_mean,'baseline_persistence_m': pred_persist,'ridge_prediction_m': pred_ridge,'random_forest_prediction_m': pred_rf})
pred_df.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
joblib.dump(best_rf, MODEL_DIR / 'distance_model.joblib')
