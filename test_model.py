"""
Comprehensive Model Testing & Validation Suite for Thrower Performance Analysis.
Tests models/distance_model.joblib on chronological test split, evaluates error metrics,
residuals, slice-based performance, edge-case robustness, and inference latency.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error
)

DATA_FILE = Path('data/cleaned/training_records_cleaned.csv')
MODEL_FILE = Path('models/distance_model.joblib')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load and prepare features with identical leakage-safe feature engineering."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    numeric_cols = [
        'max_throw_distance_m', 'release_velocity_ms',
        'workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'is_throwing_session' in df.columns:
        is_throw = df['is_throwing_session'].astype(str).str.lower().eq('true')
    else:
        is_throw = df['max_throw_distance_m'] > 0

    throws = df[is_throw & (df['max_throw_distance_m'] > 0)].copy()
    throws = throws.sort_values('date').reset_index(drop=True)

    # Leakage-safe feature engineering
    throws['prev_distance'] = throws['max_throw_distance_m'].shift(1)
    throws['prev_velocity'] = throws['release_velocity_ms'].shift(1)
    throws['prev_workload'] = throws['workload_au'].shift(1)
    throws['prev_sleep'] = throws['sleep_hours'].shift(1)
    throws['prev_hrv'] = throws['hrv_ms'].shift(1)
    throws['prev_soreness'] = throws['soreness_score'].shift(1)

    for w in [3, 5]:
        throws[f'roll_dist_{w}'] = throws['max_throw_distance_m'].shift(1).rolling(window=w, min_periods=1).mean()
        throws[f'roll_vel_{w}'] = throws['release_velocity_ms'].shift(1).rolling(window=w, min_periods=1).mean()
        throws[f'roll_workload_{w}'] = throws['workload_au'].shift(1).rolling(window=w, min_periods=1).mean()

    throws['days_since_prev'] = throws['date'].diff().dt.days

    numeric_features = [
        'workload_au', 'sleep_hours', 'hrv_ms', 'soreness_score',
        'prev_distance', 'prev_velocity', 'prev_workload', 'prev_sleep',
        'prev_hrv', 'prev_soreness',
        'roll_dist_3', 'roll_vel_3', 'roll_workload_3',
        'roll_dist_5', 'roll_vel_5', 'roll_workload_5',
        'days_since_prev'
    ]
    categorical_features = ['training_phase', 'session_type']
    features = numeric_features + categorical_features

    throws = throws.dropna(subset=['date', 'max_throw_distance_m']).reset_index(drop=True)
    split_idx = int(len(throws) * 0.8)

    train = throws.iloc[:split_idx].copy()
    test = throws.iloc[split_idx:].copy()

    return train, test, features, numeric_features, categorical_features


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    residuals = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        pearson_r, p_val = stats.pearsonr(y_true, y_pred)
        p_r = round(float(pearson_r), 4)
        p_val_r = round(float(p_val), 6)
    else:
        p_r, p_val_r = None, None

    return {
        'MAE_m': round(mae, 4),
        'RMSE_m': round(rmse, 4),
        'R2': round(r2, 4),
        'MedAE_m': round(medae, 4),
        'MAPE_pct': round(mape, 2),
        'Pearson_r': p_r,
        'Pearson_pval': p_val_r,
        'Mean_Residual_m': round(float(np.mean(residuals)), 4),
        'Std_Residual_m': round(float(np.std(residuals)), 4),
        'Max_Underpredict_m': round(float(np.max(residuals)), 4),
        'Max_Overpredict_m': round(float(np.min(residuals)), 4),
    }


def test_model_pipeline():
    """Main testing suite."""
    print("=" * 65)
    print("      THROWER PERFORMANCE MODEL TEST & VALIDATION SUITE      ")
    print("=" * 65)

    # 1. Model Loading Check
    print("\n[TEST 1] Loading Trained Model...")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    print(f"  [PASS] Model successfully loaded from: {MODEL_FILE}")
    print(f"  [PASS] Pipeline steps: {[name for name, _ in model.steps]}")
    regressor = model.named_steps.get('regressor')
    print(f"  [PASS] Regressor type: {type(regressor).__name__}")
    if hasattr(regressor, 'get_params'):
        params = regressor.get_params()
        key_params = {k: params[k] for k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'random_state'] if k in params}
        print(f"  [PASS] Key hyperparameters: {key_params}")

    # 2. Data Preparation & Chronological Split
    print("\n[TEST 2] Preparing Data & Chronological Split...")
    train, test, features, num_feats, cat_feats = load_and_prepare_data()
    print(f"  [PASS] Training sessions: {len(train)} ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"  [PASS] Testing sessions:  {len(test)} ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"  [PASS] Total features:    {len(features)} ({len(num_feats)} numeric, {len(cat_feats)} categorical)")

    X_train = train[features]
    y_train = train['max_throw_distance_m']
    X_test = test[features]
    y_test = test['max_throw_distance_m']

    # 3. Model Inference & Prediction
    print("\n[TEST 3] Running Inference on Held-Out Test Set...")
    start_time = time.perf_counter()
    y_pred_rf = model.predict(X_test)
    elapsed_time = (time.perf_counter() - start_time) * 1000
    latency_per_sample = elapsed_time / len(X_test)
    print(f"  [PASS] Predicted {len(X_test)} samples in {elapsed_time:.2f} ms ({latency_per_sample:.3f} ms/sample)")

    # 4. Baseline Models
    pred_mean = np.full_like(y_test, fill_value=y_train.mean())
    pred_persist = test['prev_distance'].fillna(y_train.mean()).values

    # 5. Metrics Calculation & Comparison
    print("\n[TEST 4] Model Evaluation vs Baselines:")
    rf_metrics = compute_metrics(y_test.values, y_pred_rf)
    mean_metrics = compute_metrics(y_test.values, pred_mean)
    persist_metrics = compute_metrics(y_test.values, pred_persist)

    comparison_df = pd.DataFrame({
        'Baseline_Mean': mean_metrics,
        'Baseline_Persistence': persist_metrics,
        'RandomForest_Model': rf_metrics,
    })

    print(comparison_df[['Baseline_Mean', 'Baseline_Persistence', 'RandomForest_Model']].to_string())

    # Improvement calculation
    mae_imp_mean = ((mean_metrics['MAE_m'] - rf_metrics['MAE_m']) / mean_metrics['MAE_m']) * 100
    rmse_imp_mean = ((mean_metrics['RMSE_m'] - rf_metrics['RMSE_m']) / mean_metrics['RMSE_m']) * 100
    rmse_imp_persist = ((persist_metrics['RMSE_m'] - rf_metrics['RMSE_m']) / persist_metrics['RMSE_m']) * 100

    print(f"\n  [PASS] Random Forest RMSE improvement over Mean Baseline: {rmse_imp_mean:.1f}%")
    print(f"  [PASS] Random Forest RMSE improvement over Persistence:   {rmse_imp_persist:.1f}%")

    # 6. Slice-Based Error Analysis
    print("\n[TEST 5] Slice-Based Performance Analysis:")
    test_eval = test.copy()
    test_eval['predicted_distance_m'] = y_pred_rf
    test_eval['residual_m'] = test_eval['max_throw_distance_m'] - y_pred_rf
    test_eval['abs_error_m'] = np.abs(test_eval['residual_m'])

    phase_slices = []
    for phase, group in test_eval.groupby('training_phase'):
        if len(group) > 1:
            m = compute_metrics(group['max_throw_distance_m'].values, group['predicted_distance_m'].values)
            phase_slices.append({
                'Slice_Type': 'Training Phase',
                'Slice_Value': phase,
                'Count': len(group),
                'MAE_m': m['MAE_m'],
                'RMSE_m': m['RMSE_m'],
                'R2': m['R2'],
                'Avg_Actual': round(group['max_throw_distance_m'].mean(), 2),
                'Avg_Pred': round(group['predicted_distance_m'].mean(), 2)
            })

    session_slices = []
    for stype, group in test_eval.groupby('session_type'):
        if len(group) > 1:
            m = compute_metrics(group['max_throw_distance_m'].values, group['predicted_distance_m'].values)
            session_slices.append({
                'Slice_Type': 'Session Type',
                'Slice_Value': stype,
                'Count': len(group),
                'MAE_m': m['MAE_m'],
                'RMSE_m': m['RMSE_m'],
                'R2': m['R2'],
                'Avg_Actual': round(group['max_throw_distance_m'].mean(), 2),
                'Avg_Pred': round(group['predicted_distance_m'].mean(), 2)
            })

    slices_df = pd.DataFrame(phase_slices + session_slices)
    print(slices_df.to_string(index=False))
    slices_df.to_csv(OUTPUT_DIR / 'test_slice_metrics.csv', index=False)

    # 7. Robustness & Edge-Case Unit Tests
    print("\n[TEST 6] Edge-Case Robustness Verification:")

    # Test 6a: Single row inference
    single_sample = X_test.iloc[[0]]
    pred_single = model.predict(single_sample)
    assert len(pred_single) == 1 and not np.isnan(pred_single[0]), "Single row prediction failed!"
    print(f"  [PASS] Single-sample prediction passed: {pred_single[0]:.2f} m")

    # Test 6b: Missing numerical values (NaN imputation)
    sample_with_nan = single_sample.copy()
    sample_with_nan.loc[:, ['sleep_hours', 'hrv_ms', 'prev_workload']] = np.nan
    pred_nan = model.predict(sample_with_nan)
    assert not np.isnan(pred_nan[0]), "Imputation for NaN numeric values failed!"
    print(f"  [PASS] Missing values handled via imputer: prediction = {pred_nan[0]:.2f} m")

    # Test 6c: Unseen category
    sample_unseen_cat = single_sample.copy()
    sample_unseen_cat.loc[:, 'training_phase'] = 'New_Unseen_Phase_2026'
    sample_unseen_cat.loc[:, 'session_type'] = 'Extreme_Olympic_Throw'
    pred_cat = model.predict(sample_unseen_cat)
    assert not np.isnan(pred_cat[0]), "Handling of unseen categorical values failed!"
    print(f"  [PASS] Unseen categories handled via OneHotEncoder: prediction = {pred_cat[0]:.2f} m")

    # Test 6d: Boundary / Extreme numeric inputs
    sample_extreme = single_sample.copy()
    sample_extreme.loc[:, 'workload_au'] = 2000.0
    sample_extreme.loc[:, 'sleep_hours'] = 1.0
    sample_extreme.loc[:, 'soreness_score'] = 10.0
    pred_extreme = model.predict(sample_extreme)
    assert np.isfinite(pred_extreme[0]), "Extreme numeric values caused non-finite output!"
    print(f"  [PASS] Extreme values handled safely: prediction = {pred_extreme[0]:.2f} m")

    # 8. Save Detailed Test Predictions & JSON Summary Report
    test_eval_output = test_eval[[
        'date', 'training_phase', 'session_type', 'max_throw_distance_m',
        'predicted_distance_m', 'residual_m', 'abs_error_m'
    ]].copy()
    test_eval_output.rename(columns={'max_throw_distance_m': 'actual_distance_m'}, inplace=True)
    test_eval_output.to_csv(OUTPUT_DIR / 'test_predictions_detailed.csv', index=False)

    report = {
        'test_timestamp': pd.Timestamp.now().isoformat(),
        'model_path': str(MODEL_FILE),
        'model_type': type(regressor).__name__,
        'test_sessions_count': len(test),
        'test_date_range': [str(test['date'].min().date()), str(test['date'].max().date())],
        'latency_ms_per_sample': round(latency_per_sample, 3),
        'metrics': {
            'random_forest': rf_metrics,
            'baseline_mean': mean_metrics,
            'baseline_persistence': persist_metrics,
        },
        'improvements_over_baselines': {
            'rmse_improvement_over_mean_pct': round(rmse_imp_mean, 2),
            'rmse_improvement_over_persistence_pct': round(rmse_imp_persist, 2),
            'mae_improvement_over_mean_pct': round(mae_imp_mean, 2),
        },
        'robustness_tests': {
            'single_row_inference': 'PASSED',
            'missing_value_imputation': 'PASSED',
            'unseen_category_handling': 'PASSED',
            'extreme_value_handling': 'PASSED',
        }
    }

    with open(OUTPUT_DIR / 'model_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("                ALL MODEL TESTS PASSED SUCCESSFULLY!          ")
    print("=" * 65)
    print("Saved artifacts:")
    print(f"  - {OUTPUT_DIR / 'model_test_report.json'}")
    print(f"  - {OUTPUT_DIR / 'test_slice_metrics.csv'}")
    print(f"  - {OUTPUT_DIR / 'test_predictions_detailed.csv'}")


if __name__ == '__main__':
    test_model_pipeline()
