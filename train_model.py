from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path("data/cleaned/training_records_cleaned.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run python preprocess.py first."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Keep sessions with an actual positive throwing distance.
# Rest days and sessions without a recorded distance are excluded.
df["max_throw_distance_m"] = pd.to_numeric(
    df["max_throw_distance_m"], errors="coerce"
)

df = df[
    (df["max_throw_distance_m"] > 0)
    & (df["is_throwing_session"].astype(str).str.lower() == "true")
].copy()

# Features that could be available when planning or starting a session.
# Release velocity is excluded because it is measured during the throw.
numeric_features = [
    "workload_au",
    "sleep_hours",
    "hrv_ms",
    "soreness_score",
]

categorical_features = [
    "training_phase",
    "session_type",
]

features = numeric_features + categorical_features
target = "max_throw_distance_m"

required = features + [target, "date"]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Exclude rows without a target or date; impute missing input features
df = df.dropna(subset=["date", target])
df = df.sort_values("date").reset_index(drop=True)

if len(df) < 20:
    raise ValueError("Too few throwing sessions to make a reliable split.")

# Chronological split: first 80% for training, final 20% for testing
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print("Training sessions:", len(train))
print("Testing sessions:", len(test))
print("Training dates:", train["date"].min().date(), "to",
      train["date"].max().date())
print("Testing dates:", test["date"].min().date(), "to",
      test["date"].max().date())

# Preprocess numeric and categorical inputs separately
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Baseline: always predict the training-set average distance
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

# Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

def evaluate(actual, predicted):
    return {
        "MAE_m": mean_absolute_error(actual, predicted),
        "RMSE_m": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

results = pd.DataFrame({
    "Baseline": evaluate(y_test, baseline_predictions),
    "Random Forest": evaluate(y_test, predictions),
}).T

print("\n=== TEST RESULTS ===")
print(results.round(3))

results.to_csv(OUTPUT_DIR / "model_metrics.csv")

# Save test predictions for later charts and review
prediction_output = pd.DataFrame({
    "date": test["date"].to_numpy(),
    "actual_distance_m": y_test.to_numpy(),
    "baseline_prediction_m": baseline_predictions,
    "random_forest_prediction_m": predictions,
})

prediction_output.to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

# Save the fitted pipeline, including preprocessing
joblib.dump(model, MODEL_DIR / "distance_model.joblib")

print("\nSaved:")
print(OUTPUT_DIR / "model_metrics.csv")
print(OUTPUT_DIR / "test_predictions.csv")
print(MODEL_DIR / "distance_model.joblib")
