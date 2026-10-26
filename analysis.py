from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path('data/cleaned/training_records_cleaned.csv')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=['date'])
df = df.sort_values('date')

throws = df[df['is_throwing_session'] == True].copy()
if throws.empty:
    raise ValueError('No throwing sessions found in the cleaned data.')

plt.figure(figsize=(12, 5))
plt.scatter(throws['date'], throws['max_throw_distance_m'], s=18, alpha=0.7)
plt.title('Throwing Distance Over Time')
plt.xlabel('Date')
plt.ylabel('Maximum Throw Distance (m)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distance_over_time.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 5))
plt.scatter(throws['date'], throws['release_velocity_ms'], s=18, alpha=0.7)
plt.title('Release Velocity Over Time')
plt.xlabel('Date')
plt.ylabel('Release Velocity (m/s)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'velocity_over_time.png', dpi=150)
plt.close()

phase_summary = (
    throws.groupby('training_phase')
    .agg(
        sessions=('max_throw_distance_m', 'count'),
        average_distance=('max_throw_distance_m', 'mean'),
        average_velocity=('release_velocity_ms', 'mean'),
        average_workload=('workload_au', 'mean'),
    )
    .round(2)
)

print('\n=== PERFORMANCE BY TRAINING PHASE ===')
print(phase_summary)
phase_summary.to_csv(OUTPUT_DIR / 'phase_summary.csv')
print('\nCharts and summary saved in:', OUTPUT_DIR.resolve())
