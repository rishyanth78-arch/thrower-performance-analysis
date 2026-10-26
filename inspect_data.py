from pathlib import Path
import pandas as pd

file_path = Path('data/raw/training_records.csv')

if not file_path.exists():
    print('Training records were not found.')
    print(f'Please place your CSV file here: {file_path.resolve()}')
    raise SystemExit(1)

df = pd.read_csv(file_path)

print('\n=== DATASET OVERVIEW ===')
print('Number of rows:', len(df))
print('Number of columns:', len(df.columns))

print('\n=== COLUMN NAMES ===')
print(df.columns.tolist())

print('\n=== FIRST FIVE RECORDS ===')
print(df.head())

print('\n=== MISSING VALUES ===')
print(df.isna().sum())

print('\n=== DUPLICATE ROWS ===')
print(df.duplicated().sum())

if 'Session_Type' in df.columns:
    print('\n=== SESSION TYPES ===')
    print(df['Session_Type'].value_counts(dropna=False))

if 'Training_Phase' in df.columns:
    print('\n=== TRAINING PHASES ===')
    print(df['Training_Phase'].value_counts(dropna=False))

if 'Date' in df.columns:
    dates = pd.to_datetime(df['Date'], errors='coerce')
    print('\n=== DATE CHECK ===')
    print('Valid dates:', dates.notna().sum())
    print('Invalid or missing dates:', dates.isna().sum())
    print('Date range:', dates.min(), 'to', dates.max())

print('\n=== KEY METRICS (non-zero counts) ===')
if 'Release_Velocity_ms' in df.columns:
    print('Non-zero Release_Velocity_ms:', (df['Release_Velocity_ms'] > 0).sum())
if 'Max_Throw_Distance_m' in df.columns:
    print('Non-zero Max_Throw_Distance_m:', (df['Max_Throw_Distance_m'] > 0).sum())

print('\nData inspection complete.')
print('Your original CSV has not been modified.')
