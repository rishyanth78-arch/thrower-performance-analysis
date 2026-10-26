from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_FILE = BASE_DIR / 'data/raw/training_records.csv'
DEFAULT_CLEAN_FILE = BASE_DIR / 'data/cleaned/training_records_cleaned.csv'


def process_and_clean_data(source=None, output_path=None):
    """
    Cleans raw training records into the standardized schema.
    source: can be a DataFrame, file path (str or Path), or None (uses DEFAULT_RAW_FILE).
    output_path: Path to save the cleaned CSV, or None.
    Returns: cleaned DataFrame.
    """
    if source is None:
        source = DEFAULT_RAW_FILE

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f'Raw dataset not found: {source_path.resolve()}')
        df = pd.read_csv(source_path)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)

    required = [
        'date',
        'training_phase',
        'session_type',
        'workload_au',
        'sleep_hours',
        'hrv_ms',
        'soreness_score',
        'release_velocity_ms',
        'max_throw_distance_m',
    ]

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    numeric_columns = [
        'workload_au',
        'sleep_hours',
        'hrv_ms',
        'soreness_score',
        'release_velocity_ms',
        'max_throw_distance_m',
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    for column in ['training_phase', 'session_type']:
        df[column] = df[column].astype('string').str.strip()

    df = df.drop_duplicates()
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    is_rest = df['session_type'].str.casefold().eq('rest').fillna(False)
    df['is_throwing_session'] = (~is_rest) & (df['release_velocity_ms'].gt(0) | df['max_throw_distance_m'].gt(0))

    df['data_quality_flag'] = ''
    negative_rows = df[numeric_columns].lt(0).any(axis=1)
    df.loc[negative_rows, 'data_quality_flag'] = 'Review negative measurement'

    rest_with_performance = is_rest & (df['release_velocity_ms'].gt(0) | df['max_throw_distance_m'].gt(0))
    df.loc[rest_with_performance, 'data_quality_flag'] = (
        df.loc[rest_with_performance, 'data_quality_flag']
        .apply(lambda x: f'{x}; ' if x else '')
        + 'Rest day has non-zero performance'
    )

    if output_path is not None:
        try:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
        except Exception as e:
            # Pass silently if running in read-only environment
            print(f'Warning: Could not write output file: {e}')

    return df


if __name__ == '__main__':
    raw_file = DEFAULT_RAW_FILE
    clean_file = DEFAULT_CLEAN_FILE
    print(f"Reading raw data from: {raw_file}")
    df = process_and_clean_data(source=raw_file, output_path=clean_file)

    is_rest = df['session_type'].str.casefold().eq('rest').fillna(False)
    print('\n=== CLEANING SUMMARY ===')
    print('Rows saved:', len(df))
    print('Throwing sessions:', int(df['is_throwing_session'].sum()))
    print('Rest sessions:', int(is_rest.sum()))
    print('Rows flagged for review:', int(df['data_quality_flag'].ne('').sum()))
    print('Output:', clean_file.resolve())
    print('Original dataset was not modified.')
