import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_RAW_DATA_PATH = Path("data/raw/demand_2024_2026.xlsx")


def load_raw_data(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw demand data from an Excel or CSV file."""
    file_path = Path(path)  # Ensures 'path' is always a Path object
    if not file_path.exists():
        raise FileNotFoundError(f"File was not found in: {file_path}")

    if file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path, skiprows=3)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    
    else:
        raise ValueError(
            f"Unsupported file format '{file_path.suffix}'. Expected .xlsx or .csv"
        )

    return df


def parse_and_index(df, date_col='FECHA'):
    """
    Parse DATE column to datetime64, set as DatetimeIndex named DATE,
    sort ascending, and drop exact duplicate timestamps.
    """
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y %H:%M')
    df.set_index(date_col, inplace=True)
    df.index.name = 'DATE'

    df.sort_index(inplace=True)

    duplicate_count = df.index.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate timestamps, dropping duplicates")
        df = df[~df.index.duplicated(keep='first')]

    df.rename(columns={'EJECUTADO': 'Demand'}, inplace=True)
    df = df[['Demand']]

    return df


def enforce_30min_grid(df):
    """
    Reindex to full 30-min date_range so all slots exist.
    Interpolate gaps <= 4 steps (2 hours) with time-based linear interpolation.
    """
    df = df.copy()

    start_date = df.index.min()
    end_date = df.index.max()

    expected_index = pd.date_range(start=start_date, end=end_date, freq='30min')

    original_len = len(df)
    df = df.reindex(expected_index)
    df.index.name = 'DATE'

    if len(df) > original_len:
        logger.info(f"Reindexed from {original_len} to {len(df)} rows")

    missing_mask = df['Demand'].isna()
    if missing_mask.any():
        missing_indices = np.where(missing_mask)[0]

        gap_groups = np.split(missing_indices, np.where(np.diff(missing_indices) > 1)[0] + 1)

        no_interpolate_mask = pd.Series(False, index=df.index)

        for gap in gap_groups:
            gap_size = len(gap)
            if gap_size > 4:
                logger.warning(
                    f"Gap of {gap_size} steps (~{gap_size * 30} minutes) found at "
                    f"index {gap[0]} ({df.index[gap[0]]}). Not interpolating."
                )

        df['Demand'] = df['Demand'].interpolate(method='time', limit_area='inside')
        df.loc[no_interpolate_mask, "Demand"] = np.nan

    return df


def validate_series(df):
    """
    Return validation dictionary with series metrics.
    
    Raises ValueError if any Demand values are negative.
    """
    df = df.copy()

    if (df['Demand'] < 0).any():
        raise ValueError("Negative demand values found in the data")

    start = df.index.min()
    end = df.index.max()
    expected_count = len(pd.date_range(start=start, end=end, freq='30min'))

    report = {
        'row_count': len(df),
        'expected_count': expected_count,
        'actual_count': len(df),
        'missing_timestamp_count': df.index.isna().sum(),
        'duplicate_timestamp_count': df.index.duplicated().sum(),
        'min_demand': df['Demand'].min(),
        'max_demand': df['Demand'].max(),
        'min_date': df.index.min(),
        'max_date': df.index.max(),
        'nan_count': df['Demand'].isna().sum(),
        'non_positive_count': (df['Demand'] <= 0).sum(),
    }

    return report

def save_processed_dataset(
    df: pd.DataFrame, path: str = "data/processed/demand.parquet"
) -> None:
    """Save processed dataframe into Parquet format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Processed dataset saved to {output_path}")


def load_processed_dataset(
    path: str = 'data/processed/demand.parquet',
)-> pd.DataFrame:
    """Reload the cleaned dataset from parquet format."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")

    df = pd.read_parquet(path)
    df.index.name = 'DATE'

    return df
