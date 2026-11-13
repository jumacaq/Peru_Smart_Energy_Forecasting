import pandas as pd
import numpy as np
import holidays
from datetime import datetime

holidays_peru = holidays.Peru()


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features at 1, 2, 48, 336 steps (30min, 1hr, 24hr, 1 week)."""
    df = df.copy()
    lags = [1, 2, 48, 336]
    for lag in lags:
        df[f'Demand_lag{lag}'] = df['Demand'].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max at windows 48, 336 steps.
    Critical: shift(1) first to prevent current row from being included.
    """
    df = df.copy()
    windows = [48, 336]
    stats = ['mean', 'std', 'min', 'max']

    for window in windows:
        shifted = df['Demand'].shift(1)
        for stat in stats:
            col_name = f'Demand_roll{window}_{stat}'
            if stat == 'mean':
                df[col_name] = shifted.rolling(window=window).mean()
            elif stat == 'std':
                df[col_name] = shifted.rolling(window=window).std()
            elif stat == 'min':
                df[col_name] = shifted.rolling(window=window).min()
            elif stat == 'max':
                df[col_name] = shifted.rolling(window=window).max()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features: hour, half_hour_of_day, day_of_week,
    day_of_month, month, quarter, is_weekend, is_holiday.
    """
    df = df.copy()

    df['hour'] = df.index.hour
    df['half_hour_of_day'] = (df.index.hour * 2) + (df.index.minute // 30)
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df["is_holiday"] = [
        int(date_val in holidays_peru) for date_val in df.index.date
    ]
    return df


def add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encodings for cyclical features.
    hour (24), day_of_week (7), month (12), half_hour_of_day (48).
    """
    df = df.copy()

    # Hour: 24 hours in a day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Day of week: 7 days
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month: 12 months
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Half-hour of day: 48 slots
    df['half_hour_of_day_sin'] = np.sin(2 * np.pi * df['half_hour_of_day'] / 48)
    df['half_hour_of_day_cos'] = np.cos(2 * np.pi * df['half_hour_of_day'] / 48)

    return df


def build_features(
    df: pd.DataFrame, drop_na: bool = False
) -> pd.DataFrame:
    """Orchestrate all feature engineering steps."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_cyclical_encodings(df)

    if drop_na:
        df = df.dropna()
    return df
