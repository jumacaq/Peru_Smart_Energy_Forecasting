import numpy as np
import pandas as pd


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    epsilon: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (MAPE).
    epsilon prevents division by zero when y_true is 0.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return float(
        np.mean(np.abs((y_true_arr - y_pred_arr) / (np.abs(y_true_arr) + epsilon)))
        * 100
    )


def smape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    epsilon: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    Ranges from 0 to 200 (or 0 to 2 if expressed as decimal).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    numerator = np.abs(y_true_arr - y_pred_arr)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + epsilon

    return float(np.mean(2 * numerator / denominator) * 100)


def evaluate_all(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """Compute all metrics and return as a dictionary."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }

def evaluation_table(
    models_metrics: dict[str, dict[str, float]],
    sort_by: str = "RMSE",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Converts a dictionary of metrics into a sorted comparison DataFrame. 

    Parameters
    models_metrics : dict
    Dictionary formatted as {'Model': {'MAE': x, 'RMSE': y, ...}}
    sort_by : str, default='RMSE'
    Primary metric by which to sort the table ('RMSE', 'MAE', 'MAPE', 'sMAPE').
    ascending : bool, default=True
    If True, sorts from lowest to highest (lower errors come first).

    Returns

    pd.DataFrame
    Sorted DataFrame with models in the rows and metrics in the columns.
    """
    df = pd.DataFrame(models_metrics).T

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    else:
        # Si la métrica especificada no existe, intenta ordenar por la primera columna disponible
        df = df.sort_values(by=df.columns[0], ascending=ascending)

    return df