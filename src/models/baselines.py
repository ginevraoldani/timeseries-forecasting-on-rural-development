import pandas as pd
import numpy as np

def predict_historical_mean(train_data, forecast_index, **kwargs):
    """
    Historical Mean: Predicts future values using the average of all historical observations.
    
    This baseline method computes the mean of the entire training dataset and projects it 
    as a constant forecast for all future periods. It is suitable for stationary series 
    and serves as a simple baseline for comparison.
    
    Args:
        train_data (pd.DataFrame or pd.Series): historical training data.
        forecast_index (pd.Index or list): index for the forecast period.
        **kwargs: additional keyword arguments (unused).
    
    Returns:
        pd.Series: forecast series with constant values equal to the historical mean,
                indexed by forecast_index and named 'pred'.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    hist_mean = series.mean()
    
    return pd.Series(
        data=[hist_mean] * len(forecast_index),
        index=forecast_index,
        name='pred'
    )

def predict_random_walk(train_data, forecast_index, **kwargs):
    """
    Naive / Random Walk: projects the last observed value into the future as a flat line.
    
    This baseline method uses the most recent observation and repeats it for all future 
    periods. It is suitable for non-stationary series (e.g., GDP) where the trend is 
    expected to continue.
    
    Args:
        train_data (pd.DataFrame or pd.Series): historical training data.
        forecast_index (pd.Index or list): index for the forecast period.
        **kwargs: additional keyword arguments (unused).
    
    Returns:
        pd.Series: forecast series with constant values equal to the last observation,
                indexed by forecast_index and named 'pred'.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    
    last_value = series.iloc[-1]
    
    return pd.Series(
        data=[last_value] * len(forecast_index),
        index=forecast_index,
        name='pred'
    )
    
def predict_random_walk_drift(train_data, forecast_index, **kwargs):
    """
    Random Walk with Drift: Projects future values using a linear trend based on historical drift.
    
    This baseline method calculates the average slope (drift) from the first to the last observation
    and uses it to extrapolate future values. It is suitable for non-stationary series with a 
    consistent trend.
    
    Args:
        train_data (pd.DataFrame or pd.Series): historical training data.
        forecast_index (pd.Index or list): index for the forecast period.
        **kwargs: additional keyword arguments (unused).
    
    Returns:
        pd.Series: forecast series with values following the linear trend,
                indexed by forecast_index and named 'pred'.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    y_T = series.iloc[-1]
    y_1 = series.iloc[0]
    T = len(series)
    
    slope = (y_T - y_1) / (T - 1)
    preds = []
    for h in range(1, len(forecast_index) + 1):
        preds.append(y_T + (h * slope))
        
    return pd.Series(
        data=preds,
        index=forecast_index,
        name='pred'
    )

def predict_seasonal_naive(train_data, forecast_index, season_length=3):
    """
    Seasonal Naive: Repeats the last observed seasonal cycle into the future.
    
    This baseline method replicates the most recent seasonal pattern for forecasting.
    If data is annual with no seasonality, season_length=1 behaves like Last Value.
    For multi-year cycles (e.g., 5-year cycles), set season_length accordingly.
    To predict T+1 I take T+1-S. For multi-step forecasting, I repeat the last block of "season_length" values.
    
    Args:
        train_data (pd.DataFrame or pd.Series): historical training data.
        forecast_index (pd.Index or list): index for the forecast period.
        season_length (int, optional): length of the seasonal cycle. Default is 1 (no seasonality).
    
    Returns:
        pd.Series: forecast series with values repeating the last seasonal cycle,
                indexed by forecast_index and named 'pred'.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    last_season = series.iloc[-season_length:].values
    
    n_required = len(forecast_index)
    repetitions = int(np.ceil(n_required / season_length))
    forecast_values = np.tile(last_season, repetitions)[:n_required]
    
    return pd.Series(
        data=forecast_values,
        index=forecast_index,
        name='pred'
    )

# ---------- WRAPPER TO CALL ONE BASELINE -------------------
BASELINE_MODELS = {
    'mean': predict_historical_mean,
    'naive': predict_random_walk,
    'drift': predict_random_walk_drift,
    'season': predict_seasonal_naive
}

def get_baseline_prediction(method_name, train_data, forecast_index):
    if method_name not in BASELINE_MODELS:
        raise ValueError(f"Baseline '{method_name}' not found.")
    
    model_func = BASELINE_MODELS[method_name]
    return model_func(train_data, forecast_index)