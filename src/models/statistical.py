from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def predict_moving_average_projection(train_data, forecast_index, window=3):
    """
    Moving Average Projection: Computes and projects the average of the last N periods.
    
    This baseline method calculates the mean of the most recent observations within the 
    specified window and uses it as a constant forecast for all future periods. It smooths 
    recent fluctuations while remaining responsive to recent trends.
    
    Args:
        train_data (pd.DataFrame or pd.Series): Historical training data.
        forecast_index (pd.Index or list): Index for the forecast period.
        window (int, optional): Number of recent periods to include in the average. Default is 3.
    
    Returns:
        pd.Series: Forecast series with constant values equal to the moving average,
                indexed by forecast_index and named 'pred'.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    
    # Prende gli ultimi 'window' valori. Se non ce ne sono abbastanza, prende tutto.
    if len(series) < window:
        subset = series
    else:
        subset = series.iloc[-window:]
        
    ma_value = subset.mean()
    
    return pd.Series(
        data=[ma_value] * len(forecast_index),
        index=forecast_index,
        name='pred'
    )

def predict_arima_family(train_data, forecast_index, order=(1,0,0), **kwargs):
    """
    Gestisce l'intera famiglia dei modelli statistici classici (Box-Jenkins).
    
    Args:
        train_data: Serie storica di training.
        forecast_index: Indice per le previsioni.
        order (tuple): (p, d, q)
            - AR puro: (p, 0, 0)
            - MA puro: (0, 0, q)
            - ARMA:    (p, 0, q)
            - ARIMA:   (p, d, q)
            
    Returns:
        pd.Series: Predizioni.
    """
    # Statsmodels richiede serie 1D
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    
    # 1. Fit del modello
    # Suppress warnings se necessario, statsmodels è molto verboso
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        
        # 2. Forecast
        # steps è il numero di passi futuri da predire
        steps = len(forecast_index)
        forecast_result = model_fit.forecast(steps=steps)
        
        # Ri-assegniamo l'indice corretto (statsmodels a volte perde l'indice temporale preciso)
        forecast_series = pd.Series(
            data=forecast_result.values,
            index=forecast_index,
            name='pred'
        )
        return forecast_series
        
    except Exception as e:
        print(f"ARIMA{order} failed: {e}")
        # Fallback in caso di errore di convergenza (può succedere con dati brutti)
        return pd.Series([series.mean()] * len(forecast_index), index=forecast_index)