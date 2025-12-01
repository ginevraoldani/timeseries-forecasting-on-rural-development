import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_errors(y_true, y_pred):
    """
    Calcola RMSE, MAE e un MAPE 'sicuro' (gestisce divisione per zero).
    y_true: Array dei valori reali
    y_pred: Array dei valori predetti
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Evitiamo divisione per zero: calcoliamo MAPE solo dove y_true != 0
    mask = y_true != 0
    if np.any(mask):
        # Calcolo standard MAPE: mean(|(true - pred) / true|) * 100
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan  # Se tutti i valori sono 0, il MAPE è impossibile
        
    # Se il MAPE è enorme (>1000%), è probabile che ci siano valori vicini allo zero.
    # In quel caso, meglio restituire NaN o un cap, per non rovinare i grafici.
    if mape > 5000: 
        mape = np.nan

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2) if not np.isnan(mape) else None
    }