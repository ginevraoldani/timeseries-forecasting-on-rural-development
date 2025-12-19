import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_errors(y_true, y_pred):
    """
    Calculates error metrics to evaluate models' performance.
    
    Automatically manages:
    - list/array input ( -> flattens).
    - NaN deletion.
    - division for zero in MAPE ( -> excludes values where y_true == 0).
    - MAPE > 5000% ( -> conversion in NaN).

    Args:
        y_true (array-like): real values.
        y_pred (array-like): predicted values.

    Returns:
        dict: dictionary containing rounded RMSE, MAE, MAPE, R2.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            "RMSE": np.nan, 
            "MAE": np.nan, 
            "MAPE": np.nan, 
            "R2": np.nan
        }
    
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    non_zero_mask = y_true_clean != 0
    
    if np.any(non_zero_mask):
        y_t_safe = y_true_clean[non_zero_mask]
        y_p_safe = y_pred_clean[non_zero_mask]
        
        # mean(abs((True-Pred)/True))*100
        mape_val = np.mean(np.abs((y_t_safe - y_p_safe) / y_t_safe)) * 100
        
        if mape_val > 5000:
            mape = np.nan
        else:
            mape = mape_val
    else:
        mape = np.nan

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2) if not np.isnan(mape) else None,
        "R2": round(r2, 4)
    }