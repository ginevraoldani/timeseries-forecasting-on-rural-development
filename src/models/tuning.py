import itertools
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def tune_arima_parameters(train_data, p_range, d, q_range):
    """
    Find the best ARIMA parameters (p, q) based on AIC criterion. 
    Args:
        train_data (pd.DataFrame or pd.Series): time series data to fit.
                Should be stationary or will be differenced internally based on d parameter.
        p_range (range or list): range of p values to test for autoregressive order.
        d (int): differencing order. Statsmodels will difference internally if d > 0.
        q_range (range or list): range of q values to test for moving average order.
    
    Returns:
        tuple: best ARIMA order (p, d, q) with lowest AIC value.
    """
    best_aic = float('inf')
    best_order = None
    pdq_combinations = list(itertools.product(p_range, [d], q_range))
    
    for order in pdq_combinations:
        try:
            model = ARIMA(train_data, order=order)
            results = model.fit()
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        except:
            continue
            
    print(f"    -> Best Params: {best_order} with AIC: {best_aic:.2f}")
    return best_order