import numpy as np

def compute_residuals(y_true, y_pred):
    """
    Calcola i residui e le statistiche di base.
    NON fa grafici.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    residuals = y_true - y_pred
    
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }
    
    return residuals, stats