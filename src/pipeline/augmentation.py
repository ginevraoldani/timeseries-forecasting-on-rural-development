import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def augment_step_function(x, y, scale_factor):
    """
    Augments data points of initial time serie by scale_factor times

    Args:
        x (_type_): first sequence to be interpolated ['Year']
        y (_type_): second sequence to be interpolated ['Value']
        scale_factor (int): by how many points the time serie have to be augmented

    Returns:
        DataFrame{'Year', 'Value'}: df with columns of Year and Value interpolated by scale_factor
    """
    
    # 'kind=previous' mantiene il valore precedente (comportamento a gradino)
    f = interp1d(x, y, kind='previous', fill_value="extrapolate")
    x_step = np.linspace(x.min(), x.max(), len(x) * scale_factor)
    y_step = f(x_step)
    return pd.DataFrame({'Year': x_step, 'Value': y_step})

def augment_linear_with_jitter(x, y, scale_factor, noise_level):
    """
    Creates new data points by interpolation = tracing a line between existing data points
    and adds jittering = casual noise for resilience

    Args:
        x (_type_): first sequence to be interpolated ['Year']
        y (_type_): second sequence to be interpolated ['Value']
        scale_factor (int): by how many points the time serie have to be augmented
        noise_level (float): standard deviation of noise (0.01 = 1% variation)

    Returns:
        DataFrame{'Year', 'Value'}: df with columns of Year and Value interpolated by scale_factor
    """
    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_interp = np.linspace(x.min(), x.max(), len(x) * scale_factor)
    y_interp = f(x_interp)
    
    noise = np.random.normal(loc=0.0, scale=noise_level, size=y_interp.shape)
    y_new = y_interp + noise
    return pd.DataFrame({'Year': x_interp, 'Value': y_new})