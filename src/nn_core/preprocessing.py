import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_datasets(orig_aug_subsets):
    """ 
    Scales data in the dictionary using only the train_orig for fitting (to avoid Data Leakage).
    - Initialize two scalers: one for X (Year) and one for y (Value).
    - FIT on train_orig, reshape(-1,1) because sklearn requires rank-2 tensors.
    - Save scaler for later use (de-scaling)
    - TRANSFORM applies scaler to each subset in dictionary orig_aug_subsets

    Args:
        orig_aug_subsets (dict): dictionary containing train set, validation set and test set of original time serie +
                                                df_step (dataframe augmented through step function) + 
                                                df_jitter (dataframe augmented through linear interpolation + jitter)

    Returns:
        scaled_dict: dictionary containing scaled data
        scalers: scalers used for X (Year) and y (Value)
    """
    
    scaled_dict = {}
    scalers = {}
    
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    train_orig = orig_aug_subsets['orig_train']
    x_train_vals = train_orig['Year'].values.reshape(-1, 1)
    y_train_vals = train_orig['Value'].values.reshape(-1, 1)
    
    scaler_x.fit(x_train_vals)
    scaler_y.fit(y_train_vals)
    
    scalers['scaler_x'] = scaler_x
    scalers['scaler_y'] = scaler_y
    
    for key, df in orig_aug_subsets.items():
        if df is None or df.empty:
            scaled_dict[key] = None
            continue
            
        df_scaled = df.copy()
        
        # Trasformo X
        x_vals = df['Year'].values.reshape(-1, 1)
        df_scaled['Year'] = scaler_x.transform(x_vals).flatten()
        
        # Trasformo y
        y_vals = df['Value'].values.reshape(-1, 1)
        df_scaled['Value'] = scaler_y.transform(y_vals).flatten()
        
        scaled_dict[key] = df_scaled
        
    return scaled_dict, scalers