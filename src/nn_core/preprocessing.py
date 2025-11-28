import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# avevo scalato gli anni ma è inutile quindi scalo solamente i valori
def scale_datasets(single_orig_aug_subsets):
    """ 
    Scales data in the dictionary using only the train_orig for fitting (to avoid Data Leakage).
    - Initialize one scaler for y (Value).
    - FIT on train_orig, reshape(-1,1) because sklearn requires rank-2 tensors.
    - Save scaler for later use (de-scaling)
    - TRANSFORM applies scaler to each subset in dictionary single_orig_aug_subsets

    Args:
        single_orig_aug_subsets (dict): dictionary containing train set, validation set and test set of original time serie +
                                                df_step (dataframe augmented through step function) + 
                                                df_jitter (dataframe augmented through linear interpolation + jitter)
                                                of a specific indicator

    Returns:
        scaled_dict: dictionary containing scaled data
        scalers: scalers used for y (Value)
    """
    
    scaled_dict = {}
    scalers = {}
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    if 'orig_train' not in single_orig_aug_subsets:
        raise ValueError("'orig_train' missing in data dictionary")
    
    train_orig = single_orig_aug_subsets['orig_train']
    y_train_vals = train_orig['Value'].values.reshape(-1, 1)
    scaler_y.fit(y_train_vals)
    scalers['scaler_y'] = scaler_y
    
    for key, df in single_orig_aug_subsets.items():
        if df is None or df.empty:
            scaled_dict[key] = None
            continue
            
        df_scaled = df.copy()
        
        # Trasformo y
        y_vals = df['Value'].values.reshape(-1, 1)
        df_scaled['Value'] = scaler_y.transform(y_vals).flatten()
        
        scaled_dict[key] = df_scaled
        
    return scaled_dict, scalers