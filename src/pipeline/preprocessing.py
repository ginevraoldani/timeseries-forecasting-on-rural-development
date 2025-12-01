import numpy as np
import pandas as pd
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

# SLIDING WINDOW = tecnica, "cornice" fisica che si sposta di un passo alla volta sui dati
# SEQUENCE = contenuto, ciò che si trova dentro la finestra in un dato momento
# INPUT SEQUENCE (X) = ciò che la rete vede (anni t-3, t-3, t-1)
# TARGET SEQUENCE (y) = ciò che la rete deve predire (anno t)
# uso tecnica di sliding window per tagliare dati in tante sequenze di input/output
def create_sequences(data, n_steps_in, n_steps_out=1):
    """ Transforms time series in samples for supervised training.
    handles extra dimension --> if i get a shape (N, 1) i flatten into (N,)
    
    Args:
        data: 
        n_steps_in: length of input window (X)
        n_steps_out: horizon of prediction (y) (default to 1 year)
    Returns:
        X: numpy array (samples, n_steps_in) if univariate.
        y: numpy array (samples, n_steps_out).
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.values
        
    if data.ndim > 1 and data.shape[1] == 1:
        data = data.flatten()
        
    X, y = [], []
    
    # Sliding Window
    for i in range(len(data)):
        # Trova la fine del pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # checks if we went over dataset length
        if out_end_ix > len(data):
            break
            
        # Raccogli input e output
        seq_x = data[i:end_ix]
        seq_y = data[end_ix:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)