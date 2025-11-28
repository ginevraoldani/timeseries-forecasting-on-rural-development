import numpy as np
import pandas as pd

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