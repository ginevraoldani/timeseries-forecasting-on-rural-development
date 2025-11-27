import numpy as np

def create_sequences(data, n_steps_in, n_steps_out=1):
    """
    Trasforma una serie temporale in campioni per Apprendimento Supervisionato.
    n_steps_in: quanti step guardare indietro (es. 12 mesi)
    n_steps_out: quanti step predire in avanti (es. 1 mese)
    """
    X, y = [], []
    for i in range(len(data)):
        # Trova la fine del pattern corrente
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # Controlla se siamo oltre la lunghezza del dataset
        if out_end_ix > len(data):
            break
            
        # Raccogli input e output
        seq_x = data[i:end_ix]
        seq_y = data[end_ix:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)