import numpy as np

def recursive_forecast(model, initial_sequence, n_steps_ahead):
    """
    Esegue predizioni ricorsive nel futuro.
    Funziona sia per MLP (input 2D) che per CNN/RNN (input 3D).
    - finds if model requires 2D input (MLP) or 3D input (CNN)
    - extracts last available window
    - dynamic reshape according to model
    - 
    Input:
        model: modello Keras addestrato
        initial_sequence: array (n_input_steps,) con gli ultimi dati noti
        n_steps_ahead: quanti anni predire
    Output:
        Array con le predizioni future
    """
    current_seq = np.array(initial_sequence).flatten().copy()
    n_input_steps = len(current_seq)
    future_preds = []
    
    # model.input_shape restituisce es: (None, 5) per MLP -> len 2
    # model.input_shape restituisce es: (None, 5, 1) per CNN -> len 3
    is_3d_input = len(model.input_shape) == 3
    
    for _ in range(n_steps_ahead):
        last_window = current_seq[-n_input_steps:]
        
        if is_3d_input:
            input_pattern = last_window.reshape(1, n_input_steps, 1)    # CNN: (Batch, TimeSteps, Channels) -> (1, N, 1)
        else:
            input_pattern = last_window.reshape(1, n_input_steps)       # MLP: (Batch, Features) -> (1, N)
        
        pred_value = model.predict(input_pattern, verbose=0)[0, 0]
        future_preds.append(pred_value)
        current_seq = np.append(current_seq, pred_value)
        
    return np.array(future_preds)