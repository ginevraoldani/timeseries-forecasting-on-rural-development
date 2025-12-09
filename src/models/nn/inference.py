import numpy as np

def recursive_forecast(model, initial_sequence, n_steps_ahead):
    """
    Esegue predizioni ricorsive nel futuro.
    Funziona sia per MLP (input 2D) che per CNN/RNN (input 3D).
    
    Input:
        model: modello Keras addestrato
        initial_sequence: array (n_input_steps,) con gli ultimi dati noti
        n_steps_ahead: quanti anni predire
    Output:
        Array con le predizioni future
    """
    # Ci assicuriamo che la sequenza iniziale sia un array piatto
    current_seq = np.array(initial_sequence).flatten().copy()
    n_input_steps = len(current_seq)
    future_preds = []
    
    # Rileviamo se il modello vuole input 3D (CNN/LSTM) o 2D (MLP)
    # model.input_shape restituisce es: (None, 5, 1) per CNN -> len 3
    # model.input_shape restituisce es: (None, 5) per MLP -> len 2
    is_3d_input = len(model.input_shape) == 3
    
    for _ in range(n_steps_ahead):
        # 1. Estrai l'ultima finestra disponibile
        last_window = current_seq[-n_input_steps:]
        
        # 2. Reshape dinamico in base al modello
        if is_3d_input:
            # Per CNN: (Batch, TimeSteps, Channels) -> (1, N, 1)
            input_pattern = last_window.reshape(1, n_input_steps, 1)
        else:
            # Per MLP: (Batch, Features) -> (1, N)
            input_pattern = last_window.reshape(1, n_input_steps)
        
        # 3. Predici
        pred_value = model.predict(input_pattern, verbose=0)[0, 0]
        
        # 4. Salva e aggiorna
        future_preds.append(pred_value)
        current_seq = np.append(current_seq, pred_value)
        
    return np.array(future_preds)