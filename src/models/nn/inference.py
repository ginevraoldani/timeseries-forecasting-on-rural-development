import numpy as np

def recursive_forecast(model, initial_sequence, n_steps_ahead):
    """
    Esegue predizioni ricorsive nel futuro.
    Input:
        model: modello Keras addestrato
        initial_sequence: array (n_input_steps,) con gli ultimi dati noti
        n_steps_ahead: quanti anni predire
    Output:
        Array con le predizioni future
    """
    current_seq = initial_sequence.copy()
    future_preds = []
    
    # Reshape per Keras (1, n_steps) se MLP, (1, n_steps, 1) se CNN
    # Qui assumiamo MLP (input piatto)
    # Se fosse CNN: current_seq.reshape(1, len(current_seq), 1)
    
    for _ in range(n_steps_ahead):
        # Prepara l'input: ultimi n_steps dalla sequenza corrente
        input_pattern = current_seq[-len(initial_sequence):].reshape(1, -1)
        
        # Predici
        pred_value = model.predict(input_pattern, verbose=0)[0, 0]
        
        # Salva e aggiorna
        future_preds.append(pred_value)
        current_seq = np.append(current_seq, pred_value)
        
    return np.array(future_preds)