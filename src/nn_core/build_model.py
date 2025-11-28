import tensorflow as tf
from tensorflow.keras.models import Sequential              # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout   # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore
from tensorflow.keras.callbacks import EarlyStopping        # type: ignore

def build_mlp_model(params, input_dim):
    """
    Costruisce un modello MLP dinamico basato sui parametri passati.
    Specifico per MLP.
    """
    model = Sequential()
    
    # Input Layer esplicito
    model.add(Input(shape=(input_dim,)))
    
    # Primo Hidden Layer
    model.add(Dense(params['units'], activation=params['activation']))
    
    # Altri Hidden Layers (dinamici)
    # n_layers include il primo, quindi iteriamo n_layers - 1 volte
    for _ in range(params['n_layers'] - 1):
        model.add(Dense(params['units'], activation=params['activation']))
        if params['dropout'] > 0:
            model.add(Dropout(params['dropout']))
            
    # Output Layer (Regressione)
    model.add(Dense(1))
    
    # Compilazione
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    
    return model

def train_mlp_model(model, X_train, y_train, X_val, y_val, batch_size, epochs=100, patience=10):
    """
    Funzione di training GENERICA (riutilizzabile per CNN, LSTM, ecc.).
    Gestisce il fit e l'Early Stopping.
    """
    # Callback per fermare il training se non migliora
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    return history, model