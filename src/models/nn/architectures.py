from tensorflow.keras.models import Sequential              # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout   # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore

def build_mlp_model(params, input_dim):
    """
    Costruisce un modello MLP dinamico basato sui parametri passati.
    Specifico per MLP.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(params['units'], activation=params['activation']))
    for _ in range(params['n_layers'] - 1):
        model.add(Dense(params['units'], activation=params['activation']))
        if params['dropout'] > 0:
            model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    
    return model