from tensorflow.keras.models import Sequential                      # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout           # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten   # type: ignore
from tensorflow.keras.optimizers import Adam                        # type: ignore

def build_mlp_model(params, input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    base_units = params['units']
    n_layers = params['n_layers']
    
    for i in range(n_layers):
        # Funnel 
        if i == 0:
            # Primo Layer: 2 * n
            current_units = base_units * 2
        elif i == 1:
            # Secondo Layer: n / 2
            current_units = base_units // 2
        elif i == 2:
            # Terzo Layer: n / 4
            current_units = base_units // 4
        else:
            current_units = base_units // (2 ** i)
        current_units = max(4, current_units) 
        model.add(Dense(current_units, activation=params['activation']))
        if params['dropout'] > 0:
            model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    
    return model

def build_cnn_model(params, input_shape):
    """
    Costruisce un modello CNN 1D dinamico basato sui parametri passati.
    Specifico per CNN 1D time series.
    
    Args:
        params (dict): dict containing architectural params
        input_shape (tuple): (window_size, n_features) es. (3, 1)
    
    Returns:
        model: keras compiled model
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Layer convoluzionali
    for i in range(params['n_conv_layers']):
        model.add(Conv1D(
            filters=params['n_filters'],
            kernel_size=params['kernel_size'],
            activation=params['activation'],
            padding='same'  # mantiene dimensione temporale
        ))
        if params.get('use_pooling', True):  # opzionale
            model.add(MaxPooling1D(pool_size=params.get('pool_size', 2)))
    
    # Flatten e Dense layers
    model.add(Flatten())
    
    if params.get('dense_units', 0) > 0:  # layer Dense opzionale
        model.add(Dense(params['dense_units'], activation=params['activation']))
    
    if params.get('dropout', 0) > 0:
        model.add(Dropout(params['dropout']))
    
    # Output layer
    model.add(Dense(1))  # forecasting univariato
    
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model
