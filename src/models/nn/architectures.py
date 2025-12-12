from tensorflow.keras.models import Sequential                                              # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Dense, Dropout    # type: ignore
from tensorflow.keras.optimizers import Adam                                                # type: ignore

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

def build_cnn_model(params):
    """ Builds a dynamic 1D CNN, based on params in input.

    Args:
        params (dict): dict containing architectural params

    Returns:
        model: keras compiled model
    """
    model = Sequential()
    model.add(Input(shape=params['input_shape']))
    model.add(Conv1D(filters=params['filters'], 
                    kernel_size=params['kernel_size'], 
                    activation='relu'))
    
    if params.get('pool_size'):
        model.add(MaxPooling1D(pool_size=params['pool_size']))
    
    model.add(Dropout(params['dropout']))

    for _ in range(params['n_conv_layers'] - 1):
        model.add(Conv1D(filters=params['filters'] * 2, # Spesso si raddoppiano i filtri scendendo
                        kernel_size=params['kernel_size'], 
                        activation=params['activation'],
                        padding = 'same'
        ))
        if params.get('use_pooling', True):  # opzionale
            model.add(MaxPooling1D(pool_size=params.get('pool_size', 2)))
        model.add(Dropout(params['dropout']))

    model.add(Flatten())
    
    if params.get('dense_units', 0) > 0:
        model.add(Dense(params['dense_units'], activation=params['activation']))
    
    if params.get('dropout', 0) > 0:
        model.add(Dropout(params['dropout']))
    
    # Output Layer (Regressione: 1 unità lineare) -> forecasting univariato
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer = Adam(learning_rate=params['learning_rate']),
                loss='mse',
                metrics=['mae']
    )
    
    return model