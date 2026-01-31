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
            # Secondo Layer: n
            current_units = base_units
        elif i == 2:
            # Terzo Layer: n / 2
            current_units = base_units // 2
        else:
            current_units = base_units // (2 ** i)
        current_units = max(4, current_units) 
        model.add(Dense(current_units, activation=params['activation']))
        if params['dropout'] > 0:
            model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    optimizer = Adam(
        learning_rate=params['learning_rate'],
        clipvalue=1.0
        )
    model.compile(optimizer=optimizer, loss='mse')
    return model

def build_cnn_model(params):    
    model = Sequential()
    
    model.add(Conv1D(filters=params['filters'], 
                    kernel_size=params['kernel_size'], 
                    activation='relu',
                    padding='same',
                    input_shape=params['input_shape']))
    
    current_dim = params['input_shape'][0]
    pool_size = params.get('pool_size', 2)

    # Pooling sicuro: lo facciamo solo se c'è spazio
    if current_dim >= pool_size:
        model.add(MaxPooling1D(pool_size=pool_size))
        current_dim = current_dim // pool_size
    
    model.add(Dropout(params['dropout']))
    
    # Questo ciclo aggiunge layer SOLO se la dimensione dei dati lo permette
    for _ in range(params['n_conv_layers'] - 1):
        
        # Con padding='same', la Conv1D non riduce la dimensione, quindi è sempre sicura
        # se il kernel non supera la dimensione (ma padding same gestisce anche quello in TF recenti).
        # Tuttavia, per logica, evitiamo di aggiungere layer su feature map minuscole (es. 1 o 2 steps).
        if current_dim >= params['kernel_size']:
            
            model.add(Conv1D(filters=params['filters'] * 2, # Raddoppio filtri
                            kernel_size=params['kernel_size'], 
                            padding='same', 
                            activation='relu'))
            
            # Check per il Pooling nel layer successivo
            if current_dim >= pool_size:
                model.add(MaxPooling1D(pool_size=pool_size))
                current_dim = current_dim // pool_size
            
            model.add(Dropout(params['dropout']))
        else:
            # STOP: La rete è diventata troppo profonda per questi dati (input troppo corto).
            # Usciamo dal ciclo senza aggiungere altri layer convoluzionali.
            break

    # HEAD (Classificatore/Regressore)
    model.add(Flatten())
    
    if params.get('dense_units', 0) > 0:
        model.add(Dense(params['dense_units'], activation='relu'))
    
    if params.get('dropout', 0) > 0:
        model.add(Dropout(params['dropout']))
    
    # Output Layer (Regressione univara)
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                loss='mse',
                metrics=['mae'])
    
    return model